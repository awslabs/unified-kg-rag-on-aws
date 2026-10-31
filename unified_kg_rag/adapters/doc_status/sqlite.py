# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite adapter implementing the document-status registry (DocStatusPort).

The local counterpart of the DynamoDB store, for ``providers.doc_status.kind:
sqlite``: one database file (``providers.doc_status.sqlite_path``) holds the
records, by index namespace (``doc_id_namespace``). Behaviour matches
``FakeDocStatusStore`` and ``DynamoDBDocStatusStore`` (the shared conformance
suite runs all three): the same diff partition and ``stored_scopes``.

SQLite has no item limit, so every record fits (``record_fits``). Each record
is stored as its JSON dump next to the columns ``diff`` reads, so the diff
never deserializes the text-unit lists. (The table of the earlier format,
``doc_status``, is left as it is: run once with ``indexing.reset: true``.)
Writes are transactions; a single connection is shared under a lock (the
pipeline calls the store from worker threads) and waits up to
``search.store_timeout_seconds`` for another writer's lock. Every SQLite error is raised as ``DocStatusRegistryError``, as
the DynamoDB store raises its errors.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import ClassVar

from unified_kg_rag.adapters.local.sqlite_util import (
    connect,
    typed_errors,
    write_transaction,
)
from unified_kg_rag.domain.ingestion.delta_detector import scope_namespace
from unified_kg_rag.domain.ingestion.document_identity import doc_id_namespace
from unified_kg_rag.domain.models import (
    Config,
    DocStatusRecord,
    DocumentDelta,
    ExtractionSettings,
    StoredDocument,
)
from unified_kg_rag.shared import DocStatusRegistryError, get_logger

logger = get_logger(__name__)

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS documents (
        namespace TEXT NOT NULL,
        doc_id TEXT NOT NULL,
        content_hash TEXT NOT NULL,
        status TEXT NOT NULL,
        scope TEXT,
        record TEXT NOT NULL,
        extraction_fingerprint TEXT,
        PRIMARY KEY (namespace, doc_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS leases (
        namespace TEXT PRIMARY KEY,
        owner TEXT NOT NULL,
        expires_at REAL NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS extraction_settings (
        scope TEXT PRIMARY KEY,
        fingerprint TEXT NOT NULL,
        settings TEXT NOT NULL
    )
    """,
)


# SQLite's default limit on bound parameters per statement is 999 on older
# builds; stay under it when reading many keys.
_BATCH = 500


class SQLiteDocStatusStore:
    """SQLite-backed implementation of :class:`DocStatusPort`."""

    # Written in this process (see DataIngestionPipeline._stores_in_process).
    in_process: ClassVar[bool] = True

    def __init__(self, config: Config, path: str | Path | None = None) -> None:
        """Args:
        config: Framework config; ``providers.doc_status.sqlite_path`` is the
            database file unless ``path`` is given.
        path: Database file, or ``":memory:"`` for a private in-memory one.
        """
        self.config = config
        self.path = str(path or config.providers.doc_status.sqlite_path)
        self.timeout_seconds = config.search.store_timeout_seconds
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    @property
    def connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._connection is None:
                with self._registry_errors("open the database"):
                    if self.path != ":memory:":
                        logger.info(
                            "Doc-status registry: SQLite database '%s'",
                            Path(self.path).resolve(),
                        )
                    self._connection = connect(
                        self.path,
                        timeout_seconds=self.timeout_seconds,
                        schema=_SCHEMA,
                    )
            return self._connection

    def close(self) -> None:
        """Close the database connection (reopened on next use)."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _registry_errors(self, operation: str) -> AbstractContextManager[None]:
        """Re-raise SQLite and file errors as :class:`DocStatusRegistryError`."""
        return typed_errors(
            lambda e: DocStatusRegistryError(
                f"Doc-status registry: could not {operation}: SQLite database "
                f"'{self.path}' stayed locked for {self.timeout_seconds}s "
                f"(search.store_timeout_seconds; {e}). Another process is "
                "writing to it; SQLite allows one writer at a time: wait for "
                "the other run, or raise the timeout."
            ),
            lambda e: DocStatusRegistryError(
                f"Doc-status registry: could not {operation} on SQLite database "
                f"'{self.path}': {type(e).__name__}: {e}. Check the path "
                "(providers.doc_status.sqlite_path) is writable, then re-run. "
                "Incremental indexing needs the registry; set "
                "providers.doc_status.enabled: false to run without it."
            ),
        )

    @contextmanager
    def _transaction(self, operation: str) -> Iterator[sqlite3.Connection]:
        with (
            self._lock,
            self._registry_errors(operation),
            write_transaction(self.connection) as connection,
        ):
            yield connection

    # --- records -----------------------------------------------------------

    def get(self, doc_id: str) -> DocStatusRecord | None:
        return self.get_many([doc_id]).get(doc_id)

    def get_many(self, doc_ids: Iterable[str]) -> dict[str, DocStatusRecord]:
        ids = list(dict.fromkeys(doc_ids))
        found: dict[str, DocStatusRecord] = {}
        with self._lock, self._registry_errors("read records"):
            for start in range(0, len(ids), _BATCH):
                batch = ids[start : start + _BATCH]
                marks = ",".join("?" * len(batch))
                rows = self.connection.execute(
                    f"SELECT doc_id, record FROM documents WHERE doc_id IN ({marks})",
                    batch,
                ).fetchall()
                for doc_id, record in rows:
                    found[doc_id] = DocStatusRecord.model_validate_json(record)
        return {doc_id: found[doc_id] for doc_id in ids if doc_id in found}

    def put(self, record: DocStatusRecord) -> None:
        self.put_many([record])

    def put_many(self, records: Iterable[DocStatusRecord]) -> None:
        latest = {record.doc_id: record for record in records}
        if not latest:
            return
        with self._transaction("write records") as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO documents (namespace, doc_id, content_hash, "
                "status, scope, record, extraction_fingerprint) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        doc_id_namespace(record.doc_id),
                        record.doc_id,
                        record.content_hash,
                        record.status.value,
                        record.scope,
                        record.model_dump_json(),
                        record.extraction_fingerprint,
                    )
                    for record in latest.values()
                ],
            )

    def record_fits(self, record: DocStatusRecord) -> bool:
        return True

    def delete(self, doc_id: str) -> None:
        with self._transaction("delete a record") as connection:
            connection.execute(
                "DELETE FROM documents WHERE namespace = ? AND doc_id = ?",
                (doc_id_namespace(doc_id), doc_id),
            )

    # --- the namespace write lease -------------------------------------------

    def try_acquire_lease(self, namespace: str, owner: str, seconds: float) -> bool:
        now = time.time()
        with self._transaction("take the namespace lease") as connection:
            row = connection.execute(
                "SELECT owner, expires_at FROM leases WHERE namespace = ?",
                (namespace,),
            ).fetchone()
            if row is not None and row[0] != owner and row[1] > now:
                return False
            connection.execute(
                "INSERT OR REPLACE INTO leases (namespace, owner, expires_at) "
                "VALUES (?, ?, ?)",
                (namespace, owner, now + seconds),
            )
            return True

    def lease_holder(self, namespace: str) -> tuple[str, float] | None:
        with self._lock, self._registry_errors("read the namespace lease"):
            row = self.connection.execute(
                "SELECT owner, expires_at FROM leases WHERE namespace = ?",
                (namespace,),
            ).fetchone()
        return (row[0], row[1]) if row else None

    def release_lease(self, namespace: str, owner: str) -> None:
        with self._transaction("release the namespace lease") as connection:
            connection.execute(
                "DELETE FROM leases WHERE namespace = ? AND owner = ?",
                (namespace, owner),
            )

    # --- the extraction settings of a scope's last commit ---------------------

    def get_extraction_settings(
        self, scope: str
    ) -> tuple[str, ExtractionSettings] | None:
        with self._lock, self._registry_errors("read the extraction settings"):
            row = self.connection.execute(
                "SELECT fingerprint, settings FROM extraction_settings "
                "WHERE scope = ?",
                (scope,),
            ).fetchone()
        if row is None:
            return None
        return row[0], ExtractionSettings.model_validate_json(row[1])

    def put_extraction_settings(
        self, scope: str, fingerprint: str, settings: ExtractionSettings
    ) -> None:
        with self._transaction("record the extraction settings") as connection:
            connection.execute(
                "INSERT OR REPLACE INTO extraction_settings "
                "(scope, fingerprint, settings) VALUES (?, ?, ?)",
                (scope, fingerprint, settings.model_dump_json()),
            )

    def list_namespace(self, namespace: str) -> list[DocStatusRecord]:
        with self._lock, self._registry_errors("read the records"):
            rows = self.connection.execute(
                "SELECT record FROM documents WHERE namespace = ? ORDER BY rowid",
                (namespace,),
            ).fetchall()
        return [DocStatusRecord.model_validate_json(row[0]) for row in rows]

    def list_all(self) -> list[DocStatusRecord]:
        with self._lock, self._registry_errors("read the records"):
            rows = self.connection.execute(
                "SELECT record FROM documents ORDER BY rowid"
            ).fetchall()
        return [DocStatusRecord.model_validate_json(row[0]) for row in rows]

    # --- diff --------------------------------------------------------------

    def diff(
        self,
        incoming: dict[str, str],
        scope: str | None = None,
        extraction_fingerprint: str | None = None,
    ) -> DocumentDelta:
        """Classify ``{doc_id: content_hash}`` against the stored records.

        With ``scope``, only the scope's namespace is read; the partition is
        ``DocumentDelta.classify``, as for every registry.
        """
        columns = "doc_id, content_hash, status, scope, extraction_fingerprint"
        namespace = scope_namespace(scope)
        with self._lock, self._registry_errors("read the records"):
            if namespace is None:
                rows = self.connection.execute(
                    f"SELECT {columns} FROM documents ORDER BY rowid"
                ).fetchall()
            else:
                rows = self.connection.execute(
                    f"SELECT {columns} FROM documents WHERE namespace = ? "
                    "ORDER BY rowid",
                    (namespace,),
                ).fetchall()
        stored = {
            doc_id: StoredDocument(content_hash, stored_scope, status, extraction)
            for doc_id, content_hash, status, stored_scope, extraction in rows
        }
        return DocumentDelta.classify(incoming, stored, scope, extraction_fingerprint)
