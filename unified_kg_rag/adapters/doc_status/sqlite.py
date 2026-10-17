# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite adapter implementing the document-status registry (DocStatusPort).

The local counterpart of the DynamoDB store, for ``providers.doc_status.kind:
sqlite``: one database file (``providers.doc_status.sqlite_path``) holds the
records and their lineage overflow. Behaviour matches ``FakeDocStatusStore``
and ``DynamoDBDocStatusStore`` (the shared conformance suite runs all three):
the same diff partition, ``stored_scopes`` and ``orphan_overflow``.

SQLite has no item limit, so every record fits (``record_fits``) and the
incremental run never needs overflow; the overflow methods are implemented
anyway so a run written by another store, or a test, round-trips. Each record
is stored as its JSON dump next to the columns ``diff`` reads, so the diff
never deserializes the artifact-id lists. Writes are transactions; a single
connection is shared under a lock (the pipeline calls the store from worker
threads) and waits up to ``search.store_timeout_seconds`` for another
writer's lock. Every SQLite error is raised as ``DocStatusRegistryError``, as
the DynamoDB store raises its errors.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path

from unified_kg_rag.adapters.local.sqlite_util import (
    connect,
    typed_errors,
    write_transaction,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    DocumentDelta,
    DocumentLineage,
)
from unified_kg_rag.shared import DocStatusRegistryError, get_logger

logger = get_logger(__name__)

_LINEAGE_FIELDS = (
    "entity_ids",
    "relationship_ids",
    "text_unit_ids",
    "community_ids",
    "claim_ids",
    "community_report_ids",
)

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS doc_status (
        doc_id TEXT PRIMARY KEY,
        content_hash TEXT NOT NULL,
        status TEXT NOT NULL,
        scope TEXT,
        record TEXT NOT NULL,
        extraction_fingerprint TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS lineage_overflow (
        doc_id TEXT NOT NULL,
        field TEXT NOT NULL,
        artifact_id TEXT NOT NULL,
        PRIMARY KEY (doc_id, field, artifact_id)
    )
    """,
)

# Columns added after the table shipped: a database created before them gets
# them on open, NULL for its existing rows (which then read as the record's
# default, e.g. no extraction fingerprint).
_ADDED_COLUMNS = {"extraction_fingerprint": "TEXT"}


def _add_missing_columns(connection: sqlite3.Connection) -> None:
    existing = {row[1] for row in connection.execute("PRAGMA table_info(doc_status)")}
    for name, kind in _ADDED_COLUMNS.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE doc_status ADD COLUMN {name} {kind}")


# SQLite's default limit on bound parameters per statement is 999 on older
# builds; stay under it when reading many keys.
_BATCH = 500


class SQLiteDocStatusStore:
    """SQLite-backed implementation of :class:`DocStatusPort`."""

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
                        migrate=_add_missing_columns,
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
                    f"SELECT doc_id, record FROM doc_status WHERE doc_id IN ({marks})",
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
                "INSERT OR REPLACE INTO doc_status (doc_id, content_hash, status, "
                "scope, record, extraction_fingerprint) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
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
            connection.execute("DELETE FROM doc_status WHERE doc_id = ?", (doc_id,))

    def list_all(self) -> list[DocStatusRecord]:
        with self._lock, self._registry_errors("read the records"):
            rows = self.connection.execute(
                "SELECT record FROM doc_status ORDER BY rowid"
            ).fetchall()
        return [DocStatusRecord.model_validate_json(row[0]) for row in rows]

    # --- lineage overflow --------------------------------------------------

    def add_lineage_overflow(self, lineages: Iterable[DocumentLineage]) -> None:
        rows = [
            (lineage.doc_id, name, artifact_id)
            for lineage in lineages
            for name in _LINEAGE_FIELDS
            for artifact_id in getattr(lineage, name)
        ]
        if not rows:
            return
        with self._transaction("write lineage overflow") as connection:
            connection.executemany(
                "INSERT OR IGNORE INTO lineage_overflow (doc_id, field, artifact_id) "
                "VALUES (?, ?, ?)",
                rows,
            )

    def get_lineage_overflow(
        self, doc_ids: Iterable[str]
    ) -> dict[str, DocumentLineage]:
        ids = list(dict.fromkeys(doc_ids))
        stored: dict[str, dict[str, list[str]]] = {}
        with self._lock, self._registry_errors("read lineage overflow"):
            for start in range(0, len(ids), _BATCH):
                batch = ids[start : start + _BATCH]
                marks = ",".join("?" * len(batch))
                rows = self.connection.execute(
                    "SELECT doc_id, field, artifact_id FROM lineage_overflow "
                    f"WHERE doc_id IN ({marks}) ORDER BY artifact_id",
                    batch,
                ).fetchall()
                for doc_id, name, artifact_id in rows:
                    stored.setdefault(doc_id, {}).setdefault(name, []).append(
                        artifact_id
                    )
        return {
            doc_id: DocumentLineage.model_validate(
                {
                    "doc_id": doc_id,
                    **{name: stored[doc_id].get(name, []) for name in _LINEAGE_FIELDS},
                }
            )
            for doc_id in ids
            if doc_id in stored
        }

    def delete_lineage_overflow(self, doc_ids: Iterable[str]) -> None:
        ids = list(dict.fromkeys(doc_ids))
        if not ids:
            return
        with self._transaction("delete lineage overflow") as connection:
            connection.executemany(
                "DELETE FROM lineage_overflow WHERE doc_id = ?",
                [(doc_id,) for doc_id in ids],
            )

    # --- diff --------------------------------------------------------------

    def diff(
        self,
        incoming: dict[str, str],
        scope: str | None = None,
        extraction_fingerprint: str | None = None,
    ) -> DocumentDelta:
        """Classify ``{doc_id: content_hash}`` against the stored records.

        Same partition as ``FakeDocStatusStore.diff``: with ``scope``, only
        stored records of that scope can be deleted; a FAILED record, or one
        whose extraction fingerprint is set and differs from
        ``extraction_fingerprint``, is changed even with an unchanged hash.
        """
        with self._lock, self._registry_errors("read the records"):
            rows = self.connection.execute(
                "SELECT doc_id, content_hash, status, scope, extraction_fingerprint "
                "FROM doc_status ORDER BY rowid"
            ).fetchall()
            overflow_owners = [
                row[0]
                for row in self.connection.execute(
                    "SELECT DISTINCT doc_id FROM lineage_overflow"
                ).fetchall()
            ]
        stored = {
            doc_id: (content_hash, status, stored_scope, stored_extraction)
            for doc_id, content_hash, status, stored_scope, stored_extraction in rows
        }
        delta = DocumentDelta()
        for doc_id, content_hash in incoming.items():
            existing = stored.get(doc_id)
            if existing is None:
                delta.new.append(doc_id)
            elif existing[0] != content_hash or existing[1] == DocStatus.FAILED.value:
                delta.changed.append(doc_id)
            elif extraction_fingerprint is not None and (
                existing[3] not in (None, extraction_fingerprint)
            ):
                delta.changed.append(doc_id)
                delta.settings_changed.append(doc_id)
            else:
                delta.unchanged.append(doc_id)
        incoming_ids = set(incoming)
        delta.deleted = [
            doc_id
            for doc_id, (_, _, stored_scope, _) in stored.items()
            if doc_id not in incoming_ids and (scope is None or stored_scope == scope)
        ]
        delta.stored_scopes = sorted(
            {s for _, _, s, _ in stored.values() if s is not None}
        )
        delta.orphan_overflow = sorted(
            owner
            for owner in overflow_owners
            if owner not in stored or stored[owner][1] != DocStatus.PENDING.value
        )
        return delta
