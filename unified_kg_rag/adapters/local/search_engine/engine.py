# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``SQLiteSearchEngine``: an embedded store that answers like OpenSearch.

It implements ``SearchEngineClient``, so the unchanged ``OpenSearchIndexer``
and ``OpenSearchRetriever`` run on it (``providers.vector.kind: sqlite``): both
backends share one copy of the indexing and retrieval logic. One SQLite file
holds the indices (mapping, settings and a write generation), the aliases,
the documents (``_source`` as JSON) and their vectors
(float32 blobs, one row per ``knn_vector`` field).

Searches run in memory. An index is loaded once per process into a
module-level cache keyed by (resolved database path, index name), shared by
every engine on the file (the indexer's and the retriever's), and reloaded
when its generation changes: every write transaction bumps it. kNN is exact
(numpy over the filtered documents), BM25 is Lucene's (``bm25``), text is
analyzed by the registry in ``analyzers``, and the query DSL is the subset in
``query``; anything else is rejected (``StoreRequestRejectedError``). Memory
is about documents x dimension x 4 bytes per vector field; loading an index of
more than 100,000 documents logs a warning.

There is no ``hybrid`` query and no search pipeline: ``OpenSearchRetriever``
fuses a lexical and a kNN search itself
(``domain.retrieval.hybrid_normalization``). Queries return raw scores; ties
break by document id.

Concurrency: one connection per engine behind a re-entrant lock, WAL
journaling and a busy timeout of ``timeout_seconds``
(``search.store_timeout_seconds``). SQLite allows one writer at a time, so a
database still locked after the timeout raises ``StoreTimeoutError``; a
corrupt or unwritable file raises ``StoreConfigurationError``. Network
filesystems are not supported. A missing index raises opensearch-py's
``NotFoundError(404, "index_not_found_exception", ...)``, which
``is_index_not_found`` recognizes as for a real cluster. ``refresh`` is
ignored: a write is visible as soon as it commits.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import uuid
import weakref
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

import numpy as np
from opensearchpy.exceptions import NotFoundError
from opensearchpy.serializer import JSONSerializer

from unified_kg_rag.adapters.local.search_engine.query import (
    NUMERIC_TYPES,
    SPACE_TYPES,
    LoadedIndex,
    VectorField,
    evaluate,
    field_specs,
    rejected,
    sort_hits,
    top_hits,
)
from unified_kg_rag.adapters.local.sqlite_util import (
    connect,
    read_snapshot,
    typed_errors,
    write_transaction,
)
from unified_kg_rag.shared import AWSServiceError, get_logger
from unified_kg_rag.shared.exceptions import (
    StoreConfigurationError,
    StoreTimeoutError,
)

logger = get_logger(__name__)

LARGE_INDEX_DOCS = 100_000
_DEFAULT_SIZE = 10
_DEFAULT_MAX_RESULT_WINDOW = 10_000
# Bound parameters per statement stay under SQLite's older 999 limit.
_BATCH = 500

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS indices (
        name TEXT PRIMARY KEY, mappings TEXT NOT NULL, settings TEXT NOT NULL,
        generation INTEGER NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS aliases (
        alias TEXT NOT NULL, index_name TEXT NOT NULL,
        PRIMARY KEY (alias, index_name))""",
    """CREATE TABLE IF NOT EXISTS docs (
        index_name TEXT NOT NULL, id TEXT NOT NULL, source TEXT NOT NULL,
        PRIMARY KEY (index_name, id))""",
    """CREATE TABLE IF NOT EXISTS vectors (
        index_name TEXT NOT NULL, id TEXT NOT NULL, field TEXT NOT NULL,
        vec BLOB NOT NULL, PRIMARY KEY (index_name, id, field))""",
    # The last generation handed out, so a re-created index never reuses the
    # generation of the one it replaces (a cached copy would look current).
    """CREATE TABLE IF NOT EXISTS counters (
        name TEXT PRIMARY KEY, value INTEGER NOT NULL)""",
)

_SERIALIZER = JSONSerializer()

# (resolved database path, index name) -> the loaded index.
_INDEX_CACHE: dict[tuple[str, str], LoadedIndex] = {}
_CACHE_LOCK = threading.Lock()


def _dumps(value: Any) -> str:
    return json.dumps(value, default=_SERIALIZER.default, ensure_ascii=False)


def _as_json(document: dict[str, Any], knn_fields: set[str]) -> dict[str, Any]:
    """``document`` with its values as JSON types (numpy values, dates, ...
    as the OpenSearch client sends them), through a JSON round trip that
    skips a vector given as a list: it is stored as float32 either way, and
    its round trip cost more than the rest of indexing the document."""
    vectors = {
        k: v for k, v in document.items() if k in knn_fields and isinstance(v, list)
    }
    rest = {k: v for k, v in document.items() if k not in vectors}
    return {**json.loads(_dumps(rest)), **vectors}


def index_not_found(name: str) -> NotFoundError:
    """The error opensearch-py raises for a missing index (a 404)."""
    return NotFoundError(
        404,
        "index_not_found_exception",
        {
            "error": {
                "type": "index_not_found_exception",
                "reason": f"no such index [{name}]",
                "index": name,
            },
            "status": 404,
        },
    )


def _names(value: str | list[str]) -> list[str]:
    parts = value if isinstance(value, list) else value.split(",")
    return [part.strip() for part in parts if part.strip()]


def _is_pattern(name: str) -> bool:
    return any(ch in name for ch in "*?[")


class _Patterns:
    """Field names and ``*?[`` patterns: plain names match by set membership."""

    __slots__ = ("names", "patterns")

    def __init__(self, values: list[str]) -> None:
        self.names = {value for value in values if not _is_pattern(value)}
        self.patterns = [value for value in values if _is_pattern(value)]

    def match(self, field: str) -> bool:
        return field in self.names or any(
            fnmatchcase(field, pattern) for pattern in self.patterns
        )


class _SourceFilter:
    """The top-level ``_source`` fields a search returns: those matching an
    ``includes`` pattern (all without one) and no ``excludes`` pattern.

    Asked per field of every hit, so each field name's answer is kept for
    the search (a page of hits repeats the same few names)."""

    __slots__ = ("_excludes", "_includes", "_kept")

    def __init__(self, includes: list[str] | None, excludes: list[str]) -> None:
        self._includes = None if includes is None else _Patterns(includes)
        self._excludes = _Patterns(excludes)
        self._kept: dict[str, bool] = {}

    def keeps(self, field: str) -> bool:
        kept = self._kept.get(field)
        if kept is None:
            kept = self._kept[field] = (
                self._includes is None or self._includes.match(field)
            ) and not self._excludes.match(field)
        return kept


def _merge(target: dict[str, Any], partial: dict[str, Any]) -> dict[str, Any]:
    """``target`` with ``partial`` merged in, objects recursively (a partial update)."""
    merged = dict(target)
    for key, value in partial.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _id_values(clause: Any) -> list[str] | None:
    """The ids an ``ids`` query or an ``id`` term(s) query names, else None."""
    if not isinstance(clause, dict) or len(clause) != 1:
        return None
    ((name, body),) = clause.items()
    if not isinstance(body, dict) or len(body) != 1:
        return None
    ((key, value),) = body.items()
    if name == "ids" and key == "values" and isinstance(value, list):
        return [str(v) for v in value]
    if name == "terms" and key == "id" and isinstance(value, list):
        return [str(v) for v in value]
    if name == "term" and key == "id" and isinstance(value, str | int):
        return [str(value)]
    return None


def _id_lookup(query: Any) -> tuple[list[str], float] | None:
    """``(ids, score)`` when ``query`` only looks documents up by id.

    A top-level ``ids`` query scores 1, a filter-only ``bool`` of one id
    clause 0. The indexer writes every document's ``id`` as its ``_id``, so
    both name the same documents.
    """
    if not isinstance(query, dict):
        return None
    if "ids" in query:
        ids = _id_values(query)
        return (ids, 1.0) if ids is not None else None
    body = query.get("bool")
    if len(query) != 1 or not isinstance(body, dict) or set(body) != {"filter"}:
        return None
    filters = body["filter"]
    filters = [filters] if isinstance(filters, dict) else filters
    if not isinstance(filters, list) or len(filters) != 1:
        return None
    ids = _id_values(filters[0])
    return (ids, 0.0) if ids is not None else None


def _validate_mappings(body: dict[str, Any]) -> None:
    unknown = set(body) - {"settings", "mappings"}
    if unknown:
        raise rejected(f"create index does not support {sorted(unknown)}")
    for path, spec in field_specs(body.get("mappings") or {}).items():
        if spec.type != "knn_vector":
            continue
        if not isinstance(spec.dimension, int) or spec.dimension < 1:
            raise rejected(f"knn_vector field [{path}] needs a positive dimension")
        if spec.space_type not in SPACE_TYPES:
            raise rejected(
                f"knn_vector field [{path}] uses space_type [{spec.space_type}]; "
                f"supported: {', '.join(sorted(SPACE_TYPES))}"
            )


class SQLiteSearchEngine:
    """The ``SearchEngineClient`` of an embedded SQLite file (see the module doc)."""

    def __init__(self, path: str | Path, *, timeout_seconds: float = 30.0) -> None:
        """Args:
        path: Database file (created on first use with its directory), or
            ``":memory:"`` for a private in-memory one.
        timeout_seconds: How long a statement waits for another writer's
            lock before ``StoreTimeoutError``.
        """
        self.path = str(path)
        self.timeout_seconds = timeout_seconds
        self._memory = self.path == ":memory:"
        # An in-memory database is private to this engine: so are its cached
        # indices, freed when the engine closes or is collected.
        self._cache_path = (
            f":memory:{uuid.uuid4().hex}"
            if self._memory
            else str(Path(self.path).resolve())
        )
        if self._memory:
            weakref.finalize(self, _drop_cached, self._cache_path)
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    # --- connection and errors --------------------------------------------

    def _errors(self, operation: str) -> AbstractContextManager[None]:
        """Re-raise SQLite and file errors as the store's typed errors."""
        return typed_errors(
            lambda e: StoreTimeoutError(
                f"Embedded search engine: could not {operation}: the SQLite "
                f"database '{self.path}' stayed locked by another writer for "
                f"{self.timeout_seconds} s (search.store_timeout_seconds). "
                "SQLite allows one writer at a time: wait for the other run, "
                "or raise the timeout."
            ),
            lambda e: StoreConfigurationError(
                f"Embedded search engine: could not {operation} on the SQLite "
                f"database '{self.path}': {type(e).__name__}: {e}. Check "
                "that the file (providers.vector.sqlite_path) is a database "
                "this release wrote and that its directory is writable "
                "(network filesystems are not supported)."
            ),
        )

    @property
    def connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._connection is None:
                with self._errors("open the database"):
                    connection = connect(
                        self.path,
                        timeout_seconds=self.timeout_seconds,
                        schema=_SCHEMA,
                    )
                if not self._memory:
                    logger.info(
                        "Vector store: embedded SQLite search engine '%s'",
                        self._cache_path,
                    )
                self._connection = connection
            return self._connection

    @contextmanager
    def _transaction(self, operation: str) -> Iterator[sqlite3.Connection]:
        with (
            self._lock,
            self._errors(operation),
            write_transaction(self.connection) as connection,
        ):
            yield connection

    @contextmanager
    def _reading(self, operation: str) -> Iterator[sqlite3.Connection]:
        """One read transaction: every read in it sees one snapshot of the
        file (in WAL mode a reader does not block the writer). A nested block
        shares the outer one."""
        with (
            self._lock,
            self._errors(operation),
            read_snapshot(self.connection) as connection,
        ):
            yield connection

    def close(self) -> None:
        """Close the connection (reopened on next use). Never raises."""
        with self._lock:
            if self._connection is not None:
                try:
                    self._connection.close()
                except sqlite3.Error as e:
                    logger.debug("Error closing the SQLite search engine: %s", e)
                self._connection = None
        if self._memory:
            _drop_cached(self._cache_path)

    async def aclose(self) -> None:
        self.close()

    # --- catalog helpers ----------------------------------------------------

    @staticmethod
    def _next_generation(connection: sqlite3.Connection) -> int:
        connection.execute(
            "INSERT INTO counters (name, value) VALUES ('generation', 1) "
            "ON CONFLICT(name) DO UPDATE SET value = value + 1"
        )
        row = connection.execute(
            "SELECT value FROM counters WHERE name = 'generation'"
        ).fetchone()
        return int(row[0])

    def _bump(self, connection: sqlite3.Connection, index: str) -> None:
        connection.execute(
            "UPDATE indices SET generation = ? WHERE name = ?",
            (self._next_generation(connection), index),
        )

    @staticmethod
    def _index_names(connection: sqlite3.Connection) -> list[str]:
        return [
            r[0] for r in connection.execute("SELECT name FROM indices ORDER BY name")
        ]

    @staticmethod
    def _alias_rows(connection: sqlite3.Connection) -> list[tuple[str, str]]:
        return [
            (r[0], r[1])
            for r in connection.execute(
                "SELECT alias, index_name FROM aliases ORDER BY alias, index_name"
            )
        ]

    @staticmethod
    def _mappings_of(connection: sqlite3.Connection, index: str) -> dict[str, Any]:
        row = connection.execute(
            "SELECT mappings FROM indices WHERE name = ?", (index,)
        ).fetchone()
        if row is None:
            raise index_not_found(index)
        mappings: dict[str, Any] = json.loads(row[0])
        return mappings

    def _resolve(
        self, connection: sqlite3.Connection, target: str | list[str]
    ) -> list[str]:
        """The concrete indices ``target`` names: indices, aliases or patterns.

        Raises:
            NotFoundError: A name (not a pattern) is neither an index nor an alias.
        """
        indices = self._index_names(connection)
        aliases = self._alias_rows(connection)
        resolved: dict[str, None] = {}
        for name in _names(target):
            if _is_pattern(name):
                resolved.update(
                    dict.fromkeys(i for i in indices if fnmatchcase(i, name))
                )
                resolved.update(
                    dict.fromkeys(i for a, i in aliases if fnmatchcase(a, name))
                )
            elif name in indices:
                resolved[name] = None
            elif members := [i for a, i in aliases if a == name]:
                resolved.update(dict.fromkeys(members))
            else:
                raise index_not_found(name)
        return list(resolved)

    # --- writes -------------------------------------------------------------

    def _write_vectors(
        self,
        connection: sqlite3.Connection,
        index: str,
        doc_id: str,
        vectors: dict[str, list[float]],
    ) -> None:
        connection.executemany(
            "INSERT OR REPLACE INTO vectors (index_name, id, field, vec) "
            "VALUES (?, ?, ?, ?)",
            [
                (index, doc_id, name, np.asarray(vec, dtype=np.float32).tobytes())
                for name, vec in vectors.items()
            ],
        )

    @staticmethod
    def _document_error(
        op: str, index: str, doc_id: str, reason: str
    ) -> dict[str, Any]:
        return {
            op: {
                "_index": index,
                "_id": doc_id,
                "status": 400,
                "error": {"type": "mapper_parsing_exception", "reason": reason},
            }
        }

    @staticmethod
    def _invalid_field(
        specs: dict[str, Any], doc_id: str, document: dict[str, Any]
    ) -> str | None:
        """Why ``document`` cannot be indexed under ``specs``, if it cannot."""
        for name, value in document.items():
            spec = specs.get(name)
            if spec is None or value is None:
                continue
            if spec.type == "knn_vector":
                if not isinstance(value, list) or len(value) != spec.dimension:
                    given = (
                        len(value) if isinstance(value, list) else type(value).__name__
                    )
                    return (
                        f"failed to parse field [{name}] of type [knn_vector] in "
                        f"document with id '{doc_id}'. Vector dimension mismatch. "
                        f"Expected: {spec.dimension}, Given: {given}"
                    )
                if spec.space_type == "cosinesimil" and not any(value):
                    return (
                        f"zero vector is not supported in field [{name}] when "
                        "space type is [cosinesimil]"
                    )
            elif spec.type in NUMERIC_TYPES:
                for element in value if isinstance(value, list) else [value]:
                    try:
                        float(element)
                    except (TypeError, ValueError):
                        return (
                            f"failed to parse field [{name}] of type [{spec.type}] "
                            f"in document with id '{doc_id}'"
                        )
        return None

    def bulk_index(
        self, index: str, documents: list[dict[str, Any]], refresh: bool = True
    ) -> dict[str, Any]:
        """Index (create or replace) ``documents`` by their ``id``.

        Returns ``{"errors", "items"}`` with only the rejected documents in
        ``items``, as ``OpenSearchClient.bulk_index`` does.
        """
        if not documents:
            return {"errors": False, "items": []}
        failed: list[dict[str, Any]] = []
        with self._transaction("index documents") as connection:
            specs = field_specs(self._mappings_of(connection, index))
            knn_fields = {p for p, s in specs.items() if s.type == "knn_vector"}
            for document in documents:
                raw_id = document.get("id")
                doc_id = str(raw_id) if raw_id is not None else uuid.uuid4().hex
                document = _as_json(document, knn_fields)
                if reason := self._invalid_field(specs, doc_id, document):
                    failed.append(self._document_error("index", index, doc_id, reason))
                    continue
                source = {k: v for k, v in document.items() if k not in knn_fields}
                connection.execute(
                    "INSERT OR REPLACE INTO docs (index_name, id, source) "
                    "VALUES (?, ?, ?)",
                    (index, doc_id, _dumps(source)),
                )
                connection.execute(
                    "DELETE FROM vectors WHERE index_name = ? AND id = ?",
                    (index, doc_id),
                )
                self._write_vectors(
                    connection,
                    index,
                    doc_id,
                    {
                        k: v
                        for k, v in document.items()
                        if k in knn_fields and v is not None
                    },
                )
            self._bump(connection, index)
        if failed:
            logger.warning(
                "Bulk index completed with %s errors in '%s'", len(failed), index
            )
        return {"errors": bool(failed), "items": failed}

    def bulk_delete(
        self, index: str, ids: list[str], refresh: bool = True
    ) -> dict[str, Any]:
        """Delete documents by id; a missing id counts as deleted."""
        if not ids:
            return {"errors": False, "items": []}
        rows = [(index, str(doc_id)) for doc_id in ids]
        with self._transaction("delete documents") as connection:
            self._mappings_of(connection, index)
            connection.executemany(
                "DELETE FROM docs WHERE index_name = ? AND id = ?", rows
            )
            connection.executemany(
                "DELETE FROM vectors WHERE index_name = ? AND id = ?", rows
            )
            self._bump(connection, index)
        return {"errors": False, "items": []}

    def bulk_update(
        self, index: str, documents: list[dict[str, Any]], refresh: bool = True
    ) -> dict[str, Any]:
        """Merge the given fields into documents by ``id`` (a missing one is
        skipped). Vectors are kept unless a ``knn_vector`` field is given."""
        if not documents:
            return {"errors": False, "items": []}
        failed: list[dict[str, Any]] = []
        with self._transaction("update documents") as connection:
            specs = field_specs(self._mappings_of(connection, index))
            knn_fields = {p for p, s in specs.items() if s.type == "knn_vector"}
            for document in documents:
                doc_id = str(document["id"])
                partial = _as_json(
                    {k: v for k, v in document.items() if k != "id"}, knn_fields
                )
                row = connection.execute(
                    "SELECT source FROM docs WHERE index_name = ? AND id = ?",
                    (index, doc_id),
                ).fetchone()
                if row is None:
                    continue
                if reason := self._invalid_field(specs, doc_id, partial):
                    failed.append(self._document_error("update", index, doc_id, reason))
                    continue
                source = _merge(
                    json.loads(row[0]),
                    {k: v for k, v in partial.items() if k not in knn_fields},
                )
                connection.execute(
                    "UPDATE docs SET source = ? WHERE index_name = ? AND id = ?",
                    (_dumps(source), index, doc_id),
                )
                self._write_vectors(
                    connection,
                    index,
                    doc_id,
                    {k: v for k, v in partial.items() if k in knn_fields and v},
                )
            self._bump(connection, index)
        return {"errors": bool(failed), "items": failed}

    # --- index and alias administration -------------------------------------

    def create_index(
        self, index: str, body: dict[str, Any], *, exist_ok: bool = True
    ) -> None:
        """Create ``index``; an existing one is kept, or raises unless ``exist_ok``."""
        _validate_mappings(body)
        with self._transaction("create an index") as connection:
            if index in self._index_names(connection):
                if exist_ok:
                    return
                raise AWSServiceError(f"Index '{index}' already exists")
            if any(alias == index for alias, _ in self._alias_rows(connection)):
                raise rejected(
                    f"invalid_index_name_exception: [{index}] is an alias name"
                )
            connection.execute(
                "INSERT INTO indices (name, mappings, settings, generation) "
                "VALUES (?, ?, ?, ?)",
                (
                    index,
                    _dumps(body.get("mappings") or {}),
                    _dumps(body.get("settings") or {}),
                    self._next_generation(connection),
                ),
            )
        logger.info("Created index: '%s'", index)

    def delete_alias(
        self, index_names: str | list[str], alias_names: str | list[str]
    ) -> None:
        index_patterns, alias_patterns = _names(index_names), _names(alias_names)
        with self._transaction("delete an alias") as connection:
            rows = [
                (alias, index)
                for alias, index in self._alias_rows(connection)
                if any(fnmatchcase(alias, p) for p in alias_patterns)
                and any(fnmatchcase(index, p) for p in index_patterns)
            ]
            if not rows:
                raise NotFoundError(
                    404,
                    "aliases_not_found_exception",
                    {
                        "error": {
                            "type": "aliases_not_found_exception",
                            "reason": f"aliases [{','.join(alias_patterns)}] missing",
                        },
                        "status": 404,
                    },
                )
            connection.executemany(
                "DELETE FROM aliases WHERE alias = ? AND index_name = ?", rows
            )
        logger.info("Deleted alias(es) %s", ", ".join(sorted({a for a, _ in rows})))

    def delete_indices(self, index_names: list[str]) -> None:
        """Delete indices by name or pattern (a missing name is ignored),
        with their documents, vectors and aliases."""
        patterns = _names(index_names)
        with self._transaction("delete indices") as connection:
            doomed = [
                name
                for name in self._index_names(connection)
                if any(fnmatchcase(name, p) for p in patterns)
            ]
            for table, column in (
                ("docs", "index_name"),
                ("vectors", "index_name"),
                ("aliases", "index_name"),
                ("indices", "name"),
            ):
                connection.executemany(
                    f"DELETE FROM {table} WHERE {column} = ?",  # noqa: S608
                    [(name,) for name in doomed],
                )
        for name in doomed:
            _drop_cached(self._cache_path, name)
        if doomed:
            logger.info("Deleted indices: %s", ",".join(doomed))

    def get_index_mappings(self, index_name: str) -> dict[str, Any]:
        """The ``mappings`` of index ``index_name`` (``{}`` for an alias)."""
        with self._reading("read index mappings") as connection:
            if index_name not in self._index_names(connection):
                if index_name in {a for a, _ in self._alias_rows(connection)}:
                    return {}
                raise index_not_found(index_name)
            return self._mappings_of(connection, index_name)

    def put_index_meta(self, index_name: str, meta: dict[str, Any]) -> None:
        """Replace the mapping ``_meta`` of ``index_name`` (fields unchanged)."""
        with self._transaction("write index _meta") as connection:
            mappings = self._mappings_of(connection, index_name)
            mappings["_meta"] = meta
            connection.execute(
                "UPDATE indices SET mappings = ? WHERE name = ?",
                (_dumps(mappings), index_name),
            )
            self._bump(connection, index_name)

    def get_index_name_by_alias(self, alias_name: str) -> str | None:
        indices = self.get_indices_by_alias(alias_name)
        return indices[0] if indices else None

    def get_indices_by_alias(self, alias_name: str) -> list[str]:
        with self._reading("read aliases") as connection:
            return sorted(
                {
                    index
                    for alias, index in self._alias_rows(connection)
                    if any(fnmatchcase(alias, p) for p in _names(alias_name))
                }
            )

    def get_aliases_by_index(self, index_pattern: str) -> dict[str, list[str]]:
        """Map every index whose name matches ``index_pattern`` to its aliases."""
        patterns = _names(index_pattern)
        with self._reading("read aliases") as connection:
            aliases = self._alias_rows(connection)
            return {
                name: sorted(alias for alias, index in aliases if index == name)
                for name in self._index_names(connection)
                if any(fnmatchcase(name, p) for p in patterns)
            }

    def update_alias(
        self,
        alias_name: str,
        new_index_name: str,
        remove_pattern: str | None = None,
    ) -> None:
        """Point ``alias_name`` at ``new_index_name``, first removing it from the
        indices matching ``remove_pattern``, in one transaction."""
        with self._transaction("update an alias") as connection:
            indices = self._index_names(connection)
            if new_index_name not in indices:
                raise index_not_found(new_index_name)
            if alias_name in indices:
                raise rejected(
                    f"invalid_alias_name_exception: an index exists with the "
                    f"same name as the alias [{alias_name}]"
                )
            if remove_pattern:
                patterns = _names(remove_pattern)
                connection.executemany(
                    "DELETE FROM aliases WHERE alias = ? AND index_name = ?",
                    [
                        (alias, index)
                        for alias, index in self._alias_rows(connection)
                        if alias == alias_name
                        and any(fnmatchcase(index, p) for p in patterns)
                    ],
                )
            connection.execute(
                "INSERT OR IGNORE INTO aliases (alias, index_name) VALUES (?, ?)",
                (alias_name, new_index_name),
            )
        logger.info("Updated alias '%s' to point to '%s'", alias_name, new_index_name)

    # --- reads --------------------------------------------------------------

    def _load(self, index: str) -> LoadedIndex:
        """``index`` from the module cache, reloaded if a write changed it.

        The generation and the rows are read in one snapshot (``_reading``):
        a write committed meanwhile cannot leave vectors without documents.
        Cached indices of this file that another engine deleted are dropped.
        """
        key = (self._cache_path, index)
        with self._reading("read an index") as connection:
            generations = dict(
                connection.execute("SELECT name, generation FROM indices")
            )
            with _CACHE_LOCK:
                for stale in [
                    k
                    for k in _INDEX_CACHE
                    if k[0] == self._cache_path and k[1] not in generations
                ]:
                    del _INDEX_CACHE[stale]
                if index not in generations:
                    raise index_not_found(index)
                generation = int(generations[index])
                cached = _INDEX_CACHE.get(key)
                if cached is not None and cached.generation == generation:
                    return cached
                loaded = self._read_index(connection, index, generation)
                _INDEX_CACHE[key] = loaded
        return loaded

    def _read_index(
        self, connection: sqlite3.Connection, index: str, generation: int
    ) -> LoadedIndex:
        mappings_text, settings_text = connection.execute(
            "SELECT mappings, settings FROM indices WHERE name = ?", (index,)
        ).fetchone()
        mappings = json.loads(mappings_text)
        sources = {
            doc_id: json.loads(source)
            for doc_id, source in connection.execute(
                "SELECT id, source FROM docs WHERE index_name = ? ORDER BY id", (index,)
            )
        }
        if len(sources) > LARGE_INDEX_DOCS:
            logger.warning(
                "The embedded search engine loaded %s documents of index '%s' into "
                "memory; above about %s documents per index use OpenSearch "
                "(providers.vector.kind: opensearch)",
                len(sources),
                index,
                LARGE_INDEX_DOCS,
            )
        specs = field_specs(mappings)
        vectors: dict[str, VectorField] = {}
        for name, spec in specs.items():
            if spec.type != "knn_vector" or spec.dimension is None:
                continue
            rows = connection.execute(
                "SELECT id, vec FROM vectors WHERE index_name = ? AND field = ? "
                "ORDER BY id",
                (index, name),
            ).fetchall()
            ids = [r[0] for r in rows]
            if orphans := [doc_id for doc_id in ids if doc_id not in sources]:
                raise StoreConfigurationError(
                    f"Embedded search engine: index '{index}' of the SQLite "
                    f"database '{self.path}' has {len(orphans)} vector(s) of "
                    f"field [{name}] without a document (e.g. '{orphans[0]}'). "
                    "The file was changed outside this engine; re-index it "
                    "(indexing.reset: true)."
                )
            matrix = (
                np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
                if rows
                else np.zeros((0, spec.dimension), dtype=np.float32)
            )
            vectors[name] = VectorField(
                ids=ids,
                rows={doc_id: i for i, doc_id in enumerate(ids)},
                matrix=matrix,
                norms=np.linalg.norm(matrix, axis=1).astype(np.float32),
                dimension=spec.dimension,
                space_type=spec.space_type or "l2",
            )
        return LoadedIndex(
            name=index,
            generation=generation,
            mappings=mappings,
            settings=json.loads(settings_text),
            sources=sources,
            vectors=vectors,
        )

    def search(self, **kwargs: Any) -> dict[str, Any]:
        """Run ``search(index=..., body=...)``.

        Raises:
            NotFoundError: ``index`` names a missing index or alias.
            StoreRequestRejectedError: The body uses a clause or parameter the
                engine does not support.
        """
        unknown = set(kwargs) - {"index", "body"}
        if unknown:
            raise rejected(f"search does not support {sorted(unknown)}")
        body = kwargs.get("body") or {}
        if not isinstance(body, dict):
            raise rejected("the search body must be an object")
        unknown = set(body) - {"size", "query", "_source", "sort", "track_scores"}
        if unknown:
            raise rejected(f"the search body does not support {sorted(unknown)}")
        size = body.get("size", _DEFAULT_SIZE)
        if not isinstance(size, int) or size < 0:
            raise rejected(f"[size] must be a non-negative integer, got {size!r}")
        excludes = self._source_filter(body.get("_source", True))
        query = body.get("query") or {"match_all": {}}
        # Every read of one search from one snapshot: an index another
        # process deletes meanwhile is either searched whole or not found.
        with self._reading("search") as connection:
            names = self._resolve(connection, kwargs.get("index") or "*")
            for name in names:
                self._check_window(connection, name, size)
            sort = body.get("sort")
            if sort is None and (lookup := _id_lookup(query)) is not None:
                return self._fetch_by_ids(names, *lookup, size, excludes)
            loaded = [self._load(name) for name in names]
        hits = [
            (index, doc_id, score)
            for index in loaded
            for doc_id, score in evaluate(index, query).items()
        ]
        top = top_hits(hits, size) if sort is None else sort_hits(hits, sort, size)
        return self._response(
            [
                (index.name, doc_id, score, self._source(index, doc_id, excludes))
                for index, doc_id, score in top
            ],
            total=len(hits),
        )

    @staticmethod
    def _check_window(connection: sqlite3.Connection, index: str, size: int) -> None:
        (settings_text,) = connection.execute(
            "SELECT settings FROM indices WHERE name = ?", (index,)
        ).fetchone()
        settings = json.loads(settings_text)
        window = int(
            settings.get(
                "max_result_window",
                settings.get("index.max_result_window", _DEFAULT_MAX_RESULT_WINDOW),
            )
        )
        if size > window:
            raise rejected(
                f"Result window is too large, size [{size}] must be less than "
                f"or equal to: [{window}]"
            )

    def _fetch_by_ids(
        self,
        names: list[str],
        ids: list[str],
        score: float,
        size: int,
        excludes: _SourceFilter | None,
    ) -> dict[str, Any]:
        """Answer an id lookup from SQL, without loading the index."""
        wanted = sorted(set(ids))
        hits: list[tuple[str, str, float, dict[str, Any] | None]] = []
        with self._reading("read documents") as connection:
            for name in names:
                for start in range(0, len(wanted), _BATCH):
                    batch = wanted[start : start + _BATCH]
                    marks = ",".join("?" * len(batch))
                    sources = {
                        doc_id: json.loads(source)
                        for doc_id, source in connection.execute(
                            "SELECT id, source FROM docs WHERE index_name = ? "
                            f"AND id IN ({marks})",
                            (name, *batch),
                        )
                    }
                    if excludes is not None:
                        for doc_id, field, vec in connection.execute(
                            "SELECT id, field, vec FROM vectors WHERE index_name = ? "
                            f"AND id IN ({marks})",
                            (name, *batch),
                        ):
                            if doc_id in sources and excludes.keeps(field):
                                sources[doc_id][field] = np.frombuffer(
                                    vec, dtype=np.float32
                                ).tolist()
                    hits.extend(
                        (
                            name,
                            doc_id,
                            score,
                            (
                                None
                                if excludes is None
                                else {
                                    k: v
                                    for k, v in sources[doc_id].items()
                                    if excludes.keeps(k)
                                }
                            ),
                        )
                        for doc_id in batch
                        if doc_id in sources
                    )
        hits.sort(key=lambda hit: (hit[0], hit[1]))
        return self._response(hits[:size], total=len(hits))

    @staticmethod
    def _source(
        index: LoadedIndex, doc_id: str, excludes: _SourceFilter | None
    ) -> dict[str, Any] | None:
        if excludes is None:
            return None
        source = {k: v for k, v in index.sources[doc_id].items() if excludes.keeps(k)}
        for name, vectors in index.vectors.items():
            row = vectors.rows.get(doc_id)
            if row is not None and excludes.keeps(name):
                source[name] = vectors.matrix[row].tolist()
        return source

    @staticmethod
    def _source_filter(source: Any) -> _SourceFilter | None:
        """The ``_source`` setting's filter (None: no source at all)."""
        if source is True:
            return _SourceFilter(None, [])
        if source is False:
            return None
        if isinstance(source, dict) and set(source) <= {"includes", "excludes"}:

            def patterns(key: str) -> list[str]:
                value = source.get(key) or []
                return [value] if isinstance(value, str) else list(value)

            includes = patterns("includes") if "includes" in source else None
            return _SourceFilter(includes, patterns("excludes"))
        raise rejected(
            "[_source] supports only true, false or {includes: [...], "
            "excludes: [...]} of top-level fields"
        )

    @staticmethod
    def _response(
        hits: list[tuple[str, str, float, dict[str, Any] | None]], total: int
    ) -> dict[str, Any]:
        rendered = []
        for index, doc_id, score, source in hits:
            hit: dict[str, Any] = {"_index": index, "_id": doc_id, "_score": score}
            if source is not None:
                hit["_source"] = source
            rendered.append(hit)
        return {
            "took": 0,
            "timed_out": False,
            "hits": {
                "total": {"value": total, "relation": "eq"},
                "max_score": max((h["_score"] for h in rendered), default=None),
                "hits": rendered,
            },
        }

    async def asearch(self, **kwargs: Any) -> dict[str, Any]:
        return await asyncio.to_thread(self.search, **kwargs)

    def _mappings_by_index(self, index: str) -> dict[str, dict[str, Any]]:
        with self._reading("read index mappings") as connection:
            return {
                name: self._mappings_of(connection, name)
                for name in self._resolve(connection, index)
            }

    async def aget_mappings(self, index: str) -> dict[str, dict[str, Any]]:
        """The ``mappings`` of each index ``index`` (a name or alias) names."""
        return await asyncio.to_thread(self._mappings_by_index, index)

    def _exists(self, index: str) -> bool:
        with self._reading("check an index") as connection:
            try:
                return bool(self._resolve(connection, index))
            except NotFoundError:
                return False

    async def aindex_exists(self, index: str) -> bool:
        """Whether an index or alias named ``index`` exists."""
        return await asyncio.to_thread(self._exists, index)


def _drop_cached(path: str, index: str | None = None) -> None:
    with _CACHE_LOCK:
        for key in [k for k in _INDEX_CACHE if k[0] == path and index in (None, k[1])]:
            del _INDEX_CACHE[key]
