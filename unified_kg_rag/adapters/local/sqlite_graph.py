# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedded knowledge-graph store on SQLite (``providers.graph.kind: sqlite``).

The local counterpart of Neptune for a single host: one database file
(``providers.graph.sqlite_path``) holds what the Neptune indexer writes, with
the same semantics. This module is the store (schema, transactions, reads and
the traversal); ``adapters.storage.sqlite_graph_indexer`` writes to it and
``adapters.retrievers.sqlite_graph_retriever`` reads from it:

- ``vertices(label, id, props)``: entity and community vertices. ``label`` is
  the Neptune vertex label (``graph_label``), so ``suffix`` and
  ``indexing.additional_suffix`` isolate data as they do in Neptune. ``props``
  is the vertex ``valueMap()`` as JSON: every value a list, list properties
  with set cardinality.
- ``edges(label, id, type, source_id, target_id, props)``: relationship edges.
  ``label`` is the entity label of both endpoints (an edge never leaves its
  suffix), ``type`` the edge label, ``props`` the edge properties.

Dropping a vertex drops its incident edges. Each store uses
one connection behind a lock, in WAL mode (each read is one snapshot), and
waits up to ``search.store_timeout_seconds`` for another writer's lock.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Any

from unified_kg_rag.adapters.local.sqlite_util import (
    connect,
    read_snapshot,
    typed_errors,
    write_transaction,
)
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import (
    GraphQueryCompileError,
    GraphQueryExecutionError,
    StoreConfigurationError,
    StoreTimeoutError,
)
from unified_kg_rag.shared.logging import log_error
from unified_kg_rag.shared.write_fence import check_fences_held, check_write_fence

logger = get_logger(__name__)

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS vertices (
        label TEXT NOT NULL,
        id TEXT NOT NULL,
        props TEXT NOT NULL,
        PRIMARY KEY (label, id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS vertices_id ON vertices (id)",
    """
    CREATE TABLE IF NOT EXISTS edges (
        label TEXT NOT NULL,
        id TEXT NOT NULL,
        type TEXT NOT NULL,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        props TEXT NOT NULL,
        PRIMARY KEY (label, id)
    )
    """,
    # With the other endpoint last, a vertex's neighbours come off the index
    # in order (``neighbors``' ORDER BY ... LIMIT reads only them).
    "CREATE INDEX IF NOT EXISTS edges_source_target "
    "ON edges (label, source_id, target_id)",
    "CREATE INDEX IF NOT EXISTS edges_target_source "
    "ON edges (label, target_id, source_id)",
    "CREATE INDEX IF NOT EXISTS edges_id ON edges (id)",
)

# Stay under SQLite's bound-parameter limit (999 on older builds): a
# statement binds at most 2 * BATCH_SIZE + 2 values, or splits its batch.
BATCH_SIZE = 400

# The vertex properties a traversal row projects (NeptuneRetriever's
# ``value_map``): the seed-graph fields only where a vertex holds them.
_NODE_KEYS = (
    "id",
    "name",
    "type",
    "description",
    "rank",
    "text_unit_ids",
    "community_ids",
    "iri",
    "source_kinds",
)

# The tables a structured query may read (``query_read_only``'s authorizer).
_QUERY_TABLES = frozenset({"vertices", "edges"})

# A vertex filter for a traversal: its stored valueMap -> kept.
VertexFilter = Callable[[dict[str, Any]], bool]


def batches(items: list[Any], size: int = BATCH_SIZE) -> Iterator[list[Any]]:
    """``items`` in slices of at most ``size`` (one statement's bound ids)."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def placeholders(items: Collection[Any]) -> str:
    """One ``?`` per item, comma-separated, for an ``IN (...)`` list."""
    return ",".join("?" * len(items))


class SQLiteGraphStore:
    """The SQLite database of one graph: schema, transactions and reads.

    Shared by ``SQLiteGraphIndexer`` (``adapters.storage``, writes) and
    ``SQLiteGraphRetriever`` (``adapters.retrievers``, traversal). Every error is raised typed:
    :class:`StoreTimeoutError` when the database stayed locked for
    ``timeout_seconds``, :class:`StoreConfigurationError` when it cannot be
    opened, read or written (a corrupt file, an unwritable path).
    """

    def __init__(self, path: str | Path, *, timeout_seconds: float = 30.0) -> None:
        """Args:
        path: Database file (created on first use), or ``":memory:"`` for a
            private in-memory one.
        timeout_seconds: How long a statement waits for another
            connection's lock before failing with ``StoreTimeoutError``.
        """
        self.path = str(path)
        self.timeout_seconds = timeout_seconds
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        # The read-only connection of structured queries (query_read_only).
        self._read_only: sqlite3.Connection | None = None
        self._read_only_lock = threading.Lock()

    @classmethod
    def from_config(cls, config: Config) -> SQLiteGraphStore:
        """The store at ``providers.graph.sqlite_path``."""
        return cls(
            config.providers.graph.sqlite_path,
            timeout_seconds=config.search.store_timeout_seconds,
        )

    @property
    def connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._connection is None:
                with self.errors("open the database"):
                    if self.path != ":memory:":
                        logger.info(
                            "Graph store: SQLite database '%s'",
                            Path(self.path).resolve(),
                        )
                    self._connection = connect(
                        self.path,
                        timeout_seconds=self.timeout_seconds,
                        schema=_SCHEMA,
                    )
            return self._connection

    def close(self) -> None:
        """Close the connections (reopened on next use)."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
        with self._read_only_lock:
            if self._read_only is not None:
                self._read_only.close()
                self._read_only = None

    def errors(self, operation: str) -> AbstractContextManager[None]:
        """Re-raise SQLite and file errors as typed store errors."""
        return typed_errors(
            lambda e: StoreTimeoutError(
                f"Graph store: could not {operation}: SQLite database "
                f"'{self.path}' stayed locked for {self.timeout_seconds}s "
                f"({e}). Another process is writing to it; SQLite allows "
                "one writer at a time."
            ),
            lambda e: StoreConfigurationError(
                f"Graph store: could not {operation} on SQLite database "
                f"'{self.path}': {type(e).__name__}: {e}. Check the path "
                "(providers.graph.sqlite_path) is a writable SQLite file on a "
                "local filesystem."
            ),
        )

    @contextmanager
    def read(self, operation: str) -> Iterator[sqlite3.Connection]:
        """The connection, under the lock, with errors typed, in one read
        transaction: every statement of the block reads the same snapshot, so
        another connection's write between two of them is not seen half way
        (a vertex a neighbour query returned is still there to read)."""
        with (
            self._lock,
            self.errors(operation),
            read_snapshot(self.connection) as connection,
        ):
            yield connection

    @contextmanager
    def transaction(self, operation: str) -> Iterator[sqlite3.Connection]:
        """A write transaction, rolled back when the block raises. Fenced
        like a store request (``shared.write_fence``): the takeover wait
        before the lock, the fence once the database's write lock is held,
        so a write of a run that gave its lease up stops, and one past the
        check commits before the next holder's first write can begin."""
        check_write_fence()
        with (
            self._lock,
            self.errors(operation),
            write_transaction(self.connection) as connection,
        ):
            check_fences_held()
            yield connection

    # --- structured queries (docs/design.md §6.6) ------------------------------

    @contextmanager
    def _query_connection(self) -> Iterator[sqlite3.Connection]:
        """A read-only connection, held for one statement: a connection of
        its own opened ``mode=ro`` with ``query_only`` (so a structured query
        never waits on, or blocks, the store's own connection), or, for an
        in-memory database, the store's connection with ``query_only`` on
        for the statement."""
        if self.path == ":memory:":
            with self._lock:
                connection = self.connection
                connection.execute("PRAGMA query_only = ON")
                try:
                    yield connection
                finally:
                    connection.execute("PRAGMA query_only = OFF")
            return
        with self._read_only_lock:
            if self._read_only is None:
                self.connection  # noqa: B018 - creates the file and its schema
                with self.errors("open the database read-only"):
                    connection = sqlite3.connect(
                        f"{Path(self.path).resolve().as_uri()}?mode=ro",
                        uri=True,
                        timeout=self.timeout_seconds,
                        check_same_thread=False,
                        isolation_level=None,
                    )
                    # nosemgrep: python.sqlalchemy.security.sqlalchemy-execute-raw-query.sqlalchemy-execute-raw-query, python.lang.security.audit.formatted-sql-query.formatted-sql-query - an int() value
                    connection.execute(
                        f"PRAGMA busy_timeout = {int(self.timeout_seconds * 1000)}"
                    )
                    connection.execute("PRAGMA query_only = ON")
                self._read_only = connection
            yield self._read_only

    def query_read_only(
        self,
        sql: str,
        params: Sequence[Any],
        *,
        functions: Mapping[str, tuple[int, Callable[..., Any]]],
        timeout_seconds: float,
    ) -> list[tuple[Any, ...]]:
        """The rows of one checked, read-only structured query
        (``adapters.graph_query``), with ``functions`` (name -> argument
        count, implementation) registered for it alone, in one snapshot.

        Past ``timeout_seconds`` a progress handler stops the statement.

        Raises:
            GraphQueryExecutionError: The statement was stopped at its time
                limit (``timed_out``) or failed in a function it called.
            StoreTimeoutError: The database stayed locked.
            StoreConfigurationError: The database cannot be read.
        """
        deadline = time.monotonic() + timeout_seconds

        def past_deadline() -> int:
            return 1 if time.monotonic() > deadline else 0

        allowed = {name.lower() for name in functions} | {"count"}

        def authorize(
            action: int, first: str | None, second: str | None, *_: Any
        ) -> int:
            # A second wall behind the statement checker: reads of the two
            # store tables and calls of the registered functions only (no
            # schema table, no recursion, no other statement).
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and first in _QUERY_TABLES:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and (second or "").lower() in allowed:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY

        with self._query_connection() as connection:
            for name, (count, function) in functions.items():
                connection.create_function(name, count, function, deterministic=True)
            connection.set_progress_handler(past_deadline, 1000)
            try:
                with self.errors("run a structured query"):
                    try:
                        with read_snapshot(connection):
                            connection.set_authorizer(authorize)
                            try:
                                cursor = connection.execute(sql, list(params))
                                return list(cursor.fetchall())
                            finally:
                                connection.set_authorizer(None)
                    except sqlite3.DatabaseError as e:
                        message = str(e).lower()
                        if "not authorized" in message or "prohibited" in message:
                            log_error(
                                logger,
                                logging.ERROR,
                                "The structured query's authorizer refused a "
                                "statement the checker passed (a bug): %s",
                                error=e,
                            )
                            raise GraphQueryCompileError(
                                "the read-only authorizer refused the query"
                            ) from e
                        if not isinstance(e, sqlite3.OperationalError):
                            raise
                        if "interrupt" in message:
                            raise GraphQueryExecutionError(
                                "The graph store stopped the query at its time "
                                f"limit ({timeout_seconds} s)",
                                timed_out=True,
                            ) from e
                        if "user-defined function" in message:
                            raise GraphQueryExecutionError(
                                f"The structured query failed: {e}"
                            ) from e
                        raise
            finally:
                connection.set_progress_handler(None, 0)
                for name, (count, _) in functions.items():
                    connection.create_function(name, count, None, deterministic=True)

    def sample_profile(
        self, label: str, sample: int, edges_per_vertex: int
    ) -> tuple[list[dict[str, list[Any]]], list[tuple[str, str, str]]]:
        """The first ``sample`` vertices of ``label`` by id (value maps) and
        ``(source type, edge label, target type)`` of at most
        ``edges_per_vertex`` out-edges of each, by edge id."""
        with self.read("sample the graph profile") as connection:
            rows = connection.execute(
                "SELECT id, props FROM vertices WHERE label = ? ORDER BY id LIMIT ?",
                (label, max(1, sample)),
            ).fetchall()
            maps = {vertex_id: json.loads(props) for vertex_id, props in rows}
            out: list[tuple[str, str, str]] = []
            for batch in batches(list(maps)):
                edges = connection.execute(
                    "SELECT source_id, type, target_id FROM edges "  # nosec B608 - ? marks only, values bound
                    f"WHERE label = ? AND source_id IN ({placeholders(batch)}) "
                    "ORDER BY source_id, id",
                    [label, *batch],
                ).fetchall()
                out.extend(edges)
            targets = self.vertex_maps(
                connection, label, (t for _, _, t in out if t not in maps)
            )
        kept: list[tuple[str, str, str]] = []
        taken: dict[str, int] = {}
        for source, edge_type, target in out:
            if taken.get(source, 0) >= edges_per_vertex:
                continue
            taken[source] = taken.get(source, 0) + 1
            target_map = maps.get(target) or targets.get(target) or {}
            kept.append(
                (
                    str((maps[source].get("type") or [""])[0]),
                    str(edge_type),
                    str((target_map.get("type") or [""])[0]),
                )
            )
        return list(maps.values()), kept

    def read_vertex_maps(
        self, label: str, ids: Iterable[str], keys: Collection[str]
    ) -> dict[str, dict[str, list[Any]]]:
        """The ``keys`` of ``label``'s vertices among ``ids``."""
        with self.read("read vertices") as connection:
            found = self.vertex_maps(connection, label, ids)
        return {
            vertex_id: {k: list(v) for k, v in value_map.items() if k in keys and v}
            for vertex_id, value_map in found.items()
        }

    # --- reads (the caller holds ``read`` or ``transaction``) ----------------

    @staticmethod
    def vertex_maps(
        connection: sqlite3.Connection, label: str, ids: Iterable[str]
    ) -> dict[str, dict[str, list[Any]]]:
        """``{id: valueMap}`` of the stored vertices of ``label`` among ``ids``."""
        found: dict[str, dict[str, list[Any]]] = {}
        for batch in batches(list(dict.fromkeys(ids))):
            rows = connection.execute(
                f"SELECT id, props FROM vertices WHERE label = ? "  # nosec B608 - ? marks only, values bound
                f"AND id IN ({placeholders(batch)})",
                [label, *batch],
            ).fetchall()
            found.update((vertex_id, json.loads(props)) for vertex_id, props in rows)
        return found

    @staticmethod
    def neighbors(
        connection: sqlite3.Connection, label: str, vertex_id: str, limit: int
    ) -> list[str]:
        """The first ``limit`` neighbours of a vertex over its edges, by id.

        One entry per incident edge, as Gremlin's ``both()``: a vertex two
        edges connect appears twice.
        """
        rows = connection.execute(
            "SELECT target_id AS n FROM edges WHERE label = ? AND source_id = ? "
            "UNION ALL "
            "SELECT source_id AS n FROM edges WHERE label = ? AND target_id = ? "
            "ORDER BY n LIMIT ?",
            (label, vertex_id, label, vertex_id, limit),
        ).fetchall()
        return [row[0] for row in rows]

    @staticmethod
    def degrees(
        connection: sqlite3.Connection,
        label: str,
        ids: Iterable[str],
    ) -> dict[str, int]:
        """Incident relationship edges of each vertex (``both().count()``)."""
        counts: dict[str, int] = {}
        # The batch is bound twice.
        for batch in batches(list(dict.fromkeys(ids)), BATCH_SIZE // 2):
            marks = placeholders(batch)
            rows = connection.execute(
                f"SELECT source_id FROM edges WHERE label = ? AND source_id IN ({marks}) "  # nosec B608 - ? marks only, values bound
                "UNION ALL "
                f"SELECT target_id FROM edges WHERE label = ? AND target_id IN ({marks})",
                [label, *batch, label, *batch],
            ).fetchall()
            for (vertex_id,) in rows:
                counts[vertex_id] = counts.get(vertex_id, 0) + 1
        return counts

    def edges_within(
        self, label: str, ids: Iterable[str]
    ) -> list[tuple[str, str, str, dict[str, Any]]]:
        """``(id, source_id, target_id, props)`` of every edge of ``label``
        between two of ``ids``, in one statement: the edges leaving the
        sources are read off the index (``edges_source_target``) and their
        targets checked there, so only the edges within the set are read and
        decoded (a hub's thousands of others are index entries). The ``+``
        keeps SQLite from seeking every source-target pair instead, which
        costs the square of the set's size where the index read costs the
        sources' degrees. A set over :data:`BATCH_SIZE` is bound as one JSON
        array (``json_each``) rather than in batches, whose every pair re-read
        the sources' index entries: 20-30% faster at 1,000-2,000 members,
        where the bound ``?`` list is about 3% faster below it."""
        members = list(dict.fromkeys(str(i) for i in ids))
        if not members:
            return []
        if len(members) <= BATCH_SIZE:
            marks = placeholders(members)
            within, values = f"IN ({marks})", members
        else:
            within, values = "IN (SELECT value FROM json_each(?))", [
                json.dumps(members)
            ]
        with self.read("read neighbourhood edges") as connection:
            rows = connection.execute(
                "SELECT id, source_id, target_id, props FROM edges "  # nosec B608 - ? marks only, values bound
                f"WHERE label = ? AND source_id {within} AND +target_id {within}",
                [label, *values, *values],
            ).fetchall()
        return [
            (edge_id, source, target, json.loads(props))
            for edge_id, source, target, props in rows
        ]

    def traverse(
        self,
        label: str,
        seed_ids: list[str],
        *,
        max_hops: int,
        max_results_per_hop: int,
        per_seed_limit: int | Mapping[str, int],
        keep: VertexFilter | None = None,
        with_degree: bool = False,
    ) -> list[dict[str, Any]]:
        """Rows of the seeds and of the vertices a walk from each one reaches.

        The walk is NeptuneRetriever's Gremlin traversal: per seed, a
        breadth-first walk of ``max_hops`` over ``label``'s edges, taking each
        vertex's first ``max_results_per_hop`` neighbours by id, skipping the
        seeds and the vertices this seed already reached, and stopping after
        ``per_seed_limit`` vertices (with a seed id -> limit mapping, that
        seed's own: ``seed_shares``). ``keep`` filters the rows (seeds
        included); it does not change the walk. Rows have the shape
        ``GraphExpansionRetriever`` documents.
        """
        seeds = list(dict.fromkeys(str(s) for s in seed_ids))
        seed_set = set(seeds)
        with self.read("traverse the graph") as connection:
            stored_seeds = self.vertex_maps(connection, label, seeds)
            walks: list[list[str]] = []
            for seed in seeds:
                if seed not in stored_seeds:
                    continue
                walks.append([seed])
                walks.extend(
                    self._walk(
                        connection,
                        label,
                        seed,
                        seed_set,
                        max_hops,
                        max_results_per_hop,
                        (
                            per_seed_limit.get(seed, 0)
                            if isinstance(per_seed_limit, Mapping)
                            else per_seed_limit
                        ),
                    )
                )
            maps = self.vertex_maps(
                connection, label, (vertex for walk in walks for vertex in walk)
            )
            walks = [walk for walk in walks if keep is None or keep(maps[walk[-1]])]
            degrees = (
                self.degrees(connection, label, (walk[-1] for walk in walks))
                if with_degree
                else {}
            )
        rows = []
        for walk in walks:
            node = maps[walk[-1]]
            row: dict[str, Any] = {
                "node": {key: node[key] for key in _NODE_KEYS if key in node},
                "path": [
                    {
                        "id": [v],
                        **({"name": maps[v]["name"]} if "name" in maps[v] else {}),
                    }
                    for v in walk
                ],
                "node_type": label,
            }
            if with_degree:
                row["degree"] = degrees.get(walk[-1], 0)
            rows.append(row)
        return rows

    def _walk(
        self,
        connection: sqlite3.Connection,
        label: str,
        seed: str,
        seeds: set[str],
        max_hops: int,
        max_results_per_hop: int,
        limit: int,
    ) -> list[list[str]]:
        """Paths (seed first) to the vertices one seed's walk reaches."""
        reached: set[str] = set()
        paths: list[list[str]] = []
        frontier = [[seed]]
        for _ in range(max_hops):
            next_frontier: list[list[str]] = []
            for path in frontier:
                for neighbor in self.neighbors(
                    connection, label, path[-1], max_results_per_hop
                ):
                    if neighbor in seeds or neighbor in reached:
                        continue
                    if len(paths) >= limit:
                        return paths
                    reached.add(neighbor)
                    extended = [*path, neighbor]
                    paths.append(extended)
                    next_frontier.append(extended)
            if not next_frontier:
                break
            frontier = next_frontier
        return paths
