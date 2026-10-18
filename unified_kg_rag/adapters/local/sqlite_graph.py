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
import sqlite3
import threading
from collections.abc import Callable, Collection, Iterable, Iterator
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
from unified_kg_rag.shared.exceptions import StoreConfigurationError, StoreTimeoutError

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
# ``value_map``).
_NODE_KEYS = ("id", "name", "description", "rank", "text_unit_ids")

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
        """Close the connection (reopened on next use)."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

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
        """A write transaction, rolled back when the block raises."""
        with (
            self._lock,
            self.errors(operation),
            write_transaction(self.connection) as connection,
        ):
            yield connection

    # --- reads (the caller holds ``read`` or ``transaction``) ----------------

    @staticmethod
    def vertex_maps(
        connection: sqlite3.Connection, label: str, ids: Iterable[str]
    ) -> dict[str, dict[str, list[Any]]]:
        """``{id: valueMap}`` of the stored vertices of ``label`` among ``ids``."""
        found: dict[str, dict[str, list[Any]]] = {}
        for batch in batches(list(dict.fromkeys(ids))):
            rows = connection.execute(
                f"SELECT id, props FROM vertices WHERE label = ? "
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
                f"SELECT source_id FROM edges WHERE label = ? AND source_id IN ({marks}) "
                "UNION ALL "
                f"SELECT target_id FROM edges WHERE label = ? AND target_id IN ({marks})",
                [label, *batch, label, *batch],
            ).fetchall()
            for (vertex_id,) in rows:
                counts[vertex_id] = counts.get(vertex_id, 0) + 1
        return counts

    def traverse(
        self,
        label: str,
        seed_ids: list[str],
        *,
        max_hops: int,
        max_results_per_hop: int,
        per_seed_limit: int,
        keep: VertexFilter | None = None,
        with_degree: bool = False,
    ) -> list[dict[str, Any]]:
        """Rows of the seeds and of the vertices a walk from each one reaches.

        The walk is NeptuneRetriever's Gremlin traversal: per seed, a
        breadth-first walk of ``max_hops`` over ``label``'s edges, taking each
        vertex's first ``max_results_per_hop`` neighbours by id, skipping the
        seeds and the vertices this seed already reached, and stopping after
        ``per_seed_limit`` vertices. ``keep`` filters the rows (seeds
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
                        per_seed_limit,
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
                    {"name": maps[v]["name"]} if "name" in maps[v] else {} for v in walk
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
