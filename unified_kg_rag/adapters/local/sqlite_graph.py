# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Embedded knowledge-graph store on SQLite (``providers.graph.kind: sqlite``).

The local counterpart of Neptune for a single host: one database file
(``providers.graph.sqlite_path``) holds what the Neptune indexer writes, with
the same semantics (``FakeGraphStore`` is the reference the tests compare
against):

- ``vertices(label, id, props)``: entity and community vertices. ``label`` is
  the Neptune vertex label (``graph_label``), so ``suffix`` and
  ``indexing.additional_suffix`` isolate data as they do in Neptune. ``props``
  is the vertex ``valueMap()`` as JSON: every value a list, list properties
  with set cardinality.
- ``edges(label, id, type, source_id, target_id, props)``: relationship edges.
  ``label`` is the entity label of both endpoints (an edge never leaves its
  suffix), ``type`` the edge label, ``props`` the edge properties.
- ``memberships``: the ``MemberOf`` edges from entities to communities.

Values are encoded and decoded by ``neptune_codec``, so
``indexing.neptune.property_max_length`` truncates strings as in Neptune.
Dropping a vertex drops its incident edges and memberships. Each store uses
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
from unified_kg_rag.adapters.storage.neptune_codec import (
    entity_from_vertex,
    entity_properties,
    item_properties,
    relationship_from_edge,
    relationship_label,
    relationship_properties,
)
from unified_kg_rag.domain.models import Community, Config, Entity, Relationship
from unified_kg_rag.ports.indexer import GraphIndexer, IndexingStats
from unified_kg_rag.shared import GraphRAGException, get_logger
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
    # The two-column indexes of earlier releases, which the new ones cover.
    "DROP INDEX IF EXISTS edges_source",
    "DROP INDEX IF EXISTS edges_target",
    "CREATE INDEX IF NOT EXISTS edges_id ON edges (id)",
    """
    CREATE TABLE IF NOT EXISTS memberships (
        community_label TEXT NOT NULL,
        community_id TEXT NOT NULL,
        entity_label TEXT NOT NULL,
        entity_id TEXT NOT NULL,
        PRIMARY KEY (community_label, community_id, entity_label, entity_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS memberships_entity "
    "ON memberships (entity_label, entity_id)",
)

# Stay under SQLite's bound-parameter limit (999 on older builds): a
# statement binds at most 2 * _BATCH + 2 values, or splits its batch.
_BATCH = 400

# The vertex properties a traversal row projects (NeptuneRetriever's
# ``value_map``).
_NODE_KEYS = ("id", "name", "description", "rank", "text_unit_ids")

# A vertex filter for a traversal: its stored valueMap -> kept.
VertexFilter = Callable[[dict[str, Any]], bool]


def _batches(items: list[Any], size: int = _BATCH) -> Iterator[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _marks(items: Collection[Any]) -> str:
    return ",".join("?" * len(items))


def vertex_value_map(vertex_id: str, props: dict[str, Any]) -> dict[str, list[Any]]:
    """The valueMap of a vertex written with ``props``, as Neptune stores it.

    The write replaces every stored property but ``id``: a scalar is a
    one-value list (single cardinality), a list its distinct values (set
    cardinality), and a ``None`` or empty list leaves the property out.
    """
    value_map: dict[str, list[Any]] = {"id": [vertex_id]}
    for key, value in props.items():
        values = value if isinstance(value, list) else [value]
        values = list(dict.fromkeys(v for v in values if v is not None))
        if values:
            value_map[key] = values
    return value_map


class SQLiteGraphStore:
    """The SQLite database of one graph: schema, transactions and reads.

    Shared by :class:`SQLiteGraphIndexer` (writes) and
    ``SQLiteGraphRetriever`` (traversal). Every error is raised typed:
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
        for batch in _batches(list(dict.fromkeys(ids))):
            rows = connection.execute(
                f"SELECT id, props FROM vertices WHERE label = ? "
                f"AND id IN ({_marks(batch)})",
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
        """Incident edges plus memberships of each vertex (``bothE().count()``)."""
        counts: dict[str, int] = {}
        # The batch is bound three times.
        for batch in _batches(list(dict.fromkeys(ids)), _BATCH // 3):
            marks = _marks(batch)
            rows = connection.execute(
                f"SELECT source_id FROM edges WHERE label = ? AND source_id IN ({marks}) "
                "UNION ALL "
                f"SELECT target_id FROM edges WHERE label = ? AND target_id IN ({marks}) "
                "UNION ALL "
                "SELECT entity_id FROM memberships WHERE entity_label = ? "
                f"AND entity_id IN ({marks})",
                [label, *batch, label, *batch, label, *batch],
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


class SQLiteGraphIndexer(GraphIndexer):
    """``GraphIndexer`` on :class:`SQLiteGraphStore` (Neptune semantics).

    Lists are set-cardinality, a vertex write replaces the stored properties
    (one its item lacks is dropped), re-adding an edge replaces it, an edge whose endpoint the suffix
    lacks is not written (its id is in ``failed_ids``), and dropping a vertex
    drops its edges and memberships. Each write is one transaction; a write
    the database refuses raises (``SQLiteGraphStore``) rather than counting
    its items as failed.
    """

    def __init__(
        self,
        config: Config,
        path: str | Path | None = None,
        *,
        store: SQLiteGraphStore | None = None,
    ) -> None:
        """Args:
        config: Framework config; ``providers.graph.sqlite_path`` is the
            database unless ``path`` or ``store`` is given.
        path: Database file, or ``":memory:"``.
        store: An open store to write to (shared with a retriever).
        """
        super().__init__(config)
        self.neptune_config = config.indexing.neptune
        self.store = store or SQLiteGraphStore(
            path or config.providers.graph.sqlite_path,
            timeout_seconds=config.search.store_timeout_seconds,
        )
        self._max_length = self.neptune_config.property_max_length

    def close(self) -> None:
        self.store.close()

    def initialize(self) -> None:
        """Open the database (creating it and its schema)."""
        with self.store.read("initialize the graph store"):
            pass

    def _entity_label(self, suffix: str | None) -> str:
        return self._get_label(self.neptune_config.entity_label_prefix, suffix)

    def _community_label(self, suffix: str | None) -> str:
        return self._get_label(self.neptune_config.community_label_prefix, suffix)

    # --- writes --------------------------------------------------------------

    def clear(self, suffixes: list[str]) -> None:
        if not suffixes:
            return
        with self.store.transaction("clear the graph") as connection:
            for suffix in suffixes:
                self._drop_label(connection, self._entity_label(suffix))
                self._drop_label(connection, self._community_label(suffix))

    def index_entities(
        self, entities: list[Entity], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        """Rebuild each suffix's entity label: clear it, then write the entities."""
        return self._write_entities(entities, clear_first=True, suffixes=suffixes)

    def upsert_entities(self, entities: list[Entity]) -> IndexingStats:
        return self._write_entities(entities, clear_first=False)

    def _write_entities(
        self,
        entities: list[Entity],
        *,
        clear_first: bool,
        suffixes: Collection[str] = (),
    ) -> IndexingStats:
        stats = IndexingStats(total_items=len(entities))
        if not entities and not suffixes:
            return stats
        grouped = self._group_items_by_suffix(entities, suffixes)
        with self.store.transaction("write entities") as connection:
            for suffix, chunk in grouped.items():
                label = self._entity_label(suffix)
                if clear_first:
                    self._drop_label(connection, label)
                self._upsert_vertices(
                    connection,
                    label,
                    [(e.id, entity_properties(e, self._max_length)) for e in chunk],
                )
                stats.add_success(len(chunk))
        return stats

    def upsert_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Replace each edge by id; one whose endpoint is missing is not written.

        As in Neptune the stored edge of that id is dropped first, so a
        re-added edge whose endpoint is gone leaves no edge behind.
        """
        stats = IndexingStats(total_items=len(relationships))
        if not relationships:
            return stats
        grouped = self._group_items_by_suffix(relationships)
        with self.store.transaction("write relationships") as connection:
            for suffix, rels in grouped.items():
                label = self._entity_label(suffix)
                self._delete_edges(connection, label, [rel.id for rel in rels])
                stored = set(
                    self.store.vertex_maps(
                        connection,
                        label,
                        (end for rel in rels for end in (rel.source_id, rel.target_id)),
                    )
                )
                for rel in rels:
                    missing = [
                        end
                        for end in (rel.source_id, rel.target_id)
                        if end not in stored
                    ]
                    if missing:
                        stats.add_error(
                            f"endpoint entity '{missing[0]}' not found", ids=[rel.id]
                        )
                        logger.warning(
                            "Relationship '%s' not written: endpoint entity '%s' "
                            "not found",
                            rel.id,
                            missing[0],
                        )
                        continue
                    props = {
                        "id": rel.id,
                        **relationship_properties(rel, self._max_length),
                    }
                    connection.execute(
                        "INSERT OR REPLACE INTO edges "
                        "(label, id, type, source_id, target_id, props) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            label,
                            rel.id,
                            relationship_label(rel),
                            rel.source_id,
                            rel.target_id,
                            json.dumps(props),
                        ),
                    )
                    stats.add_success()
        return stats

    def index_communities(
        self, communities: list[Community], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        """Rebuild each suffix's community label (vertices and memberships)."""
        return self._write_communities(communities, upsert=False, suffixes=suffixes)

    def upsert_communities(self, communities: list[Community]) -> IndexingStats:
        """Merge communities without clearing the label.

        A community with ``entity_ids`` gets exactly those members (of the
        entities stored); one without keeps its stored memberships.
        """
        return self._write_communities(communities, upsert=True)

    def _write_communities(
        self,
        communities: list[Community],
        *,
        upsert: bool,
        suffixes: Collection[str] = (),
    ) -> IndexingStats:
        stats = IndexingStats(total_items=len(communities))
        if not communities and not suffixes:
            return stats
        grouped = self._group_items_by_suffix(communities, suffixes)
        with self.store.transaction("write communities") as connection:
            for suffix, comms in grouped.items():
                community_label = self._community_label(suffix)
                entity_label = self._entity_label(suffix)
                if not upsert:
                    self._drop_label(connection, community_label)
                self._upsert_vertices(
                    connection,
                    community_label,
                    [(comm.id, self._community_properties(comm)) for comm in comms],
                )
                for comm in comms:
                    if not comm.entity_ids:
                        continue
                    connection.execute(
                        "DELETE FROM memberships WHERE community_label = ? "
                        "AND community_id = ?",
                        (community_label, comm.id),
                    )
                    members = self.store.vertex_maps(
                        connection, entity_label, comm.entity_ids
                    )
                    connection.executemany(
                        "INSERT OR IGNORE INTO memberships (community_label, "
                        "community_id, entity_label, entity_id) VALUES (?, ?, ?, ?)",
                        [
                            (community_label, comm.id, entity_label, entity_id)
                            for entity_id in members
                        ],
                    )
                stats.add_success(len(comms))
        return stats

    def _community_properties(self, comm: Community) -> dict[str, Any]:
        return item_properties(
            comm,
            {
                "name": comm.name,
                "level": comm.level,
                "parent": comm.parent,
                "size": comm.size,
                "period": comm.period,
                "children": comm.children,
            },
            self._max_length,
        )

    def delete_by_id(self, ids: list[str], suffix: str | None = None) -> IndexingStats:
        """Delete edges and vertices by id, with the vertices' edges.

        With ``suffix`` only that suffix's entity and community labels (and
        the edges between its entities) are touched; ``None`` deletes the ids
        under every label.
        """
        stats = IndexingStats(total_items=len(ids))
        if not ids:
            return stats
        with self.store.transaction("delete by id") as connection:
            if suffix is None:
                self._delete_edges(connection, None, ids)
                pairs = [
                    (label, vertex_id)
                    for batch in _batches(list(dict.fromkeys(ids)))
                    for label, vertex_id in connection.execute(
                        f"SELECT label, id FROM vertices WHERE id IN ({_marks(batch)})",
                        batch,
                    ).fetchall()
                ]
            else:
                labels = (self._entity_label(suffix), self._community_label(suffix))
                self._delete_edges(connection, labels[0], ids)
                pairs = [(label, vertex_id) for label in labels for vertex_id in ids]
            self._drop_vertices(connection, pairs)
        stats.add_success(len(ids))
        return stats

    def _upsert_vertices(
        self,
        connection: sqlite3.Connection,
        label: str,
        items: list[tuple[str, dict[str, Any]]],
    ) -> None:
        """Create or replace each vertex (the last write of an id wins)."""
        written = {
            vertex_id: vertex_value_map(vertex_id, props) for vertex_id, props in items
        }
        connection.executemany(
            "INSERT OR REPLACE INTO vertices (label, id, props) VALUES (?, ?, ?)",
            [
                (label, vertex_id, json.dumps(value_map))
                for vertex_id, value_map in written.items()
            ],
        )

    @staticmethod
    def _delete_edges(
        connection: sqlite3.Connection, label: str | None, ids: list[str]
    ) -> None:
        for batch in _batches(list(dict.fromkeys(ids))):
            if label is None:
                connection.execute(
                    f"DELETE FROM edges WHERE id IN ({_marks(batch)})", batch
                )
            else:
                connection.execute(
                    f"DELETE FROM edges WHERE label = ? AND id IN ({_marks(batch)})",
                    [label, *batch],
                )

    @staticmethod
    def _drop_vertices(
        connection: sqlite3.Connection, pairs: list[tuple[str, str]]
    ) -> None:
        """Drop ``(label, id)`` vertices with their edges and memberships."""
        for statement in (
            "DELETE FROM edges WHERE label = ? AND source_id = ?",
            "DELETE FROM edges WHERE label = ? AND target_id = ?",
            "DELETE FROM memberships WHERE entity_label = ? AND entity_id = ?",
            "DELETE FROM memberships WHERE community_label = ? AND community_id = ?",
            "DELETE FROM vertices WHERE label = ? AND id = ?",
        ):
            connection.executemany(statement, pairs)

    @staticmethod
    def _drop_label(connection: sqlite3.Connection, label: str) -> None:
        """Drop every vertex of ``label`` with its edges and memberships."""
        for statement in (
            "DELETE FROM edges WHERE label = ?",
            "DELETE FROM memberships WHERE entity_label = ?",
            "DELETE FROM memberships WHERE community_label = ?",
            "DELETE FROM vertices WHERE label = ?",
        ):
            connection.execute(statement, (label,))

    # --- reads ---------------------------------------------------------------

    def read_entities(self, ids: list[str], suffix: str | None = None) -> list[Entity]:
        """Stored entities of ``suffix`` among ``ids``, in ``ids`` order."""
        if not ids:
            return []
        with self.store.read("read entities") as connection:
            maps = self.store.vertex_maps(connection, self._entity_label(suffix), ids)
        entities = (
            entity_from_vertex(maps[i]) for i in dict.fromkeys(ids) if i in maps
        )
        return [entity for entity in entities if entity is not None]

    def read_relationships(
        self, ids: list[str], suffix: str | None = None
    ) -> list[Relationship]:
        """Stored relationships of ``suffix`` among ``ids``, in ``ids`` order."""
        if not ids:
            return []
        label = self._entity_label(suffix)
        found: dict[str, Relationship] = {}
        with self.store.read("read relationships") as connection:
            for batch in _batches(list(dict.fromkeys(ids))):
                rows = connection.execute(
                    "SELECT id, type, source_id, target_id, props FROM edges "
                    f"WHERE label = ? AND id IN ({_marks(batch)})",
                    [label, *batch],
                ).fetchall()
                for rel_id, rel_type, source_id, target_id, props in rows:
                    rel = relationship_from_edge(
                        json.loads(props), rel_type, source_id, target_id
                    )
                    if rel is not None:
                        found[rel_id] = rel
        return [found[i] for i in dict.fromkeys(ids) if i in found]

    def read_entity_names(self, suffix: str | None = None) -> list[tuple[str, str]]:
        """``(id, name)`` of every entity of ``suffix``, by id.

        Best-effort like Neptune's: ``[]`` (no fuzzy merge) when the read fails.
        """
        try:
            with self.store.read("read entity names") as connection:
                rows = connection.execute(
                    "SELECT id, props FROM vertices WHERE label = ? ORDER BY id",
                    (self._entity_label(suffix),),
                ).fetchall()
        except GraphRAGException as e:
            logger.warning(
                "read_entity_names failed (%s); fuzzy cross-run merge disabled", e
            )
            return []
        pairs = []
        for entity_id, props in rows:
            names = json.loads(props).get("name")
            if names:
                pairs.append((entity_id, str(names[0])))
        return pairs

    def read_neighbor_ids(
        self, entity_ids: list[str], suffix: str | None = None
    ) -> dict[str, list[str]]:
        """``{id: sorted ids of its neighbours}`` for the stored entities."""
        if not entity_ids:
            return {}
        label = self._entity_label(suffix)
        with self.store.read("read neighbour ids") as connection:
            stored = self.store.vertex_maps(connection, label, entity_ids)
            neighbors: dict[str, set[str]] = {i: set() for i in stored}
            for batch in _batches(list(stored)):
                marks = _marks(batch)
                # One indexed lookup per endpoint (an OR reads every edge of
                # the label).
                rows = connection.execute(
                    "SELECT source_id, target_id FROM edges WHERE label = ? "
                    f"AND source_id IN ({marks}) "
                    "UNION ALL "
                    "SELECT source_id, target_id FROM edges WHERE label = ? "
                    f"AND target_id IN ({marks})",
                    [label, *batch, label, *batch],
                ).fetchall()
                for source_id, target_id in rows:
                    if source_id in neighbors:
                        neighbors[source_id].add(target_id)
                    if target_id in neighbors:
                        neighbors[target_id].add(source_id)
        return {
            i: sorted(neighbors[i]) for i in dict.fromkeys(entity_ids) if i in neighbors
        }

    def find_incident_relationship_ids(
        self, entity_ids: list[str], suffix: str | None = None
    ) -> list[str]:
        """Sorted ids of the edges incident to ``entity_ids`` (any suffix
        when ``suffix`` is None)."""
        if not entity_ids:
            return []
        label = self._entity_label(suffix) if suffix is not None else None
        found: set[str] = set()
        with self.store.read("find incident relationships") as connection:
            for batch in _batches(list(dict.fromkeys(entity_ids))):
                marks = _marks(batch)
                if label is None:
                    # No index leads with an endpoint: one scan of the table.
                    rows = connection.execute(
                        f"SELECT id FROM edges WHERE source_id IN ({marks}) "
                        f"OR target_id IN ({marks})",
                        [*batch, *batch],
                    ).fetchall()
                else:
                    rows = connection.execute(
                        "SELECT id FROM edges WHERE label = ? "
                        f"AND source_id IN ({marks}) "
                        "UNION ALL "
                        "SELECT id FROM edges WHERE label = ? "
                        f"AND target_id IN ({marks})",
                        [label, *batch, label, *batch],
                    ).fetchall()
                found.update(row[0] for row in rows)
        return sorted(found)
