# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GraphIndexer`` on the embedded SQLite graph store (``providers.graph.kind:
sqlite``).

Writes what ``NeptuneIndexer`` writes, with the same semantics, into the
tables of ``adapters.local.sqlite_graph.SQLiteGraphStore`` (``FakeGraphStore``
is the reference the tests compare against). Values are encoded and decoded
by ``neptune_codec``, so ``indexing.neptune.property_max_length`` truncates
strings as in Neptune.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Collection
from pathlib import Path
from typing import Any

from unified_kg_rag.adapters.local.sqlite_graph import (
    SQLiteGraphStore,
    batches,
    placeholders,
)
from unified_kg_rag.adapters.storage.neptune_codec import (
    BUILD_TAG_PROPERTY,
    NAME_KEY_PROPERTY,
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

logger = get_logger(__name__)


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


class SQLiteGraphIndexer(GraphIndexer):
    """``GraphIndexer`` on :class:`SQLiteGraphStore` (Neptune semantics).

    Lists are set-cardinality, a vertex write replaces the stored properties
    (one its item lacks is dropped), re-adding an edge replaces it, an edge whose endpoint the suffix
    lacks is not written (its id is in ``failed_ids``, its stored copy stays),
    and dropping a vertex drops its edges. Each write is one transaction; a write
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
        # Set between begin_rebuild and end_rebuild (see GraphIndexer).
        self._rebuild_tag: str | None = None
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
                if clear_first and self._rebuild_tag is None:
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

        As in Neptune, such an edge fails and its stored copy stays: the
        edge an unchanged document states is not lost to a write that
        re-points it at a vertex that is not there.
        """
        stats = IndexingStats(total_items=len(relationships))
        if not relationships:
            return stats
        grouped = self._group_items_by_suffix(relationships)
        with self.store.transaction("write relationships") as connection:
            for suffix, rels in grouped.items():
                label = self._entity_label(suffix)
                stored = set(
                    self.store.vertex_maps(
                        connection,
                        label,
                        (end for rel in rels for end in (rel.source_id, rel.target_id)),
                    )
                )
                self._delete_edges(
                    connection,
                    label,
                    [
                        rel.id
                        for rel in rels
                        if rel.source_id in stored and rel.target_id in stored
                    ],
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
                    props = self._tagged(
                        {"id": rel.id, **relationship_properties(rel, self._max_length)}
                    )
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
        """Rebuild each suffix's community label."""
        return self._write_communities(communities, upsert=False, suffixes=suffixes)

    def upsert_communities(self, communities: list[Community]) -> IndexingStats:
        """Merge communities without clearing the label."""
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
                if not upsert and self._rebuild_tag is None:
                    self._drop_label(connection, community_label)
                self._upsert_vertices(
                    connection,
                    community_label,
                    [(comm.id, self._community_properties(comm)) for comm in comms],
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
                # Found by them when its documents are removed.
                "text_unit_ids": comm.text_unit_ids,
            },
            self._max_length,
        )

    def delete_by_id(self, ids: list[str], suffix: str | None = None) -> IndexingStats:
        """Delete edges and vertices by id, with the vertices' edges.

        Only the suffix's entity and community labels (and the edges between
        its entities) are touched; ``None`` is the default suffix, as for
        every other method, never every label.
        """
        stats = IndexingStats(total_items=len(ids))
        if not ids:
            return stats
        with self.store.transaction("delete by id") as connection:
            labels = (self._entity_label(suffix), self._community_label(suffix))
            self._delete_edges(connection, labels[0], ids)
            self._drop_vertices(
                connection,
                [(label, vertex_id) for label in labels for vertex_id in ids],
            )
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
            vertex_id: vertex_value_map(vertex_id, self._tagged(props))
            for vertex_id, props in items
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
        for batch in batches(list(dict.fromkeys(ids))):
            if label is None:
                connection.execute(
                    f"DELETE FROM edges WHERE id IN ({placeholders(batch)})", batch
                )
            else:
                connection.execute(
                    f"DELETE FROM edges WHERE label = ? AND id IN ({placeholders(batch)})",
                    [label, *batch],
                )

    @staticmethod
    def _drop_vertices(
        connection: sqlite3.Connection, pairs: list[tuple[str, str]]
    ) -> None:
        """Drop ``(label, id)`` vertices with their edges."""
        for statement in (
            "DELETE FROM edges WHERE label = ? AND source_id = ?",
            "DELETE FROM edges WHERE label = ? AND target_id = ?",
            "DELETE FROM vertices WHERE label = ? AND id = ?",
        ):
            connection.executemany(statement, pairs)

    def _tagged(self, props: dict[str, Any]) -> dict[str, Any]:
        """``props`` with the rebuild's tag, if one is running."""
        if self._rebuild_tag is None:
            return props
        return {**props, BUILD_TAG_PROPERTY: self._rebuild_tag}

    # --- rebuilding a suffix in place ------------------------------------------

    def begin_rebuild(self, suffixes: Collection[str]) -> bool:
        """See ``GraphIndexer.begin_rebuild``: the full-build writes keep the
        labels and tag what they write."""
        self._rebuild_tag = uuid.uuid4().hex
        return True

    def end_rebuild(self, suffixes: Collection[str], sweep: bool = True) -> None:
        """Drop what the build did not write (one transaction), and end it."""
        tag, self._rebuild_tag = self._rebuild_tag, None
        if not sweep or tag is None:
            return
        with self.store.transaction("drop the stale graph") as connection:
            for suffix in sorted(suffixes):
                entity_label = self._entity_label(suffix)
                connection.execute(
                    "DELETE FROM edges WHERE label = ? AND "
                    "json_extract(props, '$.build_tag') IS NOT ?",
                    (entity_label, tag),
                )
                for label in (entity_label, self._community_label(suffix)):
                    stale = connection.execute(
                        "SELECT id FROM vertices WHERE label = ? AND NOT EXISTS "
                        "(SELECT 1 FROM json_each(vertices.props, '$.build_tag') "
                        "WHERE value = ?)",
                        (label, tag),
                    ).fetchall()
                    self._drop_vertices(connection, [(label, row[0]) for row in stale])

    @staticmethod
    def _drop_label(connection: sqlite3.Connection, label: str) -> None:
        """Drop every vertex of ``label`` with its edges."""
        for statement in (
            "DELETE FROM edges WHERE label = ?",
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
            for batch in batches(list(dict.fromkeys(ids))):
                rows = connection.execute(
                    "SELECT id, type, source_id, target_id, props FROM edges "
                    f"WHERE label = ? AND id IN ({placeholders(batch)})",
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
            for batch in batches(list(stored)):
                marks = placeholders(batch)
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

    def find_ids_citing(
        self,
        text_unit_ids: list[str],
        suffix: str | None = None,
        *,
        communities: bool = False,
    ) -> dict[str, list[str]] | None:
        """Entity (or community) vertices of ``suffix`` citing any of
        ``text_unit_ids``, with their text units (see the port). One pass over
        the label's vertices per batch (the embedded store's scale)."""
        if not text_unit_ids:
            return {}
        label = (
            self._community_label(suffix) if communities else self._entity_label(suffix)
        )
        found: dict[str, list[str]] = {}
        with self.store.read("find vertices citing text units") as connection:
            for batch in batches(list(dict.fromkeys(text_unit_ids))):
                rows = connection.execute(
                    "SELECT id, props FROM vertices WHERE label = ? AND EXISTS "
                    "(SELECT 1 FROM json_each(vertices.props, '$.text_unit_ids') "
                    f"WHERE value IN ({placeholders(batch)}))",
                    [label, *batch],
                ).fetchall()
                for vertex_id, props in rows:
                    found[vertex_id] = list(
                        json.loads(props).get("text_unit_ids") or []
                    )
        return found

    def find_entity_ids_by_name_key(
        self, name_keys: list[str], suffix: str | None = None
    ) -> list[str] | None:
        """Entities of ``suffix`` whose stored ``name_key`` is one of
        ``name_keys`` (see the port)."""
        if not name_keys:
            return []
        label = self._entity_label(suffix)
        found: set[str] = set()
        with self.store.read("find entities by name key") as connection:
            for batch in batches(list(dict.fromkeys(name_keys))):
                rows = connection.execute(
                    "SELECT id FROM vertices WHERE label = ? AND EXISTS "
                    f"(SELECT 1 FROM json_each(vertices.props, '$.{NAME_KEY_PROPERTY}') "
                    f"WHERE value IN ({placeholders(batch)}))",
                    [label, *batch],
                ).fetchall()
                found.update(row[0] for row in rows)
        return sorted(found)

    def find_incident_relationship_ids(
        self, entity_ids: list[str], suffix: str | None = None
    ) -> list[str]:
        """Sorted ids of the suffix's edges incident to ``entity_ids``
        (``None``: the default suffix)."""
        if not entity_ids:
            return []
        label = self._entity_label(suffix)
        found: set[str] = set()
        with self.store.read("find incident relationships") as connection:
            for batch in batches(list(dict.fromkeys(entity_ids))):
                marks = placeholders(batch)
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
