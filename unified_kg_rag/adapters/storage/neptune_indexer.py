# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import random
import time
from collections.abc import Callable, Iterator
from concurrent.futures import as_completed
from typing import Any, cast

from gremlin_python.process.graph_traversal import (
    GraphTraversal,
    GraphTraversalSource,
    __,
)
from gremlin_python.process.traversal import Cardinality, P

from unified_kg_rag.adapters.aws import NeptuneClient
from unified_kg_rag.adapters.aws.neptune import is_permanent_neptune_error
from unified_kg_rag.adapters.storage.neptune_codec import (
    entity_from_vertex,
    entity_properties,
    item_properties,
    relationship_from_edge,
    relationship_label,
    relationship_properties,
)
from unified_kg_rag.domain.models import (
    Community,
    Config,
    Entity,
    Relationship,
)
from unified_kg_rag.ports.indexer import GraphIndexer, IndexingStats
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor

logger = get_logger(__name__)

# Retry-backoff seams: tests patch these module attributes rather than the
# process-global ``time.sleep``/``random.uniform``, which other threads share.
_sleep = time.sleep
# Retry-backoff jitter only; not a security/crypto context.
_jitter = random.uniform  # nosec B311


class NeptuneIndexer(GraphIndexer):
    # Emit a progress line every N edges during the per-edge relationship write
    # so a large run (tens of thousands of edges) is distinguishable from a hang.
    _PROGRESS_LOG_EVERY: int = 2000

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.neptune_config = self.config.indexing.neptune
        self.neptune_client = NeptuneClient(config=self.config)

    def close(self) -> None:
        """Close the underlying Neptune websocket + thread pool (best-effort)."""
        self.neptune_client.close()

    def clear(self, suffixes: list[str]) -> bool:
        if not suffixes:
            return True

        entity_prefix = self.neptune_config.entity_label_prefix.capitalize()
        community_prefix = self.neptune_config.community_label_prefix.capitalize()

        labels_to_delete = {self._get_name(entity_prefix, s) for s in suffixes} | {
            self._get_name(community_prefix, s) for s in suffixes
        }

        try:
            if labels_to_delete:
                logger.info("Clearing Neptune data for labels: %s", labels_to_delete)
                for label in labels_to_delete:
                    self.neptune_client.delete_vertices_in_batches(label)
            return True
        except Exception as e:
            logger.error(
                "Failed to clear Neptune data for suffixes '%s': %s", suffixes, e
            )
            return False

    def get_entity_count(self, suffixes: list[str]) -> int:
        if not suffixes:
            return 0

        entity_prefix = self.neptune_config.entity_label_prefix.capitalize()
        entity_labels = [self._get_name(entity_prefix, s) for s in suffixes]

        try:
            g = self.neptune_client.g
            result = g.V().hasLabel(*entity_labels).count().next()
            return int(result) if isinstance(result, (int | float)) else 0
        except Exception as e:
            logger.error("Failed to get entity count for '%s': %s", entity_labels, e)
            return 0

    def get_stats(self) -> dict[str, Any]:
        stats = self.neptune_client.get_graph_stats()
        if not isinstance(stats, dict):
            return {}
        return stats

    def read_entities(self, ids: list[str], suffix: str | None = None) -> list[Entity]:
        """Read existing entities by id for cross-run merge (best-effort).

        Scoped to this suffix's entity label: entity ids are suffix-independent,
        so an unscoped read could return another tenant's vertex. Inverts the
        vertex encoding (``neptune_codec``), attributes included, so a merged
        entity is written back under its own suffix. Returns ``[]`` on any
        error so cross-run merge degrades to overwrite.
        """
        if not ids:
            return []
        entity_label = self._get_name(
            self.neptune_config.entity_label_prefix.capitalize(), suffix
        )
        try:
            g = self.neptune_client.g
            entities: list[Entity] = []
            for id_batch in self._batch_iterator(ids):
                rows = (
                    g.V()
                    .hasLabel(entity_label)
                    .has("id", P.within(id_batch))
                    .valueMap()
                    .toList()
                )
                for row in rows:
                    entity = entity_from_vertex(row)
                    if entity is not None:
                        entities.append(entity)
            return entities
        except Exception as e:  # noqa: BLE001 - degrade to overwrite
            logger.warning("read_entities failed (%s); cross-run merge disabled", e)
            return []

    def read_relationships(
        self, ids: list[str], suffix: str | None = None
    ) -> list[Relationship]:
        """Read existing relationships by id for cross-run merge (best-effort).

        Scoped to edges leaving this suffix's entity label (relationship ids are
        suffix-independent). Inverts the edge encoding (``neptune_codec``): the
        type is the edge label and list properties are JSON strings. Returns
        ``[]`` on any error so cross-run merge degrades to overwrite.
        """
        if not ids:
            return []
        entity_label = self._get_name(
            self.neptune_config.entity_label_prefix.capitalize(), suffix
        )
        try:
            g = self.neptune_client.g
            rels: list[Relationship] = []
            for id_batch in self._batch_iterator(ids):
                # source_id/target_id are edge TOPOLOGY (endpoint vertex ids)
                # and the type is the edge label, so neither is in valueMap().
                rows = (
                    g.E()
                    .has("id", P.within(id_batch))
                    .where(__.outV().hasLabel(entity_label))
                    .project("props", "label", "source_id", "target_id")
                    .by(__.valueMap())
                    .by(__.label())
                    .by(__.outV().values("id"))
                    .by(__.inV().values("id"))
                    .toList()
                )
                for row in rows:
                    rel = relationship_from_edge(
                        row.get("props") or {},
                        str(row.get("label")),
                        row.get("source_id"),
                        row.get("target_id"),
                    )
                    if rel is not None:
                        rels.append(rel)
            return rels
        except Exception as e:  # noqa: BLE001 - degrade to overwrite
            logger.warning(
                "read_relationships failed (%s); cross-run merge disabled", e
            )
            return []

    def read_entity_names(self, suffix: str | None = None) -> list[tuple[str, str]]:
        """Project ``(id, name)`` for all existing entities in the suffix's label.

        Best-effort (``[]`` on error): powers cross-run *fuzzy* merge, which needs
        old entities whose ids differ from a delta entity's. Requires real
        Neptune; validated under the ``aws`` test marker.
        """
        try:
            g = self.neptune_client.g
            entity_label = self._get_name(
                self.neptune_config.entity_label_prefix.capitalize(), suffix
            )
            rows = (
                g.V()
                .hasLabel(entity_label)
                .project("id", "name")
                .by("id")
                .by("name")
                .toList()
            )
            pairs: list[tuple[str, str]] = []
            for row in rows:
                if isinstance(row, dict) and "id" in row and "name" in row:
                    pairs.append((str(row["id"]), str(row["name"])))
            return pairs
        except Exception as e:  # noqa: BLE001 - degrade to exact-name merge
            logger.warning(
                "read_entity_names failed (%s); fuzzy cross-run merge disabled", e
            )
            return []

    def initialize(self) -> bool:
        return True

    def index_entities(self, entities: list[Entity]) -> IndexingStats:
        def get_traversal_builder(label: str) -> Callable:
            def builder(g: GraphTraversalSource, batch: list[Entity]) -> GraphTraversal:
                t = g
                for entity in batch:
                    props = entity_properties(
                        entity, self.neptune_config.property_max_length
                    )
                    v_traversal = t.add_v(label).property("id", entity.id)
                    self._add_properties_to_traversal(v_traversal, props)
                    t = v_traversal
                return cast(GraphTraversal, t)

            return builder

        return self._index_generic(
            entities,
            "Entity",
            self.neptune_config.entity_label_prefix.capitalize(),
            self.neptune_config.entity_label_prefix.capitalize(),
            get_traversal_builder,
        )

    def index_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        # Full-index and incremental upsert now share one per-edge write path:
        # both drop-by-id first (idempotency) then re-add, so there is no
        # behavioural difference to justify two code paths.
        return self._write_relationships(relationships)

    def index_communities(self, communities: list[Community]) -> IndexingStats:
        return self._index_communities(communities, upsert=False)

    def upsert_communities(self, communities: list[Community]) -> IndexingStats:
        """Idempotently merge communities into the live graph (delta semantics).

        Unlike :meth:`index_communities`, this does NOT clear the community label
        first — a label-wide clear on an incremental run would wipe communities
        belonging to documents outside the delta. Community vertices are upserted
        by id (fold/coalesce) and their MemberOf edges are dropped-by-target then
        re-added so re-runs do not duplicate membership edges.
        """
        return self._index_communities(communities, upsert=True)

    def _index_communities(
        self, communities: list[Community], *, upsert: bool
    ) -> IndexingStats:
        if not communities:
            return IndexingStats()

        total_stats = IndexingStats()
        grouped_items = self._group_items_by_suffix(communities)

        for suffix, comms in grouped_items.items():
            stats = IndexingStats()
            start_time = time.time()
            community_label = self._get_name(
                self.neptune_config.community_label_prefix.capitalize(), suffix
            )
            entity_label = self._get_name(
                self.neptune_config.entity_label_prefix.capitalize(), suffix
            )

            if not upsert:
                self._clear_existing_data_by_label(community_label)

            def community_vertex_builder(
                g: GraphTraversalSource,
                batch: list[Community],
                community_label: str = community_label,
                upsert: bool = upsert,
            ) -> GraphTraversal:
                t = g
                for comm in batch:
                    props = self._build_vertex_properties(
                        comm,
                        {
                            "name": comm.name,
                            "level": comm.level,
                            "parent": comm.parent,
                            "size": comm.size,
                            "period": comm.period,
                            "children": comm.children,
                        },
                    )
                    if upsert:
                        # Create-or-match by id so an incremental re-run updates
                        # in place instead of adding a duplicate community vertex.
                        v_traversal = (
                            t.V()
                            .has(community_label, "id", comm.id)
                            .fold()
                            .coalesce(
                                __.unfold(),
                                __.add_v(community_label).property("id", comm.id),
                            )
                        )
                        self._set_properties_on_traversal(v_traversal, props)
                    else:
                        v_traversal = t.add_v(community_label).property("id", comm.id)
                        self._add_properties_to_traversal(v_traversal, props)
                    t = v_traversal
                return cast(GraphTraversal, t)

            logger.info(
                "Indexing %s communities for '%s'...", len(comms), community_label
            )
            vertex_stats = self._execute_batch_traversal(
                comms, community_vertex_builder, "Community vertex indexing"
            )
            stats.merge(vertex_stats)

            logger.info("Indexing 'MemberOf' edges for %s communities...", len(comms))
            for comm in comms:
                if not comm.entity_ids:
                    continue
                if upsert:
                    # Drop this community's existing membership edges so re-adding
                    # does not create duplicate MemberOf edges on an incremental run.
                    try:
                        self.neptune_client.g.V().hasLabel(community_label).has(
                            "id", comm.id
                        ).inE("MemberOf").drop().iterate()
                    except Exception as e:
                        logger.warning(
                            "Failed clearing MemberOf edges for community '%s': %s",
                            comm.id,
                            e,
                        )
                for entity_id_batch in self._batch_iterator(comm.entity_ids):
                    try:
                        edge_traversal = (
                            self.neptune_client.g.V()
                            .hasLabel(entity_label)
                            .has("id", P.within(entity_id_batch))
                            .addE("MemberOf")
                            .to(__.V().hasLabel(community_label).has("id", comm.id))
                        )
                        self._execute_with_retries(
                            edge_traversal, "Community edge indexing"
                        )
                    except Exception as e:
                        stats.add_error(str(e))
                        logger.warning(
                            "Community edge indexing failed for community '%s': %s",
                            comm.id,
                            e,
                        )

            stats.processing_time = time.time() - start_time
            total_stats.merge(stats)

        self._log_indexing_summary("communities", total_stats)
        return total_stats

    def upsert_entities(self, entities: list[Entity]) -> IndexingStats:
        """Idempotently merge entities into the live graph (delta semantics).

        Uses ``V().has('id', x).fold().coalesce(unfold(), addV(label))`` so an
        existing vertex is updated in place and a missing one is created — no
        label-wide clear, no duplicate vertices on re-run.
        """

        def get_traversal_builder(label: str) -> Callable:
            def builder(g: GraphTraversalSource, batch: list[Entity]) -> GraphTraversal:
                t = g
                for entity in batch:
                    props = entity_properties(
                        entity, self.neptune_config.property_max_length
                    )
                    v_traversal = (
                        t.V()
                        .has(label, "id", entity.id)
                        .fold()
                        .coalesce(
                            __.unfold(),
                            __.add_v(label).property("id", entity.id),
                        )
                    )
                    self._set_properties_on_traversal(v_traversal, props)
                    t = v_traversal
                return cast(GraphTraversal, t)

            return builder

        return self._index_generic(
            entities,
            "Entity",
            self.neptune_config.entity_label_prefix.capitalize(),
            "",  # no clear: upsert is non-destructive
            get_traversal_builder,
        )

    def upsert_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Idempotently merge relationship edges into the live graph.

        Drops any existing edge with the same id before re-adding it, so the
        operation is repeatable without creating duplicate parallel edges.
        Shares the per-edge write path with :meth:`index_relationships`.
        """
        return self._write_relationships(relationships)

    def _build_add_edge_traversal(
        self, g: GraphTraversalSource, rel: Relationship, entity_label: str
    ) -> GraphTraversal:
        """Build a single ``addE`` traversal for one relationship.

        Written per-edge on purpose. Fanning many edges out of one root via
        ``sideEffect`` (the previous shape) makes Neptune evaluate a single giant
        traversal whose per-edge cost is ~2 orders of magnitude higher than a
        small standalone traversal (measured ~1.7 s/edge vs ~7 ms/edge on a real
        cluster), so a real corpus's tens of thousands of edges effectively never
        finished. Each edge as its own traversal keeps the cost linear.

        Edge properties take no cardinality (Neptune rejects it) and hold one
        value each, so ``relationship_properties`` serializes lists to JSON.
        """
        add_edge = (
            g.V()
            .hasLabel(entity_label)
            .has("id", rel.source_id)
            .addE(relationship_label(rel))
            .to(__.V().hasLabel(entity_label).has("id", rel.target_id))
            .property("id", rel.id)
        )
        props = relationship_properties(rel, self.neptune_config.property_max_length)
        for key, value in props.items():
            add_edge.property(key, value)
        # Return the new edge's id: with no source vertex addE gets no traverser
        # and raises nothing (a missing target does raise), so an empty result
        # is the only sign of a dropped edge.
        return cast(GraphTraversal, add_edge.id_())

    def _write_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Write relationship edges one small traversal per edge (idempotent).

        For each suffix group: drop existing edges by id in batches (so re-runs
        don't duplicate parallel edges), then add each edge as its own traversal.
        Emits a progress log every ``_PROGRESS_LOG_EVERY`` edges so a large run is
        distinguishable from a hang. A single failed edge is recorded and skipped
        (does not abort the batch).
        """
        if not relationships:
            return IndexingStats()

        total_stats = IndexingStats()
        grouped_items = self._group_items_by_suffix(relationships)

        for suffix, rels in grouped_items.items():
            stats = IndexingStats(total_items=len(rels))
            start_time = time.time()
            entity_label = self._get_name(
                self.neptune_config.entity_label_prefix.capitalize(), suffix
            )

            logger.info(
                "Indexing %s relationships for '%s'...", len(rels), entity_label
            )

            # Drop existing edges by id first (batched) for idempotency. Edges
            # are not removed by the entity label-clear, so without this a re-run
            # would create duplicate parallel edges with the same id. Relationship
            # ids are suffix-independent (the same "Vendor -> Buyer" edge has the
            # same id in every tenant), so the drop is scoped to edges whose
            # source vertex carries THIS suffix's entity label (the scope
            # delete_by_id uses); unscoped, indexing one tenant would drop
            # another tenant's identical edges.
            for id_batch in self._batch_iterator([rel.id for rel in rels]):
                try:
                    self.neptune_client.g.E().has("id", P.within(id_batch)).where(
                        __.outV().hasLabel(entity_label)
                    ).drop().iterate()
                except Exception as e:
                    logger.warning("Failed dropping existing edge batch: %s", e)

            for index, rel in enumerate(rels, start=1):
                try:
                    traversal = self._build_add_edge_traversal(
                        self.neptune_client.g, rel, entity_label
                    )
                    written = self._execute_with_retries(
                        traversal, "Relationship indexing", results=True
                    )
                    if written:
                        stats.add_success(1)
                    else:
                        stats.add_error("source entity vertex not found")
                        logger.warning(
                            "Relationship '%s' not written: source entity '%s' "
                            "not found",
                            rel.id,
                            rel.source_id,
                        )
                except Exception as e:
                    stats.add_error(str(e))
                    logger.warning("Failed indexing relationship '%s': %s", rel.id, e)
                if index % self._PROGRESS_LOG_EVERY == 0:
                    logger.info(
                        "  ...%s/%s relationships written for '%s'",
                        index,
                        len(rels),
                        entity_label,
                    )

            stats.processing_time = time.time() - start_time
            total_stats.merge(stats)

        self._log_indexing_summary("relationships", total_stats)
        return total_stats

    def find_incident_relationship_ids(
        self, entity_ids: list[str], suffix: str | None = None
    ) -> list[str]:
        """Return ids of edges incident to any of the given entity vertices.

        Deleting an entity exclusive to a removed document leaves any
        relationship pointing AT it dangling. Neptune drops incident edges when
        the vertex is dropped, but the relationship's document in the OpenSearch
        relationship index survives (it is keyed by its own id, attributed to a
        *different*, still-present document's lineage). We query the incident
        edge ids here so the caller can also remove those orphaned relationship
        documents from OpenSearch. Best-effort: returns [] on any failure (the
        caller still performs the id-set based deletion).
        """
        if not entity_ids:
            return []
        try:
            g = self.neptune_client.g
            base = (
                g.V().hasLabel(
                    self._get_name(
                        self.neptune_config.entity_label_prefix.capitalize(), suffix
                    )
                )
                if suffix is not None
                else g.V()
            )
            rel_ids = (
                base.has("id", P.within(entity_ids))
                .bothE()
                .values("id")
                .dedup()
                .toList()
            )
            return [str(rid) for rid in rel_ids if rid is not None]
        except Exception as e:  # noqa: BLE001 - best-effort orphan cleanup
            logger.warning("find_incident_relationship_ids failed: %s", e)
            return []

    def delete_by_id(self, ids: list[str], suffix: str | None = None) -> IndexingStats:
        """Delete vertices and edges by their ``id`` property (delta removals).

        When ``suffix`` is given, the drop is scoped to that suffix's entity and
        community labels. Entity ids are content-hash derived, so the SAME id can
        exist under another suffix (another tenant/version); an unscoped
        ``V().has('id', ...)`` would delete that other tenant's vertex too.
        Scoping by label prevents cross-suffix data loss. ``suffix=None`` keeps
        the legacy unscoped behaviour (single-tenant).
        """
        stats = IndexingStats(total_items=len(ids))
        if not ids:
            return stats

        entity_label = community_label = None
        if suffix is not None:
            entity_label = self._get_name(
                self.neptune_config.entity_label_prefix.capitalize(), suffix
            )
            community_label = self._get_name(
                self.neptune_config.community_label_prefix.capitalize(), suffix
            )

        for id_batch in self._batch_iterator(ids):
            try:
                g = self.neptune_client.g
                if entity_label and community_label:
                    # Scope the drop to this suffix. Edges are labeled by their
                    # relationship type (e.g. RELATED_TO / MemberOf), NOT by the
                    # entity label, so filtering edges with hasLabel(entity_label)
                    # would match nothing and leak stale edges. Instead scope
                    # edges by id AND by an endpoint vertex carrying this suffix's
                    # entity label (where(outV().hasLabel(...))), which both
                    # restricts to this tenant and matches real edges.
                    g.E().has("id", P.within(id_batch)).where(
                        __.outV().hasLabel(entity_label)
                    ).drop().iterate()
                    g.V().hasLabel(entity_label, community_label).has(
                        "id", P.within(id_batch)
                    ).drop().iterate()
                else:
                    g.E().has("id", P.within(id_batch)).drop().iterate()
                    g.V().has("id", P.within(id_batch)).drop().iterate()
                stats.add_success(len(id_batch))
            except Exception as e:
                stats.add_error(str(e))
                logger.warning("Failed to delete ids batch: %s", e)

        return stats

    def _add_properties_to_traversal(
        self, traversal: GraphTraversal, props: dict[str, Any]
    ) -> None:
        for key, value in props.items():
            if isinstance(value, list):
                for item in value:
                    if item is not None:
                        traversal.property(key, item)
            else:
                traversal.property(key, value)

    def _set_properties_on_traversal(
        self, traversal: GraphTraversal, props: dict[str, Any]
    ) -> None:
        """Set properties idempotently for upserts, matching full-index encoding.

        Scalars use ``Cardinality.single`` so re-running an upsert overwrites
        rather than accumulates. List values are written as multi-valued
        ``Cardinality.set`` properties — the SAME encoding as the full-index
        write path (:meth:`_add_properties_to_traversal`) and what the read path
        expects — so an incremental run produces vertices indistinguishable from
        a full run. The existing set is cleared first so re-upserts do not grow
        stale members.
        """
        for key, value in props.items():
            if value is None:
                continue
            if isinstance(value, list):
                # Replace the whole multi-valued property: drop then re-add.
                traversal.sideEffect(__.properties(key).drop())
                for item in value:
                    if item is not None:
                        traversal.property(Cardinality.set_, key, item)
            else:
                traversal.property(Cardinality.single, key, value)

    def _build_vertex_properties(
        self, item: Any, base_props: dict[str, Any]
    ) -> dict[str, Any]:
        return item_properties(
            item, base_props, self.neptune_config.property_max_length
        )

    def _index_generic(
        self,
        items: list[Any],
        item_type_name: str,
        label_prefix: str,
        clear_label_prefix: str,
        traversal_builder_func: Callable,
        **kwargs: Any,
    ) -> IndexingStats:
        if not items:
            return IndexingStats()

        grouped_items = self._group_items_by_suffix(items)
        total_stats = IndexingStats()

        for suffix, chunk in grouped_items.items():
            label = self._get_name(label_prefix, suffix)
            if clear_label_prefix:
                clear_label = self._get_name(clear_label_prefix, suffix)
                self._clear_existing_data_by_label(clear_label)

            start_time = time.time()

            logger.info(
                "Indexing %s %ss for '%s'...", len(chunk), item_type_name.lower(), label
            )
            final_kwargs = kwargs.copy()
            if "entity_label" in final_kwargs:
                final_kwargs["entity_label"] = self._get_name(
                    final_kwargs["entity_label"], suffix
                )

            traversal_builder = traversal_builder_func(label=label, **final_kwargs)
            stats = self._execute_batch_traversal(
                chunk, traversal_builder, f"{item_type_name} indexing"
            )

            stats.processing_time = time.time() - start_time
            total_stats.merge(stats)

        self._log_indexing_summary(f"{item_type_name.lower()}s", total_stats)
        return total_stats

    def _clear_existing_data_by_label(self, label: str) -> None:
        try:
            count_result = (
                self.neptune_client.g.V().hasLabel(label).limit(1).count().next()
            )
            count = count_result[0] if isinstance(count_result, list) else count_result
            if count > 0:
                self.neptune_client.delete_vertices_in_batches(label)
        except Exception as e:
            logger.error("Failed to clear data for label '%s': %s", label, e)
            raise

    def _execute_batch_traversal(
        self,
        items: list[Any],
        traversal_builder: Callable[[GraphTraversalSource, list[Any]], GraphTraversal],
        operation_name: str,
    ) -> IndexingStats:
        stats = IndexingStats(total_items=len(items))
        if not items:
            return stats

        batches = list(self._batch_iterator(items))
        concurrency = min(self.neptune_config.index_concurrency, len(batches))

        if concurrency <= 1:
            # Sequential path (default). Each batch mutates the shared `stats`
            # directly; no cross-thread access so this is safe.
            for batch in batches:
                self._execute_single_batch(
                    batch, traversal_builder, operation_name, stats
                )
            return stats

        # Concurrent path: each batch accumulates into its OWN IndexingStats so
        # there is no shared-mutable state across worker threads; results are
        # merged on the main thread. Traversals are built off the shared
        # GraphTraversalSource (each step spawns independent bytecode, never
        # mutating `g`) and submitted over the Gremlin connection pool
        # (aws.neptune.pool_size). Order does not matter for upserts.
        logger.info(
            "Indexing %s in %s batches across %s concurrent workers.",
            operation_name,
            len(batches),
            concurrency,
        )
        # Reset total_items to 0 here: each per-batch stats carries its own
        # batch total, and stats.merge() sums total_items. Without this reset
        # the seeded len(items) above would be double-counted on the concurrent
        # path. (_execute_single_batch only adds successes/errors, so the
        # per-batch total_items seed is what makes success_rate correct.)
        stats.total_items = 0
        with ContextThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    self._execute_single_batch,
                    batch,
                    traversal_builder,
                    operation_name,
                    IndexingStats(total_items=len(batch)),
                )
                for batch in batches
            ]
            for future in as_completed(futures):
                stats.merge(future.result())

        return stats

    def _execute_single_batch(
        self,
        batch: list[Any],
        traversal_builder: Callable[[GraphTraversalSource, list[Any]], GraphTraversal],
        operation_name: str,
        stats: IndexingStats,
    ) -> IndexingStats:
        """Execute one batch, falling back to per-item indexing on batch failure.

        Accumulates into the supplied ``stats`` and returns it, so the caller can
        either share one stats object (sequential) or merge per-batch objects
        (concurrent) without any locking.
        """
        try:
            g = self.neptune_client.g
            traversal = traversal_builder(g, batch)
            self._execute_with_retries(traversal, operation_name)
            stats.add_success(len(batch))
        except Exception as batch_error:
            logger.warning(
                "Batch %s failed (%s items), falling back to individual indexing: %s",
                operation_name,
                len(batch),
                batch_error,
            )
            for item in batch:
                try:
                    g = self.neptune_client.g
                    traversal = traversal_builder(g, [item])
                    self._execute_with_retries(traversal, operation_name)
                    stats.add_success(1)
                except Exception as item_error:
                    stats.add_error(str(item_error))
                    logger.warning(
                        "Individual %s failed: %s", operation_name, item_error
                    )
        return stats

    def _execute_with_retries(
        self, traversal: GraphTraversal, operation_name: str, *, results: bool = False
    ) -> list[Any]:
        """Run ``traversal`` with retries; with ``results``, return its results.

        Without ``results`` the traversal is iterated (nothing sent back) and
        ``[]`` is returned.
        """
        max_attempts = self.neptune_config.max_attempts
        delay = self.neptune_config.retry_delay_seconds
        attempt = 0
        while True:
            try:
                if results:
                    return list(traversal.toList())
                traversal.iterate()
                return []
            except Exception as e:
                if is_permanent_neptune_error(e) or attempt + 1 == max_attempts:
                    logger.error(
                        "Failed %s after %s attempt(s): %s",
                        operation_name,
                        attempt + 1,
                        e,
                    )
                    raise
                # Exponential backoff with full jitter so concurrent workers do
                # not retry a throttled endpoint in lock-step.
                backoff = delay * (2**attempt)
                sleep_for = _jitter(0, backoff)
                logger.warning(
                    "%s attempt %s failed, retrying in %.2fs: %s",
                    operation_name,
                    attempt + 1,
                    sleep_for,
                    e,
                )
                _sleep(sleep_for)
                attempt += 1

    def _batch_iterator(self, items: list[Any]) -> Iterator[list[Any]]:
        batch_size = self.neptune_config.batch_size
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def _log_indexing_summary(self, item_type_name: str, stats: IndexingStats) -> None:
        if stats.failed_items > 0:
            logger.warning(
                "Indexed %s/%s %s (%s failed) in %.2fs.",
                stats.successful_items,
                stats.total_items,
                item_type_name,
                stats.failed_items,
                stats.processing_time,
            )
        else:
            logger.info(
                "Successfully indexed %s %s in %.2fs.",
                stats.successful_items,
                item_type_name,
                stats.processing_time,
            )
