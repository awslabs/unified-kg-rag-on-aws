# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import random
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, cast

from gremlin_python.process.graph_traversal import (
    GraphTraversal,
    GraphTraversalSource,
    __,
)
from gremlin_python.process.traversal import Cardinality, P

from aws_graphrag.adapters.aws import NeptuneClient
from aws_graphrag.domain.models import (
    Community,
    Config,
    Constants,
    Entity,
    Relationship,
)
from aws_graphrag.ports.indexer import GraphIndexer, IndexingStats
from aws_graphrag.shared import get_logger

logger = get_logger(__name__)


class NeptuneIndexer(GraphIndexer):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.neptune_config = self.config.indexing.neptune
        self.neptune_client = NeptuneClient(config=self.config)

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

    def read_entities(self, ids: list[str]) -> list[Entity]:
        """Read existing entities by id for cross-run merge (best-effort).

        Returns ``[]`` on any error so cross-run merge degrades to overwrite
        rather than failing the run. Reconstructs only the fields the merge needs
        (id/name/description/text_unit_ids/frequency/type). Requires real
        Neptune; validated under the ``aws`` test marker.
        """
        if not ids:
            return []
        try:
            g = self.neptune_client.g
            rows = g.V().has("id", P.within(ids)).valueMap(True).toList()
            entities = []
            for row in rows:
                props = self._flatten_value_map(row)
                if "id" not in props or "name" not in props:
                    continue
                entities.append(
                    Entity.model_validate(
                        {
                            "id": str(props["id"]),
                            "name": str(props["name"]),
                            "type": props.get("type"),
                            "description": props.get("description"),
                            "text_unit_ids": self._as_list(props.get("text_unit_ids")),
                            "frequency": props.get("frequency"),
                        }
                    )
                )
            return entities
        except Exception as e:  # noqa: BLE001 - degrade to overwrite
            logger.warning("read_entities failed (%s); cross-run merge disabled", e)
            return []

    def read_relationships(self, ids: list[str]) -> list[Relationship]:
        """Read existing relationships by id for cross-run merge (best-effort)."""
        if not ids:
            return []
        try:
            g = self.neptune_client.g
            # source_id/target_id are edge TOPOLOGY (endpoint vertex ids), not
            # edge properties, so valueMap() does not contain them. Project the
            # edge's own properties alongside the endpoint vertex ids.
            rows = (
                g.E()
                .has("id", P.within(ids))
                .project("props", "source_id", "target_id")
                .by(__.valueMap(True))
                .by(__.outV().values("id"))
                .by(__.inV().values("id"))
                .toList()
            )
            rels = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                props = self._flatten_value_map(row.get("props"))
                if "id" not in props:
                    continue
                rels.append(
                    Relationship.model_validate(
                        {
                            "id": str(props["id"]),
                            "source_id": str(row.get("source_id")),
                            "target_id": str(row.get("target_id")),
                            "description": props.get("description"),
                            "weight": props.get("weight"),
                            "text_unit_ids": self._as_list(props.get("text_unit_ids")),
                        }
                    )
                )
            return rels
        except Exception as e:  # noqa: BLE001 - degrade to overwrite
            logger.warning(
                "read_relationships failed (%s); cross-run merge disabled", e
            )
            return []

    @staticmethod
    def _flatten_value_map(row: Any) -> dict[str, Any]:
        """Gremlin valueMap returns {key: [value]}; flatten single-element lists."""
        if not isinstance(row, dict):
            return {}
        flat: dict[str, Any] = {}
        for key, value in row.items():
            k = str(key)
            if isinstance(value, list):
                flat[k] = value[0] if len(value) == 1 else value
            else:
                flat[k] = value
        return flat

    @staticmethod
    def _as_list(value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [str(v) for v in value]
        return [str(value)]

    def initialize(self) -> bool:
        return True

    def index_entities(self, entities: list[Entity]) -> IndexingStats:
        def get_traversal_builder(label: str) -> Callable:
            def builder(g: GraphTraversalSource, batch: list[Entity]) -> GraphTraversal:
                t = g
                for entity in batch:
                    props = self._build_vertex_properties(
                        entity,
                        {
                            "name": entity.name,
                            "type": entity.type,
                            "description": entity.description,
                            "rank": entity.rank,
                            "confidence": entity.confidence,
                            "text_unit_ids": entity.text_unit_ids,
                            "community_ids": entity.community_ids,
                        },
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
        def get_traversal_builder(label: str, entity_label: str) -> Callable:
            def builder(
                g: GraphTraversalSource, batch: list[Relationship]
            ) -> GraphTraversal:
                # Drop any pre-existing edges for this batch's ids first so a
                # re-run (or overlap with a prior partial run) does not create
                # duplicate parallel edges with the same id. Edges are not
                # cleared by the entity label-clear, so without this the full
                # index path is not idempotent (unlike upsert_relationships).
                rel_ids = [rel.id for rel in batch]
                g.E().has("id", P.within(rel_ids)).drop().iterate()

                # Each addE() traversal terminates on the created edge, so the
                # next edge must start from the source again (not from the prior
                # edge's traversal — that raised "'GraphTraversal' object is not
                # callable"). Fan each addE out via sideEffect from one root.
                t: GraphTraversal = g.inject(1)
                for rel in batch:
                    props = self._build_vertex_properties(
                        rel,
                        {
                            "source_name": rel.source_name,
                            "target_name": rel.target_name,
                            "weight": rel.weight,
                            "description": rel.description,
                            "rank": rel.rank,
                            "text_unit_ids": rel.text_unit_ids,
                        },
                    )
                    add_edge = (
                        __.V()
                        .hasLabel(entity_label)
                        .has("id", rel.source_id)
                        .addE(rel.type or "RELATED_TO")
                        .to(__.V().hasLabel(entity_label).has("id", rel.target_id))
                        .property("id", rel.id)
                    )
                    self._set_edge_properties_on_traversal(add_edge, props)
                    t = t.sideEffect(add_edge)
                return cast(GraphTraversal, t)

            return builder

        return self._index_generic(
            relationships,
            "Relationship",
            self.neptune_config.entity_label_prefix.capitalize(),
            "",
            get_traversal_builder,
            entity_label=self.neptune_config.entity_label_prefix.capitalize(),
        )

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
                    props = self._build_vertex_properties(
                        entity,
                        {
                            "name": entity.name,
                            "type": entity.type,
                            "description": entity.description,
                            "rank": entity.rank,
                            "confidence": entity.confidence,
                            "text_unit_ids": entity.text_unit_ids,
                            "community_ids": entity.community_ids,
                        },
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
        """

        def get_traversal_builder(label: str, entity_label: str) -> Callable:
            def builder(
                g: GraphTraversalSource, batch: list[Relationship]
            ) -> GraphTraversal:
                # Drop any pre-existing edges for this batch's ids in one
                # statement, then build a single chained add-edge traversal.
                # Each edge MUST start from g.V() (the source), not from the
                # previous edge's traversal: an addE() traversal ends on the
                # created edge, so chaining `.V()`/`.E()` off it raised
                # "'GraphTraversal' object is not callable". sideEffect lets us
                # fan each addE out from the same source within one traversal.
                rel_ids = [rel.id for rel in batch]
                g.E().has("id", P.within(rel_ids)).drop().iterate()

                t: GraphTraversal = g.inject(1)
                for rel in batch:
                    props = self._build_vertex_properties(
                        rel,
                        {
                            "source_name": rel.source_name,
                            "target_name": rel.target_name,
                            "weight": rel.weight,
                            "description": rel.description,
                            "rank": rel.rank,
                            "text_unit_ids": rel.text_unit_ids,
                        },
                    )
                    add_edge = (
                        __.V()
                        .hasLabel(entity_label)
                        .has("id", rel.source_id)
                        .addE(rel.type or "RELATED_TO")
                        .to(__.V().hasLabel(entity_label).has("id", rel.target_id))
                        .property("id", rel.id)
                    )
                    self._set_edge_properties_on_traversal(add_edge, props)
                    t = t.sideEffect(add_edge)
                return cast(GraphTraversal, t)

            return builder

        return self._index_generic(
            relationships,
            "Relationship",
            self.neptune_config.entity_label_prefix.capitalize(),
            "",
            get_traversal_builder,
            entity_label=self.neptune_config.entity_label_prefix.capitalize(),
        )

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
                    # Edges live between entity-label vertices; scope the edge and
                    # vertex drops to this suffix's labels only.
                    g.E().hasLabel(entity_label).has(
                        "id", P.within(id_batch)
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

    def _set_edge_properties_on_traversal(
        self, traversal: GraphTraversal, props: dict[str, Any]
    ) -> None:
        """Set properties on an EDGE traversal.

        Neptune edge properties differ from vertex properties in two ways that
        make the vertex helpers (:meth:`_add_properties_to_traversal` /
        :meth:`_set_properties_on_traversal`) unsafe here:

        1. Cardinality (``single``/``set``) may NOT be specified for edge
           properties — doing so raises ``UnsupportedOperationException``
           ("Cardinality specification may not be used with Edge properties").
        2. Edges cannot hold multi-valued properties at all, so a list (e.g.
           ``text_unit_ids``) must be serialized to a single JSON string rather
           than emitted as repeated ``property(key, item)`` calls (which on a
           vertex create a multi-property but on an edge silently keep only the
           last value).

        Edge ``id`` is always single-valued and is set positionally; re-running
        an upsert overwrites a scalar in place, so no explicit cardinality is
        needed for idempotency.
        """
        for key, value in props.items():
            if value is None:
                continue
            if isinstance(value, list):
                serialized = self._truncate(json.dumps(value))
                traversal.property(key, serialized)
            else:
                traversal.property(key, value)

    def _build_vertex_properties(
        self, item: Any, base_props: dict[str, Any]
    ) -> dict[str, Any]:
        properties = {
            k: self._safe_property_value(v)
            for k, v in base_props.items()
            if v is not None
        }
        if hasattr(item, "attributes") and item.attributes:
            for key, value in item.attributes.items():
                if value is not None:
                    properties[f"{Constants.ATTRIBUTE_PREFIX.value}_{key}"] = (
                        self._safe_property_value(value)
                    )
        return properties

    def _safe_property_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return self._truncate(json.dumps(value))
        return self._truncate(value)

    def _truncate(self, value: Any) -> Any:
        max_len = self.neptune_config.property_max_length
        return (
            value[:max_len]
            if isinstance(value, str) and len(value) > max_len
            else value
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
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    self._execute_single_batch,
                    batch,
                    traversal_builder,
                    operation_name,
                    IndexingStats(),
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
        self, traversal: GraphTraversal, operation_name: str
    ) -> None:
        max_retries = self.neptune_config.max_retries
        delay = self.neptune_config.retry_delay_seconds
        for attempt in range(max_retries + 1):
            try:
                traversal.iterate()
                return
            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        "Failed %s after %s attempts: %s",
                        operation_name,
                        attempt + 1,
                        e,
                    )
                    raise
                # Exponential backoff with full jitter so concurrent workers do
                # not retry a throttled endpoint in lock-step.
                backoff = delay * (2**attempt)
                sleep_for = random.uniform(0, backoff)
                logger.warning(
                    "%s attempt %s failed, retrying in %.2fs: %s",
                    operation_name,
                    attempt + 1,
                    sleep_for,
                    e,
                )
                time.sleep(sleep_for)

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
