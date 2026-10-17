# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import math
import random
import time
from collections.abc import Callable, Collection, Iterator
from concurrent.futures import as_completed
from typing import TYPE_CHECKING, Any, cast

from gremlin_python.process.graph_traversal import (
    GraphTraversal,
    GraphTraversalSource,
    __,
)
from gremlin_python.process.traversal import Cardinality, P, Traversal

from unified_kg_rag.adapters.aws.neptune import (
    INGESTION_REQUEST_TIMEOUT_SECONDS,
    NeptuneClient,
    is_permanent_neptune_error,
    run_traversal,
)
from unified_kg_rag.adapters.storage.neptune_codec import (
    NAME_KEY_PROPERTY,
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
from unified_kg_rag.shared import AWSServiceError, GraphRAGException, get_logger
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor

if TYPE_CHECKING:
    import boto3

logger = get_logger(__name__)

# Retry-backoff seams: tests patch these module attributes rather than the
# process-global ``time.sleep``/``random.uniform``, which other threads share.
_sleep = time.sleep
# Retry-backoff jitter only; drawn from the OS so CWE-338 scanners do not
# flag the seedable module PRNG.
_jitter = random.SystemRandom().uniform

# Vertices one clear request drops, and the passes a clear allows beyond what
# the label's count needs.
_DELETE_BATCH_SIZE = 500
_EXTRA_DELETE_PASSES = 3


class NeptuneIndexer(GraphIndexer):
    # Emit a progress line every N edges during the per-edge relationship write
    # so a large run (tens of thousands of edges) is distinguishable from a hang.
    _PROGRESS_LOG_EVERY: int = 2000

    def __init__(
        self, config: Config, boto_session: "boto3.Session | None" = None
    ) -> None:
        super().__init__(config)
        self.neptune_config = self.config.indexing.neptune
        self.neptune_client = NeptuneClient(
            config=self.config,
            boto_session=boto_session,
            timeout_seconds=INGESTION_REQUEST_TIMEOUT_SECONDS,
            timeout_setting=None,
        )

    def close(self) -> None:
        """Close the underlying Neptune websocket + thread pool (best-effort)."""
        self.neptune_client.close()

    def clear(self, suffixes: list[str]) -> None:
        labels_to_delete = {
            self._get_label(prefix, s)
            for prefix in (
                self.neptune_config.entity_label_prefix,
                self.neptune_config.community_label_prefix,
            )
            for s in suffixes
        }
        if labels_to_delete:
            logger.info("Clearing Neptune data for labels: %s", labels_to_delete)
        for label in labels_to_delete:
            self._clear_existing_data_by_label(label)

    def read_entities(self, ids: list[str], suffix: str | None = None) -> list[Entity]:
        """Read existing entities by id (cross-run merge, text-unit removal).

        Scoped to this suffix's entity label: entity ids are suffix-independent,
        so an unscoped read could return another tenant's vertex. Inverts the
        vertex encoding (``neptune_codec``), attributes included, so a merged
        entity is written back under its own suffix.

        Raises:
            AWSServiceError: The read failed. An empty result would read as "not
                stored", so the caller would overwrite stored lineage or report
                a text-unit removal that never happened.
        """
        if not ids:
            return []
        entity_label = self._get_label(self.neptune_config.entity_label_prefix, suffix)
        try:
            rows = self._read_batches(
                ids,
                lambda g, batch: g.V()
                .hasLabel(entity_label)
                .has("id", P.within(batch))
                .valueMap(),
                "read_entities",
            )
            entities = [entity_from_vertex(row) for row in rows]
            return [entity for entity in entities if entity is not None]
        except Exception as e:
            raise AWSServiceError(f"Neptune read_entities failed: {e}") from e

    def read_relationships(
        self, ids: list[str], suffix: str | None = None
    ) -> list[Relationship]:
        """Read existing relationships by id (see :meth:`read_entities`).

        Scoped to edges leaving this suffix's entity label (relationship ids are
        suffix-independent). Inverts the edge encoding (``neptune_codec``): the
        type is the edge label and list properties are JSON strings.

        Raises:
            AWSServiceError: The read failed (see :meth:`read_entities`).
        """
        if not ids:
            return []
        entity_label = self._get_label(self.neptune_config.entity_label_prefix, suffix)
        try:
            # source_id/target_id are edge TOPOLOGY (endpoint vertex ids) and
            # the type is the edge label, so neither is in valueMap().
            rows = self._read_batches(
                ids,
                lambda g, batch: g.E()
                .has("id", P.within(batch))
                .where(__.outV().hasLabel(entity_label))
                .project("props", "label", "source_id", "target_id")
                .by(__.valueMap())
                .by(__.label())
                .by(__.outV().values("id"))
                .by(__.inV().values("id")),
                "read_relationships",
            )
            rels = [
                relationship_from_edge(
                    row.get("props") or {},
                    str(row.get("label")),
                    row.get("source_id"),
                    row.get("target_id"),
                )
                for row in rows
            ]
            return [rel for rel in rels if rel is not None]
        except Exception as e:
            raise AWSServiceError(f"Neptune read_relationships failed: {e}") from e

    def find_ids_citing(
        self,
        text_unit_ids: list[str],
        suffix: str | None = None,
        *,
        communities: bool = False,
    ) -> dict[str, list[str]] | None:
        """Entity (or community) vertices of ``suffix`` citing any of
        ``text_unit_ids``, with their text units (see the port).

        Raises:
            AWSServiceError: The read failed.
        """
        if not text_unit_ids:
            return {}
        prefix = (
            self.neptune_config.community_label_prefix
            if communities
            else self.neptune_config.entity_label_prefix
        )
        label = self._get_label(prefix, suffix)
        try:
            rows = self._read_batches(
                text_unit_ids,
                lambda g, batch: g.V()
                .hasLabel(label)
                .has("text_unit_ids", P.within(batch))
                .dedup()
                .project("id", "text_unit_ids")
                .by("id")
                .by(__.values("text_unit_ids").fold()),
                "find_ids_citing",
            )
        except Exception as e:
            raise AWSServiceError(f"Neptune find_ids_citing failed: {e}") from e
        return {
            str(row["id"]): [str(u) for u in row.get("text_unit_ids") or []]
            for row in rows
            if isinstance(row, dict) and "id" in row
        }

    def find_entity_ids_by_name_key(
        self, name_keys: list[str], suffix: str | None = None
    ) -> list[str] | None:
        """Entities of ``suffix`` whose stored ``name_key`` is one of
        ``name_keys`` (see the port).

        Raises:
            AWSServiceError: The read failed.
        """
        if not name_keys:
            return []
        label = self._get_label(self.neptune_config.entity_label_prefix, suffix)
        try:
            rows = self._read_batches(
                name_keys,
                lambda g, batch: g.V()
                .hasLabel(label)
                .has(NAME_KEY_PROPERTY, P.within(batch))
                .values("id"),
                "find_entity_ids_by_name_key",
            )
        except Exception as e:
            raise AWSServiceError(
                f"Neptune find_entity_ids_by_name_key failed: {e}"
            ) from e
        return sorted({str(row) for row in rows})

    def read_neighbor_ids(
        self, entity_ids: list[str], suffix: str | None = None
    ) -> dict[str, list[str]]:
        """Read the ids of the entities each entity is related to.

        Scoped to this suffix's entity label (community vertices are not
        neighbours). Raises :class:`AWSServiceError` when the read fails, as
        :meth:`read_entities` does.
        """
        if not entity_ids:
            return {}
        entity_label = self._get_label(self.neptune_config.entity_label_prefix, suffix)
        try:
            rows = self._read_batches(
                entity_ids,
                lambda g, batch: g.V()
                .hasLabel(entity_label)
                .has("id", P.within(batch))
                .project("id", "neighbors")
                .by("id")
                .by(__.both().hasLabel(entity_label).values("id").dedup().fold()),
                "read_neighbor_ids",
            )
            return {str(row["id"]): [str(n) for n in row["neighbors"]] for row in rows}
        except Exception as e:
            raise AWSServiceError(f"Neptune read_neighbor_ids failed: {e}") from e

    def read_entity_names(self, suffix: str | None = None) -> list[tuple[str, str]]:
        """Project ``(id, name)`` for all existing entities in the suffix's label.

        Best-effort (``[]`` on error): powers cross-run *fuzzy* merge, which needs
        old entities whose ids differ from a delta entity's. Requires real
        Neptune; validated under the ``aws`` test marker.
        """
        entity_label = self._get_label(self.neptune_config.entity_label_prefix, suffix)
        try:
            rows = self._execute_with_retries(
                lambda g: g.V()
                .hasLabel(entity_label)
                .project("id", "name")
                .by("id")
                .by("name"),
                "read_entity_names",
                results=True,
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

    def index_entities(
        self, entities: list[Entity], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        """Rebuild each suffix's entity label: clear it, then write the entities.

        Written with the upsert builder (:meth:`_entity_vertex_builder`): a
        retry after a batch whose response was lost finds the vertices it
        wrote instead of adding them a second time.
        """
        return self._index_generic(
            entities,
            "Entity",
            self.neptune_config.entity_label_prefix,
            self._entity_vertex_builder,
            clear_first=True,
            suffixes=suffixes,
        )

    def index_communities(
        self, communities: list[Community], *, suffixes: Collection[str] = ()
    ) -> IndexingStats:
        return self._index_communities(communities, upsert=False, suffixes=suffixes)

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
        self,
        communities: list[Community],
        *,
        upsert: bool,
        suffixes: Collection[str] = (),
    ) -> IndexingStats:
        if not communities and not suffixes:
            return IndexingStats()

        total_stats = IndexingStats()
        grouped_items = self._group_items_by_suffix(communities, suffixes)

        for suffix, comms in grouped_items.items():
            stats = IndexingStats()
            start_time = time.time()
            community_label = self._get_label(
                self.neptune_config.community_label_prefix, suffix
            )
            entity_label = self._get_label(
                self.neptune_config.entity_label_prefix, suffix
            )

            if not upsert:
                self._clear_existing_data_by_label(community_label)
            if not comms:
                continue

            def community_vertex_builder(
                g: GraphTraversalSource,
                batch: list[Community],
                community_label: str = community_label,
            ) -> GraphTraversal:
                # Create-or-match by id, so neither an incremental re-run nor
                # a retry after a lost response adds a duplicate vertex.
                t: Any = g
                for comm in batch:
                    t = self._upsert_vertex(t, community_label, comm.id)
                    self._set_properties_on_traversal(
                        t,
                        self._build_vertex_properties(
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
                        ),
                    )
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
                    # Drop this community's existing membership edges, so an
                    # entity that left the community does not stay a member;
                    # after a failed drop the membership is left as it was.
                    def drop_members(
                        g: GraphTraversalSource,
                        community_label: str = community_label,
                        community_id: str = comm.id,
                    ) -> GraphTraversal:
                        return (
                            g.V()
                            .hasLabel(community_label)
                            .has("id", community_id)
                            .inE("MemberOf")
                            .drop()
                        )

                    try:
                        self._execute_with_retries(
                            drop_members, "Clearing MemberOf edges"
                        )
                    except Exception as e:
                        stats.add_error(str(e), ids=[comm.id])
                        logger.warning(
                            "Failed clearing MemberOf edges for community '%s': %s",
                            comm.id,
                            e,
                        )
                        continue
                for entity_id_batch in self._batch_iterator(comm.entity_ids):

                    def add_members(
                        g: GraphTraversalSource,
                        entity_label: str = entity_label,
                        community_label: str = community_label,
                        entity_ids: list[str] = entity_id_batch,
                        community_id: str = comm.id,
                    ) -> GraphTraversal:
                        # Only entities not yet members: a retry after a lost
                        # response must not add the edges a second time.
                        return (
                            g.V()
                            .hasLabel(entity_label)
                            .has("id", P.within(entity_ids))
                            .not_(
                                __.out("MemberOf")
                                .hasLabel(community_label)
                                .has("id", community_id)
                            )
                            .addE("MemberOf")
                            .to(
                                __.V().hasLabel(community_label).has("id", community_id)
                            )
                        )

                    try:
                        self._execute_with_retries(
                            add_members, "Community edge indexing"
                        )
                    except Exception as e:
                        stats.add_error(str(e), ids=[comm.id])
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

        The same writes as :meth:`index_entities` without the label-wide
        clear: an existing vertex is updated in place and a missing one is
        created, so a re-run adds no duplicate vertices.
        """
        return self._index_generic(
            entities,
            "Entity",
            self.neptune_config.entity_label_prefix,
            self._entity_vertex_builder,
            clear_first=False,  # upsert is non-destructive
        )

    def _entity_vertex_builder(
        self, label: str
    ) -> Callable[[GraphTraversalSource, list[Entity]], GraphTraversal]:
        """Build one traversal that creates or updates a batch of entity vertices.

        ``V().has(label, 'id', x).fold().coalesce(unfold(), addV(label))``
        matches an existing vertex or creates it, and its properties are
        replaced (:meth:`_set_properties_on_traversal`), so running the same
        batch twice leaves the graph as running it once.
        """

        def builder(g: GraphTraversalSource, batch: list[Entity]) -> GraphTraversal:
            t: Any = g
            for entity in batch:
                t = self._upsert_vertex(t, label, entity.id)
                self._set_properties_on_traversal(
                    t,
                    entity_properties(entity, self.neptune_config.property_max_length),
                )
            return cast(GraphTraversal, t)

        return builder

    @staticmethod
    def _upsert_vertex(t: Any, label: str, vertex_id: str) -> GraphTraversal:
        """Continue ``t`` with the vertex ``label``/``vertex_id``, created if missing."""
        return cast(
            GraphTraversal,
            t.V()
            .has(label, "id", vertex_id)
            .fold()
            .coalesce(__.unfold(), __.add_v(label).property("id", vertex_id)),
        )

    def upsert_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Idempotently merge relationship edges into the live graph.

        Drops any existing edge with the same id before re-adding it, so the
        operation is repeatable without creating duplicate parallel edges.
        The port's :meth:`index_relationships` delegates here, so a full run
        and a delta share this per-edge write path.
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

        Idempotent (``fold().coalesce(unfold(), addE)``): the caller dropped
        the edge first, so the existing edge is found only when a retry
        follows a write whose response was lost; adding it again would leave
        two parallel edges with one id.
        """
        add_edge = (
            __.V()
            .hasLabel(entity_label)
            .has("id", rel.source_id)
            .addE(relationship_label(rel))
            .to(__.V().hasLabel(entity_label).has("id", rel.target_id))
            .property("id", rel.id)
        )
        props = relationship_properties(rel, self.neptune_config.property_max_length)
        for key, value in props.items():
            add_edge.property(key, value)
        # Return the edge's id: with no source vertex addE gets no traverser
        # and raises nothing (a missing target does raise), so an empty result
        # is the only sign of a dropped edge.
        return cast(
            GraphTraversal,
            g.E()
            .has("id", rel.id)
            .where(__.outV().hasLabel(entity_label))
            .fold()
            .coalesce(__.unfold(), add_edge)
            .id_(),
        )

    def _write_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Write relationship edges one small traversal per edge (idempotent).

        For each suffix group: drop existing edges by id in batches (so re-runs
        don't duplicate parallel edges), then add each edge as its own
        traversal, ``index_concurrency`` at a time. An edge whose drop failed
        is not re-added (that would duplicate it) and counts as failed, as does
        a single failed write; neither aborts the others. Emits a progress log
        every ``_PROGRESS_LOG_EVERY`` edges so a large run is distinguishable
        from a hang.
        """
        if not relationships:
            return IndexingStats()

        total_stats = IndexingStats()
        grouped_items = self._group_items_by_suffix(relationships)

        for suffix, rels in grouped_items.items():
            stats = IndexingStats(total_items=len(rels))
            start_time = time.time()
            entity_label = self._get_label(
                self.neptune_config.entity_label_prefix, suffix
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
            writable: list[Relationship] = []
            for batch in self._batch_iterator(rels):

                def drop_edges(
                    g: GraphTraversalSource,
                    entity_label: str = entity_label,
                    batch: list[Relationship] = batch,
                ) -> GraphTraversal:
                    return (
                        g.E()
                        .has("id", P.within([rel.id for rel in batch]))
                        .where(__.outV().hasLabel(entity_label))
                        .drop()
                    )

                try:
                    self._execute_with_retries(drop_edges, "Dropping existing edges")
                    writable.extend(batch)
                except Exception as e:
                    stats.add_error(
                        f"dropping existing edges failed: {e}",
                        len(batch),
                        ids=[rel.id for rel in batch],
                    )
                    logger.warning("Failed dropping existing edge batch: %s", e)

            def write(
                rel: Relationship, entity_label: str = entity_label
            ) -> str | None:
                """Write one edge; the error, or None once it is written."""
                try:
                    if self._execute_with_retries(
                        lambda g: self._build_add_edge_traversal(g, rel, entity_label),
                        "Relationship indexing",
                        results=True,
                    ):
                        return None
                    return f"source entity '{rel.source_id}' not found"
                except Exception as e:
                    return str(e)

            workers = max(1, min(self.neptune_config.index_concurrency, len(writable)))
            with ContextThreadPoolExecutor(max_workers=workers) as executor:
                for index, (rel, error) in enumerate(
                    zip(writable, executor.map(write, writable), strict=True),
                    start=1,
                ):
                    if error is None:
                        stats.add_success(1)
                    else:
                        stats.add_error(error, ids=[rel.id])
                        logger.warning(
                            "Relationship '%s' not written: %s", rel.id, error
                        )
                    if index % self._PROGRESS_LOG_EVERY == 0:
                        logger.info(
                            "  ...%s/%s relationships written for '%s'",
                            index,
                            len(writable),
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
        documents from OpenSearch.

        Raises:
            AWSServiceError: The read failed. An empty result would leave the
                orphaned relationship documents behind, or miss the stored
                edges a remapped delta edge must merge into.
        """
        if not entity_ids:
            return []
        label = (
            self._get_label(self.neptune_config.entity_label_prefix, suffix)
            if suffix is not None
            else None
        )
        try:
            rows = self._read_batches(
                entity_ids,
                lambda g, batch: (g.V().hasLabel(label) if label is not None else g.V())
                .has("id", P.within(batch))
                .bothE()
                .values("id")
                .dedup(),
                "find_incident_relationship_ids",
            )
            return sorted({str(rid) for rid in rows if rid is not None})
        except Exception as e:
            raise AWSServiceError(
                f"Neptune find_incident_relationship_ids failed: {e}"
            ) from e

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
            entity_label = self._get_label(
                self.neptune_config.entity_label_prefix, suffix
            )
            community_label = self._get_label(
                self.neptune_config.community_label_prefix, suffix
            )

        for id_batch in self._batch_iterator(ids):

            # Scope the drop to this suffix. Edges are labeled by their
            # relationship type (e.g. RELATED_TO / MemberOf), NOT by the entity
            # label, so filtering edges with hasLabel(entity_label) would match
            # nothing and leak stale edges. Instead scope edges by id AND by an
            # endpoint vertex carrying this suffix's entity label
            # (where(outV().hasLabel(...))), which both restricts to this
            # tenant and matches real edges. A drop is safe to retry.
            def drop_edges(
                g: GraphTraversalSource, id_batch: list[str] = id_batch
            ) -> GraphTraversal:
                edges = g.E().has("id", P.within(id_batch))
                if entity_label:
                    edges = edges.where(__.outV().hasLabel(entity_label))
                return edges.drop()

            def drop_vertices(
                g: GraphTraversalSource, id_batch: list[str] = id_batch
            ) -> GraphTraversal:
                vertices = (
                    g.V().hasLabel(entity_label, community_label)
                    if entity_label and community_label
                    else g.V()
                )
                return vertices.has("id", P.within(id_batch)).drop()

            try:
                self._execute_with_retries(drop_edges, "Deleting edges by id")
                self._execute_with_retries(drop_vertices, "Deleting vertices by id")
                stats.add_success(len(id_batch))
            except Exception as e:
                stats.add_error(str(e), len(id_batch), ids=list(id_batch))
                logger.warning("Failed to delete ids batch: %s", e)

        return stats

    def _set_properties_on_traversal(
        self, traversal: GraphTraversal, props: dict[str, Any]
    ) -> None:
        """Replace the vertex's properties with ``props`` (every vertex write).

        Every stored property but ``id`` is dropped first, so a property the
        item no longer has (a value now ``None``, a removed attribute) does
        not keep its old value, and a re-write does not grow stale list
        members. Scalars use ``Cardinality.single``; list values are written
        as multi-valued ``Cardinality.set`` properties, what the read path
        expects.
        """
        traversal.sideEffect(__.properties().has_key(P.neq("id")).drop())
        for key, value in props.items():
            if value is None:
                continue
            if isinstance(value, list):
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
        traversal_builder_func: Callable,
        *,
        clear_first: bool,
        suffixes: Collection[str] = (),
    ) -> IndexingStats:
        if not items and not suffixes:
            return IndexingStats()

        grouped_items = self._group_items_by_suffix(items, suffixes)
        total_stats = IndexingStats()

        for suffix, chunk in grouped_items.items():
            label = self._get_label(label_prefix, suffix)
            if clear_first:
                self._clear_existing_data_by_label(label)
            if not chunk:
                continue

            start_time = time.time()

            logger.info(
                "Indexing %s %ss for '%s'...", len(chunk), item_type_name.lower(), label
            )
            traversal_builder = traversal_builder_func(label=label)
            stats = self._execute_batch_traversal(
                chunk, traversal_builder, f"{item_type_name} indexing"
            )

            stats.processing_time = time.time() - start_time
            total_stats.merge(stats)

        self._log_indexing_summary(f"{item_type_name.lower()}s", total_stats)
        return total_stats

    def _clear_existing_data_by_label(
        self, label: str, batch_size: int = _DELETE_BATCH_SIZE
    ) -> None:
        """Drop every vertex of ``label``, ``batch_size`` per request.

        Bounded requests keep each drop's transaction small, and each request
        is retried like a write (a drop is safe to repeat). A request returns
        how many vertices it dropped, so a short one is the last: no count of
        the label per pass. The label is counted once up front to bound the
        passes: if requests keep reporting full batches past that bound (a
        backend whose count does not reflect the drop, or writers adding
        vertices meanwhile) the clear fails instead of looping forever.

        Raises:
            AWSServiceError: A request failed after its retries, or the label
                still reported full batches after
                ``ceil(count / batch_size) + _EXTRA_DELETE_PASSES`` passes.
        """
        try:
            (initial,) = self._execute_with_retries(
                lambda g: g.V().hasLabel(label).count(),
                "Counting vertices to clear",
                results=True,
            )
            max_passes = math.ceil(int(initial) / batch_size) + _EXTRA_DELETE_PASSES
            total = 0
            for _ in range(max_passes):
                (dropped,) = self._execute_with_retries(
                    lambda g: g.V()
                    .hasLabel(label)
                    .limit(batch_size)
                    .sideEffect(__.drop())
                    .count(),
                    "Clearing vertices",
                    results=True,
                )
                total += int(dropped)
                if dropped < batch_size:
                    break
            else:
                raise AWSServiceError(
                    f"Deleting vertices with label '{label}' did not finish: "
                    f"{max_passes} passes of {batch_size} each dropped a full "
                    f"batch ({total} reported dropped, {initial} counted before)"
                )
        except Exception as e:
            logger.error("Failed to clear data for label '%s': %s", label, e)
            if isinstance(e, GraphRAGException):
                raise
            raise AWSServiceError(
                f"Clearing Neptune label '{label}' failed: {e}"
            ) from e
        if total:
            logger.info("Deleted %s vertices with label '%s'", total, label)

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
            # Not an error yet: the per-item fallback below may still index
            # every item; one that fails there logs it.
            self._execute_with_retries(
                lambda g: traversal_builder(g, batch),
                operation_name,
                exhausted_level=logging.WARNING,
            )
            stats.add_success(len(batch))
        except Exception as batch_error:
            logger.warning(
                "Batch %s failed (%s items), falling back to individual indexing: %s",
                operation_name,
                len(batch),
                batch_error,
            )
            for item in batch:

                def build_one(
                    g: GraphTraversalSource, item: Any = item
                ) -> GraphTraversal:
                    return traversal_builder(g, [item])

                try:
                    self._execute_with_retries(build_one, operation_name)
                    stats.add_success(1)
                except Exception as item_error:
                    item_id = getattr(item, "id", None)
                    stats.add_error(
                        str(item_error),
                        ids=[str(item_id)] if item_id is not None else None,
                    )
                    logger.warning(
                        "Individual %s failed: %s", operation_name, item_error
                    )
        return stats

    def _execute_with_retries(
        self,
        build: Callable[[GraphTraversalSource], Traversal],
        operation_name: str,
        *,
        results: bool = False,
        exhausted_level: int = logging.ERROR,
    ) -> list[Any]:
        """Run the traversal ``build`` makes, with retries (``run_traversal``).

        Each attempt builds it anew from the client's current source, so after
        a lost connection (a Neptune restart or failover) the retry goes over
        a new one. With ``results`` its results are returned, else ``[]``.
        The final failure is logged at ``exhausted_level``.
        """
        max_attempts = self.neptune_config.max_attempts
        delay = self.neptune_config.retry_delay_seconds
        attempt = 0
        while True:
            try:
                return run_traversal(self.neptune_client, build, results=results)
            except Exception as e:
                if is_permanent_neptune_error(e) or attempt + 1 == max_attempts:
                    logger.log(
                        exhausted_level,
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

    def _read_batches(
        self,
        ids: list[str],
        query: Callable[[GraphTraversalSource, list[str]], Traversal],
        operation_name: str,
    ) -> list[Any]:
        """The rows ``query`` reads for each batch of ``ids``, retried like a write."""
        rows: list[Any] = []
        for id_batch in self._batch_iterator(ids):

            def build(
                g: GraphTraversalSource, id_batch: list[str] = id_batch
            ) -> Traversal:
                return query(g, id_batch)

            rows.extend(self._execute_with_retries(build, operation_name, results=True))
        return rows

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
