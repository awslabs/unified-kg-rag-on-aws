# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import math
import random
import threading
import time
import uuid
from collections.abc import Callable, Collection, Iterator
from concurrent.futures import as_completed
from typing import TYPE_CHECKING, Any, cast

from gremlin_python.process.graph_traversal import (
    GraphTraversal,
    GraphTraversalSource,
    __,
)
from gremlin_python.process.traversal import Cardinality, P, T, TextP, Traversal

from unified_kg_rag.adapters.aws.neptune import (
    CUSTOM_VERTEX_IDS,
    VERTEX_ID_MODE_PROPERTY,
    VERTEX_ID_SWITCH_LABEL,
    NeptuneClient,
    is_permanent_neptune_error,
    run_traversal,
    sample_vertex_query,
    stored_vertex_id_mode,
    vertex_id,
    vertex_id_mode_error,
    vertex_id_switch_error,
    vertex_id_switch_query,
    vertex_ids,
)
from unified_kg_rag.adapters.storage.neptune_codec import (
    BUILD_TAG_PROPERTY,
    NAME_KEY_PROPERTY,
    edge_text_unit_ids,
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
from unified_kg_rag.shared import (
    AWSServiceError,
    GraphRAGException,
    NamespaceLeaseError,
    get_logger,
)
from unified_kg_rag.shared.exceptions import StoreConfigurationError
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor
from unified_kg_rag.shared.write_fence import (
    INGESTION_REQUEST_TIMEOUT_SECONDS,
    check_write_fence,
)

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


def _stale_edges(
    entity_label: str, tag: str
) -> Callable[[GraphTraversalSource], GraphTraversal]:
    """The edges leaving ``entity_label`` vertices a build tagged ``tag`` did
    not write."""

    def select(g: GraphTraversalSource) -> GraphTraversal:
        # From the label's vertices (label-indexed) rather than a scan of
        # every edge: relationships (and an older build's MemberOf edges)
        # leave an entity.
        return g.V().hasLabel(entity_label).outE().not_(__.has(BUILD_TAG_PROPERTY, tag))

    return select


def _stale_vertices(
    label: str, tag: str
) -> Callable[[GraphTraversalSource], GraphTraversal]:
    """The ``label`` vertices a build tagged ``tag`` did not write."""

    def select(g: GraphTraversalSource) -> GraphTraversal:
        return g.V().hasLabel(label).not_(__.has(BUILD_TAG_PROPERTY, tag))

    return select


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
        # Set between begin_rebuild and end_rebuild (see there).
        self._rebuild_tag: str | None = None
        # indexing.neptune.vertex_ids: the labels found in this mode (or
        # empty) and those a full build is switching to it
        # (_check_vertex_ids).
        self._custom_ids = self.neptune_config.vertex_ids == CUSTOM_VERTEX_IDS
        self._checked_labels: set[str] = set()
        self._switching_labels: set[str] = set()
        self._labels_lock = threading.Lock()

    def close(self) -> None:
        """Close the underlying Neptune websocket + thread pool (best-effort)."""
        self.neptune_client.close()

    # --- vertex ids (indexing.neptune.vertex_ids) ------------------------------

    def _label(self, prefix: str, suffix: str | None) -> str:
        """The label of ``prefix`` under ``suffix``, checked for this run's
        vertex ids (:meth:`_check_vertex_ids`)."""
        label = self._get_label(prefix, suffix)
        self._check_vertex_ids(label)
        return label

    def _check_vertex_ids(self, label: str) -> None:
        """Refuse ``label`` if its vertices are of the other vertex-id mode.

        Checked once per label, before its first read or write. One stored
        vertex shows the mode (its vertex id is ``<label>:<id>`` or not):
        every write is checked first, so a label holds one mode, except
        while a full build switches it. That build writes a switch vertex
        first (``VERTEX_ID_SWITCH_LABEL``) and drops it after its sweep,
        which drops every vertex the build did not write. A switch left
        unfinished (a failed write, a kill) is therefore refused too, until
        a full build finishes it. Unchecked, custom-mode lookups would find
        no property-mode vertex (reads see nothing stored, writes add a
        second copy), and property-mode writes would add vertices that
        custom-mode lookups never find.

        Raises:
            StoreConfigurationError: ``label`` holds the other mode's
                vertices, or a switch left unfinished, outside a full build.
            AWSServiceError: The check's read failed.
        """
        if label in self._checked_labels:
            return
        with self._labels_lock:
            if label in self._checked_labels:
                return
            mode = self.neptune_config.vertex_ids
            try:
                switching = self._execute_with_retries(
                    vertex_id_switch_query(label),
                    f"Reading the vertex-id switch of '{label}'",
                    results=True,
                    fenced=False,
                )
                stored = stored_vertex_id_mode(
                    label,
                    self._execute_with_retries(
                        sample_vertex_query(label),
                        f"Reading the vertex ids of '{label}'",
                        results=True,
                        fenced=False,
                    ),
                )
            except NamespaceLeaseError:
                raise
            except Exception as e:
                if isinstance(e, GraphRAGException):
                    raise
                raise AWSServiceError(
                    f"Neptune vertex-id check of '{label}' failed: {e}"
                ) from e
            if self._rebuild_tag is not None:
                if switching or stored not in (None, mode):
                    self._write_vertex_id_switch(label, mode)
                    self._switching_labels.add(label)
            elif switching:
                raise StoreConfigurationError(
                    vertex_id_switch_error(label, str(switching[0]))
                )
            elif stored is not None and stored != mode:
                raise StoreConfigurationError(vertex_id_mode_error(label, stored, mode))
            self._checked_labels.add(label)

    def _write_vertex_id_switch(self, label: str, mode: str) -> None:
        """Record that a full build is switching ``label`` to ``mode``."""
        logger.warning(
            "Neptune label '%s' holds vertices of the other vertex-id mode: "
            "the full build rewrites it with vertex_ids '%s' (queries see "
            "only what it has rewritten until it ends)",
            label,
            mode,
        )
        self._execute_with_retries(
            lambda g: g.V()
            .hasLabel(VERTEX_ID_SWITCH_LABEL)
            .has("id", label)
            .fold()
            .coalesce(
                __.unfold(),
                __.add_v(VERTEX_ID_SWITCH_LABEL).property("id", label),
            )
            .property(Cardinality.single, VERTEX_ID_MODE_PROPERTY, mode)
            .id_(),
            f"Recording the vertex-id switch of '{label}'",
        )

    def _drop_vertex_id_switch(self, label: str) -> None:
        self._execute_with_retries(
            lambda g: g.V().hasLabel(VERTEX_ID_SWITCH_LABEL).has("id", label).drop(),
            f"Dropping the vertex-id switch of '{label}'",
        )

    def _vertices(
        self, t: Any, labels: tuple[str, ...], ids: list[str]
    ) -> GraphTraversal:
        """Continue ``t`` with the vertices of ``labels`` whose id is one of
        ``ids`` (non-empty): looked up by vertex id in custom mode, else by
        the ``id`` property."""
        if self._custom_ids:
            return cast(
                GraphTraversal,
                t.V(*(vid for label in labels for vid in vertex_ids(label, ids))),
            )
        return cast(GraphTraversal, t.V().hasLabel(*labels).has("id", P.within(ids)))

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
        entity_label = self._label(self.neptune_config.entity_label_prefix, suffix)
        try:
            rows = self._read_batches(
                ids,
                lambda g, batch: self._vertices(g, (entity_label,), batch).valueMap(),
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
        entity_label = self._label(self.neptune_config.entity_label_prefix, suffix)
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
        label = self._label(prefix, suffix)
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

    def find_relationship_ids_citing(
        self, text_unit_ids: list[str], suffix: str | None = None
    ) -> dict[str, list[str]] | None:
        """Edges of ``suffix`` citing any of ``text_unit_ids``, with their text
        units (see the port). An edge holds its text units as one JSON string
        (edge properties are single-valued), so this reads the suffix's
        out-edges once per batch of text units.

        Raises:
            AWSServiceError: The read failed.
        """
        if not text_unit_ids:
            return {}
        label = self._label(self.neptune_config.entity_label_prefix, suffix)
        try:
            rows = self._read_batches(
                list(dict.fromkeys(text_unit_ids)),
                lambda g, batch: g.V()
                .hasLabel(label)
                .outE()
                .or_(
                    *(
                        __.has("text_unit_ids", TextP.containing(f'"{unit}"'))
                        for unit in batch
                    )
                )
                .dedup()
                .project("id", "text_unit_ids")
                .by("id")
                .by("text_unit_ids"),
                "find_relationship_ids_citing",
            )
        except Exception as e:
            raise AWSServiceError(
                f"Neptune find_relationship_ids_citing failed: {e}"
            ) from e
        return {
            str(row["id"]): edge_text_unit_ids(row.get("text_unit_ids"))
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
        label = self._label(self.neptune_config.entity_label_prefix, suffix)
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
        entity_label = self._label(self.neptune_config.entity_label_prefix, suffix)
        try:
            rows = self._read_batches(
                entity_ids,
                lambda g, batch: self._vertices(g, (entity_label,), batch)
                .project("id", "neighbors")
                .by("id")
                .by(__.both().hasLabel(entity_label).values("id").dedup().fold()),
                "read_neighbor_ids",
            )
            return {str(row["id"]): [str(n) for n in row["neighbors"]] for row in rows}
        except Exception as e:
            raise AWSServiceError(f"Neptune read_neighbor_ids failed: {e}") from e

    def count_neighbors(
        self,
        entity_ids: list[str],
        suffix: str | None = None,
        at_most: int | None = None,
    ) -> dict[str, int]:
        """Count the entities each entity is related to, in the store (see
        :meth:`read_neighbor_ids`; no ids are returned), stopping at
        ``at_most`` (a hub's neighbours are not all counted). Raises
        :class:`AWSServiceError` when the read fails."""
        if not entity_ids:
            return {}
        entity_label = self._label(self.neptune_config.entity_label_prefix, suffix)

        def neighbors() -> GraphTraversal:
            distinct = __.both().hasLabel(entity_label).values("id").dedup()
            return distinct if at_most is None else distinct.limit(at_most)

        try:
            rows = self._read_batches(
                entity_ids,
                lambda g, batch: self._vertices(g, (entity_label,), batch)
                .project("id", "count")
                .by("id")
                .by(neighbors().count()),
                "count_neighbors",
            )
            return {str(row["id"]): int(row["count"]) for row in rows}
        except Exception as e:
            raise AWSServiceError(f"Neptune count_neighbors failed: {e}") from e

    def read_entity_names(self, suffix: str | None = None) -> list[tuple[str, str]]:
        """Project ``(id, name)`` for all existing entities in the suffix's label.

        Best-effort (``[]`` on error): powers cross-run *fuzzy* merge, which needs
        old entities whose ids differ from a delta entity's. Requires real
        Neptune; validated under the ``aws`` test marker.
        """
        entity_label = self._label(self.neptune_config.entity_label_prefix, suffix)
        try:
            rows = self._execute_with_retries(
                lambda g: g.V()
                .hasLabel(entity_label)
                .project("id", "name")
                .by("id")
                .by("name"),
                "read_entity_names",
                results=True,
                fenced=False,
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
        by id (fold/coalesce). Membership is not written as edges: no query
        reads it (an entity names its communities, ``community_ids``).
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

            if not upsert and self._rebuild_tag is None:
                self._clear_existing_data_by_label(community_label)
            else:
                self._check_vertex_ids(community_label)
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

    def _upsert_vertex(self, t: Any, label: str, item_id: str) -> GraphTraversal:
        """Continue ``t`` with the vertex ``label``/``item_id``, created if missing.

        In custom mode the vertex is found and created by its vertex id
        (``V(id).fold().coalesce(unfold(), addV())``, the chained upsert
        Neptune's guide gives; ``mergeV`` would need engine 1.2.1.0 and, with
        the label in its map, matches less efficiently).
        """
        if self._custom_ids:
            vid = vertex_id(label, item_id)
            return cast(
                GraphTraversal,
                t.V(vid)
                .fold()
                .coalesce(
                    __.unfold(),
                    __.add_v(label).property(T.id, vid).property("id", item_id),
                ),
            )
        return cast(
            GraphTraversal,
            self._rewritten(t.V().has(label, "id", item_id), label)
            .fold()
            .coalesce(__.unfold(), __.add_v(label).property("id", item_id)),
        )

    def _rewritten(self, t: Any, label: str) -> Any:
        """``t`` (property-mode lookups in ``label``), kept to the vertices
        this build wrote when it is switching the label to property ids: a
        custom-id vertex has the ``id`` property too, and rewritten in place
        it would keep its custom vertex id. The new vertex is added instead
        and the sweep drops the old one."""
        if self._rebuild_tag is not None and label in self._switching_labels:
            return t.has(BUILD_TAG_PROPERTY, self._rebuild_tag)
        return t

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
        """Build the one traversal that rewrites one relationship's edge.

        It drops the edge's stored copies (edges with its id whose source
        vertex carries ``entity_label``: relationship ids are
        suffix-independent, so another tenant's identical edge is left
        alone) and adds the edge, in the same request. Neptune runs a
        request as one transaction, so the edge is replaced or left as it
        was: with the drop a request of its own, a run interrupted between
        the two (a kill, a stop, a lost lease, a write that kept failing)
        lost the stored edge for good, as nothing finds it again to rewrite
        it. A retry after a lost response drops and adds again: still one
        edge.

        Written per edge on purpose. Fanning many edges out of one root via
        ``sideEffect`` makes Neptune evaluate a single giant traversal whose
        per-edge cost is ~2 orders of magnitude higher than a small
        standalone traversal (measured ~1.7 s/edge vs ~7 ms/edge on a real
        cluster), so a real corpus's tens of thousands of edges effectively
        never finished. Each edge as its own traversal keeps the cost linear.

        Edge properties take no cardinality (Neptune rejects it) and hold one
        value each, so ``relationship_properties`` serializes lists to JSON.

        Both endpoint vertices are looked up before the drop, so an edge
        whose endpoint is gone keeps its stored copy and the result is empty
        (the only sign of an edge not written).
        """

        stored = (
            g.E()
            .has("id", rel.id)
            .where(__.outV().hasLabel(entity_label))
            .fold()
            .as_("stored")
        )
        # Both endpoints first, each looked up once: without them nothing is
        # dropped either.
        if self._custom_ids:
            ends = (
                stored.V(vertex_id(entity_label, rel.source_id))
                .as_("source")
                .V(vertex_id(entity_label, rel.target_id))
                .as_("target")
            )
        else:
            source = self._rewritten(
                stored.V().hasLabel(entity_label).has("id", rel.source_id),
                entity_label,
            ).as_("source")
            ends = self._rewritten(
                source.V().hasLabel(entity_label).has("id", rel.target_id),
                entity_label,
            ).as_("target")
        edge = (
            ends.sideEffect(__.select("stored").unfold().drop())
            .addE(relationship_label(rel))
            .from_("source")
            .to("target")
            .property("id", rel.id)
        )
        props = relationship_properties(rel, self.neptune_config.property_max_length)
        for key, value in props.items():
            edge.property(key, value)
        self._tagged(edge)
        return cast(GraphTraversal, edge.id_())

    def _write_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        """Write relationship edges one small traversal per edge (idempotent).

        For each suffix group: rewrite each edge as its own traversal
        (``_build_add_edge_traversal``: both endpoints looked up, the old copy
        dropped and the new one added in one request), ``index_concurrency``
        at a time. An edge whose endpoint is not stored, and a failed write,
        count as failed and leave the stored edge as it was; neither aborts
        the others. Emits a progress log every
        ``_PROGRESS_LOG_EVERY`` edges so a large run is distinguishable from
        a hang.
        """
        if not relationships:
            return IndexingStats()

        total_stats = IndexingStats()
        grouped_items = self._group_items_by_suffix(relationships)

        for suffix, rels in grouped_items.items():
            stats = IndexingStats(total_items=len(rels))
            start_time = time.time()
            entity_label = self._label(self.neptune_config.entity_label_prefix, suffix)

            logger.info(
                "Indexing %s relationships for '%s'...", len(rels), entity_label
            )

            # An edge whose endpoint vertex is missing is not written (the
            # traversal finds both before it drops anything): its stored copy
            # (the edge an unchanged document states, before a re-key
            # re-pointed it) stays.
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
                    return (
                        f"source entity '{rel.source_id}' or target entity "
                        f"'{rel.target_id}' not found"
                    )
                except NamespaceLeaseError:
                    raise
                except Exception as e:
                    return str(e)

            workers = max(1, min(self.neptune_config.index_concurrency, len(rels)))
            with ContextThreadPoolExecutor(max_workers=workers) as executor:
                for index, (rel, error) in enumerate(
                    zip(rels, executor.map(write, rels), strict=True),
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
        documents from OpenSearch.

        Raises:
            AWSServiceError: The read failed. An empty result would leave the
                orphaned relationship documents behind, or miss the stored
                edges a remapped delta edge must merge into.
        """
        if not entity_ids:
            return []
        label = self._label(self.neptune_config.entity_label_prefix, suffix)
        try:
            rows = self._read_batches(
                entity_ids,
                lambda g, batch: self._vertices(g, (label,), batch)
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

        The drop is scoped to the suffix's entity and community labels
        (``None``: the default suffix's, as for every other method). Entity
        ids are content-hash derived, so the SAME id can exist under another
        suffix (another tenant/version); an unscoped ``V().has('id', ...)``
        would delete that other tenant's vertex too.
        """
        stats = IndexingStats(total_items=len(ids))
        if not ids:
            return stats

        entity_label = self._label(self.neptune_config.entity_label_prefix, suffix)
        community_label = self._label(
            self.neptune_config.community_label_prefix, suffix
        )

        for id_batch in self._batch_iterator(ids):

            # Scope the drop to this suffix. Edges are labeled by their
            # relationship type (e.g. RELATED_TO), NOT by the entity
            # label, so filtering edges with hasLabel(entity_label) would match
            # nothing and leak stale edges. Instead scope edges by id AND by an
            # endpoint vertex carrying this suffix's entity label
            # (where(outV().hasLabel(...))), which both restricts to this
            # tenant and matches real edges. A drop is safe to retry.
            def drop_edges(
                g: GraphTraversalSource, id_batch: list[str] = id_batch
            ) -> GraphTraversal:
                return (
                    g.E()
                    .has("id", P.within(id_batch))
                    .where(__.outV().hasLabel(entity_label))
                    .drop()
                )

            def drop_vertices(
                g: GraphTraversalSource, id_batch: list[str] = id_batch
            ) -> GraphTraversal:
                return self._vertices(
                    g, (entity_label, community_label), id_batch
                ).drop()

            try:
                self._execute_with_retries(drop_edges, "Deleting edges by id")
                self._execute_with_retries(drop_vertices, "Deleting vertices by id")
                stats.add_success(len(id_batch))
            except NamespaceLeaseError:
                raise
            except Exception as e:
                stats.add_error(str(e), len(id_batch), ids=list(id_batch))
                logger.warning("Failed to delete ids batch: %s", e)

        return stats

    def _tagged(self, traversal: GraphTraversal) -> GraphTraversal:
        """``traversal`` (adding an edge) with the rebuild's tag, if any."""
        if self._rebuild_tag is not None:
            traversal.property(BUILD_TAG_PROPERTY, self._rebuild_tag)
        return traversal

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
        if self._rebuild_tag is not None:
            props = {**props, BUILD_TAG_PROPERTY: self._rebuild_tag}
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
            if clear_first and self._rebuild_tag is None:
                self._clear_existing_data_by_label(label)
            else:
                self._check_vertex_ids(label)
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
        """Drop every vertex of ``label`` (see :meth:`_drop_in_batches`), and
        a vertex-id switch it was left in: empty, it takes either mode."""
        self._drop_in_batches(
            lambda g: g.V().hasLabel(label),
            f"vertices with label '{label}'",
            batch_size,
        )
        self._drop_vertex_id_switch(label)
        with self._labels_lock:
            self._switching_labels.discard(label)
            self._checked_labels.add(label)

    def _drop_in_batches(
        self,
        select: Callable[[GraphTraversalSource], GraphTraversal],
        what: str,
        batch_size: int = _DELETE_BATCH_SIZE,
    ) -> int:
        """Drop the elements ``select`` yields, ``batch_size`` per request.

        Bounded requests keep each drop's transaction small, and each request
        is retried like a write (a drop is safe to repeat). A request returns
        how many elements it dropped, so a short one is the last: no count per
        pass. The elements are counted once up front to bound the passes: if
        requests keep reporting full batches past that bound (a backend whose
        count does not reflect the drop, or writers adding elements meanwhile)
        the drop fails instead of looping forever. Returns how many it
        dropped.

        Raises:
            AWSServiceError: A request failed after its retries, or the
                selection still reported full batches after
                ``ceil(count / batch_size) + _EXTRA_DELETE_PASSES`` passes.
        """
        try:
            (initial,) = self._execute_with_retries(
                lambda g: select(g).count(),
                f"Counting {what} to drop",
                results=True,
            )
            max_passes = math.ceil(int(initial) / batch_size) + _EXTRA_DELETE_PASSES
            total = 0
            for _ in range(max_passes):
                (dropped,) = self._execute_with_retries(
                    lambda g: select(g).limit(batch_size).sideEffect(__.drop()).count(),
                    f"Dropping {what}",
                    results=True,
                )
                total += int(dropped)
                if dropped < batch_size:
                    break
            else:
                raise AWSServiceError(
                    f"Dropping {what} did not finish: {max_passes} passes of "
                    f"{batch_size} each dropped a full batch ({total} reported "
                    f"dropped, {initial} counted before)"
                )
        except Exception as e:
            logger.error("Failed to drop %s: %s", what, e)
            if isinstance(e, GraphRAGException):
                raise
            raise AWSServiceError(f"Dropping Neptune {what} failed: {e}") from e
        if total:
            logger.info("Dropped %s %s", total, what)
        return total

    # --- rebuilding a suffix in place ------------------------------------------

    def begin_rebuild(self, suffixes: Collection[str]) -> bool:
        """Start a full build that replaces the suffixes' graph in place.

        Until :meth:`end_rebuild`, the full-build writes no longer clear a
        label first: every vertex and edge they write carries this build's
        tag (``build_tag``), so the graph is never empty while the new one is
        written (queries read the rewritten items next to the previous ones
        not reached yet), and :meth:`end_rebuild` then drops what the build
        did not write.
        """
        self._rebuild_tag = uuid.uuid4().hex
        return True

    def end_rebuild(self, suffixes: Collection[str], sweep: bool = True) -> None:
        """Drop the suffixes' vertices and edges the build did not write
        (``sweep``; skipped when a write failed, so a retry still finds the
        previous copies), and end the rebuild.

        A label the build switched to this run's vertex ids
        (:meth:`_check_vertex_ids`) holds only them once swept: its switch
        vertex is dropped. Unswept, it holds both modes and stays refused
        until a full build finishes it.
        """
        tag, self._rebuild_tag = self._rebuild_tag, None
        switching, self._switching_labels = self._switching_labels, set()
        swept: set[str] = set()
        try:
            if not sweep or tag is None:
                return
            for suffix in sorted(suffixes):
                entity_label = self._get_label(
                    self.neptune_config.entity_label_prefix, suffix
                )
                # Edges first: they leave the entities.
                self._drop_in_batches(
                    _stale_edges(entity_label, tag),
                    f"stale edges of '{entity_label}'",
                )
                for prefix in (
                    self.neptune_config.entity_label_prefix,
                    self.neptune_config.community_label_prefix,
                ):
                    label = self._get_label(prefix, suffix)
                    self._drop_in_batches(
                        _stale_vertices(label, tag), f"stale vertices of '{label}'"
                    )
                    swept.add(label)
            for label in sorted(switching & swept):
                self._drop_vertex_id_switch(label)
                switching.discard(label)
        finally:
            # Checked again on their next use, and refused.
            with self._labels_lock:
                self._checked_labels -= switching

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
        (concurrent) without any locking. A run that lost its write lease
        raises before the request (``check_write_fence``).
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
        except NamespaceLeaseError:
            raise
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
                except NamespaceLeaseError:
                    raise
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
        fenced: bool = True,
    ) -> list[Any]:
        """Run the traversal ``build`` makes, with retries (``run_traversal``).

        Each attempt builds it anew from the client's current source, so after
        a lost connection (a Neptune restart or failover) the retry goes over
        a new one. With ``results`` its results are returned, else ``[]``.
        The final failure is logged at ``exhausted_level``. Unless it only
        reads (``fenced=False``), each attempt checks the run's write fence
        first, so a run that lost its lease stops between attempts.
        """
        max_attempts = self.config.retry.graph_store_max_attempts
        delay = self.config.retry.base_delay_seconds
        attempt = 0
        while True:
            if fenced:
                check_write_fence()
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

            rows.extend(
                self._execute_with_retries(
                    build, operation_name, results=True, fenced=False
                )
            )
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
