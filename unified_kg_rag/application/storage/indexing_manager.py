# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections.abc import Callable, Sequence
from concurrent.futures import as_completed
from functools import partial
from typing import Any, NamedTuple, TypeVar

from pydantic import BaseModel, Field

from unified_kg_rag.adapters.ingestion.description_summarizer import (
    DescriptionSummarizer,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.ingestion.base_resolver import FuzzyMatcher
from unified_kg_rag.domain.ingestion.graph_rank import with_graph_ranks
from unified_kg_rag.domain.ingestion.merge import (
    merge_claims,
    merge_entities,
    merge_relationships,
    relationship_id_remap,
    remove_claim_text_units,
    remove_text_units,
)
from unified_kg_rag.domain.ingestion.relationship_endpoints import (
    sync_relationship_endpoint_names,
)
from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Config,
    Constants,
    Entity,
    Relationship,
    TextUnit,
)
from unified_kg_rag.ports.indexer import (
    BaseIndexer,
    GraphIndexer,
    IndexingStats,
    VectorIndexer,
)
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor

logger = get_logger(__name__)

# A graph item whose description the cross-run merge reads back.
_Described = TypeVar("_Described", Entity, Relationship)


class CrossRunMergeResult(BaseModel):
    """A delta merged with the stored graph state it touches."""

    entities: list[Entity] | None = None
    relationships: list[Relationship] | None = None
    # Per index suffix: delta entity/relationship id -> the stored id it merged
    # into, so lineage can record the ids actually written.
    id_remap_by_suffix: dict[str, dict[str, str]] = Field(default_factory=dict)


class IndexingTask(NamedTuple):
    fn: Callable
    args: list
    key: str
    # Run even without items (a full build replacing a type with nothing).
    run_empty: bool = False


class IndexingManager:
    """Coordinates writes across the vector and graph stores.

    The two backends are injected as :class:`VectorIndexer` / :class:`GraphIndexer`
    ports; otherwise the provider bundle builds the ones ``config.providers``
    selects (OpenSearch/Neptune by default). This keeps the application layer
    off any concrete backend import and lets tests drive the *real* manager
    with the in-memory fakes. The per-backend fan-out (entities to both stores;
    entities-before-edges phasing; the orphan-edge cascade across stores) is
    deliberate domain knowledge, not arbitrary dispatch, so it stays explicit
    here rather than behind a uniform registry loop.
    """

    def __init__(
        self,
        config: Config,
        *,
        vector_indexer: VectorIndexer | None = None,
        graph_indexer: GraphIndexer | None = None,
        providers: Providers | None = None,
    ) -> None:
        self.config = config
        # Session, embedding/LLM providers and store builders shared with the
        # pipeline, used for the default indexers and the merge summarizer.
        self.providers = Providers.resolve(config, providers)
        # The attribute names predate the store-agnostic ports; they hold the
        # vector and graph writers whatever their backend.
        self.opensearch_indexer: VectorIndexer = (
            vector_indexer or self.providers.build_vector_indexer()
        )
        self.neptune_indexer: GraphIndexer = (
            graph_indexer or self.providers.build_graph_indexer()
        )
        # Built lazily on first cross-run merge so a manager only used for
        # indexing (no incremental read-back) never constructs a Bedrock client.
        self._description_summarizer: DescriptionSummarizer | None = None

    @property
    def description_summarizer(self) -> DescriptionSummarizer:
        if self._description_summarizer is None:
            self._description_summarizer = DescriptionSummarizer(
                self.config, providers=self.providers
            )
        return self._description_summarizer

    def close(self) -> None:
        """Close both indexers' backing clients (best-effort, never raises)."""
        for indexer in (self.opensearch_indexer, self.neptune_indexer):
            try:
                indexer.close()
            except Exception as e:  # noqa: BLE001 - teardown must never raise
                logger.debug("Error closing %s: %s", type(indexer).__name__, e)

    def clear_all_data(self, text_units: list[TextUnit]) -> bool:
        suffixes = self._discover_suffixes(text_units)
        if not suffixes:
            logger.warning("No suffixes found to clear data")
            return True

        logger.info("Clearing data for suffixes: '%s'", suffixes)
        try:
            with ContextThreadPoolExecutor(max_workers=2) as executor:
                os_future = executor.submit(self.opensearch_indexer.clear, suffixes)
                neptune_future = executor.submit(self.neptune_indexer.clear, suffixes)
                opensearch_success = os_future.result()
                neptune_success = neptune_future.result()

            success = opensearch_success and neptune_success
            if not success:
                logger.error(
                    "Clear operation failed - OpenSearch: %s, Neptune: %s",
                    opensearch_success,
                    neptune_success,
                )
            return success
        except Exception as e:
            logger.error("Clear operation failed: %s", e)
            return False

    @staticmethod
    def _group_by_suffix(items: list[Any]) -> dict[str, list[Any]]:
        grouped: dict[str, list[Any]] = {}
        for item in items:
            grouped.setdefault(BaseIndexer.get_suffix(item), []).append(item)
        return grouped

    @staticmethod
    def _discover_suffixes(items: list[Any] | None) -> list[str]:
        if not items:
            return []
        return list({BaseIndexer.get_suffix(item) for item in items})

    def initialize(self) -> bool:
        try:
            opensearch_ok = self.opensearch_indexer.initialize()
            neptune_ok = self.neptune_indexer.initialize()

            if not opensearch_ok or not neptune_ok:
                logger.error("Failed to initialize indexers")
                return False

            return True
        except Exception as e:
            logger.error("Indexer initialization failed: %s", e)
            return False

    def index_all_data(
        self,
        text_units: list[TextUnit] | None = None,
        entities: list[Entity] | None = None,
        relationships: list[Relationship] | None = None,
        communities: list[Community] | None = None,
        community_reports: list[CommunityReport] | None = None,
        claims: list[Claim] | None = None,
    ) -> dict[str, IndexingStats]:
        """Replace the stored artifacts of the run's suffixes (full build).

        Every index type is rebuilt for every suffix the run's artifacts carry,
        including a type the run has none of (no claims, no reports): its index
        is emptied rather than left serving a previous run's items.
        """
        start_time = time.time()
        results: dict[str, IndexingStats] = {}

        self._enrich_text_units(text_units, communities)
        suffixes = self._discover_suffixes(
            [
                *(text_units or []),
                *(entities or []),
                *(relationships or []),
                *(communities or []),
                *(community_reports or []),
                *(claims or []),
            ]
        )

        def replace(fn: Callable, items: list[Any] | None, key: str) -> IndexingTask:
            return IndexingTask(
                partial(fn, suffixes=suffixes), [items or []], key, run_empty=True
            )

        phase1_tasks = [
            replace(
                self.opensearch_indexer.index_text_units,
                text_units,
                "opensearch_text_units",
            ),
            replace(
                self.opensearch_indexer.index_entities,
                entities,
                "opensearch_entities",
            ),
            replace(
                self.opensearch_indexer.index_community_reports,
                community_reports,
                "opensearch_community_reports",
            ),
            replace(
                self.opensearch_indexer.index_claims,
                claims,
                "opensearch_claims",
            ),
            replace(self.neptune_indexer.index_entities, entities, "neptune_entities"),
        ]
        # The OpenSearch relationship VECTOR index is only consumed by LightRAG
        # high-level keyword retrieval (mix/hybrid). Skip it for a GraphRAG-only
        # deployment (symmetric with community_detection.enabled). The Neptune
        # relationship EDGES below are always built — both methodologies traverse
        # them for graph expansion.
        if self.config.indexing.opensearch.build_relationship_vector_index:
            phase1_tasks.append(
                replace(
                    self.opensearch_indexer.index_relationships,
                    relationships,
                    "opensearch_relationships",
                )
            )

        logger.info("--- Starting Indexing Phase 1 ---")
        results.update(self._run_indexing_phase(phase1_tasks))

        phase2_tasks = [
            # Replacing the entity vertices already dropped the old edges.
            IndexingTask(
                self.neptune_indexer.index_relationships,
                [relationships],
                "neptune_relationships",
            ),
            replace(
                self.neptune_indexer.index_communities,
                communities,
                "neptune_communities",
            ),
        ]

        logger.info("--- Starting Indexing Phase 2 ---")
        results.update(self._run_indexing_phase(phase2_tasks))

        elapsed_time = time.time() - start_time
        self._log_completion_summary(results, elapsed_time)
        return results

    def index_delta(
        self,
        text_units: list[TextUnit] | None = None,
        entities: list[Entity] | None = None,
        relationships: list[Relationship] | None = None,
        communities: list[Community] | None = None,
        community_reports: list[CommunityReport] | None = None,
        claims: list[Claim] | None = None,
    ) -> dict[str, IndexingStats]:
        """Idempotently upsert a delta set into the live stores (incremental run).

        Routes to the indexers' ``upsert_*`` methods instead of the full
        rebuild path, so only changed/new artifacts are written and existing
        data is preserved. Entities (graph + vector) are upserted before
        relationships, which depend on entity vertices existing. Writes the
        given artifacts as they are: the cross-run merge with stored state is
        :meth:`merge_with_existing_graph`, run by the incremental commit.
        """
        start_time = time.time()
        results: dict[str, IndexingStats] = {}

        self._enrich_text_units(text_units, communities)

        phase1_tasks = [
            IndexingTask(
                self.opensearch_indexer.upsert_text_units,
                [text_units],
                "opensearch_text_units",
            ),
            IndexingTask(
                self.opensearch_indexer.upsert_entities,
                [entities],
                "opensearch_entities",
            ),
            IndexingTask(
                # Delta path MUST upsert, not index: index_community_reports does
                # a full alias-swap that would wipe unchanged docs' reports (the
                # delta only carries reports for the changed subgraph).
                self.opensearch_indexer.upsert_community_reports,
                [community_reports],
                "opensearch_community_reports",
            ),
            IndexingTask(
                self.opensearch_indexer.upsert_claims,
                [claims],
                "opensearch_claims",
            ),
            IndexingTask(
                self.neptune_indexer.upsert_entities, [entities], "neptune_entities"
            ),
        ]
        # LightRAG-only vector index — gate symmetrically with the full build.
        # Neptune relationship edges (phase 2) are always upserted.
        if self.config.indexing.opensearch.build_relationship_vector_index:
            phase1_tasks.append(
                IndexingTask(
                    self.opensearch_indexer.upsert_relationships,
                    [relationships],
                    "opensearch_relationships",
                )
            )
        logger.info("--- Starting Delta Indexing Phase 1 (upsert) ---")
        results.update(self._run_indexing_phase(phase1_tasks))

        phase2_tasks = [
            IndexingTask(
                self.neptune_indexer.upsert_relationships,
                [relationships],
                "neptune_relationships",
            ),
            IndexingTask(
                self.neptune_indexer.upsert_communities,
                [communities],
                "neptune_communities",
            ),
        ]
        logger.info("--- Starting Delta Indexing Phase 2 (upsert) ---")
        results.update(self._run_indexing_phase(phase2_tasks))

        elapsed_time = time.time() - start_time
        self._log_completion_summary(results, elapsed_time)
        return results

    def merge_with_existing_graph(
        self,
        entities: list[Entity] | None,
        relationships: list[Relationship] | None,
    ) -> CrossRunMergeResult:
        """Union delta artifacts with existing graph state before upsert.

        Reads the existing entities/relationships the delta touches back from the
        graph store and merges (description/text_unit_ids union, frequency/weight
        recompute) via the pure merge functions, so a cross-run upsert accumulates
        rather than overwriting. If the adapter does not support read-back it
        returns ``[]`` and this degenerates to the existing overwrite behaviour.
        A failed read raises: overwriting would drop the stored lineage, and
        raising before any write leaves the docs unrecorded for a retry.
        With ``indexing.cross_run_merge`` off the delta is returned unchanged.

        Merges one index suffix at a time: entity and relationship ids are
        suffix-independent, so each tenant's delta must meet that tenant's stored
        state only (and its entity-id remap must not leak into another tenant).
        """
        if not self.config.indexing.cross_run_merge:
            return CrossRunMergeResult(entities=entities, relationships=relationships)
        entity_groups = self._group_by_suffix(entities or [])
        relationship_groups = self._group_by_suffix(relationships or [])
        merged_entities: list[Entity] = []
        merged_relationships: list[Relationship] = []
        id_remap_by_suffix: dict[str, dict[str, str]] = {}
        for suffix in sorted(entity_groups.keys() | relationship_groups.keys()):
            suffix_entities, suffix_relationships, id_remap = self._merge_suffix(
                suffix,
                entity_groups.get(suffix, []),
                relationship_groups.get(suffix, []),
            )
            merged_entities.extend(suffix_entities)
            merged_relationships.extend(suffix_relationships)
            if id_remap:
                id_remap_by_suffix[suffix] = id_remap
        return CrossRunMergeResult(
            entities=merged_entities if entities is not None else None,
            relationships=(merged_relationships if relationships is not None else None),
            id_remap_by_suffix=id_remap_by_suffix,
        )

    def merge_with_stored_claims(
        self, claims: list[Claim] | None
    ) -> list[Claim] | None:
        """Fold the stored state of each delta claim into it before upsert.

        A claim's id does not depend on its document, so the claim another
        document already stored has the delta claim's id: without the merge
        the upsert replaces its descriptions and text units with this run's
        (see ``merge_claims``). Read per index suffix from the vector store; a
        failed read raises (see :meth:`merge_with_existing_graph`). With
        ``indexing.cross_run_merge`` off the claims are returned unchanged.
        """
        if not claims or not self.config.indexing.cross_run_merge:
            return claims
        merged: list[Claim] = []
        for suffix, group in sorted(self._group_by_suffix(claims).items()):
            stored = self.opensearch_indexer.read_claims(
                sorted({c.id for c in group}), suffix=suffix
            )
            merged.extend(merge_claims(stored, group) if stored else group)
        return merged

    def _merge_suffix(
        self,
        suffix: str,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> tuple[list[Entity], list[Relationship], dict[str, str]]:
        """Cross-run merge of one suffix's delta (see merge_with_existing_graph).

        Returns the merged entities and relationships plus ``{delta id: kept
        id}`` for every delta entity or relationship that merged into a stored
        one under a different id.
        """
        merged_entities = entities
        entity_id_remap: dict[str, str] = {}
        if entities:
            # The full build also merges names that differ only by legal form
            # or spacing (and by similarity with fuzzy_entity_resolution);
            # their stored entities have other ids than the delta's, so they
            # are found through the stored names and read back as well.
            name_matcher, matched_ids = self._match_stored_names(suffix, entities)
            existing = self._with_full_descriptions(
                self.neptune_indexer.read_entities(
                    sorted({e.id for e in entities} | matched_ids), suffix=suffix
                ),
                self.opensearch_indexer.read_entity_descriptions,
                suffix,
            )
            if existing:
                merged_entities, entity_id_remap = merge_entities(
                    existing, entities, fuzzy_matcher=name_matcher
                )
                # The cross-run merge concatenates descriptions, so an entity
                # seen across many runs can grow unbounded — re-summarize the
                # over-threshold ones (no-op below the threshold / when disabled).
                merged_entities = self.description_summarizer.summarize_entities(
                    merged_entities
                )
        merged_relationships = relationships
        relationship_remap: dict[str, str] = {}
        if relationships:
            rel_ids = {r.id for r in relationships}
            if entity_id_remap:
                # A delta edge whose endpoint merged into a stored entity can only
                # match a stored edge of that entity: its id derives from the
                # stored names, not the delta's, so read the incident edges too.
                rel_ids |= set(
                    self.neptune_indexer.find_incident_relationship_ids(
                        sorted(set(entity_id_remap.values())), suffix=suffix
                    )
                )
            existing_rels = self._with_full_descriptions(
                self.neptune_indexer.read_relationships(sorted(rel_ids), suffix=suffix),
                self.opensearch_indexer.read_relationship_descriptions,
                suffix,
            )
            # Even when no existing relationships are read back, delta endpoints
            # must follow any entity-id collapse from the entity merge above —
            # otherwise a remapped entity leaves its relationships pointing at a
            # now-nonexistent id. Pass the remap through unconditionally.
            if existing_rels or entity_id_remap:
                merged_relationships = merge_relationships(
                    existing_rels,
                    relationships,
                    entity_id_remap=entity_id_remap or None,
                )
                relationship_remap = relationship_id_remap(
                    relationships, merged_relationships, entity_id_remap
                )
                # The incident edges were read only to find matches: write
                # back just the edges a delta edge became or merged into.
                written = {relationship_remap.get(r.id, r.id) for r in relationships}
                merged_relationships = (
                    self.description_summarizer.summarize_relationships(
                        [r for r in merged_relationships if r.id in written]
                    )
                )
        if merged_entities is not entities and merged_relationships:
            # Merged with stored entities: an edge re-pointed at one, or at one
            # whose stored name differs from the delta's, must name it as the
            # entity does.
            merged_relationships = [r.model_copy() for r in merged_relationships]
            sync_relationship_endpoint_names(merged_relationships, merged_entities)
        merged_entities, merged_relationships = self._rank_with_stored_graph(
            suffix, merged_entities, merged_relationships
        )
        return (
            merged_entities,
            merged_relationships,
            {**entity_id_remap, **relationship_remap},
        )

    def _with_full_descriptions(
        self,
        items: list[_Described],
        read: Callable[..., dict[str, str]],
        suffix: str,
    ) -> list[_Described]:
        """``items`` read back from the graph, with each description the graph
        capped at ``indexing.neptune.property_max_length`` replaced by the
        vector store's full one (the merged description is written there)."""
        limit = self.config.indexing.neptune.property_max_length
        capped = [i.id for i in items if len(i.description or "") >= limit]
        if not capped:
            return items
        full = read(capped, suffix=suffix)
        return [
            i.model_copy(update={"description": full[i.id]}) if i.id in full else i
            for i in items
        ]

    def _rank_with_stored_graph(
        self,
        suffix: str,
        entities: list[Entity],
        relationships: list[Relationship],
    ) -> tuple[list[Entity], list[Relationship]]:
        """Recompute the ranks (degrees) of the items the delta writes over the
        stored graph plus the delta's relationships.

        A stored entity the delta does not write keeps its rank until a full
        rebuild, also when a removed document dropped one of its edges.
        """
        ids = sorted(
            {e.id for e in entities}
            | {i for r in relationships for i in (r.source_id, r.target_id)}
        )
        stored_neighbors = self.neptune_indexer.read_neighbor_ids(ids, suffix=suffix)
        return with_graph_ranks(entities, relationships, stored_neighbors)

    def _match_stored_names(
        self, suffix: str, entities: list[Entity]
    ) -> tuple[FuzzyMatcher | None, set[str]]:
        """Match delta entity names against the stored names of ``suffix``.

        Returns a matcher over the stored names (built with the full build's
        settings) and the ids of the stored entities some delta name matches.
        Returns ``(None, set())`` when the adapter projects no names, which
        leaves the merge to ids and exact keys.
        """
        id_by_name: dict[str, str] = {}
        for eid, name in self.neptune_indexer.read_entity_names(suffix):
            id_by_name.setdefault(name, eid)
        if not id_by_name:
            return None, set()

        processing = self.config.processing
        matcher = FuzzyMatcher(
            candidates=list(id_by_name.keys()),
            resolution_method=processing.resolution_method,
            similarity_threshold=processing.similarity_threshold,
            fuzzy=processing.fuzzy_entity_resolution,
        )
        old_ids = {
            id_by_name[name]
            for entity in entities
            for name, _score in matcher.find_all_matches(entity.name)
            if name in id_by_name
        }
        return matcher, old_ids

    def remove_text_units_from_shared(
        self,
        suffix: str,
        *,
        entity_ids: list[str],
        relationship_ids: list[str],
        text_unit_ids: list[str],
        claim_ids: Sequence[str] = (),
    ) -> dict[str, IndexingStats]:
        """Strip removed text units from kept entities/relationships/claims of a
        suffix.

        Reads the stored items back, removes ``text_unit_ids`` from their
        lineage (recomputing frequency/weight, see ``remove_text_units``) and
        writes every item read back to both stores (the vector store updates
        only those fields in place), so a retry also repairs a store an earlier,
        interrupted strip did not reach. A graph adapter without read-back
        leaves them unchanged.

        Claims live in the vector store only: they are read back from it and
        only those still citing a removed text unit are updated.

        A failed read is reported as failed items, so the caller keeps the
        documents' registry rows and retries the removal.
        """
        results: dict[str, IndexingStats] = {}
        if claim_ids:
            results.update(
                self._remove_text_units_from_claims(
                    suffix, list(claim_ids), text_unit_ids
                )
            )
        try:
            entities = (
                self.neptune_indexer.read_entities(entity_ids, suffix=suffix)
                if entity_ids
                else []
            )
            relationships = (
                self.neptune_indexer.read_relationships(relationship_ids, suffix=suffix)
                if relationship_ids
                else []
            )
        except Exception as e:  # noqa: BLE001 - reported as failed items
            logger.error("Reading shared artifacts in '%s' failed: %s", suffix, e)
            stats = IndexingStats(total_items=len(entity_ids) + len(relationship_ids))
            stats.add_error(str(e), count=stats.total_items)
            return {**results, f"neptune_read_{suffix}": stats}
        stripped_entities, stripped_relationships = remove_text_units(
            entities, relationships, text_unit_ids
        )
        # Every item read back is rewritten, stripped or not: the graph copy
        # is the reference, and after a strip that was interrupted or partly
        # failed between the stores it no longer cites the removed text units
        # while the vector copy still does. Rewriting only the items still
        # citing them would leave that copy stale on every retry.
        stripped = {item.id: item for item in stripped_entities}
        entities = [stripped.get(e.id, e) for e in entities]
        stripped_rels = {item.id: item for item in stripped_relationships}
        relationships = [stripped_rels.get(r.id, r) for r in relationships]
        if not entities and not relationships:
            return results
        logger.info(
            "Removing %d text units from %d shared entities and %d shared "
            "relationships in '%s'",
            len(text_unit_ids),
            len(entities),
            len(relationships),
            suffix,
        )
        # Only lineage and weight changed: the vector store updates them in
        # place instead of re-embedding unchanged text.
        phase1 = [
            IndexingTask(
                self.neptune_indexer.upsert_entities, [entities], "neptune_entities"
            ),
            IndexingTask(
                self.opensearch_indexer.update_entity_lineage,
                [entities],
                "opensearch_entities",
            ),
        ]
        if self.config.indexing.opensearch.build_relationship_vector_index:
            phase1.append(
                IndexingTask(
                    self.opensearch_indexer.update_relationship_lineage,
                    [relationships],
                    "opensearch_relationships",
                )
            )
        results.update(self._run_indexing_phase(phase1))
        results.update(
            self._run_indexing_phase(
                [
                    IndexingTask(
                        self.neptune_indexer.upsert_relationships,
                        [relationships],
                        "neptune_relationships",
                    )
                ]
            )
        )
        return results

    def _remove_text_units_from_claims(
        self, suffix: str, claim_ids: list[str], text_unit_ids: list[str]
    ) -> dict[str, IndexingStats]:
        try:
            claims = self.opensearch_indexer.read_claims(claim_ids, suffix=suffix)
        except Exception as e:  # noqa: BLE001 - reported as failed items
            logger.error("Reading shared claims in '%s' failed: %s", suffix, e)
            stats = IndexingStats(total_items=len(claim_ids))
            stats.add_error(str(e), count=len(claim_ids), ids=claim_ids)
            return {f"opensearch_read_claims_{suffix}": stats}
        # The write-back groups by the claim's own index tag, so tag what was
        # read with the index it came from.
        for claim in claims:
            claim.attributes = {
                **(claim.attributes or {}),
                Constants.INDEX.value: suffix,
            }
        stripped = remove_claim_text_units(claims, text_unit_ids)
        if not stripped:
            return {}
        return {
            f"opensearch_claims_{suffix}": (
                self.opensearch_indexer.update_claim_lineage(stripped)
            )
        }

    def delete_documents(
        self, ids_by_suffix: dict[str, list[str]]
    ) -> dict[str, IndexingStats]:
        """Delete artifacts for removed documents from both stores by id.

        ``ids_by_suffix`` maps a tenant/version suffix to the entity/relationship/
        text-unit/community ids that belonged only to deleted documents.
        """
        results: dict[str, IndexingStats] = {}
        for suffix, ids in ids_by_suffix.items():
            if not ids:
                continue
            # Orphan-edge cleanup: a relationship pointing AT one of these
            # deleted entities is owned (in lineage) by a SURVIVING document, so
            # it is not in `ids`. Neptune drops the incident edge on vertex drop,
            # but its OpenSearch relationship document would survive as a
            # dangling edge. Collect those incident edge ids BEFORE the Neptune
            # drop, then fold them into the relationship-index deletion below.
            try:
                incident_rel_ids = self.neptune_indexer.find_incident_relationship_ids(
                    ids, suffix=suffix
                )
            except Exception as e:  # noqa: BLE001 - reported as failed items
                # Deleting without them would orphan their vector documents
                # for good; fail the removal so the registry rows are kept.
                logger.error("Reading incident edges in '%s' failed: %s", suffix, e)
                stats = IndexingStats(total_items=len(ids))
                stats.add_error(str(e), count=len(ids))
                results[f"neptune_read_{suffix}"] = stats
                continue
            results[f"neptune_delete_{suffix}"] = self.neptune_indexer.delete_by_id(
                ids, suffix=suffix
            )
            results.update(
                self.opensearch_indexer.delete_document_artifacts(
                    ids, suffix, extra_relationship_ids=incident_rel_ids
                )
            )
        return results

    def _run_indexing_phase(
        self, tasks: list[IndexingTask]
    ) -> dict[str, IndexingStats]:
        phase_results: dict[str, IndexingStats] = {}
        valid_tasks = [task for task in tasks if task.run_empty or task.args[0]]

        if not valid_tasks:
            logger.info("No tasks to run in this phase.")
            return phase_results

        total_items = sum(len(task.args[0] or []) for task in valid_tasks)
        logger.info(
            "Executing %s tasks with %s total items...", len(valid_tasks), total_items
        )

        # Indexing tasks are IO-bound (OpenSearch/Neptune network writes), so size
        # the pool by the number of independent tasks rather than CPU count — a
        # cpu*0.8 cap (~1-2 on a 2-vCPU Fargate task) would needlessly serialize
        # the independent per-backend writes. Cap at 8 as a safety bound.
        pool_size = min(len(valid_tasks), 8)
        with ContextThreadPoolExecutor(max_workers=pool_size) as executor:
            futures_map = {
                executor.submit(task.fn, *task.args): task.key for task in valid_tasks
            }

            for future in as_completed(futures_map):
                task_name = futures_map[future]
                try:
                    phase_results[task_name] = future.result()
                except Exception as e:
                    logger.error("Task '%s' failed: %s", task_name, e)
                    stats = IndexingStats()
                    stats.add_error(str(e))
                    phase_results[task_name] = stats

        return phase_results

    @staticmethod
    def _enrich_text_units(
        text_units: list[TextUnit] | None, communities: list[Community] | None
    ) -> None:
        if not text_units or not communities:
            return

        text_unit_map = {tu.id: tu for tu in text_units}
        for community in communities:
            if not community.text_unit_ids:
                continue
            for text_unit_id in community.text_unit_ids:
                if text_unit := text_unit_map.get(text_unit_id):
                    if text_unit.community_ids is None:
                        text_unit.community_ids = []
                    if community.id not in text_unit.community_ids:
                        text_unit.community_ids.append(community.id)

    @staticmethod
    def _log_completion_summary(
        results: dict[str, IndexingStats], elapsed_time: float
    ) -> None:
        total_items = total_successful = total_failed = 0

        for stats in results.values():
            if stats:
                total_items += stats.total_items
                total_successful += stats.successful_items
                total_failed += stats.failed_items

        success_rate = (total_successful / total_items * 100) if total_items > 0 else 0
        logger.info(
            "Indexing completed in %.2fs: %s/%s items (%.1f%%)",
            elapsed_time,
            total_successful,
            total_items,
            success_rate,
        )

        if total_failed > 0:
            logger.warning("Failed items: %s", total_failed)
            for data_type, stats in results.items():
                if stats and stats.errors:
                    logger.warning("%s errors: %s", data_type, stats.errors[:2])

        for task_name, stats in results.items():
            if stats and stats.total_items > 0:
                failure_rate = stats.failed_items / stats.total_items
                if failure_rate > 0.5:
                    logger.warning(
                        "High failure rate for '%s': %s/%s (%.1f%%)",
                        task_name,
                        stats.failed_items,
                        stats.total_items,
                        failure_rate * 100,
                    )
