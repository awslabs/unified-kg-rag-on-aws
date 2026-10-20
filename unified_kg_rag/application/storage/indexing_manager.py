# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import as_completed
from functools import partial
from typing import Any, NamedTuple, TypeVar

from pydantic import BaseModel, Field

from unified_kg_rag.adapters.ingestion.description_summarizer import (
    DescriptionSummarizer,
    extend_summaries,
    needs_summary,
    reuse_summaries,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.ingestion.base_resolver import (
    FuzzyMatcher,
    compact_name_key,
)
from unified_kg_rag.domain.ingestion.community_hierarchy import attach_report_hierarchy
from unified_kg_rag.domain.ingestion.contributions import (
    RECORD_KEYS_ATTRIBUTE,
    attribute_record_keys,
    claims_from_payload,
    record_from_payload,
    record_keys,
)
from unified_kg_rag.domain.ingestion.derivation import Derived, derive
from unified_kg_rag.domain.ingestion.graph_rank import with_graph_ranks
from unified_kg_rag.domain.ingestion.merge import (
    merge_claims,
    merge_entities,
    merge_relationships,
    relationship_id_remap,
    relationship_key,
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
    TextUnitExtraction,
)
from unified_kg_rag.ports.indexer import (
    BaseIndexer,
    GraphIndexer,
    IndexingStats,
    VectorIndexer,
)
from unified_kg_rag.shared import DataProcessingError, NamespaceLeaseError, get_logger
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor
from unified_kg_rag.shared.write_fence import check_write_fence

logger = get_logger(__name__)

# A graph item whose description the cross-run merge reads back.
_Described = TypeVar("_Described", Entity, Relationship)

# Extraction-store key of a suffix's strip note: the stored items a
# removal's strip merges or re-keys, kept until their old ids are deleted
# (record keys are ``fingerprint:hash``: no clash).
_PENDING_STRIP = "pending-strip"


def _noted(notes: Mapping[str, Mapping[str, Any]], field: str) -> set[str]:
    """The ids the strip notes ``notes`` list under ``field``."""
    return {
        i
        for note in notes.values()
        for i in note.get(field) or []
        if isinstance(i, str)
    }


def _other_spellings_of_parts(
    split_from: Mapping[str, str], derived: Mapping[str, Entity]
) -> list[str]:
    """The ids of the ``derived`` entities (by id) sharing a
    ``compact_name_key`` with a part of ``split_from``, the parts excepted:
    a stored claim naming a part's spelling may be on one of them."""
    keys = {compact_name_key(derived[p].name) for p in split_from if p in derived}
    return sorted(
        i
        for i, e in derived.items()
        if i not in split_from and compact_name_key(e.name) in keys
    )


def _vector_content(item: Entity | Relationship) -> tuple[Any, ...]:
    """What the vector store holds of an item besides its description,
    lineage and rank."""
    if isinstance(item, Entity):
        return item.name, item.type, item.confidence
    return item.source_name, item.target_name, item.type


def _claim_text(claim: Claim) -> tuple[Any, ...]:
    """What a claim states beyond its lineage: its text and the names of the
    entities it is about (a renamed entity renames them)."""
    return (
        claim.type,
        claim.status,
        claim.description,
        claim.source_text,
        claim.subject_name,
        claim.object_name,
    )


def _other_spellings(
    derived: Sequence[Entity], given: Sequence[Entity]
) -> dict[str, str]:
    """``{derived id: given id}`` of each given entity of another id whose
    name has a derived entity's ``compact_name_key``."""
    by_key: dict[str, list[Entity]] = defaultdict(list)
    for entity in derived:
        by_key[compact_name_key(entity.name)].append(entity)
    pairs: dict[str, str] = {}
    for other in given:
        for entity in by_key.get(compact_name_key(other.name), ()):
            if entity.id != other.id:
                pairs.setdefault(entity.id, other.id)
    return pairs


def _with_old_ids(ids: Iterable[str], rekey: Mapping[str, str]) -> list[str]:
    """``ids`` and the stored ids ``rekey`` replaced by them (an item merged
    from several stored copies has its edges under each)."""
    wanted = set(ids)
    return sorted(wanted | {old for old, new in rekey.items() if new in wanted})


def _edges_of_written_entities(
    relationships: list[Relationship] | None, entity_stats: IndexingStats | None
) -> tuple[list[Relationship] | None, IndexingStats | None]:
    """The relationships to write after the graph entity writes, and the
    stats of those held back.

    An edge whose endpoint's write failed is not written: a re-keyed
    entity's edges would be re-pointed at a vertex that is not there. The
    held back edges count as failed, so their documents are written again.
    A failure that names no entity holds back every edge.
    """
    if not relationships or entity_stats is None or not entity_stats.failed_items:
        return relationships, None
    failed = set(entity_stats.failed_ids)
    held = [
        r
        for r in relationships
        if entity_stats.unattributed_failures
        or r.source_id in failed
        or r.target_id in failed
    ]
    if not held:
        return relationships, None
    logger.warning(
        "Not writing %d relationships: an endpoint entity's graph write failed",
        len(held),
    )
    stats = IndexingStats(total_items=len(held))
    stats.add_error("endpoint entity write failed", len(held), ids=[r.id for r in held])
    held_ids = {id(r) for r in held}
    return [r for r in relationships if id(r) not in held_ids], stats


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
        # Extraction-store keys named by removed text units, by suffix: the
        # candidates of collect_extraction_records.
        self._removed_record_keys: dict[str, set[str]] = {}
        # By suffix, the stored ids derivation re-keyed to follow the current
        # name -> their new ids: the items are written under the new ids,
        # then the old ones are deleted (delete_retired_ids).
        self._rekeyed: dict[str, dict[str, str]] = {}
        # By suffix, the entities derivation split off a stored one -> its id
        # (``Derived.split_from``): the stored claims about it follow them.
        self._split_from: dict[str, dict[str, str]] = {}
        # Stored entities a derivation of this run renamed under their id (a
        # case or spelling variant won): their claims name them anew.
        self._renamed: dict[str, set[str]] = {}
        # Ids of the stored edges this run rewrites for their rank only.
        self._rank_only: set[str] = set()
        # The suffixes whose graph merge found nothing stored.
        self._fresh: set[str] = set()
        # By suffix, the stored items a full build does not have (edges
        # ``Derived.dropped``, copies under a folded delta id): deleted with
        # the retired ids.
        self._dropped: dict[str, set[str]] = {}
        # The stored community reports a removal kept although they cite a
        # removed text unit (see community_drift).
        self.outdated_reports: set[str] = set()
        # Stored items derivation kept as they were: a text unit they cite
        # names no stored record (``Derived.underived``).
        self.underived_items = 0
        # By suffix, the entities this run's graph merge or strip derived (to
        # be written): the claims about them are derived against these.
        self._derived_entities: dict[str, dict[str, Entity]] = {}

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

    def initialize(self) -> None:
        """Prepare both stores for writes; a store's error propagates."""
        self.opensearch_indexer.initialize()
        self.neptune_indexer.initialize()

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
        results = self._write_artifacts(
            text_units,
            entities,
            relationships,
            communities,
            community_reports,
            claims,
            suffixes=suffixes,
        )
        written = results.get("opensearch_text_units")
        if written is not None and not written.failed_items:
            self._retain_extraction_records(text_units or [], suffixes)
        return results

    def _retain_extraction_records(
        self, text_units: list[TextUnit], suffixes: list[str]
    ) -> None:
        """Keep, per rebuilt suffix, only the extraction records its stored
        text units name: the records of other extraction settings, and of
        texts no stored unit holds any more (deleted or edited documents),
        go. Best-effort: a failure keeps the records, which cost space only.
        """
        named: dict[str, set[str]] = {suffix: set() for suffix in suffixes}
        for unit in text_units:
            named.setdefault(BaseIndexer.get_suffix(unit), set()).update(
                record_keys(unit).values()
            )
        for suffix, keys in sorted(named.items()):
            try:
                kept = self.opensearch_indexer.retain_extraction_records(
                    sorted(keys), suffix=suffix
                )
            except NamespaceLeaseError:
                raise
            except Exception as e:  # noqa: BLE001 - garbage only costs space
                logger.warning(
                    "Dropping the extraction records of '%s' its text units no "
                    "longer name failed; they stay until the next full build: %s",
                    suffix,
                    e,
                )
                continue
            logger.info(
                "Kept %d extraction records of '%s' (the ones its text units name)",
                kept.successful_items,
                suffix,
            )

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
        return self._write_artifacts(
            text_units,
            entities,
            relationships,
            communities,
            community_reports,
            claims,
            suffixes=None,
        )

    def _write_artifacts(
        self,
        text_units: list[TextUnit] | None,
        entities: list[Entity] | None,
        relationships: list[Relationship] | None,
        communities: list[Community] | None,
        community_reports: list[CommunityReport] | None,
        claims: list[Claim] | None,
        *,
        suffixes: list[str] | None,
    ) -> dict[str, IndexingStats]:
        """Write the artifacts in two phases: entities before the edges.

        With ``suffixes`` (full build) each type is replaced in those
        suffixes, even with no items. Without (delta) each is upserted: the
        delta path MUST NOT index, because index_community_reports does a full
        alias-swap (and index_communities a label clear) that would wipe the
        unchanged documents' artifacts the delta does not carry.
        """
        start_time = time.time()
        self._enrich_text_units(text_units, communities)
        attach_report_hierarchy(community_reports, communities)
        vector, graph = self.opensearch_indexer, self.neptune_indexer
        # (full-build writer, delta writer, items, result key)
        phase1: list[tuple[Callable, Callable, list[Any] | None, str]] = [
            (
                vector.index_text_units,
                vector.upsert_text_units,
                text_units,
                "opensearch_text_units",
            ),
            (
                vector.index_entities,
                vector.upsert_entities,
                entities,
                "opensearch_entities",
            ),
            (
                vector.index_community_reports,
                vector.upsert_community_reports,
                community_reports,
                "opensearch_community_reports",
            ),
            (vector.index_claims, vector.upsert_claims, claims, "opensearch_claims"),
            (graph.index_entities, graph.upsert_entities, entities, "neptune_entities"),
        ]
        # The OpenSearch relationship VECTOR index is only consumed by LightRAG
        # high-level keyword retrieval (mix/hybrid). Skip it for a GraphRAG-only
        # deployment (symmetric with community_detection.enabled). The Neptune
        # relationship EDGES of phase 2 are always written — both methodologies
        # traverse them for graph expansion.
        if self.config.indexing.opensearch.build_relationship_vector_index:
            phase1.append(
                (
                    vector.index_relationships,
                    self._upsert_vector_relationships,
                    relationships,
                    "opensearch_relationships",
                )
            )

        def tasks(
            specs: list[tuple[Callable, Callable, list[Any] | None, str]],
        ) -> list[IndexingTask]:
            if suffixes is None:
                return [IndexingTask(fn, [items], key) for _, fn, items, key in specs]
            return [
                IndexingTask(
                    partial(fn, suffixes=suffixes), [items or []], key, run_empty=True
                )
                for fn, _, items, key in specs
            ]

        # A full build's cleared (or, rebuilt in place, swept) entity labels
        # drop the old edges.
        edges = (
            graph.upsert_relationships
            if suffixes is None
            else graph.index_relationships
        )

        prefix, tail = ("", "") if suffixes is not None else ("Delta ", " (upsert)")
        # A full build replaces the graph in place where the backend can: the
        # labels are not cleared first, so queries read the items the build
        # already rewrote next to the previous ones it has not reached, and
        # what the build did not write is dropped at the end (the vector
        # indices swap aliases).
        # A run that lost its write lease stops before each phase and before
        # the sweep, which would drop what the new holder wrote.
        check_write_fence()
        rebuilding = suffixes is not None and graph.begin_rebuild(suffixes)
        try:
            logger.info("--- Starting %sIndexing Phase 1%s ---", prefix, tail)
            results = self._run_indexing_phase(tasks(phase1))
            check_write_fence()
            writable, skipped = _edges_of_written_entities(
                relationships, results.get("neptune_entities")
            )
            phase2 = [
                IndexingTask(edges, [writable], "neptune_relationships"),
                *tasks(
                    [
                        (
                            graph.index_communities,
                            graph.upsert_communities,
                            communities,
                            "neptune_communities",
                        )
                    ]
                ),
            ]
            logger.info("--- Starting %sIndexing Phase 2%s ---", prefix, tail)
            results.update(self._run_indexing_phase(phase2))
            check_write_fence()
            if skipped is not None:
                results.setdefault("neptune_relationships", IndexingStats()).merge(
                    skipped
                )
        except BaseException:
            if rebuilding:
                graph.end_rebuild(suffixes or [], sweep=False)
            raise
        if rebuilding:
            graph_failed = any(
                stats.failed_items
                for key, stats in results.items()
                if key.startswith("neptune_")
            )
            if graph_failed:
                logger.warning(
                    "Graph writes of the full build failed: keeping the graph's "
                    "previous vertices and edges beside the new ones until a "
                    "build succeeds"
                )
                # Reported as a failed write: the rebuild did not replace the
                # graph, so it must not be recorded as the corpus's build (a
                # registry cleared now would leave the previous vertices and
                # edges with nothing to remove them).
                skipped = IndexingStats(total_items=1)
                skipped.add_error(
                    "previous graph not dropped: some graph writes failed"
                )
                results["neptune_sweep"] = skipped
            try:
                graph.end_rebuild(suffixes or [], sweep=not graph_failed)
            except Exception as e:  # noqa: BLE001 - reported as a failed write
                logger.error("Dropping the previous graph failed: %s", e)
                stats = IndexingStats(total_items=1)
                stats.add_error(str(e))
                results["neptune_sweep"] = stats

        self._log_completion_summary(results, time.time() - start_time)
        return results

    def _upsert_vector_relationships(
        self, relationships: list[Relationship]
    ) -> IndexingStats:
        """Delta upsert of the relationship vectors. An edge this run rewrites
        for its rank only is updated in place (no embedding), unless its
        vector copy is missing (an interrupted write): that one is upserted."""
        rank_only, self._rank_only = self._rank_only, set()
        vector = self.opensearch_indexer
        candidates = [r for r in relationships if r.id in rank_only]
        if not candidates:
            return vector.upsert_relationships(relationships)
        stored: set[str] = set()
        for suffix, group in self._group_by_suffix(candidates).items():
            stored |= set(
                vector.read_relationship_descriptions(
                    [r.id for r in group], suffix=suffix
                )
            )
        partial = [r for r in candidates if r.id in stored]
        kept = {r.id for r in partial}
        full = [r for r in relationships if r.id not in kept]
        stats = vector.update_relationship_lineage(partial)
        if full:
            stats.merge(vector.upsert_relationships(full))
        stats.total_items = len(relationships)
        return stats

    def merge_with_existing_graph(
        self,
        entities: list[Entity] | None,
        relationships: list[Relationship] | None,
        text_units: Sequence[TextUnit] = (),
    ) -> CrossRunMergeResult:
        """Union delta artifacts with existing graph state before upsert.

        Reads the existing entities/relationships the delta touches back from the
        graph store and merges (description/text_unit_ids union, weight
        recompute) via the pure merge functions, so a cross-run upsert accumulates
        rather than overwriting. If the adapter does not support read-back it
        returns ``[]`` and this degenerates to the existing overwrite behaviour.
        A failed read raises: overwriting would drop the stored lineage, and
        raising before any write leaves the docs unrecorded for a retry.

        Merges one index suffix at a time: entity and relationship ids are
        suffix-independent, so each tenant's delta must meet that tenant's stored
        state only (and its entity-id remap must not leak into another tenant).

        The content of every item written (name, type, description,
        confidence, attributes) is then derived from the records of the text
        units it cites (``domain.ingestion.derivation``): ``text_units`` are
        the delta's, the others' record keys are read from the vector store.
        The stored edges of an entity whose name or rank changed are written
        too, with its name and their ranks.
        """
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
                [u for u in text_units if BaseIndexer.get_suffix(u) == suffix],
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
        self, claims: list[Claim] | None, text_units: Sequence[TextUnit] = ()
    ) -> list[Claim] | None:
        """Fold the stored state of each delta claim into it before upsert.

        A claim's id does not depend on its document, so the claim another
        document already stored has the delta claim's id: without the merge
        the upsert replaces its descriptions and text units with this run's
        (see ``merge_claims``). Read per index suffix from the vector store; a
        failed read raises (see :meth:`merge_with_existing_graph`).

        The stored claims about an entity the graph merge re-keyed are
        derived too, so they follow its new id (ids follow the current name,
        see ``derivation``), and so are those citing a delta text unit (a
        resumed run's removal derived the killed run's claims of the delta
        against a graph without the delta's entities).
        """
        if not (claims or self._rekeyed or self._split_from or self._renamed):
            return claims
        groups = self._group_by_suffix(claims or [])
        merged: list[Claim] = []
        for suffix in sorted(
            groups.keys()
            | self._rekeyed.keys()
            | self._split_from.keys()
            | self._renamed.keys()
        ):
            group = groups.get(suffix, [])
            stored = self.opensearch_indexer.read_claims(
                sorted({c.id for c in group}), suffix=suffix
            )
            group = merge_claims(stored, group) if stored else group
            units = [u for u in text_units if BaseIndexer.get_suffix(u) == suffix]
            # The stored claims about an entity the graph merge re-keyed
            # follow it (see merge_with_existing_graph), and those citing a
            # delta text unit are derived with the delta's: a resumed
            # removal may have derived one the killed run wrote under an id
            # this delta does not write.
            citing = (
                self._vector_citing(
                    self.config.indexing.opensearch.claims_index_prefix,
                    [u.id for u in units],
                    suffix,
                )
                if units
                else {}
            )
            about = sorted(
                (
                    set(self._claims_about(suffix, self._rederived_entity_ids(suffix)))
                    | set(citing)
                )
                - {c.id for c in group}
            )
            for claim in self.opensearch_indexer.read_claims(about, suffix=suffix):
                claim.attributes = {
                    **(claim.attributes or {}),
                    Constants.INDEX.value: suffix,
                }
                group.append(claim)
            if not group:
                continue
            if not stored and not about and suffix in self._fresh:
                # Nothing stored: the delta claims are the full build's.
                merged.extend(group)
                continue
            merged.extend(self._derive_claims(suffix, group, units))
        return merged if claims is not None or merged else claims

    def _merge_suffix(
        self,
        suffix: str,
        entities: list[Entity],
        relationships: list[Relationship],
        text_units: Sequence[TextUnit] = (),
    ) -> tuple[list[Entity], list[Relationship], dict[str, str]]:
        """Cross-run merge of one suffix's delta (see merge_with_existing_graph).

        Returns the entities and relationships to write plus ``{delta id: kept
        id}`` for every delta entity or relationship that merged into a stored
        one under a different id.
        """
        merged_entities = entities
        existing: list[Entity] = []
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
        merged_relationships = relationships
        existing_rels: list[Relationship] = []
        relationship_remap: dict[str, str] = {}
        if relationships:
            rel_ids = {r.id for r in relationships}
            if merged_entities:
                # A delta edge can match a stored edge of other id between the
                # same entities (extracted with other spellings, or with an
                # endpoint that merged into a stored entity): the full build
                # merges them, so the entities' stored edges are read too.
                rel_ids |= set(
                    self.neptune_indexer.find_incident_relationship_ids(
                        sorted({e.id for e in merged_entities}), suffix=suffix
                    )
                )
            delta_ids = {r.id for r in relationships}
            pairs = {
                (
                    entity_id_remap.get(r.source_id, r.source_id),
                    entity_id_remap.get(r.target_id, r.target_id),
                )
                for r in relationships
            }
            existing_rels = self._with_full_descriptions(
                [
                    r
                    for r in self.neptune_indexer.read_relationships(
                        sorted(rel_ids), suffix=suffix
                    )
                    if r.id in delta_ids or (r.source_id, r.target_id) in pairs
                ],
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
                merged_relationships = [
                    r for r in merged_relationships if r.id in written
                ]
        if merged_entities is not entities and merged_relationships:
            # Merged with stored entities: an edge re-pointed at one, or at one
            # whose stored name differs from the delta's, must name it as the
            # entity does.
            merged_relationships = [r.model_copy() for r in merged_relationships]
            sync_relationship_endpoint_names(merged_relationships, merged_entities)
        if existing or existing_rels:
            derived = self.derive(
                suffix, merged_entities, merged_relationships, (), text_units
            )
        else:
            # Nothing stored (a first build with the registry on): the delta
            # is what a full build over its records gives, summaries included.
            self._fresh.add(suffix)
            derived = Derived(
                entities=list(merged_entities),
                relationships=list(merged_relationships),
            )
        rekey = derived.id_remap
        self._note_renamed(suffix, derived.entities, existing)

        def final_id(delta_id: str, merged_into: Mapping[str, str]) -> str:
            """The id a delta item ends under: the stored one it merged into
            (another spelling, say), as the derivation re-keyed it."""
            kept = merged_into.get(delta_id, delta_id)
            return rekey.get(kept, kept)

        # A stored summary that only grew is summarized with what the delta
        # added (see extend_summaries).
        reuse_summaries(derived.entities, entities)
        reuse_summaries(derived.relationships, relationships)
        extend_summaries(
            derived.entities,
            [e.model_copy(update={"id": rekey.get(e.id, e.id)}) for e in existing],
            {final_id(e.id, entity_id_remap): e.description for e in entities},
        )
        extend_summaries(
            derived.relationships,
            [r.model_copy(update={"id": rekey.get(r.id, r.id)}) for r in existing_rels],
            {final_id(r.id, relationship_remap): r.description for r in relationships},
        )
        merged_entities, merged_relationships = self._finish(
            suffix,
            derived.entities,
            derived.relationships,
            # The delta's summaries (graph resolution made them) are reused
            # as the stored ones are: by the description they were made from.
            stored_entities=[*existing, *entities],
            stored_relationships=[*existing_rels, *relationships],
            rekey=rekey,
            rewired=set(derived.rewired),
            dropped=derived.dropped,
        )
        # A delta id the merge folded into a stored item of another id has no
        # graph copy (the stored one of that id would have been merged), but
        # a run that died between the stores may have left its vector copy:
        # it goes with the retired ids, unless the derivation writes it (an
        # entity split off under it).
        written = {e.id for e in merged_entities} | {r.id for r in merged_relationships}
        folded = [
            delta_id
            for delta_id, kept in {**entity_id_remap, **relationship_remap}.items()
            if delta_id != kept and delta_id not in written
        ]
        self._retire(
            suffix,
            rekey,
            derived.split_from,
            [i for i in derived.dropped if i not in written] + folded,
        )
        self._derived_entities.setdefault(suffix, {}).update(
            {e.id: e for e in merged_entities}
        )
        # A delta id that merged into a stored one follows that one's re-key;
        # the stored ids map to their new ids for what else references them.
        return (
            merged_entities,
            merged_relationships,
            {
                **rekey,
                **{
                    delta_id: rekey.get(kept, kept)
                    for delta_id, kept in {
                        **entity_id_remap,
                        **relationship_remap,
                    }.items()
                },
            },
        )

    def derive(
        self,
        suffix: str,
        entities: Sequence[Entity],
        relationships: Sequence[Relationship],
        claims: Sequence[Claim],
        text_units: Sequence[TextUnit] = (),
        rekeyed: Mapping[str, str] | None = None,
        split_from: Mapping[str, str] | None = None,
    ) -> Derived:
        """``derivation.derive`` over the records of the items' text units:
        ``text_units`` carry their record keys, the others are read from the
        vector store. A failed read raises (see merge_with_existing_graph)."""
        items = (list(entities), list(relationships), list(claims))
        blocked: set[str] = set()
        # A new id another stored item holds: that item is a copy an
        # interrupted re-key left, or another spelling a full build merges
        # with this one, so it is derived with them (and merged when it is
        # the same item). One still held after that keeps its holder: the
        # item keeps its id rather than overwrite another. The stored edges
        # of an entity that was re-keyed or split are derived with it
        # (``Derived.rewired``): they may become one with a derived edge, or
        # belong to a part it split into.
        for attempt in range(4):
            unit_ids = sorted(
                {
                    u
                    for group in items
                    for item in group
                    for u in item.text_unit_ids or []
                }
            )
            records, claim_records = self._unit_records(suffix, unit_ids, text_units)
            stored_edges = (
                None
                if claims or not items[0]
                else self.neptune_indexer.find_incident_relationship_ids(
                    sorted(_with_old_ids([e.id for e in items[0]], rekeyed or {})),
                    suffix=suffix,
                )
            )
            derived = derive(
                self.config,
                *items,
                records,
                claim_records,
                rekeyed=rekeyed,
                blocked=blocked,
                split_from=split_from,
                stored_relationship_ids=stored_edges,
            )
            given = {item.id for group in items for item in group}
            # Stored entities of a derived name's key (another spelling a
            # full build merges with it, that cites none of these units), and
            # stored claims of a derived claim's id, are held too.
            siblings = (
                self.neptune_indexer.find_entity_ids_by_name_key(
                    sorted({compact_name_key(e.name) for e in derived.entities}),
                    suffix,
                )
                if derived.entities and not claims
                else None
            )
            targets = sorted(
                (
                    set(derived.id_remap.values())
                    | set(derived.split_from)
                    | set(siblings or ())
                    | {c.id for c in derived.claims}
                )
                - given
                - blocked
            )
            held = self._read_held(suffix, targets) if targets else ([], [], [])
            # Another spelling of a derived entity: a run that split them and
            # died before their edges and claims were written leaves those on
            # the other spelling, and only its entity copies record the split
            # (the two need not share a text unit). Such a leftover cites a
            # unit of the spelling it belongs to: only a spelling holding one
            # is rederived (all its edges, and its claims follow through
            # ``split_from``). A spelling a full build keeps apart holds none,
            # and is left as stored.
            spellings = (
                {
                    part: whole
                    for part, whole in _other_spellings(
                        derived.entities, items[0]
                    ).items()
                    if whole not in derived.id_remap
                }
                if not claims
                else {}
            )
            moved = set(derived.rewired)
            candidates = sorted({*moved, *spellings.values()})
            parted: list[Relationship] = []
            if candidates and not claims:
                incident = set(
                    self.neptune_indexer.find_incident_relationship_ids(
                        candidates, suffix=suffix
                    )
                )
                if incident - given:
                    parted = self.neptune_indexer.read_relationships(
                        sorted(incident - given), suffix=suffix
                    )
            if spellings:
                spellings = self._leftover_spellings(
                    suffix, derived.entities, spellings, parted
                )
                moved |= set(spellings.values())
                parted = [r for r in parted if {r.source_id, r.target_id} & moved]
            rewired = sorted(moved)
            if (not any(held) and not parted) or attempt == 3:
                self.underived_items += len(set(derived.underived))
                return derived.model_copy(
                    update={
                        "rewired": rewired,
                        "split_from": {**spellings, **derived.split_from},
                    }
                )
            if attempt == 0:
                items = (
                    [*items[0], *held[0]],
                    [*items[1], *held[1]],
                    [*items[2], *held[2]],
                )
            else:
                blocked |= {item.id for group in held for item in group}
            items[1].extend(parted)
            # Their other endpoints too: a re-pointed edge changes their rank.
            ends = {i for r in parted for i in (r.source_id, r.target_id)}
            known = {e.id for e in items[0]}
            if ends - known:
                items[0].extend(
                    self.neptune_indexer.read_entities(
                        sorted(ends - known), suffix=suffix
                    )
                )
        self.underived_items += len(set(derived.underived))
        return derived

    def _leftover_spellings(
        self,
        suffix: str,
        derived: Sequence[Entity],
        spellings: Mapping[str, str],
        edges: Sequence[Relationship],
    ) -> dict[str, str]:
        """The ``{part: whole}`` of ``spellings`` whose whole holds an edge
        among ``edges``, or a stored claim, that cites a unit of a derived
        spelling of its name (the part's or another's: a whole may hold a
        third spelling's statement): what a split a killed run left
        half-written leaves behind."""
        units_of = {e.id: set(e.text_unit_ids or []) for e in derived}
        key_of = {e.id: compact_name_key(e.name) for e in derived}
        parts_by_key: dict[str, list[str]] = defaultdict(list)
        for part in spellings:
            parts_by_key[key_of.get(part, "")].append(part)

        def stated(part: str, whole: str) -> set[str]:
            # The units of every part of its name but the whole itself (a
            # spelling a full build keeps apart is a part too: its own
            # edges and claims cite its own units).
            return set().union(
                *(
                    units_of.get(other, set())
                    for other in parts_by_key[key_of.get(part, "")]
                    if other != whole
                )
            )

        left: dict[str, str] = {}
        for part, whole in spellings.items():
            units = stated(part, whole)
            if any(
                whole in (r.source_id, r.target_id)
                and units.intersection(r.text_unit_ids or [])
                for r in edges
            ):
                left[part] = whole
        pending = {p: w for p, w in spellings.items() if p not in left}
        if pending:
            prefix = self.config.indexing.opensearch.claims_index_prefix
            for part, whole in pending.items():
                part_units = sorted(stated(part, whole))
                citing = (
                    self.opensearch_indexer.find_documents(
                        prefix, "text_unit_ids", part_units, suffix
                    )
                    or {}
                    if part_units
                    else {}
                )
                if citing and set(citing) & set(self._claims_about(suffix, [whole])):
                    left[part] = whole
        return left

    def _read_held(
        self, suffix: str, ids: list[str]
    ) -> tuple[list[Entity], list[Relationship], list[Claim]]:
        """The stored entities, relationships and claims of ``ids``."""
        claims = self.opensearch_indexer.read_claims(ids, suffix=suffix)
        for claim in claims:
            claim.attributes = {
                **(claim.attributes or {}),
                Constants.INDEX.value: suffix,
            }
        return (
            self.neptune_indexer.read_entities(ids, suffix=suffix),
            self.neptune_indexer.read_relationships(ids, suffix=suffix),
            claims,
        )

    def _unit_records(
        self,
        suffix: str,
        unit_ids: Sequence[str],
        text_units: Sequence[TextUnit],
    ) -> tuple[dict[str, TextUnitExtraction], dict[str, list[Claim]]]:
        """The graph and claim records of ``unit_ids`` (those found)."""
        units = {u.id: u for u in text_units if u.id in set(unit_ids)}
        missing = [i for i in unit_ids if i not in units]
        if missing:
            for (
                unit_id,
                attributes,
            ) in self.opensearch_indexer.read_text_unit_attributes(
                missing, suffix=suffix
            ).items():
                units[unit_id] = TextUnit.model_validate(
                    {"id": unit_id, "text": "", "attributes": attributes}
                )
        keys = {u.id: record_keys(u) for u in units.values()}
        payloads = self.opensearch_indexer.read_extraction_records(
            sorted({key for k in keys.values() for key in k.values()}), suffix=suffix
        )
        records: dict[str, TextUnitExtraction] = {}
        claim_records: dict[str, list[Claim]] = {}
        for unit_id, unit_keys in keys.items():
            unit = units[unit_id]
            try:
                if (payload := payloads.get(unit_keys.get("graph", ""))) is not None:
                    records[unit_id] = record_from_payload(payload, unit)
                if (payload := payloads.get(unit_keys.get("claims", ""))) is not None:
                    claim_records[unit_id] = claims_from_payload(payload, unit)
            except (ValueError, TypeError) as e:
                logger.warning(
                    "Ignoring an unreadable extraction record of text unit '%s': %s",
                    unit.short_id or unit_id,
                    type(e).__name__,
                )
        return records, claim_records

    def _finish(
        self,
        suffix: str,
        entities: list[Entity],
        relationships: list[Relationship],
        *,
        stored_entities: Sequence[Entity] = (),
        stored_relationships: Sequence[Relationship] = (),
        rekey: Mapping[str, str] | None = None,
        rewired: Collection[str] = (),
        dropped: Collection[str] = (),
    ) -> tuple[list[Entity], list[Relationship]]:
        """Summarize, rank and complete derived items for writing.

        A description over the summarization threshold is summarized again
        only when it differs from the one the stored summary was made from.
        Ranks are recomputed over the stored graph. The stored edges of the
        entities whose names or ranks now differ (an entity renamed, or its
        degree or a neighbour's changed) are returned with the items, with
        the entities' names and their recomputed rank; ``rekey`` (stored id
        -> derived id, see ``Derived.id_remap``) re-points the stored edges
        of a re-keyed entity at its new id. The stored edges of the
        ``rewired`` entities (``Derived.rewired``) are all among
        ``relationships``: their stored links do not count.
        """
        rekey = rekey or {}
        reuse_summaries(entities, stored_entities)
        reuse_summaries(relationships, stored_relationships)
        if any(needs_summary(self.config, i.description) for i in entities):
            entities = self.description_summarizer.summarize_entities(entities)
        if any(needs_summary(self.config, i.description) for i in relationships):
            relationships = self.description_summarizer.summarize_relationships(
                relationships
            )
        entities, relationships = self._rank_with_stored_graph(
            suffix, entities, relationships, rekey, rewired
        )
        # The edges the derivation dropped go with the retired ids: they are
        # not rewritten as other stored edges of the touched entities.
        written = {r.id for r in relationships} | set(rekey) | set(dropped)
        extra_ids = sorted(
            set(
                self.neptune_indexer.find_incident_relationship_ids(
                    _with_old_ids([e.id for e in entities], rekey), suffix=suffix
                )
            )
            - written
        )
        if not extra_ids:
            return entities, relationships
        # Their stored copies are compared with the names and ranks the
        # touched entities now give them (not the entities with their stored
        # copies): a run that wrote the entities but died before their edges
        # leaves edges a retry still finds stale.
        stored_extra = self.neptune_indexer.read_relationships(extra_ids, suffix=suffix)
        extra = [
            r.model_copy(
                deep=True,
                update={
                    "source_id": rekey.get(r.source_id, r.source_id),
                    "target_id": rekey.get(r.target_id, r.target_id),
                },
            )
            for r in stored_extra
        ]
        # A stored edge of a written edge's key (endpoints and type, as the
        # merge matches them) under another id, citing no unit the written
        # one does not, is that edge under an earlier id (the derivation
        # re-pointed the written one onto these endpoints): a full build has
        # one edge, so it goes with the retired ids.
        written_by_key = {
            relationship_key(r.source_id, r.target_id, r): r for r in relationships
        }
        duplicates = {
            r.id
            for r in extra
            if (w := written_by_key.get(relationship_key(r.source_id, r.target_id, r)))
            is not None
            and w.id != r.id
            and set(r.text_unit_ids or []) <= set(w.text_unit_ids or [])
        }
        # An edge between two entities the re-key merged is a self-loop a
        # full build does not draw: it goes with the retired ids.
        loops = {r.id for r in extra if r.source_id == r.target_id} | duplicates
        if loops:
            self._dropped.setdefault(suffix, set()).update(loops)
            extra = [r for r in extra if r.id not in loops]
            stored_extra = [r for r in stored_extra if r.id not in loops]
        sync_relationship_endpoint_names(extra, entities)
        endpoints = sorted({i for r in extra for i in (r.source_id, r.target_id)})
        _, ranked = with_graph_ranks(
            [],
            [*relationships, *extra],
            self._stored_neighbors(suffix, endpoints, rekey, rewired),
        )
        ranks = {r.id: r.rank for r in ranked}
        for rel in extra:
            rel.rank = ranks[rel.id]

        def shown(r: Relationship) -> tuple[Any, ...]:
            return r.source_id, r.target_id, r.source_name, r.target_name, r.rank

        before = {r.id: shown(r) for r in stored_extra}
        extra = [r for r in extra if before[r.id] != shown(r)]
        if not extra:
            return entities, relationships
        # Only the rank changed (a neighbour's degree): the vector copy is
        # updated in place, not embedded again (_upsert_vector_relationships).
        self._rank_only.update(r.id for r in extra if before[r.id][:4] == shown(r)[:4])
        extra = self._with_full_descriptions(
            extra, self.opensearch_indexer.read_relationship_descriptions, suffix
        )
        changed = {i for r in extra for i in (r.source_id, r.target_id)}
        logger.info(
            "Rewriting the name or rank of %d stored relationships of %d "
            "entities in '%s'",
            len(extra),
            len(changed & {e.id for e in entities}),
            suffix,
        )
        return entities, [*relationships, *extra]

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
        rekey: Mapping[str, str],
        rewired: Collection[str] = (),
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
        return with_graph_ranks(
            entities,
            relationships,
            self._stored_neighbors(suffix, ids, rekey, rewired),
        )

    def _stored_neighbors(
        self,
        suffix: str,
        ids: Sequence[str],
        rekey: Mapping[str, str],
        rewired: Collection[str] = (),
    ) -> dict[str, list[str]]:
        """``read_neighbor_ids`` of ``ids``, some of which may be re-keyed
        (``rekey``: stored id -> new id), stated in the new ids, without the
        links of the ``rewired`` stored entities (the caller holds all their
        edges)."""
        stored = self.neptune_indexer.read_neighbor_ids(
            _with_old_ids(ids, rekey), suffix=suffix
        )
        merged: dict[str, set[str]] = defaultdict(set)
        for i, neighbors in stored.items():
            if i in rewired:
                continue
            merged[rekey.get(i, i)].update(
                rekey.get(n, n) for n in neighbors if n not in rewired
            )
        return {i: sorted(neighbors - {i}) for i, neighbors in merged.items()}

    def _retire(
        self,
        suffix: str,
        rekey: Mapping[str, str],
        split_from: Mapping[str, str] | None = None,
        dropped: Iterable[str] = (),
    ) -> None:
        """Mark the stored ids ``rekey`` replaced, and the ``dropped`` edges,
        for deletion once the items are written under their new ids
        (:meth:`delete_retired_ids`), and note the entities split off stored
        ones (their claims follow)."""
        if rekey:
            self._rekeyed.setdefault(suffix, {}).update(rekey)
        if split_from:
            self._split_from.setdefault(suffix, {}).update(split_from)
        if dropped:
            self._dropped.setdefault(suffix, set()).update(dropped)

    def _rederived_entity_ids(self, suffix: str) -> list[str]:
        """The stored entity ids whose claims this run derives again: the
        re-keyed ones, those an entity was split off and the parts split off
        them (a stored claim may name a part already), the other spellings
        of a split-off entity's name this run derived (a claim naming the
        part may resolve to one of them) and those renamed."""
        split_from = self._split_from.get(suffix, {})
        return sorted(
            {
                *self._rekeyed.get(suffix, {}),
                *split_from.values(),
                *split_from,
                *_other_spellings_of_parts(
                    split_from, self._derived_entities.get(suffix, {})
                ),
                *self._renamed.get(suffix, set()),
            }
        )

    def _note_renamed(
        self, suffix: str, derived: Sequence[Entity], stored: Sequence[Entity]
    ) -> list[str]:
        """Record (and return) the stored entities ``derived`` renamed under
        their own id."""
        names = {e.id: e.name for e in stored}
        renamed = [e.id for e in derived if e.id in names and names[e.id] != e.name]
        if renamed:
            self._renamed.setdefault(suffix, set()).update(renamed)
        return renamed

    def delete_retired_ids(self) -> dict[str, IndexingStats]:
        """Delete the items whose ids a derivation replaced (see
        ``Derived.id_remap``) from both stores.

        Run after their items are written under the new ids: the re-pointed
        edges then hold the new endpoints, and deleting the old entity vertex
        drops no edge still in use.
        """
        retired = {
            suffix: sorted(
                (set(self._rekeyed.get(suffix, {})) | self._dropped.get(suffix, set()))
                - set(self._rekeyed.get(suffix, {}).values())
            )
            for suffix in self._rekeyed.keys() | self._dropped.keys()
        }
        self._rekeyed.clear()
        self._dropped.clear()
        self._split_from.clear()
        self._renamed.clear()
        self._fresh.clear()
        self._rank_only.clear()
        self._derived_entities.clear()
        if retired:
            logger.info(
                "Deleting %d items re-keyed to follow their current name",
                sum(len(ids) for ids in retired.values()),
            )
        return self.delete_documents(retired)

    def discard_retired_ids(self) -> None:
        """Keep the re-keyed items' old ids (their writes failed: deleting
        them would lose the items)."""
        self._rekeyed.clear()
        self._dropped.clear()
        self._split_from.clear()
        self._renamed.clear()
        self._fresh.clear()
        self._rank_only.clear()
        self._derived_entities.clear()

    def _match_stored_names(
        self, suffix: str, entities: list[Entity]
    ) -> tuple[FuzzyMatcher | None, set[str]]:
        """Match delta entity names against the stored names of ``suffix``.

        Returns a matcher over the stored names (built with the full build's
        settings) and the ids of the stored entities some delta name matches.
        Returns ``(None, set())`` when the adapter projects no names, which
        leaves the merge to ids and exact keys.

        Without ``fuzzy_entity_resolution`` names match by their
        ``compact_name_key`` only, so just the stored entities with a delta
        name's key are read (``GraphIndexer.find_entity_ids_by_name_key``);
        fuzzy matching, or a graph store that cannot look keys up, reads
        every stored name.
        """
        processing = self.config.processing
        graph = self.neptune_indexer
        keyed = (
            None
            if processing.fuzzy_entity_resolution
            else graph.find_entity_ids_by_name_key(
                sorted({compact_name_key(e.name) for e in entities}), suffix
            )
        )
        stored_names = (
            graph.read_entity_names(suffix)
            if keyed is None
            else [(e.id, e.name) for e in graph.read_entities(keyed, suffix=suffix)]
        )
        id_by_name: dict[str, str] = {}
        for eid, name in stored_names:
            id_by_name.setdefault(name, eid)
        if not id_by_name:
            return None, set()

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
        community_ids: Collection[str] = (),
        earlier_notes: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, IndexingStats]:
        """Strip removed text units from kept entities/relationships/claims of a
        suffix, and derive their content again from the text units left.

        Reads the stored items back, removes ``text_unit_ids`` from their
        lineage (recomputing weight, see ``remove_text_units``), derives their
        name, type, description, confidence and attributes from the records of
        the text units they still cite (``derive``), so nothing a removed
        document contributed stays, and recomputes their ranks (and the ranks
        and endpoint names of their other stored edges, see ``_finish``).
        Every item read back is written to both stores (the vector store
        updates lineage and rank in place when nothing it embeds changed), so a
        retry also repairs a store an earlier, interrupted strip did not
        reach. A graph adapter without read-back leaves them unchanged.

        Claims live in the vector store only: they are read back from it,
        stripped and derived the same way.

        An item whose derived name has another id takes it (with its edges
        and claims, see ``derivation``); the old ids are deleted once every
        write succeeded, else kept (a retry re-keys them).

        A failed read is reported as failed items, so the caller keeps the
        documents' registry rows and retries the removal.

        A strip that merges or re-keys notes its items first
        (:meth:`_note_pending_strip`), with those of the ``earlier_notes``
        (``{key: note}``) of a strip a retry finishes; the notes are deleted
        once the old ids are.
        """
        earlier_notes = earlier_notes or {}
        results, noted = self._strip_shared(
            suffix,
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
            text_unit_ids=text_unit_ids,
            claim_ids=claim_ids,
            community_ids=community_ids,
            earlier_notes=earlier_notes,
        )
        # A note is kept until the old ids are deleted (a retry needs it).
        if any(stats.failed_items for stats in results.values() if stats):
            self.discard_retired_ids()
            return results
        results = {**results, **self.delete_retired_ids()}
        if (noted or earlier_notes) and not any(
            s.failed_items for s in results.values() if s
        ):
            self._clear_pending_strip(suffix, earlier_notes)
        return results

    def _note_pending_strip(
        self, suffix: str, entity_ids: list[str], relationship_ids: list[str]
    ) -> None:
        """Note the stored items a strip merges or re-keys, in one record of
        ``suffix``'s extraction store (kept until :meth:`_clear_pending_strip`),
        so a retry of the removal strips them again (:meth:`remove_text_units`).
        Raises when the write fails, before the strip writes anything."""
        note = {"entity_ids": entity_ids, "relationship_ids": relationship_ids}
        stats = self.opensearch_indexer.write_extraction_records(
            {_PENDING_STRIP: note}, suffix=suffix
        )
        if stats.failed_items:
            raise DataProcessingError(
                f"Noting the pending strip of '{suffix}' failed: {stats.errors[:1]}"
            )

    def _clear_pending_strip(
        self, suffix: str, earlier_notes: Collection[str] = ()
    ) -> None:
        """Delete the note of :meth:`_note_pending_strip` and the
        ``earlier_notes`` keys (best-effort: one left only makes a retry strip
        the items again)."""
        try:
            self.opensearch_indexer.delete_extraction_records(
                sorted({_PENDING_STRIP, *earlier_notes}), suffix=suffix
            )
        except NamespaceLeaseError:
            raise
        except Exception as e:  # noqa: BLE001 - a leftover note is harmless
            logger.warning("Deleting the strip notes of '%s' failed: %s", suffix, e)

    def _strip_shared(
        self,
        suffix: str,
        *,
        entity_ids: list[str],
        relationship_ids: list[str],
        text_unit_ids: list[str],
        claim_ids: Sequence[str] = (),
        community_ids: Collection[str] = (),
        earlier_notes: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> tuple[dict[str, IndexingStats], bool]:
        """:meth:`remove_text_units_from_shared` before the old ids of the
        re-keyed items are deleted, and whether it noted them
        (:meth:`_note_pending_strip`; with the ``earlier_notes`` items, so
        the note it replaces is not lost)."""
        results: dict[str, IndexingStats] = {}
        noted = False
        doing = "Reading shared artifacts"
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
            # An edge is derived with its endpoints (the resolved entities map
            # to them). They cite the removed units too, unless a killed run
            # stripped them and left the edge (re-keyed, not yet deleted).
            ends = {i for r in relationships for i in (r.source_id, r.target_id)}
            ends -= {e.id for e in entities}
            if ends:
                entities += self.neptune_indexer.read_entities(
                    sorted(ends), suffix=suffix
                )
            entities = self._with_full_descriptions(
                entities, self.opensearch_indexer.read_entity_descriptions, suffix
            )
            relationships = self._with_full_descriptions(
                relationships,
                self.opensearch_indexer.read_relationship_descriptions,
                suffix,
            )
            stripped_entities, stripped_relationships = remove_text_units(
                entities, relationships, text_unit_ids, community_ids
            )
            # Every item read back is rewritten, stripped or not: the graph
            # copy is the reference, and after a strip that was interrupted
            # or partly failed between the stores it no longer cites the
            # removed text units while the vector copy still does.
            stripped = {item.id: item for item in stripped_entities}
            stripped_rels = {item.id: item for item in stripped_relationships}
            derived = self.derive(
                suffix,
                [stripped.get(e.id, e) for e in entities],
                [stripped_rels.get(r.id, r) for r in relationships],
                (),
            )
            new_entities, new_relationships = self._finish(
                suffix,
                derived.entities,
                derived.relationships,
                stored_entities=entities,
                stored_relationships=relationships,
                rekey=derived.id_remap,
                rewired=set(derived.rewired),
                dropped=derived.dropped,
            )
            self._retire(suffix, derived.id_remap, derived.split_from, derived.dropped)
            self._derived_entities.setdefault(suffix, {}).update(
                {e.id: e for e in new_entities}
            )
            entity_ids_now = {e.id for e in derived.entities}
            edge_ids_now = {r.id for r in derived.relationships}
            claim_ids_now = {c.id for c in derived.claims}
            merged = {
                i
                for pair in derived.id_remap.items()
                if pair[1] in entity_ids_now
                for i in pair
            }
            # The old ids of the edges re-keyed into a derived edge, and of
            # those dropped: no claim.
            retired_edges = {
                old for old, new in derived.id_remap.items() if new in edge_ids_now
            } | {i for i in derived.dropped if i not in claim_ids_now}
            if merged or retired_edges:
                # Their old ids need not cite the removed text units (another
                # spelling merged in, or an edge re-pointed at it): once the
                # items are written, a retry finds them through this note
                # only, and strips them again. Noted: the entities on both
                # sides of a merge and the edges re-keyed or dropped. Not a
                # split's part and origin: a split keeps the stored id, and
                # stripping them again on a retry gave a claim the wrong
                # subject.
                doing = "Noting the pending strip"
                earlier = earlier_notes or {}
                self._note_pending_strip(
                    suffix,
                    sorted(merged | _noted(earlier, "entity_ids")),
                    sorted(retired_edges | _noted(earlier, "relationship_ids")),
                )
                noted = True
            doing = "Reading the claims of re-keyed entities"
            # The stored claims about a re-keyed, split or renamed entity
            # follow it.
            renamed = self._note_renamed(suffix, derived.entities, entities)
            claim_ids = sorted(
                {
                    *claim_ids,
                    *self._claims_about(
                        suffix,
                        [
                            *derived.id_remap,
                            *derived.split_from.values(),
                            *_other_spellings_of_parts(
                                derived.split_from,
                                {e.id: e for e in derived.entities},
                            ),
                            *renamed,
                        ],
                    ),
                }
            )
        except Exception as e:  # noqa: BLE001 - reported as failed items
            logger.error("%s in '%s' failed: %s", doing, suffix, e)
            stats = IndexingStats(total_items=len(entity_ids) + len(relationship_ids))
            stats.add_error(str(e), count=stats.total_items)
            return {**results, f"neptune_read_{suffix}": stats}, noted
        if claim_ids:
            results.update(
                self._remove_text_units_from_claims(
                    suffix, list(claim_ids), text_unit_ids
                )
            )
        if not new_entities and not new_relationships:
            return results, noted
        logger.info(
            "Removing %d text units from %d shared entities and %d shared "
            "relationships in '%s'",
            len(text_unit_ids),
            len(entities),
            len(relationships),
            suffix,
        )
        # The vector store re-embeds only what changed: the content the
        # graph copy had (name, type, confidence) and the description the
        # vector copy holds (an earlier strip may have reached only the graph).
        graph_copy = {e.id: _vector_content(e) for e in entities} | {
            r.id: _vector_content(r) for r in relationships
        }
        vector_text = {
            **self.opensearch_indexer.read_entity_descriptions(
                [e.id for e in new_entities], suffix=suffix
            ),
            **self.opensearch_indexer.read_relationship_descriptions(
                [r.id for r in new_relationships], suffix=suffix
            ),
        }

        def unchanged(item: Entity | Relationship) -> bool:
            return graph_copy.get(item.id) == _vector_content(item) and (
                vector_text.get(item.id) or None
            ) == (item.description or None)

        # Written so an interruption never hides an edge still citing a
        # removed text unit: a retry finds the edges through the entities
        # that cite it, so those are rewritten last. A re-keyed entity's new
        # vertex comes first (its edges are re-pointed at it); its old
        # vertex still cites the text units until it is deleted.
        rekeyed = [e for e in new_entities if e.id not in graph_copy]
        kept = [e for e in new_entities if e.id in graph_copy]
        relationship_tasks = [
            IndexingTask(
                self.neptune_indexer.upsert_relationships,
                [new_relationships],
                "neptune_relationships",
            )
        ]
        if self.config.indexing.opensearch.build_relationship_vector_index:
            relationship_tasks += [
                IndexingTask(
                    self.opensearch_indexer.update_relationship_lineage,
                    [[r for r in new_relationships if unchanged(r)]],
                    "opensearch_relationship_lineage",
                ),
                IndexingTask(
                    self._upsert_vector_relationships,
                    [[r for r in new_relationships if not unchanged(r)]],
                    "opensearch_relationships",
                ),
            ]
        phases = [
            [
                IndexingTask(
                    self.neptune_indexer.upsert_entities,
                    [rekeyed],
                    "neptune_rekeyed_entities",
                )
            ],
            relationship_tasks,
            [
                IndexingTask(
                    self.neptune_indexer.upsert_entities, [kept], "neptune_entities"
                ),
                IndexingTask(
                    self.opensearch_indexer.update_entity_lineage,
                    [[e for e in new_entities if unchanged(e)]],
                    "opensearch_entity_lineage",
                ),
                IndexingTask(
                    self.opensearch_indexer.upsert_entities,
                    [[e for e in new_entities if not unchanged(e)]],
                    "opensearch_entities",
                ),
            ],
        ]
        for phase in phases:
            check_write_fence()
            results.update(self._run_indexing_phase(phase))
        return results, noted

    def _remove_text_units_from_claims(
        self, suffix: str, claim_ids: list[str], text_unit_ids: list[str]
    ) -> dict[str, IndexingStats]:
        try:
            claims = self.opensearch_indexer.read_claims(claim_ids, suffix=suffix)
            # The write-back groups by the claim's own index tag, so tag what
            # was read with the index it came from.
            for claim in claims:
                claim.attributes = {
                    **(claim.attributes or {}),
                    Constants.INDEX.value: suffix,
                }
            # The claims citing none of them were read because their subject
            # or object was re-keyed: they are derived again too.
            stripped = {c.id: c for c in remove_claim_text_units(claims, text_unit_ids)}
            claims = [stripped.get(c.id, c) for c in claims]
            if not claims:
                return {}
            derived = self._derive_claims(suffix, claims, removed=text_unit_ids)
        except Exception as e:  # noqa: BLE001 - reported as failed items
            logger.error("Reading shared claims in '%s' failed: %s", suffix, e)
            stats = IndexingStats(total_items=len(claim_ids))
            stats.add_error(str(e), count=len(claim_ids), ids=claim_ids)
            return {f"opensearch_read_claims_{suffix}": stats}
        before = {c.id: c for c in claims}
        lineage_only = [
            d
            for d in derived
            if d.id in before and _claim_text(d) == _claim_text(before[d.id])
        ]
        kept = {d.id for d in lineage_only}
        changed = [d for d in derived if d.id not in kept]
        results: dict[str, IndexingStats] = {}
        if lineage_only:
            results[f"opensearch_claims_{suffix}"] = (
                self.opensearch_indexer.update_claim_lineage(lineage_only)
            )
        if changed:
            results[f"opensearch_claim_content_{suffix}"] = (
                self.opensearch_indexer.upsert_claims(changed)
            )
        return results

    def _claims_about(self, suffix: str, entity_ids: Iterable[str]) -> list[str]:
        """Ids of the stored claims whose subject or object is one of
        ``entity_ids``."""
        ids = sorted(set(entity_ids))
        if not ids:
            return []
        prefix = self.config.indexing.opensearch.claims_index_prefix
        found: set[str] = set()
        for field in ("subject_id", "object_id"):
            found.update(
                self.opensearch_indexer.find_documents(prefix, field, ids, suffix) or {}
            )
        return sorted(found)

    def _derive_claims(
        self,
        suffix: str,
        claims: list[Claim],
        text_units: Sequence[TextUnit] = (),
        removed: Sequence[str] = (),
    ) -> list[Claim]:
        """``claims`` with their content derived from their text units'
        records, resolved against their subject and object entities: as this
        run derived them, else as stored (stripped of ``removed`` text
        units). A claim whose derived id differs is retired (see
        :meth:`delete_retired_ids`)."""
        rekeyed = self._rekeyed.get(suffix, {})
        split_from = self._split_from.get(suffix, {})
        derived_entities = self._derived_entities.get(suffix, {})
        named = {i for c in claims for i in (c.subject_id, c.object_id) if i}
        entity_ids = {rekeyed.get(i, i) for i in named} | {
            part for part, origin in split_from.items() if origin in named
        }
        entities = self.neptune_indexer.read_entities(
            sorted(entity_ids - set(derived_entities)), suffix=suffix
        )
        if removed:
            stripped = {e.id: e for e in remove_text_units(entities, [], removed)[0]}
            entities = [stripped.get(e.id, e) for e in entities]
        entities += [
            derived_entities[i] for i in sorted(entity_ids) if i in derived_entities
        ]
        # The other spellings of their names this run derived (not stored
        # yet): their text units' records decide, with the claims' own,
        # which spelling a claim's subject and object resolve to.
        keys = {compact_name_key(e.name) for e in entities}
        given = {e.id for e in entities}
        entities += [
            e
            for i, e in sorted(derived_entities.items())
            if i not in given and compact_name_key(e.name) in keys
        ]
        derived = self.derive(
            suffix,
            entities,
            [],
            claims,
            text_units,
            rekeyed=rekeyed,
            split_from=split_from,
        )
        claim_ids = {c.id for c in claims}
        self._retire(
            suffix,
            {old: new for old, new in derived.id_remap.items() if old in claim_ids},
            dropped=[i for i in derived.dropped if i in claim_ids],
        )
        return derived.claims

    def remove_text_units(
        self, suffix: str, text_unit_ids: list[str]
    ) -> dict[str, IndexingStats]:
        """Remove removed documents' text units and what only they stated.

        The items citing a removed text unit are found in the stores: the
        entities and communities in the graph store
        (``GraphIndexer.find_ids_citing``), the relationships among their
        incident edges, and the entities, relationships, claims and community
        reports in the vector store (``VectorIndexer.find_documents``). An entity, relationship or claim
        citing only removed text units is deleted with them; one citing others
        too is stripped of them and its content derived again
        (:meth:`remove_text_units_from_shared`). A community or report is
        deleted once none of the text units it cites is stored any more (its
        content is not derived again: see the docs on delta communities).
        The extraction-store keys the removed text units named are kept for
        :meth:`collect_extraction_records`.

        A failed read is reported as failed items, so the caller keeps the
        documents' registry rows and retries the removal.
        """
        removed = set(text_unit_ids)
        graph, vector = self.neptune_indexer, self.opensearch_indexer
        prefixes = self.config.indexing.opensearch
        try:
            found = graph.find_ids_citing(text_unit_ids, suffix=suffix)
            if found is None:
                logger.warning(
                    "The graph store cannot find what text units state: the "
                    "removed documents' entities and relationships stay"
                )
            # The vector store's copies are searched too: an interrupted run
            # may have written one without its graph copy.
            cited = {
                **self._vector_citing(
                    prefixes.entities_index_prefix, text_unit_ids, suffix
                ),
                **(found or {}),
            }
            # The edges citing them are among the cited entities' edges,
            # except after an earlier removal of these units that got past
            # deleting them (their vector copies are gone) and died: it may
            # have stripped the entities and re-keyed an edge whose old graph
            # copy it did not get to delete. Only then are the graph's edges
            # looked up by text unit (on Neptune a scan of the suffix's edges).
            still = vector.read_text_unit_ids(text_unit_ids, suffix=suffix)
            resumed = still is None or len(set(still)) < len(removed)
            edges = (
                graph.find_relationship_ids_citing(text_unit_ids, suffix=suffix)
                if resumed
                else None
            )
            if edges is None:
                incident = (
                    graph.find_incident_relationship_ids(sorted(cited), suffix=suffix)
                    if cited
                    else []
                )
                edges = {
                    r.id: list(r.text_unit_ids or [])
                    for r in graph.read_relationships(incident, suffix=suffix)
                    if removed.intersection(r.text_unit_ids or [])
                }
            relationships = {
                **self._vector_citing(
                    prefixes.relationships_index_prefix, text_unit_ids, suffix
                ),
                **edges,
            }
            communities = (
                graph.find_ids_citing(text_unit_ids, suffix=suffix, communities=True)
                or {}
            )
            claims = self._vector_citing(
                prefixes.claims_index_prefix, text_unit_ids, suffix
            )
            reports = self._vector_citing(
                prefixes.community_reports_index_prefix, text_unit_ids, suffix
            )
            other_units = sorted(
                {
                    unit_id
                    for units in (*communities.values(), *reports.values())
                    for unit_id in units
                }
                - removed
            )
            stored = vector.read_text_unit_ids(other_units, suffix=suffix)
            # A backend that cannot tell keeps every community still citing
            # other text units.
            live = set(other_units if stored is None else stored)
            self._removed_record_keys.setdefault(suffix, set()).update(
                key
                for attributes in vector.read_text_unit_attributes(
                    text_unit_ids, suffix=suffix
                ).values()
                for key in attribute_record_keys(attributes).values()
            )
            # What an earlier strip merged or re-keyed and may not have
            # finished (see _note_pending_strip): the items need not cite the
            # removed units any more. Read on every removal (one record): a
            # retry need not remove the same units.
            notes = vector.read_extraction_records([_PENDING_STRIP], suffix=suffix)
            # A claim the earlier strip re-keyed onto a noted entity cites no
            # removed unit: it is found through its subject or object.
            noted_claims = self._claims_about(suffix, _noted(notes, "entity_ids"))
        except Exception as e:  # noqa: BLE001 - reported as failed items
            logger.error("Finding the removed text units' items failed: %s", e)
            stats = IndexingStats(total_items=len(text_unit_ids))
            stats.add_error(str(e), count=stats.total_items)
            return {f"find_removed_{suffix}": stats}

        def exclusive(units: Iterable[str]) -> bool:
            return set(units) <= removed

        def gone(units: Iterable[str]) -> bool:
            return not (set(units) - removed) & live

        outdated = sorted(i for i, units in reports.items() if not gone(units))
        if outdated:
            logger.warning(
                "%d community reports in '%s' cite removed text units and other "
                "ones: they are kept, and may state the removed text until "
                "indexing.reset: true rebuilds them",
                len(outdated),
                suffix,
            )
            self.outdated_reports.update(outdated)

        shared_entities = sorted(
            i for i, units in cited.items() if not exclusive(units)
        )
        shared_relationships = sorted(
            i for i, units in relationships.items() if not exclusive(units)
        )
        shared_claims = sorted(i for i, units in claims.items() if not exclusive(units))
        doomed = sorted(
            {
                *text_unit_ids,
                *(i for i, units in cited.items() if exclusive(units)),
                *(i for i, units in relationships.items() if exclusive(units)),
                *(i for i, units in claims.items() if exclusive(units)),
                *(i for i, units in communities.items() if gone(units)),
                *(i for i, units in reports.items() if gone(units)),
            }
        )
        if notes:
            shared_entities = sorted(
                (set(shared_entities) | _noted(notes, "entity_ids")) - set(doomed)
            )
            shared_relationships = sorted(
                (set(shared_relationships) | _noted(notes, "relationship_ids"))
                - set(doomed)
            )
            shared_claims = sorted(
                (set(shared_claims) | set(noted_claims)) - set(doomed)
            )
        logger.info(
            "Removing %d artifacts of %d text units in '%s'; stripping %d "
            "shared entities, %d relationships and %d claims",
            len(doomed),
            len(text_unit_ids),
            suffix,
            len(shared_entities),
            len(shared_relationships),
            len(shared_claims),
        )
        results = self.delete_documents({suffix: doomed})
        if any(stats.failed_items for stats in results.values() if stats):
            # Strip nothing until the removed text units are gone: a retry
            # tells a removal that got further by their missing copies.
            return results
        deleted_communities = [i for i, u in communities.items() if gone(u)]
        if deleted_communities:
            # A kept text unit may still name one: a later delta wrote the
            # community anew under its id without that unit.
            results[f"opensearch_text_unit_communities_{suffix}"] = (
                vector.prune_text_unit_communities(deleted_communities, suffix)
            )
        if shared_entities or shared_relationships or shared_claims:
            results.update(
                self.remove_text_units_from_shared(
                    suffix,
                    entity_ids=shared_entities,
                    relationship_ids=shared_relationships,
                    text_unit_ids=text_unit_ids,
                    claim_ids=shared_claims,
                    community_ids=deleted_communities,
                    earlier_notes=notes,
                )
            )
        elif notes and not any(s.failed_items for s in results.values() if s):
            # The noted items are gone: the earlier strip is finished.
            self._clear_pending_strip(suffix, notes)
        return results

    def _vector_citing(
        self, alias_prefix: str, text_unit_ids: list[str], suffix: str
    ) -> dict[str, list[str]]:
        """``{id: text_unit_ids}`` of the documents of ``alias_prefix`` citing
        any of ``text_unit_ids`` (``{}`` for a backend that cannot tell)."""
        return (
            self.opensearch_indexer.find_documents(
                alias_prefix, "text_unit_ids", text_unit_ids, suffix
            )
            or {}
        )

    def collect_extraction_records(self) -> dict[str, IndexingStats]:
        """Delete the extraction records the removed text units named that no
        stored text unit names any more (see :meth:`remove_text_units`).

        Run after the delta is written: a changed document's unchanged chunk
        names the same record as its removed copy. Best-effort: a failure
        leaves records no text unit names, which cost space only.
        """
        results: dict[str, IndexingStats] = {}
        vector = self.opensearch_indexer
        prefix = self.config.indexing.opensearch.text_units_index_prefix
        for suffix, keys in sorted(self._removed_record_keys.items()):
            try:
                naming: set[str] = set()
                for kind in ("graph", "claims"):
                    naming.update(
                        vector.find_documents(
                            prefix,
                            f"attributes.{RECORD_KEYS_ATTRIBUTE}.{kind}",
                            sorted(keys),
                            suffix,
                        )
                        or {}
                    )
                still_named = {
                    key
                    for attributes in vector.read_text_unit_attributes(
                        sorted(naming), suffix=suffix
                    ).values()
                    for key in attribute_record_keys(attributes).values()
                }
                unused = sorted(keys - still_named)
                if unused:
                    logger.info(
                        "Deleting %d extraction records no text unit of '%s' "
                        "names any more",
                        len(unused),
                        suffix,
                    )
                    results[f"extraction_records_{suffix}"] = (
                        vector.delete_extraction_records(unused, suffix=suffix)
                    )
            except NamespaceLeaseError:
                raise
            except Exception as e:  # noqa: BLE001 - garbage only costs space
                logger.warning(
                    "Collecting the unused extraction records of '%s' failed: %s",
                    suffix,
                    e,
                )
        self._removed_record_keys.clear()
        return results

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
            check_write_fence()
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
            # The vector copies first: a retry finds what to delete (and the
            # re-keyed items it retires) through the graph, so a run that
            # died between the stores must leave the graph copy, not an
            # orphaned vector copy nothing finds again.
            results.update(
                self.opensearch_indexer.delete_document_artifacts(
                    ids, suffix, extra_relationship_ids=incident_rel_ids
                )
            )
            results[f"neptune_delete_{suffix}"] = self.neptune_indexer.delete_by_id(
                ids, suffix=suffix
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
                except NamespaceLeaseError:
                    raise
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
