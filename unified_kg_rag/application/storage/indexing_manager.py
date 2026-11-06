# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import as_completed
from functools import partial
from typing import Any

from unified_kg_rag.adapters.ingestion.description_summarizer import (
    DescriptionSummarizer,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.bootstrap import resolve_extraction_schema
from unified_kg_rag.application.storage.cross_run_merge import (
    CrossRunMerge,
    CrossRunMergeResult,
)
from unified_kg_rag.application.storage.reannotation import (
    SchemaReannotation,
    SchemaReannotator,
)
from unified_kg_rag.application.storage.removal import TextUnitRemoval
from unified_kg_rag.application.storage.stored_items import (
    _PENDING_COMMIT,
    _STRIPPED_SUMMARIES,
    IndexingTask,
    _annotations,
    _Described,
    _note_ids,
    _with_sources,
    stripped_descriptions,
    unit_records,
)
from unified_kg_rag.domain.ingestion.community_hierarchy import attach_report_hierarchy
from unified_kg_rag.domain.ingestion.contributions import (
    REPORT_RECORD_ATTRIBUTE,
    record_keys,
)
from unified_kg_rag.domain.ingestion.derivation import Derived
from unified_kg_rag.domain.ingestion.methodologies import artifacts_to_build
from unified_kg_rag.domain.ingestion.schema_annotation import SchemaAnnotator
from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Config,
    Constants,
    Entity,
    IndexArtifact,
    Relationship,
    TextUnit,
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

__all__ = [
    "CrossRunMergeResult",
    "IndexingManager",
    "IndexingTask",
    "SchemaReannotation",
    "stripped_descriptions",
    "unit_records",
]


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

    The manager writes, and holds the run state; three collaborators do the
    rest over its stores and that state, behind its public methods: the
    cross-run merge and derivation (:mod:`.cross_run_merge`), the removal
    of text units (:mod:`.removal`) and schema re-annotation
    (:mod:`.reannotation`).
    """

    def __init__(
        self,
        config: Config,
        *,
        vector_indexer: VectorIndexer | None = None,
        graph_indexer: GraphIndexer | None = None,
        providers: Providers | None = None,
    ) -> None:
        # The schema a schema source names, loaded once (the pipeline loads
        # it before it builds the manager; then this changes nothing).
        config = resolve_extraction_schema(config, providers=providers)
        self.config = config
        # Session, embedding/LLM providers and store builders shared with the
        # pipeline, used for the default indexers and the merge summarizer.
        self.providers = Providers.resolve(config, providers)
        # The attribute names predate the store-agnostic ports; they hold the
        # vector and graph writers whatever their backend.
        self.vector_indexer: VectorIndexer = (
            vector_indexer or self.providers.build_vector_indexer()
        )
        self.graph_indexer: GraphIndexer = (
            graph_indexer or self.providers.build_graph_indexer()
        )
        # Built lazily on first cross-run merge so a manager only used for
        # indexing (no incremental read-back) never constructs a Bedrock client.
        self._description_summarizer: DescriptionSummarizer | None = None
        # The schema annotations of every entity and relationship written,
        # from its stored type.
        self._annotator = SchemaAnnotator(
            config.processing.graph_extraction.extraction_schema()
        )
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
        # Per suffix: the retired entities only the vector store held (see
        # _strip_shared), whose edges delete_documents finds there.
        self._vector_only: dict[str, set[str]] = {}
        # Per suffix: the entities a commit's derivation split, re-keyed
        # (old and new ids) or rewired, for its write-ahead note.
        self._commit_entities: dict[str, set[str]] = {}
        # Per suffix: the entities an earlier commit's note names (see
        # note_pending_commit); and the suffixes whose commit note this run
        # wrote or read (clear_pending_commit deletes those).
        self._noted_commit_entities: dict[str, set[str]] = {}
        self._commit_note_suffixes: set[str] = set()
        # The stored community reports a removal kept although they cite a
        # removed text unit (see community_drift).
        self.outdated_reports: set[str] = set()
        # By suffix, the stored communities this run's removals deleted
        # (the build record's community counters subtract them).
        self.deleted_communities: dict[str, set[str]] = {}
        # Stored items derivation kept as they were: a text unit they cite
        # names no stored record (``Derived.underived``).
        self.underived_items = 0
        # By suffix, the entities this run's graph merge or strip derived (to
        # be written): the claims about them are derived against these.
        self._derived_entities: dict[str, dict[str, Entity]] = {}
        # By suffix, the summaries this run made before indexing (see
        # offer_summaries): reused as the stored ones are.
        self._offered_summaries: dict[str, list[Entity]] = {}
        # By suffix, the ids of the items this run's removals summarized or
        # wrote unsummarized (left to the commit, or another spelling folded
        # in): the stored summary, if any, is not of what the strip left, so
        # the commit summarizes such an item from its records rather than
        # extending that summary (see _merge_suffix). Not a summary a removal
        # reused (community detection's, made for the report the run stores).
        # Noted in the extraction store before the strip writes them, for a
        # commit of a later run if this one dies before its own.
        self._stripped_ids: dict[str, set[str]] = {}
        # By suffix, the ids the stored summary note lists (see
        # note_stripped_summaries), once read or written by this run: a
        # removal that dies before the commit leaves them to its successor.
        self._noted_stripped_ids: dict[str, set[str]] = {}
        # By suffix, the ids of the run's delta items (see expect_delta): a
        # removal that derives one again leaves its summary to the commit.
        self._delta_ids: dict[str, set[str]] = {}
        # The collaborators the public methods delegate to; they read and
        # write the stores and the run state above through this manager.
        self._merge = CrossRunMerge(self)
        self._removal = TextUnitRemoval(self)
        self._reannotation = SchemaReannotator(self)

    @property
    def description_summarizer(self) -> DescriptionSummarizer:
        if self._description_summarizer is None:
            self._description_summarizer = DescriptionSummarizer(
                self.config, providers=self.providers
            )
        return self._description_summarizer

    def close(self) -> None:
        """Close both indexers' backing clients (best-effort, never raises)."""
        for indexer in (self.vector_indexer, self.graph_indexer):
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
        self.vector_indexer.initialize()
        self.graph_indexer.initialize()

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
        written = results.get("vector_text_units")
        if written is not None and not written.failed_items:
            self._retain_extraction_records(
                text_units or [], suffixes, community_reports or []
            )
            # The summaries of text the rebuild no longer holds go: a summary
            # tree, if any, is written again by the summary_tree stage. A
            # rebuild without raptor deletes the tree whole, as it replaces
            # the indices of the other artifacts it no longer builds.
            units = self._group_by_suffix(text_units or [])
            keeps_tree = IndexArtifact.SUMMARY_TREE in artifacts_to_build(self.config)
            for suffix in sorted(suffixes):
                summaries = (
                    self.vector_indexer.delete_summary_nodes_outside(
                        [unit.id for unit in units.get(suffix, [])], suffix
                    )
                    if keeps_tree
                    else self.vector_indexer.delete_summary_tree(suffix)
                )
                if summaries.total_items or summaries.failed_items:
                    results[f"vector_summary_nodes_{suffix}"] = summaries
        return results

    def _retain_extraction_records(
        self,
        text_units: list[TextUnit],
        suffixes: list[str],
        community_reports: Sequence[CommunityReport] = (),
    ) -> None:
        """Keep, per rebuilt suffix, only the extraction records its stored
        text units and community reports name (``REPORT_RECORD_ATTRIBUTE``):
        the records of other extraction settings, and of texts no stored
        unit holds any more (deleted or edited documents), go. Best-effort: a
        failure keeps the records, which cost space only.
        """
        named: dict[str, set[str]] = {suffix: set() for suffix in suffixes}
        for unit in text_units:
            named.setdefault(BaseIndexer.get_suffix(unit), set()).update(
                record_keys(unit).values()
            )
        for report in community_reports:
            key = (report.attributes or {}).get(REPORT_RECORD_ATTRIBUTE)
            if isinstance(key, str):
                named.setdefault(BaseIndexer.get_suffix(report), set()).add(key)
        for suffix, keys in sorted(named.items()):
            try:
                kept = self.vector_indexer.retain_extraction_records(
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

    def _annotated_entities(self, entities: Iterable[Entity]) -> list[Entity]:
        """``entities`` annotated as they are written: the schema annotations
        and ``source_kinds``."""
        return [_with_sources(e) for e in self._annotator.entities(entities)]

    def _annotated_relationships(
        self, relationships: Iterable[Relationship]
    ) -> list[Relationship]:
        """``relationships`` annotated as they are written (see
        :meth:`_annotated_entities`)."""
        return [_with_sources(r) for r in self._annotator.relationships(relationships)]

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
        if entities is not None:
            entities = self._annotated_entities(entities)
        if relationships is not None:
            relationships = self._annotated_relationships(relationships)
        vector, graph = self.vector_indexer, self.graph_indexer
        # (full-build writer, delta writer, items, result key)
        phase1: list[
            tuple[Callable[..., Any], Callable[..., Any], list[Any] | None, str]
        ] = [
            (
                vector.index_text_units,
                vector.upsert_text_units,
                text_units,
                "vector_text_units",
            ),
            (
                vector.index_entities,
                vector.upsert_entities,
                entities,
                "vector_entities",
            ),
            (
                vector.index_community_reports,
                vector.upsert_community_reports,
                community_reports,
                "vector_community_reports",
            ),
            (vector.index_claims, vector.upsert_claims, claims, "vector_claims"),
            (graph.index_entities, graph.upsert_entities, entities, "graph_entities"),
        ]
        # The OpenSearch relationship VECTOR index: written when the run
        # builds it (its switch, a methodology reading it); otherwise a full
        # build only empties a live one. The Neptune relationship EDGES of
        # phase 2 are always written: every graph methodology traverses them.
        phase1.append(
            (
                vector.index_relationships,
                self._upsert_vector_relationships,
                (
                    relationships
                    if IndexArtifact.RELATIONSHIP_VECTORS
                    in artifacts_to_build(self.config)
                    else None
                ),
                "vector_relationships",
            )
        )

        def tasks(
            specs: list[
                tuple[Callable[..., Any], Callable[..., Any], list[Any] | None, str]
            ],
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
                relationships, results.get("graph_entities")
            )
            phase2 = [
                IndexingTask(edges, [writable], "graph_relationships"),
                *tasks(
                    [
                        (
                            graph.index_communities,
                            graph.upsert_communities,
                            communities,
                            "graph_communities",
                        )
                    ]
                ),
            ]
            logger.info("--- Starting %sIndexing Phase 2%s ---", prefix, tail)
            results.update(self._run_indexing_phase(phase2))
            check_write_fence()
            if skipped is not None:
                results.setdefault("graph_relationships", IndexingStats()).merge(
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
                if key.startswith("graph_")
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
                results["graph_sweep"] = skipped
            try:
                graph.end_rebuild(suffixes or [], sweep=not graph_failed)
            except Exception as e:  # noqa: BLE001 - reported as a failed write
                logger.error("Dropping the previous graph failed: %s", e)
                stats = IndexingStats(total_items=1)
                stats.add_error(str(e))
                results["graph_sweep"] = stats

        self._log_completion_summary(results, time.time() - start_time)
        return results

    def _upsert_vector_relationships(
        self, relationships: list[Relationship]
    ) -> IndexingStats:
        """Delta upsert of the relationship vectors. An edge this run rewrites
        for its rank only is updated in place (no embedding), unless its
        vector copy is missing (an interrupted write): that one is upserted."""
        rank_only, self._rank_only = self._rank_only, set()
        vector = self.vector_indexer
        candidates = [r for r in relationships if r.id in rank_only]
        if not candidates:
            return vector.upsert_relationships(relationships)
        stored: dict[str, Mapping[str, Any]] = {}
        for suffix, group in self._group_by_suffix(candidates).items():
            stored |= vector.read_vector_copies(
                "relationships", [r.id for r in group], suffix=suffix
            )
        # A copy whose schema annotations differ is written in full too.
        partial = [
            r
            for r in candidates
            if r.id in stored
            and all(stored[r.id].get(k) == v for k, v in _annotations(r).items())
        ]
        kept = {r.id for r in partial}
        full = [r for r in relationships if r.id not in kept]
        stats = vector.update_relationship_lineage(partial)
        if full:
            stats.merge(vector.upsert_relationships(full))
        stats.total_items = len(relationships)
        return stats

    @property
    def annotation_digest(self) -> str:
        """The digest of the annotations this manager writes
        (``SchemaAnnotator.digest``)."""
        return self._annotator.digest

    def stores_hold_items(self, suffix: str) -> bool:
        """Whether either store holds an entity or relationship of
        ``suffix`` (``holds_any_item``, a bounded read each). A store that
        cannot tell counts as empty, and so do stores :meth:`reannotate`
        cannot rewrite (a pass would read nothing). A read that fails
        raises."""
        return self._reannotation.stores_hold_items(suffix)

    def reannotate(self, suffixes: Collection[str]) -> SchemaReannotation:
        """Rewrite the schema annotations of every stored entity and
        relationship of ``suffixes`` (every scope's) to what the current
        schema gives their stored types, and nothing else: no model call,
        no embedding.

        An incremental run calls it when the stores of its namespace may
        hold other annotations than the run's schema gives
        (``incremental.reannotate_on_schema_change``): the extraction caches
        still hold, so no document is extracted again, and afterwards the
        stores hold what a full build would write. Each store is read in
        bounded pages, annotated from its own type (a relationship's vector
        copy from its edge's, as an older copy stores none; one without an
        edge is left as it is), and only the annotations that differ are written.

        Nothing is read unless both stores can enumerate and rewrite their
        items (``rewrites_schema_annotations``): the result is then
        ``unsupported``, and a reset rewrites them. A store read or write
        that raises stops the pass with ``error`` set instead of failing the
        run (the next run tries again); a lost namespace lease still
        propagates.
        """
        return self._reannotation.reannotate(suffixes)

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
        return self._merge.merge_with_existing_graph(
            entities, relationships, text_units
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
        return self._merge.merge_with_stored_claims(claims, text_units)

    def _read_stored_claims(self, ids: list[str], suffix: str) -> list[Claim]:
        """The stored claims of ``ids`` in ``suffix``, tagged with it (the
        write-back groups claims by their own index tag)."""
        claims = self.vector_indexer.read_claims(ids, suffix=suffix) if ids else []
        for claim in claims:
            claim.attributes = {
                **(claim.attributes or {}),
                Constants.INDEX.value: suffix,
            }
        return claims

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
        return self._merge.derive(
            suffix, entities, relationships, claims, text_units, rekeyed, split_from
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
        vector_only = dict(self._vector_only)
        self._reset_run_state()
        if retired:
            logger.info(
                "Deleting %d items re-keyed to follow their current name",
                sum(len(ids) for ids in retired.values()),
            )
        return self.delete_documents(retired, vector_only)

    def note_pending_commit(self) -> None:
        """Note, per suffix, the stored entities this commit's derivation
        split, re-keyed or rewired (with those of an earlier note it
        finishes), before its first write. Those decisions live in memory
        only: a run killed between the writes leaves the stored state half
        changed, and the next run, deriving from it, would not split or
        re-key them again, so their stored edges and claims would stay as
        they were. The next commit of the suffix derives them again
        (:meth:`_merge_suffix`). Raises when a write fails, before anything
        is written."""
        for suffix in sorted(
            self._commit_entities.keys() | self._noted_commit_entities.keys()
        ):
            ids = {
                *self._commit_entities.get(suffix, set()),
                *self._noted_commit_entities.get(suffix, set()),
            }
            if not ids:
                continue
            self._write_note(_PENDING_COMMIT, suffix, {"entity_ids": sorted(ids)})
            self._commit_note_suffixes.add(suffix)

    def clear_pending_commit(self) -> None:
        """Delete the commit notes of the suffixes this run noted or read
        (best-effort: one left only makes the next commit derive the
        entities again)."""
        for suffix in sorted(self._commit_note_suffixes):
            self._delete_note(_PENDING_COMMIT, suffix)
        self._commit_note_suffixes.clear()

    def read_stripped_summaries(
        self, suffix: str, records: Mapping[str, Any] | None = None
    ) -> None:
        """Add the ids of the stored summary note of ``suffix`` (see
        :meth:`note_stripped_summaries`) to the items this run's commit does
        not extend; read once per run (from ``records``, the extraction
        records a caller read with it, when given). A failed read raises."""
        if suffix in self._noted_stripped_ids:
            return
        if records is None:
            records = self.vector_indexer.read_extraction_records(
                [_STRIPPED_SUMMARIES], suffix=suffix
            )
        ids = _note_ids(records.get(_STRIPPED_SUMMARIES), "ids")
        self._noted_stripped_ids[suffix] = ids
        if ids:
            self._stripped_ids.setdefault(suffix, set()).update(ids)

    def note_stripped_summaries(self, suffix: str) -> None:
        """Note the items of ``suffix`` whose stored summary the commit must
        not extend (``_stripped_ids``, with those of a stored note), before a
        strip writes them. They live in memory only: a run that dies between
        the strip and the commit leaves the stored copies stripped, and its
        successor's removal finds nothing left to strip, so its commit
        would extend them (a summary of a summary, or kept removed lines).
        Raises when the write fails, before the strip writes anything."""
        self.read_stripped_summaries(suffix)
        ids = self._stripped_ids.get(suffix, set())
        if ids <= self._noted_stripped_ids[suffix]:
            return
        self._write_note(_STRIPPED_SUMMARIES, suffix, {"ids": sorted(ids)})
        self._noted_stripped_ids[suffix] = set(ids)

    def clear_stripped_summaries(self) -> None:
        """Delete the summary notes this run read or wrote, once its
        commit's writes succeeded (best-effort: one left only makes a later
        commit summarize its items from their records, as a full build)."""
        for suffix, ids in sorted(self._noted_stripped_ids.items()):
            if ids:
                self._delete_note(_STRIPPED_SUMMARIES, suffix)
        self._noted_stripped_ids.clear()

    def discard_retired_ids(self) -> None:
        """Keep the re-keyed items' old ids (their writes failed, or the
        commit did not get to them: deleting them would lose the items), and
        the stored notes (a retry finishes from them)."""
        self._reset_run_state()
        self._commit_note_suffixes.clear()

    def _reset_run_state(self) -> None:
        """Forget what this run's derivations decided: the re-keyed, split,
        renamed, dropped and vector-only ids, the derived entities, the
        commit's noted entities. Held in memory between a merge and its
        writes only; a failed commit's, kept, would make the next run of a
        reused pipeline delete old ids whose replacements were never
        written."""
        for state in (
            self._rekeyed,
            self._dropped,
            self._vector_only,
            self._commit_entities,
            self._noted_commit_entities,
            self._split_from,
            self._renamed,
            self._fresh,
            self._rank_only,
            self._derived_entities,
        ):
            state.clear()

    def reset_run_state(self) -> None:
        """Start a run with no state left from an earlier one: what its
        derivations decided (see :meth:`discard_retired_ids`), and what its
        removals found for after the commit, the community reports they kept
        the communities they deleted, the extraction-record keys of their
        text units and the items they derived again (an earlier run that
        failed before its commit measured, counted or collected them)."""
        self.discard_retired_ids()
        self.outdated_reports.clear()
        self.deleted_communities.clear()
        self._removed_record_keys.clear()
        self._offered_summaries.clear()
        self._stripped_ids.clear()
        self._noted_stripped_ids.clear()
        self._delta_ids.clear()

    def expect_delta(
        self, entities: Sequence[Entity], relationships: Sequence[Relationship]
    ) -> None:
        """The items the run's commit derives (its delta), told before the
        removals: one a removal derives again under the same id is derived
        once more by the commit, with the delta's lines, so the removal
        writes its description unsummarized and the commit summarizes it
        (one call, from every line its text units state, as a full build).
        Grouped by their index suffix."""
        for suffix, group in self._group_by_suffix([*entities, *relationships]).items():
            self._delta_ids.setdefault(suffix, set()).update(i.id for i in group)

    def offer_summaries(self, entities: Sequence[Entity]) -> None:
        """Summaries this run made before indexing (community detection's
        of its stored members, ``PipelineContext.member_summaries``), each
        marked with the description it was made from: an entity derived
        with that description takes the summary (``reuse_summaries``) rather
        than being summarized again. Grouped by their index suffix."""
        for suffix, group in self._group_by_suffix(list(entities)).items():
            self._offered_summaries.setdefault(suffix, []).extend(group)

    def _write_note(self, key: str, suffix: str, note: dict[str, Any]) -> None:
        """Write the note ``key`` (one extraction-store record of ``suffix``).
        Raises when the write fails: the caller writes nothing before it."""
        stats = self.vector_indexer.write_extraction_records({key: note}, suffix=suffix)
        if stats.failed_items:
            raise DataProcessingError(
                f"Writing the note '{key}' of '{suffix}' failed: {stats.errors[:1]}"
            )

    def _delete_note(self, key: str, suffix: str) -> None:
        """Delete the note ``key`` of ``suffix`` (best-effort: one left only
        makes a later run derive its items again)."""
        try:
            self.vector_indexer.delete_extraction_records([key], suffix=suffix)
        except NamespaceLeaseError:
            raise
        except Exception as e:  # noqa: BLE001 - a leftover note is harmless
            logger.warning("Deleting the note '%s' of '%s' failed: %s", key, suffix, e)

    def remove_text_units_from_shared(
        self,
        suffix: str,
        *,
        entity_ids: list[str],
        relationship_ids: list[str],
        text_unit_ids: list[str],
        claim_ids: Sequence[str] = (),
        community_ids: Collection[str] = (),
        earlier_note: Mapping[str, Any] | None = None,
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
        (:meth:`_note_pending_strip`), with those of the ``earlier_note`` of
        a strip a retry finishes; the note is deleted once the old ids are.
        """
        return self._removal.remove_text_units_from_shared(
            suffix,
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
            text_unit_ids=text_unit_ids,
            claim_ids=claim_ids,
            community_ids=community_ids,
            earlier_note=earlier_note,
        )

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
                self.vector_indexer.find_documents(prefix, field, ids, suffix) or {}
            )
        return sorted(found)

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
        return self._removal.remove_text_units(suffix, text_unit_ids)

    def _vector_incident_relationship_ids(
        self, entity_ids: list[str], suffix: str
    ) -> list[str]:
        """Ids of the relationship documents of ``suffix`` with an endpoint
        among ``entity_ids`` (``[]`` for a backend that cannot tell)."""
        if not entity_ids:
            return []
        prefix = self.config.indexing.opensearch.relationships_index_prefix
        found: set[str] = set()
        for field in ("source_id", "target_id"):
            found.update(
                self.vector_indexer.find_documents(prefix, field, entity_ids, suffix)
                or {}
            )
        return sorted(found)

    def _vector_citing(
        self, alias_prefix: str, text_unit_ids: list[str], suffix: str
    ) -> dict[str, list[str]]:
        """``{id: text_unit_ids}`` of the documents of ``alias_prefix`` citing
        any of ``text_unit_ids`` (``{}`` for a backend that cannot tell)."""
        return (
            self.vector_indexer.find_documents(
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
        return self._removal.collect_extraction_records()

    def delete_documents(
        self,
        ids_by_suffix: dict[str, list[str]],
        vector_only: Mapping[str, Collection[str]] | None = None,
    ) -> dict[str, IndexingStats]:
        """Delete artifacts for removed documents from both stores by id.

        ``ids_by_suffix`` maps a tenant/version suffix to the entity/relationship/
        text-unit/community ids that belonged only to deleted documents;
        ``vector_only`` (by suffix) the entities among them that only the
        vector store holds, whose edges are found in its relationship index.
        """
        return self._removal.delete_documents(ids_by_suffix, vector_only)

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
                except Exception as e:  # noqa: BLE001 - reported as a failed task
                    logger.error("Task '%s' failed: %s", task_name, e, exc_info=True)
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
