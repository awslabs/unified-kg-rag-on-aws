# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental-indexing orchestrator.

Ties together the M2 pieces around a :class:`DocStatusPort`:

1. detect the corpus delta (new / changed / unchanged / deleted),
2. write ahead: record every new and changed document PENDING, with a
   content hash no document has and the union of its stored and planned
   text units, before any store write (:meth:`IncrementalIndexer.write_ahead`),
3. for deleted (and changed) documents, remove what their text units stated
   from the live stores, found there by the text units every artifact cites
   (``IndexingManager.remove_text_units``),
4. upsert the freshly extracted delta artifacts (idempotent),
5. commit: replace the PENDING records with the new content hashes and
   text units.

A run interrupted anywhere after step 2 leaves PENDING records: the next run
classifies those documents changed (re-extracted, even when their content went
back to the indexed version) or, when they are gone, deleted, and removes
everything the interrupted run may have written for them.

Extraction of the delta documents themselves is delegated to the caller (the
existing 12-stage pipeline run on the filtered subset), so this orchestrator
stays storage-focused and is exercised end-to-end with in-memory fakes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from unified_kg_rag.application.ingestion.namespace_lease import check_lease
from unified_kg_rag.domain.ingestion.attempts import (
    closed_attempt,
    opened_attempt,
    settled_attempts,
    unproductive_attempt,
)
from unified_kg_rag.domain.ingestion.claim_resolver import resolved_claim_id
from unified_kg_rag.domain.ingestion.community_drift import (
    CommunityDrift,
    measure_community_drift,
)
from unified_kg_rag.domain.ingestion.delta_detector import (
    assign_registry_source,
    detect_delta,
    document_doc_id,
    filter_documents_to_process,
    fingerprint_documents,
    other_local_scopes,
    registry_scope,
    scope_namespace,
    validate_retired_scopes,
)
from unified_kg_rag.domain.ingestion.merge import merge_claims
from unified_kg_rag.domain.models import (
    PENDING_CONTENT_HASH,
    Claim,
    Community,
    CommunityReport,
    Config,
    Constants,
    DocStatus,
    DocStatusRecord,
    Document,
    DocumentDelta,
    DocumentLineage,
    Entity,
    ExtractionSettings,
    PipelineContext,
    Relationship,
    TextUnit,
)
from unified_kg_rag.ports import DocStatusPort
from unified_kg_rag.ports.indexer import BaseIndexer
from unified_kg_rag.shared import (
    ConfigurationError,
    DataProcessingError,
    DocStatusRegistryError,
    PipelineStageError,
    get_logger,
)
from unified_kg_rag.shared.utils.cache_keys import (
    changed_settings,
    extraction_settings,
)
from unified_kg_rag.shared.utils.document_identity import (
    DEFAULT_NAMESPACE,
    RELATIVE_PATH_KEY,
    compute_doc_id,
    config_registry_namespace,
    normalize_source_path,
    normalize_source_scope,
    relative_source_path,
)

if TYPE_CHECKING:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager
    from unified_kg_rag.ports.indexer import IndexingStats

logger = get_logger(__name__)


def build_document_lineage(
    documents: list[Document],
    text_units: list[TextUnit],
    suffix: str = DEFAULT_NAMESPACE,
) -> list[DocumentLineage]:
    """Each document's text units, keyed by its stable ``doc_id``.

    A text unit belongs to one document (its id derives from the document's
    registry key, see ``assign_registry_source``), and every artifact cites
    the text units it was stated in, so a document's text units are all a
    later run needs to find what it contributed.
    """
    docid_to_stable = {doc.document_id: document_doc_id(doc) for doc in documents}
    units_by_doc: dict[str, list[str]] = defaultdict(list)
    for unit in text_units:
        for document_id in unit.document_ids or []:
            if document_id in docid_to_stable:
                units_by_doc[docid_to_stable[document_id]].append(unit.id)
    return [
        DocumentLineage(
            doc_id=docid_to_stable[doc.document_id],
            suffix=suffix,
            file_path=doc.metadata.get(RELATIVE_PATH_KEY) or doc.file_path,
            text_unit_ids=sorted(set(units_by_doc[docid_to_stable[doc.document_id]])),
        )
        for doc in documents
    ]


def artifact_owners(
    lineages: Sequence[DocumentLineage],
    artifacts: Iterable[Any],
) -> dict[str, set[str]]:
    """``{artifact id: doc ids}``: the documents whose text units each
    artifact (any item with ``text_unit_ids``, or a text unit) cites."""
    owner_of_unit = {
        unit_id: lineage.doc_id
        for lineage in lineages
        for unit_id in lineage.text_unit_ids
    }
    owners: dict[str, set[str]] = defaultdict(set)
    for item in artifacts:
        if item.id in owner_of_unit:
            owners[item.id].add(owner_of_unit[item.id])
        for unit_id in getattr(item, "text_unit_ids", None) or []:
            if unit_id in owner_of_unit:
                owners[item.id].add(owner_of_unit[unit_id])
    return owners


# Indexed documents' text units probed in the stores before a delta run writes
# (IncrementalIndexer.check_graph_holds_registry). Stored records are read in
# batches of _PROBE_DOCUMENTS until that many with text units are found.
_PROBE_DOCUMENTS = 20
_PROBE_UNITS = 100

_EXTRACTION_FAILED = "extraction failed on some text units"
_QUARANTINED_INFO = (
    "quarantined: its processing killed the run {attempts} consecutive runs, "
    "the last one processing it alone (or it produced no text units); skipped "
    "until its content or the extraction settings change (or "
    "indexing.max_document_attempts is raised)"
)
_WRITE_FAILED = "writing some of its artifacts to the stores failed"
_WRITE_AHEAD_INFO = (
    "indexing in progress (write-ahead record); a run that ends with this "
    "record re-extracts the document, or removes its artifacts if it is gone"
)


def _remap_ids(ids: list[str] | None, id_remap: dict[str, str]) -> list[str]:
    return list(dict.fromkeys(id_remap.get(i, i) for i in ids or []))


def _remap_claims(
    claims: list[Claim], id_remap_by_suffix: dict[str, dict[str, str]]
) -> list[Claim]:
    """Point claims' subjects/objects at the entity ids the merge kept.

    A claim's id derives from its subject and object entity ids
    (``resolved_claim_id``), so a remapped claim is re-keyed: it then has the
    id the full build gives it and that a stored claim of the same fact has,
    so ``merge_claims`` folds the two instead of storing a duplicate. Delta
    claims that now share an id are merged.
    """
    by_key: dict[tuple[str, str], Claim] = {}
    for claim in claims:
        suffix = BaseIndexer.get_suffix(claim)
        remap = id_remap_by_suffix.get(suffix, {})
        if remap:
            claim = claim.model_copy(
                update={
                    "subject_id": remap.get(claim.subject_id, claim.subject_id),
                    "object_id": (
                        remap.get(claim.object_id, claim.object_id)
                        if claim.object_id
                        else claim.object_id
                    ),
                }
            )
            claim_id = resolved_claim_id(claim)
            if claim_id != claim.id:
                claim = claim.model_copy(
                    update={"id": claim_id, "short_id": claim_id[:8]}
                )
        key = (suffix, claim.id)
        if key in by_key:
            claim = merge_claims([by_key[key]], [claim])[0]
        by_key[key] = claim
    return list(by_key.values())


def _remap_references(
    communities: list[Community] | None,
    claims: list[Claim] | None,
    id_remap_by_suffix: dict[str, dict[str, str]],
) -> tuple[list[Community] | None, list[Claim] | None]:
    """Point communities' members and claims' subjects/objects at the ids the
    cross-run merge kept, so none references an id that is never written."""
    if not id_remap_by_suffix:
        return communities, claims
    remapped_communities = None
    if communities is not None:
        remapped_communities = []
        for community in communities:
            remap = id_remap_by_suffix.get(BaseIndexer.get_suffix(community), {})
            remapped_communities.append(
                community.model_copy(
                    update={
                        "entity_ids": _remap_ids(community.entity_ids, remap),
                        "relationship_ids": _remap_ids(
                            community.relationship_ids, remap
                        ),
                    }
                )
                if remap
                else community
            )
    if claims is None:
        return remapped_communities, None
    return remapped_communities, _remap_claims(claims, id_remap_by_suffix)


class IncrementalIndexer:
    """Orchestrates an incremental indexing run against a document registry."""

    def __init__(
        self,
        doc_status: DocStatusPort,
        indexing_manager: IndexingManager,
        suffix: str = DEFAULT_NAMESPACE,
        scope: str | None = None,
        extraction_fingerprint: str | None = None,
        extraction_settings: ExtractionSettings | None = None,
    ) -> None:
        self.doc_status = doc_status
        self.indexing_manager = indexing_manager
        self.suffix = suffix
        # Registry scope (delta_detector.registry_scope) the delta is computed
        # in and recorded under; None diffs against the whole registry.
        self.scope = scope
        # The run's extraction fingerprint (run_extraction_fingerprint),
        # recorded on every committed document, and the settings it stands
        # for, recorded once for the scope; None records none.
        self.extraction_fingerprint = extraction_fingerprint
        self.extraction_settings = extraction_settings
        # Write-ahead state of the current run (see write_ahead): the records
        # stored before it, the PENDING records written, and the documents
        # being (re)indexed.
        self._prior: dict[str, DocStatusRecord] = {}
        self._pending: dict[str, DocStatusRecord] = {}
        self._in_flight: set[str] = set()
        # Community drift of the last commit (None without communities).
        self.community_drift: CommunityDrift | None = None

    def check_graph_holds_registry(self, delta: DocumentDelta) -> None:
        """Raise before any write when the stores lost the registry's documents.

        The registry and the stores must describe the same corpus: with a
        registry of indexed documents over an emptied graph (an in-memory
        Gremlin Server restarted, say) or vector store (its database file
        deleted or moved, ``providers.vector.sqlite_path`` changed), the diff
        calls those documents unchanged and they would never be indexed
        again. Probes the text units of a few indexed documents of the run's
        namespace in the vector store (``VectorIndexer.read_text_unit_ids``),
        and the entities citing them in the graph store
        (``GraphIndexer.find_ids_citing``) against those the vector store
        holds; when a store holds none, raises :class:`DataProcessingError`.
        A backend that cannot tell skips its probe.
        """
        stored = [*delta.unchanged, *delta.changed]
        text_unit_ids: dict[str, list[str]] = defaultdict(list)
        probed = 0
        for start in range(0, len(stored), _PROBE_DOCUMENTS):
            batch = stored[start : start + _PROBE_DOCUMENTS]
            for record in self.doc_status.get_many(batch).values():
                if record.status is not DocStatus.PROCESSED or not record.text_unit_ids:
                    continue
                text_unit_ids[record.suffix].extend(record.text_unit_ids)
                probed += 1
            if probed >= _PROBE_DOCUMENTS:
                break
        probes = {
            suffix: sorted(set(ids))[:_PROBE_UNITS]
            for suffix, ids in text_unit_ids.items()
        }
        if not probes:
            return
        graph = self.indexing_manager.neptune_indexer
        vector = self.indexing_manager.opensearch_indexer
        lost = "doc-status registry lists indexed documents of this namespace whose"
        advice = (
            "Re-run with indexing.reset: true to rebuild the stores and the "
            "registry, or delete the registry's records of this namespace."
        )
        found_units = [
            vector.read_text_unit_ids(ids, suffix=suffix)
            for suffix, ids in probes.items()
        ]
        if all(found is not None and not found for found in found_units):
            raise DataProcessingError(
                f"The {lost} text units the vector store does not hold (its "
                "database file was deleted or moved, providers.vector."
                f"sqlite_path changed, or its indices were deleted). {advice}"
            )
        # The documents probed may state no entity (cover pages): the graph
        # lost them only if the vector store holds entities citing them.
        opensearch = self.indexing_manager.config.indexing.opensearch
        lost_graph = []
        for suffix, ids in probes.items():
            in_graph = graph.find_ids_citing(ids, suffix=suffix)
            lost_graph.append(
                in_graph is not None
                and not in_graph
                and bool(
                    vector.find_documents(
                        opensearch.entities_index_prefix, "text_unit_ids", ids, suffix
                    )
                )
            )
        if all(lost_graph):
            raise DataProcessingError(
                f"The {lost} entities the graph store does not hold (an "
                f"in-memory Gremlin Server loses its graph on restart). {advice}"
            )

    def write_ahead(
        self,
        delta: DocumentDelta,
        lineages: list[DocumentLineage],
    ) -> None:
        """Record the documents about to be (re)indexed PENDING, before any
        store write.

        Call after extraction and before removing or writing anything for the
        delta. Each new and changed document (and any other document with a
        lineage) gets a PENDING record with the content hash
        ``PENDING_CONTENT_HASH`` and the union of its stored text units and
        the ones this run is about to write (``lineages``), under its current
        key and the run's scope (the stored scope when the run has none), so:

        - an interrupted run leaves a record the next diff classifies changed
          (never unchanged, even when the document reverted to the indexed
          content), or deleted when the document is gone;
        - removing it then finds every artifact this run may have written:
          they cite those text units.

        A record over the store's item limit (``DocStatusPort.record_fits``)
        raises :class:`DataProcessingError` here, before anything is written.

        :meth:`commit` replaces the records with the real ones. A PENDING
        record has ``failure_count`` 0 and is never FAILED: an interruption is
        not counted as a failure, but it resets the consecutive-failure count
        of a document that had failed before (the next run counts from the
        PENDING record). The count of a failure in this run continues from
        the record stored before the PENDING one. The stored records are read
        once (batched) and kept for :meth:`commit`.

        The record's attempt fields are kept as stored: the run opened its
        attempt before its first stage (:func:`open_attempts`), and a full
        rebuild opens none, so a rebuild that dies never counts toward
        quarantine.
        """
        check_lease()
        planned = {lineage.doc_id: lineage for lineage in lineages}
        doc_ids = list(dict.fromkeys([*delta.to_process, *planned]))
        if not doc_ids:
            return
        self._prior = self.doc_status.get_many(doc_ids)
        self._in_flight = set(doc_ids)
        self._pending = {
            doc_id: self._pending_record(
                doc_id, self._prior.get(doc_id), planned.get(doc_id)
            )
            for doc_id in doc_ids
        }
        too_large = [
            f"{record.file_path or record.doc_id} "
            f"({len(record.text_unit_ids)} text units)"
            for record in self._pending.values()
            if not self.doc_status.record_fits(record)
        ]
        if too_large:
            raise DataProcessingError(
                f"{len(too_large)} documents have more text units than one "
                "doc-status registry record holds (DynamoDB: 400 KB, roughly "
                "10,000); split them into smaller files. Nothing was written "
                "for this run's delta: " + ", ".join(too_large[:10])
            )
        self.doc_status.put_many(list(self._pending.values()))
        logger.info(
            "Recorded %d documents PENDING before indexing the delta",
            len(self._pending),
        )

    def _pending_record(
        self,
        doc_id: str,
        prior: DocStatusRecord | None,
        lineage: DocumentLineage | None,
    ) -> DocStatusRecord:
        suffix = lineage.suffix if lineage else self.suffix
        base = prior or DocStatusRecord(
            doc_id=doc_id, content_hash=PENDING_CONTENT_HASH, suffix=suffix
        )
        return base.model_copy(
            update={
                "content_hash": PENDING_CONTENT_HASH,
                "status": DocStatus.PENDING,
                "error_info": _WRITE_AHEAD_INFO,
                "failure_count": 0,
                # The stored suffix is kept: removal groups by it, and a
                # changed document's old artifacts were written under it. A
                # record without text units (a new document's attempt
                # record) takes the run's.
                "suffix": base.suffix if base.text_unit_ids else suffix,
                "scope": self.scope if self.scope is not None else base.scope,
                "file_path": (lineage.file_path if lineage else None) or base.file_path,
                "text_unit_ids": sorted(
                    set(base.text_unit_ids)
                    | set(lineage.text_unit_ids if lineage else ())
                ),
            }
        )

    def remove_obsolete_artifacts(self, doc_ids: list[str]) -> bool:
        """Remove what the given (deleted/changed) docs contributed.

        Their text units are read from the registry (for a document of this
        run, from the record stored before its PENDING one; after an
        interrupted run, that run's PENDING record, so whatever it wrote is
        found too) and grouped by the suffix they were written under. The
        stores are then asked for the items citing them
        (``IndexingManager.remove_text_units``): an item citing only removed
        text units is deleted, one citing others too is stripped of them and
        its content derived again from the rest.

        Returns True if every store write reported no failures (or there was
        nothing to remove). The store paths swallow exceptions into error
        stats rather than raising, so the caller must inspect this result
        before deleting the registry record — otherwise a transient failure
        would orphan the artifacts (record gone, artifacts still live).
        """
        check_lease()
        if not doc_ids:
            return True
        stored = self.doc_status.get_many(
            [doc_id for doc_id in doc_ids if doc_id not in self._in_flight]
        )
        units_by_suffix: dict[str, set[str]] = defaultdict(set)
        for doc_id in doc_ids:
            record = (
                self._prior.get(doc_id)
                if doc_id in self._in_flight
                else stored.get(doc_id)
            )
            if record is not None:
                units_by_suffix[record.suffix].update(record.text_unit_ids)
        results: list[IndexingStats] = []
        for suffix, unit_ids in sorted(units_by_suffix.items()):
            if not unit_ids:
                continue
            logger.info(
                "Removing the %d text units of %d obsolete documents from '%s'",
                len(unit_ids),
                len(doc_ids),
                suffix,
            )
            results.extend(
                self.indexing_manager.remove_text_units(
                    suffix, sorted(unit_ids)
                ).values()
            )
        failed = sum(s.failed_items for s in results if s is not None)
        if failed:
            logger.warning(
                "Artifact removal for obsolete docs had %d failures; caller "
                "should not treat the removal as complete.",
                failed,
            )
        return failed == 0

    def commit(
        self,
        lineages: list[DocumentLineage],
        fingerprints: dict[str, str],
        *,
        failed_doc_ids: Collection[str] = (),
        text_units: list[TextUnit] | None = None,
        entities: list[Entity] | None = None,
        relationships: list[Relationship] | None = None,
        communities: list[Community] | None = None,
        community_reports: list[CommunityReport] | None = None,
        claims: list[Claim] | None = None,
    ) -> dict[str, IndexingStats]:
        """Upsert delta artifacts and update the registry for processed docs.

        ``lineages`` lists each processed doc's text units, by which a later
        run finds what the doc contributed. The artifact lists passed
        separately are the union actually written to the stores this run. Docs in ``failed_doc_ids`` (an
        extraction stage failed on some of their text units) are recorded
        FAILED with their lineage, so the next run re-extracts them and first
        prunes what this run wrote.

        Entities and relationships are first merged with the stored graph state
        they touch; the communities and claims
        referencing a delta item that merged into a stored one under another
        id are pointed at that id. Claims are merged with the stored claims
        of the same id, which other documents may have contributed to. Once
        the docs are recorded, the extraction records only removed text units
        named are deleted (``IndexingManager.collect_extraction_records``).
        """
        check_lease()
        merged = self.indexing_manager.merge_with_existing_graph(
            entities, relationships, text_units or ()
        )
        entities, relationships = merged.entities, merged.relationships
        communities, claims = _remap_references(
            communities, claims, merged.id_remap_by_suffix
        )
        self.community_drift = self._measure_drift(entities, communities)
        claims = self.indexing_manager.merge_with_stored_claims(
            claims, text_units or ()
        )
        results = self.indexing_manager.index_delta(
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            communities=communities,
            community_reports=community_reports,
            claims=claims,
        )
        # The items the merge re-keyed to follow their current name are now
        # written under the new ids: delete the old ones, unless a write
        # failed (deleting them would lose the item; the next run re-keys it).
        if any(stats.failed_items for stats in results.values() if stats):
            self.indexing_manager.discard_retired_ids()
        else:
            # Cleanup, not the delta: a failure leaves an old copy beside the
            # new one (the next run that touches it re-keys it again).
            retired = self.indexing_manager.delete_retired_ids()
            failed = sum(stats.failed_items for stats in retired.values())
            if failed:
                logger.warning(
                    "Deleting %d re-keyed items' old ids failed; their old "
                    "copies stay until a run touches them again",
                    failed,
                )
        owners = artifact_owners(
            lineages,
            [
                *(text_units or []),
                *(entities or []),
                *(relationships or []),
                *(communities or []),
                *(community_reports or []),
                *(claims or []),
            ],
        )
        if self.record(lineages, fingerprints, results, failed_doc_ids, owners):
            # After the delta's writes: a changed document's new text units
            # name the records its removed ones did.
            self.indexing_manager.collect_extraction_records()
        return results

    def _measure_drift(
        self, entities: list[Entity] | None, communities: list[Community] | None
    ) -> CommunityDrift | None:
        """Measure and log the commit's community drift (see community_drift)."""
        if not communities:
            return None
        drift = measure_community_drift(
            entities or [],
            communities,
            self.indexing_manager.config.indexing.community_drift_threshold,
        )
        if drift is None:
            return None
        if drift.rebuild_recommended:
            logger.warning(
                "Community drift: %d of the %d entities this run clustered also "
                "belong to an earlier level-0 community (%d earlier communities "
                "have stale reports). Incremental runs cluster only their delta; "
                "rebuild the communities with indexing.reset: true.",
                drift.overlapping_entities,
                drift.clustered_entities,
                drift.stale_communities,
            )
        else:
            logger.info(
                "Community drift: %d of %d clustered entities also in an earlier "
                "level-0 community (%d stale communities)",
                drift.overlapping_entities,
                drift.clustered_entities,
                drift.stale_communities,
            )
        return drift

    def record(
        self,
        lineages: list[DocumentLineage],
        fingerprints: dict[str, str],
        results: dict[str, IndexingStats],
        failed_doc_ids: Collection[str] = (),
        owners: Mapping[str, set[str]] | None = None,
    ) -> bool:
        """Record the processed docs in the registry if their writes landed.

        Used after a delta upsert (:meth:`commit`) and after a reset's full
        rebuild. The indexing manager SWALLOWS per-task write errors into stats
        and never raises, so recording PROCESSED unconditionally would persist a
        doc's new content_hash + lineage even when its backend writes failed —
        the doc would then be classified `unchanged` on the next run and its
        missing artifacts never re-indexed. Returns whether the docs were
        recorded.

        Under the tolerated failure rate the docs are recorded, but a doc that
        owns an artifact whose write failed is recorded FAILED (see
        :meth:`_documents_with_failed_writes`), so the next run rewrites it.
        """
        check_lease()
        if self._delta_writes_succeeded(results):
            write_failed = self._documents_with_failed_writes(
                lineages, results, owners or {}
            )
            self._record_processed(lineages, fingerprints, failed_doc_ids, write_failed)
            if (
                self.scope is not None
                and self.extraction_fingerprint is not None
                and self.extraction_settings is not None
            ):
                self.doc_status.put_extraction_settings(
                    self.scope, self.extraction_fingerprint, self.extraction_settings
                )
            return True
        logger.error(
            "Indexing failed for at least one artifact type (complete failure "
            "or failure rate above the tolerated threshold); NOT recording docs "
            "as PROCESSED so they are retried on the next run. Stats: %s",
            {k: (v.successful_items, v.failed_items) for k, v in results.items()},
        )
        return False

    def _delta_writes_succeeded(self, results: dict[str, IndexingStats]) -> bool:
        """False if any index type failed hard enough that the delta must retry.

        Must mirror the SAME failure criteria as
        IndexingStage._validate_backend_success, because that gate runs AFTER
        commit() has already written the registry: if this gate is more lenient,
        commit records docs PROCESSED, then _validate_backend_success raises and
        fails the pipeline, but the DynamoDB write is not rolled back — the
        partially-indexed docs are classified `unchanged` next run and their
        failed artifacts are never retried. Two failure modes, matching that gate:

        1. Complete failure: an index type with zero successes. "Work to do" is
           measured by total_items OR failed_items, since an index type that
           raises before _perform_indexing seeds total_items surfaces as
           total_items=0, failed_items=N (gating on total_items alone would let
           that pass as success).
        2. Partial failure over threshold: failure rate exceeds the configured
           max_failure_rate (same tolerance as the pipeline gate).
        """
        max_failure_rate = self.indexing_manager.config.indexing.max_failure_rate
        for stats in results.values():
            if not stats:
                continue
            had_work = stats.total_items > 0 or stats.failed_items > 0
            if had_work and stats.successful_items == 0:
                return False
            if stats.total_items > 0:
                failure_rate = stats.failed_items / stats.total_items
                if failure_rate > max_failure_rate:
                    return False
        return True

    @staticmethod
    def _documents_with_failed_writes(
        lineages: list[DocumentLineage],
        results: dict[str, IndexingStats],
        owners: Mapping[str, set[str]],
    ) -> set[str]:
        """Registry ids of the documents owning an artifact whose write failed.

        The failed artifact ids come from the indexers' stats, their owners
        from ``owners`` (see :func:`artifact_owners`). A failure the indexer
        reported without an id, or for an id no owner is known for, cannot be
        attributed, so every document of the commit is returned: retrying them all re-extracts more than
        needed, while recording them PROCESSED would leave the artifact missing
        with the documents classified unchanged on every later run.
        """
        failed_ids: set[str] = set()
        unattributed = 0
        for stats in results.values():
            if not stats:
                continue
            # failed_ids keeps one entry per failed item, so the unattributed
            # count must come from the list, not the deduplicated set: one
            # artifact failing in several batches is still fully attributed.
            failed_ids.update(stats.failed_ids)
            unattributed += stats.unattributed_failures
        if unattributed:
            logger.warning(
                "%d write failures carry no artifact id; recording all %d "
                "documents of this run FAILED so the next run rewrites them",
                unattributed,
                len(lineages),
            )
            return {lineage.doc_id for lineage in lineages}
        if not failed_ids:
            return set()
        unowned = [i for i in failed_ids if not owners.get(i)]
        if unowned:
            logger.warning(
                "%d failed writes belong to no document of this run; recording "
                "all %d documents of this run FAILED so the next run rewrites "
                "them",
                len(unowned),
                len(lineages),
            )
            return {lineage.doc_id for lineage in lineages}
        failed_docs = {doc_id for i in failed_ids for doc_id in owners[i]}
        logger.warning(
            "%d artifact writes failed; recording the %d documents that own "
            "them FAILED so the next run rewrites them",
            len(failed_ids),
            len(failed_docs),
        )
        return failed_docs

    def _delete_records(self, doc_ids: list[str]) -> None:
        """Delete removed documents' records."""
        for doc_id in doc_ids:
            self.doc_status.delete(doc_id)

    def remove_changed_and_deleted(self, delta: DocumentDelta) -> bool:
        """Remove the stale artifacts of changed AND deleted docs in one pass.

        A changed document's old entities/edges are dropped (unless shared
        with surviving docs) before its re-extraction is committed, so an
        artifact the new extraction no longer produces is not left orphaned.
        The removal is planned once over changed + deleted, so an artifact only
        those documents reference is removed even when it is referenced by one
        changed and one deleted document; planning the two sets separately
        would treat each as a survivor of the other.

        The deleted documents' registry records are deleted only when the
        removal fully succeeded: deleting them first would strand the
        still-live artifacts with no lineage to clean them up. Returns False
        on a partial failure. The caller MUST NOT :meth:`commit` changed
        documents on False: commit replaces their lineage with the new
        artifacts, so the old ids that failed to delete would never be
        cleaned up. Not committing keeps the old lineage (inside the PENDING
        record :meth:`write_ahead` wrote), so the next run retries.

        Quarantined documents (``DocumentDelta.quarantined``) are recorded
        QUARANTINED. One whose attempts reached the indexing stage (its record
        is the write-ahead ``PENDING`` one) may have written artifacts and
        removed some of its indexed version: it is removed in the same pass
        and recorded without text units. One whose attempts died before (its
        record is still the indexed version's) keeps that version: nothing of
        the new one was written.
        """
        held = self.doc_status.get_many(delta.quarantined) if delta.quarantined else {}
        kept = {
            doc_id
            for doc_id, record in held.items()
            if record.status is not DocStatus.PENDING and record.text_unit_ids
        }
        doc_ids = list(
            dict.fromkeys(
                delta.changed
                + delta.deleted
                + [doc_id for doc_id in delta.quarantined if doc_id not in kept]
            )
        )
        if not doc_ids:
            check_lease()
            self._record_quarantined(delta.quarantined, kept)
            return True
        if not self.remove_obsolete_artifacts(doc_ids):
            if delta.deleted or delta.quarantined:
                logger.warning(
                    "Keeping %d deleted or quarantined documents' registry "
                    "records because artifact removal did not fully succeed; "
                    "will retry next run.",
                    len(delta.deleted) + len(delta.quarantined),
                )
            return False
        check_lease()
        self._delete_records(delta.deleted)
        self._record_quarantined(delta.quarantined, kept)
        return True

    def _record_quarantined(
        self, doc_ids: list[str], kept: Collection[str] = ()
    ) -> None:
        """Replace the quarantined documents' records with QUARANTINED ones of
        the content they died on (unchanged while it stays), with their deaths
        settled. Those in ``kept`` keep the text units of their indexed
        version, the others have none left."""
        if not doc_ids:
            return
        records = []
        for doc_id, record in self.doc_status.get_many(doc_ids).items():
            attempted = record.attempted_hash or record.content_hash
            deaths, crash_batch = settled_attempts(record, attempted)
            records.append(
                record.model_copy(
                    update={
                        "status": DocStatus.QUARANTINED,
                        "content_hash": attempted,
                        "text_unit_ids": (
                            record.text_unit_ids if doc_id in kept else []
                        ),
                        "error_info": _QUARANTINED_INFO.format(attempts=deaths),
                        "attempt_count": deaths,
                        "crash_batch": crash_batch,
                        "attempt_batch": None,
                        "scope": (
                            self.scope if self.scope is not None else record.scope
                        ),
                        "extraction_fingerprint": (
                            self.extraction_fingerprint or record.extraction_fingerprint
                        ),
                    }
                )
            )
        self.doc_status.put_many(records)

    def _record_processed(
        self,
        lineages: list[DocumentLineage],
        fingerprints: dict[str, str],
        failed_doc_ids: Collection[str] = (),
        write_failed_doc_ids: Collection[str] = (),
    ) -> None:
        if failed_doc_ids:
            logger.warning(
                "Extraction failed on part of %d documents; recording them FAILED "
                "so the next run re-extracts them",
                len(failed_doc_ids),
            )
        # The records as stored before this run's PENDING records (read by
        # write_ahead); the others in one batched read.
        prior = {
            **self.doc_status.get_many(
                [lin.doc_id for lin in lineages if lin.doc_id not in self._pending],
            ),
            **self._prior,
        }
        records: list[DocStatusRecord] = []
        for lineage in lineages:
            error_info = None
            if lineage.doc_id in failed_doc_ids:
                error_info = _EXTRACTION_FAILED
            elif lineage.doc_id in write_failed_doc_ids:
                error_info = _WRITE_FAILED
            failed = error_info is not None
            existing = prior.get(lineage.doc_id)
            content_hash = fingerprints.get(
                lineage.doc_id, existing.content_hash if existing else ""
            )
            # Consecutive failures of this content; new content starts over.
            failure_count = 0
            if failed:
                failure_count = 1
                if (
                    existing is not None
                    and existing.status is DocStatus.FAILED
                    and existing.content_hash == content_hash
                ):
                    failure_count += existing.failure_count
            record = DocStatusRecord(
                doc_id=lineage.doc_id,
                content_hash=content_hash,
                status=DocStatus.FAILED if failed else DocStatus.PROCESSED,
                error_info=error_info,
                failure_count=failure_count,
                extraction_fingerprint=self.extraction_fingerprint,
                suffix=lineage.suffix,
                scope=(
                    self.scope
                    if self.scope is not None
                    else (existing.scope if existing else None)
                ),
                file_path=lineage.file_path,
                text_unit_ids=lineage.text_unit_ids,
            )
            records.append(record)
        unrecorded = sorted(set(self._pending) - {r.doc_id for r in records})
        # Counted as an attempt of the document alone (unproductive_attempt),
        # so one that never produces a text unit is quarantined instead of
        # re-extracted on every run.
        records += [
            unproductive_attempt(
                self._pending[doc_id],
                fingerprints.get(doc_id, self._pending[doc_id].attempted_hash or ""),
            )
            for doc_id in unrecorded
        ]
        self.doc_status.put_many(records)
        if unrecorded:
            logger.warning(
                "%d documents of the delta produced no lineage this run; their "
                "records stay PENDING so the next run re-extracts them: %s",
                len(unrecorded),
                ", ".join(
                    self._pending[doc_id].file_path or doc_id
                    for doc_id in unrecorded[:10]
                ),
            )


# --- the pipeline's incremental steps -----------------------------------------
# The document_loading stage diffs the corpus (``filter_to_delta``) and the
# indexing stage writes the delta (``index_delta``) or clears the registry of a
# reset (``clear_registry_namespaces``); both stages only delegate here.


def run_registry_scope(
    config: Config, source_directory: Path, failed_source_files: list[str]
) -> tuple[str, str, str, list[str]]:
    """The run's registry scope, namespace, source scope and failed files.

    The scope is the index namespace the run writes (``index_value`` +
    ``indexing.additional_suffix``, tagged with the index and label prefixes
    when they are not the defaults, see ``store_namespace_tag``) and the
    corpus source
    (``document_parsing.source_scope``, else the resolved source
    directory), so a run never deletes another tenant's or another
    corpus's documents. The source scope is also part of every document's
    registry key, so two corpora written to one namespace never share a
    record. Failed files are keyed the same way their documents would have
    been.
    """
    parsing = config.processing.document_parsing
    namespace = config_registry_namespace(config.indexing, parsing.index_value)
    root = source_directory
    source_scope = parsing.source_scope or root.as_posix()
    failed_doc_ids = [
        compute_doc_id(
            relative_source_path(path, root) or normalize_source_path(path),
            namespace,
            source_scope,
        )
        for path in failed_source_files
    ]
    return (
        registry_scope(namespace, source_scope),
        namespace,
        source_scope,
        failed_doc_ids,
    )


def dedup_preferences(
    documents: list[Document],
    context: PipelineContext,
    config: Config,
    doc_status: Callable[[], DocStatusPort],
    source_directory: Path,
) -> tuple[Callable[[Document], bool], Callable[[Document], bool]]:
    """Which near-duplicate to keep: ``(indexed, failed)`` predicates.

    The incremental diff treats a dropped document as deleted, so a new
    near-copy that happened to be kept would delete the indexed original and
    index itself instead. A document the registry holds as PROCESSED is
    therefore kept over one it does not, and one
    it holds as FAILED gives way to any other: a FAILED document may have used
    up its retries (``indexing.max_document_failures``), so keeping it over
    its copy would leave neither indexed. A PENDING record (an interrupted
    run) is neither.
    """
    _, _, source_scope, _ = run_registry_scope(
        config, source_directory, context.failed_source_files
    )
    keys: dict[int, str] = {}
    for document in documents:
        assign_registry_source(document, source_scope)
        keys[id(document)] = document_doc_id(document)
    try:
        stored = doc_status().get_many(set(keys.values()))
    except Exception as e:
        raise DocStatusRegistryError(
            f"Doc-status registry: could not read it to deduplicate the "
            f"corpus ({type(e).__name__}: {e})"
        ) from e

    def has_status(status: DocStatus) -> Callable[[Document], bool]:
        def check(document: Document) -> bool:
            record = stored.get(keys[id(document)])
            return record is not None and record.status is status

        return check

    indexed = has_status(DocStatus.PROCESSED)
    failed = has_status(DocStatus.FAILED)
    return indexed, lambda document: failed(document) and not indexed(document)


def run_extraction_fingerprint(
    store: DocStatusPort, scope: str, settings: ExtractionSettings
) -> str:
    """The extraction fingerprint a run in ``scope`` diffs with and records.

    The fingerprint the scope's last commit recorded while its settings
    match ``settings`` field by field (``cache_keys.changed_settings``: a
    field a newer version added still at its default is no change), so an
    upgrade that adds a setting does not re-extract the corpus; otherwise
    the fingerprint of ``settings``, and a WARNING names what changed.
    """
    stored = store.get_extraction_settings(scope)
    if stored is None:
        return settings.fingerprint
    fingerprint, recorded = stored
    changed = changed_settings(recorded, settings)
    if not changed:
        return fingerprint
    logger.warning(
        "The extraction settings changed since this corpus was last committed "
        "(%s%s); the documents extracted with the earlier settings are "
        "re-extracted",
        ", ".join(changed[:10]),
        f" and {len(changed) - 10} more" if len(changed) > 10 else "",
    )
    return settings.fingerprint


def filter_to_delta(
    documents: list[Document],
    context: PipelineContext,
    config: Config,
    doc_status: Callable[[], DocStatusPort],
    source_directory: Path,
) -> tuple[list[Document], int]:
    """Keep only new/changed documents per the DynamoDB doc-status registry.

    Also stashes the computed delta + fingerprints on the context so the
    IndexingStage can prune stale artifacts, propagate deletions, and record
    per-document lineage (the full incremental commit path).

    Raises :class:`DocStatusRegistryError` when the registry cannot be
    read. There is no safe fallback: without a delta the indexing stage
    takes the full-rebuild path, which replaces the suffix's live index
    content with this run's documents and never records them, so other
    scopes' documents would be lost and every later run would rebuild.
    """
    scope, namespace, source_scope, failed_doc_ids = run_registry_scope(
        config, source_directory, context.failed_source_files
    )
    # Fails before any registry read or write when the run would retire
    # its own scope.
    retired = validate_retired_scopes(
        config.indexing.retire_source_scopes, source_scope, namespace
    )
    for document in documents:
        assign_registry_source(document, source_scope)
    settings = extraction_settings(config)
    context.incremental_extraction_settings = settings
    context.incremental_extraction_fingerprint = settings.fingerprint
    if config.indexing.reset:
        # A reset clears the stores and the registry before indexing, so
        # every document is new: diffing against the registry here would
        # rebuild from the delta alone and lose the unchanged documents.
        fingerprints = fingerprint_documents(documents)
        context.incremental_delta = DocumentDelta(new=list(fingerprints))
        context.incremental_fingerprints = fingerprints
        context.incremental_scope = scope
        logger.info(
            "Incremental filter skipped (indexing.reset): processing all "
            "%d documents",
            len(documents),
        )
        if retired:
            logger.info(
                "indexing.retire_source_scopes ignored: the reset clears "
                "every record"
            )
        return documents, 0

    try:
        store = doc_status()
        extraction = run_extraction_fingerprint(store, scope, settings)
        context.incremental_extraction_fingerprint = extraction
        delta, fingerprints = detect_delta(
            documents,
            store,
            scope=scope,
            failed_doc_ids=failed_doc_ids,
            max_failures=config.indexing.max_document_failures,
            max_attempts=config.indexing.max_document_attempts,
            retired_scopes=retired,
            extraction_fingerprint=extraction,
        )
    except (DocStatusRegistryError, ConfigurationError):
        raise
    except Exception as e:
        raise DocStatusRegistryError(
            f"Doc-status registry: could not diff the corpus against the "
            f"registry ({type(e).__name__}: {e}). Incremental indexing needs "
            "the registry; fix it and re-run, or turn incremental indexing off "
            "(providers.doc_status.enabled: false) to "
            "run without it."
        ) from e
    stored_spelling = _stored_spelling(source_scope, namespace, scope, delta)
    if stored_spelling is not None:
        # The corpus was indexed under another spelling of its source (an
        # s3:// URI with or without a trailing slash): diffing under this
        # one would index every document again beside the stored copies and
        # never delete those. Key the run by the stored spelling instead.
        logger.warning(
            "The registry holds this corpus under the source scope %r, another "
            "spelling of %r; using the stored spelling (set "
            "processing.document_parsing.source_scope to it to skip this check)",
            stored_spelling,
            source_scope,
        )
        respelled = config.model_copy(deep=True)
        respelled.processing.document_parsing.source_scope = stored_spelling
        return filter_to_delta(
            documents, context, respelled, doc_status, source_directory
        )
    context.incremental_delta = delta
    context.incremental_fingerprints = fingerprints
    context.incremental_scope = scope
    context.incremental_retired_scopes = list(config.indexing.retire_source_scopes)
    open_attempts(
        store,
        delta,
        fingerprints,
        scope=scope,
        suffix=config.processing.document_parsing.index_value
        or Constants.DEFAULT_SUFFIX.value,
        file_paths={
            document_doc_id(document): document.metadata.get(RELATIVE_PATH_KEY)
            or document.file_path
            for document in documents
        },
    )
    if delta.new and not config.processing.document_parsing.source_scope:
        _warn_about_other_local_scopes(delta, scope, namespace, retired)
    if delta.is_empty:
        logger.info("Incremental: no new/changed documents detected")
    to_process = filter_documents_to_process(documents, delta)
    skipped = len(documents) - len(to_process)
    logger.info(
        "Incremental filter: %d to process, %d unchanged (skipped), %d deleted, "
        "%d deferred",
        len(to_process),
        skipped,
        len(delta.deleted),
        len(delta.deferred),
    )
    return to_process, skipped


def open_attempts(
    store: DocStatusPort,
    delta: DocumentDelta,
    fingerprints: Mapping[str, str],
    *,
    scope: str | None = None,
    suffix: str = DEFAULT_NAMESPACE,
    file_paths: Mapping[str, str | None] | None = None,
) -> None:
    """Open an attempt on every document the run processes (``delta.to_process``).

    Called before the first stage that processes them: by the
    document_loading stage right after the diff, and by a run resumed after
    it. Each record gets ``attempts.opened_attempt`` (an attempt a dead run
    left open is counted first); a new document gets a ``PENDING`` record
    without text units, under the run's ``scope``, so a run that dies before
    the indexing stage still counts against it. Status, content hash and
    text units of a stored record are kept: the diff, the failure count and
    the removal read them as before. One batched read and write.
    """
    check_lease()
    batch = delta.to_process
    if not batch:
        return
    stored = store.get_many(batch)
    paths = file_paths or {}
    store.put_many(
        [
            opened_attempt(
                stored.get(doc_id)
                or DocStatusRecord(
                    doc_id=doc_id,
                    content_hash=PENDING_CONTENT_HASH,
                    status=DocStatus.PENDING,
                    error_info=_WRITE_AHEAD_INFO,
                    suffix=suffix,
                    scope=scope,
                    file_path=paths.get(doc_id),
                ),
                fingerprints[doc_id],
                len(batch),
            )
            for doc_id in batch
        ]
    )


def close_attempts(
    store: DocStatusPort, delta: DocumentDelta, fingerprints: Mapping[str, str]
) -> int:
    """Close the attempts :func:`open_attempts` opened for ``delta``.

    Called whenever the run ends in the process (it finished, a stage
    failed, or it was stopped): only a run that died leaves its attempts
    open, so only deaths count toward quarantine. Records the run committed
    are new and have none. Returns how many it closed.
    """
    check_lease()
    stored = store.get_many(delta.to_process)
    still_open = [
        closed_attempt(record)
        for doc_id, record in stored.items()
        if record.attempt_batch is not None
        and record.attempted_hash == fingerprints.get(doc_id)
    ]
    if still_open:
        store.put_many(still_open)
    return len(still_open)


def delta_still_applies(
    store: DocStatusPort,
    delta: DocumentDelta,
    fingerprints: Mapping[str, str],
    max_attempts: int,
) -> bool:
    """Whether a resumed run may index ``delta`` as it was computed.

    Every document the delta processes holds the attempt its run opened
    (``attempted_hash`` at its content) until a commit replaces the record
    with one of that content. A record that is gone (another run found the
    document deleted) or holds other content (another run indexed another
    version) means indexing the delta would undo what that run did. Redoing
    a commit of the same content is idempotent, so a stage of a finished
    pipeline can be re-run. A document whose open attempt shows the last run
    died, with the deaths now at ``max_attempts``, needs the plan of a new
    diff (``attempts.plan_attempts``), or the resumed runs would keep dying
    on the same delta.
    """
    stored = store.get_many(delta.to_process)
    for doc_id in delta.to_process:
        record = stored.get(doc_id)
        content_hash = fingerprints.get(doc_id)
        if record is None or content_hash is None:
            return False
        if content_hash not in (record.attempted_hash, record.content_hash):
            return False
        deaths, _ = settled_attempts(record, content_hash)
        if record.attempt_batch is not None and deaths >= max_attempts:
            return False
    return True


def _stored_spelling(
    source_scope: str, namespace: str, scope: str, delta: DocumentDelta
) -> str | None:
    """Another spelling of ``source_scope`` the registry holds this
    namespace's records under, when it holds none under ``scope`` itself
    (see :func:`normalize_source_scope`)."""
    if scope in delta.stored_scopes:
        return None
    prefix = f"{namespace}|"
    own = normalize_source_scope(source_scope)
    spellings = sorted(
        stored[len(prefix) :]
        for stored in delta.stored_scopes
        if stored.startswith(prefix)
        and stored[len(prefix) :] != source_scope
        and normalize_source_scope(stored[len(prefix) :]) == own
    )
    return spellings[0] if spellings else None


def _warn_about_other_local_scopes(
    delta: DocumentDelta, scope: str, namespace: str, retired: list[str]
) -> None:
    """Name the other local source directories of the run's namespace.

    Only a run whose scope defaulted to its source directory and that has
    new documents gets here: that is what a moved corpus looks like. The
    records stay: a directory that is missing here may be another host's
    corpus sharing the registry, so only the user can tell a move from a
    second corpus. Uses the scopes the diff already read.
    """
    others = other_local_scopes(delta, scope, namespace, retired)
    if not others:
        return
    logger.warning(
        "The doc-status registry also holds records of index namespace %r "
        "under other local source directories: %s. If this corpus was "
        "moved from one of them, its old records keep the previous "
        "content indexed: run once with --retire-source-scope <old "
        "directory> (indexing.retire_source_scopes) to remove it. If they "
        "are other corpora, set document_parsing.source_scope to a stable "
        "name per corpus.",
        namespace,
        ", ".join(others),
    )


def reset_namespaces(
    config: Config, context: PipelineContext, text_units: list[TextUnit]
) -> set[str]:
    """The registry namespaces a reset rebuilds.

    The full build replaces the suffixes of the run's text units under
    ``indexing.additional_suffix`` (in the stores the configured prefixes
    name); the run's own namespace (its registry scope, else the configured
    ``index_value``) is always included.
    """
    namespaces = {
        config_registry_namespace(config.indexing, BaseIndexer.get_suffix(unit))
        for unit in text_units
    }
    namespaces.add(
        scope_namespace(context.incremental_scope)
        or config_registry_namespace(
            config.indexing, config.processing.document_parsing.index_value
        )
    )
    return namespaces


def clear_registry_namespaces(
    doc_status: Callable[[], DocStatusPort],
    namespaces: set[str],
    keep: Collection[str] = (),
) -> None:
    """Delete the registry records of ``namespaces`` but ``keep`` (reset path).

    A reset rebuilt the namespaces from the corpus and recorded its
    documents (``keep``); the other records describe documents the rebuild
    replaced. Only those namespaces: other index suffixes sharing the table
    keep their records, which still describe their intact stores. Each
    namespace is its own registry partition (``DocStatusPort.list_namespace``).

    The registry adapters raise :class:`DocStatusRegistryError` when the
    registry cannot be cleared, and it propagates as is (like any other
    typed error, e.g. a configuration error): a stale record would make a
    later incremental run remove what its document's file no longer holds,
    so the reset fails here rather than leave the registry out of step.
    """
    check_lease()
    store = doc_status()
    kept = set(keep)
    cleared = 0
    for namespace in sorted(namespaces):
        for record in store.list_namespace(namespace):
            if record.doc_id not in kept:
                check_lease()
                store.delete(record.doc_id)
                cleared += 1
    logger.info(
        "Cleared %s doc-status registry records of namespaces %s on reset",
        cleared,
        ", ".join(sorted(namespaces)),
    )


def warn_about_unapplied_retired_scopes(
    config: Config, context: PipelineContext
) -> None:
    """Warn about ``indexing.retire_source_scopes`` the delta did not apply.

    Scopes are retired when the document_loading stage computes the
    delta. A run resumed after it indexes the delta an earlier run
    computed, so scopes added since then are not removed by this run.
    """
    if context.incremental_retired_scopes is None:
        # Saved before the delta recorded its retired scopes: whether
        # they were applied is unknown.
        if config.indexing.retire_source_scopes:
            logger.warning(
                "indexing.retire_source_scopes / --retire-source-scope %s may "
                "not have been applied: this run resumed after the "
                "document_loading stage from a pipeline context that does "
                "not record which scopes its delta retired. Re-run from "
                "document_loading (--resume-from-stage document_loading) to "
                "make sure they are retired.",
                ", ".join(repr(s) for s in config.indexing.retire_source_scopes),
            )
        return
    applied = {normalize_source_scope(s) for s in context.incremental_retired_scopes}
    ignored = [
        scope
        for scope in config.indexing.retire_source_scopes
        if normalize_source_scope(scope) not in applied
    ]
    if ignored:
        logger.warning(
            "indexing.retire_source_scopes / --retire-source-scope %s NOT "
            "applied: this run resumed after the document_loading stage and "
            "indexes the delta an earlier run computed without them, so "
            "their records and exclusive artifacts are kept. Re-run from "
            "document_loading (--resume-from-stage document_loading) to "
            "retire them.",
            ", ".join(repr(s) for s in ignored),
        )


def documents_with_failed_units(
    context: PipelineContext, text_units: list[TextUnit]
) -> set[str]:
    """Registry ids of the documents an extraction stage failed on."""
    failed_units = {
        unit_id for ids in context.failed_text_unit_ids.values() for unit_id in ids
    }
    run_document_ids = {
        document_id
        for unit in text_units
        if unit.id in failed_units
        for document_id in unit.document_ids or []
    }
    return {
        document_doc_id(document)
        for document in context.documents
        if document.document_id in run_document_ids
    }


def run_incremental_indexer(
    doc_status: DocStatusPort,
    indexing_manager: IndexingManager,
    context: PipelineContext,
    text_units: list[TextUnit],
) -> IncrementalIndexer:
    """The incremental indexer of a run, on the run's suffix and scope."""
    # Artifacts carry their own index suffix (multi-tenant/version aware);
    # derive the run's suffix from the text units the same way the indexers
    # do. A run that only deletes extracts none: its suffix is the one its
    # documents would carry (document_parsing.index_value).
    suffix = (
        BaseIndexer.get_suffix(text_units[0])
        if text_units
        else indexing_manager.config.processing.document_parsing.index_value
        or Constants.DEFAULT_SUFFIX.value
    )
    return IncrementalIndexer(
        doc_status,
        indexing_manager,
        suffix=suffix,
        scope=context.incremental_scope,
        extraction_fingerprint=context.incremental_extraction_fingerprint,
        extraction_settings=context.incremental_extraction_settings,
    )


def context_lineages(
    context: PipelineContext, suffix: str, text_units: list[TextUnit]
) -> list[DocumentLineage]:
    """Per-document lineage of the run's documents (``build_document_lineage``)."""
    return build_document_lineage(
        documents=context.documents, text_units=text_units, suffix=suffix
    )


def index_delta(
    doc_status: Callable[[], DocStatusPort],
    indexing_manager: IndexingManager,
    context: PipelineContext,
    text_units: list[TextUnit],
    entities: list[Entity],
    relationships: list[Relationship],
    communities: list[Community],
    community_reports: list[CommunityReport],
    claims: list[Claim],
) -> tuple[dict[str, IndexingStats], CommunityDrift | None]:
    """Idempotent delta indexing + stale-artifact pruning + registry write-back.

    Routed to when a doc-status delta is present (incremental mode). The new
    and changed documents are first recorded PENDING (write-ahead), then
    the stale artifacts of changed/deleted documents are removed, the
    freshly extracted delta is upserted and the PENDING records replaced
    with per-document lineage so subsequent runs diff correctly.
    """
    delta = context.incremental_delta
    if delta is None:
        # A full replace here would wipe the documents the run did not load.
        raise PipelineStageError("Incremental indexing needs the run's delta")
    incremental = run_incremental_indexer(
        doc_status(), indexing_manager, context, text_units
    )
    lineages = context_lineages(context, incremental.suffix, text_units)
    # Checked before any write: the live indices must have been built for
    # this run's embedding model and languages by a run that kept the
    # registry (unchanged documents are not rewritten), and the graph must
    # still hold the registry's documents.
    indexing_manager.opensearch_indexer.check_index_settings(incremental.suffix)
    incremental.check_graph_holds_registry(delta)
    # Write ahead before any store write: the new and changed docs are
    # recorded PENDING with their stored + planned lineage, so a run
    # interrupted from here on is repaired by the next one (the docs
    # read as changed, or as deleted when gone, and every artifact this
    # run may write is found by their lineage).
    incremental.write_ahead(delta, lineages)

    # Drop the stale artifacts of changed and deleted docs in ONE removal
    # plan (before re-upsert): planning them separately keeps whatever a
    # changed and a deleted doc share, orphaning it. A failed removal must
    # stop the run before commit when docs changed: committing would
    # overwrite their lineage and orphan the stale artifacts. A failure
    # that only concerns deleted docs keeps their registry rows for a
    # retry and need not hold back the delta, so it fails the stage only
    # after the commit.
    removed = incremental.remove_changed_and_deleted(delta)
    if not removed and delta.changed:
        raise PipelineStageError(
            "Removing stale artifacts of changed or deleted documents "
            "failed; not committing the delta so the registry keeps their "
            "PENDING records (old + planned lineage) and the next run "
            "retries the removal"
        )

    results = incremental.commit(
        lineages=lineages,
        fingerprints=context.incremental_fingerprints,
        failed_doc_ids=documents_with_failed_units(context, text_units),
        text_units=text_units,
        entities=entities,
        relationships=relationships,
        communities=communities,
        community_reports=community_reports,
        claims=claims,
    )
    if not removed:
        raise PipelineStageError(
            "Removing the artifacts of deleted documents failed; their "
            "registry records are kept so the next run retries the removal"
        )
    return results, incremental.community_drift
