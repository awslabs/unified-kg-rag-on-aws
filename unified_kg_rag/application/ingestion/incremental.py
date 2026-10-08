# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental-indexing orchestrator.

Ties together the M2 pieces around a :class:`DocStatusPort`:

1. detect the corpus delta (new / changed / unchanged / deleted),
2. for deleted (and changed) documents, remove their previously indexed
   artifacts from the live stores using the lineage recorded in the registry,
3. upsert the freshly extracted delta artifacts (idempotent),
4. update the registry with the new content hashes and artifact lineage.

Extraction of the delta documents themselves is delegated to the caller (the
existing 12-stage pipeline run on the filtered subset), so this orchestrator
stays storage-focused and is exercised end-to-end with in-memory fakes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Collection
from typing import TYPE_CHECKING

from unified_kg_rag.domain.ingestion.delta_detector import (
    detect_delta,
    document_doc_id,
    filter_documents_to_process,
)
from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    DocStatus,
    DocStatusRecord,
    Document,
    DocumentDelta,
    DocumentLineage,
    Entity,
    Relationship,
    TextUnit,
)
from unified_kg_rag.ports import DocStatusPort
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.document_identity import RELATIVE_PATH_KEY

if TYPE_CHECKING:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager
    from unified_kg_rag.ports.indexer import IndexingStats

logger = get_logger(__name__)


def build_document_lineage(
    documents: list[Document],
    text_units: list[TextUnit],
    entities: list[Entity],
    relationships: list[Relationship],
    communities: list[Community],
    claims: list[Claim],
    community_reports: list[CommunityReport] | None = None,
    suffix: str = "default",
) -> list[DocumentLineage]:
    """Attribute extracted artifacts to their source documents.

    Walks the per-run ``document_id`` linkage (``TextUnit.document_ids`` and the
    artifacts' ``text_unit_ids``) and keys the resulting lineage by the *stable*
    ``doc_id`` so a later run can remove a document's exclusive artifacts. An
    artifact shared across documents is attributed to each — the registry-level
    exclusive-id computation later subtracts ids still referenced by survivors.
    """
    # Map per-run document_id -> stable doc_id.
    docid_to_stable = {doc.document_id: document_doc_id(doc) for doc in documents}
    # Map text_unit id -> set of stable doc_ids it belongs to.
    tu_to_docs: dict[str, set[str]] = {}
    docs_by_stable: dict[str, set[str]] = defaultdict(set)
    for tu in text_units:
        stable_ids = {
            docid_to_stable[d] for d in (tu.document_ids or []) if d in docid_to_stable
        }
        if stable_ids:
            tu_to_docs[tu.id] = stable_ids
            for s in stable_ids:
                docs_by_stable[s].add(tu.id)

    entity_ids: dict[str, set[str]] = defaultdict(set)
    relationship_ids: dict[str, set[str]] = defaultdict(set)
    claim_ids: dict[str, set[str]] = defaultdict(set)
    community_ids: dict[str, set[str]] = defaultdict(set)
    community_report_ids: dict[str, set[str]] = defaultdict(set)

    def _attribute(artifact_id: str, tu_ids: list[str] | None, bucket: dict) -> None:
        for tu_id in tu_ids or []:
            for stable in tu_to_docs.get(tu_id, ()):  # noqa: B007
                bucket[stable].add(artifact_id)

    for e in entities:
        _attribute(e.id, e.text_unit_ids, entity_ids)
    for r in relationships:
        _attribute(r.id, r.text_unit_ids, relationship_ids)
    for c in claims:
        _attribute(c.id, c.text_unit_ids, claim_ids)

    # Communities (and the reports that summarize them) attach to documents
    # through the community's member text units.
    community_docs: dict[str, set[str]] = {}
    for comm in communities:
        _attribute(comm.id, comm.text_unit_ids, community_ids)
        community_docs[comm.id] = {
            stable
            for tu_id in (comm.text_unit_ids or [])
            for stable in tu_to_docs.get(tu_id, ())
        }
    for report in community_reports or []:
        for stable in community_docs.get(report.community_id, ()):  # noqa: B007
            community_report_ids[stable].add(report.id)

    lineages = []
    for doc in documents:
        stable = docid_to_stable[doc.document_id]
        lineages.append(
            DocumentLineage(
                doc_id=stable,
                suffix=suffix,
                file_path=doc.metadata.get(RELATIVE_PATH_KEY) or doc.file_path,
                text_unit_ids=sorted(docs_by_stable.get(stable, set())),
                entity_ids=sorted(entity_ids.get(stable, set())),
                relationship_ids=sorted(relationship_ids.get(stable, set())),
                claim_ids=sorted(claim_ids.get(stable, set())),
                community_ids=sorted(community_ids.get(stable, set())),
                community_report_ids=sorted(community_report_ids.get(stable, set())),
            )
        )
    return lineages


class IncrementalIndexer:
    """Orchestrates an incremental indexing run against a document registry."""

    def __init__(
        self,
        doc_status: DocStatusPort,
        indexing_manager: IndexingManager,
        suffix: str = "default",
        scope: str | None = None,
    ) -> None:
        self.doc_status = doc_status
        self.indexing_manager = indexing_manager
        self.suffix = suffix
        # Registry scope (delta_detector.registry_scope) the delta is computed
        # in and recorded under; None diffs against the whole registry.
        self.scope = scope

    def plan(
        self, documents: list[Document], failed_doc_ids: list[str] | None = None
    ) -> tuple[DocumentDelta, dict[str, str]]:
        """Compute the delta for ``documents`` without mutating any store."""
        return detect_delta(
            documents, self.doc_status, self.scope, failed_doc_ids or ()
        )

    def documents_to_process(
        self, documents: list[Document], delta: DocumentDelta
    ) -> list[Document]:
        """Return the documents requiring extraction this run (new + changed)."""
        return filter_documents_to_process(documents, delta)

    def remove_obsolete_artifacts(self, doc_ids: list[str]) -> bool:
        """Delete artifacts belonging only to the given (deleted/changed) docs.

        Lineage is read from the registry; only ids not still referenced by a
        *surviving* document are removed, so shared artifacts are preserved.
        Removals are grouped by the suffix each document's artifacts were
        written under (multi-tenant/multi-index safe).

        Returns True if every store delete reported no failures (or there was
        nothing to remove). The store delete_by_id paths swallow exceptions into
        error stats rather than raising, so the caller must inspect this result
        before deleting the registry record — otherwise a transient delete
        failure would orphan the artifacts (record gone, artifacts still live).
        """
        if not doc_ids:
            return True

        obsolete_by_suffix = self._collect_exclusive_artifact_ids(doc_ids)
        if not obsolete_by_suffix:
            return True

        total = sum(len(ids) for ids in obsolete_by_suffix.values())
        logger.info(
            "Removing %d artifacts for %d obsolete documents",
            total,
            len(doc_ids),
        )
        results = self.indexing_manager.delete_documents(obsolete_by_suffix)
        failed = sum(s.failed_items for s in results.values() if s is not None)
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

        ``lineages`` attributes artifacts to their source document (one entry
        per processed doc), so per-document deletion later removes only a doc's
        *exclusive* artifacts. The artifact lists passed separately are the union
        actually written to the stores this run. Docs in ``failed_doc_ids`` (an
        extraction stage failed on some of their text units) are recorded
        FAILED with their lineage, so the next run re-extracts them and first
        prunes what this run wrote.
        """
        results = self.indexing_manager.index_delta(
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            communities=communities,
            community_reports=community_reports,
            claims=claims,
        )
        # Only write the registry back if the delta actually landed. index_delta
        # (via _run_indexing_phase) SWALLOWS per-task write errors into stats and
        # never raises, so recording PROCESSED unconditionally would persist a
        # doc's new content_hash + lineage even when its backend writes failed —
        # the doc would then be classified `unchanged` on the next run and its
        # missing artifacts never re-indexed. If any index type had items but
        # zero successes (a hard write failure), skip the registry write-back so
        # the affected docs are re-detected and retried next run.
        if self._delta_writes_succeeded(results):
            self._record_processed(lineages, fingerprints, failed_doc_ids)
        else:
            logger.error(
                "Delta indexing failed for at least one artifact type (complete "
                "failure or failure rate above the tolerated threshold); NOT "
                "recording docs as PROCESSED so they are retried on the next "
                "run. Stats: %s",
                {k: (v.successful_items, v.failed_items) for k, v in results.items()},
            )
        return results

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

    def remove_deleted(self, delta: DocumentDelta) -> None:
        """Remove artifacts and registry records for deleted documents.

        The registry record is only deleted when artifact removal fully
        succeeded. If removal partially failed, the record is kept so the next
        run retries the removal — deleting the record first would strand the
        still-live artifacts with no lineage to ever clean them up (permanent
        orphans).
        """
        if not delta.deleted:
            return
        removed = self.remove_obsolete_artifacts(delta.deleted)
        if not removed:
            logger.warning(
                "Keeping %d deleted-doc registry records because artifact "
                "removal did not fully succeed; will retry next run.",
                len(delta.deleted),
            )
            return
        for doc_id in delta.deleted:
            self.doc_status.delete(doc_id)

    def prune_changed(self, delta: DocumentDelta) -> bool:
        """Remove the now-stale artifacts of changed docs before re-extraction.

        A changed document's old entities/edges are dropped first (unless shared
        with surviving docs) so a re-extraction that no longer produces some
        artifact does not leave it orphaned in the graph. Call this before
        re-running extraction + :meth:`commit` on the changed documents.

        Returns True when every stale artifact was removed (or there was nothing
        to remove). The caller MUST NOT :meth:`commit` on False: commit replaces
        the changed docs' registry lineage with the new artifacts, so the old
        ids that failed to delete would no longer be referenced anywhere and
        could never be cleaned up (permanent orphans). Not committing keeps the
        old lineage and content hash, so the next run re-detects the docs as
        changed and retries the prune.
        """
        if not delta.changed:
            return True
        return self.remove_obsolete_artifacts(delta.changed)

    def _collect_exclusive_artifact_ids(
        self, doc_ids: list[str]
    ) -> dict[str, list[str]]:
        target = set(doc_ids)
        # Artifact ids referenced by SURVIVING documents -> keep them. Tracked
        # PER SUFFIX: artifact ids are suffix-independent (an entity "Vendor"
        # yields the same uuid5 id in every tenant), so a global retained set
        # would let a surviving doc in tenant B suppress the deletion of the same
        # id in tenant A. Only a same-suffix survivor should retain an id.
        retained_by_suffix: dict[str, set[str]] = defaultdict(set)
        removing_by_suffix: dict[str, set[str]] = defaultdict(set)
        for record in self.doc_status.list_all():
            ids = (
                record.entity_ids
                + record.relationship_ids
                + record.text_unit_ids
                + record.community_ids
                + record.claim_ids
                + record.community_report_ids
            )
            if record.doc_id in target:
                removing_by_suffix[record.suffix].update(ids)
            else:
                retained_by_suffix[record.suffix].update(ids)

        result: dict[str, list[str]] = {}
        for suffix, suffix_ids in removing_by_suffix.items():
            exclusive = sorted(suffix_ids - retained_by_suffix[suffix])
            if exclusive:
                result[suffix] = exclusive
        return result

    def _record_processed(
        self,
        lineages: list[DocumentLineage],
        fingerprints: dict[str, str],
        failed_doc_ids: Collection[str] = (),
    ) -> None:
        if failed_doc_ids:
            logger.warning(
                "Extraction failed on part of %d documents; recording them FAILED "
                "so the next run re-extracts them",
                len(failed_doc_ids),
            )
        for lineage in lineages:
            failed = lineage.doc_id in failed_doc_ids
            existing = self.doc_status.get(lineage.doc_id)
            record = DocStatusRecord(
                doc_id=lineage.doc_id,
                content_hash=fingerprints.get(
                    lineage.doc_id, existing.content_hash if existing else ""
                ),
                status=DocStatus.FAILED if failed else DocStatus.PROCESSED,
                error_info=("extraction failed on some text units" if failed else None),
                suffix=lineage.suffix,
                scope=(
                    self.scope
                    if self.scope is not None
                    else (existing.scope if existing else None)
                ),
                file_path=lineage.file_path,
                entity_ids=lineage.entity_ids,
                relationship_ids=lineage.relationship_ids,
                text_unit_ids=lineage.text_unit_ids,
                community_ids=lineage.community_ids,
                claim_ids=lineage.claim_ids,
                community_report_ids=lineage.community_report_ids,
            )
            self.doc_status.put(record)
