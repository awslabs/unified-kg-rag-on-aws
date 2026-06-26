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
from typing import TYPE_CHECKING

from unified_kg_rag.domain.ingestion.delta_detector import (
    compute_doc_id,
    detect_delta,
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
    docid_to_stable = {
        doc.document_id: compute_doc_id(doc.file_path) for doc in documents
    }
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
    ) -> None:
        self.doc_status = doc_status
        self.indexing_manager = indexing_manager
        self.suffix = suffix

    def plan(self, documents: list[Document]) -> tuple[DocumentDelta, dict[str, str]]:
        """Compute the delta for ``documents`` without mutating any store."""
        return detect_delta(documents, self.doc_status)

    def documents_to_process(
        self, documents: list[Document], delta: DocumentDelta
    ) -> list[Document]:
        """Return the documents requiring extraction this run (new + changed)."""
        return filter_documents_to_process(documents, delta)

    def remove_obsolete_artifacts(self, doc_ids: list[str]) -> None:
        """Delete artifacts belonging only to the given (deleted/changed) docs.

        Lineage is read from the registry; only ids not still referenced by a
        *surviving* document are removed, so shared artifacts are preserved.
        Removals are grouped by the suffix each document's artifacts were
        written under (multi-tenant/multi-index safe).
        """
        if not doc_ids:
            return

        obsolete_by_suffix = self._collect_exclusive_artifact_ids(doc_ids)
        if obsolete_by_suffix:
            total = sum(len(ids) for ids in obsolete_by_suffix.values())
            logger.info(
                "Removing %d artifacts for %d obsolete documents",
                total,
                len(doc_ids),
            )
            self.indexing_manager.delete_documents(obsolete_by_suffix)

    def commit(
        self,
        lineages: list[DocumentLineage],
        fingerprints: dict[str, str],
        *,
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
        actually written to the stores this run.
        """
        results = self.indexing_manager.index_delta(
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            communities=communities,
            community_reports=community_reports,
            claims=claims,
        )
        self._record_processed(lineages, fingerprints)
        return results

    def remove_deleted(self, delta: DocumentDelta) -> None:
        """Remove artifacts and registry records for deleted documents."""
        if not delta.deleted:
            return
        self.remove_obsolete_artifacts(delta.deleted)
        for doc_id in delta.deleted:
            self.doc_status.delete(doc_id)

    def prune_changed(self, delta: DocumentDelta) -> None:
        """Remove the now-stale artifacts of changed docs before re-extraction.

        A changed document's old entities/edges are dropped first (unless shared
        with surviving docs) so a re-extraction that no longer produces some
        artifact does not leave it orphaned in the graph. Call this before
        re-running extraction + :meth:`commit` on the changed documents.
        """
        if delta.changed:
            self.remove_obsolete_artifacts(delta.changed)

    def _collect_exclusive_artifact_ids(
        self, doc_ids: list[str]
    ) -> dict[str, list[str]]:
        target = set(doc_ids)
        # Artifact ids referenced by documents NOT being removed -> keep them.
        retained: set[str] = set()
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
                retained.update(ids)

        result: dict[str, list[str]] = {}
        for suffix, suffix_ids in removing_by_suffix.items():
            exclusive = sorted(suffix_ids - retained)
            if exclusive:
                result[suffix] = exclusive
        return result

    def _record_processed(
        self, lineages: list[DocumentLineage], fingerprints: dict[str, str]
    ) -> None:
        for lineage in lineages:
            existing = self.doc_status.get(lineage.doc_id)
            record = DocStatusRecord(
                doc_id=lineage.doc_id,
                content_hash=fingerprints.get(
                    lineage.doc_id, existing.content_hash if existing else ""
                ),
                status=DocStatus.PROCESSED,
                suffix=lineage.suffix,
                entity_ids=lineage.entity_ids,
                relationship_ids=lineage.relationship_ids,
                text_unit_ids=lineage.text_unit_ids,
                community_ids=lineage.community_ids,
                claim_ids=lineage.claim_ids,
                community_report_ids=lineage.community_report_ids,
            )
            self.doc_status.put(record)
