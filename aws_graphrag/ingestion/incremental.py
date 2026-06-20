# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
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

from aws_graphrag.core import get_logger
from aws_graphrag.core.ports import DocStatusPort
from aws_graphrag.ingestion.delta_detector import (
    compute_doc_id,
    detect_delta,
    filter_documents_to_process,
)
from aws_graphrag.models import (
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

if TYPE_CHECKING:
    from aws_graphrag.storage import IndexingManager, IndexingStats

logger = get_logger(__name__)


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
            )
            self.doc_status.put(record)

    @staticmethod
    def doc_id_for(document: Document) -> str:
        """Expose the stable doc-id derivation for callers/tests."""
        return compute_doc_id(document.file_path)
