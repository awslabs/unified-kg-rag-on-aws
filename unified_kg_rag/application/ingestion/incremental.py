# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental-indexing orchestrator.

Ties together the M2 pieces around a :class:`DocStatusPort`:

1. detect the corpus delta (new / changed / unchanged / deleted),
2. write ahead: record every new and changed document PENDING, with a
   content hash no document has and the union of its stored and planned
   lineage (the part over the registry's item limit in lineage overflow),
   before any store write (:meth:`IncrementalIndexer.write_ahead`),
3. for deleted (and changed) documents, remove their previously indexed
   artifacts from the live stores using the lineage recorded in the registry,
4. upsert the freshly extracted delta artifacts (idempotent),
5. commit: replace the PENDING records with the new content hashes and
   artifact lineage.

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
from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from unified_kg_rag.domain.ingestion.delta_detector import (
    detect_delta,
    document_doc_id,
    scope_namespace,
)
from unified_kg_rag.domain.models import (
    PENDING_CONTENT_HASH,
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
from unified_kg_rag.shared import DataProcessingError, get_logger
from unified_kg_rag.shared.utils.document_identity import (
    RELATIVE_PATH_KEY,
    compute_doc_id,
)

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


class _SuffixRemoval(BaseModel):
    """What removing some documents does to one suffix's artifacts."""

    # Artifacts no surviving document references: deleted.
    exclusive_ids: list[str] = Field(default_factory=list)
    # Entities/relationships a survivor still references: kept, but stripped
    # of the removed text units.
    shared_entity_ids: list[str] = Field(default_factory=list)
    shared_relationship_ids: list[str] = Field(default_factory=list)
    removed_text_unit_ids: list[str] = Field(default_factory=list)


_EXTRACTION_FAILED = "extraction failed on some text units"
_WRITE_FAILED = "writing some of its artifacts to the stores failed"
_WRITE_AHEAD_INFO = (
    "indexing in progress (write-ahead record); a run that ends with this "
    "record re-extracts the document, or removes its artifacts if it is gone"
)
_LINEAGE_FIELDS = (
    "entity_ids",
    "relationship_ids",
    "text_unit_ids",
    "community_ids",
    "claim_ids",
    "community_report_ids",
)


def _artifact_ids(record: DocStatusRecord | DocumentLineage) -> list[str]:
    return (
        record.entity_ids
        + record.relationship_ids
        + record.text_unit_ids
        + record.community_ids
        + record.claim_ids
        + record.community_report_ids
    )


# (item suffix, index namespace or None when the record has no scope).
_NamespaceKey = tuple[str, str | None]


def _namespace_key(
    record: DocStatusRecord, known: Collection[str] = ()
) -> _NamespaceKey:
    """The removal-planning group of ``record``.

    ``record.suffix`` is the item suffix the indexers are called with
    (``index_value``, ``default`` when unset); the index namespace adds
    ``indexing.additional_suffix`` and is read from the record's scope (see
    ``scope_namespace``).

    A record without a scope has its namespace only when its key proves it:
    a ``file_path`` whose legacy key ``compute_doc_id(file_path, namespace)``
    is its ``doc_id`` for one of the ``known`` namespaces. Any other
    scope-less record is of an unknown namespace (``None``), which removal
    planning treats as part of every namespace of its suffix: a record of
    the initial release (keyed by the path alone, no ``file_path``), a
    pipeline record of a namespace no scope in the registry names yet, a
    record of the direct API without a scope (keyed by the bare suffix,
    whatever ``indexing.additional_suffix`` the indexers write under) or
    one under a caller's own ``doc_id``.
    """
    if record.scope is not None:
        return record.suffix, scope_namespace(record.scope)
    if record.file_path is not None:
        for namespace in known:
            if record.doc_id == compute_doc_id(record.file_path, namespace):
                return record.suffix, namespace
    return record.suffix, None


def _read_many(
    doc_status: DocStatusPort, doc_ids: Iterable[str]
) -> dict[str, DocStatusRecord]:
    """``doc_status.get_many``; one ``get`` per id for a custom store that
    predates ``get_many``."""
    ids = list(dict.fromkeys(doc_ids))
    if not ids:
        return {}
    get_many = getattr(doc_status, "get_many", None)
    if get_many is None:
        records = {doc_id: doc_status.get(doc_id) for doc_id in ids}
        return {doc_id: r for doc_id, r in records.items() if r is not None}
    return dict(get_many(ids))


def _write_many(doc_status: DocStatusPort, records: list[DocStatusRecord]) -> None:
    """``doc_status.put_many``; one ``put`` per record for a custom store that
    predates ``put_many``."""
    if not records:
        return
    put_many = getattr(doc_status, "put_many", None)
    if put_many is None:
        for record in records:
            doc_status.put(record)
    else:
        put_many(records)


def _union_lineage(
    record: DocStatusRecord | DocumentLineage,
    lineage: DocumentLineage | DocStatusRecord | None,
) -> dict[str, list[str]]:
    """``record``'s artifact ids plus ``lineage``'s, per lineage field."""
    if lineage is None:
        return {name: list(getattr(record, name)) for name in _LINEAGE_FIELDS}
    return {
        name: sorted(set(getattr(record, name)) | set(getattr(lineage, name)))
        for name in _LINEAGE_FIELDS
    }


def _lineage_minus(
    lineage: DocStatusRecord | DocumentLineage,
    *others: DocStatusRecord | DocumentLineage | None,
) -> dict[str, list[str]]:
    """``lineage``'s artifact ids that none of ``others`` lists, per field."""
    return {
        name: sorted(
            set(getattr(lineage, name)).difference(
                *(getattr(other, name) for other in others if other is not None)
            )
        )
        for name in _LINEAGE_FIELDS
    }


def _lineage(doc_id: str, ids: dict[str, list[str]]) -> DocumentLineage:
    """A lineage of ``doc_id`` with ``{lineage field: ids}``."""
    return DocumentLineage.model_validate({"doc_id": doc_id, **ids})


def _record_fits(doc_status: DocStatusPort, record: DocStatusRecord) -> bool:
    """``doc_status.record_fits``; always True for a custom store that
    predates it (it has no item limit the run knows of)."""
    fits = getattr(doc_status, "record_fits", None)
    return True if fits is None else bool(fits(record))


def _read_overflow(
    doc_status: DocStatusPort, doc_ids: Iterable[str]
) -> dict[str, DocumentLineage]:
    """``doc_status.get_lineage_overflow``; none for a custom store that
    predates it."""
    ids = list(dict.fromkeys(doc_ids))
    get = getattr(doc_status, "get_lineage_overflow", None)
    if not ids or get is None:
        return {}
    return dict(get(ids))


def _delete_overflow(doc_status: DocStatusPort, doc_ids: Iterable[str]) -> None:
    """``doc_status.delete_lineage_overflow``; a no-op for a custom store
    that predates it."""
    ids = list(dict.fromkeys(doc_ids))
    delete = getattr(doc_status, "delete_lineage_overflow", None)
    if ids and delete is not None:
        delete(ids)


def _remap_lineage(
    lineage: DocumentLineage, id_remap: dict[str, str]
) -> DocumentLineage:
    """Point a lineage's entity/relationship ids at the ids the merge kept."""
    if not id_remap:
        return lineage
    return lineage.model_copy(
        update={
            "entity_ids": sorted({id_remap.get(i, i) for i in lineage.entity_ids}),
            "relationship_ids": sorted(
                {id_remap.get(i, i) for i in lineage.relationship_ids}
            ),
        }
    )


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
        # Write-ahead state of the current run (see write_ahead): the records
        # stored before it (with an interrupted run's lineage overflow), the
        # PENDING records written, the ids held in lineage overflow for them,
        # and the documents being (re)indexed.
        self._prior: dict[str, DocStatusRecord] = {}
        self._pending: dict[str, DocStatusRecord] = {}
        self._overflow: dict[str, DocumentLineage] = {}
        self._in_flight: set[str] = set()

    def plan(
        self,
        documents: list[Document],
        failed_doc_ids: list[str] | None = None,
        max_failures: int | None = None,
    ) -> tuple[DocumentDelta, dict[str, str]]:
        """Compute the delta for ``documents`` without mutating any store."""
        return detect_delta(
            documents,
            self.doc_status,
            self.scope,
            failed_doc_ids or (),
            max_failures=max_failures,
        )

    def write_ahead(
        self, delta: DocumentDelta, lineages: list[DocumentLineage]
    ) -> None:
        """Record the documents about to be (re)indexed PENDING, before any
        store write.

        Call after extraction and before removing or writing anything for the
        delta. Each new and changed document (and any other document with a
        lineage) gets a PENDING record with the content hash
        ``PENDING_CONTENT_HASH`` and the union of its stored lineage and the
        lineage this run is about to write (``lineages``), under its current
        key and the run's scope (the stored scope when the run has none), so:

        - an interrupted run leaves a record the next diff classifies changed
          (never unchanged, even when the document reverted to the indexed
          content), or deleted when the document is gone;
        - removing it then finds every artifact this run may have written.

        A union over the store's item limit (``DocStatusPort.record_fits``;
        a document whose stored and planned lineage each fit, but not
        together) keeps the stored lineage in the record and the other
        planned ids in lineage overflow (``add_lineage_overflow``, written
        after the record and still before any store write), which a later
        run reads back with the record (only a stored PENDING record's is
        read) and which the commit and the removal of the document delete. A document whose planned lineage alone
        cannot fit in a record raises :class:`DataProcessingError` here,
        before anything is written.

        Overflow the diff found left over (``delta.orphan_overflow``: next to
        a record that is not PENDING, after a commit or removal interrupted
        before deleting it) is deleted first when its owner is a record of
        the run's namespace (any record when the run has no scope) or a
        document of this run, as re-read here. Overflow of another
        namespace's owner, or of an owner whose record is gone, is left
        alone.

        :meth:`commit` replaces the records with the real ones. A PENDING
        record has ``failure_count`` 0 and is never FAILED: an interruption is
        not counted as a failure, but it resets the consecutive-failure count
        of a document that had failed before (the next run counts from the
        PENDING record). The count of a failure in this run continues from
        the record stored before the PENDING one. The stored records are read
        once (batched) and kept for :meth:`commit`, which therefore neither
        re-reads them nor takes the PENDING record for the previous state.
        """
        planned = {lineage.doc_id: lineage for lineage in lineages}
        doc_ids = list(dict.fromkeys([*delta.to_process, *planned]))
        if not doc_ids and not delta.orphan_overflow:
            return
        read = _read_many(self.doc_status, [*doc_ids, *delta.orphan_overflow])
        self._collect_leftover_overflow(delta.orphan_overflow, read, set(doc_ids))
        if not doc_ids:
            return
        stored = {doc_id: read[doc_id] for doc_id in doc_ids if doc_id in read}
        self._check_committable(stored, planned)
        # Overflow belongs to a PENDING record (an interrupted run's): its ids
        # are part of that record's lineage, so only those documents' is
        # read. Next to any other record, or none, it is left over from a
        # commit or removal interrupted before deleting it, and was collected
        # above (delta.orphan_overflow).
        kept = _read_overflow(
            self.doc_status,
            [
                doc_id
                for doc_id, record in stored.items()
                if record.status is DocStatus.PENDING
            ],
        )
        self._prior = {
            doc_id: (
                record.model_copy(update=_union_lineage(record, kept[doc_id]))
                if doc_id in kept
                else record
            )
            for doc_id, record in stored.items()
        }
        self._in_flight = set(doc_ids)
        self._pending = {}
        self._overflow = {}
        additions: list[DocumentLineage] = []
        for doc_id in doc_ids:
            full = self._pending_record(
                doc_id, self._prior.get(doc_id), planned.get(doc_id)
            )
            record, spilled = self._fit(full, stored.get(doc_id))
            self._pending[doc_id] = record
            if spilled is None:
                continue
            self._overflow[doc_id] = spilled
            added = _lineage_minus(spilled, kept.get(doc_id))
            if any(added.values()):
                additions.append(_lineage(doc_id, added))
        _write_many(self.doc_status, list(self._pending.values()))
        if additions:
            self.doc_status.add_lineage_overflow(additions)
        # A record that now holds its interrupted run's overflow ids itself.
        _delete_overflow(
            self.doc_status, [doc_id for doc_id in kept if doc_id not in self._overflow]
        )
        logger.info(
            "Recorded %d documents PENDING before indexing the delta%s",
            len(self._pending),
            (
                f" ({len(self._overflow)} with lineage overflow)"
                if self._overflow
                else ""
            ),
        )

    def _collect_leftover_overflow(
        self,
        owners: list[str],
        records: dict[str, DocStatusRecord],
        documents: set[str],
    ) -> None:
        """Delete the leftover overflow of ``owners`` this run may delete.

        ``records`` are the owners' records as read now: an owner that
        became PENDING since the diff has its overflow read with it. An
        owner without a record is only this run's when it is one of its
        ``documents`` (keyed in its namespace).
        """
        namespace = scope_namespace(self.scope)
        leftover = []
        for owner in owners:
            record = records.get(owner)
            if record is None:
                if owner in documents:
                    leftover.append(owner)
                continue
            if record.status is DocStatus.PENDING:
                continue
            if self.scope is None or (
                namespace is not None
                and _namespace_key(record, [namespace])[1] == namespace
            ):
                leftover.append(owner)
        if leftover:
            logger.info(
                "Deleting the leftover lineage overflow of %d documents "
                "(an interrupted commit or removal did not delete it)",
                len(leftover),
            )
            _delete_overflow(self.doc_status, leftover)

    def _fit(
        self, full: DocStatusRecord, stored: DocStatusRecord | None
    ) -> tuple[DocStatusRecord, DocumentLineage | None]:
        """``full`` as stored: itself when it fits, else a record listing the
        stored lineage (or none) plus the overflow holding the rest."""
        if _record_fits(self.doc_status, full):
            return full, None
        empty = DocumentLineage(doc_id=full.doc_id)
        for base in (stored, empty):
            if base is None:
                continue
            record = full.model_copy(
                update={name: list(getattr(base, name)) for name in _LINEAGE_FIELDS}
            )
            if _record_fits(self.doc_status, record):
                return record, _lineage(full.doc_id, _lineage_minus(full, record))
        raise DataProcessingError(
            f"Doc-status record for '{full.file_path or full.doc_id}' does not "
            "fit the registry's item limit even without artifact ids"
        )

    def _check_committable(
        self, stored: dict[str, DocStatusRecord], planned: dict[str, DocumentLineage]
    ) -> None:
        """Raise before any write when a document's commit record cannot fit.

        The commit records each document with this run's lineage; one over
        the store's item limit would fail after the stores were written, and
        so on every later run. Checked with the longer (FAILED) record.
        """
        too_large = []
        for doc_id, lineage in planned.items():
            existing = stored.get(doc_id)
            record = DocStatusRecord(
                doc_id=doc_id,
                content_hash="0" * 64,
                status=DocStatus.FAILED,
                error_info=_WRITE_FAILED,
                failure_count=1,
                suffix=lineage.suffix,
                scope=(
                    self.scope
                    if self.scope is not None
                    else (existing.scope if existing else None)
                ),
                file_path=lineage.file_path,
                **{name: getattr(lineage, name) for name in _LINEAGE_FIELDS},
            )
            if not _record_fits(self.doc_status, record):
                too_large.append(
                    f"{lineage.file_path or doc_id} "
                    f"({len(_artifact_ids(lineage))} artifact ids)"
                )
        if too_large:
            raise DataProcessingError(
                f"{len(too_large)} documents produce more artifact ids than one "
                "doc-status registry record holds (DynamoDB: 400 KB, roughly "
                "10,000 ids); split them into smaller files. Nothing was "
                "written for this run's delta: " + ", ".join(too_large[:10])
            )

    def _pending_record(
        self,
        doc_id: str,
        prior: DocStatusRecord | None,
        lineage: DocumentLineage | None,
    ) -> DocStatusRecord:
        base = prior or DocStatusRecord(
            doc_id=doc_id,
            content_hash=PENDING_CONTENT_HASH,
            suffix=lineage.suffix if lineage else self.suffix,
        )
        return base.model_copy(
            update={
                "content_hash": PENDING_CONTENT_HASH,
                "status": DocStatus.PENDING,
                "error_info": _WRITE_AHEAD_INFO,
                "failure_count": 0,
                # The stored suffix is kept: removal groups by it, and a
                # changed document's old artifacts were written under it.
                "scope": self.scope if self.scope is not None else base.scope,
                "file_path": (lineage.file_path if lineage else None) or base.file_path,
                **_union_lineage(base, lineage),
            }
        )

    def _extend_pending(self, lineages: list[DocumentLineage]) -> None:
        """Add the ids the cross-run merge kept to the PENDING lineage (to the
        record, or to its overflow when it has some or would not fit)."""
        updated = []
        additions = []
        for lineage in lineages:
            doc_id = lineage.doc_id
            pending = self._pending.get(doc_id)
            if pending is None:
                continue
            held = self._overflow.get(doc_id)
            added = _lineage_minus(lineage, pending, held)
            if not any(added.values()):
                continue
            if held is None:
                extended = pending.model_copy(update=_union_lineage(pending, lineage))
                if _record_fits(self.doc_status, extended):
                    self._pending[doc_id] = extended
                    updated.append(extended)
                    continue
                held = DocumentLineage(doc_id=doc_id)
            self._overflow[doc_id] = held.model_copy(
                update=_union_lineage(held, _lineage(doc_id, added))
            )
            additions.append(_lineage(doc_id, added))
        _write_many(self.doc_status, updated)
        if additions:
            self.doc_status.add_lineage_overflow(additions)

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

        Artifacts the documents share with survivors are kept, but the text
        units being removed are stripped from them and their frequency/weight
        recomputed, so they no longer cite chunks that no longer exist. Their
        descriptions keep the removed documents' contribution: re-deriving a
        description needs an LLM call (a full rebuild does it).
        """
        if not doc_ids:
            return True

        removals = self._plan_removal(doc_ids)
        exclusive_by_suffix = {
            suffix: removal.exclusive_ids
            for suffix, removal in removals.items()
            if removal.exclusive_ids
        }
        if not exclusive_by_suffix:
            return True

        total = sum(len(ids) for ids in exclusive_by_suffix.values())
        logger.info(
            "Removing %d artifacts for %d obsolete documents",
            total,
            len(doc_ids),
        )
        results = list(
            self.indexing_manager.delete_documents(exclusive_by_suffix).values()
        )
        for suffix, removal in removals.items():
            if removal.removed_text_unit_ids and (
                removal.shared_entity_ids or removal.shared_relationship_ids
            ):
                results.extend(
                    self.indexing_manager.remove_text_units_from_shared(
                        suffix,
                        entity_ids=removal.shared_entity_ids,
                        relationship_ids=removal.shared_relationship_ids,
                        text_unit_ids=removal.removed_text_unit_ids,
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

        ``lineages`` attributes artifacts to their source document (one entry
        per processed doc), so per-document deletion later removes only a doc's
        *exclusive* artifacts. The artifact lists passed separately are the union
        actually written to the stores this run. Docs in ``failed_doc_ids`` (an
        extraction stage failed on some of their text units) are recorded
        FAILED with their lineage, so the next run re-extracts them and first
        prunes what this run wrote.

        Entities and relationships are first merged with the stored graph state
        they touch (``indexing.cross_run_merge``); a delta item that merged into
        a stored one under another id is recorded in the lineage by that id
        (and added to the document's PENDING record before the write, see
        :meth:`write_ahead`).
        """
        merged = self.indexing_manager.merge_with_existing_graph(
            entities, relationships
        )
        entities, relationships = merged.entities, merged.relationships
        lineages = [
            _remap_lineage(lineage, merged.id_remap_by_suffix.get(lineage.suffix, {}))
            for lineage in lineages
        ]
        # A stored id the merge kept in place of a delta id is written too:
        # the PENDING record must reference it before the write.
        self._extend_pending(lineages)
        results = self.indexing_manager.index_delta(
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            communities=communities,
            community_reports=community_reports,
            claims=claims,
        )
        self.record(lineages, fingerprints, results, failed_doc_ids)
        return results

    def record(
        self,
        lineages: list[DocumentLineage],
        fingerprints: dict[str, str],
        results: dict[str, IndexingStats],
        failed_doc_ids: Collection[str] = (),
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
        if self._delta_writes_succeeded(results):
            write_failed = self._documents_with_failed_writes(lineages, results)
            self._record_processed(lineages, fingerprints, failed_doc_ids, write_failed)
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
        lineages: list[DocumentLineage], results: dict[str, IndexingStats]
    ) -> set[str]:
        """Registry ids of the documents owning an artifact whose write failed.

        The failed artifact ids come from the indexers' stats. A failure the
        indexer reported without an id cannot be attributed, so every document
        of the commit is returned: retrying them all re-extracts more than
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
        owners = {
            lineage.doc_id
            for lineage in lineages
            if failed_ids.intersection(_artifact_ids(lineage))
        }
        logger.warning(
            "%d artifact writes failed; recording the %d documents that own "
            "them FAILED so the next run rewrites them",
            len(failed_ids),
            len(owners),
        )
        return owners

    def remove_deleted(self, delta: DocumentDelta) -> bool:
        """Remove artifacts and registry records for deleted documents.

        The registry record is only deleted when artifact removal fully
        succeeded. If removal partially failed, the record is kept so the next
        run retries the removal — deleting the record first would strand the
        still-live artifacts with no lineage to ever clean them up (permanent
        orphans).

        Returns True when every artifact was removed (or nothing was deleted).
        The caller must report False as a failed run: the stores still hold
        the deleted documents' content. A delta that also changes documents
        must use :meth:`remove_changed_and_deleted` instead.
        """
        if not delta.deleted:
            return True
        removed = self.remove_obsolete_artifacts(delta.deleted)
        if not removed:
            logger.warning(
                "Keeping %d deleted-doc registry records because artifact "
                "removal did not fully succeed; will retry next run.",
                len(delta.deleted),
            )
            return False
        self._delete_records(delta.deleted)
        return True

    def _delete_records(self, doc_ids: list[str]) -> None:
        """Delete removed documents' records, their lineage overflow first: an
        interruption in between leaves the record, listing artifacts that
        are already gone, rather than overflow nothing references."""
        _delete_overflow(self.doc_status, doc_ids)
        for doc_id in doc_ids:
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
        old lineage (inside the PENDING record when :meth:`write_ahead` ran),
        so the next run re-detects the docs as changed and retries the prune.

        A delta that also deletes documents must use
        :meth:`remove_changed_and_deleted` instead: pruning and deleting in
        separate passes keeps what only a changed and a deleted doc share.
        """
        if not delta.changed:
            return True
        return self.remove_obsolete_artifacts(delta.changed)

    def remove_changed_and_deleted(self, delta: DocumentDelta) -> bool:
        """Remove the stale artifacts of changed AND deleted docs in one pass.

        The removal is planned once over changed + deleted, so an artifact only
        those documents reference is removed even when it is referenced by one
        changed and one deleted document. Planning the two sets separately
        (:meth:`prune_changed` then :meth:`remove_deleted`) treats each set as
        a survivor of the other and leaves such artifacts behind as orphans
        with no text units.

        The deleted documents' registry records are deleted only when the
        removal fully succeeded (see :meth:`remove_deleted`). Returns False on
        a partial failure; the caller MUST NOT :meth:`commit` changed
        documents on False, for the reason given in :meth:`prune_changed`.
        """
        doc_ids = list(dict.fromkeys(delta.changed + delta.deleted))
        if not doc_ids:
            return True
        if not self.remove_obsolete_artifacts(doc_ids):
            if delta.deleted:
                logger.warning(
                    "Keeping %d deleted-doc registry records because artifact "
                    "removal did not fully succeed; will retry next run.",
                    len(delta.deleted),
                )
            return False
        self._delete_records(delta.deleted)
        return True

    def _plan_removal(self, doc_ids: list[str]) -> dict[str, _SuffixRemoval]:
        target = set(doc_ids)
        # Artifact ids referenced by SURVIVING documents -> keep them. Tracked
        # PER NAMESPACE: artifact ids are namespace-independent (an entity
        # "Vendor" yields the same uuid5 id in every tenant), so a global
        # retained set would let a surviving doc in tenant B suppress the
        # deletion of the same id in tenant A. Only a survivor of the same
        # namespace should retain an id. The namespace is read from the
        # record's scope, not from ``record.suffix``: that is the item suffix
        # the indexers are called with, which every
        # ``indexing.additional_suffix`` shares (see _namespace_key).
        retained: dict[_NamespaceKey, set[str]] = defaultdict(set)
        removing: dict[_NamespaceKey, list[DocStatusRecord]] = defaultdict(list)
        records = self.doc_status.list_all()
        # Another run's interrupted PENDING record may keep part of its
        # lineage in overflow (this run's documents have theirs in _prior).
        overflow = _read_overflow(
            self.doc_status,
            [
                r.doc_id
                for r in records
                if r.status is DocStatus.PENDING and r.doc_id not in self._in_flight
            ],
        )
        planned: list[DocStatusRecord] = []
        for record in records:
            if record.doc_id in overflow:
                record = record.model_copy(
                    update=_union_lineage(record, overflow[record.doc_id])
                )
            if record.doc_id in self._in_flight:
                # This run's PENDING record also lists what the run writes
                # after this removal: plan from the record stored before it
                # (after an interrupted run, that run's PENDING record, so
                # whatever it wrote is removed). A new document has none and
                # neither removes nor retains anything.
                prior = self._prior.get(record.doc_id)
                if prior is None:
                    continue
                record = prior
            planned.append(record)
        # The namespaces a record without a scope is matched against (by its
        # legacy key): the run's and every one a scope in the registry names.
        known = sorted(
            {
                namespace
                for namespace in (
                    scope_namespace(self.scope),
                    *(scope_namespace(r.scope) for r in (*records, *planned)),
                )
                if namespace is not None
            }
        )
        for record in planned:
            key = _namespace_key(record, known)
            if record.doc_id in target:
                removing[key].append(record)
            else:
                retained[key].update(_artifact_ids(record))

        plan: dict[str, _SuffixRemoval] = {}
        for suffix in dict.fromkeys(key[0] for key in removing):
            records = [
                record
                for key, group in removing.items()
                if key[0] == suffix
                for record in group
            ]
            # The ids a survivor of any removed record's namespace keeps. A
            # record of an unknown namespace (see _namespace_key) may belong
            # to any namespace of its suffix: it retains for all of them, and
            # all of them retain for it.
            kept: set[str] = set()
            for key in removing:
                if key[0] != suffix:
                    continue
                for other, ids in retained.items():
                    if other[0] == suffix and (
                        key[1] is None or other[1] is None or other[1] == key[1]
                    ):
                        kept |= ids
            ids = {i for record in records for i in _artifact_ids(record)}
            text_units = {i for record in records for i in record.text_unit_ids}
            plan[suffix] = _SuffixRemoval(
                exclusive_ids=sorted(ids - kept),
                shared_entity_ids=sorted(
                    {i for record in records for i in record.entity_ids} & kept
                ),
                shared_relationship_ids=sorted(
                    {i for record in records for i in record.relationship_ids} & kept
                ),
                removed_text_unit_ids=sorted(text_units - kept),
            )
        return plan

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
            **_read_many(
                self.doc_status,
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
            records.append(record)
        _write_many(self.doc_status, records)
        # The committed records replace the PENDING ones: their overflow is
        # no longer referenced (deleted after the write, so an interruption
        # in between leaves overflow next to a committed record, which the
        # next write-ahead ignores and drops).
        committed = [r.doc_id for r in records if r.doc_id in self._overflow]
        _delete_overflow(self.doc_status, committed)
        for doc_id in committed:
            del self._overflow[doc_id]
        unrecorded = sorted(set(self._pending) - {r.doc_id for r in records})
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
