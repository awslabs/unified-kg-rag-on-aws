# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-run delta detection for incremental indexing.

Given the freshly loaded corpus and a :class:`DocStatusPort`, computes a stable
``doc_id`` and ``content_hash`` per document and classifies the corpus into
new / changed / unchanged / deleted / failed (:class:`DocumentDelta`).

``doc_id`` is derived from the document's index namespace, its corpus source
scope and its path relative to the corpus root (not the parser-assigned
``document_id``, which changes with the content) so the same file maps to the
same registry entry on every run, wherever the corpus is staged, and two
corpora written to one namespace never share an entry. ``content_hash`` is
computed over the aggregated document text so a content edit is detected even
when the path is unchanged.

Deletion is scoped: a run only classifies registry documents of its own scope
(index namespace + corpus source, see :func:`registry_scope`) as deleted, and
never a corpus file that merely failed to parse this run.

Registries written before the source scope was part of ``doc_id`` are keyed
by namespace and path only. :func:`detect_delta` adopts such a record under
the new key when it belongs to the run's scope (or predates scopes), so an
upgrade neither re-extracts nor deletes the corpus.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from unified_kg_rag.domain.models import (
    Constants,
    DocStatus,
    Document,
    DocumentDelta,
)
from unified_kg_rag.ports import DocStatusPort
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.document_identity import (
    REGISTRY_NAMESPACE_KEY,
    REGISTRY_SOURCE_KEY,
    RELATIVE_PATH_KEY,
    compute_doc_id,
    compute_document_id,
    compute_text_hash,
    normalize_source_path,
    registry_namespace,
    relative_source_path,
)

__all__ = [
    "adopt_legacy_records",
    "assign_document_identity",
    "assign_registry_source",
    "compute_content_hash",
    "compute_doc_id",
    "detect_delta",
    "document_doc_id",
    "filter_documents_to_process",
    "fingerprint_documents",
    "legacy_doc_id",
    "registry_scope",
]

logger = get_logger(__name__)


def compute_content_hash(document: Document) -> str:
    """Compute a content hash for change detection over the document text."""
    text = ""
    if document.content and document.content.text:
        text = document.content.text
    elif document.page_content:
        text = document.page_content
    return compute_text_hash(text)


def _document_suffix(document: Document) -> str | None:
    """The ``index`` metadata value the document's artifacts inherit."""
    value = document.metadata.get(Constants.INDEX.value)
    if isinstance(value, str) and value:
        return value
    if isinstance(value, (list, tuple)) and value:
        return str(value[0])
    return None


def assign_document_identity(
    document: Document,
    source_root: str | Path,
    additional_suffix: str | None = None,
) -> None:
    """Re-derive ``document``'s identity against the corpus root, in place.

    Records the path relative to ``source_root`` and the registry namespace in
    the metadata, and sets ``document_id`` from that path plus the full-content
    hash, so the id does not depend on where the corpus was synced or checked
    out and two files with the same name in different folders never share an
    id (or text-unit ids, which derive from it).

    A document whose ``file_path`` lies outside the root (a pre-parsed JSON
    document records its original path) keeps the relative path the loader
    recorded for the file it was read from.
    """
    relative = (
        relative_source_path(document.file_path, source_root)
        or document.metadata.get(RELATIVE_PATH_KEY)
        or normalize_source_path(document.file_path)
    )
    document.metadata[RELATIVE_PATH_KEY] = relative
    document.metadata[REGISTRY_NAMESPACE_KEY] = registry_namespace(
        _document_suffix(document), additional_suffix
    )
    document.document_id = compute_document_id(relative, compute_content_hash(document))


def assign_registry_source(document: Document, source_scope: str) -> None:
    """Record the corpus source scope ``document`` was read from, in place.

    Part of the registry key (see :func:`document_doc_id`), so two corpora
    written to one index namespace keep separate records even for files with
    the same relative path.
    """
    document.metadata[REGISTRY_SOURCE_KEY] = source_scope


def _identity(document: Document) -> tuple[str, str]:
    relative = document.metadata.get(RELATIVE_PATH_KEY) or document.file_path
    namespace = document.metadata.get(REGISTRY_NAMESPACE_KEY) or registry_namespace(
        _document_suffix(document)
    )
    return relative, namespace


def document_doc_id(document: Document) -> str:
    """The registry key of ``document`` (see :func:`compute_doc_id`).

    Uses the identity :func:`assign_document_identity` and
    :func:`assign_registry_source` recorded; a document that never went
    through them (built directly by a caller) is keyed by its ``file_path`` as
    given, the namespace of its ``index`` metadata and no source scope.
    """
    relative, namespace = _identity(document)
    return compute_doc_id(
        relative, namespace, document.metadata.get(REGISTRY_SOURCE_KEY)
    )


def legacy_doc_id(document: Document) -> str:
    """The key ``document`` had before the source scope joined the registry
    key (namespace and relative path only)."""
    relative, namespace = _identity(document)
    return compute_doc_id(relative, namespace)


def registry_scope(namespace: str, source_scope: str) -> str:
    """The scope a run diffs in: its index namespace and its corpus source.

    Two runs share a scope only when they write the same index namespace from
    the same corpus source, so one tenant's run, or a run over a subfolder,
    never classifies another scope's documents as deleted.
    """
    return f"{namespace}|{source_scope}"


def fingerprint_documents(documents: list[Document]) -> dict[str, str]:
    """Map each document to ``{doc_id: content_hash}`` for diffing.

    When two documents resolve to the same ``doc_id`` (same source path), the
    last one wins, matching load-order precedence.
    """
    fingerprints: dict[str, str] = {}
    for document in documents:
        fingerprints[document_doc_id(document)] = compute_content_hash(document)
    return fingerprints


def detect_delta(
    documents: list[Document],
    doc_status: DocStatusPort,
    scope: str | None = None,
    failed_doc_ids: Iterable[str] = (),
    max_failures: int | None = None,
    legacy_doc_ids: Mapping[str, str] | None = None,
) -> tuple[DocumentDelta, dict[str, str]]:
    """Classify ``documents`` against the persisted registry.

    Args:
        documents: The documents read successfully this run.
        doc_status: The registry.
        scope: The run's :func:`registry_scope`. Only stored documents of this
            scope can be classified deleted. ``None`` diffs against the whole
            registry (single-corpus deployments that predate scopes).
        failed_doc_ids: Registry ids of corpus files that failed to parse or
            load this run. They are reported as ``failed`` and never deleted.
        max_failures: A document whose unchanged content was recorded FAILED
            this many consecutive times is classified unchanged instead of
            changed, so a deterministic failure is not re-extracted every run.
            ``None`` retries without limit.
        legacy_doc_ids: ``{doc_id: legacy_doc_id}`` for this run's documents
            and failed files (see :func:`legacy_doc_id`). With a ``scope``, a
            record stored under the legacy key whose scope is this run's (or
            that predates scopes) is re-keyed to ``doc_id`` before the diff
            (see :func:`adopt_legacy_records`), so it keeps its content hash
            and lineage instead of reading as new plus deleted. This is the
            only write this function makes.

    Returns:
        The :class:`DocumentDelta` plus the ``{doc_id: content_hash}``
        fingerprint map (so callers can persist new/changed records without
        recomputing hashes).
    """
    fingerprints = fingerprint_documents(documents)
    delta = (
        doc_status.diff(fingerprints)
        if scope is None
        else doc_status.diff(fingerprints, scope=scope)
    )
    if (
        scope is not None
        and legacy_doc_ids
        and adopt_legacy_records(doc_status, delta, legacy_doc_ids, scope)
    ):
        # The adopted records now sit under this run's keys: diff again so
        # they classify like any other stored document (one extra scan, only
        # on the first run after an upgrade).
        delta = doc_status.diff(fingerprints, scope=scope)
    if max_failures is not None and delta.changed:
        _stop_retrying_exhausted(delta, doc_status, fingerprints, max_failures)
    failed = set(failed_doc_ids) - set(fingerprints)
    if failed:
        delta.deleted = [doc_id for doc_id in delta.deleted if doc_id not in failed]
        delta.failed = sorted(failed)
        logger.warning(
            "%d corpus files failed to parse or load; their indexed content is "
            "kept (not treated as deleted) until a run reads them",
            len(failed),
        )
    logger.info(
        "Delta detected: %d new, %d changed, %d unchanged, %d deleted, %d failed",
        len(delta.new),
        len(delta.changed),
        len(delta.unchanged),
        len(delta.deleted),
        len(delta.failed),
    )
    return delta, fingerprints


def adopt_legacy_records(
    doc_status: DocStatusPort,
    delta: DocumentDelta,
    legacy_doc_ids: Mapping[str, str],
    scope: str,
) -> int:
    """Re-key this scope's legacy-keyed records to the current ``doc_id``.

    A record stored under the legacy key is adopted when its scope is
    ``scope``, or ``None`` (written before scopes existed, which the legacy key
    matched for any run): it is written under the current key with its content
    hash, status and lineage, then the legacy key is deleted. Only ids
    without a current record (``delta.new`` and the failed files) are looked
    up, plus those whose legacy key ``delta`` lists as deleted: those already
    have a current record (an adoption interrupted between the write and the
    delete), so only the stale legacy key is deleted. A legacy record of
    another scope is left untouched, so that scope adopts it on its own run.

    Returns:
        The number of legacy keys adopted or deleted.
    """
    current = set(delta.changed) | set(delta.unchanged)
    deleted = set(delta.deleted)
    adopted: list[str] = []
    for doc_id, legacy_id in sorted(legacy_doc_ids.items()):
        if legacy_id == doc_id or (doc_id in current and legacy_id not in deleted):
            continue
        legacy = doc_status.get(legacy_id)
        if legacy is None or legacy.scope not in (scope, None):
            continue
        if doc_id not in current and doc_status.get(doc_id) is None:
            doc_status.put(legacy.model_copy(update={"doc_id": doc_id, "scope": scope}))
        doc_status.delete(legacy_id)
        adopted.append(legacy.file_path or doc_id)
    if adopted:
        logger.info(
            "Adopted %d doc-status records keyed without the source scope "
            "(written before it was part of the key), e.g. %s",
            len(adopted),
            ", ".join(adopted[:10]),
        )
    return len(adopted)


def _stop_retrying_exhausted(
    delta: DocumentDelta,
    doc_status: DocStatusPort,
    fingerprints: dict[str, str],
    max_failures: int,
) -> None:
    """Move changed documents that used up their retries to ``unchanged``."""
    exhausted: list[str] = []
    labels: list[str] = []
    for doc_id in delta.changed:
        record = doc_status.get(doc_id)
        if (
            record is not None
            and record.status is DocStatus.FAILED
            and record.content_hash == fingerprints[doc_id]
            and record.failure_count >= max_failures
        ):
            exhausted.append(doc_id)
            labels.append(record.file_path or doc_id)
    if not exhausted:
        return
    delta.changed = [doc_id for doc_id in delta.changed if doc_id not in exhausted]
    delta.unchanged.extend(exhausted)
    logger.warning(
        "Not retrying %d documents that failed %d consecutive runs with unchanged "
        "content (kept FAILED; edit the file or raise "
        "indexing.max_document_failures to retry): %s",
        len(exhausted),
        max_failures,
        ", ".join(labels),
    )


def filter_documents_to_process(
    documents: list[Document], delta: DocumentDelta
) -> list[Document]:
    """Return only the documents that need (re)indexing this run (new + changed)."""
    to_process = set(delta.to_process)
    return [
        document for document in documents if document_doc_id(document) in to_process
    ]
