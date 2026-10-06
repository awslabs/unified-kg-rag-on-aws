# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-run delta detection for incremental indexing.

Given the freshly loaded corpus and a :class:`DocStatusPort`, computes a stable
``doc_id`` and ``content_hash`` per document and classifies the corpus into
new / changed / unchanged / deleted (:class:`DocumentDelta`).

``doc_id`` is derived from the *source path* (not the parser-assigned
``document_id``, which is not stable across runs) so the same file maps to the
same registry entry on every run. ``content_hash`` is computed over the
aggregated document text so a content edit is detected even when the path is
unchanged.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from unified_kg_rag.domain.models import Document, DocumentDelta
from unified_kg_rag.ports import DocStatusPort
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.common import compute_hash
from unified_kg_rag.shared.utils.document_identity import (
    RELATIVE_PATH_KEY,
    compute_document_id,
    compute_text_hash,
    normalize_source_path,
    relative_source_path,
)

logger = get_logger(__name__)


def compute_doc_id(file_path: str) -> str:
    """Derive a stable document id from a source file path.

    Normalises separators and strips a leading ``./`` so the same logical file
    yields the same id regardless of how the path was expressed.
    """
    normalized = PurePosixPath(str(file_path).replace("\\", "/")).as_posix()
    return compute_hash(normalized, algorithm="sha256", length=32)


def compute_content_hash(document: Document) -> str:
    """Compute a content hash for change detection over the document text."""
    text = ""
    if document.content and document.content.text:
        text = document.content.text
    elif document.page_content:
        text = document.page_content
    return compute_text_hash(text)


def assign_document_identity(document: Document, source_root: str | Path) -> None:
    """Re-derive ``document``'s id against the corpus root, in place.

    Records the path relative to ``source_root`` in the metadata and sets
    ``document_id`` from that path plus the full-content hash, so the id does
    not depend on where the corpus was synced or checked out and two files
    with the same name in different folders never share an id (or text-unit
    ids, which derive from it).

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
    document.document_id = compute_document_id(relative, compute_content_hash(document))


def fingerprint_documents(documents: list[Document]) -> dict[str, str]:
    """Map each document to ``{doc_id: content_hash}`` for diffing.

    When two documents resolve to the same ``doc_id`` (same source path), the
    last one wins, matching load-order precedence.
    """
    fingerprints: dict[str, str] = {}
    for document in documents:
        doc_id = compute_doc_id(document.file_path)
        fingerprints[doc_id] = compute_content_hash(document)
    return fingerprints


def detect_delta(
    documents: list[Document], doc_status: DocStatusPort
) -> tuple[DocumentDelta, dict[str, str]]:
    """Classify ``documents`` against the persisted registry.

    Returns the :class:`DocumentDelta` plus the ``{doc_id: content_hash}``
    fingerprint map (so callers can persist new/changed records without
    recomputing hashes).
    """
    fingerprints = fingerprint_documents(documents)
    delta = doc_status.diff(fingerprints)
    logger.info(
        "Delta detected: %d new, %d changed, %d unchanged, %d deleted",
        len(delta.new),
        len(delta.changed),
        len(delta.unchanged),
        len(delta.deleted),
    )
    return delta, fingerprints


def filter_documents_to_process(
    documents: list[Document], delta: DocumentDelta
) -> list[Document]:
    """Return only the documents that need (re)indexing this run (new + changed)."""
    to_process = set(delta.to_process)
    return [
        document
        for document in documents
        if compute_doc_id(document.file_path) in to_process
    ]
