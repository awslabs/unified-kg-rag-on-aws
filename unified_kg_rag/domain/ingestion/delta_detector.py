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

from pathlib import PurePosixPath

from unified_kg_rag.domain.models import Document, DocumentDelta
from unified_kg_rag.ports import DocStatusPort
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.common import compute_hash

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
    return compute_hash(text, algorithm="sha256", length=32)


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
