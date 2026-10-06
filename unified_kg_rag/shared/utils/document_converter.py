# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from langchain_core.documents import Document as LangChainDocument

from unified_kg_rag.domain.models import Constants, Document, DocumentContent, Page
from unified_kg_rag.shared.utils.document_identity import (
    compute_document_id,
    compute_text_hash,
    normalize_source_path,
)


def convert_langchain_to_document(
    langchain_docs: list[LangChainDocument],
    file_path: str | Path,
    n_chars: int = 100,  # unused: the id now hashes the full text (kept for API)
    index_value: str | None = None,
) -> Document:
    path = Path(file_path)

    combined_text = "\n\n".join(
        doc.page_content for doc in langchain_docs if doc.page_content
    )
    # Strip a leading UTF-8 BOM (U+FEFF): loaders that decode as plain "utf-8"
    # (not "utf-8-sig") leave the BOM in the text, so the same logical document
    # with/without a BOM would hash to a different content-derived document_id
    # and break incremental-indexing dedup. Normalize once, here, since this is
    # the single source of both the id and the stored content.
    combined_text = combined_text.lstrip("\ufeff")

    # Copy rather than alias the loader's metadata dict: page 0's raw_data below
    # holds the same object as langchain_docs[0].metadata, so mutating the
    # top-level metadata (e.g. the INDEX key) would otherwise leak into page 0.
    metadata = dict(langchain_docs[0].metadata) if langchain_docs else {}
    if index_value is not None:
        metadata[Constants.INDEX.value] = index_value
    # Path + full-content hash: a name + text-prefix id collided for two files
    # with the same name in different folders and the same opening lines, which
    # in turn collided their text-unit ids. The pipeline re-derives this against
    # the corpus root (delta_detector.assign_document_identity).
    document_id = compute_document_id(
        normalize_source_path(path), compute_text_hash(combined_text)
    )

    pages = [
        Page(
            page_number=i + 1,
            text_content=doc.page_content,
            elements=[],
            raw_data=doc.metadata,
        )
        for i, doc in enumerate(langchain_docs)
    ]

    content = DocumentContent(text=combined_text, html=None, markdown=None)

    file_type = path.suffix.lower().lstrip(".")

    return Document(
        document_id=document_id,
        file_name=path.name,
        file_path=str(path),
        file_type=file_type,
        detected_language="Unknown",
        total_pages=len(langchain_docs),
        pages=pages,
        elements=[],
        content=content,
        metadata=metadata,
        page_content=combined_text,
    )
