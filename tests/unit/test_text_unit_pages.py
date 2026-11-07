# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Page numbers of text units (docs/design.md §7.1): the converter records
where each page of a paged document starts, the chunker stores each chunk's
first and last page beside its relative path, and nothing else changes."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document as LangChainDocument

from unified_kg_rag.adapters.ingestion.chunker import SimpleTextChunker
from unified_kg_rag.adapters.ingestion.document_converter import (
    convert_langchain_to_document,
)
from unified_kg_rag.domain.ingestion.delta_detector import assign_document_identity
from unified_kg_rag.domain.ingestion.pages import (
    PAGE_OFFSETS_KEY,
    chunk_page_ranges,
    valid_offsets,
)
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit

_PAGES = [
    "Vendor supplies widgets to Buyer under the agreement.",
    "",
    "공급사는 구매사에 부품을 공급한다. 창고는 공급사의 부품을 보관한다.",
    "倉庫は供給者の部品を保管する。",
]


def _pdf(pages: list[str], *, bom: bool = False) -> list[LangChainDocument]:
    docs = [
        LangChainDocument(page_content=text, metadata={"page": i, "source": "x.pdf"})
        for i, text in enumerate(pages)
    ]
    if bom:
        docs[0].page_content = "\ufeff" + docs[0].page_content
    return docs


@pytest.mark.parametrize("bom", [False, True])
def test_the_converter_records_where_each_page_starts(bom: bool) -> None:
    document = convert_langchain_to_document(_pdf(_PAGES, bom=bom), "x.pdf")
    offsets = document.metadata[PAGE_OFFSETS_KEY]
    # The empty page has no text and no entry; pages are 1-based.
    assert [page for _, page in offsets] == [1, 3, 4]
    for (offset, page), text in zip(
        offsets, [_PAGES[0], _PAGES[2], _PAGES[3]], strict=True
    ):
        assert document.page_content[offset : offset + len(text)] == text, page
    # The id and text are those of an unpaged conversion of the same text.
    plain = convert_langchain_to_document(
        [LangChainDocument(page_content=document.page_content)], "x.pdf"
    )
    assert plain.document_id == document.document_id
    assert PAGE_OFFSETS_KEY not in plain.metadata


def test_a_document_without_page_numbers_records_none() -> None:
    docs = [LangChainDocument(page_content="a", metadata={"seq_num": 1})]
    assert (
        PAGE_OFFSETS_KEY not in convert_langchain_to_document(docs, "a.json").metadata
    )


def test_chunks_map_to_the_pages_they_span() -> None:
    content = "aaaa\n\nbbbb\n\ncccc"
    offsets = [[0, 1], [6, 2], [12, 3]]
    assert chunk_page_ranges(
        content, offsets, ["aaaa", "aa\n\nbbbb", "bbbb\n\ncc", "zzz", "cccc"]
    ) == [(1, 1), (1, 2), (2, 3), None, (3, 3)]
    # Repeated text is matched after the previous chunk, never before it.
    assert chunk_page_ranges("xx\n\nxx", [[0, 1], [4, 2]], ["xx", "xx"]) == [
        (1, 1),
        (2, 2),
    ]


@pytest.mark.parametrize(
    "value", [None, [], [[0]], [[5, 1], [2, 2]], [[0, 0]], [[0, True]], "x"]
)
def test_malformed_offsets_are_ignored(value: object) -> None:
    assert valid_offsets(value) is None


def test_text_units_carry_their_pages_and_relative_path(tmp_path) -> None:
    config = Config()
    chunking = config.processing.chunking
    chunking.fallback_chunk_size = 40
    chunking.chunk_overlap = 0
    chunking.min_chunk_size = 1
    source = tmp_path / "corpus"
    path = source / "legal" / "x.pdf"
    document = convert_langchain_to_document(_pdf(_PAGES), path)
    assign_document_identity(document, source)
    units = SimpleTextChunker(config, show_progress=False).chunk_documents([document])
    assert len(units) >= 3
    pages = [(u.attributes["page_start"], u.attributes["page_end"]) for u in units]
    assert pages[0] == (1, 1)
    assert pages[-1] == (4, 4)
    assert all(first <= last for first, last in pages)
    assert {p for pair in pages for p in pair} == {1, 3, 4}
    assert {u.attributes["relative_path"] for u in units} == {"legal/x.pdf"}

    # Ids and texts do not depend on the pages.
    unpaged = document.model_copy(deep=True)
    del unpaged.metadata[PAGE_OFFSETS_KEY]
    plain = SimpleTextChunker(config, show_progress=False).chunk_documents([unpaged])
    assert [(u.id, u.text) for u in plain] == [(u.id, u.text) for u in units]
    assert all("page_start" not in u.attributes for u in plain)
