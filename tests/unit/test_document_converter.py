# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for convert_langchain_to_document (AWS-free)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document as LangChainDocument

from unified_kg_rag.domain.models import Document
from unified_kg_rag.shared.utils.document_converter import (
    convert_langchain_to_document,
)

pytestmark = pytest.mark.unit


def _lc(text: str) -> list[LangChainDocument]:
    return [LangChainDocument(page_content=text, metadata={})]


def test_leading_bom_stripped_from_content() -> None:
    doc = convert_langchain_to_document(_lc("﻿Hello world"), "a.txt")
    assert doc.page_content == "Hello world"
    assert not doc.page_content.startswith("﻿")


def test_bom_does_not_change_document_id() -> None:
    # The content-derived document_id must be identical with/without a BOM, so
    # incremental-indexing content-hash dedup is not broken by an encoding quirk.
    with_bom = convert_langchain_to_document(_lc("﻿Same content"), "a.txt")
    without_bom = convert_langchain_to_document(_lc("Same content"), "a.txt")
    assert with_bom.document_id == without_bom.document_id


def test_non_bom_content_unchanged() -> None:
    doc = convert_langchain_to_document(_lc("Plain text"), "a.txt")
    assert doc.page_content == "Plain text"


# "보증 기간" written with conjoining jamo (NFD), as macOS file systems and
# many PDF text layers produce it.
_NFD_KOREAN = "\u1107\u1169\u110c\u1173\u11bc \u1100\u1175\u1100\u1161\u11ab"


def test_decomposed_hangul_is_composed() -> None:
    doc = convert_langchain_to_document(_lc(_NFD_KOREAN), "a.txt")
    assert doc.page_content == "보증 기간"
    assert doc.page_content == "보증 기간"


def test_nfd_and_nfc_text_share_a_document_id() -> None:
    nfd = convert_langchain_to_document(_lc(_NFD_KOREAN), "a.txt")
    nfc = convert_langchain_to_document(_lc("보증 기간"), "a.txt")
    assert nfd.document_id == nfc.document_id


def test_compatibility_characters_are_kept() -> None:
    # NFC, not NFKC: full-width letters and circled designators stay as written.
    doc = convert_langchain_to_document(_lc("ＡＢＣ ㈜"), "a.txt")
    assert doc.page_content == "ＡＢＣ ㈜"


def test_text_is_stored_once() -> None:
    # A Document used to hold its text three times (page_content, every
    # page's text_content and content.text), tripling the parsing and
    # loading caches and the parsed-output exports.
    text = "Vendor ships to Buyer. " * 50
    doc = convert_langchain_to_document(_lc(text), "a.txt")
    assert doc.to_json_file().count("Vendor ships to Buyer.") == 50


# A parsed export as earlier versions wrote it: the text three times, page
# elements, page sizes, loader raw_data and detected_language.
def _old_shape_export(text: str, page_content: str) -> dict:
    return {
        "page_content": page_content,
        "document_id": "doc-old",
        "file_name": "a.txt",
        "file_path": "a.txt",
        "file_type": "txt",
        "total_pages": 1,
        "pages": [
            {
                "page_number": 1,
                "text_content": text,
                "elements": [],
                "raw_data": {},
                "height": 10,
                "width": 5,
            }
        ],
        "content": {"text": text, "html": None, "markdown": None},
        "metadata": {"source": "a.txt"},
        "error_info": None,
        "detected_language": "Unknown",
        "elements": [],
    }


def test_old_shape_cached_document_validates() -> None:
    # Stage caches restore documents with model_validate, not from_json_file.
    doc = Document.model_validate(_old_shape_export("same", "same"))
    assert doc.page_content == "same"
    assert set(doc.model_dump()) == {
        "page_content",
        "document_id",
        "file_name",
        "file_path",
        "metadata",
        "error_info",
    }


def test_new_export_round_trips(tmp_path) -> None:
    doc = convert_langchain_to_document(_lc("Vendor ships to Buyer"), "a.txt")
    path = tmp_path / "a.json"
    doc.to_json_file(path)
    assert Document.from_json_file(path) == doc
