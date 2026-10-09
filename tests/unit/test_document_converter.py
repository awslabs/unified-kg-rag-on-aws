# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for convert_langchain_to_document (AWS-free)."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document as LangChainDocument

from unified_kg_rag.shared.utils.document_converter import (
    convert_langchain_to_document,
)

pytestmark = pytest.mark.unit


def _lc(text: str) -> list[LangChainDocument]:
    return [LangChainDocument(page_content=text, metadata={})]


def test_leading_bom_stripped_from_content() -> None:
    doc = convert_langchain_to_document(_lc("﻿Hello world"), "a.txt")
    assert doc.content.text == "Hello world"
    assert not doc.content.text.startswith("﻿")


def test_bom_does_not_change_document_id() -> None:
    # The content-derived document_id must be identical with/without a BOM, so
    # incremental-indexing content-hash dedup is not broken by an encoding quirk.
    with_bom = convert_langchain_to_document(_lc("﻿Same content"), "a.txt")
    without_bom = convert_langchain_to_document(_lc("Same content"), "a.txt")
    assert with_bom.document_id == without_bom.document_id


def test_non_bom_content_unchanged() -> None:
    doc = convert_langchain_to_document(_lc("Plain text"), "a.txt")
    assert doc.content.text == "Plain text"


# "보증 기간" written with conjoining jamo (NFD), as macOS file systems and
# many PDF text layers produce it.
_NFD_KOREAN = "\u1107\u1169\u110c\u1173\u11bc \u1100\u1175\u1100\u1161\u11ab"


def test_decomposed_hangul_is_composed() -> None:
    doc = convert_langchain_to_document(_lc(_NFD_KOREAN), "a.txt")
    assert doc.content.text == "보증 기간"
    assert doc.page_content == "보증 기간"
    assert doc.pages[0].text_content == "보증 기간"


def test_nfd_and_nfc_text_share_a_document_id() -> None:
    nfd = convert_langchain_to_document(_lc(_NFD_KOREAN), "a.txt")
    nfc = convert_langchain_to_document(_lc("보증 기간"), "a.txt")
    assert nfd.document_id == nfc.document_id


def test_compatibility_characters_are_kept() -> None:
    # NFC, not NFKC: full-width letters and circled designators stay as written.
    doc = convert_langchain_to_document(_lc("ＡＢＣ ㈜"), "a.txt")
    assert doc.content.text == "ＡＢＣ ㈜"
