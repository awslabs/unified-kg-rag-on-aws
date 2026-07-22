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
