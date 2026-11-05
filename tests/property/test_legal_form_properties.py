# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariants of ``strip_legal_forms``, the legal-form part of the compact
name key: a stripped name strips to itself, and a name with any content
never strips to nothing (an empty key would merge unrelated entities)."""

from __future__ import annotations

import unicodedata

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from unified_kg_rag.domain.ingestion.base_resolver import compact_name_key
from unified_kg_rag.shared.utils.legal_forms import strip_legal_forms

pytestmark = pytest.mark.property

# Names built from designators, articles, conjunctions and plain words, so
# the generated names exercise every rule rather than only random text.
_PARTS = st.sampled_from(
    [
        "Acme",
        "the",
        "&",
        "and",
        "Inc",
        "Inc.",
        "Co.,",
        "Ltd",
        "L.L.C.",
        "S.p.A.",
        "spa",
        "GmbH",
        "Pty",
        "(주)",
        "㈜",
        "주식회사",
        "가나다",
        "株式会社",
        "甲乙",
        "股份有限公司",
        "有限公司",
    ]
)
_NAMES = st.one_of(
    st.text(),
    st.lists(_PARTS, min_size=1, max_size=6).map(" ".join),
    st.lists(_PARTS, min_size=1, max_size=6).map("".join),
)


def _has_content(name: str) -> bool:
    folded = unicodedata.normalize(
        "NFKC", unicodedata.normalize("NFKC", name).casefold()
    )
    return bool(folded.strip())


@given(name=_NAMES)
@example(name="ß́")
@example(name="(주)주식회사")
@example(name="Acme Pty Ltd")
def test_stripping_is_idempotent(name: str) -> None:
    once = strip_legal_forms(name)
    assert strip_legal_forms(once) == once


@given(name=_NAMES)
@example(name="Inc.")
@example(name="주식회사 (주)")
@example(name="the co")
def test_stripping_never_empties_a_name(name: str) -> None:
    if _has_content(name):
        assert strip_legal_forms(name)
        assert compact_name_key(name)
