# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The query entity list splits in every language (``split_entity_list``)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from unified_kg_rag.domain.retrieval.query_entities import split_entity_list

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Vendor, Buyer, Depot", ["Vendor", "Buyer", "Depot"]),
        # Japanese: the ideographic comma.
        ("供給者、購入者、倉庫", ["供給者", "購入者", "倉庫"]),
        # Chinese: the fullwidth comma and semicolon.
        ("供应商，采购方；仓库", ["供应商", "采购方", "仓库"]),
        # Korean: ASCII commas between Hangul words with spaces.
        ("공급사, 구매사, 물류 창고", ["공급사", "구매사", "물류 창고"]),
        # One per line, with list markers.
        ("- 공급사\n- 구매사\n1. 창고", ["공급사", "구매사", "창고"]),
        # The whole list quoted, as the prompt's example shows it.
        ('"Vendor, Buyer"', ["Vendor", "Buyer"]),
        ("「供給者」、「倉庫」。", ["供給者", "倉庫"]),
        # Arabic commas and semicolons.
        ("المورد، المشتري؛ المستودع", ["المورد", "المشتري", "المستودع"]),
        # A quoted item keeps its comma; a number keeps its separator.
        ('"Vendor, Inc.", Buyer', ["Vendor, Inc.", "Buyer"]),
        ("1,000 units, Depot", ["1,000 units", "Depot"]),
        # Repeats and empty items dropped.
        ("Depot,, Depot ,", ["Depot"]),
        ("", []),
    ],
)
def test_the_entity_list_splits_at_every_separator(
    answer: str, expected: list[str]
) -> None:
    assert split_entity_list(answer) == expected


_NAME = st.text(
    alphabet=st.sampled_from("ab가나다あいう漢字XYZ"), min_size=1, max_size=6
)


@given(
    names=st.lists(_NAME, min_size=1, max_size=6, unique=True),
    separator=st.sampled_from([",", ", ", "，", "、", ";", "；", "\n", " 、 "]),
)
def test_names_joined_by_any_separator_split_back(
    names: list[str], separator: str
) -> None:
    assert split_entity_list(separator.join(names)) == names


@given(st.text(max_size=80))
def test_items_are_stripped_non_empty_and_unique(text: str) -> None:
    items = split_entity_list(text)
    assert all(item and item == item.strip() for item in items)
    assert len(items) == len(set(items))
