# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``merge_attributes``: the attribute rule of graph resolution and the merge.

A full build resolves the N instances of an item at once; an incremental
build folds them into the stored item two at a time. Both call
``merge_attributes``, so the rule itself must make the two agree: a
multi-valued document filter is a sorted list (the incremental two-way merge
once kept a delta's list unsorted when the stored item had no filters, while
a full build sorted it), one value stays a scalar, and nothing is ``None``.
"""

from __future__ import annotations

from functools import reduce
from itertools import product
from typing import Any

import pytest

from unified_kg_rag.domain.ingestion.base_processor import merge_attributes

pytestmark = pytest.mark.unit

ATTRIBUTES: list[dict[str, Any] | None] = [
    None,
    {},
    {"filters": {}},
    {"filters": {"tenant": "b"}},
    {"filters": {"tenant": ["b", "a"]}},
    {"filters": {"tenant": ["a"]}, "suffix": "s1"},
    {"filters": {"tenant": "a", "region": "x"}},
    {"suffix": "s2"},
]


@pytest.mark.parametrize(
    ("maps", "expected"),
    [
        ([], None),
        ([None, {}], None),
        # An empty filters dict is kept as no filters.
        ([{"filters": {}}], {"filters": {}}),
        ([{"filters": {"tenant": "b"}}], {"filters": {"tenant": "b"}}),
        # One distinct value stays a scalar, a one-element list included.
        ([{"filters": {"tenant": ["a"]}}], {"filters": {"tenant": "a"}}),
        ([{"filters": {"tenant": "a"}}, {"filters": {"tenant": ["a"]}}],
         {"filters": {"tenant": "a"}}),
        # Several values: a sorted list, also from one unsorted list.
        ([{"filters": {"tenant": ["b", "a"]}}], {"filters": {"tenant": ["a", "b"]}}),
        ([{"filters": {"tenant": "b"}}, {"filters": {"tenant": "a"}}],
         {"filters": {"tenant": ["a", "b"]}}),
        # A None filter value is dropped, and a key left with none with it.
        ([{"filters": {"tenant": None, "region": "x"}}],
         {"filters": {"region": "x"}}),
        # Other keys: the later map wins; filters keys are unioned.
        ([{"suffix": "s1", "filters": {"tenant": "a"}},
          {"suffix": "s2", "filters": {"region": "x"}}],
         {"suffix": "s2", "filters": {"tenant": "a", "region": "x"}}),
    ],
)  # fmt: skip
def test_merge_attributes(maps: list, expected: dict | None) -> None:
    assert merge_attributes(maps) == expected


def test_merge_attributes_does_not_alias_its_inputs() -> None:
    tenants = ["b", "a"]
    merged = merge_attributes([{"filters": {"tenant": tenants}}])
    assert merged is not None
    merged["filters"]["tenant"].append("c")
    assert tenants == ["b", "a"]


@pytest.mark.parametrize(
    "maps", [*product(ATTRIBUTES, repeat=2), *product(ATTRIBUTES[3:], repeat=3)]
)
def test_folding_two_at_a_time_equals_merging_at_once(maps) -> None:
    # The incremental merge folds runs into the stored item; a full build
    # merges every instance in one call.
    folded = reduce(lambda stored, new: merge_attributes((stored, new)), maps)
    assert folded == merge_attributes(maps)


def test_unhashable_filter_values_are_unioned_once() -> None:
    # A filter value the set cannot hold (a dict) falls back to the list;
    # a repeat of it must still be dropped, not listed twice.
    from unified_kg_rag.domain.ingestion.base_processor import (
        union_all_document_filters,
    )

    region = {"code": "eu"}
    merged = union_all_document_filters(
        [
            {"filters": {"region": region}},
            {"filters": {"region": dict(region)}},
            {"filters": {"region": {"code": "us"}}},
        ]
    )
    assert merged is not None
    assert merged["filters"]["region"] == [{"code": "eu"}, {"code": "us"}]
