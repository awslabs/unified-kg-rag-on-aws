# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property-based tests for merge laws (M2 incremental indexing).

The merge functions key entities/relationships by ``normalize_name`` (NFKC +
casefold + separator/punctuation handling). Earlier versions of these tests drew
names from ``alphabet="ABCDE"``, which never exercised that normalization — the
union law was trivially true on single-case ASCII. These strategies deliberately
include the hard inputs (mixed case, ``_``/``-`` separators, surrounding
whitespace, full-width forms, unicode, punctuation-only) so the merge key's
collision behavior is actually tested. ``normalize_name`` is used as the oracle
(we assert the merge invariant, not a re-derivation of normalization).
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from aws_graphrag.domain.ingestion.merge import merge_entities, merge_relationships
from aws_graphrag.domain.models import Entity, Relationship
from aws_graphrag.shared.utils.common import normalize_name

pytestmark = pytest.mark.property

# Names that actually stress normalize_name: case, separators, whitespace,
# full-width digits/letters, accented/CJK scripts, and punctuation-only strings
# (which fall back to the casefolded original rather than collapsing to "").
_hard_names = st.lists(
    st.sampled_from(
        [
            "Acme",
            "acme",
            "ACME",
            "Acme-Corp",
            "Acme Corp",
            "acme_corp",
            "  Acme  ",
            "ＡＣＭＥ",  # full-width -> "acme" under NFKC
            "Café",
            "café",
            "東京",
            "!!!",
            "@@@",
            "résumé",
            "Data-Science",
            "data science",
        ]
    ),
    min_size=0,
    max_size=8,
)


def _entities_unique_ids(names: list[str], prefix: str) -> list[Entity]:
    # One entity per list position (NOT deduped by raw name) so that raw names
    # which normalize to the same key are present as separate inputs — that is
    # exactly the collision the merge key must collapse.
    return [
        Entity(id=f"{prefix}{i}", name=name, text_unit_ids=[f"{prefix}t{i}"])
        for i, name in enumerate(names)
    ]


def _distinct_keys(names: list[str]) -> set[str]:
    return {normalize_name(n) for n in names}


@given(old_names=_hard_names, delta_names=_hard_names)
def test_merged_count_equals_distinct_normalized_keys(
    old_names: list[str], delta_names: list[str]
) -> None:
    # The merged set has exactly one entity per distinct normalize_name value
    # across old+delta. This is the property that protects against both
    # duplicate entities (under-merging) and unrelated entities collapsing
    # (over-merging) in production incremental indexing.
    old = _entities_unique_ids(old_names, "o")
    delta = _entities_unique_ids(delta_names, "d")
    merged, _ = merge_entities(old, delta)
    assert {normalize_name(e.name) for e in merged} == _distinct_keys(
        old_names + delta_names
    )
    assert len(merged) == len(_distinct_keys(old_names + delta_names))


@given(old_names=_hard_names)
def test_merging_empty_delta_is_identity(old_names: list[str]) -> None:
    old = _entities_unique_ids(old_names, "o")
    merged, remap = merge_entities(old, [])
    # Old entities collapse to their distinct normalized keys; empty delta adds
    # nothing and remaps nothing.
    assert len(merged) == len(_distinct_keys(old_names))
    assert remap == {}


@given(names=_hard_names)
def test_self_merge_is_idempotent(names: list[str]) -> None:
    old = _entities_unique_ids(names, "o")
    delta = _entities_unique_ids(names, "d")
    merged, remap = merge_entities(old, delta)
    # Re-applying the same logical set must not grow the entity count beyond the
    # distinct keys, and every delta entity must remap onto a surviving one.
    assert len(merged) == len(_distinct_keys(names))
    assert len(remap) == len(delta)


# --------------------------------------------------------------------------- #
# merge_relationships — previously UNFUZZED. Keyed by (source_id, target_id,
# type.strip().lower()); a different type between the same endpoints stays a
# distinct edge. weights sum; the resulting key set is order-independent.
# --------------------------------------------------------------------------- #
_entity_ids = st.sampled_from(["e1", "e2", "e3"])
# Types that stress the key's case/whitespace handling.
_rel_types = st.sampled_from(["KNOWS", "knows", " knows ", "WORKS_AT", ""])


@st.composite
def _relationships(draw, prefix: str) -> list[Relationship]:
    n = draw(st.integers(min_value=0, max_value=6))
    rels = []
    for i in range(n):
        rels.append(
            Relationship(
                id=f"{prefix}{i}",
                source_id=draw(_entity_ids),
                target_id=draw(_entity_ids),
                type=draw(_rel_types),
                weight=draw(st.floats(min_value=0.0, max_value=5.0)),
                description=f"{prefix}-desc-{i}",
            )
        )
    return rels


def _rel_key(r: Relationship) -> tuple[str, str, str]:
    # Mirror merge_relationships._key exactly (no entity_id_remap in these tests).
    return (r.source_id, r.target_id, (r.type or "").strip().lower())


@given(old=_relationships("o"), delta=_relationships("d"))
def test_relationship_merge_collapses_to_distinct_keys(
    old: list[Relationship], delta: list[Relationship]
) -> None:
    merged = merge_relationships(old, delta)
    expected_keys = {_rel_key(r) for r in (old + delta)}
    assert {_rel_key(r) for r in merged} == expected_keys
    assert len(merged) == len(expected_keys)


@given(old=_relationships("o"), delta=_relationships("d"))
def test_relationship_merge_is_order_independent(
    old: list[Relationship], delta: list[Relationship]
) -> None:
    # The resulting key set must not depend on delta ordering.
    merged_fwd = merge_relationships(old, delta)
    merged_rev = merge_relationships(old, list(reversed(delta)))
    assert {_rel_key(r) for r in merged_fwd} == {_rel_key(r) for r in merged_rev}
