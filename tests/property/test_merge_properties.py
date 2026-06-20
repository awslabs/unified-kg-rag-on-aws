# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Property-based tests for merge laws (M2 incremental indexing)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from aws_graphrag.domain.ingestion.merge import merge_entities
from aws_graphrag.domain.models import Entity

pytestmark = pytest.mark.property

_names = st.lists(
    st.text(alphabet="ABCDE", min_size=1, max_size=2), min_size=0, max_size=6
)


def _entities(names: list[str], prefix: str) -> list[Entity]:
    # One entity per distinct name (merge_entities collapses by name anyway).
    return [
        Entity(id=f"{prefix}{i}", name=name, text_unit_ids=[f"{prefix}t{i}"])
        for i, name in enumerate(dict.fromkeys(names))
    ]


@given(old_names=_names, delta_names=_names)
def test_merged_names_are_union(old_names: list[str], delta_names: list[str]) -> None:
    old = _entities(old_names, "o")
    delta = _entities(delta_names, "d")
    merged, _ = merge_entities(old, delta)
    expected = {n.upper() for n in (old_names + delta_names)}
    # normalize_name lowercases; names here are already single-case letters.
    assert {e.name.upper() for e in merged} == expected


@given(old_names=_names)
def test_merging_empty_delta_is_identity_on_names(old_names: list[str]) -> None:
    old = _entities(old_names, "o")
    merged, remap = merge_entities(old, [])
    assert {e.id for e in merged} == {e.id for e in old}
    assert remap == {}


@given(names=_names)
def test_self_merge_is_idempotent(names: list[str]) -> None:
    old = _entities(names, "o")
    # Merging the same logical set (same names) must not grow the entity count
    # and must remap every delta entity onto an existing one.
    delta = _entities(names, "d")
    merged, remap = merge_entities(old, delta)
    assert len(merged) == len(old)
    assert len(remap) == len(delta)
