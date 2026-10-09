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

import functools

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from unified_kg_rag.domain.ingestion.graph_resolver import RelationshipResolver
from unified_kg_rag.domain.ingestion.merge import (
    merge_entities,
    merge_relationships,
    remove_text_units,
)
from unified_kg_rag.domain.models import Config, Entity, Relationship
from unified_kg_rag.shared.utils.common import normalize_name

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


def _stored_entities(names: list[str]) -> list[Entity]:
    # Stored state after a merge holds one entity per key; the merge keeps every
    # stored id as it is, so two stored entities sharing a key stay two.
    by_key = {normalize_name(n): n for n in names}
    return _entities_unique_ids(list(by_key.values()), "o")


@given(old_names=_hard_names, delta_names=_hard_names)
def test_merged_count_equals_distinct_normalized_keys(
    old_names: list[str], delta_names: list[str]
) -> None:
    # The merged set has exactly one entity per distinct normalize_name value
    # across old+delta. This is the property that protects against both
    # duplicate entities (under-merging) and unrelated entities collapsing
    # (over-merging) in production incremental indexing.
    old = _stored_entities(old_names)
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
    # Every stored entity is kept as it is (including two sharing a key); an
    # empty delta adds nothing and remaps nothing.
    assert merged == old
    assert remap == {}


@given(names=_hard_names)
def test_self_merge_is_idempotent(names: list[str]) -> None:
    old = _stored_entities(names)
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
    return (r.source_id, r.target_id, (r.type or "RELATED_TO").strip().lower())


def _stored(rels: list[Relationship]) -> list[Relationship]:
    # Stored state after a merge holds one edge per key (see _stored_entities).
    return list({_rel_key(r): r for r in rels}.values())


@given(old=_relationships("o"), delta=_relationships("d"))
def test_relationship_merge_collapses_to_distinct_keys(
    old: list[Relationship], delta: list[Relationship]
) -> None:
    old = _stored(old)
    merged = merge_relationships(old, delta)
    # Self-loops are dropped (parity with the full-build resolver), so they are
    # excluded from the expected key set.
    expected_keys = {_rel_key(r) for r in (old + delta) if r.source_id != r.target_id}
    assert {_rel_key(r) for r in merged} == expected_keys
    assert len(merged) == len(expected_keys)
    # No merged edge is a self-loop.
    assert all(r.source_id != r.target_id for r in merged)


@given(old=_relationships("o"), delta=_relationships("d"))
def test_relationship_merge_is_idempotent_in_weight(
    old: list[Relationship], delta: list[Relationship]
) -> None:
    # Re-applying the same delta must not change any edge's weight (the prior
    # summed-weight semantics inflated it on every re-application).
    once = merge_relationships(old, delta)
    twice = merge_relationships(once, delta)
    weights_once = {_rel_key(r): r.weight for r in once}
    weights_twice = {_rel_key(r): r.weight for r in twice}
    assert weights_once == weights_twice


@given(old=_relationships("o"), delta=_relationships("d"))
def test_relationship_merge_is_order_independent(
    old: list[Relationship], delta: list[Relationship]
) -> None:
    # The resulting key set must not depend on delta ordering.
    merged_fwd = merge_relationships(old, delta)
    merged_rev = merge_relationships(old, list(reversed(delta)))
    assert {_rel_key(r) for r in merged_fwd} == {_rel_key(r) for r in merged_rev}


# --------------------------------------------------------------------------- #
# Full build vs incremental merge: relationship weights converge.
# A "full build" is the resolver over every extracted instance of the corpus;
# an incremental run is the full build of the old documents followed by
# merge_relationships of the new documents' full build. Weight is the sum of
# the per-text-unit strengths, so both must agree on edges and weights.
# --------------------------------------------------------------------------- #
@st.composite
def _instances(draw) -> list[Relationship]:
    """Extracted edge instances, each from one of text units t0..t5."""
    n = draw(st.integers(min_value=0, max_value=12))
    instances = []
    for _ in range(n):
        source = draw(_entity_ids)
        target = draw(_entity_ids)
        rel_type = draw(st.sampled_from(["KNOWS", "knows", "WORKS_AT"]))
        instances.append(
            Relationship(
                # Ids derive from the endpoints and normalized type, as in
                # extraction, so the same edge has the same id in every run.
                id=f"r-{source}-{target}-{rel_type.lower()}",
                source_id=source,
                target_id=target,
                type=rel_type,
                weight=draw(st.floats(min_value=0.1, max_value=10.0)),
                text_unit_ids=[f"t{draw(st.integers(min_value=0, max_value=5))}"],
            )
        )
    return instances


@functools.cache
def _resolver() -> RelationshipResolver:
    return RelationshipResolver(Config(), max_workers=1, use_process_pool=False)


def _full_build(instances: list[Relationship]) -> list[Relationship]:
    copies = [r.model_copy(deep=True) for r in instances]
    resolved, _ = _resolver().resolve(copies, {})
    return resolved


def _edge_state(rels: list[Relationship]) -> dict[tuple[str, str, str], tuple]:
    return {_rel_key(r): (set(r.text_unit_ids or []), r.weight) for r in rels}


def _assert_same_graph(left: list[Relationship], right: list[Relationship]) -> None:
    left_state, right_state = _edge_state(left), _edge_state(right)
    assert left_state.keys() == right_state.keys()
    for key, (units, weight) in left_state.items():
        assert units == right_state[key][0]
        assert weight == pytest.approx(right_state[key][1])


@settings(deadline=None)
@given(
    instances=_instances(),
    new_units=st.sets(st.sampled_from([f"t{i}" for i in range(6)])),
)
def test_incremental_merge_converges_with_full_build(
    instances: list[Relationship], new_units: set[str]
) -> None:
    # Text units belong to one document, so old and new runs partition them.
    old = [r for r in instances if r.text_unit_ids[0] not in new_units]
    new = [r for r in instances if r.text_unit_ids[0] in new_units]

    full = _full_build(instances)
    incremental = merge_relationships(_full_build(old), _full_build(new))

    _assert_same_graph(full, incremental)
    # A retried commit re-applies the same delta without changing anything.
    _assert_same_graph(full, merge_relationships(incremental, _full_build(new)))


@settings(deadline=None)
@given(
    instances=_instances(),
    removed=st.sets(st.sampled_from([f"t{i}" for i in range(6)])),
)
def test_removing_text_units_converges_with_full_build(
    instances: list[Relationship], removed: set[str]
) -> None:
    kept = [r for r in instances if r.text_unit_ids[0] not in removed]
    full = _full_build(instances)
    _, updated = remove_text_units([], full, removed)
    by_id = {r.id: r for r in full} | {r.id: r for r in updated}
    # Edges left without text units are deleted by the caller.
    remaining = [r for r in by_id.values() if r.text_unit_ids]

    _assert_same_graph(_full_build(kept), remaining)
