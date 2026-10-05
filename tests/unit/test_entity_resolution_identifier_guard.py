# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fuzzy entity resolution must not merge distinct identifiers or types.

Regression: MinHash over 3-char shingles scored "purchase order 1001" vs
"purchase order 1002" at ~0.96 and "vendor a" vs "vendor b" at ~0.73 (above the
0.6 default), and the BFS grouping ignored entity types and chained
transitively. Distinct identifiers collapsed into one entity with merged
descriptions and re-pointed edges. All names below are synthetic.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from unified_kg_rag.domain.ingestion.base_resolver import (
    FuzzyMatcher,
    discriminator_tokens,
    entity_types_compatible,
)
from unified_kg_rag.domain.ingestion.graph_resolver import (
    EntityResolver,
    GraphResolver,
    _TypeAwareUnionFind,
)
from unified_kg_rag.domain.ingestion.merge import merge_entities
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    Relationship,
    ResolutionMethod,
)

pytestmark = pytest.mark.unit


def _resolver(config: Config | None = None) -> EntityResolver:
    resolver = EntityResolver(config or Config(), max_workers=1, use_process_pool=False)
    resolver.show_progress = False
    return resolver


def _group_names(entities: list[Entity], config: Config | None = None) -> list[set]:
    groups = _resolver(config)._group_similar_entities(entities)
    return [{e.name for e in group} for group in groups]


def _together(groups: list[set], a: str, b: str) -> bool:
    return any(a in g and b in g for g in groups)


# --------------------------------------------------------------------------- #
# Discriminator tokens / type compatibility
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Purchase Order 1001", {"1001"}),
        ("Vendor A", {"a"}),
        ("Phase II", {"ii"}),
        ("Phase 2", {"2"}),
        ("Section 4.2", {"4", "2"}),
        ("Model v2", {"v2"}),
        ("Acme Corporation", set()),
        ("Mix Studio", set()),  # "mix" is not a Roman numeral
        ("가 나다", set()),  # single non-ASCII letters are not identifiers
    ],
)
def test_discriminator_tokens(name: str, expected: set[str]) -> None:
    assert discriminator_tokens(name) == frozenset(expected)


@pytest.mark.parametrize(
    ("a", "b", "compatible"),
    [
        ("organization", "ORGANIZATION", True),
        ("organization", "", True),
        ("Unknown", "person", True),
        (None, "person", True),
        ("organization", "person", False),
    ],
)
def test_entity_types_compatible(a, b, compatible: bool) -> None:
    assert entity_types_compatible(a, b) is compatible


@pytest.mark.parametrize(
    "method", [ResolutionMethod.MINHASH, ResolutionMethod.SEQUENCE_MATCHER]
)
def test_find_all_matches_drops_identifier_conflicts(method) -> None:
    names = ["Purchase Order 1001", "Purchase Order 1002", "Purchase Order 1003"]
    matcher = FuzzyMatcher(names, resolution_method=method, similarity_threshold=0.6)
    matched = {name for name, _ in matcher.find_all_matches("Purchase Order 1001")}
    assert "Purchase Order 1002" not in matched
    assert "Purchase Order 1003" not in matched


# --------------------------------------------------------------------------- #
# Full-build grouping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "names",
    [
        ["Purchase Order 1001", "Purchase Order 1002", "Purchase Order 1003"],
        ["Article 12", "Article 13 Termination", "Article 12 Termination"],
        ["Vendor A", "Vendor B"],
        ["Phase 1", "Phase 2", "Phase II"],
    ],
)
def test_distinct_identifiers_stay_separate(names: list[str]) -> None:
    entities = [
        Entity(id=f"e{i}", name=name, type="concept") for i, name in enumerate(names)
    ]
    groups = _group_names(entities)
    assert len(groups) == len(names)


def test_true_variants_still_merge() -> None:
    entities = [
        Entity(id="e1", name="Acme Corp", type="organization"),
        Entity(id="e2", name="Acme Corp.", type="organization"),
        Entity(id="e3", name="ACME Corporation", type="organization"),
        Entity(id="e4", name="Acme Corporation Inc", type=""),
        Entity(id="e5", name="Purchase Order 1001", type="document"),
        Entity(id="e6", name="purchase order 1001", type="document"),
    ]
    groups = _group_names(entities)
    assert _together(groups, "Acme Corp", "Acme Corp.")
    assert _together(groups, "ACME Corporation", "Acme Corporation Inc")
    assert _together(groups, "Purchase Order 1001", "purchase order 1001")


def test_type_mismatch_stays_separate() -> None:
    entities = [
        Entity(id="e1", name="Acme Corp", type="organization"),
        Entity(id="e2", name="Acme Corp.", type="product"),
    ]
    groups = _group_names(entities)
    assert not _together(groups, "Acme Corp", "Acme Corp.")


def test_transitive_chain_cannot_join_conflicting_types() -> None:
    # All three names match each other (score 1.0); the untyped middle member
    # would otherwise bridge the two incompatible types into one group.
    entities = [
        Entity(id="e1", name="Acme Corp", type="organization"),
        Entity(id="e2", name="acme corp", type=""),
        Entity(id="e3", name="ACME CORP", type="person"),
    ]
    groups = _group_names(entities)
    assert not _together(groups, "Acme Corp", "ACME CORP")
    # The untyped member still joins exactly one side.
    assert sum("acme corp" in g for g in groups) == 1
    assert len(groups) == 2


def test_same_name_entities_still_grouped_together() -> None:
    # Exact-name identity is not a fuzzy link: the type guard does not split it.
    entities = [
        Entity(id="e1", name="Mercury", type="planet"),
        Entity(id="e2", name="Mercury", type="element"),
    ]
    groups = _resolver()._group_similar_entities(entities)
    assert [sorted(e.id for e in g) for g in groups] == [["e1", "e2"]]


def test_grouping_is_order_independent() -> None:
    entities = [
        Entity(id="e1", name="Acme Corp", type="organization"),
        Entity(id="e2", name="acme corp", type=""),
        Entity(id="e3", name="ACME CORP", type="person"),
        Entity(id="e4", name="Vendor A", type="organization"),
        Entity(id="e5", name="Vendor B", type="organization"),
    ]

    def canon(groups: list[set]) -> set[frozenset]:
        return {frozenset(g) for g in groups}

    assert canon(_group_names(entities)) == canon(_group_names(entities[::-1]))


def test_relationships_keep_distinct_identifier_endpoints() -> None:
    resolver = GraphResolver(Config(), max_workers=1, use_process_pool=False)
    entities = [
        Entity(id="buyer", name="Buyer", type="organization"),
        Entity(id="po1", name="Purchase Order 1001", type="document"),
        Entity(id="po2", name="Purchase Order 1002", type="document"),
    ]
    relationships = [
        Relationship(id="r1", source_id="buyer", target_id="po1", type="ISSUED"),
        Relationship(id="r2", source_id="buyer", target_id="po2", type="ISSUED"),
    ]
    result, _ = resolver.resolve_graph(entities, relationships)
    assert {e.id for e in result["entities"]} == {"buyer", "po1", "po2"}
    assert {(r.source_id, r.target_id) for r in result["relationships"]} == {
        ("buyer", "po1"),
        ("buyer", "po2"),
    }


# --------------------------------------------------------------------------- #
# Incremental merge applies the same guards
# --------------------------------------------------------------------------- #


def test_incremental_merge_does_not_collapse_distinct_identifier() -> None:
    old = [Entity(id="po1", name="Purchase Order 1001", type="document")]
    delta = [Entity(id="po2", name="Purchase Order 1002", type="document")]
    matcher = FuzzyMatcher(["Purchase Order 1001"], similarity_threshold=0.6)
    merged, remap = merge_entities(old, delta, fuzzy_matcher=matcher)
    assert remap == {}
    assert {e.id for e in merged} == {"po1", "po2"}


def test_incremental_merge_respects_type() -> None:
    old = [Entity(id="o1", name="Acme Corporation", type="organization")]
    delta = [Entity(id="d1", name="Acme Corporatio", type="product")]
    matcher = FuzzyMatcher(["Acme Corporation"], similarity_threshold=0.5)
    merged, remap = merge_entities(old, delta, fuzzy_matcher=matcher)
    assert remap == {}
    assert len(merged) == 2


def test_incremental_merge_still_merges_untyped_variant() -> None:
    old = [Entity(id="o1", name="Acme Corporation", type="organization")]
    delta = [Entity(id="d1", name="Acme Corporatio", type="")]
    matcher = FuzzyMatcher(["Acme Corporation"], similarity_threshold=0.5)
    _, remap = merge_entities(old, delta, fuzzy_matcher=matcher)
    assert remap == {"d1": "o1"}


# --------------------------------------------------------------------------- #
# Invariant: no finished group contains a pairwise-conflicting pair
# --------------------------------------------------------------------------- #

_TYPES = st.sampled_from(["", "organization", "person", "product"])
_NAMES = st.sampled_from(
    [
        "Acme Corp",
        "acme corp",
        "Acme Corp.",
        "ACME Corporation",
        "Acme Corporation Inc",
        "Vendor A",
        "Vendor B",
        "Phase 1",
        "Phase 2",
        "Purchase Order 1001",
        "Purchase Order 1002",
    ]
)


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(st.lists(st.tuples(_NAMES, _TYPES), min_size=2, max_size=8, unique=True))
def test_groups_never_contain_conflicting_members(items) -> None:
    entities = [
        Entity(id=f"e{i}", name=name, type=type_)
        for i, (name, type_) in enumerate(items)
    ]
    for group in _resolver()._group_similar_entities(entities):
        names = {e.name for e in group}
        assert len({discriminator_tokens(n) for n in names}) == 1
        for a in group:
            for b in group:
                if a.name != b.name:
                    assert entity_types_compatible(a.type, b.type)


def test_union_find_tracks_common_type() -> None:
    uf = _TypeAwareUnionFind(
        {
            "a": frozenset({"organization"}),
            "b": frozenset({"organization"}),
            "c": frozenset({"person"}),
            "d": None,  # untyped
            "e": frozenset(),  # one name extracted with conflicting types
        }
    )
    assert uf.union("a", "b")
    assert not uf.union("a", "c")
    assert uf.union("d", "a")
    assert uf.find("d") == uf.find("b")
    assert uf.find("c") != uf.find("a")
    assert not uf.union("e", "a")  # a conflicted name joins no typed group
    assert not uf.union("e", "c")
