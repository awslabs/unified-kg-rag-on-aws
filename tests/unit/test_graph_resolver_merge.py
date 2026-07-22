# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the full-build graph resolver merge/resolution logic.

graph_resolver.py is pure domain (no AWS) and carries the entity/relationship
merge arithmetic that feeds ranking and the incremental path. These lock the
merge semantics — max-confidence, frequency = |text_unit_ids|, evidence-count
weight, self-loop removal, type normalization, determinism — and the stats
reduction-rate guards. Run with use_process_pool=False so grouping is
deterministic and in-process.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.graph_resolver import (
    EntityResolutionStats,
    GraphResolutionStats,
    GraphResolver,
    RelationshipResolutionStats,
)
from unified_kg_rag.domain.models import Config, Entity, Relationship

pytestmark = pytest.mark.unit


@pytest.fixture
def resolver() -> GraphResolver:
    # In-process, single worker -> deterministic grouping for assertions.
    return GraphResolver(Config(), max_workers=1, use_process_pool=False)


def _ent(id_: str, name: str, **kw) -> Entity:
    return Entity(id=id_, name=name, **kw)


# --------------------------------------------------------------------------- #
# Entity merge
# --------------------------------------------------------------------------- #


def test_merge_takes_max_confidence(resolver: GraphResolver) -> None:
    entities = [
        _ent("e1", "Acme", confidence=0.3, text_unit_ids=["t1"]),
        _ent("e2", "Acme", confidence=0.9, text_unit_ids=["t2"]),
        _ent("e3", "Acme", confidence=None, text_unit_ids=["t3"]),
    ]
    merged = resolver.entity_resolver._merge_entities(entities)
    # Max (not mean): confidence is monotonic in evidence.
    assert merged.confidence == 0.9


def test_merge_recomputes_frequency_from_text_unit_union(
    resolver: GraphResolver,
) -> None:
    entities = [
        _ent("e1", "Acme", text_unit_ids=["t1"]),
        _ent("e2", "Acme", text_unit_ids=["t1", "t2"]),
    ]
    merged = resolver.entity_resolver._merge_entities(entities)
    # Deduped union {t1, t2} -> frequency 2.
    assert merged.frequency == 2
    assert set(merged.text_unit_ids) == {"t1", "t2"}


def test_merge_canonical_name_is_most_common(resolver: GraphResolver) -> None:
    entities = [
        _ent("e1", "Acme"),
        _ent("e2", "Acme"),
        _ent("e3", "ACME"),
    ]
    merged = resolver.entity_resolver._merge_entities(entities)
    assert merged.name == "Acme"
    # The surviving id is the primary (most-common-name) entity's.
    assert merged.id in {"e1", "e2"}


def test_merge_single_entity_returns_input_unchanged(resolver: GraphResolver) -> None:
    e = _ent("e1", "Solo", text_unit_ids=["t1"])
    assert resolver.entity_resolver._merge_entities([e]) is e


def test_merge_created_at_is_none_when_unknown(resolver: GraphResolver) -> None:
    # No member carries created_at -> merged created_at stays None (not a
    # fabricated wall-clock time), so the merge is a pure function of its inputs.
    entities = [_ent("e1", "Acme"), _ent("e2", "Acme")]
    merged = resolver.entity_resolver._merge_entities(entities)
    assert merged.created_at is None


def test_entity_merge_is_deterministic(resolver: GraphResolver) -> None:
    entities = [
        _ent("e1", "Acme", description="d1", text_unit_ids=["t1"]),
        _ent("e2", "Acme", description="d2", text_unit_ids=["t2"]),
    ]
    a = resolver.entity_resolver._merge_entities(entities)
    b = resolver.entity_resolver._merge_entities(entities)
    # Everything except the wall-clock updated_at must be identical run to run.
    assert (a.id, a.name, a.description, set(a.text_unit_ids), a.frequency) == (
        b.id,
        b.name,
        b.description,
        set(b.text_unit_ids),
        b.frequency,
    )


# --------------------------------------------------------------------------- #
# Relationship resolution
# --------------------------------------------------------------------------- #


def test_resolve_removes_self_referencing_edges(resolver: GraphResolver) -> None:
    rels = [Relationship(id="r1", source_id="a", target_id="b")]
    # entity_mapping collapses both endpoints onto the same id -> self-loop.
    resolved, stats = resolver.relationship_resolver.resolve(rels, {"a": "x", "b": "x"})
    assert resolved == []
    assert stats.self_referencing_removed == 1


def test_resolve_remaps_endpoints(resolver: GraphResolver) -> None:
    rels = [Relationship(id="r1", source_id="a", target_id="b")]
    resolved, _ = resolver.relationship_resolver.resolve(rels, {"a": "x", "b": "y"})
    assert (resolved[0].source_id, resolved[0].target_id) == ("x", "y")


def test_group_similar_relationships_normalizes_type(resolver: GraphResolver) -> None:
    rels = [
        Relationship(id="r1", source_id="a", target_id="b", type="WORKS_FOR"),
        Relationship(id="r2", source_id="a", target_id="b", type=" works_for "),
    ]
    resolved, _ = resolver.relationship_resolver.resolve(rels, {})
    # Case/whitespace variants of the same type between the same endpoints merge.
    assert len(resolved) == 1


def test_relationship_weight_is_supporting_text_unit_count(
    resolver: GraphResolver,
) -> None:
    rels = [
        Relationship(id="r1", source_id="a", target_id="b", text_unit_ids=["t1"]),
        Relationship(id="r2", source_id="a", target_id="b", text_unit_ids=["t1", "t2"]),
    ]
    resolved, _ = resolver.relationship_resolver.resolve(rels, {})
    assert len(resolved) == 1
    # Deduped union {t1, t2} -> weight 2.0 (evidence count, not summed strength).
    assert resolved[0].weight == 2.0


def test_relationship_weight_falls_back_without_text_units(
    resolver: GraphResolver,
) -> None:
    rels = [
        Relationship(id="r1", source_id="a", target_id="b", weight=3.0),
        Relationship(id="r2", source_id="a", target_id="b", weight=4.0),
    ]
    resolved, _ = resolver.relationship_resolver.resolve(rels, {})
    # No lineage -> primary edge's own weight is kept (not summed to 7.0).
    assert resolved[0].weight == 3.0


# --------------------------------------------------------------------------- #
# resolve_graph end-to-end + stats
# --------------------------------------------------------------------------- #


def test_resolve_graph_dedups_entities_and_keeps_edges(resolver: GraphResolver) -> None:
    entities = [
        _ent("e1", "Acme", text_unit_ids=["t1"]),
        _ent("e2", "Acme", text_unit_ids=["t2"]),
        _ent("e3", "Beta", text_unit_ids=["t3"]),
    ]
    relationships = [Relationship(id="r1", source_id="e1", target_id="e3")]
    result, stats = resolver.resolve_graph(entities, relationships)

    # The two "Acme" entities collapse to one; "Beta" stays.
    assert len(result["entities"]) == 2
    assert stats.entity_stats.original_entities == 3
    assert stats.entity_stats.resolved_entities == 2
    # The edge survives, remapped onto the surviving Acme id.
    assert len(result["relationships"]) == 1


def test_resolve_graph_never_increases_counts(resolver: GraphResolver) -> None:
    entities = [_ent(f"e{i}", "Same") for i in range(5)]
    rels = [Relationship(id=f"r{i}", source_id="e0", target_id="e1") for i in range(3)]
    result, _ = resolver.resolve_graph(entities, rels)
    assert len(result["entities"]) <= len(entities)
    assert len(result["relationships"]) <= len(rels)


def test_resolve_graph_is_idempotent_fixed_point(resolver: GraphResolver) -> None:
    entities = [_ent("e1", "Acme", text_unit_ids=["t1"]), _ent("e2", "Beta")]
    rels = [Relationship(id="r1", source_id="e1", target_id="e2")]
    once, _ = resolver.resolve_graph(entities, rels)
    twice, _ = resolver.resolve_graph(once["entities"], once["relationships"])
    # Resolving an already-resolved graph changes neither count.
    assert len(twice["entities"]) == len(once["entities"])
    assert len(twice["relationships"]) == len(once["relationships"])


# --------------------------------------------------------------------------- #
# Stats reduction-rate guards (zero-division branches)
# --------------------------------------------------------------------------- #


def test_entity_stats_reduction_rate_zero_when_empty() -> None:
    assert EntityResolutionStats(original_entities=0).reduction_rate == 0.0


def test_relationship_stats_reduction_rate_zero_when_empty() -> None:
    assert RelationshipResolutionStats(original_relationships=0).reduction_rate == 0.0


def test_graph_stats_overall_reduction_rate() -> None:
    stats = GraphResolutionStats(
        entity_stats=EntityResolutionStats(original_entities=10, resolved_entities=6),
        relationship_stats=RelationshipResolutionStats(
            original_relationships=10, resolved_relationships=4
        ),
    )
    # (20 - 10) / 20 * 100 == 50.0
    assert stats.overall_reduction_rate == 50.0


def test_graph_stats_overall_reduction_rate_zero_when_empty() -> None:
    assert GraphResolutionStats().overall_reduction_rate == 0.0
