# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the full-build graph resolver merge/resolution logic.

graph_resolver.py is pure domain (no AWS) and carries the entity/relationship
merge arithmetic that feeds ranking and the incremental path. These lock the
merge semantics — max-confidence, text-unit union, summed per-text-unit
weight, self-loop removal, type normalization, determinism — and the stats
reduction-rate guards. Run with use_process_pool=False so grouping is
deterministic and in-process.
"""

from __future__ import annotations

import logging

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


def test_merge_unions_text_units(
    resolver: GraphResolver,
) -> None:
    entities = [
        _ent("e1", "Acme", text_unit_ids=["t1"]),
        _ent("e2", "Acme", text_unit_ids=["t1", "t2"]),
    ]
    merged = resolver.entity_resolver._merge_entities(entities)
    assert set(merged.text_unit_ids) == {"t1", "t2"}


def test_merge_canonical_name_is_the_best_supported(resolver: GraphResolver) -> None:
    entities = [
        _ent("e1", "ACME", text_unit_ids=["t1"]),
        _ent("e2", "Acme", text_unit_ids=["t1", "t2"]),
        _ent("e3", "Acme Inc", text_unit_ids=["t3"]),
    ]
    for ordering in (entities, entities[::-1]):
        merged = resolver.entity_resolver._merge_entities(ordering)
        # The name cited by the most text units wins, whatever the order.
        assert (merged.name, merged.id) == ("Acme", "e2")


def test_merge_canonical_name_tie_takes_the_smallest_name(
    resolver: GraphResolver,
) -> None:
    entities = [
        _ent("e1", "Acme", text_unit_ids=["t1"]),
        _ent("e2", "ACME", text_unit_ids=["t2"]),
    ]
    for ordering in (entities, entities[::-1]):
        assert resolver.entity_resolver._merge_entities(ordering).id == "e2"


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
    assert (a.id, a.name, a.description, set(a.text_unit_ids)) == (
        b.id,
        b.name,
        b.description,
        set(b.text_unit_ids),
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


def test_relationship_weight_sums_strengths_per_text_unit(
    resolver: GraphResolver,
) -> None:
    rels = [
        Relationship(
            id="r1", source_id="a", target_id="b", weight=8.0, text_unit_ids=["t1"]
        ),
        Relationship(
            id="r2", source_id="a", target_id="b", weight=3.0, text_unit_ids=["t2"]
        ),
        Relationship(
            id="r3", source_id="a", target_id="b", weight=1.0, text_unit_ids=["t1"]
        ),
    ]
    resolved, _ = resolver.relationship_resolver.resolve(rels, {})
    assert len(resolved) == 1
    # MS GraphRAG sums instance strengths; the per-text-unit sums are kept so
    # the incremental merge can reproduce the weight.
    assert resolved[0].weight == 12.0
    assert resolved[0].text_unit_ids == ["t1", "t2"]
    assert resolved[0].attributes == {"text_unit_weights": [9.0, 3.0]}


def test_single_chunk_relationship_keeps_its_strength(
    resolver: GraphResolver,
) -> None:
    rels = [
        Relationship(
            id="r1", source_id="a", target_id="b", weight=8.0, text_unit_ids=["t1"]
        )
    ]
    resolved, _ = resolver.relationship_resolver.resolve(rels, {})
    assert resolved[0].weight == 8.0
    assert resolved[0].attributes == {"text_unit_weights": [8.0]}


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


def test_resolve_graph_renames_edges_onto_the_merged_entity(
    resolver: GraphResolver,
) -> None:
    # "Acme" merges into the better-supported "Acme Inc"; the edge extracted
    # as "Acme" -> "Beta" must name the surviving entity, which the indexers
    # and the retrieval context read.
    entities = [
        _ent("acme-inc", "Acme Inc", text_unit_ids=["t1", "t2"]),
        _ent("acme", "Acme", text_unit_ids=["t3"]),
        _ent("beta", "Beta", text_unit_ids=["t3"]),
    ]
    rels = [
        Relationship(
            id="r1",
            source_id="acme",
            source_name="Acme",
            target_id="beta",
            target_name="Beta",
            type="SUPPLIES",
        )
    ]
    result, _ = resolver.resolve_graph(entities, rels)
    (rel,) = result["relationships"]
    assert (rel.source_id, rel.source_name) == ("acme-inc", "Acme Inc")
    assert rel.target_name == "Beta"
    # The input edge is not mutated.
    assert rels[0].source_name == "Acme"


def test_resolve_graph_unions_document_filters(resolver: GraphResolver) -> None:
    entities = [
        _ent(
            "acme-inc",
            "Acme Inc",
            text_unit_ids=["t1", "t2"],
            attributes={"filters": {"category": "research"}, "index": "a"},
        ),
        _ent(
            "acme",
            "Acme",
            text_unit_ids=["t3"],
            attributes={"filters": {"category": "finance"}, "index": "a"},
        ),
    ]
    result, _ = resolver.resolve_graph(entities, [])
    (entity,) = result["entities"]
    assert entity.attributes == {
        "filters": {"category": ["finance", "research"]},
        "index": "a",
    }


def _fuzzy_pool_resolver(*, ignore_errors: bool) -> GraphResolver:
    config = Config()
    config.processing.fuzzy_entity_resolution = True
    config.processing.ignore_errors = ignore_errors
    return GraphResolver(config, max_workers=2, use_process_pool=False)


def _failing_match(name: str) -> list:
    raise MemoryError("worker died")


@pytest.mark.parametrize("ignore_errors", [False, True])
def test_fuzzy_match_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    ignore_errors: bool,
) -> None:
    from unified_kg_rag.domain.ingestion import graph_resolver as module
    from unified_kg_rag.shared import GraphError

    # A failed match used to be logged and dropped even with ignore_errors
    # off, so the name's variants silently stayed duplicate entities.
    monkeypatch.setattr(module, "find_all_matches_for_entity_task", _failing_match)
    entities = [
        _ent("e1", "Acme Corporation Holdings", text_unit_ids=["t1"]),
        _ent("e2", "Acme Corporation Holding", text_unit_ids=["t2"]),
    ]
    resolver = _fuzzy_pool_resolver(ignore_errors=ignore_errors)
    caplog.set_level(logging.INFO)
    if ignore_errors:
        result, _ = resolver.resolve_graph(entities, [])
        assert len(result["entities"]) == 2
        assert "Failed to find matches" in caplog.text
    else:
        with pytest.raises(GraphError, match="Fuzzy matching failed") as excinfo:
            resolver.resolve_graph(entities, [])
        # The error is logged by the pipeline at ERROR: no entity name in it.
        assert "Acme" not in str(excinfo.value)
    # Corpus entity names stay out of INFO+ records.
    assert "Acme" not in caplog.text


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
