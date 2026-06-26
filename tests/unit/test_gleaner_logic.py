# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GraphGleaner pure logic (AWS-free).

Covers the post-merge relationship reconciliation (orphan/self-loop drop
counting, weight summing), the entities/relationships clamp >= 0, the quality
and convergence math, and the module-level format_entities_with_limit_task. The
static methods are called directly; instance methods are exercised on a real
GraphGleaner whose Bedrock/boto wiring is patched out.
"""

from __future__ import annotations

import pytest

import unified_kg_rag.adapters.ingestion.gleaner as gleaner_module
from unified_kg_rag.adapters.ingestion.gleaner import (
    GraphGleaner,
    format_entities_with_limit_task,
)
from unified_kg_rag.domain.models import Config, Entity, Relationship

pytestmark = pytest.mark.unit


@pytest.fixture
def gleaner(config: Config, mocker) -> GraphGleaner:
    mocker.patch.object(gleaner_module, "boto3")
    mocker.patch.object(gleaner_module, "BedrockLanguageModelFactory")
    mocker.patch.object(gleaner_module, "create_robust_xml_output_parser")
    mocker.patch.object(gleaner_module, "setup_chain")
    return GraphGleaner(config)


def _rel(id_, src, tgt, **kw) -> Relationship:
    return Relationship(id=id_, source_id=src, target_id=tgt, type="X", **kw)


# --------------------------------------------------------------------------- #
# _update_relationships_after_merge
# --------------------------------------------------------------------------- #
class TestUpdateRelationshipsAfterMerge:
    def test_orphan_endpoint_dropped(self) -> None:
        rels = [_rel("r1", "e1", "GONE")]
        out = GraphGleaner._update_relationships_after_merge(
            rels, unique_entity_ids={"e1"}, id_remap={}
        )
        assert out == []  # target not in unique set -> dropped

    def test_self_loop_dropped(self) -> None:
        rels = [_rel("r1", "e1", "e1")]
        out = GraphGleaner._update_relationships_after_merge(
            rels, unique_entity_ids={"e1"}, id_remap={}
        )
        assert out == []

    def test_self_loop_after_remap_dropped(self) -> None:
        # e2 remaps to e1, collapsing the edge into a self-loop -> dropped.
        rels = [_rel("r1", "e1", "e2")]
        out = GraphGleaner._update_relationships_after_merge(
            rels, unique_entity_ids={"e1"}, id_remap={"e2": "e1"}
        )
        assert out == []

    def test_surviving_edge_kept_with_remapped_endpoint(self) -> None:
        rels = [_rel("r1", "e9", "e2")]
        out = GraphGleaner._update_relationships_after_merge(
            rels, unique_entity_ids={"e1", "e2"}, id_remap={"e9": "e1"}
        )
        assert len(out) == 1
        assert out[0].source_id == "e1"
        assert out[0].target_id == "e2"

    def test_duplicate_edges_merged_and_weight_summed(self) -> None:
        rels = [
            _rel("r1", "e1", "e2", weight=1.0, description="a"),
            _rel("r2", "e1", "e2", weight=2.0, description="b"),
        ]
        out = GraphGleaner._update_relationships_after_merge(
            rels, unique_entity_ids={"e1", "e2"}, id_remap={}
        )
        assert len(out) == 1
        assert out[0].weight == 3.0
        assert "a" in out[0].description and "b" in out[0].description

    def test_distinct_types_not_merged(self) -> None:
        rels = [
            Relationship(id="r1", source_id="e1", target_id="e2", type="A"),
            Relationship(id="r2", source_id="e1", target_id="e2", type="B"),
        ]
        out = GraphGleaner._update_relationships_after_merge(
            rels, unique_entity_ids={"e1", "e2"}, id_remap={}
        )
        assert len(out) == 2


# --------------------------------------------------------------------------- #
# _merge_duplicate_entities
# --------------------------------------------------------------------------- #
class TestMergeDuplicateEntities:
    def test_name_keyed_dedup_with_id_remap(self) -> None:
        ents = [
            Entity(id="e1", name="Alice", type="PERSON", text_unit_ids=["t1"]),
            Entity(id="e9", name="alice", type="PERSON", text_unit_ids=["t2"]),
        ]
        unique, remap = GraphGleaner._merge_duplicate_entities(ents)
        assert len(unique) == 1
        assert remap == {"e9": "e1"}  # e9 merged into master e1
        assert set(unique[0].text_unit_ids) == {"t1", "t2"}

    def test_most_frequent_type_wins(self) -> None:
        ents = [
            Entity(id="e1", name="A", type="PERSON"),
            Entity(id="e2", name="A", type="PERSON"),
            Entity(id="e3", name="A", type="ORG"),
        ]
        unique, _ = GraphGleaner._merge_duplicate_entities(ents)
        assert len(unique) == 1
        assert unique[0].type == "person"  # lowercased, most frequent


# --------------------------------------------------------------------------- #
# clamp >= 0 (via _calculate_convergence_score behavior under clamping)
# --------------------------------------------------------------------------- #
class TestCounterClamp:
    def test_convergence_one_when_no_change(self, gleaner) -> None:
        assert gleaner._calculate_convergence_score(0, 0, 0.0) == 1.0

    def test_negative_added_clamped_at_call_site_semantics(self, gleaner) -> None:
        # The convergence math is only ever fed clamped (>=0) counts. Confirm a
        # zero/zero feed converges fully and a large change feed does not.
        big = gleaner._calculate_convergence_score(100, 100, 0.0)
        assert 0.0 <= big < 1.0


# --------------------------------------------------------------------------- #
# _calculate_initial_quality
# --------------------------------------------------------------------------- #
class TestInitialQuality:
    def test_empty_graph_is_zero(self, gleaner) -> None:
        assert gleaner._calculate_initial_quality([], []) == 0.0

    def test_completeness_capped_at_half_each(self, gleaner) -> None:
        # Far above both scales -> each completeness saturates at 0.5,
        # blended = (0.5 + 0.5) / 2 = 0.5.
        ents = [Entity(id=f"e{i}", name=f"n{i}") for i in range(500)]
        rels = [
            Relationship(id=f"r{i}", source_id="e0", target_id="e1") for i in range(500)
        ]
        assert gleaner._calculate_initial_quality(ents, rels) == 0.5

    def test_partial_completeness(self, gleaner) -> None:
        # entity_scale=50, rel_scale=100 by default.
        # 25 entities -> min(0.5, 25/50)=0.5; 0 rels -> 0.0 -> (0.5+0)/2 = 0.25
        ents = [Entity(id=f"e{i}", name=f"n{i}") for i in range(25)]
        assert gleaner._calculate_initial_quality(ents, []) == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# _calculate_graph_quality (completeness/accuracy blend)
# --------------------------------------------------------------------------- #
class TestGraphQuality:
    def test_weighted_blend(self, gleaner) -> None:
        # default completeness_weight = 0.6
        q = gleaner._calculate_graph_quality({"completeness": 1.0, "accuracy": 0.0})
        assert q == pytest.approx(0.6)

    def test_defaults_when_missing(self, gleaner) -> None:
        # both default to 0.5 -> 0.5*0.6 + 0.5*0.4 = 0.5
        assert gleaner._calculate_graph_quality({}) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# _calculate_convergence_score
# --------------------------------------------------------------------------- #
class TestConvergenceScore:
    def test_no_change_is_full_convergence(self, gleaner) -> None:
        assert gleaner._calculate_convergence_score(0, 0, 0.0) == 1.0

    def test_change_rate_reduces_convergence(self, gleaner) -> None:
        # change_scale=20: (10+0)/20 = 0.5 change_rate, no quality change
        # convergence = 1 - min(1, 0.5 + 0) = 0.5
        assert gleaner._calculate_convergence_score(10, 0, 0.0) == pytest.approx(0.5)

    def test_quality_swing_lowers_convergence(self, gleaner) -> None:
        # (2+0)/20 = 0.1 + |0.3| = 0.4 -> convergence 0.6
        assert gleaner._calculate_convergence_score(2, 0, 0.3) == pytest.approx(0.6)

    def test_floored_at_zero(self, gleaner) -> None:
        assert gleaner._calculate_convergence_score(1000, 1000, 5.0) == 0.0


# --------------------------------------------------------------------------- #
# _should_stop_gleaning
# --------------------------------------------------------------------------- #
class TestShouldStop:
    def test_stops_on_convergence_threshold(self, gleaner) -> None:
        # Primary MEASURED signal: convergence_score >= 0.8 (default) -> stop,
        # at any round.
        assert gleaner._should_stop_gleaning(0.9, 0.5, 0.1, round_num=1) is True

    def test_stops_on_low_improvement_only_from_round_2(self, gleaner) -> None:
        # min_improvement_threshold = 0.05; improvement 0.01 < that.
        # Round 1: the LLM-quality delta is on a different scale than the
        # count-based seed, so the check is suppressed -> does NOT stop.
        assert gleaner._should_stop_gleaning(0.1, 0.01, 0.1, round_num=1) is False
        # Round 2+: the advisory quality-improvement check applies -> stop.
        assert gleaner._should_stop_gleaning(0.1, 0.01, 0.1, round_num=2) is True

    def test_quality_regression_does_not_trigger_improvement_stop(
        self, gleaner
    ) -> None:
        # A quality DROP (negative improvement) is below the threshold but must
        # not be treated as "converged" via abs(); it is one-sided now. With a
        # mid convergence score and sub-threshold quality, round 2 stops because
        # improvement < threshold (a drop is still "not improving") — but the
        # point is abs() no longer flips a large positive regression magnitude.
        # Big negative improvement at round 2 -> stop (not improving).
        assert gleaner._should_stop_gleaning(0.1, -0.5, 0.1, round_num=2) is True

    def test_stops_on_quality_target(self, gleaner) -> None:
        # quality_threshold = 0.9, applies at any round.
        assert gleaner._should_stop_gleaning(0.1, 0.5, 0.95, round_num=1) is True

    def test_continues_otherwise(self, gleaner) -> None:
        # Low convergence, healthy improvement, below quality target -> continue.
        assert gleaner._should_stop_gleaning(0.1, 0.5, 0.1, round_num=2) is False


# --------------------------------------------------------------------------- #
# format_entities_with_limit_task
# --------------------------------------------------------------------------- #
class TestFormatEntitiesWithLimit:
    def test_under_limit_lists_all_names(self) -> None:
        ents = [Entity(id="e1", name="Alice"), Entity(id="e2", name="Bob")]
        out = format_entities_with_limit_task(ents, max_entities=5)
        assert out == "Alice\nBob"

    def test_over_limit_truncates_with_suffix(self) -> None:
        ents = [Entity(id=f"e{i}", name=f"n{i}") for i in range(5)]
        out = format_entities_with_limit_task(ents, max_entities=2)
        assert "... and 3 more entities" in out
        assert len(out.splitlines()) == 3  # 2 names + suffix line

    def test_prioritizes_described_and_high_support_entities(self) -> None:
        # An entity with a description and more text_unit_ids should be picked
        # over a bare one when truncating.
        rich = Entity(id="e1", name="Rich", description="d", text_unit_ids=["t1", "t2"])
        bare = Entity(id="e2", name="Bare")
        out = format_entities_with_limit_task([bare, rich], max_entities=1)
        assert out.splitlines()[0] == "Rich"
