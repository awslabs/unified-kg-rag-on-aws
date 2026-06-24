# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for incremental merge logic (M2)."""

from __future__ import annotations

import pytest

from aws_graphrag.domain.ingestion.merge import (
    merge_communities,
    merge_community_reports,
    merge_entities,
    merge_relationships,
)
from aws_graphrag.domain.models import Community, CommunityReport, Entity, Relationship

pytestmark = pytest.mark.unit


def _entity(id_: str, name: str, **kw) -> Entity:
    return Entity(id=id_, name=name, **kw)


class TestMergeEntities:
    def test_new_entity_appended(self) -> None:
        old = [_entity("e1", "Alice", text_unit_ids=["t1"])]
        delta = [_entity("e9", "Bob", text_unit_ids=["t2"])]
        merged, remap = merge_entities(old, delta)
        assert {e.name for e in merged} == {"Alice", "Bob"}
        assert remap == {}

    def test_same_name_merges_into_old_id(self) -> None:
        old = [_entity("e1", "Alice", description="desc A", text_unit_ids=["t1"])]
        delta = [_entity("e9", "alice", description="desc B", text_unit_ids=["t2"])]
        merged, remap = merge_entities(old, delta)
        assert len(merged) == 1
        survivor = merged[0]
        assert survivor.id == "e1"  # old id preserved
        assert remap == {"e9": "e1"}
        assert set(survivor.text_unit_ids) == {"t1", "t2"}
        assert "desc A" in survivor.description and "desc B" in survivor.description

    def test_frequency_recomputed_from_text_units(self) -> None:
        old = [_entity("e1", "Alice", rank=5, text_unit_ids=["t1", "t2"])]
        delta = [_entity("e9", "Alice", text_unit_ids=["t2", "t3"])]
        merged, _ = merge_entities(old, delta)
        # union {t1,t2,t3} -> frequency 3; rank (graph importance) is preserved.
        assert merged[0].frequency == 3
        assert merged[0].rank == 5

    def test_type_backfilled_when_old_missing(self) -> None:
        old = [_entity("e1", "Alice", type=None)]
        delta = [_entity("e9", "Alice", type="PERSON")]
        merged, _ = merge_entities(old, delta)
        assert merged[0].type == "PERSON"

    def test_idempotent_merge_of_identical_delta(self) -> None:
        old = [_entity("e1", "Alice", text_unit_ids=["t1"])]
        delta = [_entity("e1", "Alice", text_unit_ids=["t1"])]
        merged, _ = merge_entities(old, delta)
        assert len(merged) == 1
        assert merged[0].text_unit_ids == ["t1"]


class TestMergeRelationships:
    def test_new_relationship_appended(self) -> None:
        old = [Relationship(id="r1", source_id="e1", target_id="e2")]
        delta = [Relationship(id="r2", source_id="e2", target_id="e3")]
        merged = merge_relationships(old, delta)
        assert len(merged) == 2

    def test_same_endpoints_merge_and_sum_weight(self) -> None:
        # Weights are summed (MS GraphRAG semantics): order-independent and
        # associative, unlike a running pairwise mean.
        old = [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                weight=1.0,
                text_unit_ids=["t1"],
            )
        ]
        delta = [
            Relationship(
                id="r2",
                source_id="e1",
                target_id="e2",
                weight=3.0,
                text_unit_ids=["t2"],
            )
        ]
        merged = merge_relationships(old, delta)
        assert len(merged) == 1
        assert merged[0].weight == 4.0
        assert set(merged[0].text_unit_ids) == {"t1", "t2"}

    def test_endpoints_remapped_via_entity_remap(self) -> None:
        old = [Relationship(id="r1", source_id="e1", target_id="e2")]
        # delta edge references e9 which merged into e1.
        delta = [Relationship(id="r2", source_id="e9", target_id="e2")]
        merged = merge_relationships(old, delta, entity_id_remap={"e9": "e1"})
        # After remap, (e1,e2) collides with old -> merged into one.
        assert len(merged) == 1

    def test_distinct_types_between_same_endpoints_kept_separate(self) -> None:
        # A delta edge of a DIFFERENT type between the same endpoints must stay
        # a distinct edge (merge key includes normalized type) — not collapse
        # into the old edge and silently drop the delta type.
        old = [Relationship(id="r1", source_id="e1", target_id="e2", type="WORKS_AT")]
        delta = [Relationship(id="r2", source_id="e1", target_id="e2", type="FOUNDED")]
        merged = merge_relationships(old, delta)
        assert len(merged) == 2
        assert {r.type for r in merged} == {"WORKS_AT", "FOUNDED"}

    def test_same_type_case_insensitive_merges(self) -> None:
        # Type matching is normalized (strip + lowercase): "WORKS_AT" and
        # " works_at " are the same edge and merge.
        old = [
            Relationship(
                id="r1", source_id="e1", target_id="e2", type="WORKS_AT", weight=1.0
            )
        ]
        delta = [
            Relationship(
                id="r2", source_id="e1", target_id="e2", type=" works_at ", weight=2.0
            )
        ]
        merged = merge_relationships(old, delta)
        assert len(merged) == 1
        assert merged[0].weight == 3.0


class TestMergeCommunities:
    def test_appends_and_disambiguates_colliding_ids(self) -> None:
        old = [Community(id="c1", name="A", level="0", parent="", children=[])]
        delta = [
            Community(id="c1", name="B", level="0", parent="", children=[]),
            Community(id="c2", name="C", level="0", parent="", children=[]),
        ]
        merged = merge_communities(old, delta)
        ids = [c.id for c in merged]
        assert ids == ["c1", "c1-delta", "c2"]

    def test_remerging_same_delta_is_idempotent(self) -> None:
        # Re-applying a delta must not grow `c1-delta` into `c1-delta-delta`.
        old = [Community(id="c1", name="A", level="0", parent="", children=[])]
        delta = [Community(id="c1", name="B", level="0", parent="", children=[])]
        once = merge_communities(old, delta)
        twice = merge_communities(once, delta)
        assert [c.id for c in twice] == ["c1", "c1-delta"]

    def test_reports_appended(self) -> None:
        old = [CommunityReport(id="cr1", community_id="c1", name="R1")]
        delta = [CommunityReport(id="cr1", community_id="c1", name="R2")]
        merged = merge_community_reports(old, delta)
        assert [r.id for r in merged] == ["cr1", "cr1-delta"]
        # Idempotent on re-merge.
        twice = merge_community_reports(merged, delta)
        assert [r.id for r in twice] == ["cr1", "cr1-delta"]
