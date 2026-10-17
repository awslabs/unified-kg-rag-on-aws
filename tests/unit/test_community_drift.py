# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Community drift of incremental runs (domain/ingestion/community_drift)."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.community_drift import (
    measure_community_drift,
    with_level0_community_ids,
)
from unified_kg_rag.domain.models import Community, Entity

pytestmark = pytest.mark.unit


def _community(cid: str, level: int, members: list[str]) -> Community:
    return Community(
        id=cid, name=cid, level=str(level), parent="", children=[], entity_ids=members
    )


class TestWithLevel0CommunityIds:
    def test_adds_only_level0_ids_and_keeps_stored_ones(self) -> None:
        entities = [
            Entity(id="a", name="A", community_ids=["old"]),
            Entity(id="b", name="B"),
            Entity(id="c", name="C"),
        ]
        communities = [
            _community("c0", 0, ["a", "b"]),
            _community("c1", 1, ["a", "b", "c"]),
        ]
        a, b, c = with_level0_community_ids(entities, communities)
        assert a.community_ids == ["old", "c0"]
        assert b.community_ids == ["c0"]
        assert c is entities[2]
        # The input entities are left as they were.
        assert entities[0].community_ids == ["old"]

    def test_is_idempotent(self) -> None:
        communities = [_community("c0", 0, ["a"])]
        once = with_level0_community_ids([Entity(id="a", name="A")], communities)
        assert with_level0_community_ids(once, communities)[0].community_ids == ["c0"]


class TestMeasureCommunityDrift:
    def test_none_without_communities(self) -> None:
        assert measure_community_drift([Entity(id="a", name="A")], [], 0.5) is None

    def test_a_first_clustering_has_no_drift(self) -> None:
        communities = [_community("c0", 0, ["a", "b"]), _community("p", 1, ["a"])]
        entities = with_level0_community_ids(
            [Entity(id="a", name="A"), Entity(id="b", name="B")], communities
        )
        drift = measure_community_drift(entities, communities, 0.5)
        assert drift is not None
        assert drift.model_dump() == {
            "delta_communities": 2,
            "clustered_entities": 2,
            "overlapping_entities": 0,
            "stale_communities": 0,
            "overlap_share": 0.0,
            "rebuild_recommended": False,
        }

    def test_counts_entities_an_earlier_clustering_placed(self) -> None:
        # a and b were clustered before (stored ids e1, e2); c and d are new.
        communities = [_community("c0", 0, ["a", "b", "c", "d"])]
        entities = with_level0_community_ids(
            [
                Entity(id="a", name="A", community_ids=["e1"]),
                Entity(id="b", name="B", community_ids=["e2"]),
                Entity(id="c", name="C"),
                Entity(id="d", name="D"),
            ],
            communities,
        )
        drift = measure_community_drift(entities, communities, 0.5)
        assert drift is not None
        assert drift.overlapping_entities == 2
        assert drift.stale_communities == 2
        assert drift.overlap_share == 0.5
        assert drift.rebuild_recommended
        below = measure_community_drift(entities, communities, 0.6)
        assert below is not None and not below.rebuild_recommended
