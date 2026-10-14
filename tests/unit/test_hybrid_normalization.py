# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``combine_min_max`` reproduces the neural-search min_max + arithmetic_mean."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.retrieval.hybrid_normalization import (
    MIN_SCORE,
    combine_min_max,
)

pytestmark = pytest.mark.unit


def test_min_max_per_sub_query_then_weighted_mean() -> None:
    lexical = [("a", 10.0), ("b", 6.0), ("c", 2.0)]
    vector = [("b", 0.9), ("d", 0.5)]
    fused = dict(combine_min_max([lexical, vector], [0.3, 0.7]))
    # lexical normalized: a 1.0, b 0.5, c 0 -> MIN_SCORE
    # vector normalized:  b 1.0, d 0 -> MIN_SCORE
    assert fused["a"] == pytest.approx(0.3 * 1.0)
    assert fused["b"] == pytest.approx(0.3 * 0.5 + 0.7 * 1.0)
    assert fused["c"] == pytest.approx(0.3 * MIN_SCORE)
    assert fused["d"] == pytest.approx(0.7 * MIN_SCORE)


def test_missing_sub_query_scores_zero_and_keeps_its_weight() -> None:
    # "a" is only a lexical hit: the vector weight still divides its score.
    fused = dict(combine_min_max([[("a", 3.0), ("b", 1.0)], []], [1.0, 3.0]))
    assert fused["a"] == pytest.approx(1.0 / 4.0)


def test_equal_scores_normalize_to_one() -> None:
    fused = combine_min_max([[("a", 2.0), ("b", 2.0)], [("c", 0.4)]], [0.5, 0.5])
    assert fused == [("a", 0.5), ("b", 0.5), ("c", 0.5)]


def test_ties_keep_first_appearance_order() -> None:
    fused = combine_min_max([[("x", 1.0)], [("y", 1.0)]], [0.5, 0.5])
    assert [doc_id for doc_id, _ in fused] == ["x", "y"]


def test_min_score_floor_can_rank_the_minimum_above_a_near_minimum_hit() -> None:
    # The plugin floors only an exact 0: a hit just above the minimum
    # normalizes to less than the floored minimum.
    fused = combine_min_max([[("top", 10.0), ("near", 0.0001), ("min", 0.0)]], [1.0])
    assert [doc_id for doc_id, _ in fused] == ["top", "min", "near"]


def test_zero_weights_score_zero() -> None:
    assert combine_min_max([[("a", 1.0)]], [0.0]) == [("a", 0.0)]


def test_no_hits() -> None:
    assert combine_min_max([[], []], [0.5, 0.5]) == []


@pytest.mark.parametrize("weights", [[1.0], [0.5, 0.5, 0.0], [1.5, -0.5]])
def test_weights_must_match_sub_queries_and_be_non_negative(weights) -> None:
    with pytest.raises(ValueError):
        combine_min_max([[("a", 1.0)], [("b", 1.0)]], weights)
