# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for GlobalSearchStrategy community relevance scoring.

Regression: the LLM emits a 0-10 relevance score, but relevance_threshold is
config-constrained to [0.0, 1.0]. The threshold filter must compare like-for-like
(0-1), or it becomes a near no-op (default 0.5 threshold vs 0-10 scores passes
almost everything). These tests construct the strategy via __new__ and stub the
relevance scorer so no Bedrock client is built.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from aws_graphrag.adapters.search_strategies.global_search import GlobalSearchStrategy
from aws_graphrag.domain.models import RetrievalResult, SearchQuery

pytestmark = pytest.mark.unit


class _ScorerStub:
    """Async runnable returning a fixed 0-10 LLM relevance string."""

    def __init__(self, score_text: str) -> None:
        self._score_text = score_text

    async def ainvoke(self, _inputs: dict) -> str:
        return self._score_text


def _strategy(*, threshold: float, score_text: str) -> GlobalSearchStrategy:
    strat = GlobalSearchStrategy.__new__(GlobalSearchStrategy)
    strat.global_search_config = SimpleNamespace(
        max_communities=100,
        use_dynamic_selection=True,
        relevance_threshold=threshold,
    )
    strat.ignore_errors = False
    strat.community_relevance_scorer = _ScorerStub(score_text)
    return strat


def _communities(n: int) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            content=f"community {i}", score=0.5, source=f"c{i}", retriever_type="graph"
        )
        for i in range(n)
    ]


async def test_high_llm_score_passes_threshold() -> None:
    # LLM says 8/10 (=0.8 normalized); default threshold 0.5 -> kept.
    strat = _strategy(threshold=0.5, score_text="8")
    query = SearchQuery(query="q", retrieval_multiplier=1)
    kept = await strat._select_relevant_communities(_communities(3), query)
    assert len(kept) == 3


async def test_low_llm_score_filtered_out() -> None:
    # LLM says 2/10 (=0.2 normalized); threshold 0.5 -> dropped. Before the fix,
    # the raw 0-10 score (2.0) was >= 0.5 so nothing was ever filtered.
    strat = _strategy(threshold=0.5, score_text="2")
    query = SearchQuery(query="q", retrieval_multiplier=1)
    kept = await strat._select_relevant_communities(_communities(3), query)
    assert kept == []


async def test_score_normalized_into_unit_interval() -> None:
    # A perfect 10/10 must blend to a score within [0, 1], not blow past 1.
    strat = _strategy(threshold=0.0, score_text="10")
    query = SearchQuery(query="q", retrieval_multiplier=1)
    kept = await strat._select_relevant_communities(_communities(1), query)
    assert kept and 0.0 <= kept[0].score <= 1.0


def _strategy_static(*, max_communities: int) -> GlobalSearchStrategy:
    strat = GlobalSearchStrategy.__new__(GlobalSearchStrategy)
    strat.global_search_config = SimpleNamespace(
        max_communities=max_communities,
        use_dynamic_selection=False,
        relevance_threshold=0.5,
    )
    strat.ignore_errors = False
    return strat


def _communities_with_rank(ranks: list[float]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            content=f"community {i}",
            score=0.5,
            source=f"c{i}",
            retriever_type="graph",
            metadata={"rank": r},
        )
        for i, r in enumerate(ranks)
    ]


async def test_static_selection_keeps_highest_rank_communities() -> None:
    # use_dynamic_selection=False: select by community-report `rank` (graph
    # importance) instead of arbitrary retrieval order, capped at max_communities.
    strat = _strategy_static(max_communities=2)
    query = SearchQuery(query="q", retrieval_multiplier=1)
    # ranks deliberately out of order; top-2 by rank are c2(9.0) and c0(5.0).
    kept = await strat._select_relevant_communities(
        _communities_with_rank([5.0, 1.0, 9.0, 2.0]), query
    )
    assert [c.source for c in kept] == ["c2", "c0"]


async def test_static_selection_missing_rank_sorts_last() -> None:
    strat = _strategy_static(max_communities=2)
    query = SearchQuery(query="q", retrieval_multiplier=1)
    ranked = _communities_with_rank([3.0])
    unranked = [
        RetrievalResult(
            content="no-rank", score=0.5, source="cX", retriever_type="graph"
        )
    ]
    kept = await strat._select_relevant_communities(ranked + unranked, query)
    # The ranked community comes first; the rank-less one sorts last.
    assert kept[0].source == "c0"
