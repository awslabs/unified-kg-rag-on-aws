# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
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
