# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for GlobalSearchStrategy community relevance rating.

Regression: the LLM emits a 0-10 relevance score, but relevance_threshold is
config-constrained to [0.0, 1.0]. The rating must be normalized to 0-1 so the
dynamic selection's threshold compares like-for-like. The strategy is built
via __new__ and the relevance scorer stubbed so no Bedrock client is built.
"""

from __future__ import annotations

import asyncio

import pytest

from unified_kg_rag.adapters.search_strategies.global_search import GlobalSearchStrategy
from unified_kg_rag.domain.models import RetrievalResult, SearchQuery

pytestmark = pytest.mark.unit


class _ScorerStub:
    """Async runnable returning a fixed 0-10 LLM relevance string."""

    def __init__(self, score_text: str) -> None:
        self._score_text = score_text

    async def ainvoke(self, _inputs: dict, config=None) -> str:
        return self._score_text


def _strategy(score_text: str) -> GlobalSearchStrategy:
    strat = GlobalSearchStrategy.__new__(GlobalSearchStrategy)
    strat.community_relevance_scorer = _ScorerStub(score_text)
    return strat


async def _rate(score_text: str) -> float:
    report = RetrievalResult(
        content="community", score=0.5, source="c0", retriever_type="community"
    )
    return await _strategy(score_text)._rate_report(
        report, SearchQuery(query="q"), asyncio.Semaphore(1)
    )


@pytest.mark.parametrize(("text", "expected"), [("8", 0.8), ("2", 0.2), ("10", 1.0)])
async def test_rating_is_normalized_into_the_unit_interval(
    text: str, expected: float
) -> None:
    # Before the fix the raw 0-10 score was compared with the 0-1 threshold,
    # so every report passed.
    assert await _rate(text) == pytest.approx(expected)


async def test_unparseable_rating_defaults_to_the_middle() -> None:
    assert await _rate("no number here") == pytest.approx(0.5)
