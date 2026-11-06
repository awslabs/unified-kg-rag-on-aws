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


async def _rate(score_text: str) -> float | None:
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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # float() of the whole answer read each of these as the default 5.
        ("8/10", 0.8),
        ("Score: 2", 0.2),
        ("1\n\nIrrelevant: the community is about cats", 0.1),
        ("1\n\nIrrelevant", 0.1),
        ("10/10", 1.0),
        ("Rating: 3 out of 10", 0.3),
        # A mention of the scale is not the rating (it was read as 1).
        ("On a scale of 1-10, I would rate this 8", 0.8),
        ("On a scale of 1 to 10: 7", 0.7),
        ("On a scale of 0\u201310, 6", 0.6),
        # The scale mentioned in Korean, Japanese or Chinese (each was read
        # as 10, or as 1).
        ("10점 만점에 7점", 0.7),
        ("7점 (10점 만점)", 0.7),
        ("1에서 10 사이에서 4점", 0.4),
        ("10点満点中6点", 0.6),
        ("1から10で3", 0.3),
        ("满分10分，我给5分", 0.5),
        ("1到10分之间：9", 0.9),
        ("10分制：2分", 0.2),
        # Clamped to the prompt's 1-10 scale.
        ("85", 1.0),
        ("0", 0.1),
    ],
)
async def test_a_rating_is_read_by_its_first_number(text: str, expected: float) -> None:
    assert await _rate(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text", ["no number here", "nan", "inf", "", "On a scale of 1-10: high"]
)
async def test_a_rating_without_a_number_is_unrated(text: str) -> None:
    # "nan" was read as NaN, which no threshold drops.
    assert await _rate(text) is None
