# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavioral tests for GlobalSearchStrategy community relevance rating.

The rating is MS GraphRAG's (``rate_prompt.py``): a JSON object with a
reason and a 0-5 rating, divided by 5 so ``relevance_threshold`` (0-1)
compares like for like. It replaced a bare 1-10 number read by its first
digits, which needed a list of scale mentions in four languages ("Auf einer
Skala von 1 bis 10: 8" read 1, pruning the report's whole subtree). The
strategy is built via __new__ and the relevance scorer stubbed so no Bedrock
client is built.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from unified_kg_rag.adapters.search_strategies.global_search import GlobalSearchStrategy
from unified_kg_rag.domain.models import Config, RetrievalResult, SearchQuery

pytestmark = pytest.mark.unit


class _ScorerStub:
    """Async runnable returning a fixed LLM rating answer."""

    def __init__(self, score_text: str) -> None:
        self._score_text = score_text

    async def ainvoke(self, _inputs: dict, config=None) -> str:
        return self._score_text


def _strategy(score_text: str, fixed: str = "still no JSON") -> GlobalSearchStrategy:
    strat = GlobalSearchStrategy.__new__(GlobalSearchStrategy)
    strat.config = Config()
    strat.providers = SimpleNamespace(  # type: ignore[assignment]
        llm_factory=ScriptedLLMFactory(lambda _system, _human: fixed)
    )
    strat.community_relevance_scorer = _ScorerStub(score_text)
    return strat


async def _rate(score_text: str, fixed: str = "still no JSON") -> float | None:
    report = RetrievalResult(
        content="community", score=0.5, source="c0", retriever_type="community"
    )
    return await _strategy(score_text, fixed)._rate_report(
        report, SearchQuery(query="q"), asyncio.Semaphore(1)
    )


def _answer(rating: object, reason: str = "It names the parties.") -> str:
    return json.dumps({"reason": reason, "rating": rating}, ensure_ascii=False)


@pytest.mark.parametrize(
    ("rating", "expected"), [(5, 1.0), (4, 0.8), (1, 0.2), (0, 0.0), ("3", 0.6)]
)
async def test_rating_is_normalized_into_the_unit_interval(
    rating: object, expected: float
) -> None:
    # 0.2 (the default relevance_threshold) is MS GraphRAG's rating of 1.
    assert await _rate(_answer(rating)) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        # The reason in any language mentions any scale; only "rating" counts.
        _answer(4, "Auf einer Skala von 1 bis 10 wäre das 8."),
        _answer(4, "10점 만점에 8점 정도로 관련이 있다."),
        _answer(4, "1から10で言えば8"),
        # Other families' JSON: a Python dict, a trailing comma, a fence.
        "{'reason': 'Names the Vendor.', 'rating': 4}",
        '{"reason": "Names the Vendor.", "rating": 4,}',
        'Here is my rating:\n```json\n{"reason": "x", "rating": 4}\n```',
    ],
)
async def test_the_rating_is_read_whatever_the_reason_says(text: str) -> None:
    assert await _rate(text) == pytest.approx(0.8)


@pytest.mark.parametrize(("rating", "expected"), [(85, 1.0), (-2, 0.0)])
async def test_a_rating_off_the_scale_is_clamped(rating: int, expected: float) -> None:
    assert await _rate(_answer(rating)) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text", ["no number here", "8", "", _answer("high"), '{"reason": "x"}']
)
async def test_a_rating_the_fixer_cannot_repair_is_unrated(text: str) -> None:
    assert await _rate(text) is None


async def test_an_unparsed_rating_gets_one_repair() -> None:
    assert await _rate("I'd rate it 4 of 5", fixed=_answer(4)) == pytest.approx(0.8)


async def test_a_non_finite_rating_is_unrated() -> None:
    # json reads NaN, which no threshold drops.
    assert await _rate('{"reason": "x", "rating": NaN}') is None
