# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for the MS GraphRAG global-search map-reduce.

Covers the MAP -> FILTER+RANK -> PACK -> REDUCE pipeline added to
``GlobalSearchStrategy``: the map step produces scored key points, the filter
drops score<=threshold, ranking orders by score descending, the token budget
caps the packed points, the reduce step receives the ranked points (not the raw
report concatenation), and a parse failure degrades to the legacy
concat-and-reduce path. Style mirrors ``test_global_search_logic.py`` /
``test_global_search_scoring.py``: the strategy is built via ``__new__`` so its
Bedrock ``__init__`` never runs, and chains are replaced with fakes / mocks.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from unified_kg_rag.adapters.search_strategies.global_search import (
    GlobalSearchStrategy,
    _MapPoint,
)
from unified_kg_rag.domain.models import RetrievalResult, SearchQuery

pytestmark = pytest.mark.unit


def _community(i: int, *, content: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        content=content or f"community report {i}",
        score=0.5,
        source=f"c{i}",
        retriever_type="graph",
        metadata={"community_id": f"cid-{i}"},
    )


def _communities(n: int) -> list[RetrievalResult]:
    return [_community(i) for i in range(n)]


class _SyncBatchChain:
    """Stub map chain exposing .batch / .invoke (used by BatchProcessor)."""

    def __init__(self, outputs: list[str]) -> None:
        # outputs are returned in call order, one per input dict.
        self._outputs = list(outputs)
        self.batch_inputs: list[dict] = []

    def batch(self, inputs: list[dict], config: Any = None) -> list[str]:
        self.batch_inputs.extend(inputs)
        out = self._outputs[: len(inputs)]
        self._outputs = self._outputs[len(inputs) :]
        return out

    def invoke(self, single_input: dict) -> str:
        return self.batch([single_input])[0]


def _reducer(return_value: str) -> Any:
    """A reduce chain stub whose ``.ainvoke`` is an awaitable mock."""
    chain = SimpleNamespace()
    chain.ainvoke = AsyncMock(return_value=return_value)
    return chain


class _TokenCounter:
    """Token counter that returns a fixed cost per call (1 token/char by len)."""

    def __init__(self, cost: int = 10) -> None:
        self._cost = cost

    def count_tokens(self, text: str) -> int:
        return self._cost


def _map_payload(*scored: tuple[str, int]) -> str:
    return json.dumps({"points": [{"description": d, "score": s} for d, s in scored]})


def _strategy(
    *,
    map_batch_size: int = 2,
    map_relevance_threshold: int = 0,
    max_map_reduce_tokens: int = 8000,
    map_reduce_min_results: int = 1,
    ignore_errors: bool = True,
    map_outputs: list[str] | None = None,
    reducer: Any = None,
    token_cost: int = 10,
) -> GlobalSearchStrategy:
    strat = GlobalSearchStrategy.__new__(GlobalSearchStrategy)
    strat.global_search_config = SimpleNamespace(
        map_batch_size=map_batch_size,
        map_relevance_threshold=map_relevance_threshold,
        max_map_reduce_tokens=max_map_reduce_tokens,
        map_reduce_min_results=map_reduce_min_results,
        enable_map_reduce=True,
    )
    strat.ignore_errors = ignore_errors
    strat.target_language = "English"
    strat.map_rater = _SyncBatchChain(map_outputs or [])
    strat.map_reducer = (
        reducer if reducer is not None else _reducer("SYNTHESIZED ANSWER")
    )
    strat.token_manager = _TokenCounter(cost=token_cost)
    # Real BatchProcessor with batch_size=1 (one prepared input per LLM call).
    from unified_kg_rag.shared.utils import BatchProcessor

    strat.batch_processor = BatchProcessor(batch_size=1, max_concurrency=4)
    return strat


# --------------------------------------------------------------------------- #
# MAP phase — produces scored points (mocked chain)
# --------------------------------------------------------------------------- #


async def test_map_phase_produces_scored_points() -> None:
    strat = _strategy(
        map_batch_size=1,
        map_outputs=[
            _map_payload(("point A", 90)),
            _map_payload(("point B", 30), ("point C", 0)),
        ],
    )
    points = await strat._run_map_phase(_communities(2), SearchQuery(query="q"))
    descriptions = {p.description: p.score for p in points}
    assert descriptions == {"point A": 90, "point B": 30, "point C": 0}


async def test_map_phase_batches_reports_per_call() -> None:
    # 3 reports, batch size 2 -> 2 map calls (batches of 2 and 1).
    strat = _strategy(
        map_batch_size=2,
        map_outputs=[_map_payload(("p1", 50)), _map_payload(("p2", 60))],
    )
    await strat._run_map_phase(_communities(3), SearchQuery(query="q"))
    # Two prepared inputs were sent (one per batch); each carries query/lang.
    assert len(strat.map_rater.batch_inputs) == 2
    first = strat.map_rater.batch_inputs[0]
    assert first["query"] == "q"
    assert first["target_language"] == "English"
    # The first batch packs two reports into one "reports" string.
    assert "cid-0" in first["reports"] and "cid-1" in first["reports"]


# --------------------------------------------------------------------------- #
# FILTER + RANK
# --------------------------------------------------------------------------- #


def test_filter_drops_at_or_below_threshold() -> None:
    strat = _strategy(map_relevance_threshold=0)
    points = [_MapPoint("zero", 0), _MapPoint("pos", 5)]
    kept = strat._filter_and_rank_points(points)
    assert [p.description for p in kept] == ["pos"]  # score 0 dropped (<= 0)


def test_filter_respects_configurable_threshold() -> None:
    strat = _strategy(map_relevance_threshold=50)
    points = [_MapPoint("low", 40), _MapPoint("edge", 50), _MapPoint("high", 80)]
    kept = strat._filter_and_rank_points(points)
    # 40 and 50 dropped (<= 50); only 80 kept.
    assert [p.description for p in kept] == ["high"]


def test_rank_orders_by_score_descending() -> None:
    strat = _strategy(map_relevance_threshold=0)
    points = [_MapPoint("mid", 50), _MapPoint("hi", 90), _MapPoint("lo", 10)]
    kept = strat._filter_and_rank_points(points)
    assert [p.score for p in kept] == [90, 50, 10]


# --------------------------------------------------------------------------- #
# PACK — token budget caps points
# --------------------------------------------------------------------------- #


def test_pack_caps_points_at_token_budget() -> None:
    # Each point costs 10 tokens; budget 25 -> first two fit (20), third (30) drops.
    strat = _strategy(max_map_reduce_tokens=25, token_cost=10)
    points = [_MapPoint(f"p{i}", 100 - i) for i in range(5)]
    packed = strat._pack_points_within_budget(points)
    assert [p.description for p in packed] == ["p0", "p1"]


def test_pack_always_keeps_at_least_one_point() -> None:
    # A single point exceeding the budget is still kept (avoid empty reduce).
    strat = _strategy(max_map_reduce_tokens=5, token_cost=10)
    points = [_MapPoint("big", 100)]
    packed = strat._pack_points_within_budget(points)
    assert [p.description for p in packed] == ["big"]


# --------------------------------------------------------------------------- #
# REDUCE — receives ranked points, not the raw concatenation
# --------------------------------------------------------------------------- #


async def test_reduce_receives_ranked_points() -> None:
    reducer = _reducer("FINAL")
    strat = _strategy(reducer=reducer)
    points = [_MapPoint("alpha", 90), _MapPoint("beta", 40)]
    results = _communities(3)
    out = await strat._reduce_from_points(points, results, SearchQuery(query="q"))

    assert out[0].source == "synthesized_summary"
    assert out[0].content == "FINAL"
    assert out[0].metadata["ranked_key_points"] == 2
    assert [r.source for r in out[1:]] == ["c0", "c1", "c2"]

    sent = reducer.ainvoke.await_args.args[0]
    # The reduce input is the ranked key points (with relevance), NOT the raw
    # report bodies.
    assert "alpha" in sent["summaries"] and "beta" in sent["summaries"]
    assert "relevance 90" in sent["summaries"]
    assert "community report 0" not in sent["summaries"]


# --------------------------------------------------------------------------- #
# End-to-end _apply_map_reduce
# --------------------------------------------------------------------------- #


async def test_apply_map_reduce_below_min_returns_unchanged() -> None:
    strat = _strategy(map_reduce_min_results=5)
    results = _communities(3)
    out = await strat._apply_map_reduce(results, SearchQuery(query="q"))
    assert out is results


async def test_apply_map_reduce_full_pipeline_prepends_synthesis() -> None:
    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=2,
        map_outputs=[
            _map_payload(("strong point", 95)),
            _map_payload(("weak point", 0)),  # dropped by filter
        ],
        reducer=_reducer("THE ANSWER"),
    )
    results = _communities(2)
    out = await strat._apply_map_reduce(results, SearchQuery(query="q"))
    assert out[0].content == "THE ANSWER"
    assert out[0].metadata["ranked_key_points"] == 1  # only the 95 survived
    assert [r.source for r in out[1:]] == ["c0", "c1"]


# --------------------------------------------------------------------------- #
# Parse failure / empty map -> degrade to concat-and-reduce
# --------------------------------------------------------------------------- #


def test_parse_map_points_handles_garbage() -> None:
    assert GlobalSearchStrategy._parse_map_points("not json at all") == []
    assert GlobalSearchStrategy._parse_map_points("") == []


def test_parse_map_points_strips_code_fence() -> None:
    raw = "```json\n" + _map_payload(("fenced", 70)) + "\n```"
    points = GlobalSearchStrategy._parse_map_points(raw)
    assert [(p.description, p.score) for p in points] == [("fenced", 70)]


def test_parse_map_points_clamps_scores() -> None:
    raw = _map_payload(("over", 250), ("under", -5))
    points = GlobalSearchStrategy._parse_map_points(raw)
    assert {p.description: p.score for p in points} == {"over": 100, "under": 0}


async def test_apply_map_reduce_parse_failure_degrades_to_concat() -> None:
    reducer = _reducer("CONCAT SUMMARY")
    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=2,
        map_outputs=["garbage", "still garbage"],  # all map calls unparseable
        reducer=reducer,
    )
    results = _communities(2)
    out = await strat._apply_map_reduce(results, SearchQuery(query="q"))
    assert out[0].content == "CONCAT SUMMARY"
    # Concat path metadata has no ranked_key_points key.
    assert "ranked_key_points" not in out[0].metadata
    # Reduce was fed the raw report bodies (concat path).
    sent = reducer.ainvoke.await_args.args[0]
    assert "community report 0" in sent["summaries"]


async def test_apply_map_reduce_all_filtered_degrades_to_concat() -> None:
    reducer = _reducer("CONCAT SUMMARY")
    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=2,
        map_relevance_threshold=50,
        map_outputs=[_map_payload(("low1", 10)), _map_payload(("low2", 20))],
        reducer=reducer,
    )
    out = await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))
    assert out[0].content == "CONCAT SUMMARY"
    assert "ranked_key_points" not in out[0].metadata
