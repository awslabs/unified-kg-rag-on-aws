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
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from botocore.exceptions import ClientError

from unified_kg_rag.adapters.search_strategies.global_search import (
    GlobalSearchStrategy,
    _MapPoint,
)
from unified_kg_rag.domain.models import Config, RetrievalResult, SearchQuery
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED

pytestmark = pytest.mark.unit


def _community(i: int, *, content: str | None = None) -> RetrievalResult:
    return RetrievalResult(
        content=content or f"community report {i}",
        score=0.5,
        source=f"c{i}",
        retriever_type="community",
        metadata={"community_id": f"cid-{i}"},
    )


def _communities(n: int) -> list[RetrievalResult]:
    return [_community(i) for i in range(n)]


class _SyncMapChain:
    """Stub map chain exposing .invoke (called per item by BatchProcessor)."""

    def __init__(self, outputs: list[str]) -> None:
        # outputs are returned in call order, one per input dict.
        self._outputs = list(outputs)
        self.batch_inputs: list[dict] = []

    def invoke(self, single_input: dict, config: Any = None) -> str:
        self.batch_inputs.append(single_input)
        return self._outputs.pop(0) if self._outputs else ""


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

    def count_tokens_many(self, texts: list[str]) -> list[int]:
        return [self.count_tokens(t) for t in texts]


def _batches(
    strat: GlobalSearchStrategy, reports: list[RetrievalResult]
) -> list[list[RetrievalResult]]:
    return strat._map_batches(reports)[0]


def _map_payload(*scored: tuple[str, int]) -> str:
    return json.dumps({"points": [{"description": d, "score": s} for d, s in scored]})


def _no_rerank_scorer() -> Any:
    """A real scorer with reranking off (asearch asks it for a rerank trace)."""
    from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer

    config = Config()
    config.search.reranking.enabled = False
    return HybridScorer(config)


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
    reduce_with_llm: bool = False,
) -> GlobalSearchStrategy:
    strat = GlobalSearchStrategy.__new__(GlobalSearchStrategy)
    strat.global_search_config = SimpleNamespace(
        # Every report costs token_cost tokens, so this packs map_batch_size
        # reports into one map call.
        map_context_tokens=map_batch_size * token_cost,
        map_relevance_threshold=map_relevance_threshold,
        max_map_reduce_tokens=max_map_reduce_tokens,
        map_reduce_min_results=map_reduce_min_results,
        enable_map_reduce=True,
        reduce_with_llm=reduce_with_llm,
        use_dynamic_selection=False,
        community_depth=2,
        text_unit_slots=0,
    )
    strat.retrievers = {}
    strat.ignore_errors = ignore_errors
    strat.target_language = "English"
    strat.hybrid_scorer = _no_rerank_scorer()
    strat.map_rater = _SyncMapChain(map_outputs or [])
    strat.map_reducer = (
        reducer if reducer is not None else _reducer("SYNTHESIZED ANSWER")
    )
    strat.token_manager = _TokenCounter(cost=token_cost)
    # The stub assigns outputs in call order, so run the map calls serially.
    # Concurrency and result ordering have dedicated BatchProcessor tests.
    from unified_kg_rag.shared.utils.langchain import BatchProcessor

    strat.batch_processor = BatchProcessor(max_concurrency=1)
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
    points, unrated = await strat._run_map_phase(
        _batches(strat, _communities(2)), SearchQuery(query="q")
    )
    descriptions = {p.description: p.score for p in points}
    assert descriptions == {"point A": 90, "point B": 30, "point C": 0}
    assert unrated == []


async def test_map_phase_reports_unrated_batches() -> None:
    # Batch 0 is rated (even with no points); batch 1 is unparseable; batch 2
    # failed (marked BATCH_ITEM_FAILED). Only 1 and 2 are unrated.
    strat = _strategy(
        map_batch_size=1,
        map_outputs=[json.dumps({"points": []}), "not json", BATCH_ITEM_FAILED],  # type: ignore[list-item]
    )
    communities = _communities(3)
    points, unrated = await strat._run_map_phase(
        _batches(strat, communities), SearchQuery(query="q")
    )
    assert points == []
    assert [r.source for r in unrated] == ["c1", "c2"]


async def test_map_phase_batches_reports_per_call() -> None:
    # 3 reports, batch size 2 -> 2 map calls (batches of 2 and 1).
    strat = _strategy(
        map_batch_size=2,
        map_outputs=[_map_payload(("p1", 50)), _map_payload(("p2", 60))],
    )
    await strat._run_map_phase(_batches(strat, _communities(3)), SearchQuery(query="q"))
    # Two prepared inputs were sent (one per batch); each carries query/lang.
    assert len(strat.map_rater.batch_inputs) == 2
    first, second = strat.map_rater.batch_inputs
    for batch in (first, second):
        assert batch["query"] == "q"
        assert batch["target_language"] == "English"
    # The first batch packs two reports into one "reports" string.
    assert "cid-0" in first["reports"] and "cid-1" in first["reports"]
    assert "cid-2" not in first["reports"]
    # The remaining report gets its own call.
    assert "cid-2" in second["reports"]
    assert "cid-0" not in second["reports"] and "cid-1" not in second["reports"]


def test_map_batches_pack_reports_by_tokens() -> None:
    # MS GraphRAG packs reports by tokens (data_max_tokens), not by count; a
    # report over the budget gets a call of its own.
    strat = _strategy(map_batch_size=3)  # a 30-token budget
    strat.token_manager = SimpleNamespace(  # type: ignore[assignment]
        count_tokens_many=lambda texts: [40 if "big" in t else 10 for t in texts]
    )
    reports = [_community(0), _community(1, content="big"), *_communities(4)[2:]]
    batches, tokens = strat._map_batches(reports)
    assert [[r.source for r in batch] for batch in batches] == [
        ["c0"],
        ["c1"],
        ["c2", "c3"],
    ]
    assert tokens == 70


async def test_map_step_rates_only_reports() -> None:
    # Text units are not community reports: mapping them sent chunks to the
    # map LLM as "reports". The evidence after the points is the reports.
    strat = _strategy(map_batch_size=10, map_outputs=[_map_payload(("point", 80))])
    chunk = RetrievalResult(
        content="chunk text", score=0.4, source="t0", retriever_type="text"
    )
    results = [*_communities(2), chunk]

    out = await strat._apply_map_reduce(results, SearchQuery(query="q"))

    (map_input,) = strat.map_rater.batch_inputs
    assert "chunk text" not in map_input["reports"]
    assert [r.source for r in out] == ["synthesized_key_points", "c0", "c1"]


async def test_map_step_counts_only_reports_against_the_minimum() -> None:
    strat = _strategy(map_reduce_min_results=2, map_outputs=[_map_payload(("p", 80))])
    chunks = [
        RetrievalResult(
            content=f"t{i}", score=0.4, source=f"t{i}", retriever_type="text"
        )
        for i in range(3)
    ]
    results = [_community(0), *chunks]
    assert await strat._apply_map_reduce(results, SearchQuery(query="q")) == results
    assert strat.map_rater.batch_inputs == []


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
    strat = _strategy(reducer=reducer, reduce_with_llm=True)
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
        reduce_with_llm=True,
    )
    results = _communities(2)
    out = await strat._apply_map_reduce(results, SearchQuery(query="q"))
    assert out[0].content == "THE ANSWER"
    assert out[0].metadata["ranked_key_points"] == 1  # only the 95 survived
    # LLM synthesis, not evidence: flagged so it is never reported as a source.
    assert out[0].metadata["synthesized"] is True
    # Only the batch behind a packed point is evidence: c1 gave none.
    assert [r.source for r in out[1:]] == ["c0"]


# --------------------------------------------------------------------------- #
# Parse failure / empty map -> degrade to concat-and-reduce
# --------------------------------------------------------------------------- #


def test_parse_map_payload_handles_garbage() -> None:
    assert GlobalSearchStrategy._parse_map_payload("not json at all") is None
    assert GlobalSearchStrategy._parse_map_payload("") is None


def test_parse_map_payload_strips_code_fence() -> None:
    raw = "```json\n" + _map_payload(("fenced", 70)) + "\n```"
    points = GlobalSearchStrategy._parse_map_payload(raw) or []
    assert [(p.description, p.score) for p in points] == [("fenced", 70)]


def test_parse_map_payload_clamps_scores() -> None:
    raw = _map_payload(("over", 250), ("under", -5))
    points = GlobalSearchStrategy._parse_map_payload(raw) or []
    assert {p.description: p.score for p in points} == {"over": 100, "under": 0}


async def test_apply_map_reduce_parse_failure_degrades_to_concat() -> None:
    reducer = _reducer("CONCAT SUMMARY")
    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=2,
        map_outputs=["garbage", "still garbage"],  # all map calls unparseable
        reducer=reducer,
        reduce_with_llm=True,
    )
    results = _communities(2)
    out = await strat._apply_map_reduce(results, SearchQuery(query="q"))
    assert out[0].content == "CONCAT SUMMARY"
    assert out[0].metadata["synthesized"] is True
    # Concat path metadata has no ranked_key_points key.
    assert "ranked_key_points" not in out[0].metadata
    # Reduce was fed the raw report bodies (concat path).
    sent = reducer.ainvoke.await_args.args[0]
    assert "community report 0" in sent["summaries"]


async def test_apply_map_reduce_all_filtered_returns_no_context() -> None:
    # Every map point scored at/below the threshold: the map step judged the
    # reports irrelevant to the query. Synthesizing from them anyway (the old
    # concat-and-reduce degradation) produced an answer with no supporting
    # evidence. MS GraphRAG returns its no-data answer here, so the strategy
    # returns no context and the chain's empty-context guard answers instead.
    reducer = _reducer("CONCAT SUMMARY")
    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=2,
        map_relevance_threshold=50,
        map_outputs=[_map_payload(("low1", 10)), _map_payload(("low2", 50))],
        reducer=reducer,
        reduce_with_llm=True,
    )
    out = await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))
    assert out == []
    reducer.ainvoke.assert_not_awaited()


async def test_apply_map_reduce_unrated_batch_degrades_to_concat_over_unrated() -> None:
    # One batch rated every point irrelevant, the other failed (throttled): the
    # failed batch's reports were never judged, so "no relevant data" would be
    # an unfounded verdict. Synthesize from the unrated reports instead.
    reducer = _reducer("CONCAT SUMMARY")
    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=2,
        map_relevance_threshold=50,
        map_outputs=[_map_payload(("low", 0)), BATCH_ITEM_FAILED],  # type: ignore[list-item]
        reducer=reducer,
        reduce_with_llm=True,
    )
    out = await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))
    assert out[0].content == "CONCAT SUMMARY"
    assert out[0].metadata["synthesized"] is True
    assert [r.source for r in out[1:]] == ["c1"]
    sent = reducer.ainvoke.await_args.args[0]
    assert "community report 1" in sent["summaries"]
    assert "community report 0" not in sent["summaries"]


async def test_apply_map_reduce_map_exception_concats_all_reports() -> None:
    reducer = _reducer("CONCAT SUMMARY")
    strat = _strategy(map_reduce_min_results=2, reducer=reducer, reduce_with_llm=True)
    strat._run_map_phase = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("map down")
    )
    out = await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))
    assert out[0].content == "CONCAT SUMMARY"
    assert [r.source for r in out[1:]] == ["c0", "c1"]


async def test_asearch_all_filtered_yields_empty_result_with_flag() -> None:
    # End to end through asearch: an all-below-threshold map phase surfaces as an
    # empty SearchResult (no reports leak through as answer context) and is
    # flagged in metadata so callers can tell it apart from a retrieval miss.
    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=1,
        map_relevance_threshold=50,
        map_outputs=[_map_payload(("low", 5)), _map_payload(("low", 5))],
    )
    communities = _communities(2)
    strat._cut_reports = AsyncMock(  # type: ignore[method-assign]
        return_value=communities
    )

    result = await strat.asearch(SearchQuery(query="q"))

    assert result.results == []
    assert result.total_results == 0
    assert result.metadata["map_reduce_no_relevant_points"] is True
    assert result.metadata["map_reduce_applied"] is False


@pytest.mark.parametrize("failing", ["map", "reduce"])
async def test_asearch_marks_a_map_reduce_failure_ignore_errors_degraded(
    failing: str,
) -> None:
    # Regression: with ignore_errors on, a failed map or reduce call degraded
    # the answer with no marker on the result (nor error.type on the span).
    strat = _strategy(
        ignore_errors=True,
        map_batch_size=1,
        reduce_with_llm=True,
        map_outputs=[_map_payload(("p", 50))] * 2,
    )
    if failing == "map":
        strat._run_map_phase = AsyncMock(  # type: ignore[method-assign]
            side_effect=TimeoutError("map down")
        )
    else:
        strat.map_reducer.ainvoke = AsyncMock(side_effect=TimeoutError("down"))
    strat._cut_reports = AsyncMock(  # type: ignore[method-assign]
        return_value=_communities(2)
    )

    result = await strat.asearch(SearchQuery(query="q"))

    assert result.results
    assert result.metadata["degraded_error"] == "TimeoutError"


async def test_asearch_without_a_failure_has_no_degraded_marker() -> None:
    strat = _strategy(map_batch_size=1, map_outputs=[_map_payload(("p", 50))] * 2)
    strat._cut_reports = AsyncMock(  # type: ignore[method-assign]
        return_value=_communities(2)
    )

    result = await strat.asearch(SearchQuery(query="q"))

    assert "degraded_error" not in result.metadata


# --------------------------------------------------------------------------- #
# Default (reduce_with_llm=False): packed points go to the answer model as-is
# --------------------------------------------------------------------------- #


async def test_default_passes_ranked_points_without_a_reduce_call() -> None:
    reducer = _reducer("SHOULD NOT RUN")
    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=2,
        map_outputs=[
            _map_payload(("weaker point", 40)),
            _map_payload(("strong point", 95), ("noise", 0)),
        ],
        reducer=reducer,
    )
    results = _communities(2)

    out = await strat._apply_map_reduce(results, SearchQuery(query="q"))

    reducer.ainvoke.assert_not_awaited()
    points = out[0]
    assert points.source == "synthesized_key_points"
    assert points.metadata["synthesized"] is True  # never reported as a source
    assert points.metadata["ranked_key_points"] == 2
    assert points.content.index("strong point") < points.content.index("weaker")
    assert "relevance 95" in points.content
    assert "noise" not in points.content
    # The evidence follows its best point (c1's 95 before c0's 40), scored 0-1.
    assert [(r.source, r.score) for r in out[1:]] == [("c1", 0.95), ("c0", 0.4)]
    assert strat._was_map_reduce_applied(out) is True


async def test_a_point_credits_only_the_reports_it_cites() -> None:
    # Regression: a point drawn from one report of a batch credited every
    # report of the batch, so a dozen unrelated reports became context and
    # sources.
    cited = json.dumps(
        {
            "points": [
                {"description": "from c1", "score": 80, "report_ids": ["cid-1"]},
                {"description": "from c3", "score": 60, "report_ids": [" cid-3 "]},
            ]
        }
    )
    strat = _strategy(map_batch_size=4, map_outputs=[cited])
    out = await strat._apply_map_reduce(_communities(4), SearchQuery(query="q"))
    assert [(r.source, r.score) for r in out[1:]] == [("c1", 0.8), ("c3", 0.6)]


@pytest.mark.parametrize("ids", [None, [], ["cid-9"], "cid-1"])
async def test_a_point_without_valid_ids_credits_its_batch(ids: Any) -> None:
    point: dict[str, Any] = {"description": "somewhere", "score": 70}
    if ids is not None:
        point["report_ids"] = ids
    strat = _strategy(
        map_batch_size=2, map_outputs=[json.dumps({"points": [point]}), ""]
    )
    out = await strat._apply_map_reduce(_communities(3), SearchQuery(query="q"))
    assert [r.source for r in out[1:]] == ["c0", "c1"]


def test_the_map_prompt_asks_for_the_report_ids() -> None:
    from unified_kg_rag.domain.prompts import GlobalMapPrompt

    assert '"report_ids"' in GlobalMapPrompt.system_prompt_template
    formatted = GlobalSearchStrategy._format_reports(_communities(1))
    assert formatted.startswith("--- Report (id: cid-0) ---")


async def test_default_degraded_path_passes_unrated_reports_through() -> None:
    reducer = _reducer("SHOULD NOT RUN")
    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=2,
        map_outputs=["garbage", "still garbage"],
        reducer=reducer,
    )
    results = _communities(2)

    out = await strat._apply_map_reduce(results, SearchQuery(query="q"))

    reducer.ainvoke.assert_not_awaited()
    assert [r.source for r in out] == ["c0", "c1"]


async def test_pack_does_not_count_tokens_on_the_event_loop() -> None:
    import threading

    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=2,
        map_outputs=[_map_payload(("a", 90)), _map_payload(("b", 80))],
    )
    loop_thread = threading.get_ident()
    seen: list[int] = []
    counter = strat.token_manager

    def count_tokens_many(texts: list[str]) -> list[int]:
        seen.append(threading.get_ident())
        return [counter.count_tokens(t) for t in texts]

    counter.count_tokens_many = count_tokens_many  # type: ignore[attr-defined]
    out = await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))
    assert out[0].metadata["ranked_key_points"] == 2
    assert seen and loop_thread not in seen


class _DeniedMapChain:
    """Map chain whose every call fails like a model without access."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def invoke(self, single_input: dict, config: Any = None) -> str:
        raise self.error


async def test_map_phase_fatal_error_propagates_when_not_ignored() -> None:
    strat = _strategy(ignore_errors=False, map_batch_size=1)
    strat.batch_processor.max_attempts = 1  # as GlobalSearchStrategy configures
    strat.map_rater = _DeniedMapChain(
        ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "no access"}},
            "InvokeModel",
        )
    )
    with pytest.raises(ClientError, match="AccessDenied"):
        await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))


class _OverflowingFirstReport(_SyncMapChain):
    """Map chain whose prompt with report 0 is over the model's context."""

    def invoke(self, single_input: dict, config: Any = None) -> str:
        if "community report 0" in single_input["reports"]:
            from langchain_aws.chat_models.bedrock_converse import (
                BedrockContextOverflowError,
            )

            raise BedrockContextOverflowError(
                {
                    "Error": {
                        "Code": "ValidationException",
                        "Message": "Input is too long for requested model.",
                    }
                },
                "Converse",
            )
        return super().invoke(single_input, config)


async def test_map_phase_one_overflowing_batch_is_left_unrated() -> None:
    # Regression: Bedrock raises a too-long prompt as a ValidationException,
    # which failed the whole query; the other batches still rate.
    strat = _strategy(ignore_errors=False, map_batch_size=1)
    strat.batch_processor.max_attempts = 1  # as GlobalSearchStrategy configures
    strat.map_rater = _OverflowingFirstReport(
        [_map_payload((f"point {i}", 50)) for i in (1, 2, 3)]
    )
    points, unrated = await strat._run_map_phase(
        _batches(strat, _communities(4)), SearchQuery(query="q")
    )
    assert [r.source for r in unrated] == ["c0"]
    assert len(points) == 3


def _bad_request(message: str) -> Exception:
    """An OpenAI-compatible server's HTTP 400, as vLLM and llama.cpp send it."""
    import httpx
    import openai

    return openai.BadRequestError(
        message,
        response=httpx.Response(
            400, request=httpx.Request("POST", "http://localhost/v1")
        ),
        body={"object": "error", "type": "BadRequestError", "code": 400},
    )


class _VllmOverflowingFirstReport(_SyncMapChain):
    """Map chain failing report 0 the way vLLM reports a too-long prompt."""

    def invoke(self, single_input: dict, config: Any = None) -> str:
        if "community report 0" in single_input["reports"]:
            raise _bad_request("This model's maximum context length is 8192 tokens.")
        return super().invoke(single_input, config)


async def test_map_phase_a_vllm_overflow_leaves_the_batch_unrated() -> None:
    # Regression: vLLM and llama.cpp return a plain 400 for a prompt over the
    # context, which LangChain does not map to ContextOverflowError, so one
    # long batch failed the whole query.
    strat = _strategy(ignore_errors=False, map_batch_size=1)
    strat.batch_processor.max_attempts = 1  # as GlobalSearchStrategy configures
    strat.map_rater = _VllmOverflowingFirstReport(
        [_map_payload((f"point {i}", 50)) for i in (1, 2, 3)]
    )
    points, unrated = await strat._run_map_phase(
        _batches(strat, _communities(4)), SearchQuery(query="q")
    )
    assert [r.source for r in unrated] == ["c0"]
    assert len(points) == 3


async def test_map_phase_a_400_on_every_batch_fails_the_query() -> None:
    # Regression: every 400 left its batch unrated, so a map model refusing
    # every request (an unsupported parameter, an unknown model id) answered
    # from the unrated reports; with models.fast the answer
    # call runs on another model and did not fail either.
    import openai

    strat = _strategy(ignore_errors=False, map_batch_size=1, reduce_with_llm=True)
    strat.batch_processor.max_attempts = 1  # as GlobalSearchStrategy configures
    strat.map_rater = _DeniedMapChain(_bad_request("Unknown parameter: 'top_k'"))
    with pytest.raises(openai.BadRequestError, match="top_k"):
        await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))


async def test_map_phase_a_typed_overflow_on_every_batch_answers() -> None:
    # Regression: langchain-openai's OpenAIContextOverflowError is a
    # BadRequestError too, so an overflow on every batch (a level of at most
    # map_context_tokens of reports is one batch) failed the query as a refused
    # request instead of answering from the unrated reports.
    base = pytest.importorskip("langchain_openai.chat_models.base")
    import httpx

    overflow = base.OpenAIContextOverflowError(
        "too long",
        response=httpx.Response(
            400, request=httpx.Request("POST", "http://localhost/v1")
        ),
        body=None,
    )
    strat = _strategy(ignore_errors=False, map_batch_size=5, reduce_with_llm=True)
    strat.batch_processor.max_attempts = 1  # as GlobalSearchStrategy configures
    strat.map_rater = _DeniedMapChain(overflow)
    out = await strat._apply_map_reduce(_communities(3), SearchQuery(query="q"))
    assert out[0].content == "SYNTHESIZED ANSWER"
    assert [r.source for r in out[1:]] == ["c0", "c1", "c2"]


async def test_map_phase_fatal_error_degrades_when_ignored() -> None:
    strat = _strategy(ignore_errors=True, map_batch_size=1, reduce_with_llm=True)
    strat.batch_processor.max_attempts = 1  # as GlobalSearchStrategy configures
    strat.map_rater = _DeniedMapChain(
        ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "no access"}},
            "InvokeModel",
        )
    )
    out = await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))
    assert out  # degraded to concat-and-reduce over the unrated reports


async def test_map_phase_transient_failures_still_degrade_when_not_ignored() -> None:
    strat = _strategy(ignore_errors=False, map_batch_size=1, reduce_with_llm=True)
    strat.batch_processor.max_attempts = 1  # as GlobalSearchStrategy configures
    strat.map_rater = _DeniedMapChain(TimeoutError("read timed out"))
    out = await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))
    assert out


class _SlowMapChain:
    """Map chain whose calls wait for ``target`` calls in flight (or a timeout).

    Records the peak in-flight count. Waiting on the other calls instead of
    sleeping a fixed time makes the peak independent of runner speed: with
    enough fan-out every call meets ``target`` peers; with less, each call
    gives up after ``timeout`` and the peak stays below ``target``.
    """

    def __init__(self, target: int, timeout: float = 2.0) -> None:
        self.active = 0
        self.max_active = 0
        self._target = target
        self._timeout = timeout
        self._cond = threading.Condition()

    def batch(
        self, inputs: list[dict], config: Any = None, return_exceptions: bool = False
    ) -> list[str]:
        with self._cond:
            self.active += len(inputs)
            self.max_active = max(self.max_active, self.active)
            self._cond.notify_all()
            self._cond.wait_for(
                lambda: self.max_active >= self._target, timeout=self._timeout
            )
            self.active -= len(inputs)
        return [_map_payload(("p", 50))] * len(inputs)

    def invoke(self, single_input: dict, config: Any = None) -> str:
        return self.batch([single_input])[0]


async def test_map_phase_runs_max_concurrency_calls_at_once() -> None:
    # search.max_concurrency is the documented map fan-out; the ingestion
    # knob processing.max_concurrency is not.
    config = Config()
    config.search.max_concurrency = 20
    config.processing.max_concurrency = 3
    strat = GlobalSearchStrategy(config=config, retrievers={})
    strat.map_rater = _SlowMapChain(target=20)  # type: ignore[assignment]

    points, unrated = await strat._run_map_phase(
        [[report] for report in _communities(40)], SearchQuery(query="q")
    )

    assert len(points) == 40 and unrated == []
    assert strat.map_rater.max_active == 20  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Reduce LLM failure: ignore_errors decides between raising and degrading
# --------------------------------------------------------------------------- #


def _failing_reducer() -> Any:
    chain = SimpleNamespace()
    chain.ainvoke = AsyncMock(side_effect=RuntimeError("reduce model down"))
    return chain


@pytest.mark.parametrize("path", ["points", "concat"])
async def test_reduce_failure_raises_without_ignore_errors(path: str) -> None:
    strat = _strategy(
        ignore_errors=False, reducer=_failing_reducer(), reduce_with_llm=True
    )
    query = SearchQuery(query="q")
    with pytest.raises(RuntimeError, match="reduce model down"):
        if path == "points":
            await strat._reduce_from_points(
                [_MapPoint("point", 80)], _communities(2), query
            )
        else:
            await strat._concat_reduce(_communities(2), query)


@pytest.mark.parametrize("path", ["points", "concat"])
async def test_reduce_failure_degrades_with_ignore_errors(path: str) -> None:
    strat = _strategy(
        ignore_errors=True, reducer=_failing_reducer(), reduce_with_llm=True
    )
    query = SearchQuery(query="q")
    results = _communities(2)
    degraded: list[str] = []
    if path == "points":
        out = await strat._reduce_from_points(
            [_MapPoint("point", 80)], results, query, degraded=degraded
        )
    else:
        out = await strat._concat_reduce(results, query, degraded=degraded)
    # The reports pass through without a synthesis; the failure is recorded.
    assert out == results
    assert degraded == ["RuntimeError"]
