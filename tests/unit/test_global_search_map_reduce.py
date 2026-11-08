# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for the MS GraphRAG global-search map-reduce.

Covers the MAP -> FILTER+RANK -> PACK -> REDUCE pipeline added to
``GlobalSearchStrategy``: the map step produces scored key points, the filter
drops score<=threshold, ranking orders by score descending, the token budget
caps the packed points, the answer model receives the ranked points as one
section (not the raw report concatenation), and a parse failure passes the
unrated reports through. Style mirrors ``test_global_search_logic.py`` /
``test_global_search_scoring.py``: the strategy is built via ``__new__`` so its
Bedrock ``__init__`` never runs, and chains are replaced with fakes / mocks.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from botocore.exceptions import ClientError

from unified_kg_rag.adapters.retrieval.base import degradation_scope
from unified_kg_rag.adapters.search_strategies.global_search import (
    GlobalSearchStrategy,
    _MapPoint,
)
from unified_kg_rag.domain.models import Config, RetrievalResult, SearchQuery
from unified_kg_rag.domain.retrieval.degradation import DegradationLedger

pytestmark = pytest.mark.unit


def _types(ledger: DegradationLedger) -> list[tuple[str, str | None]]:
    """The ``(step, error_type)`` of each entry of ``ledger``."""
    return [(e.step, e.error_type) for e in ledger.entries]


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


def _points(out: list[RetrievalResult]) -> list[RetrievalResult]:
    """The key points of a map-reduce result: its leading synthesized items."""
    points = [r for r in out if r.metadata.get("synthesized")]
    assert out[: len(points)] == points
    return points


def _evidence(out: list[RetrievalResult]) -> list[RetrievalResult]:
    """The reports after the key points."""
    return out[len(_points(out)) :]


class _SyncMapChain:
    """Stub map chain: ``.ainvoke`` answers through ``.invoke``."""

    def __init__(self, outputs: list[str | Exception]) -> None:
        # outputs are returned (an exception raised) in call order, one per
        # input dict.
        self._outputs = list(outputs)
        self.batch_inputs: list[dict] = []

    def invoke(self, single_input: dict, config: Any = None) -> str:
        self.batch_inputs.append(single_input)
        output = self._outputs.pop(0) if self._outputs else ""
        if isinstance(output, Exception):
            raise output
        return output

    async def ainvoke(self, single_input: dict, config: Any = None) -> str:
        return self.invoke(single_input, config)


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
    config.providers.rerank.kind = "none"
    return HybridScorer(config)


class _ScriptedFixer:
    """A factory whose output fixer answers ``replies`` in turn (then text
    that is no JSON), counting its calls."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.calls = 0

    def get_model(self, **_kwargs: Any) -> Any:
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        def answer(_prompt: Any) -> AIMessage:
            self.calls += 1
            return AIMessage(
                content=self.replies.pop(0) if self.replies else "still no JSON"
            )

        return RunnableLambda(answer)


def _strategy(
    *,
    map_batch_size: int = 2,
    map_relevance_threshold: int = 0,
    max_map_reduce_tokens: int = 8000,
    map_reduce_min_results: int = 1,
    ignore_errors: bool = True,
    map_outputs: list[str | Exception] | None = None,
    token_cost: int = 10,
    evidence_tokens: int | None = None,
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
        use_dynamic_selection=False,
        community_depth=2,
        text_unit_slots=0,
        evidence_tokens=evidence_tokens,
    )
    strat.config = Config()
    strat.retrievers = {}
    strat.ignore_errors = ignore_errors
    strat.hybrid_scorer = _no_rerank_scorer()
    strat.map_rater = _SyncMapChain(map_outputs or [])
    strat.providers = SimpleNamespace(llm_factory=_ScriptedFixer())
    strat.token_manager = _TokenCounter(cost=token_cost)
    # The stub assigns outputs in call order, so run the map calls serially.
    strat.max_concurrency = 1
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
    # failed. Only 1 and 2 are unrated.
    strat = _strategy(
        map_batch_size=1,
        map_outputs=[json.dumps({"points": []}), "not json", TimeoutError("t")],
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
    assert "report 0" in first["reports"] and "report 1" in first["reports"]
    assert "report 2" not in first["reports"]
    # The remaining report gets its own call.
    assert "report 2" in second["reports"]
    assert "report 0" not in second["reports"] and "report 1" not in second["reports"]


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
    assert [r.source for r in out] == ["synthesized_key_points_1", "c0", "c1"]


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
# REDUCE — the answer model receives the ranked points, not the raw reports
# --------------------------------------------------------------------------- #


def test_reduce_hands_the_ranked_points_to_the_answer_model() -> None:
    strat = _strategy()
    points = [_MapPoint("alpha", 90), _MapPoint("beta", 40)]
    results = _communities(3)
    out = strat._reduce_from_points(points, results)

    assert [r.source for r in out[:2]] == [
        "synthesized_key_points_1",
        "synthesized_key_points_2",
    ]
    assert out[0].metadata["ranked_key_points"] == 2
    assert [r.source for r in out[2:]] == ["c0", "c1", "c2"]
    # Each section is a ranked key point (with relevance), NOT a raw report
    # body.
    assert "alpha" in out[0].content and "beta" in out[1].content
    assert "relevance 90" in out[0].content
    assert all("community report 0" not in r.content for r in out[:2])
    # Without its batch a point names no evidence: it is not citable.
    assert out[0].metadata["citable"] is False


def test_global_search_builds_no_reduce_chain() -> None:
    # The reduce LLM call is gone: the strategy builds the map chain only (it
    # built the reduce chain even with the reduce call off).
    from unittest.mock import MagicMock

    from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
    from unified_kg_rag.adapters.providers import Providers

    config = Config()
    config.providers.rerank.kind = "none"
    factory = ScriptedLLMFactory(lambda system, human: "")
    built: list[Any] = []
    get_model = factory.get_model
    factory.get_model = lambda model_id, **kw: built.append(model_id) or get_model(  # type: ignore[method-assign]
        model_id, **kw
    )
    providers = Providers(config, boto_session=MagicMock(), llm_factory=factory)
    strat = GlobalSearchStrategy(config=config, retrievers={}, providers=providers)
    assert not hasattr(strat, "map_reducer")
    assert len(built) == 1  # the map rater


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
    )
    results = _communities(2)
    out = await strat._apply_map_reduce(results, SearchQuery(query="q"))
    assert out[0].source == "synthesized_key_points_1"
    assert out[0].metadata["ranked_key_points"] == 1  # only the 95 survived
    # Map-LLM output, not evidence: flagged so it is never reported as a source.
    assert out[0].metadata["synthesized"] is True
    # Only the batch behind a packed point is evidence: c1 gave none.
    assert [r.source for r in out[1:]] == ["c0"]


# --------------------------------------------------------------------------- #
# Parse failure / empty map -> the unrated reports pass through
# --------------------------------------------------------------------------- #


def test_parse_map_payload_handles_garbage() -> None:
    assert GlobalSearchStrategy._parse_map_payload("not json at all") is None
    assert GlobalSearchStrategy._parse_map_payload("") is None


# Map outputs json.loads refuses, as a model writes them (rv6: about 1% of
# real map batches came back unparseable and left their reports unrated).
_DEFECTIVE_MAP_OUTPUTS = {
    "unescaped-quote": (
        '{"points": [{"description": "The album "Thriller" sold most", '
        '"score": 80, "report_ids": ["1", "3"]}, {"description": "공급사가 '
        '"부품"을 출하한다", "score": 40, "report_ids": ["2"]}]}',
        [
            ('The album "Thriller" sold most', 80, ["1", "3"]),
            ('공급사가 "부품"을 출하한다', 40, ["2"]),
        ],
    ),
    "line-break-in-string": (
        '{"points": [{"description": "倉庫が部品を\n保管する", "score": 70, '
        '"report_ids": ["1"]}]}',
        [("倉庫が部品を\n保管する", 70, ["1"])],
    ),
    "invalid-escape": (
        r'{"points": [{"description": "Vendor\'s depot", "score": 60, '
        '"report_ids": ["1"]}]}',
        [("Vendor's depot", 60, ["1"])],
    ),
    "trailing-comma": (
        '{"points": [{"description": "Depot", "score": 50, "report_ids": ["1"],},],}',
        [("Depot", 50, ["1"])],
    ),
    "bare-list": (
        'Here are the points:\n[{"description": "A", "score": 30, "report_ids": '
        '["1"]}, {"description": "B", "score": 20, "report_ids": ["2"]}]',
        [("A", 30, ["1"]), ("B", 20, ["2"])],
    ),
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    list(_DEFECTIVE_MAP_OUTPUTS.values()),
    ids=list(_DEFECTIVE_MAP_OUTPUTS),
)
def test_parse_map_payload_repairs_what_a_model_writes(
    raw: str, expected: list[tuple[str, int, list[str]]]
) -> None:
    points = GlobalSearchStrategy._parse_map_payload(raw)
    assert points is not None
    assert [(p.description, p.score, p.report_ids) for p in points] == expected


def test_an_object_without_points_is_unrated_not_rated_empty() -> None:
    # Regression (rv6): the last top-level object was taken for the answer,
    # so a bare list's last point (no "points") read as a rated batch with
    # no key points: its points were lost without a trace.
    assert GlobalSearchStrategy._parse_map_payload('{"answer": "none"}') is None
    assert GlobalSearchStrategy._parse_map_payload('{"points": []}') == []


async def test_an_unparseable_map_output_goes_to_the_output_fixer_once(
    caplog,
) -> None:
    # One repair of the output, not another map call over the whole batch;
    # a blank output has nothing to repair.
    strat = _strategy(
        map_batch_size=1,
        map_outputs=["Sorry, here they are: points: none", "", "still not json"],
    )
    fixer = _ScriptedFixer([_map_payload(("fixed", 70))])
    strat.providers = SimpleNamespace(llm_factory=fixer)
    with degradation_scope() as ledger:
        points, unrated = await strat._run_map_phase(
            _batches(strat, _communities(3)), SearchQuery(query="q")
        )
    assert [(p.description, p.batch) for p in points] == [("fixed", 0)]
    assert [r.source for r in unrated] == ["c1", "c2"]
    assert len(strat.map_rater.batch_inputs) == 3
    assert fixer.calls == 2
    assert _types(ledger) == [("global_map", "LanguageModelError")]
    # The log names the shape, never the text.
    messages = [r.getMessage() for r in caplog.records if "not map JSON" in r.message]
    assert len(messages) == 3
    assert "asking the output fixer" in messages[0] and "no JSON object" in messages[0]
    assert "empty" in messages[1] and "still not json" not in " ".join(messages)


async def test_a_parsed_map_output_is_not_asked_again() -> None:
    strat = _strategy(map_batch_size=1, map_outputs=[json.dumps({"points": []})])
    points, unrated = await strat._run_map_phase(
        _batches(strat, _communities(1)), SearchQuery(query="q")
    )
    assert (points, unrated) == ([], [])
    assert len(strat.map_rater.batch_inputs) == 1


def test_parse_map_payload_strips_code_fence() -> None:
    raw = "```json\n" + _map_payload(("fenced", 70)) + "\n```"
    points = GlobalSearchStrategy._parse_map_payload(raw) or []
    assert [(p.description, p.score) for p in points] == [("fenced", 70)]


def test_parse_map_payload_clamps_scores() -> None:
    raw = _map_payload(("over", 250), ("under", -5))
    points = GlobalSearchStrategy._parse_map_payload(raw) or []
    assert {p.description: p.score for p in points} == {"over": 100, "under": 0}


async def test_apply_map_reduce_all_filtered_returns_no_context() -> None:
    # Every map point scored at/below the threshold: the map step judged the
    # reports irrelevant to the query. Synthesizing from them anyway (the old
    # concat-and-reduce degradation) produced an answer with no supporting
    # evidence. MS GraphRAG returns its no-data answer here, so the strategy
    # returns no context and the chain's empty-context guard answers instead.
    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=2,
        map_relevance_threshold=50,
        map_outputs=[_map_payload(("low1", 10)), _map_payload(("low2", 50))],
    )
    out = await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))
    assert out == []


async def test_apply_map_reduce_unrated_batch_passes_the_unrated_through() -> None:
    # One batch rated every point irrelevant, the other failed (throttled): the
    # failed batch's reports were never judged, so "no relevant data" would be
    # an unfounded verdict. Answer from the unrated reports instead.
    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=2,
        map_relevance_threshold=50,
        map_outputs=[_map_payload(("low", 0)), TimeoutError("throttled")],
    )
    out = await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))
    assert [r.source for r in out] == ["c1"]


async def test_apply_map_reduce_map_exception_passes_all_reports_through() -> None:
    strat = _strategy(map_reduce_min_results=2)
    strat._run_map_phase = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("map down")
    )
    out = await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))
    assert [r.source for r in out] == ["c0", "c1"]


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


async def test_asearch_marks_a_map_failure_ignore_errors_degraded() -> None:
    # Regression: with ignore_errors on, a failed map call degraded the answer
    # with no marker on the result (nor error.type on the span).
    strat = _strategy(
        ignore_errors=True,
        map_batch_size=1,
        map_outputs=[_map_payload(("p", 50))] * 2,
    )
    strat._run_map_phase = AsyncMock(  # type: ignore[method-assign]
        side_effect=TimeoutError("map down")
    )
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
# The packed points go to the answer model as-is
# --------------------------------------------------------------------------- #


async def test_passes_each_ranked_point_as_its_own_section() -> None:
    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=2,
        map_outputs=[
            _map_payload(("weaker point", 40)),
            _map_payload(("strong point", 95), ("noise", 0)),
        ],
    )
    results = _communities(2)

    out = await strat._apply_map_reduce(results, SearchQuery(query="q"))

    points = _points(out)
    assert [p.source for p in points] == [
        "synthesized_key_points_1",
        "synthesized_key_points_2",
    ]
    for point in points:
        assert point.metadata["synthesized"] is True  # never reported as a source
        assert point.metadata["ranked_key_points"] == 2
    assert "strong point" in points[0].content and "relevance 95" in points[0].content
    assert "weaker point" in points[1].content
    assert [p.score for p in points] == [0.95, 0.4]
    assert all("noise" not in p.content for p in points)
    # The evidence follows its best point (c1's 95 before c0's 40), scored 0-1.
    assert [(r.source, r.score) for r in _evidence(out)] == [
        ("c1", 0.95),
        ("c0", 0.4),
    ]
    assert strat._was_map_reduce_applied(out) is True


async def test_a_point_credits_only_the_reports_it_cites() -> None:
    # Regression: a point drawn from one report of a batch credited every
    # report of the batch, so a dozen unrelated reports became context and
    # sources.
    cited = json.dumps(
        {
            "points": [
                {"description": "from c1", "score": 80, "report_ids": ["2"]},
                {"description": "from c3", "score": 60, "report_ids": [" 4 "]},
            ]
        }
    )
    strat = _strategy(map_batch_size=4, map_outputs=[cited])
    out = await strat._apply_map_reduce(_communities(4), SearchQuery(query="q"))
    assert [(r.source, r.score) for r in _evidence(out)] == [("c1", 0.8), ("c3", 0.6)]


def _cited_community(i: int) -> RetrievalResult:
    report = _community(i)
    report.metadata.update(
        text_unit_ids=[f"tu-{i}a", f"tu-{i}b", "tu-shared"],
        document_ids=[f"doc-{i}"],
    )
    return report


@pytest.mark.parametrize(
    "descriptions",
    [
        ("Vendor ships widgets", "Depot stores them"),
        ("공급사가 부품을 출하한다", "창고가 부품을 보관한다"),
        ("供給者が部品を出荷する", "倉庫が部品を保管する"),
    ],
    ids=["en", "ko", "ja"],
)
async def test_each_point_carries_the_lineage_of_the_reports_it_credits(
    descriptions: tuple[str, str],
) -> None:
    # Regression: the key points were one uncitable section, so an answer
    # drawn from them cited nothing, and with the evidence budget most of
    # their reports never reached the context. MS GraphRAG's reduce keeps
    # each point's [Data: Reports (...)]: here each point is its own
    # section whose lineage is the union of its credited reports'.
    one, both = descriptions
    payload = json.dumps(
        {
            "points": [
                {"description": one, "score": 80, "report_ids": ["2"]},
                {"description": both, "score": 60},  # no ids: its whole batch
            ]
        },
        ensure_ascii=False,
    )
    strat = _strategy(map_batch_size=2, map_outputs=[payload], evidence_tokens=0)
    reports = [_cited_community(0), _cited_community(1)]
    out = await strat._apply_map_reduce(reports, SearchQuery(query="q"))

    first, second = out  # evidence_tokens 0: the points alone
    assert one in first.content and both in second.content
    assert first.metadata["citable"] is True
    assert first.metadata["report_ids"] == ["c1"]
    assert first.metadata["text_unit_ids"] == ["tu-1a", "tu-1b", "tu-shared"]
    assert first.metadata["document_ids"] == ["doc-1"]
    assert second.metadata["report_ids"] == ["c0", "c1"]
    assert second.metadata["text_unit_ids"] == [
        "tu-0a",
        "tu-0b",
        "tu-shared",
        "tu-1a",
        "tu-1b",
    ]
    assert second.metadata["document_ids"] == ["doc-0", "doc-1"]


async def test_a_point_of_reports_without_lineage_is_not_citable() -> None:
    strat = _strategy(map_batch_size=2, map_outputs=[_map_payload(("p", 70))])
    out = await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))
    assert out[0].metadata["citable"] is False
    assert out[0].metadata["text_unit_ids"] == []


@pytest.mark.parametrize("ids", [None, [], ["9"], ["cid-0"], "1"])
async def test_a_point_without_valid_ids_credits_its_batch(ids: Any) -> None:
    point: dict[str, Any] = {"description": "somewhere", "score": 70}
    if ids is not None:
        point["report_ids"] = ids
    strat = _strategy(
        map_batch_size=2, map_outputs=[json.dumps({"points": [point]}), ""]
    )
    out = await strat._apply_map_reduce(_communities(3), SearchQuery(query="q"))
    assert [r.source for r in _evidence(out)] == ["c0", "c1"]


def test_the_map_prompt_asks_for_the_report_ids() -> None:
    from unified_kg_rag.domain.prompts import GlobalMapPrompt

    assert '"report_ids"' in GlobalMapPrompt.system_prompt_template
    # Each report is headed by its ordinal in the batch, not its UUID.
    formatted = GlobalSearchStrategy._format_reports(_communities(2))
    assert formatted.startswith("--- Report (id: 1) ---\ncommunity report 0")
    assert "--- Report (id: 2) ---\ncommunity report 1" in formatted
    assert "cid-" not in formatted


async def test_degraded_path_passes_unrated_reports_through() -> None:
    strat = _strategy(
        map_batch_size=1,
        map_reduce_min_results=2,
        map_outputs=["garbage", "still garbage"],  # all map calls unparseable
    )
    results = _communities(2)

    out = await strat._apply_map_reduce(results, SearchQuery(query="q"))

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

    async def ainvoke(self, single_input: dict, config: Any = None) -> str:
        return self.invoke(single_input, config)


async def test_map_phase_fatal_error_propagates_when_not_ignored() -> None:
    strat = _strategy(ignore_errors=False, map_batch_size=1)
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

    strat = _strategy(ignore_errors=False, map_batch_size=1)
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
    strat = _strategy(ignore_errors=False, map_batch_size=5)
    strat.map_rater = _DeniedMapChain(overflow)
    out = await strat._apply_map_reduce(_communities(3), SearchQuery(query="q"))
    assert [r.source for r in out] == ["c0", "c1", "c2"]


async def test_map_phase_fatal_error_raises_even_when_ignored() -> None:
    # As DRIFT's: a model the caller cannot access fails the query, where
    # ignore_errors answered from the unrated reports.
    strat = _strategy(ignore_errors=True, map_batch_size=1)
    strat.map_rater = _DeniedMapChain(
        ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "no access"}},
            "InvokeModel",
        )
    )
    with pytest.raises(ClientError, match="AccessDenied"):
        await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))


async def test_map_phase_failing_on_every_batch_raises_when_not_ignored() -> None:
    # Regression: a map model that answered no batch (throttled past its
    # retries, timing out) degraded silently to an answer from the unjudged
    # reports; DRIFT raises the first error when a whole step fails.
    strat = _strategy(ignore_errors=False, map_batch_size=1)
    strat.map_rater = _DeniedMapChain(TimeoutError("read timed out"))
    with pytest.raises(TimeoutError, match="read timed out"):
        await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))


async def test_map_phase_failing_on_every_batch_degrades_when_ignored() -> None:
    strat = _strategy(ignore_errors=True, map_batch_size=1)
    strat.map_rater = _DeniedMapChain(TimeoutError("read timed out"))
    with degradation_scope() as ledger:
        out = await strat._apply_map_reduce(_communities(2), SearchQuery(query="q"))
    assert out
    assert _types(ledger) == [("global_map", "TimeoutError")]


class _HangingMapChain:
    """Map chain whose calls never answer; records each call it starts."""

    def __init__(self) -> None:
        self.started = 0
        self.cancelled = 0

    async def ainvoke(self, single_input: dict, config: Any = None) -> str:
        self.started += 1
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        return _map_payload(("late", 50))


@pytest.mark.timeout(30)
async def test_a_deadline_cancels_the_map_calls_and_starts_no_more() -> None:
    # Regression: the map calls ran in a worker thread (BatchProcessor), so a
    # query deadline or a closed chain left them running and the queued ones
    # kept starting, each a billed model call, after the query had failed.
    strat = _strategy(ignore_errors=False, map_batch_size=1)
    strat.max_concurrency = 2
    strat.map_rater = _HangingMapChain()
    batches = _batches(strat, _communities(10))
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.2):
            await strat._run_map_phase(batches, SearchQuery(query="q"))
    await asyncio.sleep(0.3)
    assert strat.map_rater.started == 2  # type: ignore[attr-defined]
    assert strat.map_rater.cancelled == 2  # type: ignore[attr-defined]


class _TimingOutFirstReport(_SyncMapChain):
    def invoke(self, single_input: dict, config: Any = None) -> str:
        if "community report 0" in single_input["reports"]:
            raise TimeoutError("read timed out")
        return super().invoke(single_input, config)


async def test_map_phase_unrated_batches_are_recorded_as_degraded() -> None:
    # Regression: a failed or unparseable map batch left no trace in the
    # search metadata (degraded_error) or the strategy span.
    strat = _strategy(ignore_errors=False, map_batch_size=1)
    strat.map_rater = _TimingOutFirstReport([_map_payload(("point", 50))])
    with degradation_scope() as ledger:
        points, unrated = await strat._run_map_phase(
            _batches(strat, _communities(2)), SearchQuery(query="q")
        )
    assert ([r.source for r in unrated], len(points)) == (["c0"], 1)
    assert _types(ledger) == [("global_map", "TimeoutError")]

    strat.map_rater = _SyncMapChain(["not json", _map_payload(("point", 50))])
    with degradation_scope() as ledger:
        await strat._run_map_phase(
            _batches(strat, _communities(2)), SearchQuery(query="q")
        )
    assert _types(ledger) == [("global_map", "LanguageModelError")]


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
        self._met = asyncio.Event()

    async def ainvoke(self, single_input: dict, config: Any = None) -> str:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        if self.max_active >= self._target:
            self._met.set()
        try:
            await asyncio.wait_for(self._met.wait(), self._timeout)
        except TimeoutError:
            pass
        self.active -= 1
        return _map_payload(("p", 50))


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
# evidence_tokens — the cited report text after the key points
# --------------------------------------------------------------------------- #


def _ranked_three() -> list[str]:
    return [
        json.dumps(
            {
                "points": [
                    {"description": "a", "score": 90, "report_ids": ["1"]},
                    {"description": "b", "score": 60, "report_ids": ["2"]},
                    {"description": "c", "score": 30, "report_ids": ["3"]},
                ]
            }
        )
    ]


@pytest.mark.parametrize(
    ("evidence_tokens", "kept"),
    [(None, ["c0", "c1", "c2"]), (25, ["c0", "c1"]), (10, ["c0"]), (9, []), (0, [])],
)
async def test_evidence_tokens_caps_the_cited_reports(
    evidence_tokens: int | None, kept: list[str]
) -> None:
    # Each report costs 10 tokens: whole reports, best point first, until the
    # next would exceed the cap; null keeps them all.
    strat = _strategy(
        map_batch_size=3, map_outputs=_ranked_three(), evidence_tokens=evidence_tokens
    )
    stats: dict[str, Any] = {}
    out = await strat._apply_map_reduce(
        _communities(3), SearchQuery(query="q"), stats=stats
    )
    assert out[0].source == "synthesized_key_points_1"
    assert [r.source for r in _evidence(out)] == kept
    if evidence_tokens is None:
        assert "global_evidence_reports" not in stats
    else:
        assert stats["global_evidence_reports"] == len(kept)


def test_evidence_tokens_defaults_to_4000() -> None:
    # Measured: -41% cost per global query against no cap, with no
    # consistent quality difference and the sources kept.
    from unified_kg_rag.domain.models.config import GlobalSearchConfig

    assert GlobalSearchConfig().evidence_tokens == 4000


@pytest.mark.parametrize(
    ("metadata", "translation", "language"),
    [
        ({"query_language": "Korean"}, {}, "Korean"),
        # Untranslated, the index is in its source language, not the target.
        (
            {},
            {"enabled": False, "source_language": "zh", "target_language": "en"},
            "Chinese",
        ),
    ],
    ids=["from-the-chain", "index-as-written"],
)
async def test_the_map_points_are_written_in_the_query_language(
    metadata: dict[str, Any], translation: dict[str, Any], language: str
) -> None:
    strat = _strategy(map_outputs=[_map_payload(("p1", 50))])
    for name, value in translation.items():
        setattr(strat.config.processing.translation, name, value)
    await strat._run_map_phase(
        _batches(strat, _communities(1)), SearchQuery(query="q", metadata=metadata)
    )
    (batch,) = strat.map_rater.batch_inputs
    assert batch["target_language"] == language


class _RecordStore:
    """A document retriever over stored records, by index prefix: a fetch
    by id returns the stored ones it names (and records the request)."""

    def __init__(self, records: dict[str, dict[str, dict[str, Any]]]) -> None:
        self.records = records
        self.calls: list[SearchQuery] = []

    async def aindex_limits(self, prefix: str, suffix: str | None) -> Any:
        from unified_kg_rag.adapters.storage.filter_schema import IndexLimits

        return IndexLimits()

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        stored = self.records.get((query.index_prefixes or [""])[0], {})
        return [
            RetrievalResult(
                content="",
                score=1.0,
                source=record_id,
                retriever_type="document",
                metadata={"id": record_id, **stored[record_id]},
            )
            for record_id in (query.filters or {}).get("id") or []
            if record_id in stored
        ]


def _findings(*cited: tuple[list[str], list[str]]) -> list[dict[str, Any]]:
    return [
        {"summary": "s", "explanation": "e", "entity_ids": e, "relationship_ids": r}
        for e, r in cited
    ]


def _report_store(config: Config) -> _RecordStore:
    o = config.indexing.opensearch
    return _RecordStore(
        {
            o.entities_index_prefix: {
                "e-ko": {
                    "text_unit_ids": ["tu-ko-1", "tu-ko-2"],
                    "document_ids": ["ko"],
                },
                "e-ja": {"text_unit_ids": ["tu-ja"], "document_ids": ["ja"]},
            },
            o.relationships_index_prefix: {
                "r-en": {"text_unit_ids": ["tu-en"], "document_ids": ["en"]},
            },
        }
    )


@pytest.mark.parametrize(
    "description",
    ["Vendor ships widgets", "공급사가 부품을 출하한다", "供給者が部品を出荷する"],
    ids=["en", "ko", "ja"],
)
async def test_a_point_cites_what_its_reports_findings_cite(description: str) -> None:
    # Regression (rv5 M1): a key point's lineage was its reports' community
    # membership, so every point of a corpus-spanning community named the
    # same documents. It is the text units of the records the credited
    # reports' findings cite (a finding's relationships before its
    # entities), read by id; a report stored without findings keeps its
    # membership.
    payload = json.dumps(
        {
            "points": [
                {"description": description, "score": 80, "report_ids": ["1"]},
                {"description": description, "score": 70, "report_ids": ["2"]},
                {"description": description, "score": 60, "report_ids": ["3"]},
            ]
        },
        ensure_ascii=False,
    )
    strat = _strategy(map_batch_size=3, map_outputs=[payload], evidence_tokens=0)
    store = _report_store(strat.config)
    strat.retrievers = {"document": store}
    cited, gone, legacy = (_cited_community(i) for i in range(3))
    cited.metadata["findings"] = _findings((["e-ko"], ["r-en"]), (["e-ja"], []))
    gone.metadata["findings"] = _findings((["e-deleted"], ["r-deleted"]))
    out = await strat._apply_map_reduce(
        [cited, gone, legacy], SearchQuery(query="q", suffix="ns")
    )

    first, second, third = out
    assert first.metadata["text_unit_ids"] == ["tu-en", "tu-ko-1", "tu-ko-2", "tu-ja"]
    assert first.metadata["document_ids"] == ["en", "ko", "ja"]
    assert first.metadata["citable"] is True
    # Every cited record removed: the membership stands, as without findings.
    assert second.metadata["text_unit_ids"] == ["tu-1a", "tu-1b", "tu-shared"]
    assert third.metadata["text_unit_ids"] == ["tu-2a", "tu-2b", "tu-shared"]
    o = strat.config.indexing.opensearch
    # One fetch per index, of every cited id at once, in the namespace.
    assert sorted(
        (c.index_prefixes[0], tuple(c.filters["id"])) for c in store.calls
    ) == [
        (o.entities_index_prefix, ("e-ko", "e-ja", "e-deleted")),
        (o.relationships_index_prefix, ("r-en", "r-deleted")),
    ]
    assert {c.suffix for c in store.calls} == {"ns"}


async def test_findings_lineage_reads_no_relationships_index_the_namespace_lacks() -> (
    None
):
    from unified_kg_rag.domain.retrieval.index_prefixes import HELD_ARTIFACTS_KEY

    strat = _strategy(
        map_batch_size=1, map_outputs=[_map_payload(("p", 70))], evidence_tokens=0
    )
    store = _report_store(strat.config)
    strat.retrievers = {"document": store}
    report = _cited_community(0)
    report.metadata["findings"] = _findings((["e-ja"], ["r-en"]))
    held = ["text_units", "entities", "relationships", "graph", "communities"]
    out = await strat._apply_map_reduce(
        [report],
        SearchQuery(
            query="q", metadata={HELD_ARTIFACTS_KEY: [*held, "community_reports"]}
        ),
    )
    assert out[0].metadata["text_unit_ids"] == ["tu-ja"]
    o = strat.config.indexing.opensearch
    assert [c.index_prefixes for c in store.calls] == [[o.entities_index_prefix]]


async def test_a_failed_findings_read_keeps_the_membership_lineage() -> None:
    class _Failing(_RecordStore):
        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            raise RuntimeError("one request failed")

    strat = _strategy(
        map_batch_size=1, map_outputs=[_map_payload(("p", 70))], evidence_tokens=0
    )
    strat.retrievers = {"document": _Failing({})}
    report = _cited_community(0)
    report.metadata["findings"] = _findings((["e-ja"], []))
    with degradation_scope() as ledger:
        out = await strat._apply_map_reduce([report], SearchQuery(query="q"))
    assert out[0].metadata["text_unit_ids"] == ["tu-0a", "tu-0b", "tu-shared"]
    # Regression (rv6): the failure was only logged, so the answer read as
    # a clean one; it is reported as the search's degraded error.
    assert _types(ledger) == [("findings_lineage", "RuntimeError")]
    # Regression (rv7 algo-ret L9): under the default citations (global's
    # off), only the reported sources rest on the lineage: recorded apart.
    strat = _strategy(
        map_batch_size=1, map_outputs=[_map_payload(("p", 70))], evidence_tokens=0
    )
    strat.retrievers = {"document": _Failing({})}
    query = SearchQuery(query="q", metadata={"search_strategy": "global"})
    with degradation_scope() as ledger:
        await strat._apply_map_reduce([report], query)
    assert not ledger.entries
    assert ledger.metadata() == {"citations_unresolved": True}


class _FilteringRecordStore(_RecordStore):
    """A record store that applies a ``document_ids`` filter, as a real
    index does."""

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        hits = await super().aretrieve(query)
        allowed = set((query.filters or {}).get("document_ids") or [])
        return [
            h
            for h in hits
            if not allowed or allowed & set(h.metadata.get("document_ids") or [])
        ]


@pytest.mark.parametrize(
    ("description", "filters"),
    [
        ("Vendor ships widgets", {"document_ids": ["doc-0"]}),
        ("공급사가 부품을 출하한다", {"document_ids": ["doc-0"], "id": ["x"]}),
        ("供給者が部品を出荷する", None),
    ],
    ids=["en-filtered", "ko-filtered-id", "ja-unfiltered"],
)
async def test_findings_records_are_read_without_the_callers_filters(
    description: str, filters: dict[str, Any] | None
) -> None:
    # Regression (rv6 M1): the cited records were read under the caller's
    # filters, so a record of another document came back missing and the
    # point fell back to the report's membership: the caller's own
    # documents, which the findings do not cite. They are read in the
    # namespace alone; the citation applies the filters to the units.
    payload = json.dumps(
        {"points": [{"description": description, "score": 80, "report_ids": ["0"]}]},
        ensure_ascii=False,
    )
    strat = _strategy(map_batch_size=1, map_outputs=[payload])
    o = strat.config.indexing.opensearch
    store = _FilteringRecordStore(
        {
            o.entities_index_prefix: {
                "e-b": {"text_unit_ids": ["tu-b"], "document_ids": ["doc-B"]}
            }
        }
    )
    strat.retrievers = {"document": store}
    report = _cited_community(0)
    report.metadata["findings"] = _findings((["e-b"], []))
    out = await strat._apply_map_reduce(
        [report], SearchQuery(query="q", suffix="ns", filters=filters)
    )
    point, evidence = out
    assert point.metadata["text_unit_ids"] == ["tu-b"]
    assert point.metadata["document_ids"] == ["doc-B"]
    assert [c.filters for c in store.calls] == [{"id": ["e-b"]}]
    assert {c.suffix for c in store.calls} == {"ns"}
    # The report's own section cites what the point cites; its membership
    # stays its text_unit_ids.
    assert evidence.metadata["findings_text_unit_ids"] == ["tu-b"]
    assert evidence.metadata["findings_document_ids"] == ["doc-B"]
    assert evidence.metadata["text_unit_ids"] == ["tu-0a", "tu-0b", "tu-shared"]


async def test_reports_without_map_reduce_cite_their_findings() -> None:
    strat = _strategy()
    store = _report_store(strat.config)
    strat.retrievers = {"document": store}
    cited, legacy = _cited_community(0), _cited_community(1)
    cited.metadata["findings"] = _findings((["e-ja"], []))
    out = await strat._cite_report_findings(SearchQuery(query="q"), [cited, legacy])
    assert out[0].metadata["findings_text_unit_ids"] == ["tu-ja"]
    assert "findings_text_unit_ids" not in out[1].metadata
    # A report that carries its lineage is not read again.
    store.calls.clear()
    assert await strat._cite_report_findings(SearchQuery(query="q"), out) == out
    assert store.calls == []


@pytest.mark.parametrize("fail", [False, True], ids=["read", "failed-read"])
async def test_reports_passed_through_without_map_reduce_cite_their_findings(
    fail: bool,
) -> None:
    class _Failing(_RecordStore):
        async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
            raise RuntimeError("one request failed")

    strat = _strategy()
    strat.global_search_config.enable_map_reduce = False
    store = _report_store(strat.config)
    strat.retrievers = {"document": _Failing({}) if fail else store}
    report = _cited_community(0)
    report.metadata["findings"] = _findings((["e-ko"], []))

    async def cut(query: SearchQuery, stats: dict[str, Any]) -> list[RetrievalResult]:
        return [report]

    strat._cut_reports = cut  # type: ignore[method-assign]
    result = await strat.asearch(SearchQuery(query="공급사는 누구인가?"))
    (out,) = result.results
    if fail:
        assert "findings_text_unit_ids" not in out.metadata
        assert result.metadata["degraded_error"] == "RuntimeError"
    else:
        assert out.metadata["findings_text_unit_ids"] == ["tu-ko-1", "tu-ko-2"]
        assert "degraded_error" not in result.metadata
