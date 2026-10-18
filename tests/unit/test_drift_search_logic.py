# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free tests of DRIFT search (MS GraphRAG parity).

DRIFT expands the query (HyDE), primes intermediate answers and follow-up
queries from the most similar community reports in folds, and runs the
follow-ups as local searches over ``n_depth`` rounds, each answered with its
own follow-ups. The strategy is built via ``__new__``; the chains, the report
retriever and the local search are stubs, so the tests see which queries,
reports and contexts each step receives.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from botocore.exceptions import ClientError

from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer
from unified_kg_rag.adapters.search_strategies.drift_search import (
    DriftSearchStrategy,
    _score,
)
from unified_kg_rag.adapters.search_strategies.local_search import (
    LocalSearchStrategy,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    SearchQuery,
    SearchResult,
)

pytestmark = pytest.mark.unit

REPORTS_PREFIX = Config().indexing.opensearch.community_reports_index_prefix


def _result(
    content: str,
    *,
    score: float = 0.5,
    source: str | None = None,
    metadata: dict | None = None,
    retriever_type: str = "text",
) -> RetrievalResult:
    return RetrievalResult(
        content=content,
        score=score,
        source=source or content,
        retriever_type=retriever_type,
        metadata=metadata or {},
    )


class _Chain:
    """Async chain stub: ``respond(inputs)`` gives the output (or raises)."""

    def __init__(self, respond) -> None:
        self.respond = respond
        self.calls: list[dict] = []
        self.configs: list[Any] = []

    async def ainvoke(self, inputs: dict, config: Any = None) -> str:
        self.calls.append(inputs)
        self.configs.append(config)
        value = self.respond(inputs)
        if isinstance(value, BaseException):
            raise value
        return value


class _Reports:
    """The community-report index: a browse gives the template, a search the
    ``top_k`` reports (it records the query text it was given)."""

    def __init__(self, count: int = 6) -> None:
        self.reports = [
            _result(f"report {i}", retriever_type="community") for i in range(count)
        ]
        self.queries: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.queries.append(query)
        assert query.index_prefixes == [REPORTS_PREFIX]
        return self.reports[: query.top_k]


class _Local:
    """Local search: one chunk naming the follow-up query."""

    def __init__(self, empty: set[str] | None = None) -> None:
        self.queries: list[str] = []
        self.empty = empty or set()

    async def asearch(self, query: SearchQuery, config: Any = None) -> SearchResult:
        self.queries.append(query.query)
        results = (
            []
            if query.query in self.empty
            else [_result(f"chunk for {query.query}", source=f"t:{query.query}")]
        )
        return SearchResult(
            query=query,
            results=results,
            total_results=len(results),
            search_strategy="local_search",
            processing_time=0.0,
        )


class _Tokens:
    def optimize_context(self, results, query, max_tokens):
        return results

    def build_context_string(self, optimized) -> str:
        return "\n".join(r.content for r in optimized)

    def count_tokens_many(self, texts: list[str]) -> list[int]:
        return [len(t) for t in texts]


def _primer(follow_ups: dict[str, list[str]] | None = None):
    """Primer: one answer per fold, with follow-ups keyed by the fold's first
    report (default: one follow-up per report)."""

    def respond(inputs: dict) -> str:
        first = inputs["community_reports"].split("\n\n")[0]
        ups = (follow_ups or {}).get(first, [f"about {first}?"])
        return json.dumps(
            {
                "intermediate_answer": f"draft from {first}",
                "score": 40,
                "follow_up_queries": ups,
            }
        )

    return respond


def _action(tree: dict[str, tuple[int, list[str]]] | None = None):
    """Action: answers ``query`` with its score and follow-ups from ``tree``."""

    def respond(inputs: dict) -> str:
        score, ups = (tree or {}).get(inputs["query"], (50, []))
        return json.dumps(
            {
                "response": f"answer to {inputs['query']}",
                "score": score,
                "follow_up_queries": ups,
            }
        )

    return respond


def _strategy(
    *,
    reports: _Reports | None = None,
    local: _Local | None = None,
    primer=None,
    action=None,
    hyde=None,
    ignore_errors: bool = False,
    **settings: Any,
) -> DriftSearchStrategy:
    config = Config()
    config.search.reranking.enabled = False
    config.processing.ignore_errors = ignore_errors
    drift = config.search.drift_search
    drift.k_followups = 6
    drift.primer_folds = 3
    for key, value in settings.items():
        setattr(drift, key, value)
    strat = DriftSearchStrategy.__new__(DriftSearchStrategy)
    strat.config = config
    strat.drift_config = drift
    strat.ignore_errors = ignore_errors
    strat.max_concurrency = 4
    strat.target_language = "English"
    strat.retrievers = {"document": reports or _Reports()}
    strat.hybrid_scorer = HybridScorer(config)
    strat.token_manager = _Tokens()  # type: ignore[assignment]
    strat.local_search = local or _Local()  # type: ignore[assignment]
    strat.hyde = _Chain(hyde or (lambda inputs: "hypothetical report"))
    strat.primer = _Chain(primer or _primer())
    strat.action = _Chain(action or _action())
    return strat


def _answers(result: SearchResult) -> str:
    first = result.results[0]
    assert first.metadata.get("synthesized") is True
    return first.content


# --------------------------------------------------------------------------- #
# Primer
# --------------------------------------------------------------------------- #


def test_defaults_bound_the_llm_calls() -> None:
    # MS GraphRAG's 20 follow-ups over 3 rounds made up to 67 LLM calls a
    # query; the defaults make at most 27.
    drift = Config().search.drift_search
    assert (drift.k_followups, drift.primer_folds, drift.n_depth) == (10, 5, 2)
    calls = 1 + drift.primer_folds + drift.k_followups * drift.n_depth + 1
    assert calls == 27


async def test_the_primer_reads_the_reports_similar_to_the_expanded_query() -> None:
    reports = _Reports()
    strat = _strategy(reports=reports, n_depth=1)
    await strat.asearch(SearchQuery(query="question"))

    template, similar = reports.queries
    assert template.query == "" and template.top_k == 1
    # The hypothetical answer, not the question, finds the primer reports.
    assert strat.hyde.calls[0]["template"] == "report 0"
    assert similar.query == "hypothetical report"
    assert similar.top_k == 6
    # Six reports in three folds of two, one primer call each.
    folds = [c["community_reports"] for c in strat.primer.calls]
    assert sorted(folds) == [
        "report 0\n\nreport 1",
        "report 2\n\nreport 3",
        "report 4\n\nreport 5",
    ]


def test_follow_up_searches_do_not_rerank_and_use_the_search_concurrency() -> None:
    # Regression: every follow-up's local search made its own rerank call
    # (up to 60 a query) whose outcome was never reported, and the fan-out
    # read the ingestion knob processing.max_concurrency.
    config = Config()
    config.search.max_concurrency = 3
    config.processing.max_concurrency = 40
    strat = DriftSearchStrategy(config, {})
    assert strat.local_search.rerank_types == frozenset()
    assert strat.max_concurrency == 3
    assert LocalSearchStrategy(config, {}).rerank_types == {"text"}


def test_folds_are_contiguous_and_never_empty() -> None:
    items = [_result(str(i)) for i in range(5)]
    parts = DriftSearchStrategy._folds(items, 3)
    assert [[r.content for r in p] for p in parts] == [["0", "1"], ["2", "3"], ["4"]]
    assert len(DriftSearchStrategy._folds(items[:2], 5)) == 2
    assert DriftSearchStrategy._folds([], 5) == []


async def test_a_failed_expansion_searches_with_the_question() -> None:
    reports = _Reports()
    strat = _strategy(
        reports=reports, hyde=lambda _: TimeoutError("slow"), ignore_errors=True
    )
    result = await strat.asearch(SearchQuery(query="question"))
    assert reports.queries[1].query == "question"
    assert result.metadata["degraded_error"] == "TimeoutError"


async def test_a_failed_expansion_raises_without_ignore_errors() -> None:
    strat = _strategy(hyde=lambda _: TimeoutError("slow"))
    with pytest.raises(TimeoutError):
        await strat.asearch(SearchQuery(query="question"))


async def test_without_reports_the_question_itself_is_searched() -> None:
    local = _Local()
    strat = _strategy(reports=_Reports(count=0), local=local)
    result = await strat.asearch(SearchQuery(query="question"))
    assert local.queries == ["question"]
    assert strat.primer.calls == [] and strat.hyde.calls == []
    assert "answer to question" in _answers(result)


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #


async def test_follow_ups_run_as_local_searches_round_by_round() -> None:
    # Regression: DRIFT rewrote one query per iteration; follow-up queries
    # never ran as local searches answered on their own.
    local = _Local()
    tree = {
        "about report 0?": (90, ["deeper A?"]),
        "about report 2?": (30, ["deeper B?"]),
        "deeper A?": (70, ["deepest?"]),
    }
    strat = _strategy(local=local, action=_action(tree), n_depth=3)
    result = await strat.asearch(SearchQuery(query="question"))

    assert local.queries[:3] == [
        "about report 0?",
        "about report 2?",
        "about report 4?",
    ]
    # Round 2: the follow-ups of the better answer first.
    assert local.queries[3:5] == ["deeper A?", "deeper B?"]
    assert local.queries[5:] == ["deepest?"]
    assert result.metadata["drift_depth"] == 3
    assert result.metadata["drift_follow_up_queries"] == 6
    # Each follow-up's context is its own local search.
    contexts = {c["query"]: c["context"] for c in strat.action.calls}
    assert contexts["deeper A?"] == "chunk for deeper A?"
    assert all(c["global_query"] == "question" for c in strat.action.calls)


async def test_rounds_stop_at_n_depth_and_k_followups() -> None:
    local = _Local()
    tree = {f"about report {i}?": (50, [f"next {i}?"]) for i in (0, 2, 4)}
    strat = _strategy(local=local, action=_action(tree), n_depth=1, k_followups=6)
    result = await strat.asearch(SearchQuery(query="question"))
    assert len(local.queries) == 3
    assert result.metadata["drift_open_follow_ups"] == 3

    local = _Local()
    strat = _strategy(local=local, action=_action(tree), k_followups=2, n_depth=1)
    await strat.asearch(SearchQuery(query="question"))
    # Two reports, so one fold-answer per report and two follow-ups a round.
    assert local.queries == ["about report 0?", "about report 1?"]


async def test_a_round_runs_at_most_k_followups_of_the_open_queries() -> None:
    # k_followups 2: two reports, one per fold, give six open follow-ups;
    # two run, the earliest of the equally placed first, and four stay open.
    local = _Local()
    primer = _primer({f"report {i}": [f"{i}a?", f"{i}b?", f"{i}c?"] for i in (0, 1)})
    strat = _strategy(local=local, primer=primer, n_depth=1)
    strat.drift_config.k_followups = 2
    result = await strat.asearch(SearchQuery(query="question"))
    assert local.queries == ["0a?", "0b?"]
    assert result.metadata["drift_follow_up_queries"] == 2
    assert result.metadata["drift_open_follow_ups"] == 4


async def test_an_action_reads_its_local_context_within_the_budget() -> None:
    budgets: list[int] = []

    class _Budget(_Tokens):
        def optimize_context(self, results, query, max_tokens):
            budgets.append(max_tokens)
            return results[:0] if max_tokens < 1000 else results

    strat = _strategy(n_depth=1, local_context_tokens=999)
    strat.token_manager = _Budget()  # type: ignore[assignment]
    await strat.asearch(SearchQuery(query="question"))
    assert budgets and set(budgets) == {999}
    # What the budget kept is the context the action answers from.
    assert all(c["context"] == "" for c in strat.action.calls)


async def test_a_follow_up_asked_twice_runs_once() -> None:
    local = _Local()
    primer = _primer({f"report {i}": ["same?", "question"] for i in (0, 2, 4)})
    strat = _strategy(local=local, primer=primer)
    await strat.asearch(SearchQuery(query="question"))
    assert local.queries == ["same?"]


async def test_a_follow_up_without_local_results_costs_no_action_call() -> None:
    strat = _strategy(local=_Local(empty={"about report 2?"}))
    await strat.asearch(SearchQuery(query="question"))
    assert "about report 2?" not in [c["query"] for c in strat.action.calls]


async def test_a_failed_action_keeps_its_evidence_under_ignore_errors() -> None:
    strat = _strategy(action=lambda _: TimeoutError("slow"), ignore_errors=True)
    result = await strat.asearch(SearchQuery(query="question"))
    assert result.metadata["degraded_error"] == "TimeoutError"
    sources = {r.source for r in result.results}
    assert "t:about report 0?" in sources


async def test_a_failed_action_raises_without_ignore_errors() -> None:
    strat = _strategy(action=lambda _: TimeoutError("slow"))
    with pytest.raises(TimeoutError):
        await strat.asearch(SearchQuery(query="question"))


async def test_one_failed_action_leaves_only_its_answer_out() -> None:
    # Regression: one throttled action failed the whole query, after every
    # other action had been paid for.
    def action(inputs: dict):
        if inputs["query"] == "about report 2?":
            return TimeoutError("throttled")
        return _action()(inputs)

    strat = _strategy(action=action, n_depth=1)
    result = await strat.asearch(SearchQuery(query="question"))
    assert result.metadata["degraded_error"] == "TimeoutError"
    text = _answers(result)
    assert "answer to about report 0?" in text
    assert "answer to about report 2?" not in text
    # The failed action's local evidence is kept.
    assert "t:about report 2?" in {r.source for r in result.results}


async def test_a_failed_round_stops_the_rounds() -> None:
    tree = {f"about report {i}?": (50, [f"next {i}?"]) for i in (0, 2, 4)}

    def action(inputs: dict):
        if inputs["query"].startswith("next"):
            return TimeoutError("throttled")
        return _action(tree)(inputs)

    local = _Local()
    strat = _strategy(local=local, action=action, n_depth=3, k_followups=6)
    # Two follow-ups a round, so round 2 leaves one "next" query queued.
    strat.drift_config.k_followups = 2
    result = await strat.asearch(SearchQuery(query="question"))
    # Round 1 answered, round 2 failed entirely: no round 3.
    assert result.metadata["drift_depth"] == 2
    assert result.metadata["degraded_error"] == "TimeoutError"


async def test_one_failed_primer_fold_leaves_only_its_answer_out() -> None:
    def primer(inputs: dict):
        if inputs["community_reports"].startswith("report 2"):
            return TimeoutError("throttled")
        return _primer()(inputs)

    local = _Local()
    strat = _strategy(primer=primer, local=local, n_depth=1)
    result = await strat.asearch(SearchQuery(query="question"))
    assert local.queries == ["about report 0?", "about report 4?"]
    assert result.metadata["degraded_error"] == "TimeoutError"


async def test_every_primer_fold_failing_raises_without_ignore_errors() -> None:
    strat = _strategy(primer=lambda _: TimeoutError("slow"))
    with pytest.raises(TimeoutError):
        await strat.asearch(SearchQuery(query="question"))


async def test_a_fatal_error_cancels_the_other_actions() -> None:
    # Regression: the query raised while its sibling actions kept running
    # (and paying for) their LLM calls.
    denied = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "Converse"
    )
    finished: list[str] = []

    class _Slow(_Chain):
        async def ainvoke(self, inputs: dict, config: Any = None) -> str:
            if inputs["query"] == "about report 0?":
                raise denied
            await asyncio.sleep(0.2)
            finished.append(inputs["query"])
            return _action()(inputs)

    strat = _strategy(n_depth=1)
    strat.action = _Slow(None)
    with pytest.raises(ClientError):
        await strat.asearch(SearchQuery(query="question"))
    await asyncio.sleep(0.3)
    assert finished == []


async def test_a_fatal_action_error_raises_even_under_ignore_errors() -> None:
    denied = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "no"}}, "Converse"
    )
    strat = _strategy(action=lambda _: denied, ignore_errors=True)
    with pytest.raises(ClientError):
        await strat.asearch(SearchQuery(query="question"))


# --------------------------------------------------------------------------- #
# Reduce
# --------------------------------------------------------------------------- #


async def test_answers_are_ranked_and_packed_into_the_budget() -> None:
    tree = {"about report 0?": (90, []), "about report 2?": (10, [])}
    strat = _strategy(action=_action(tree))
    result = await strat.asearch(SearchQuery(query="question"))
    text = _answers(result)
    assert text.index("answer to about report 0?") < text.index("draft from report 0")
    assert text.index("draft from report 0") < text.index("answer to about report 2?")
    assert result.results[0].source == "synthesized_drift_answers"

    strat = _strategy(action=_action(tree), max_answer_tokens=128)
    text = _answers(await strat.asearch(SearchQuery(query="question")))
    assert "answer to about report 0?" in text
    assert "answer to about report 2?" not in text


async def test_evidence_is_fused_with_the_local_quota(mocker) -> None:
    strat = _strategy()
    spy = mocker.spy(strat, "_fuse_and_rerank")
    query = SearchQuery(query="question", top_k=7)
    await strat.asearch(query)
    kwargs = spy.call_args.kwargs
    assert kwargs["per_type_quota"] == strat._per_type_quota(7)
    assert kwargs["rerank_only_types"] == {"text"}
    fused = spy.call_args.args[0]["results"]
    assert len(fused) == len({r.source for r in fused})  # deduplicated


async def test_the_callers_config_reaches_every_llm_call() -> None:
    strat = _strategy(n_depth=1)
    caller_config = {"tags": ["caller"]}
    await strat.asearch(SearchQuery(query="question"), config=caller_config)
    for chain in (strat.hyde, strat.primer, strat.action):
        assert chain.configs and all(c is caller_config for c in chain.configs)


async def test_actions_run_concurrently() -> None:
    in_flight = 0
    peak = 0

    class _SlowLocal(_Local):
        async def asearch(self, query, config=None):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return await super().asearch(query, config)

    strat = _strategy(local=_SlowLocal(), n_depth=1)
    await strat.asearch(SearchQuery(query="question"))
    assert peak == 3


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("80", 80), (250, 100), (-5, 0), ("Infinity", 0), ("NaN", 0), ("x", 0)],
)
def test_scores_are_clamped_integers(raw: Any, expected: int) -> None:
    assert _score(raw) == expected


def test_rank_within_type_keeps_each_store_in_its_slots() -> None:
    # Graph entities score in distance bands; the entity index's vector hits
    # sum two kNN scores. Sorted together, the index's hits took every slot.
    graph = {"_node_type": "entity"}
    index = {"_search_index": "graphrag-entities-default-1"}
    results = [
        _result("g-hop", score=0.45, metadata=graph, retriever_type="entity"),
        _result("os-b", score=1.6, metadata=index, retriever_type="entity"),
        _result("g-seed", score=0.9, metadata=graph, retriever_type="entity"),
        _result("os-a", score=1.8, metadata=index, retriever_type="entity"),
    ]
    ranked = DriftSearchStrategy._rank_within_type(results)
    assert [r.content for r in ranked] == ["g-seed", "os-a", "g-hop", "os-b"]
