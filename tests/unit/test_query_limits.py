# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A query's limits: its deadline, model calls and input tokens.

A query had a deadline (``search.query_timeout_seconds``) and nothing else:
nothing bounded how many model calls a global, DRIFT or iterative query
fanned out or how many tokens they read, and a caller (an MCP agent) could
not lower even the deadline. ``QueryLimits`` generalizes the deadline: the
config's limits bound every query, ``RAGInput.limits`` (and the MCP tools'
``limits``) lower them for one, a call past them is refused before it is sent
(the usage callback raises the refusal, where LangChain used to swallow a
handler's error), the fan-out strategies start only what fits beside the
answer and record the rest as degraded, and auto routes only to strategies
that fit.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.fakes.model_servers import FakeOllama
from tests.unit.test_global_search_map_reduce import (
    _communities,
    _map_payload,
)
from tests.unit.test_global_search_map_reduce import _strategy as _global_strategy
from tests.unit.test_query_deadline import (
    _ANSWER,
    _chain,
    _Factory,
    _Retriever,
    _SlowChatModel,
)
from unified_kg_rag import QueryLimits, UsageLimitExceededError
from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
from unified_kg_rag.adapters.observability import usage_callback
from unified_kg_rag.adapters.observability.usage_callback import UsageCallbackHandler
from unified_kg_rag.adapters.retrieval.base import degradation_scope
from unified_kg_rag.adapters.search_strategies.drift_search import (
    DriftSearchStrategy,
)
from unified_kg_rag.adapters.search_strategies.global_search import (
    GlobalSearchStrategy,
)
from unified_kg_rag.adapters.search_strategies.iterative_search import (
    IterativeSearchStrategy,
)
from unified_kg_rag.application.cli.mcp_tools import describe_error
from unified_kg_rag.application.retrieval.rag_chain import (
    DEFAULT_ERROR_MESSAGE,
    ConversationTurn,
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    ModelPurpose,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.shared import usage
from unified_kg_rag.shared.exceptions import ErrorClass, classify

pytestmark = pytest.mark.unit


# --- the meter's limits --------------------------------------------------------


def test_a_call_past_a_limit_is_refused_and_counted_nowhere() -> None:
    with usage.usage_scope() as outer:
        with usage.usage_scope(usage.UsageLimits(max_llm_calls=2)) as meter:
            for _ in range(2):
                usage.start_llm_call("query", "m", estimated_input_tokens=5)
            with pytest.raises(UsageLimitExceededError) as refused:
                usage.start_llm_call("query", "m", estimated_input_tokens=5)
    assert refused.value.limit == "max_llm_calls" and refused.value.value == 2
    assert meter.snapshot()["llm_calls"] == 2
    # The enclosing scope counts what was sent, not the refused call.
    assert outer.snapshot()["llm_calls"] == 2


def test_an_enclosing_scope_limits_the_calls_of_an_inner_one() -> None:
    with usage.usage_scope(usage.UsageLimits(max_input_tokens=100)):
        with usage.usage_scope():
            usage.start_llm_call("query", "m", estimated_input_tokens=60)
            assert usage.remaining() == usage.Remaining(input_tokens=40)
            with pytest.raises(UsageLimitExceededError, match="past its limit 100"):
                usage.start_llm_call("query", "m", estimated_input_tokens=60)


def test_a_cost_limit_prices_the_counts_with_the_refused_call() -> None:
    seen: list[dict[str, Any]] = []

    def cost(by_model: Any) -> float:
        seen.append({m: dict(c) for m, c in by_model.items()})
        return sum(c["input_tokens"] for c in by_model.values()) / 100

    limits = usage.UsageLimits(max_cost=1.0, cost=cost, names={"max_cost": "budget"})
    with usage.usage_scope(limits):
        usage.start_llm_call("ingestion", "m", estimated_input_tokens=60)
        with pytest.raises(UsageLimitExceededError, match="budget"):
            usage.start_llm_call("ingestion", "m", estimated_input_tokens=60)
    assert seen[-1]["m"]["input_tokens"] == 120 and seen[-1]["m"]["llm_calls"] == 2
    with pytest.raises(ValueError, match="cost function"):
        usage.UsageLimits(max_cost=1.0)


def test_calls_that_fit_keep_room_for_what_must_follow() -> None:
    assert usage.calls_that_fit([10, 10, 10]) == 3  # no limits
    limits = usage.UsageLimits(max_llm_calls=3, max_input_tokens=100)
    with usage.usage_scope(limits):
        assert usage.calls_that_fit([10, 10, 10], reserve_calls=1) == 2
        assert usage.calls_that_fit([40, 40, 40], reserve_tokens=30) == 1
        assert usage.calls_that_fit([200]) == 0


# --- the model wrapper: the usage callback refuses the call ----------------------


def _ollama_model(server: FakeOllama) -> Any:
    config = Config(
        providers={
            "llm": {"kind": "ollama", "base_url": "http://model-server.test"},
            "embedding": {"kind": "ollama", "base_url": "http://model-server.test"},
        },
        models={"default": "chat-a", "embedding": {"model_id": "e", "dimension": 4}},
    )
    factory = OllamaLanguageModelFactory(config, transport=httpx.MockTransport(server))
    return factory.get_model("chat-a", model_purpose=ModelPurpose.QUERY)


def test_a_refused_call_sends_no_request() -> None:
    # Regression: LangChain logs and swallows a handler's error unless the
    # handler sets raise_error, so a refusal in the callback did not stop
    # the call.
    server = FakeOllama(lambda system, human: "a reply", "unused", 4)
    model = _ollama_model(server)
    with usage.usage_scope(usage.UsageLimits(max_llm_calls=1)):
        model.invoke([HumanMessage(content="first")])
        with pytest.raises(UsageLimitExceededError):
            model.invoke([HumanMessage(content="second")])
    assert len(server.chat_models) == 1


def test_the_callbacks_own_errors_still_never_fail_a_call(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # raise_error raises whatever a handler raises: only the refusal may.
    server = FakeOllama(lambda system, human: "a reply", "unused", 4)
    model = _ollama_model(server)

    def broken(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("accounting bug")

    monkeypatch.setattr(usage_callback, "response_token_usage", broken)
    monkeypatch.setattr(usage_callback, "estimate_token_count", broken)
    with caplog.at_level(logging.WARNING), usage.usage_scope():
        reply = model.invoke([HumanMessage(content="hello")])
    assert reply.content == "a reply"
    # Logged as a digest (log_failure), never its text at WARNING.
    warned = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("UsageCallbackHandler.on_llm_end" in m for m in warned)
    assert all("accounting bug" not in m for m in warned)
    assert UsageCallbackHandler.raise_error is True


# --- the query's limits ---------------------------------------------------------


def test_a_query_can_only_lower_the_configured_limits() -> None:
    configured = QueryLimits(deadline_seconds=180, max_model_calls=10)
    asked = QueryLimits(deadline_seconds=300, max_model_calls=4, max_input_tokens=9)
    assert configured.lowered_by(asked) == QueryLimits(
        deadline_seconds=180, max_model_calls=4, max_input_tokens=9
    )
    assert configured.lowered_by(None) is configured
    assert Config().search.query_limits() == QueryLimits(deadline_seconds=180)
    with pytest.raises(ValidationError):
        RAGInput(query="q", limits={"max_model_calls": 0})
    with pytest.raises(ValidationError):
        RAGInput(query="q", limits={"max_calls": 3})


def _metered(model: _SlowChatModel) -> _SlowChatModel:
    """``model`` with the usage callback the model factories attach."""
    model.callbacks = [UsageCallbackHandler("fake", role="query")]
    return model


def _follow_up(limits: QueryLimits | None = None) -> RAGInput:
    # A follow-up: the rewrite is one call, the answer the next.
    return RAGInput(
        query="And its price?",
        search_strategy=SearchStrategy.SIMPLE,
        enable_query_processing=False,
        history=[
            ConversationTurn(role="user", content="What does Vendor supply?"),
            ConversationTurn(role="assistant", content="Parts."),
        ],
        limits=limits,
    )


async def _calls_unlimited() -> int:
    """The model calls of the follow-up query without limits: the rewrite
    (and its output fix: the fake answers no rewrite JSON) and the answer."""
    model = _metered(_SlowChatModel())
    out = await _chain(Config(), _Factory(model)).ainvoke(_follow_up())
    assert isinstance(out, RAGOutput) and out.answer == _ANSWER
    assert out.metadata["usage"]["llm_calls"] == len(model.started) >= 2
    return len(model.started)


class _SpendingRetriever(_Retriever):
    """A retrieval that spends one model call of the query's own (as a
    third-party strategy's unguarded call): with ``max_model_calls=1`` the
    answer's call is then past the limit."""

    async def aretrieve(self, query: SearchQuery) -> list[Any]:
        usage.start_llm_call("query", "m", estimated_input_tokens=1)
        return await super().aretrieve(query)


def _spending_chain(config: Config, model: _SlowChatModel) -> GraphRAGChain:
    from unified_kg_rag.adapters.local.token_counter import (
        local_token_counter_factory,
    )
    from unified_kg_rag.adapters.providers import Providers
    from unified_kg_rag.domain.models import RetrieverRole

    retriever = _SpendingRetriever()
    providers = Providers(
        config,
        llm_factory=_Factory(model),
        token_counter_factory=local_token_counter_factory(config),
        retriever_builders={RetrieverRole.DOCUMENT: lambda: retriever},
    )
    return GraphRAGChain(config=config, providers=providers)


def _question(limits: QueryLimits | None = None) -> RAGInput:
    return RAGInput(
        query="What does Vendor supply?",
        search_strategy=SearchStrategy.SIMPLE,
        enable_query_processing=False,
        limits=limits,
    )


async def test_a_query_past_its_call_limit_fails_naming_the_limit() -> None:
    calls = await _calls_unlimited()
    assert calls == 3  # the rewrite, its output repair, the answer
    model = _metered(_SlowChatModel())
    chain = _spending_chain(Config(), model)
    with pytest.raises(UsageLimitExceededError) as refused:
        await chain.ainvoke(_question(QueryLimits(max_model_calls=1)))
    assert "limits.max_model_calls" in str(refused.value)
    assert model.started == []  # the answer was not sent
    assert classify(refused.value) is ErrorClass.LIMIT_EXCEEDED
    # The MCP caller sees its own limit's message.
    message, callers = describe_error(refused.value)
    assert callers and message.startswith(f"[limit_exceeded] {refused.value}")


async def test_the_configured_limit_is_named_and_streams_obey_it() -> None:
    calls = await _calls_unlimited()
    model = _metered(_SlowChatModel())
    chain = _spending_chain(Config(search={"query_max_model_calls": 1}), model)
    # A request above the configured limit does not raise it.
    with pytest.raises(UsageLimitExceededError, match="search.query_max_model_calls"):
        await chain.ainvoke(_question(QueryLimits(max_model_calls=calls + 5)))
    with pytest.raises(UsageLimitExceededError):
        _ = [chunk async for chunk in chain.astream(_question())]
    # The same limit on a call that fits answers in full, streamed too.
    roomy = _chain(Config(search={"query_max_model_calls": calls}), _Factory(model))
    assert (await roomy.ainvoke(_follow_up())).answer == _ANSWER
    streamed = [chunk async for chunk in roomy.astream(_follow_up())]
    assert "".join(streamed).strip() == _ANSWER
    # One call short: the rewrite's repair is not asked (it would take the
    # answer's call), and the query answers (W1 review L4).
    tight = _chain(Config(search={"query_max_model_calls": calls - 1}), _Factory(model))
    assert (await tight.ainvoke(_follow_up())).answer == _ANSWER
    streamed = [chunk async for chunk in tight.astream(_follow_up())]
    assert "".join(streamed).strip() == _ANSWER


async def test_ignore_errors_answers_with_the_limit_error_class() -> None:
    config = Config(search={"query_max_model_calls": 1})
    config.processing.ignore_errors = True
    chain = _spending_chain(config, _metered(_SlowChatModel()))
    out = await chain.ainvoke(_question())
    assert out.answer == DEFAULT_ERROR_MESSAGE
    assert out.metadata["error_class"] == "limit_exceeded"


async def test_an_input_token_limit_too_small_for_any_context_abstains() -> None:
    # The rewrite skipped (optional), the answer left no context: it
    # abstains naming the limit instead of failing, and sends nothing.
    model = _metered(_SlowChatModel())
    chain = _chain(Config(), _Factory(model))
    out = await chain.ainvoke(_follow_up(QueryLimits(max_input_tokens=10)))
    assert out.metadata["abstention_reason"] == "query_limit"
    assert model.started == []


async def test_a_lowered_deadline_is_the_querys_and_named() -> None:
    from unified_kg_rag import QueryTimeoutError

    chain = _chain(Config(), _Factory(_SlowChatModel(first_delay=30)))
    with pytest.raises(QueryTimeoutError, match="limits.deadline_seconds"):
        await chain.ainvoke(
            RAGInput(
                query="What does Vendor supply?",
                search_strategy=SearchStrategy.SIMPLE,
                enable_query_processing=False,
                limits=QueryLimits(deadline_seconds=0.2),
            )
        )


def test_the_answer_context_is_cut_to_the_tokens_left() -> None:
    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain.config = Config()
    assert chain._context_budget({"max_tokens": 500}) == 500
    from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
    from unified_kg_rag.adapters.retrieval.base import answer_prompt_tokens

    prompt = answer_prompt_tokens(chain.config) + estimate_token_count("q")
    with usage.usage_scope(usage.UsageLimits(max_input_tokens=prompt + 1000)):
        usage.start_llm_call("query", "m", estimated_input_tokens=200)
        # 800 left after the answer prompt, less the 10% context-window
        # headroom.
        assert chain._context_budget({"max_tokens": None, "query": "q"}) == 720
        assert chain._context_budget({"max_tokens": 500, "query": "q"}) == 500


# --- auto routes only to strategies that fit -----------------------------------


def _router_chain() -> GraphRAGChain:
    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain.config = Config()
    return chain


# A query without query processing: the strategies' own calls alone.
_UNPROCESSED = {"query": "q", "enable_query_processing": False}


def test_auto_drops_the_candidates_whose_fewest_calls_do_not_fit() -> None:
    chain = _router_chain()
    config = chain.config
    candidates = (SearchStrategy.LOCAL, SearchStrategy.GLOBAL, SearchStrategy.DRIFT)
    assert chain._routes_within_limits(candidates, _UNPROCESSED) == (candidates, [])
    # The fewest calls of each, past its query extraction and the answer.
    drift = DriftSearchStrategy.fewest_model_calls(config)
    assert drift == 2 + config.search.drift_search.primer_folds
    assert GlobalSearchStrategy.fewest_model_calls(config) == 1
    config.search.global_search.use_dynamic_selection = True
    assert GlobalSearchStrategy.fewest_model_calls(config) == 2
    with usage.usage_scope(usage.UsageLimits(max_llm_calls=1 + drift)):
        # The router's call, then global's rating and map call and the
        # answer fit; DRIFT's calls and the answer do not.
        assert chain._routes_within_limits(candidates, _UNPROCESSED) == (
            (SearchStrategy.LOCAL, SearchStrategy.GLOBAL),
            ["drift"],
        )
    with usage.usage_scope(usage.UsageLimits(max_llm_calls=2)):
        # One left beside the router's: no router call, local alone.
        assert chain._routes_within_limits(candidates, _UNPROCESSED) == (
            (SearchStrategy.LOCAL,),
            ["global", "drift"],
        )
    iterative = IterativeSearchStrategy.fewest_model_calls(config)
    with usage.usage_scope(usage.UsageLimits(max_llm_calls=iterative)):
        with pytest.raises(UsageLimitExceededError, match="needs at least"):
            chain._routes_within_limits((SearchStrategy.ITERATIVE,), _UNPROCESSED)


# --- the fan-out strategies stop cleanly ----------------------------------------


async def test_global_maps_only_the_batches_that_fit_beside_the_answer() -> None:
    strat = _global_strategy(
        map_batch_size=1,
        map_outputs=[_map_payload((f"point {i}", 90)) for i in range(4)],
    )
    limits = usage.UsageLimits(max_llm_calls=3)
    with usage.usage_scope(limits), degradation_scope() as ledger:
        out = await strat._apply_map_reduce(_communities(4), SearchQuery(query="q"))
    # Two map calls and the answer's: the two most relevant batches.
    assert len(strat.map_rater.batch_inputs) == 2
    points = " ".join(p.content for p in out if p.metadata.get("synthesized"))
    assert "point 0" in points and "point 1" in points and "point 2" not in points
    (entry,) = ledger.entries
    assert entry.step == "global_map"
    assert entry.reason == "query_limit: 2 of 4 calls not started"
    assert entry.error_type == "UsageLimitExceededError"


async def test_global_without_room_for_a_map_call_passes_the_reports() -> None:
    strat = _global_strategy(map_batch_size=1, map_outputs=[])
    reports = _communities(3)
    with usage.usage_scope(usage.UsageLimits(max_llm_calls=1)), degradation_scope():
        out = await strat._apply_map_reduce(reports, SearchQuery(query="q"))
    assert out == reports and strat.map_rater.batch_inputs == []


class _Counted:
    """A strategy's chain whose calls the query's meter counts, as the
    model factories' usage callback does."""

    def __init__(self, chain: Any) -> None:
        self.chain = chain

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.chain.calls

    async def ainvoke(self, inputs: dict[str, Any], config: Any = None) -> Any:
        call = usage.start_llm_call("query", "m", estimated_input_tokens=1)
        call.finish(input_tokens=1, output_tokens=1)
        return await self.chain.ainvoke(inputs, config)


async def test_drift_runs_only_the_actions_that_fit_beside_the_answer() -> None:
    from tests.unit.test_drift_search_logic import _action, _Local
    from tests.unit.test_drift_search_logic import _strategy as _drift_strategy

    local = _Local()
    tree = {f"about report {i}?": (50, [f"next {i}?"]) for i in (0, 2, 4)}
    strat = _drift_strategy(local=local, action=_action(tree), n_depth=2)
    for name in ("hyde", "primer", "action"):
        setattr(strat, name, _Counted(getattr(strat, name)))
    # The expansion, three primer folds, two of the three actions, and the
    # answer.
    with usage.usage_scope(usage.UsageLimits(max_llm_calls=7)):
        result = await strat.asearch(SearchQuery(query="question"))
    assert len(strat.action.calls) == 2
    assert result.metadata["drift_depth"] == 1
    assert result.metadata["drift_open_follow_ups"] >= 1
    (entry,) = result.metadata["degraded"]
    assert (entry["step"], entry["reason"]) == (
        "drift_action",
        "query_limit: 1 of 3 calls not started",
    )


async def test_iterative_stops_its_rounds_at_the_querys_limits() -> None:
    from tests.unit.test_iterative_search import INSUFFICIENT, _query, _Script
    from tests.unit.test_iterative_search import _strategy as _iterative_strategy

    script = _Script(controller=[INSUFFICIENT])
    strategy, _, later, _, _ = _iterative_strategy(script, max_rounds=3)
    strategy.controller = _Counted(strategy.controller)
    # One verdict and the answer fit: no follow-up round, whose verdict
    # would leave the answer no call.
    with usage.usage_scope(usage.UsageLimits(max_llm_calls=2)):
        result = await strategy.asearch(_query())
    meta = result.metadata["iterative"]
    assert (meta["rounds"], meta["stop_reason"]) == (0, "query_limit")
    assert len(script.calls["controller"]) == 1 and later.queries == []
    assert result.metadata["context_sufficient"] is False
    (entry,) = result.metadata["degraded"]
    assert entry["step"] == "iterative_controller"
    assert entry["reason"].startswith("query_limit")

    # Room for a second verdict: the round runs and is judged.
    script = _Script(controller=[INSUFFICIENT])
    strategy, _, later, _, _ = _iterative_strategy(script, max_rounds=3)
    strategy.controller = _Counted(strategy.controller)
    with usage.usage_scope(usage.UsageLimits(max_llm_calls=3)):
        result = await strategy.asearch(_query())
    assert result.metadata["iterative"]["rounds"] == 1
    assert len(script.calls["controller"]) == 2


def test_a_cache_probed_call_counts_only_once_the_cache_missed() -> None:
    # A response cache looks a call up after it started: a hit is no call
    # and holds no room while in flight; a miss is checked then.
    limits = usage.UsageLimits(max_llm_calls=1)
    with usage.usage_scope(limits) as meter:
        hits = [
            usage.start_llm_call(
                "ingestion", "m", estimated_input_tokens=5, probes_cache=True
            )
            for _ in range(3)
        ]
        assert meter.total.llm_calls == 0
        # A call that is no cache's takes the one call meanwhile.
        sent = usage.start_llm_call("ingestion", "m", estimated_input_tokens=5)
        for call in hits:
            usage.cache_probed(hit=True)
            call.served_from_cache()
        missed = usage.start_llm_call(
            "ingestion", "m", estimated_input_tokens=5, probes_cache=True
        )
        with pytest.raises(UsageLimitExceededError):
            usage.cache_probed(hit=False)
        missed.fail()
        sent.finish(input_tokens=5, output_tokens=1)
    assert (meter.total.llm_calls, meter.total.cache_hits) == (1, 3)
    assert meter.total.failed_calls == 0


class _ManyResults:
    """A retriever of many long passages: an answer context far above a
    small input-token limit."""

    async def aindex_limits(self, prefix: str, suffix: str | None) -> Any:
        from unified_kg_rag.domain.models import IndexLimits

        return IndexLimits()

    async def aretrieve(self, query: SearchQuery) -> list[Any]:
        from unified_kg_rag.domain.models import RetrievalResult, SectionType

        text = "Vendor supplies Buyer with parts under the agreement. " * 40
        return [
            RetrievalResult(
                content=f"{i} {text}",
                score=1 - i / 100,
                source=f"doc{i}.txt",
                retriever_type=SectionType.TEXT,
                metadata={"id": f"doc-{i}", "text_unit_ids": []},
            )
            for i in range(40)
        ]


@pytest.mark.parametrize("limit", [1500, 3000, 6000])
async def test_an_input_token_limit_cuts_the_context_so_the_answer_fits(
    limit: int,
) -> None:
    # Regression (W1 review L2): the context was cut to the tokens left less
    # the headroom, but the answer prompt's own templates and question were
    # not subtracted, so under any limit below about 8k tokens the answer
    # call was refused exactly where the cut should have made it fit.
    from unified_kg_rag.adapters.local.token_counter import (
        local_token_counter_factory,
    )
    from unified_kg_rag.adapters.providers import Providers
    from unified_kg_rag.domain.models import RetrieverRole

    config = Config()
    model = _metered(_SlowChatModel())
    retriever = _ManyResults()
    providers = Providers(
        config,
        llm_factory=_Factory(model),
        token_counter_factory=local_token_counter_factory(config),
        retriever_builders={RetrieverRole.DOCUMENT: lambda: retriever},
    )
    chain = GraphRAGChain(config=config, providers=providers)
    out = await chain.ainvoke(
        RAGInput(
            query="What does Vendor supply?",
            search_strategy=SearchStrategy.SIMPLE,
            enable_query_processing=False,
            limits=QueryLimits(max_input_tokens=limit),
        )
    )
    assert out.answer == _ANSWER, out.metadata.get("error")
    assert out.metadata["usage"]["input_tokens"] <= limit


def test_a_fan_out_keeps_the_answer_prompts_tokens_for_the_answer() -> None:
    # Regression (W1 review L2): global's and DRIFT's token reserves counted
    # the answer's context only, so a fan-out could leave the answer no room
    # for its own prompt.
    from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
    from unified_kg_rag.adapters.retrieval.base import answer_prompt_tokens
    from unified_kg_rag.domain.prompts import GlobalMapPrompt

    strat = _global_strategy(map_batch_size=1, map_outputs=[])
    resolved = GlobalMapPrompt.resolve(custom_prompts=strat.config.custom_prompts)
    call = 100 + estimate_token_count(
        resolved.system_prompt_template + resolved.human_prompt_template
    )
    # The answer to a global search's prompt (its strategy's style).
    answer = strat._answer_prompt_tokens()
    assert answer == answer_prompt_tokens(strat.config, SearchStrategy.GLOBAL)
    for limit, fits in ((call + answer, 1), (call + answer - 1, 0)):
        with usage.usage_scope(usage.UsageLimits(max_input_tokens=limit)):
            assert strat._calls_within_limits(GlobalMapPrompt, [100]) == fits


def test_auto_counts_the_query_extractions_each_candidate_reads() -> None:
    # Regression (W1 review L3, D5): a candidate's fewest calls left out the
    # query extraction it reads (local's entities, mix's keywords) and the
    # translation, so under a call limit of 1 auto picked local (two calls)
    # and failed where naive (the answer alone) fitted.
    chain = _router_chain()
    candidates = (SearchStrategy.LOCAL, SearchStrategy.MIX, SearchStrategy.NAIVE)
    state = {"query": "What does Vendor supply?"}
    with usage.usage_scope(usage.UsageLimits(max_llm_calls=1)):
        assert chain._routes_within_limits(candidates, state) == (
            (SearchStrategy.NAIVE,),
            ["local", "mix"],
        )
    # Without query processing no extraction is made: local fits alone.
    with usage.usage_scope(usage.UsageLimits(max_llm_calls=1)):
        assert chain._routes_within_limits(
            candidates, {**state, "enable_query_processing": False}
        ) == ((SearchStrategy.LOCAL,), ["mix", "naive"])
    # A question to translate first costs one call more for every one.
    korean = {"query": "공급업체는 무엇을 공급합니까?"}
    with usage.usage_scope(usage.UsageLimits(max_llm_calls=1)):
        with pytest.raises(UsageLimitExceededError):
            chain._routes_within_limits(candidates, korean)


def test_auto_refuses_naming_the_limit_and_its_value() -> None:
    # Regression (W1 review D3): the refusal named max_model_calls and the
    # calls left, where every other refusal names the setting and its value.
    chain = _router_chain()
    limits = usage.UsageLimits(
        max_llm_calls=2, names={"max_llm_calls": "limits.max_model_calls"}
    )
    with usage.usage_scope(limits):
        usage.start_llm_call("query", "m", estimated_input_tokens=1)
        with pytest.raises(UsageLimitExceededError) as refused:
            chain._routes_within_limits(
                (SearchStrategy.LOCAL,), {"query": "What does Vendor supply?"}
            )
    assert "limits.max_model_calls 2" in str(refused.value)
    assert (refused.value.limit, refused.value.value) == ("limits.max_model_calls", 2)


def test_an_output_repair_never_takes_the_call_kept_for_the_answer() -> None:
    # Regression (W1 review L4): the fan-out reserved the answer's call, but
    # a reply that did not parse asked the output fixer, which took that
    # call, and the answer was refused. A query's limits keep one call for
    # the answer (UsageLimits.reserved_calls): a repair is an optional call
    # (usage.optional_calls), refused when it would take it, and the reply
    # fails to parse as an unrepaired one does.
    from langchain_core.exceptions import OutputParserException
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from tests.fixtures.fakes.chat_models import fake_model_info
    from unified_kg_rag.adapters.llm.chain_factory import create_json_output_parser

    class _Fixer:
        def get_model(self, model_id: str, **_: Any) -> Any:
            return FakeListChatModel(
                responses=['{"fixed": true}'],
                callbacks=[UsageCallbackHandler(model_id, role="query")],
            )

        def get_model_info(self, model_id: str) -> Any:
            return fake_model_info(model_id)

    parser = create_json_output_parser(_Fixer(), "fixer")  # type: ignore[arg-type]
    limits = usage.UsageLimits(max_llm_calls=3, reserved_calls=1)
    with usage.usage_scope(limits) as meter:
        usage.start_llm_call("query", "m", estimated_input_tokens=1)
        usage.start_llm_call("query", "m", estimated_input_tokens=1)
        # One call left: the answer's.
        with pytest.raises(OutputParserException):
            parser.parse("no json here")
        assert meter.total.llm_calls == 2
    with usage.usage_scope(limits) as meter:
        usage.start_llm_call("query", "m", estimated_input_tokens=1)
        # Two left: the repair may take one.
        assert parser.parse("no json here") == {"fixed": True}
        assert meter.total.llm_calls == 2


def test_a_model_without_the_usage_callback_is_warned_of_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Regression (W1 review L9): a custom LLMFactoryPort whose models carry
    # no UsageCallbackHandler made every query limit and run budget count
    # nothing, without a word.
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from tests.fixtures.fakes.chat_models import fake_model_info
    from unified_kg_rag.adapters.llm.chain_factory import setup_chain
    from unified_kg_rag.domain.prompts import AnswerGenerationPrompt

    class _Bare:
        def get_model(self, model_id: str, **_: Any) -> Any:
            return FakeListChatModel(responses=["a"])

        def get_model_info(self, model_id: str) -> Any:
            return fake_model_info(model_id)

    class _Metered(_Bare):
        def get_model(self, model_id: str, **_: Any) -> Any:
            return FakeListChatModel(
                responses=["a"],
                callbacks=[UsageCallbackHandler(model_id, role="query")],
            )

    from langchain_core.output_parsers import StrOutputParser

    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            setup_chain(_Bare(), "m", AnswerGenerationPrompt, StrOutputParser())  # type: ignore[arg-type]
        setup_chain(_Metered(), "m", AnswerGenerationPrompt, StrOutputParser())  # type: ignore[arg-type]
    warned = [
        r.getMessage()
        for r in caplog.records
        if "UsageCallbackHandler" in r.getMessage()
    ]
    assert len(warned) == 1 and "_Bare" in warned[0]


async def test_global_maps_what_an_input_token_limit_leaves_beside_the_answer() -> None:
    # Regression (W1 verification 1): global reserved the config ceilings
    # of what its answer may read (max_map_reduce_tokens, evidence_tokens:
    # 12k) beside every map call, so under any smaller limit it mapped
    # nothing; the answer's context is cut to what is left anyway, so a
    # fan-out keeps only the answer prompt and the least context for it.
    from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
    from unified_kg_rag.domain.prompts import GlobalMapPrompt

    strat = _global_strategy(
        map_batch_size=1,
        map_outputs=[_map_payload((f"point {i}", 90)) for i in range(4)],
    )
    reports = _communities(4)
    batches, _ = strat._map_batches(reports)
    call = strat._prompt_tokens(GlobalMapPrompt) + estimate_token_count(
        strat._format_reports(batches[0])
    )
    limit = 2 * call + strat._answer_prompt_tokens() + strat._min_context_tokens
    assert limit < strat.global_search_config.max_map_reduce_tokens
    with usage.usage_scope(usage.UsageLimits(max_input_tokens=limit)):
        await strat._apply_map_reduce(reports, SearchQuery(query="q"))
    assert len(strat.map_rater.batch_inputs) == 2


def test_auto_makes_no_router_call_an_input_token_limit_cannot_take() -> None:
    # Regression (W1 verification 3): auto weighed a call limit but not an
    # input-token one, so at a small token limit its router call was refused
    # and the query failed where any explicit strategy degraded; the router
    # call is optional too: without room for it beside the answer, auto
    # takes its fallback as with one candidate.
    from unified_kg_rag.adapters.retrieval.base import answer_prompt_tokens

    chain = _router_chain()
    candidates = (SearchStrategy.LOCAL, SearchStrategy.MIX, SearchStrategy.NAIVE)
    state = {"query": "What does Vendor supply?", "enable_query_processing": False}
    roomy = 50_000
    with usage.usage_scope(usage.UsageLimits(max_input_tokens=roomy)):
        assert chain._routes_within_limits(candidates, state) == (candidates, [])
    small = answer_prompt_tokens(chain.config) + 200
    with usage.usage_scope(usage.UsageLimits(max_input_tokens=small)):
        assert chain._routes_within_limits(candidates, state) == (
            (SearchStrategy.LOCAL,),
            ["mix", "naive"],
        )


# --- what an optional call leaves the answer (W1 verification residuals) ----


def test_an_optional_call_leaves_the_answer_its_input_tokens() -> None:
    # Regression (W1 verification, low 1): an optional call kept the call
    # for the answer but not its input tokens, so on a Korean question at a
    # small max_input_tokens the translation and extraction took the tokens
    # the answer's prompt needed and the query failed on the answer.
    limits = usage.UsageLimits(
        max_input_tokens=1_000, reserved_calls=1, reserved_tokens=600
    )
    with usage.usage_scope(limits) as meter:
        with usage.optional_calls():
            # 300 + 600 kept: fits.
            usage.start_llm_call("query", "m", estimated_input_tokens=300)
            with pytest.raises(usage.OptionalCallSkipped) as skipped:
                usage.start_llm_call("query", "m", estimated_input_tokens=200)
        assert skipped.value.limit == "max_input_tokens"
        assert "keeps for the answer" in str(skipped.value)
        # The answer itself may take them.
        usage.start_llm_call("query", "m", estimated_input_tokens=600)
        assert meter.total.input_tokens == 900


def test_a_query_keeps_its_answer_s_prompt_and_question_tokens() -> None:
    from unified_kg_rag.adapters.retrieval.base import answer_prompt_tokens

    config = Config()
    chain = _chain(config, _Factory(_SlowChatModel()))
    rag_input = RAGInput(
        query="What does Vendor supply?", limits=QueryLimits(max_input_tokens=5_000)
    )
    limits = chain._usage_limits(rag_input)
    assert limits is not None
    kept = limits.reserved_tokens
    assert kept > answer_prompt_tokens(config)  # the prompt, question, context
    # Without an input-token limit nothing is kept for it.
    calls_only = chain._usage_limits(
        RAGInput(query="q", limits=QueryLimits(max_model_calls=3))
    )
    assert calls_only is not None and calls_only.reserved_tokens == 0


async def test_a_follow_up_at_one_call_searches_the_question_as_written() -> None:
    # Regression (W1 verification, low 2): the history rewrite was no
    # optional call, so a follow-up at max_model_calls=1 failed on its
    # answer. It answers now, the follow-up searched as written and the
    # rewrite recorded as degraded by the limit.
    model = _metered(_SlowChatModel())
    chain = _chain(Config(), _Factory(model))
    out = await chain.ainvoke(_follow_up(QueryLimits(max_model_calls=1)))
    assert out.answer == _ANSWER
    assert len(model.started) == 1  # the answer alone
    (entry,) = out.metadata["degraded"]
    assert entry["step"] == "follow_up_rewrite"
    assert entry["reason"].startswith("query_limit")
    assert out.processed_query is not None
    assert out.processed_query.final_query == "And its price?"


def test_an_optional_call_past_a_limit_itself_is_skipped_not_refused() -> None:
    # The meter checked the limit first: an optional call past it raised the
    # limit's refusal, which the step making it does not absorb (the
    # follow-up rewrite failed a query whose answer fitted).
    limits = usage.UsageLimits(
        max_input_tokens=100, reserved_calls=1, reserved_tokens=50
    )
    with usage.usage_scope(limits), usage.optional_calls():
        with pytest.raises(usage.OptionalCallSkipped):
            usage.start_llm_call("query", "m", estimated_input_tokens=500)
