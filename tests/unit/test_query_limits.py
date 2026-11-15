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

import json
import logging
from typing import Any

import httpx
import pytest
from langchain_core.messages import HumanMessage
from pydantic import Field, ValidationError

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


@pytest.mark.parametrize(("outer", "inner"), [(5, 3), (3, 10)])
def test_nested_scopes_leave_the_fewest_calls_any_of_them_allows(
    outer: int, inner: int
) -> None:
    # Whichever scope is the tighter, inner or enclosing, decides.
    with usage.usage_scope(usage.UsageLimits(max_llm_calls=outer)):
        with usage.usage_scope(usage.UsageLimits(max_llm_calls=inner)):
            usage.start_llm_call("query", "m", estimated_input_tokens=1)
            left = usage.remaining()
    assert left.llm_calls == min(outer, inner) - 1
    assert left.call_limit_value == min(outer, inner)


def test_a_call_that_reaches_the_input_token_limit_exactly_is_sent() -> None:
    # The limit is the most a query may read: reaching it is within it.
    with usage.usage_scope(usage.UsageLimits(max_input_tokens=100)) as meter:
        usage.start_llm_call("query", "m", estimated_input_tokens=60)
        usage.start_llm_call("query", "m", estimated_input_tokens=40)
        assert usage.remaining().input_tokens == 0
        with pytest.raises(UsageLimitExceededError, match="to 101, past its limit"):
            usage.start_llm_call("query", "m", estimated_input_tokens=1)
    assert meter.snapshot()["llm_calls"] == 2


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


@pytest.mark.parametrize("path", ["ainvoke", "astream", "abatch", "chain", "stream"])
async def test_a_refusal_stops_the_call_on_every_path_the_queries_take(
    path: str,
) -> None:
    # Enforcement rests on LangChain raising the usage callback's refusal
    # (raise_error): pinned on every call path the queries use, sync and
    # async, inline and through the executor, so a LangChain release (or
    # Python version) that swallows it fails here instead of a query
    # running past its limit, or an optional call hanging its step.
    import asyncio

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    server = FakeOllama(lambda system, human: "a reply", "unused", 4)
    model = _ollama_model(server)
    prompt = [HumanMessage(content="what does the Vendor ship")]

    async def call() -> Any:
        if path == "ainvoke":
            return await model.ainvoke(prompt)
        if path == "astream":
            return [chunk async for chunk in model.astream(prompt)]
        if path == "abatch":
            return await model.abatch([prompt])
        if path == "chain":
            chain = ChatPromptTemplate.from_messages([("human", "{q}")]) | model
            return await (chain | StrOutputParser()).ainvoke({"q": "ships?"})
        return await asyncio.to_thread(lambda: list(model.stream(prompt)))

    limits = usage.UsageLimits(max_llm_calls=2, reserved_calls=1)
    with usage.usage_scope(limits):
        usage.start_llm_call("query", "m", estimated_input_tokens=1)
        with usage.optional_calls():
            with pytest.raises(usage.OptionalCallSkipped):
                await asyncio.wait_for(call(), timeout=10)
        usage.start_llm_call("query", "m", estimated_input_tokens=1)
        with pytest.raises(UsageLimitExceededError):
            await asyncio.wait_for(call(), timeout=10)
    assert server.chat_models == []


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
    chain = _chain(Config(), _Factory(_SlowChatModel()))
    assert chain._context_budget({"max_tokens": 500}) == (500, False)
    from unified_kg_rag.adapters.retrieval.base import answer_call_tokens

    prompt = answer_call_tokens(chain.config, None, "q", 0)
    with usage.usage_scope(usage.UsageLimits(max_input_tokens=prompt + 1000)):
        usage.start_llm_call("query", "m", estimated_input_tokens=200)
        # 800 left after the answer prompt, less the 10% context-window
        # headroom: set by the limit.
        assert chain._context_budget({"max_tokens": None, "query": "q"}) == (720, True)
        assert chain._context_budget({"max_tokens": 500, "query": "q"}) == (500, False)


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
    # rv33: and neither counted the question each call and the answer carry.
    from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
    from unified_kg_rag.adapters.retrieval.base import answer_call_tokens
    from unified_kg_rag.domain.prompts import GlobalMapPrompt

    strat = _global_strategy(map_batch_size=1, map_outputs=[])
    query = SearchQuery(query="공급업체는 어떤 부품을 언제 구매자에게 공급합니까?")
    resolved = GlobalMapPrompt.resolve(custom_prompts=strat.config.custom_prompts)
    question = estimate_token_count(query.query)
    call = (
        100
        + question
        + estimate_token_count(
            resolved.system_prompt_template + resolved.human_prompt_template
        )
    )
    # The answer to a global search's prompt (its strategy's style), its
    # question and the least context.
    answer = strat._answer_reserve(query)
    assert answer == answer_call_tokens(
        strat.config, SearchStrategy.GLOBAL, query.query
    )
    assert answer > strat._min_context_tokens + question
    for limit, fits in ((call + answer, 1), (call + answer - 1, 0)):
        with usage.usage_scope(usage.UsageLimits(max_input_tokens=limit)):
            assert strat._calls_within_limits(GlobalMapPrompt, query, [100]) == fits
    # A search that is not answered keeps nothing for an answer.
    unanswered = query.model_copy(update={"metadata": {"answered": False}})
    with usage.usage_scope(usage.UsageLimits(max_input_tokens=call)):
        assert strat._calls_within_limits(GlobalMapPrompt, unanswered, [100]) == 1


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
    # A question to translate first costs one call more for every one: none
    # fits whole, so auto routes as if its optional calls (the translation
    # and extractions, skipped under the limit) were not made (W2 review
    # S5: it refused a query that local answered with one call).
    korean = {"query": "공급업체는 무엇을 공급합니까?"}
    with usage.usage_scope(usage.UsageLimits(max_llm_calls=1)):
        routed, dropped = chain._routes_within_limits(candidates, korean)
    assert len(routed) == 1 and sorted([*dropped, routed[0].value]) == [
        "local",
        "mix",
        "naive",
    ]


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
                (SearchStrategy.DRIFT,), {"query": "What does Vendor supply?"}
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
    query = SearchQuery(query="q")
    call = strat._call_tokens(
        GlobalMapPrompt, query, estimate_token_count(strat._format_reports(batches[0]))
    )
    limit = 2 * call + strat._answer_reserve(query)
    assert limit < strat.global_search_config.max_map_reduce_tokens
    with usage.usage_scope(usage.UsageLimits(max_input_tokens=limit)):
        await strat._apply_map_reduce(reports, query)
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
    from unified_kg_rag.adapters.retrieval.base import (
        answer_call_tokens,
        answer_prompt_tokens,
    )

    config = Config()
    chain = _chain(config, _Factory(_SlowChatModel()))
    rag_input = RAGInput(
        query="What does Vendor supply?", limits=QueryLimits(max_input_tokens=5_000)
    )
    limits = chain._usage_limits(rag_input, search=False)
    assert limits is not None
    kept = limits.reserved_tokens
    assert kept > answer_prompt_tokens(config)  # the prompt, question, context
    assert limits.reserved_calls == 1
    # Without an input-token limit nothing is kept for it.
    calls_only = chain._usage_limits(
        RAGInput(query="q", limits=QueryLimits(max_model_calls=3)), search=False
    )
    assert calls_only is not None and calls_only.reserved_tokens == 0
    # A search makes no answer: nothing is kept for one (rv33).
    searched = chain._usage_limits(rag_input, search=True)
    assert searched is not None
    assert (searched.reserved_calls, searched.reserved_tokens) == (0, 0)
    # DRIFT keeps the one action it cannot answer without beside the
    # answer, and under an input-token limit its least input too (rv34: an
    # optional extraction spent the tokens the action needed, and the query
    # abstained after paying for it); a search keeps the action alone.
    drifting = rag_input.model_copy(update={"search_strategy": SearchStrategy.DRIFT})
    (action,) = DriftSearchStrategy.answering_calls(config, rag_input.query)
    assert action > config.search.token_manager.min_truncated_section_tokens
    drift = chain._usage_limits(drifting, search=False)
    assert drift is not None and drift.reserved_calls == 2
    answer = answer_call_tokens(config, SearchStrategy.DRIFT, rag_input.query)
    assert drift.reserved_tokens == answer + action
    drift_search = chain._usage_limits(drifting, search=True)
    assert drift_search is not None
    assert (drift_search.reserved_calls, drift_search.reserved_tokens) == (1, action)


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


@pytest.mark.parametrize(
    ("limits", "call"),
    [
        (usage.UsageLimits(max_llm_calls=1), 1),
        (usage.UsageLimits(max_input_tokens=100), 500),
    ],
    ids=["calls", "tokens"],
)
def test_an_optional_call_past_a_limit_keeping_nothing_is_skipped(
    limits: usage.UsageLimits, call: int
) -> None:
    # rv33: a scope keeping nothing for an answer (a search's) refused an
    # optional call past its limit with the limit's own error, which no
    # optional step absorbs: a search's output repair failed the search.
    with usage.usage_scope(limits), usage.optional_calls():
        if limits.max_llm_calls is not None:
            usage.start_llm_call("query", "m", estimated_input_tokens=1)
        with pytest.raises(usage.OptionalCallSkipped, match="past"):
            usage.start_llm_call("query", "m", estimated_input_tokens=call)


async def test_a_streamed_answer_counts_one_call_as_an_unstreamed_one() -> None:
    # Regression (W2 review S4): a streamed answer counted two calls (and an
    # estimate), so its usage and cost read twice ainvoke's.
    from unified_kg_rag import QueryResult

    model = _metered(_SlowChatModel())
    chain = _chain(Config(), _Factory(model))
    unstreamed = (await chain.ainvoke(_question())).metadata["usage"]
    events = [event async for event in chain.aquery_events(_question())]
    assert isinstance(events[-1], QueryResult)
    streamed = events[-1].output.metadata["usage"]
    assert unstreamed["llm_calls"] == streamed["llm_calls"] == 1
    assert streamed["estimated_calls"] == unstreamed["estimated_calls"]


# --- a DRIFT search under an input-token limit (rv34) ---------------------------


class _DriftModel(_SlowChatModel):
    """Answers each DRIFT prompt as the model would: a report for the
    expansion, a primer answer with follow-ups, an action answer."""

    actions: list[bool] = Field(default_factory=list)

    async def _agenerate(self, messages: list[Any], *a: Any, **k: Any) -> Any:
        from langchain_core.messages import AIMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        system = str(messages[0].content)
        if "DRIFT action step" in system:
            self.actions.append(True)
            reply = json.dumps(
                {
                    "response": "Vendor ships parts.",
                    "score": 60,
                    "follow_up_queries": [],
                }
            )
        elif "DRIFT search primer" in system:
            reply = json.dumps(
                {
                    "intermediate_answer": "Vendor supplies Buyer.",
                    "score": 50,
                    "follow_up_queries": ["Which parts does Vendor ship to Buyer?"],
                }
            )
        else:
            reply = "Vendor supplies parts to Buyer under the agreement."
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=reply))]
        )


class _MixedLocal:
    """A follow-up's local search: one section of each type, of the sizes
    the demo corpus's are (a short relationship, a long entity, two
    passages, a long report)."""

    async def asearch(self, query: SearchQuery, config: Any = None) -> Any:
        from unified_kg_rag.domain.models import RetrievalResult, SearchResult

        def words(n: int) -> str:
            return " ".join(f"part{i}" for i in range(n))

        results = [
            RetrievalResult(
                content=words(size),
                score=score,
                source=f"{kind}-1",
                retriever_type=kind,
            )
            for kind, size, score in (
                ("relationship", 30, 0.9),
                ("entity", 120, 0.8),
                ("text", 150, 0.7),
                ("text", 160, 0.6),
                ("community", 200, 0.5),
            )
        ]
        return SearchResult(
            query=query,
            results=results,
            total_results=len(results),
            search_strategy="local_search",
            processing_time=0.0,
        )


def _drift(model: _DriftModel) -> DriftSearchStrategy:
    from tests.unit.test_drift_search_logic import _Reports
    from unified_kg_rag.adapters.local.token_counter import (
        local_token_counter_factory,
    )
    from unified_kg_rag.adapters.providers import Providers

    config = Config()
    config.providers.rerank.kind = "none"
    providers = Providers(
        config,
        llm_factory=_Factory(model),
        token_counter_factory=local_token_counter_factory(config),
    )
    strategy = DriftSearchStrategy(
        config, {"document": _Reports(count=4)}, providers=providers
    )
    strategy.local_search = _MixedLocal()  # type: ignore[assignment]
    return strategy


async def test_a_drift_search_under_any_input_token_limit_never_fails() -> None:
    # Regression (rv34, from #759): a search keeps nothing for an answer, so
    # an action's share had no slack, and the context built for it read past
    # the share (the token manager lent a type's unused share twice; the
    # pieces are counted apart from the whole the meter estimates). Every
    # action was refused and the search failed with limit_exceeded at limits
    # where lower ones succeeded. Now the built context keeps to its share,
    # and an action the meter still refuses is a recorded limit stop.
    from unified_kg_rag.adapters.retrieval.base import ANSWERED_KEY

    sent: list[int] = []
    for limit in range(400, 4000, 50):
        model = _metered(_DriftModel())
        strategy = _drift(model)
        query = SearchQuery(
            query="What does Vendor supply to Buyer?",
            metadata={ANSWERED_KEY: False},
        )
        with (
            degradation_scope(),
            usage.usage_scope(usage.UsageLimits(max_input_tokens=limit)) as meter,
        ):
            result = await strategy.asearch(query)
        assert meter.total.input_tokens <= limit
        sent.append(len(model.actions))
        # What the follow-ups' local searches found is kept.
        assert result.results or not sent[-1], limit
    # A higher limit never sends fewer actions.
    assert sent == sorted(sent), sent
    assert sent[0] < sent[-1]


async def test_a_drift_action_the_limits_refuse_when_built_is_a_limit_stop() -> None:
    # The plan counts an action's prompt in pieces, the meter the whole
    # prompt: one the meter still refuses (not sent) is left out as an
    # action that did not fit, its follow-up's evidence kept, not a failure
    # of the search (rv34).
    from tests.unit.test_drift_search_logic import _Local
    from tests.unit.test_drift_search_logic import _strategy as _drift_strategy

    refused = UsageLimitExceededError(
        "A model call would take max_input_tokens to 1202, past its limit 1200",
        limit="max_input_tokens",
        value=1200,
    )
    strat = _drift_strategy(local=_Local(), action=lambda inputs: refused, n_depth=2)
    with degradation_scope():
        result = await strat.asearch(SearchQuery(query="question"))
    assert result.metadata["drift_depth"] == 1  # no round after it
    assert result.metadata["drift_llm_calls"] == 1 + 3  # the expansion, primers
    assert any(r.source.startswith("t:") for r in result.results)
    steps = {(e["step"], e["reason"]) for e in result.metadata["degraded"]}
    assert ("drift_action", "query_limit: 1 of 1 calls not started") in steps
    # Another error of an action still fails a search none of whose actions
    # answered.
    strat = _drift_strategy(
        local=_Local(), action=lambda inputs: ValueError("boom"), n_depth=2
    )
    with degradation_scope(), pytest.raises(ValueError):
        await strat.asearch(SearchQuery(query="question"))


async def test_a_drift_action_whose_context_fits_nothing_is_a_limit_stop() -> None:
    # Regression (rv36): under an input-token limit an action whose
    # evidence no section of fits its share was still sent, asked to
    # answer from an empty context. It is left out as an action that did
    # not fit, its follow-up's evidence kept; without a limit the action
    # reads what its budget kept, as before.
    from tests.unit import test_drift_search_logic as drift

    class _Nothing(drift._Tokens):
        def optimize_context(self, results, query, max_tokens, estimated_within=None):
            return drift._Optimized(results[:0])

    strat = drift._strategy(local=drift._Local(), n_depth=2)
    strat.token_manager = _Nothing()  # type: ignore[assignment]
    with (
        degradation_scope(),
        usage.usage_scope(usage.UsageLimits(max_input_tokens=100_000)),
    ):
        result = await strat.asearch(SearchQuery(query="question"))
    assert strat.action.calls == []
    assert result.metadata["drift_depth"] == 1  # no round after it
    assert any(r.source.startswith("t:") for r in result.results)
    steps = {(e["step"], e["reason"]) for e in result.metadata["degraded"]}
    assert ("drift_action", "query_limit: 1 of 1 calls not started") in steps
    strat = drift._strategy(local=drift._Local(), n_depth=1)
    strat.token_manager = _Nothing()  # type: ignore[assignment]
    await strat.asearch(SearchQuery(query="question"))
    assert strat.action.calls and all(c["context"] == "" for c in strat.action.calls)


def _limited_context(chain: GraphRAGChain, results: list[Any], limit: int) -> Any:
    """The answer context ``chain`` builds from ``results`` under an
    input-token ``limit``, and what it recorded as degraded."""
    from unified_kg_rag.application.retrieval.rag_chain import ProcessedQuery
    from unified_kg_rag.domain.models import SearchResult

    question = "What does Vendor supply?"
    state = {
        "query": question,
        "max_tokens": None,
        "resolved_strategy": SearchStrategy.SIMPLE,
        "processed_query": ProcessedQuery(
            original_query=question, final_query=question
        ),
        "search_results": SearchResult(
            query=SearchQuery(query=question),
            results=results,
            total_results=len(results),
            search_strategy="simple",
            processing_time=0.0,
        ),
    }
    with (
        degradation_scope() as ledger,
        usage.usage_scope(usage.UsageLimits(max_input_tokens=limit)),
    ):
        optimized = chain._optimize_context(state)
    return optimized, ledger


def _local_chain() -> GraphRAGChain:
    from unified_kg_rag.adapters.local.token_counter import (
        local_token_counter_factory,
    )
    from unified_kg_rag.adapters.providers import Providers

    config = Config()
    providers = Providers(
        config,
        llm_factory=_Factory(_SlowChatModel()),
        token_counter_factory=local_token_counter_factory(config),
    )
    return GraphRAGChain(config=config, providers=providers)


@pytest.mark.parametrize("content", ["Vendor {i}", "벤더 {i} 공급"])
def test_many_short_sections_never_starve_the_answer_call(content: str) -> None:
    # Regression (rv34): each section is counted apart and rounded down,
    # the blank lines between them uncounted, so a context of 400 short
    # passages estimated past its budget by more than the answer's 10%
    # headroom (173 tokens on Korean ones): the meter refused the answer
    # call though a smaller context fitted.
    from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
    from unified_kg_rag.adapters.retrieval.base import answer_call_tokens
    from unified_kg_rag.domain.models import RetrievalResult, SectionType

    chain = _local_chain()
    results = [
        RetrievalResult(
            content=content.format(i=i),
            score=1 - i / 1000,
            source=f"doc{i}.txt",
            retriever_type=SectionType.TEXT,
        )
        for i in range(400)
    ]
    question = "What does Vendor supply?"
    for limit in range(1500, 8000, 37):
        optimized, _ = _limited_context(chain, results, limit)
        assert optimized.sections
        built = chain.token_manager.build_context_string(optimized)
        call = answer_call_tokens(
            chain.config, SearchStrategy.SIMPLE, question, estimate_token_count(built)
        )
        assert call <= limit, limit


@pytest.mark.parametrize("ratio", [0.5, 1.0, 1.8])
def test_the_answer_context_fills_the_room_at_any_counter_ratio(ratio: float) -> None:
    # Regression (rv35): the room an input-token limit leaves is in the
    # estimate's tokens (the meter's) but was packed as the counter's, so a
    # counter counting 1.8 times the estimate left the answer about half
    # the context it had room for.
    from unittest import mock

    from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
    from unified_kg_rag.adapters.providers import Providers
    from unified_kg_rag.domain.models import RetrievalResult, SectionType

    counter = mock.Mock()
    counter.count_tokens.side_effect = lambda text: max(
        1, int(estimate_token_count(text) * ratio)
    )
    config = Config()
    chain = GraphRAGChain(
        config=config,
        providers=Providers(
            config,
            llm_factory=_Factory(_SlowChatModel()),
            token_counter_factory=lambda *_, **__: counter,
        ),
    )
    results = [
        RetrievalResult(
            content="공급업체는 구매자에게 라이다 장치를 납품한다. " * (1 + i % 4),
            score=1 - i / 100,
            source=f"doc{i}.txt",
            retriever_type=SectionType.TEXT,
        )
        for i in range(200)
    ]
    for limit in (3000, 6000):
        with usage.usage_scope(usage.UsageLimits(max_input_tokens=limit)):
            room, limited = chain._context_budget(
                {
                    "max_tokens": None,
                    "resolved_strategy": SearchStrategy.SIMPLE,
                    "query": "What does Vendor supply?",
                }
            )
        assert limited
        optimized, _ = _limited_context(chain, results, limit)
        built = estimate_token_count(
            chain.token_manager.build_context_string(optimized)
        )
        assert 0.9 * room <= built <= room, (limit, room, built)


def test_a_section_the_limit_cuts_short_is_listed_as_cut() -> None:
    # Regression (rv34): a section the input-token limit truncated but kept
    # was not recorded, so the answer read less while query_limited said
    # the limit cut nothing.
    from unified_kg_rag.adapters.retrieval.base import answer_call_tokens
    from unified_kg_rag.domain.models import RetrievalResult, SectionType

    chain = _local_chain()
    text = "Vendor supplies Buyer with parts under the agreement. " * 30
    results = [
        RetrievalResult(
            content=text, score=0.9, source="doc.txt", retriever_type=SectionType.TEXT
        )
    ]
    prompt = answer_call_tokens(
        chain.config, SearchStrategy.SIMPLE, "What does Vendor supply?", 0
    )
    optimized, ledger = _limited_context(chain, results, prompt + 300)
    assert (optimized.sections_included, optimized.sections_excluded) == (1, 0)
    assert optimized.sections_truncated == 1
    (entry,) = ledger.entries
    assert (entry.step, entry.reason) == (
        "context_optimization",
        "query_limit: 1 of 1 sections cut",
    )
