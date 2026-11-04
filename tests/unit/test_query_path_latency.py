# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The query path's round trips, repeat cost and output reservations (AWS-free).

- auto starts the extractions its routable strategies read alongside the
  router (``search.auto_speculative_extraction``), with the router's
  decision unchanged;
- query entities and keywords are cached per question, language, model and
  prompt (``search.query_extraction_cache_size``);
- the short-answer query calls ask for a small max_tokens
  (``search.output_ceilings``) that still leaves the thinking budget;
- the answer prompt is chosen per strategy (``search.answer_prompt_style``).
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any

import httpx
import pytest
from langchain_core.runnables import Runnable
from pydantic import ValidationError

import unified_kg_rag.adapters.search_strategies  # noqa: F401 (registers strategies)
from tests.fixtures.fakes.boto import FakeBotoSession
from tests.fixtures.fakes.model_servers import FakeOllama
from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory
from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
from unified_kg_rag.application.bootstrap import build_providers
from unified_kg_rag.application.retrieval.rag_chain import (
    _LEFTOVER_EXTRACTIONS,
    AnswerGenerationPrompt,
    ComprehensiveAnswerPrompt,
    EntityExtractionPrompt,
    GraphRAGChain,
    KeywordsExtractionPrompt,
    StrategySelectionPrompt,
    TranslationPrompt,
)
from unified_kg_rag.domain.models import Config, LanguageModelId, SearchStrategy
from unified_kg_rag.ports.model_factory import ceiled_output_tokens
from unified_kg_rag.shared.utils.concurrency import discard

pytestmark = pytest.mark.unit

QUESTION = "What does the Vendor owe the Buyer?"
_KEYWORDS = (
    '{"high_level_keywords": ["supply terms"], "low_level_keywords": ["Vendor"]}'
)


class _Chain(Runnable[dict[str, Any], Any]):
    def __init__(self, factory: _Factory, prompt_class: type, kwargs: dict) -> None:
        self.factory = factory
        self.prompt_class = prompt_class
        self.kwargs = kwargs

    def invoke(self, inputs: dict[str, Any], config: Any = None, **_: Any) -> Any:
        raise NotImplementedError

    async def ainvoke(  # type: ignore[override]
        self, inputs: dict[str, Any], config: Any = None, **_: Any
    ) -> Any:
        f = self.factory
        f.calls.append((self.prompt_class, inputs))
        f.started[self.prompt_class].set()
        try:
            if self.prompt_class is StrategySelectionPrompt and f.router_waits_for:
                # Deadlocks (and times out) unless the extraction started
                # before the router answered.
                await asyncio.wait_for(f.started[f.router_waits_for].wait(), 2)
            if self.prompt_class in f.hang:
                await asyncio.Event().wait()
            await asyncio.sleep(f.latency.get(self.prompt_class, 0))
        except asyncio.CancelledError:
            f.cancelled.append(self.prompt_class)
            raise
        f.finished.append(self.prompt_class)
        output = f.outputs[self.prompt_class]
        if isinstance(output, Exception):
            raise output
        return output


class _Factory:
    """Stands in for ``_get_chain_for_prompt``: canned outputs, a log of the
    calls, their start events and cancellations."""

    def __init__(self, route: str = "local") -> None:
        self.outputs: dict[type, Any] = {
            StrategySelectionPrompt: route,
            EntityExtractionPrompt: ["Vendor", "Buyer"],
            KeywordsExtractionPrompt: _KEYWORDS,
            TranslationPrompt: "translated",
        }
        self.calls: list[tuple[type, dict[str, Any]]] = []
        self.kwargs: dict[type, dict[str, Any]] = {}
        self.started: dict[type, asyncio.Event] = {}
        self.latency: dict[type, float] = {}
        self.hang: set[type] = set()
        self.cancelled: list[type] = []
        self.finished: list[type] = []
        self.router_waits_for: type | None = None

    def get_chain(self, prompt_class: type, parser: Any, **kwargs: Any) -> _Chain:
        self.kwargs[prompt_class] = kwargs
        self.started.setdefault(prompt_class, asyncio.Event())
        return _Chain(self, prompt_class, kwargs)

    @property
    def counts(self) -> Counter:
        return Counter(cls for cls, _ in self.calls)


def _config(**search: Any) -> Config:
    config = Config(search=search)
    # Same-language corpus: no translation call unless a test asks for one.
    translation = config.processing.translation
    translation.source_language = translation.target_language
    return config


def _chain(config: Config, factory: _Factory) -> GraphRAGChain:
    chain = GraphRAGChain(config=config)
    chain._get_chain_for_prompt = factory.get_chain  # type: ignore[assignment]
    return chain


async def _route(chain: GraphRAGChain, strategy: str = "auto", **state: Any) -> dict:
    for prompt_class in (
        StrategySelectionPrompt,
        EntityExtractionPrompt,
        KeywordsExtractionPrompt,
        TranslationPrompt,
    ):
        chain._get_chain_for_prompt(prompt_class, None)  # type: ignore[arg-type]
    return await chain._route_and_process_step(
        {"query": QUESTION, "search_strategy": SearchStrategy(strategy), **state}
    )


# --- auto: extraction alongside the router -----------------------------------


async def test_auto_extracts_while_the_router_runs() -> None:
    # The router answers only once the entity extraction has started: with
    # the extraction after the router this times out.
    factory = _Factory(route="local")
    factory.router_waits_for = EntityExtractionPrompt
    state = await _route(_chain(_config(), factory))
    assert state["resolved_strategy"] is SearchStrategy.LOCAL
    assert state["processed_query"].entities == ["Vendor", "Buyer"]
    assert factory.counts[StrategySelectionPrompt] == 1
    assert factory.counts[EntityExtractionPrompt] == 1


async def test_without_speculation_the_extraction_waits_for_the_route() -> None:
    factory = _Factory(route="local")
    state = await _route(_chain(_config(auto_speculative_extraction=False), factory))
    assert [cls for cls, _ in factory.calls] == [
        StrategySelectionPrompt,
        EntityExtractionPrompt,
    ]
    assert state["processed_query"].entities == ["Vendor", "Buyer"]


@pytest.mark.parametrize(
    "raw",
    ["local", "mix", "global", "Global search.", "I'd use mix", "nonsense", ""],
)
async def test_the_route_is_the_same_with_and_without_speculation(raw: str) -> None:
    routed = []
    for speculative in (True, False):
        factory = _Factory(route=raw)
        state = await _route(
            _chain(_config(auto_speculative_extraction=speculative), factory)
        )
        router_inputs = [i for c, i in factory.calls if c is StrategySelectionPrompt]
        routed.append((state["resolved_strategy"], router_inputs))
    assert routed[0] == routed[1]


@pytest.mark.parametrize(
    ("route", "reads"),
    [
        ("local", {EntityExtractionPrompt}),
        ("mix", {KeywordsExtractionPrompt}),
        ("global", set()),
    ],
)
async def test_an_unread_extraction_is_never_waited_for(
    route: str, reads: set[type]
) -> None:
    # Handed to the query entry point (which leaves them running in the
    # background), never cancelled here: their requests are sent and billed.
    factory = _Factory(route=route)
    unread = {EntityExtractionPrompt, KeywordsExtractionPrompt} - reads
    factory.hang = unread
    leftovers: list[asyncio.Future[Any]] = []
    token = _LEFTOVER_EXTRACTIONS.set(leftovers)
    try:
        state = await asyncio.wait_for(_route(_chain(_config(), factory)), 2)
    finally:
        _LEFTOVER_EXTRACTIONS.reset(token)
    assert state["resolved_strategy"] is SearchStrategy(route)
    assert factory.cancelled == []
    assert len(leftovers) == len(unread)
    assert not any(task.done() for task in leftovers)
    processed = state["processed_query"]
    assert bool(processed.entities) is (EntityExtractionPrompt in reads)
    assert bool(processed.ll_keywords) is (KeywordsExtractionPrompt in reads)
    await discard(leftovers)
    assert set(factory.cancelled) == unread


async def test_outside_a_query_entry_point_the_step_awaits_the_extraction() -> None:
    factory = _Factory(route="global")
    factory.latency[KeywordsExtractionPrompt] = 0.05
    await _route(_chain(_config(), factory))
    assert factory.cancelled == []
    assert set(factory.finished) >= {EntityExtractionPrompt, KeywordsExtractionPrompt}


async def test_a_speculative_extraction_that_answers_fills_the_cache() -> None:
    # Routed to local, the keywords answer while the entities still run:
    # a later mix query on the question reuses them.
    factory = _Factory(route="local")
    factory.latency[EntityExtractionPrompt] = 0.05
    chain = _chain(_config(), factory)
    await _route(chain)
    assert factory.finished.count(KeywordsExtractionPrompt) == 1
    state = await _route(chain, "mix")
    assert factory.counts[KeywordsExtractionPrompt] == 1
    assert state["processed_query"].ll_keywords == ["Vendor"]


async def test_a_router_failure_cancels_the_speculative_calls() -> None:
    factory = _Factory()
    factory.outputs[StrategySelectionPrompt] = RuntimeError("router down")
    factory.hang = {EntityExtractionPrompt, KeywordsExtractionPrompt}
    chain = _chain(_config(), factory)
    chain.ignore_errors = False
    with pytest.raises(RuntimeError, match="router down"):
        await asyncio.wait_for(_route(chain), 2)
    assert set(factory.cancelled) == {EntityExtractionPrompt, KeywordsExtractionPrompt}


async def test_the_translation_runs_alongside_the_router() -> None:
    config = _config()
    config.processing.translation.source_language = "ja"
    config.processing.translation.target_language = "en"
    factory = _Factory(route="global")
    factory.router_waits_for = TranslationPrompt
    state = await _route(_chain(config, factory))
    assert state["processed_query"].final_query == "translated"
    assert factory.counts[TranslationPrompt] == 1


async def test_an_explicit_strategy_speculates_nothing() -> None:
    factory = _Factory()
    await _route(_chain(_config(), factory), "global")
    assert factory.calls == []


# --- the extraction cache ----------------------------------------------------


async def test_a_repeated_question_reuses_its_extraction() -> None:
    factory = _Factory()
    chain = _chain(_config(), factory)
    first = await _route(chain, "local")
    second = await _route(chain, "local")
    assert factory.counts[EntityExtractionPrompt] == 1
    assert second["processed_query"].entities == first["processed_query"].entities


async def test_comparison_arms_search_with_the_same_keywords() -> None:
    factory = _Factory()
    chain = _chain(_config(), factory)
    mix = await _route(chain, "mix")
    # The model would answer differently the second time.
    factory.outputs[KeywordsExtractionPrompt] = (
        '{"high_level_keywords": ["other"], "low_level_keywords": ["Buyer"]}'
    )
    hybrid = await _route(chain, "hybrid")
    assert factory.counts[KeywordsExtractionPrompt] == 1
    assert hybrid["processed_query"].hl_keywords == mix["processed_query"].hl_keywords
    assert hybrid["processed_query"].ll_keywords == ["Vendor"]


async def test_the_cache_is_keyed_by_the_target_language() -> None:
    factory = _Factory()
    chain = _chain(_config(), factory)
    await _route(chain, "local")
    await _route(chain, "local", target_language="ko")
    languages = [
        i["target_language"] for c, i in factory.calls if c is EntityExtractionPrompt
    ]
    assert languages == ["English", "Korean"]


def test_the_cache_is_keyed_by_the_prompt_text() -> None:
    plain = GraphRAGChain(config=_config())
    custom = GraphRAGChain(
        config=Config(
            custom_prompts={"entity_extraction_human": "Entities in {query}:"}
        )
    )
    key = plain._extraction_key(EntityExtractionPrompt, QUESTION, "English")
    assert key != custom._extraction_key(EntityExtractionPrompt, QUESTION, "English")
    assert key != plain._extraction_key(KeywordsExtractionPrompt, QUESTION, "English")


async def test_a_failed_extraction_is_not_cached() -> None:
    factory = _Factory()
    factory.outputs[KeywordsExtractionPrompt] = "not json"
    chain = _chain(_config(), factory)
    chain.ignore_errors = True
    failed = await _route(chain, "mix")
    assert failed["processed_query"].ll_keywords == []
    factory.outputs[KeywordsExtractionPrompt] = _KEYWORDS
    retried = await _route(chain, "mix")
    assert factory.counts[KeywordsExtractionPrompt] == 2
    assert retried["processed_query"].ll_keywords == ["Vendor"]


async def test_size_zero_disables_the_cache() -> None:
    factory = _Factory()
    chain = _chain(_config(query_extraction_cache_size=0), factory)
    await _route(chain, "local")
    await _route(chain, "local")
    assert factory.counts[EntityExtractionPrompt] == 2


async def test_the_cache_drops_the_least_recently_used_question() -> None:
    factory = _Factory()
    chain = _chain(_config(query_extraction_cache_size=1), factory)
    await _route(chain, "local")
    await chain._route_and_process_step(
        {"query": "Who is the Buyer?", "search_strategy": SearchStrategy.LOCAL}
    )
    await _route(chain, "local")
    assert factory.counts[EntityExtractionPrompt] == 3


# --- output ceilings ---------------------------------------------------------


async def test_the_short_query_calls_get_their_ceilings() -> None:
    factory = _Factory(route="mix")
    config = _config(
        output_ceilings={"strategy_selection": 300, "query_extraction": 700}
    )
    await _route(_chain(config, factory))
    assert factory.kwargs[StrategySelectionPrompt]["output_ceiling"] == 300
    assert factory.kwargs[EntityExtractionPrompt]["output_ceiling"] == 700
    assert factory.kwargs[KeywordsExtractionPrompt]["output_ceiling"] == 700


def _bedrock_max_tokens(config: Config, model: LanguageModelId, **kwargs: Any) -> int:
    factory = BedrockLanguageModelFactory(
        config, boto_session=FakeBotoSession()  # type: ignore[arg-type]
    )
    info = factory.get_model_info(model)
    cfg = factory._build_model_config(info, f"us.{model.value}", True, **kwargs)
    return int(cfg.get("max_tokens") or cfg["model_kwargs"]["max_tokens"])


def test_a_ceiling_keeps_the_thinking_budget_of_the_effort() -> None:
    config = Config()
    haiku = LanguageModelId.CLAUDE_V5_5_HAIKU
    fast = {"model_tier": "fast", "output_ceiling": 4096}
    # Haiku 5.5 always reasons; fast_effort low budgets 2,048 tokens.
    assert _bedrock_max_tokens(config, haiku) == 16384
    assert _bedrock_max_tokens(config, haiku, **fast) == 4096
    assert _bedrock_max_tokens(config, haiku, **{**fast, "output_ceiling": 500}) == (
        2048 + 1024
    )
    assert _bedrock_max_tokens(config, haiku, **fast, effort="high") == 8192 + 1024
    # A model that does not reason gets the ceiling, never under 1,024.
    no_thinking = LanguageModelId.CLAUDE_V3_5_SONNET_V2
    assert _bedrock_max_tokens(config, no_thinking, output_ceiling=2000) == 2000
    assert _bedrock_max_tokens(config, no_thinking, output_ceiling=500) == 1024


def test_a_ceiling_stays_above_a_manual_thinking_budget() -> None:
    config = Config()
    model = LanguageModelId.CLAUDE_V3_7_SONNET
    info = BedrockLanguageModelFactory(
        config, boto_session=FakeBotoSession()  # type: ignore[arg-type]
    ).get_model_info(model)
    assert info.supports_thinking and not info.uses_adaptive_thinking
    kwargs = {"output_ceiling": 1000, "enable_thinking": True}
    assert _bedrock_max_tokens(config, model, **kwargs) == 2048 + 1024
    assert (
        _bedrock_max_tokens(config, model, **kwargs, thinking_budget_tokens=5000)
        == 5000 + 1024
    )


def test_a_ceiling_never_raises_max_tokens() -> None:
    config = Config(aws={"bedrock": {"default_max_output_tokens": 2000}})
    haiku = LanguageModelId.CLAUDE_V5_5_HAIKU
    assert (
        _bedrock_max_tokens(config, haiku, model_tier="fast", output_ceiling=4096)
        == 2000
    )
    assert ceiled_output_tokens(2000, 4096, 2048) == 2000
    # An explicit max_tokens wins over the ceiling.
    assert (
        _bedrock_max_tokens(config, haiku, max_tokens=9000, output_ceiling=100) == 9000
    )


def _ollama_config() -> Config:
    config = Config(
        backend="local",
        models={
            "default": "synthetic-llm",
            "fast": "synthetic-fast",
            "embedding": {"model_id": "synthetic-embed", "dimension": 8},
        },
        aws={"bedrock": {"default_max_output_tokens": 16384}},
        providers={"llm": {"max_output_tokens": 128000}},
    )
    translation = config.processing.translation
    translation.source_language = translation.target_language
    return config


async def test_an_auto_query_reserves_at_most_30k_output_tokens() -> None:
    # Router + entities + (speculative) keywords + answer, as the model
    # server sees them; before the ceilings 3 x 16,384 without speculation.
    requested: list[int] = []

    def respond(system: str, human: str) -> str:
        if "search strategy selector" in system:
            return "local"
        if "keyword extractor" in system:
            return _KEYWORDS
        if "entity extraction expert" in system:
            return "Vendor, Buyer"
        raise AssertionError  # the answer

    server = FakeOllama(respond, "The Vendor owes delivery.", 8)

    def recording(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            requested.append(json.loads(request.content)["options"]["num_predict"])
        return server(request)

    config = _ollama_config()
    transport = httpx.MockTransport(recording)
    providers = build_providers(
        config, llm_factory=OllamaLanguageModelFactory(config, transport=transport)
    )
    chain = GraphRAGChain(config, providers=providers)
    try:
        state = await chain._route_and_process_step(
            {"query": QUESTION, "search_strategy": SearchStrategy.AUTO}
        )
        answer = await chain._answer_generation_step(
            {**state, "context": "The Vendor owes delivery."}
        ).ainvoke({**state, "context": "The Vendor owes delivery."})
    finally:
        await chain.aclose()
    assert answer == "The Vendor owes delivery."
    assert state["resolved_strategy"] is SearchStrategy.LOCAL
    assert len(requested) == 4
    assert sum(requested) <= 30000
    assert sorted(requested) == [4096, 4096, 4096, 16384]


def test_the_relevance_rating_and_answer_span_get_their_ceilings(mocker) -> None:
    from unified_kg_rag.adapters.search_strategies import global_search
    from unified_kg_rag.evaluation import answer_match_evaluator

    calls: list[dict[str, Any]] = []

    def record(*args: Any, **kwargs: Any) -> None:
        calls.append({"prompt": (kwargs.get("prompt_class") or args[2]), **kwargs})

    for module in (global_search, answer_match_evaluator):
        mocker.patch.object(module, "setup_chain", side_effect=record)
    config = _ollama_config()
    config.search.global_search.use_dynamic_selection = True
    config.evaluation.answer_span = "extracted"
    config.evaluation.answer_span_output_ceiling = 333
    providers = build_providers(
        config,
        llm_factory=OllamaLanguageModelFactory(
            config, transport=httpx.MockTransport(lambda r: httpx.Response(500))
        ),
    )
    global_search.GlobalSearchStrategy(config, {}, providers=providers)
    answer_match_evaluator.AnswerMatchEvaluator(config, providers=providers)
    ceilings = {c["prompt"].__name__: c.get("output_ceiling") for c in calls}
    assert ceilings["CommunityRelevancePrompt"] == 4096
    assert ceilings["AnswerSpanExtractionPrompt"] == 333
    assert ceilings["GlobalMapPrompt"] is None


# --- answer prompt per strategy ----------------------------------------------


def test_every_strategy_keeps_the_concise_answer_by_default() -> None:
    chain = GraphRAGChain(config=Config())
    for strategy in SearchStrategy:
        assert chain._answer_prompt(strategy) is AnswerGenerationPrompt
    assert chain._answer_prompt(None) is AnswerGenerationPrompt


def test_a_strategy_can_take_the_comprehensive_answer() -> None:
    chain = GraphRAGChain(
        config=Config(
            search={
                "answer_prompt_style": {
                    "global": "comprehensive",
                    "drift": "comprehensive",
                }
            }
        )
    )
    assert chain._answer_prompt(SearchStrategy.GLOBAL) is ComprehensiveAnswerPrompt
    assert chain._answer_prompt(SearchStrategy.DRIFT) is ComprehensiveAnswerPrompt
    assert chain._answer_prompt(SearchStrategy.LOCAL) is AnswerGenerationPrompt


async def test_the_answer_step_uses_the_resolved_strategys_prompt() -> None:
    config = Config(search={"answer_prompt_style": {"global": "comprehensive"}})
    factory = _Factory()
    factory.outputs[ComprehensiveAnswerPrompt] = "long answer"
    factory.outputs[AnswerGenerationPrompt] = "short answer"
    chain = _chain(config, factory)
    answers = {}
    for strategy in (SearchStrategy.GLOBAL, SearchStrategy.LOCAL):
        state = {"query": QUESTION, "context": "facts", "resolved_strategy": strategy}
        answers[strategy] = await chain._answer_generation_step(state).ainvoke(state)
    assert answers == {
        SearchStrategy.GLOBAL: "long answer",
        SearchStrategy.LOCAL: "short answer",
    }


def test_auto_takes_no_answer_style() -> None:
    with pytest.raises(ValidationError, match="auto resolves"):
        Config(search={"answer_prompt_style": {"auto": "comprehensive"}})


def test_the_comprehensive_answer_is_overridable() -> None:
    custom = Config(
        custom_prompts={"comprehensive_answer_system": "Answer at length."}
    ).custom_prompts
    resolved = ComprehensiveAnswerPrompt.resolve(custom_prompts=custom)
    assert resolved.system_prompt_template == "Answer at length."
    with pytest.raises(ValidationError, match="context"):
        Config(custom_prompts={"comprehensive_answer_human": "Q: {query}"})


def test_the_comprehensive_answer_shares_the_answer_variables() -> None:
    assert ComprehensiveAnswerPrompt.input_variables == (
        AnswerGenerationPrompt.input_variables
    )
