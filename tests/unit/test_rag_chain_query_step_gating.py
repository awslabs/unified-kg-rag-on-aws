# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-strategy gating of the query-side LLM calls in GraphRAGChain (AWS-free).

Entity extraction is only paid for strategies that read ``entity_focus``
(local, DRIFT) and dual-keyword extraction only for the LightRAG modes that
read high/low keywords (mix, hybrid). The decision is made on the RESOLVED
strategy, so AUTO routing to local still extracts entities.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401 (registers strategies)
from unified_kg_rag.application.retrieval.rag_chain import (
    EntityExtractionPrompt,
    GraphRAGChain,
    KeywordsExtractionPrompt,
    ProcessedQuery,
    StrategySelectionPrompt,
    TranslationPrompt,
)
from unified_kg_rag.domain.models import Config, SearchStrategy

pytestmark = pytest.mark.unit

_KEYWORDS_JSON = (
    '{"high_level_keywords": ["supply terms"], "low_level_keywords": ["Vendor"]}'
)


class _CountingChain:
    def __init__(self, prompt_class: type, result: Any, log: list) -> None:
        self._prompt_class = prompt_class
        self._result = result
        self._log = log

    async def ainvoke(self, inputs: dict[str, Any]) -> Any:
        self._log.append((self._prompt_class, inputs))
        return self._result


class _FakeFactory:
    """Counts LLM invocations per prompt class and returns canned outputs."""

    def __init__(self, *, route: str = "local", translation: str = "") -> None:
        self.calls: list[tuple[type, dict[str, Any]]] = []
        self.outputs: dict[type, Any] = {
            EntityExtractionPrompt: ["Vendor", "Buyer"],
            KeywordsExtractionPrompt: _KEYWORDS_JSON,
            StrategySelectionPrompt: route,
            TranslationPrompt: translation,
        }

    def get_chain(self, prompt_class: type, parser: Any, **kwargs: Any) -> Any:
        return _CountingChain(
            prompt_class, self.outputs.get(prompt_class, ""), self.calls
        )

    @property
    def counts(self) -> Counter:
        return Counter(cls for cls, _ in self.calls)


@pytest.fixture
def same_language_config(config: Config) -> Config:
    config.processing.translation.source_language = (
        config.processing.translation.target_language
    )
    return config


def _chain_with(config: Config, factory: _FakeFactory) -> GraphRAGChain:
    chain = GraphRAGChain(config=config)
    chain._get_chain_for_prompt = factory.get_chain  # type: ignore[assignment]
    return chain


async def _run_query_side(
    chain: GraphRAGChain, strategy: SearchStrategy
) -> ProcessedQuery:
    # Same order as the real chain: resolve (incl. AUTO routing), then process.
    state = await chain._resolve_strategy(
        {"query": "What does the Vendor owe the Buyer?", "search_strategy": strategy}
    )
    return await chain._process_query_step(state)


@pytest.mark.parametrize(
    "strategy",
    [
        SearchStrategy.SIMPLE,
        SearchStrategy.GLOBAL,
        SearchStrategy.MIX,
        SearchStrategy.HYBRID,
        SearchStrategy.NAIVE,
    ],
)
async def test_no_entity_extraction_when_strategy_ignores_entity_focus(
    same_language_config: Config, strategy: SearchStrategy
) -> None:
    factory = _FakeFactory()
    processed = await _run_query_side(
        _chain_with(same_language_config, factory), strategy
    )
    assert factory.counts[EntityExtractionPrompt] == 0
    assert processed.entities == []


@pytest.mark.parametrize("strategy", [SearchStrategy.LOCAL, SearchStrategy.DRIFT])
async def test_entity_extraction_kept_for_entity_focus_consumers(
    same_language_config: Config, strategy: SearchStrategy
) -> None:
    # Local seeds from entity_focus; DRIFT sizes its entity candidate pool by it.
    factory = _FakeFactory()
    processed = await _run_query_side(
        _chain_with(same_language_config, factory), strategy
    )
    assert factory.counts[EntityExtractionPrompt] == 1
    assert factory.counts[KeywordsExtractionPrompt] == 0
    assert processed.entities == ["Vendor", "Buyer"]


async def test_naive_makes_no_query_side_llm_call(
    same_language_config: Config,
) -> None:
    factory = _FakeFactory()
    processed = await _run_query_side(
        _chain_with(same_language_config, factory), SearchStrategy.NAIVE
    )
    assert factory.calls == []
    assert processed.hl_keywords == []
    assert processed.ll_keywords == []


@pytest.mark.parametrize("strategy", [SearchStrategy.MIX, SearchStrategy.HYBRID])
async def test_mix_hybrid_still_extract_keywords(
    same_language_config: Config, strategy: SearchStrategy
) -> None:
    factory = _FakeFactory()
    processed = await _run_query_side(
        _chain_with(same_language_config, factory), strategy
    )
    assert factory.counts[KeywordsExtractionPrompt] == 1
    assert processed.hl_keywords == ["supply terms"]
    assert processed.ll_keywords == ["Vendor"]


async def test_keywords_use_translated_query_when_translating(config: Config) -> None:
    config.processing.translation.source_language = "ja"
    config.processing.translation.target_language = "en"
    factory = _FakeFactory(translation="What does the Vendor owe the Buyer?")
    chain = _chain_with(config, factory)
    processed = await chain._process_query_step(
        {"query": "ベンダーは何を負うか", "resolved_strategy": SearchStrategy.HYBRID}
    )
    keyword_inputs = [i for cls, i in factory.calls if cls is KeywordsExtractionPrompt]
    assert keyword_inputs == [
        {"query": "What does the Vendor owe the Buyer?", "target_language": "en"}
    ]
    assert processed.ll_keywords == ["Vendor"]


async def test_keyword_failure_propagates_without_ignore_errors(
    same_language_config: Config,
) -> None:
    factory = _FakeFactory()
    factory.outputs[KeywordsExtractionPrompt] = "not json"
    chain = _chain_with(same_language_config, factory)
    chain.ignore_errors = False
    with pytest.raises(json.JSONDecodeError):
        await chain._process_query_step(
            {"query": "q", "resolved_strategy": SearchStrategy.MIX}
        )


async def test_auto_routed_to_local_extracts_entities(
    same_language_config: Config,
) -> None:
    factory = _FakeFactory(route="local")
    processed = await _run_query_side(
        _chain_with(same_language_config, factory), SearchStrategy.AUTO
    )
    assert factory.counts[StrategySelectionPrompt] == 1
    assert factory.counts[EntityExtractionPrompt] == 1
    assert processed.entities == ["Vendor", "Buyer"]


async def test_auto_router_is_told_the_routable_strategies(
    same_language_config: Config,
) -> None:
    factory = _FakeFactory(route="mix")
    await _run_query_side(
        _chain_with(same_language_config, factory), SearchStrategy.AUTO
    )
    route_inputs = [i for cls, i in factory.calls if cls is StrategySelectionPrompt]
    assert route_inputs and route_inputs[0]["strategies"] == "local, mix, global, drift"
    # Routed to mix -> the LightRAG keyword extraction runs.
    assert factory.counts[KeywordsExtractionPrompt] == 1


async def test_auto_routed_to_global_skips_entities(
    same_language_config: Config,
) -> None:
    factory = _FakeFactory(route="global")
    processed = await _run_query_side(
        _chain_with(same_language_config, factory), SearchStrategy.AUTO
    )
    assert factory.counts[StrategySelectionPrompt] == 1
    assert factory.counts[EntityExtractionPrompt] == 0
    assert processed.entities == []


async def test_entity_focus_order_is_deterministic(config: Config) -> None:
    chain = GraphRAGChain(config=config)
    captured: dict[str, Any] = {}

    class _Strategy:
        async def asearch(self, query: Any) -> str:
            captured["query"] = query
            return "result"

    chain._get_strategy_instance = lambda _s: _Strategy()  # type: ignore[assignment]
    await chain._search_step(
        {
            "resolved_strategy": SearchStrategy.LOCAL,
            "processed_query": ProcessedQuery(
                original_query="q",
                final_query="q",
                entities=["Vendor", "Buyer", "Warehouse", "Vendor"],
            ),
            "relevant_entities": ["Carrier", "Buyer"],
        }
    )
    assert captured["query"].entity_focus == [
        "Vendor",
        "Buyer",
        "Warehouse",
        "Carrier",
    ]
