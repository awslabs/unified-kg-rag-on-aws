# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraphRAGChain reuses strategy instances per event loop (AWS-free).

A strategy used to be constructed per query (a boto3 Session, scorer, token
manager and, for global/DRIFT, an LLM factory and several chains each time).
It is now cached per (strategy, loop), dropped with the retrievers when the
loop changes, and safe to share between concurrent queries on one loop.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, ProcessedQuery
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)

pytestmark = pytest.mark.unit


class _LLMFactory:
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return FakeListChatModel(responses=["ok"])

    def get_model_info(self, model_id: Any) -> Any:
        return None


class _TokenCounter:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> tuple[str, int]:
        return text, len(text.split())


class _EchoRetriever:
    """Document retriever whose results name the query that produced them.

    The first query sleeps longer, so two concurrent searches interleave and
    finish in the opposite order to how they started.
    """

    def __init__(self) -> None:
        self.closed = 0

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        await asyncio.sleep(0.05 if query.query == "first" else 0.0)
        return [
            RetrievalResult(
                content=f"{query.query} chunk {i}",
                score=1.0 - i / 10,
                source=f"{query.query}-{i}",
                retriever_type="text",
                metadata={"filters": dict(query.filters or {})},
            )
            for i in range(query.top_k)
        ]

    def filter_fields(self) -> None:
        return None

    def close(self) -> None:
        self.closed += 1


def _chain(config: Config | None = None) -> GraphRAGChain:
    config = config or Config()
    providers = Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=_LLMFactory(),
        embedding_factory=MagicMock(),
        rerank_factory=MagicMock(),
        token_counter_factory=lambda *_, **__: _TokenCounter(),
    )
    return GraphRAGChain(
        config=config,
        providers=providers,
        retriever_builders={
            RetrieverRole.DOCUMENT: _EchoRetriever,  # type: ignore[dict-item]
            RetrieverRole.GRAPH: _EchoRetriever,  # type: ignore[dict-item]
        },
    )


def _run_on_fresh_loop(coro: Any) -> Any:
    # Not asyncio.run, which nest_asyncio (applied by the CLI modules) turns
    # into a reuse of one loop.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _state(query: str, strategy: SearchStrategy, **extra: Any) -> dict[str, Any]:
    return {
        "resolved_strategy": strategy,
        "processed_query": ProcessedQuery(original_query=query, final_query=query),
        "top_k": extra.pop("top_k", 3),
        **extra,
    }


async def test_strategy_reused_across_queries_on_one_loop() -> None:
    chain = _chain()
    first = chain._get_strategy_instance(SearchStrategy.LOCAL)
    second = chain._get_strategy_instance(SearchStrategy.LOCAL)
    other = chain._get_strategy_instance(SearchStrategy.SIMPLE)

    assert first is second
    assert other is not first
    # The cached strategy holds the cached retrievers.
    assert first.document_retriever is chain._get_retriever(RetrieverRole.DOCUMENT)


def test_strategy_not_shared_across_loops() -> None:
    chain = _chain()

    async def get() -> Any:
        return chain._get_strategy_instance(SearchStrategy.SIMPLE)

    first = _run_on_fresh_loop(get())
    second = _run_on_fresh_loop(get())

    assert first is not second
    assert first.document_retriever is not second.document_retriever
    # The evicted strategy's retriever was released with it.
    assert first.document_retriever.closed == 1
    assert second.document_retriever.closed == 0
    chain.close()


def test_strategy_construction_uses_bundle_once(mocker) -> None:
    from unified_kg_rag.adapters.search_strategies import global_search

    chain = _chain()
    spy = mocker.spy(global_search.GlobalSearchStrategy, "__init__")

    async def queries() -> None:
        for _ in range(5):
            chain._get_strategy_instance(SearchStrategy.GLOBAL)

    _run_on_fresh_loop(queries())
    assert spy.call_count == 1
    chain.close()


@pytest.mark.parametrize("strategy", [SearchStrategy.SIMPLE, SearchStrategy.NAIVE])
async def test_concurrent_queries_on_a_cached_strategy_are_isolated(
    strategy: SearchStrategy,
) -> None:
    chain = _chain()
    instance = chain._get_strategy_instance(strategy)

    first, second = await asyncio.gather(
        chain._search_step(
            _state("first", strategy, top_k=2, filters={"document_id": ["d-1"]})
        ),
        chain._search_step(
            _state("second", strategy, top_k=4, filters={"document_id": ["d-2"]})
        ),
    )

    # Both queries ran on the one cached instance ...
    assert chain._get_strategy_instance(strategy) is instance
    # ... yet each result carries only its own query, size and filters.
    assert first.query.query == "first" and second.query.query == "second"
    assert first.query.top_k == 2 and second.query.top_k == 4
    assert {r.content.split()[0] for r in first.results} == {"first"}
    assert {r.content.split()[0] for r in second.results} == {"second"}
    assert len(first.results) == 2 and len(second.results) == 4
    assert {r.metadata["filters"]["document_id"][0] for r in first.results} == {"d-1"}
    assert {r.metadata["filters"]["document_id"][0] for r in second.results} == {"d-2"}
    assert first.metadata is not second.metadata
    assert first.query.metadata == {"search_strategy": strategy.value}
    assert second.query.metadata == {"search_strategy": strategy.value}


async def test_aclose_drops_cached_strategies() -> None:
    chain = _chain()
    chain._get_strategy_instance(SearchStrategy.SIMPLE)
    await chain.aclose()
    assert chain._strategy_cache == {}
    assert chain._retriever_cache == {}
