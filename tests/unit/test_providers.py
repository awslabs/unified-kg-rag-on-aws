# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ``Providers`` bundle and its reach through the query path (AWS-free).

An injected provider must reach every component the chain builds — strategies
(including the global/DRIFT LLM chains), the hybrid scorer, both token
managers and conversation memory — without any Bedrock factory being
constructed. Defaults must be built lazily and at most once.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from unified_kg_rag.adapters import providers as providers_module
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.retrieval.memory_manager import MemoryManager
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import Config, SearchStrategy

pytestmark = pytest.mark.unit


class FakeLLMFactory:
    """Structural LLMFactoryPort returning an offline chat model."""

    def __init__(self) -> None:
        self.model_ids: list[str] = []

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        self.model_ids.append(str(model_id))
        return FakeListChatModel(responses=["ok"])

    def get_model_info(self, model_id: Any) -> Any:
        return None


class FakeRerankFactory:
    def __init__(self) -> None:
        self.calls = 0

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return MagicMock(top_n=kwargs.get("top_k"))

    def get_model_info(self, model_id: Any) -> Any:
        return None


class FakeTokenCounter:
    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> tuple[str, int]:
        words = text.split()[:max_tokens]
        return " ".join(words), len(words)


@pytest.fixture
def no_bedrock(mocker) -> dict[str, MagicMock]:
    """Fail loudly if any default Bedrock factory/counter is constructed."""

    def _forbid(name: str) -> MagicMock:
        return mocker.patch.object(
            providers_module,
            name,
            side_effect=AssertionError(f"{name} must not be constructed"),
        )

    return {
        name: _forbid(name)
        for name in (
            "BedrockLanguageModelFactory",
            "BedrockEmbeddingModelFactory",
            "BedrockRerankModelFactory",
            "BedrockTokenCounter",
        )
    }


def _fake_providers(config: Config) -> Providers:
    return Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=FakeLLMFactory(),
        embedding_factory=MagicMock(),
        rerank_factory=FakeRerankFactory(),
        token_counter_factory=lambda *_, **__: FakeTokenCounter(),
    )


# --- the bundle itself ------------------------------------------------------


def test_defaults_are_lazy_and_built_once(mocker) -> None:
    llm_cls = mocker.patch.object(providers_module, "BedrockLanguageModelFactory")
    rerank_cls = mocker.patch.object(providers_module, "BedrockRerankModelFactory")
    providers = Providers(Config(), boto_session=MagicMock())

    llm_cls.assert_not_called()
    assert providers.llm_factory is providers.llm_factory
    llm_cls.assert_called_once()
    rerank_cls.assert_not_called()  # never touched, never built


def test_default_session_comes_from_the_profile(mocker) -> None:
    session_cls = mocker.patch.object(providers_module.boto3, "Session")
    config = Config()
    providers = Providers(config)
    assert providers.boto_session is providers.boto_session
    session_cls.assert_called_once_with(profile_name=config.aws.profile_name)


def test_injected_providers_are_returned_verbatim(no_bedrock) -> None:
    config = Config()
    llm, embedding, rerank = FakeLLMFactory(), MagicMock(), FakeRerankFactory()
    providers = Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=llm,
        embedding_factory=embedding,
        rerank_factory=rerank,
    )
    assert providers.llm_factory is llm
    assert providers.embedding_factory is embedding
    assert providers.rerank_factory is rerank


def test_default_token_counters_share_one_client(mocker) -> None:
    counter_cls = mocker.patch.object(providers_module, "BedrockTokenCounter")
    session = MagicMock()
    providers = Providers(Config(), boto_session=session)

    providers.token_counter("model-a")
    providers.token_counter("model-b")
    providers.token_counter("model-c", api_supported=False)

    session.client.assert_called_once()
    clients = [c.kwargs["client"] for c in counter_cls.call_args_list]
    assert clients[0] is clients[1] is session.client.return_value
    assert clients[2] is None


def test_resolve_prefers_the_given_bundle() -> None:
    config = Config()
    bundle = Providers(config, boto_session=MagicMock())
    assert Providers.resolve(config, bundle) is bundle
    session = MagicMock()
    assert Providers.resolve(config, None, session).boto_session is session


# --- reach through the query path -------------------------------------------


@pytest.mark.parametrize("strategy", [SearchStrategy.GLOBAL, SearchStrategy.DRIFT])
def test_injected_llm_factory_reaches_global_and_drift(
    strategy: SearchStrategy, no_bedrock
) -> None:
    config = Config()
    config.search.reranking.enabled = True
    providers = _fake_providers(config)
    chain = GraphRAGChain(config=config, providers=providers)
    chain._get_retriever = lambda role: MagicMock()  # type: ignore[method-assign]

    instance = chain._get_strategy_instance(strategy)

    assert instance.providers is providers
    llm = providers.llm_factory
    assert isinstance(llm, FakeLLMFactory)
    # The strategy's own LLM chains were built from the injected factory.
    expected = (
        config.search.global_search.map_model_id
        if strategy is SearchStrategy.GLOBAL
        else config.search.drift_search.query_refinement_model_id
    )
    assert str(expected) in llm.model_ids
    # Scorer and token counter came from the bundle too.
    assert instance.hybrid_scorer.rerank_factory is providers.rerank_factory
    assert isinstance(instance.token_manager._token_counter, FakeTokenCounter)
    assert isinstance(chain.token_manager._token_counter, FakeTokenCounter)


def test_model_factory_shorthand_reaches_strategies(no_bedrock) -> None:
    # The pre-existing seam: GraphRAGChain(model_factory=...) alone.
    no_bedrock["BedrockTokenCounter"].side_effect = None
    config = Config()
    config.search.reranking.enabled = False
    fake = FakeLLMFactory()
    chain = GraphRAGChain(config=config, boto_session=MagicMock(), model_factory=fake)
    chain._get_retriever = lambda role: MagicMock()  # type: ignore[method-assign]

    instance = chain._get_strategy_instance(SearchStrategy.GLOBAL)

    assert chain.factory is fake
    assert instance.providers.llm_factory is fake
    assert fake.model_ids  # global's map/reduce chains


def test_injected_llm_factory_reaches_conversation_memory(no_bedrock) -> None:
    config = Config()
    providers = _fake_providers(config)
    chain = GraphRAGChain(config=config, providers=providers)

    # The first chain creates the shared memory from its config and providers.
    assert isinstance(chain.memory_manager, MemoryManager)
    assert chain.memory_manager.providers is providers
    history = asyncio.run(chain.memory_manager.get_or_create_memory("c-1"))
    assert history.boto_session is providers.boto_session
    assert str(config.search.entity_extraction_model_id) in (
        providers.llm_factory.model_ids  # type: ignore[attr-defined]
    )


def test_default_chains_share_conversation_history(no_bedrock) -> None:
    from unified_kg_rag.domain.models import MessageRole

    config = Config()
    first = GraphRAGChain(config=config, providers=_fake_providers(config))
    second = GraphRAGChain(config=config, providers=_fake_providers(config))

    async def scenario() -> list[str]:
        await first.memory_manager.add_message(
            "conv-1", MessageRole.USER, "about Vendor", entities=["Vendor"]
        )
        history = await second.memory_manager.get_or_create_memory("conv-1")
        return [str(m.content) for m in history.messages]

    # A chain built per request still sees the earlier request's turn.
    assert second.memory_manager is first.memory_manager
    assert asyncio.run(scenario()) == ["about Vendor"]


def test_injected_memory_manager_isolates_a_chain(no_bedrock) -> None:
    from unified_kg_rag.domain.models import MessageRole

    config = Config()
    shared = GraphRAGChain(config=config, providers=_fake_providers(config))
    isolated_manager = MemoryManager(config, providers=_fake_providers(config))
    isolated = GraphRAGChain(
        config=config,
        providers=_fake_providers(config),
        memory_manager=isolated_manager,
    )

    async def scenario() -> list[str]:
        await shared.memory_manager.add_message(
            "conv-1", MessageRole.USER, "about Vendor", entities=["Vendor"]
        )
        history = await isolated.memory_manager.get_or_create_memory("conv-1")
        return [str(m.content) for m in history.messages]

    assert isolated.memory_manager is isolated_manager
    assert asyncio.run(scenario()) == []


def test_mismatched_config_keeps_shared_memory_and_logs_once(
    no_bedrock, mocker
) -> None:
    from unified_kg_rag.adapters.retrieval import memory_manager as mm

    warn = mocker.spy(mm.logger, "warning")
    config = Config()
    first = GraphRAGChain(config=config, providers=_fake_providers(config))
    other = Config()
    other.memory.max_conversations = 3
    second = GraphRAGChain(config=other, providers=_fake_providers(other))
    third = GraphRAGChain(config=other, providers=_fake_providers(other))

    assert first.memory_manager is second.memory_manager is third.memory_manager
    assert first.memory_manager.config is config
    assert warn.call_count == 1


def test_providers_and_model_factory_are_exclusive() -> None:
    config = Config()
    with pytest.raises(ValueError, match="either model_factory or providers"):
        GraphRAGChain(
            config=config,
            providers=_fake_providers(config),
            model_factory=FakeLLMFactory(),
        )


def test_strategy_without_providers_still_constructs(mocker) -> None:
    # Backward compatibility: direct construction builds a default bundle.
    from unified_kg_rag.adapters.search_strategies import LocalSearchStrategy

    mocker.patch.object(providers_module, "BedrockTokenCounter")
    session = MagicMock()
    strategy = LocalSearchStrategy(config=Config(), retrievers={}, boto_session=session)
    assert strategy.boto_session is session
    assert strategy.providers.boto_session is session


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [(SearchStrategy.LOCAL, ["Vendor"]), (SearchStrategy.SIMPLE, None)],
)
async def test_memory_save_reuses_query_step_entities(
    strategy: SearchStrategy, expected: list[str] | None
) -> None:
    from unified_kg_rag.application.retrieval.rag_chain import (
        ProcessedQuery,
        RAGOutput,
    )
    from unified_kg_rag.domain.models import SearchQuery, SearchResult

    config = Config()
    chain = GraphRAGChain(config=config, providers=_fake_providers(config))
    calls: list[dict[str, Any]] = []

    class _Recorder:
        async def add_message(
            self, conv_id: str, role: Any, content: str, **kw: Any
        ) -> None:
            calls.append({"role": role, **kw})

    chain.memory_manager = _Recorder()  # type: ignore[assignment]
    output = RAGOutput(
        answer="a",
        sources=[],
        search_results=SearchResult(
            query=SearchQuery(query="q"),
            results=[],
            total_results=0,
            search_strategy=strategy.value,
            processing_time=0.0,
        ),
        conversation_id="c-1",
        processed_query=ProcessedQuery(
            original_query="q", final_query="q", entities=["Vendor"]
        ),
    )

    await chain._save_memory(output, query_processing=True)
    await chain._save_memory(output)  # query processing off: never reused

    assert calls[0].get("entities") == expected
    assert "entities" not in calls[2]
