# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The composition root and the provider registry (AWS-free).

``config.providers`` selects every role through the registry, and with
``backend: local`` the selection reaches every component the orchestrators
build without constructing a single Bedrock factory, Bedrock token counter,
DynamoDB store or boto3 client: those constructors are patched to fail.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import boto3
import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from unified_kg_rag.adapters import provider_registry
from unified_kg_rag.adapters import providers as providers_module
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.aws import dynamodb as dynamodb_module
from unified_kg_rag.adapters.aws import token_counter as token_counter_module
from unified_kg_rag.adapters.local.embeddings import OllamaEmbeddingFactory
from unified_kg_rag.adapters.local.models import (
    OllamaLanguageModelFactory,
    OpenAICompatibleLanguageModelFactory,
)
from unified_kg_rag.adapters.local.rerank import NoRerankFactory
from unified_kg_rag.adapters.local.sqlite_doc_status import SQLiteDocStatusStore
from unified_kg_rag.adapters.local.token_counter import LocalTokenCounter
from unified_kg_rag.adapters.provider_registry import (
    ProviderRole,
    register_provider,
    registered_kinds,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.bootstrap import (
    build_providers,
    check_providers,
    with_provider_overrides,
)
from unified_kg_rag.domain.models import (
    Backend,
    Config,
    PipelineConfig,
    PipelineStageType,
    RetrieverRole,
)
from unified_kg_rag.domain.retrieval.strategy_registry import registered_strategies
from unified_kg_rag.shared import ConfigurationError

pytestmark = pytest.mark.unit


def _local_config(tmp_path: Path, **overrides: Any) -> Config:
    return Config(
        backend="local",
        providers={
            "rerank": {"kind": "none"},
            "doc_status": {
                "enabled": True,
                "sqlite_path": str(tmp_path / "doc-status.sqlite3"),
            },
        },
        **overrides,
    )


@pytest.fixture
def no_aws(mocker) -> None:
    """Fail if any Bedrock factory/counter, DynamoDB store or boto client is built."""

    def forbid(target: Any, name: str) -> None:
        mocker.patch.object(
            target, name, side_effect=AssertionError(f"{target}.{name} called")
        )

    forbid(bedrock_module.BaseBedrockModelFactory, "__init__")
    forbid(token_counter_module.BedrockTokenCounter, "__init__")
    forbid(dynamodb_module.DynamoDBDocStatusStore, "__init__")
    forbid(boto3.Session, "client")
    for name in (
        "BedrockLanguageModelFactory",
        "BedrockEmbeddingModelFactory",
        "BedrockRerankModelFactory",
        "BedrockTokenCounter",
    ):
        forbid(providers_module, name)


@pytest.fixture
def custom_kind() -> Iterator[str]:
    kind = "synthetic_server"
    yield kind
    for role in ProviderRole:
        provider_registry._REGISTRY.pop((role, kind), None)


# --- registry ---------------------------------------------------------------


def test_built_in_kinds_are_registered() -> None:
    assert registered_kinds(ProviderRole.LLM) == [
        "bedrock",
        "ollama",
        "openai_compatible",
    ]
    assert registered_kinds(ProviderRole.EMBEDDING) == [
        "bedrock",
        "ollama",
        "openai_compatible",
    ]
    assert registered_kinds(ProviderRole.RERANK) == ["bedrock", "cross_encoder", "none"]
    assert registered_kinds(ProviderRole.GRAPH_INDEXER) == ["gremlin", "neptune"]
    assert registered_kinds(ProviderRole.GRAPH_RETRIEVER) == ["gremlin", "neptune"]
    assert registered_kinds(ProviderRole.VECTOR_INDEXER) == ["opensearch"]
    assert registered_kinds(ProviderRole.DOC_STATUS) == ["dynamodb", "sqlite"]


def test_check_providers_reports_the_selection() -> None:
    selected = check_providers(Config())
    assert selected[ProviderRole.LLM] == "bedrock"
    assert selected[ProviderRole.TOKEN_COUNTER] == "bedrock"
    assert selected[ProviderRole.DOC_STATUS] == "dynamodb"


def test_an_unregistered_kind_fails_fast_with_the_registered_ones() -> None:
    config = Config(providers={"vector": {"kind": "lancedb"}})
    with pytest.raises(ConfigurationError, match="Registered kinds: opensearch"):
        build_providers(config)
    with pytest.raises(ConfigurationError, match="providers.vector.kind 'lancedb'"):
        Providers(config).build_vector_indexer()


def test_a_custom_kind_is_selected_without_code_changes(custom_kind: str) -> None:
    built: list[Providers] = []

    @register_provider(ProviderRole.LLM, custom_kind)
    def _build(providers: Providers) -> Any:
        built.append(providers)
        return OpenAICompatibleLanguageModelFactory(providers.config)

    @register_provider(ProviderRole.TOKEN_COUNTER, custom_kind)
    def _counter(providers: Providers) -> Any:
        return lambda model_id, **_: LocalTokenCounter(model_id)

    config = Config(providers={"llm": {"kind": custom_kind, "default_model_id": "m"}})
    providers = build_providers(config)
    assert isinstance(providers.llm_factory, OpenAICompatibleLanguageModelFactory)
    assert built == [providers]
    assert isinstance(providers.token_counter("m"), LocalTokenCounter)


def test_default_config_builds_the_bedrock_providers(mocker) -> None:
    llm = mocker.patch.object(providers_module, "BedrockLanguageModelFactory")
    store = mocker.patch.object(dynamodb_module, "DynamoDBDocStatusStore")
    providers = build_providers(Config(), boto_session=MagicMock())
    assert providers.llm_factory is llm.return_value
    assert providers.build_doc_status() is store.return_value
    store.assert_called_once()


def test_injected_components_win_over_the_configured_kind(no_aws, tmp_path) -> None:
    config = _local_config(tmp_path)
    llm, store = MagicMock(), MagicMock()
    graph, retriever = MagicMock(), MagicMock()
    providers = build_providers(
        config,
        llm_factory=llm,
        doc_status=store,
        graph_indexer_factory=lambda: graph,
        retriever_builders={RetrieverRole.GRAPH: lambda: retriever},
    )
    assert providers.llm_factory is llm
    assert providers.build_doc_status() is store
    assert providers.build_doc_status() is store
    assert providers.build_graph_indexer() is graph
    assert providers.build_retriever(RetrieverRole.GRAPH) is retriever


def test_llm_model_info_reads_the_catalog_without_a_client(mocker) -> None:
    llm = mocker.patch.object(providers_module, "BedrockLanguageModelFactory")
    config = Config()
    info = Providers(config).llm_model_info(config.search.answer_generation_model_id)
    assert info.context_window_size >= 200000
    llm.assert_not_called()


def test_llm_model_info_comes_from_a_local_factory() -> None:
    config = Config(
        providers={
            "llm": {
                "kind": "ollama",
                "default_model_id": "m",
                "context_window_size": 4096,
            }
        }
    )
    assert Providers(config).llm_model_info("m").context_window_size == 4096


def test_llm_model_info_falls_back_to_the_catalog_for_untyped_fakes() -> None:
    fake = MagicMock()
    fake.get_model_info.return_value = None
    config = Config()
    info = Providers(config, llm_factory=fake).llm_model_info(
        config.search.answer_generation_model_id
    )
    assert info.context_window_size >= 200000


# --- with backend: local nothing AWS is built --------------------------------


def test_local_bundle_selects_the_local_adapters(no_aws, tmp_path) -> None:
    providers = build_providers(_local_config(tmp_path))
    assert isinstance(providers.llm_factory, OllamaLanguageModelFactory)
    assert isinstance(providers.embedding_factory, OllamaEmbeddingFactory)
    assert isinstance(providers.rerank_factory, NoRerankFactory)
    assert isinstance(providers.token_counter("qwen"), LocalTokenCounter)
    assert isinstance(providers.build_doc_status(), SQLiteDocStatusStore)
    # A new registry per call: runs sharing a bundle keep their own state.
    assert providers.build_doc_status() is not providers.build_doc_status()


def test_local_backend_reaches_every_query_component(no_aws, tmp_path) -> None:
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

    config = _local_config(tmp_path)
    chain = GraphRAGChain(
        config,
        retriever_builders={
            RetrieverRole.GRAPH: MagicMock,
            RetrieverRole.DOCUMENT: MagicMock,
        },
    )
    providers = chain.providers
    assert isinstance(chain.factory, OllamaLanguageModelFactory)
    assert isinstance(chain.token_manager._token_counter, LocalTokenCounter)
    assert chain.memory_manager.providers is providers
    for strategy in registered_strategies():
        instance = chain._get_strategy_instance(strategy)
        assert instance.providers is providers, strategy
        assert isinstance(instance.hybrid_scorer.rerank_factory, NoRerankFactory)
        assert instance.hybrid_scorer.rerank_model is None
        assert isinstance(instance.token_manager._token_counter, LocalTokenCounter)


def test_local_backend_builds_the_store_retrievers(no_aws, tmp_path, mocker) -> None:
    from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
    from unified_kg_rag.adapters.retrievers.opensearch_retriever import (
        OpenSearchRetriever,
    )
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

    chain = GraphRAGChain(_local_config(tmp_path))
    graph = chain._build_retriever(RetrieverRole.GRAPH)
    document = chain._build_retriever(RetrieverRole.DOCUMENT)
    assert isinstance(graph, NeptuneRetriever)
    assert isinstance(document, OpenSearchRetriever)
    assert isinstance(document._embedding_factory, OllamaEmbeddingFactory)


def test_local_backend_reaches_every_ingestion_component(no_aws, tmp_path) -> None:
    from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
    from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline

    config = _local_config(tmp_path)
    config.processing.translation.enabled = True
    config.processing.gleaning.enabled = True
    config.processing.claim_extraction.enabled = True
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(local_directory=str(tmp_path / "cache")),
        source_directory=tmp_path,
    )
    stages = {pipeline.name_to_type_map[s.name]: s for s in pipeline.stages}
    providers = pipeline.providers
    llm = providers.llm_factory
    assert isinstance(llm, OllamaLanguageModelFactory)
    for stage_type in DataIngestionPipeline.PROVIDER_STAGES:
        assert stages[stage_type].providers is providers, stage_type

    T = PipelineStageType
    assert stages[T.TEXT_CHUNKING].chunker.factory is llm
    assert stages[T.TRANSLATION].translator.factory is llm
    assert stages[T.GRAPH_EXTRACTION].extractor.factory is llm
    assert stages[T.GLEANING].gleaner.factory is llm
    assert stages[T.CLAIM_EXTRACTION].extractor.factory is llm
    assert stages[T.GRAPH_RESOLUTION].description_summarizer.factory is llm
    assert stages[T.COMMUNITY_DETECTION].detector.factory is llm
    manager = stages[T.INDEXING].indexing_manager
    assert isinstance(manager.opensearch_indexer, OpenSearchIndexer)
    assert isinstance(manager.neptune_indexer, NeptuneIndexer)
    assert manager.opensearch_indexer.embedding_factory is providers.embedding_factory
    assert manager.description_summarizer.factory is llm
    # The registry the pipeline injects is the configured SQLite store.
    assert isinstance(stages[T.DOCUMENT_LOADING]._doc_status, SQLiteDocStatusStore)
    # One registry per pipeline, shared by its stages.
    assert stages[T.INDEXING]._doc_status is stages[T.DOCUMENT_LOADING]._doc_status
    pipeline.close()


def test_local_backend_reaches_evaluators_and_prompt_tuning(no_aws, tmp_path) -> None:
    from unified_kg_rag.application.prompts.tuner import PromptTuner
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
    from unified_kg_rag.evaluation import EvaluationManager

    config = _local_config(tmp_path)
    chain = GraphRAGChain(config)
    manager = EvaluationManager(config, rag_chain=chain)
    assert manager.providers is chain.providers
    assert set(manager.evaluators) == set(config.evaluation.enabled_evaluators)
    for evaluator in manager.evaluators.values():
        if hasattr(evaluator, "providers"):
            assert evaluator.providers is chain.providers
    ragas = manager.evaluators[
        next(e for e in manager.evaluators if e.value == "ragas")
    ]
    assert isinstance(ragas._token_counter, LocalTokenCounter)

    tuner = PromptTuner(config)
    assert isinstance(tuner.factory, OllamaLanguageModelFactory)


def test_standalone_components_follow_the_config(no_aws, tmp_path) -> None:
    # Built without a bundle, each component still resolves config.providers.
    from unified_kg_rag.adapters.ingestion.chunker import IntelligentTextChunker
    from unified_kg_rag.adapters.retrieval.memory_manager import MemoryManager
    from unified_kg_rag.adapters.retrieval.token_manager import TokenManager

    config = _local_config(tmp_path)
    assert isinstance(
        IntelligentTextChunker(config).factory, OllamaLanguageModelFactory
    )
    assert isinstance(TokenManager(config)._token_counter, LocalTokenCounter)
    assert isinstance(
        MemoryManager(config).providers.llm_factory, OllamaLanguageModelFactory
    )


# --- CLI overrides ------------------------------------------------------------


def test_overrides_apply_the_local_preset() -> None:
    config = with_provider_overrides(Config(), backend="local", llm_model_id="my-model")
    assert config.backend is Backend.LOCAL
    assert config.providers.llm.kind == "ollama"
    assert config.search.answer_generation_model_id == "my-model"
    assert config.aws.neptune.use_iam is False


def test_overrides_keep_what_the_file_set() -> None:
    base = Config(
        logging={"level": "DEBUG"}, search={"answer_generation_model_id": "x"}
    )
    config = with_provider_overrides(
        base,
        llm_kind="openai_compatible",
        llm_model_id="served",
        base_url="http://model-server.test/v1",
        embedding_kind="openai_compatible",
        embedding_model_id="embed",
        embedding_dimension=16,
        rerank_kind="none",
    )
    assert config.logging.level == "DEBUG"
    assert config.search.answer_generation_model_id == "x"
    assert config.processing.chunking.chunking_model_id == "served"
    assert config.providers.embedding.base_url == "http://model-server.test/v1"
    assert config.indexing.opensearch.embedding_model_id == "embed"
    assert config.providers.rerank.kind == "none"


def test_no_overrides_returns_the_same_config() -> None:
    config = Config()
    assert with_provider_overrides(config) is config


def test_invalid_overrides_are_configuration_errors() -> None:
    with pytest.raises(ConfigurationError, match="providers.embedding.model_id"):
        with_provider_overrides(Config(), embedding_kind="ollama")


def test_the_guard_catches_the_aws_defaults(no_aws) -> None:
    # Guards the guard: the AWS defaults do trip it.
    providers = Providers(Config())
    with pytest.raises(AssertionError):
        providers.llm_factory
    with pytest.raises(AssertionError):
        providers.token_counter("anthropic.claude-sonnet-5-5")
    with pytest.raises(AssertionError):
        providers.build_doc_status()
