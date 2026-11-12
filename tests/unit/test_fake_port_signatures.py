# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fakes and the adapters must keep the exact call shape of their ports.

They are duck-typed (they do not subclass the ports), so nothing else stops a
port parameter rename from leaving one behind: tests would keep passing
against a call shape the real adapters no longer accept.
"""

from __future__ import annotations

import inspect

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.graph_query import FakeGraphQueryRetriever
from tests.fixtures.fakes.retrievers import FakeRetriever, StaticRetriever
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from tests.fixtures.fakes.tokens import WordTokenCounter
from tests.unit.test_query_guardrail import _RecordingGuard
from unified_kg_rag.adapters.aws.bedrock import (
    BedrockEmbeddingModelFactory,
    BedrockLanguageModelFactory,
    BedrockRerankModelFactory,
)
from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient
from unified_kg_rag.adapters.aws.token_counter import BedrockTokenCounter
from unified_kg_rag.adapters.doc_status.dynamodb import DynamoDBDocStatusStore
from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore
from unified_kg_rag.adapters.guardrail.bedrock import BedrockGuardrail
from unified_kg_rag.adapters.local.embeddings import (
    OllamaEmbeddingFactory,
    OpenAICompatibleEmbeddingFactory,
)
from unified_kg_rag.adapters.local.models import (
    OllamaLanguageModelFactory,
    OpenAICompatibleLanguageModelFactory,
)
from unified_kg_rag.adapters.local.rerank import (
    CrossEncoderRerankFactory,
    NoRerankFactory,
)
from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
from unified_kg_rag.adapters.local.token_counter import LocalTokenCounter
from unified_kg_rag.adapters.retrieval.base import BaseGraphRAGRetriever
from unified_kg_rag.adapters.retrievers.graph_expansion import GraphExpansionRetriever
from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from unified_kg_rag.adapters.retrievers.sqlite_graph_retriever import (
    SQLiteGraphRetriever,
)
from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.adapters.storage.search_engine import SearchEngineClient
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
from unified_kg_rag.domain.ingestion.delta_detector import DocStatusReader
from unified_kg_rag.ports import (
    ContentGuardPort,
    DocStatusPort,
    GraphIndexer,
    GraphQueryPort,
    ModelFactoryPort,
    TokenCounterPort,
    VectorIndexer,
)


def _port_methods(port: type) -> list[str]:
    """Public instance methods of ``port`` (class/static helpers excluded).

    For the retriever base, only the ones its own source defines: the rest
    are LangChain's ``BaseRetriever`` (and Pydantic's) surface, which the
    strategies never call.
    """
    own = port in _OWN_METHODS_ONLY
    return sorted(
        name
        for name in (vars(port) if own else dir(port))
        if not name.startswith("_")
        and inspect.isfunction(method := inspect.getattr_static(port, name))
        and (not own or method.__module__ == port.__module__)
    )


_OWN_METHODS_ONLY = (BaseGraphRAGRetriever,)


def _shape(func: object) -> list[tuple[str, inspect._ParameterKind, bool]]:
    return [
        (p.name, p.kind, p.default is not inspect.Parameter.empty)
        for p in inspect.signature(func).parameters.values()  # type: ignore[arg-type]
    ]


PAIRS = [
    (FakeGraphStore, GraphIndexer),
    (SQLiteGraphIndexer, GraphIndexer),
    (NeptuneIndexer, GraphIndexer),
    (FakeVectorStore, VectorIndexer),
    (OpenSearchIndexer, VectorIndexer),
    (FakeDocStatusStore, DocStatusPort),
    (SQLiteDocStatusStore, DocStatusPort),
    (DynamoDBDocStatusStore, DocStatusPort),
    # The domain's view of the registry is the port's read subset.
    (DocStatusPort, DocStatusReader),
    (ScriptedLLMFactory, ModelFactoryPort),
    (HashingEmbeddingFactory, ModelFactoryPort),
    (OpenAICompatibleLanguageModelFactory, ModelFactoryPort),
    (OllamaLanguageModelFactory, ModelFactoryPort),
    (OpenAICompatibleEmbeddingFactory, ModelFactoryPort),
    (OllamaEmbeddingFactory, ModelFactoryPort),
    (CrossEncoderRerankFactory, ModelFactoryPort),
    (NoRerankFactory, ModelFactoryPort),
    (BedrockLanguageModelFactory, ModelFactoryPort),
    (BedrockEmbeddingModelFactory, ModelFactoryPort),
    (BedrockRerankModelFactory, ModelFactoryPort),
    (WordTokenCounter, TokenCounterPort),
    (LocalTokenCounter, TokenCounterPort),
    (BedrockTokenCounter, TokenCounterPort),
    (OpenSearchClient, SearchEngineClient),
    (SQLiteSearchEngine, SearchEngineClient),
    (BedrockGuardrail, ContentGuardPort),
    (FakeRetriever, BaseGraphRAGRetriever),
    # The concrete retrievers override the base's methods; a strategy calls
    # them through the base's shapes.
    (OpenSearchRetriever, BaseGraphRAGRetriever),
    (NeptuneRetriever, BaseGraphRAGRetriever),
    (SQLiteGraphRetriever, BaseGraphRAGRetriever),
    (GraphExpansionRetriever, BaseGraphRAGRetriever),
    (FakeGraphQueryRetriever, BaseGraphRAGRetriever),
    (FakeGraphQueryRetriever, GraphQueryPort),
    (NeptuneRetriever, GraphQueryPort),
    (SQLiteGraphRetriever, GraphQueryPort),
]


@pytest.mark.parametrize(
    ("fake", "port", "method"),
    [
        pytest.param(fake, port, method, id=f"{fake.__name__}.{method}")
        for fake, port in PAIRS
        for method in _port_methods(port)
    ],
)
def test_fake_matches_port_signature(fake: type, port: type, method: str) -> None:
    assert hasattr(fake, method), f"{fake.__name__} lacks port method {method}"
    assert _shape(getattr(fake, method)) == _shape(getattr(port, method))


@pytest.mark.parametrize(("fake", "port"), PAIRS)
def test_port_surface_is_not_empty(fake: type, port: type) -> None:
    # Guards the discovery above: an empty list would make the parametrized
    # test vacuously pass. The smallest port (the guard) has one method.
    assert len(_port_methods(port)) >= 1


# Fakes that stand in for a port with only the methods their tests call:
# each of those keeps the port's shape.
PARTIAL = [(StaticRetriever, BaseGraphRAGRetriever)]


@pytest.mark.parametrize(
    ("fake", "port", "method"),
    [
        pytest.param(fake, port, method, id=f"{fake.__name__}.{method}")
        for fake, port in PARTIAL
        for method in _port_methods(port)
        if method in vars(fake)
    ],
)
def test_a_partial_fakes_methods_match_the_port(
    fake: type, port: type, method: str
) -> None:
    assert _shape(getattr(fake, method)) == _shape(getattr(port, method))


@pytest.mark.parametrize(("fake", "port"), PARTIAL)
def test_a_partial_fake_implements_some_port_method(fake: type, port: type) -> None:
    assert set(_port_methods(port)) & set(vars(fake))


def test_recording_guard_accepts_every_port_call() -> None:
    # It takes keyword arguments as **kwargs to record them, so its shape
    # differs from the port's on purpose; it must still accept every call
    # the port allows.
    guard = inspect.signature(_RecordingGuard.acheck)
    port = inspect.signature(ContentGuardPort.acheck).parameters.values()
    positional = [p.name for p in port if p.kind is p.POSITIONAL_OR_KEYWORD]
    keyword = {p.name: None for p in port if p.kind is p.KEYWORD_ONLY}
    assert keyword, "the port's keyword arguments moved"
    guard.bind(*positional, **keyword)
    guard.bind(*positional)
