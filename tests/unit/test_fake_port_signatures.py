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
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.aws.bedrock import (
    BedrockEmbeddingModelFactory,
    BedrockLanguageModelFactory,
    BedrockRerankModelFactory,
)
from unified_kg_rag.adapters.aws.dynamodb import DynamoDBDocStatusStore
from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient
from unified_kg_rag.adapters.aws.token_counter import BedrockTokenCounter
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
from unified_kg_rag.adapters.local.sqlite_doc_status import SQLiteDocStatusStore
from unified_kg_rag.adapters.local.sqlite_graph import SQLiteGraphIndexer
from unified_kg_rag.adapters.local.token_counter import LocalTokenCounter
from unified_kg_rag.adapters.storage.search_engine import SearchEngineClient
from unified_kg_rag.ports import (
    DocStatusPort,
    GraphIndexer,
    ModelFactoryPort,
    TokenCounterPort,
    VectorIndexer,
)


def _port_methods(port: type) -> list[str]:
    """Public instance methods of ``port`` (class/static helpers excluded)."""
    return sorted(
        name
        for name in dir(port)
        if not name.startswith("_")
        and inspect.isfunction(inspect.getattr_static(port, name))
    )


def _shape(func: object) -> list[tuple[str, inspect._ParameterKind, bool]]:
    return [
        (p.name, p.kind, p.default is not inspect.Parameter.empty)
        for p in inspect.signature(func).parameters.values()  # type: ignore[arg-type]
    ]


PAIRS = [
    (FakeGraphStore, GraphIndexer),
    (SQLiteGraphIndexer, GraphIndexer),
    (FakeVectorStore, VectorIndexer),
    (FakeDocStatusStore, DocStatusPort),
    (SQLiteDocStatusStore, DocStatusPort),
    (DynamoDBDocStatusStore, DocStatusPort),
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
    (LocalTokenCounter, TokenCounterPort),
    (BedrockTokenCounter, TokenCounterPort),
    (OpenSearchClient, SearchEngineClient),
    (SQLiteSearchEngine, SearchEngineClient),
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
    # test vacuously pass. The smallest ports (model factory, token counter)
    # have two methods.
    assert len(_port_methods(port)) >= 2
