# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model-factory port DI seam.

A conforming ModelFactoryPort can be injected into the embedding/rerank call
sites so no Bedrock client is constructed — proving the provider boundary is
real (a non-Bedrock provider needs only a conforming factory, no call-site
edits). Verifies structural conformance and that injection short-circuits the
default Bedrock construction.
"""

from __future__ import annotations

from typing import Any

import pytest

from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import Config
from unified_kg_rag.ports.model_factory import (
    EmbeddingFactoryPort,
    ModelFactoryPort,
    RerankFactoryPort,
)

pytestmark = pytest.mark.unit


class _FakeEmbeddings:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.0, 0.1, 0.2] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.1, 0.2]


class _FakeModelInfo:
    dimensions = 3


class FakeEmbeddingFactory:
    """A non-Bedrock embedding factory conforming to EmbeddingFactoryPort."""

    def __init__(self) -> None:
        self.get_model_calls = 0

    def get_model(self, model_id: Any, **kwargs: Any) -> _FakeEmbeddings:
        self.get_model_calls += 1
        return _FakeEmbeddings()

    def get_model_info(self, model_id: Any) -> _FakeModelInfo:
        return _FakeModelInfo()


class FakeRerankFactory:
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return object()

    def get_model_info(self, model_id: Any) -> Any:
        return None


def test_fake_factories_conform_to_port() -> None:
    # runtime_checkable Protocol: the fakes structurally satisfy the boundary.
    assert isinstance(FakeEmbeddingFactory(), ModelFactoryPort)
    assert isinstance(FakeEmbeddingFactory(), EmbeddingFactoryPort)
    assert isinstance(FakeRerankFactory(), RerankFactoryPort)


def test_opensearch_indexer_uses_injected_embedding_factory(mocker) -> None:
    # Injecting the factory must short-circuit Bedrock construction entirely:
    # the constructor builds no Bedrock embedding factory and no AWS embed call
    # goes through Bedrock. We patch the OpenSearch client (network) but NOT the
    # embedding factory — that comes from injection.
    mocker.patch(
        "unified_kg_rag.adapters.storage.opensearch_indexer.OpenSearchClient",
        return_value=mocker.MagicMock(),
    )
    bedrock_factory = mocker.patch(
        "unified_kg_rag.adapters.storage.opensearch_indexer."
        "BedrockEmbeddingModelFactory"
    )
    fake = FakeEmbeddingFactory()

    indexer = OpenSearchIndexer(
        config=Config(),
        boto_session=mocker.MagicMock(),
        embedding_factory=fake,
    )

    assert indexer.embedding_factory is fake
    # The Bedrock factory was never constructed (the port short-circuited it).
    bedrock_factory.assert_not_called()
    # The dimension was resolved through the injected factory's get_model_info.
    assert indexer._embedding_dimension == 3


def test_hybrid_scorer_uses_injected_rerank_factory() -> None:
    config = Config()
    config.search.reranking.enabled = True
    fake = FakeRerankFactory()
    scorer = HybridScorer(config=config, rerank_factory=fake)
    # The injected factory is used; its get_model produced the rerank model.
    assert scorer.rerank_factory is fake
    assert scorer.rerank_model is not None
