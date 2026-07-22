# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BedrockNodeEmbedder (visualization) — AWS-free via an injected
fake embedding factory (EmbeddingFactoryPort)."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from unified_kg_rag.domain.models import Config
from unified_kg_rag.visualization.embeddings.node2vec import (
    BedrockNodeEmbedder,
    NodeEmbeddings,
)

pytestmark = pytest.mark.unit


class _FakeEmbeddingModel:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.seen: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.seen.extend(texts)
        # deterministic non-random vectors so assertions are stable
        return [[float(len(t))] * self.dim for t in texts]


class _FakeEmbeddingFactory:
    """Structurally an EmbeddingFactoryPort (get_model / get_model_info)."""

    def __init__(self, dim: int = 4) -> None:
        self.model = _FakeEmbeddingModel(dim)
        self.dim = dim

    def get_model_info(self, model_id):  # noqa: ANN001
        return SimpleNamespace(dimensions=self.dim)

    def get_model(self, model_id, **kwargs):  # noqa: ANN001, ANN003
        return self.model


@pytest.fixture
def embedder(config: Config) -> BedrockNodeEmbedder:
    return BedrockNodeEmbedder(config=config, embedding_factory=_FakeEmbeddingFactory())


def test_empty_graph_returns_empty_embeddings(embedder) -> None:
    out = embedder.generate_embeddings(nx.Graph())
    assert isinstance(out, NodeEmbeddings)
    assert out.nodes == [] and out.embeddings == {}


def test_embeds_every_node(embedder) -> None:
    g = nx.Graph()
    g.add_node("e1", name="Alice", description="a person")
    g.add_node("e2", name="Acme", description="a company")
    g.add_edge("e1", "e2")
    out = embedder.generate_embeddings(g)
    assert set(out.nodes) == {"e1", "e2"}
    assert all(isinstance(v, np.ndarray) for v in out.embeddings.values())
    assert all(v.shape == (4,) for v in out.embeddings.values())


def test_embedding_text_includes_name_and_description(embedder) -> None:
    g = nx.Graph()
    g.add_node("e1", name="Alice", description="a person")
    embedder.generate_embeddings(g)
    # The text fed to the model is "name: description".
    assert embedder.embedding_model.seen == ["Alice: a person"]


def test_fallback_to_random_on_embed_error(config: Config, mocker) -> None:
    factory = _FakeEmbeddingFactory()
    embedder = BedrockNodeEmbedder(config=config, embedding_factory=factory)
    mocker.patch.object(
        embedder.embedding_model,
        "embed_documents",
        side_effect=RuntimeError("bedrock down"),
    )
    g = nx.Graph()
    g.add_node("e1", name="Alice")
    out = embedder.generate_embeddings(g)
    # Degrades to random embeddings of the right dimension rather than raising.
    assert out.nodes == ["e1"]
    assert out.embeddings["e1"].shape == (4,)


def test_unsupported_model_raises(config: Config) -> None:
    class _NoInfoFactory(_FakeEmbeddingFactory):
        def get_model_info(self, model_id):  # noqa: ANN001
            return None

    with pytest.raises(ValueError, match="Unsupported Bedrock model"):
        BedrockNodeEmbedder(config=config, embedding_factory=_NoInfoFactory())
