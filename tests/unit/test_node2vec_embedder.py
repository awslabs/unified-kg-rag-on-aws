# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BedrockNodeEmbedder (visualization) — AWS-free via an injected
fake embedding factory (EmbeddingFactoryPort)."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import EmbeddingModelError
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


def _failing_embedder(config: Config, mocker) -> BedrockNodeEmbedder:  # noqa: ANN001
    embedder = BedrockNodeEmbedder(
        config=config, embedding_factory=_FakeEmbeddingFactory()
    )
    mocker.patch.object(
        embedder.embedding_model,
        "embed_documents",
        side_effect=RuntimeError("bedrock down"),
    )
    return embedder


def _one_node_graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node("e1", name="Alice")
    return g


def test_embed_error_raises_by_default(config: Config, mocker) -> None:
    # Fail fast (ignore_errors=False): random vectors would produce a
    # meaningless layout that still looks valid.
    embedder = _failing_embedder(config, mocker)
    with pytest.raises(EmbeddingModelError, match="bedrock down"):
        embedder.generate_embeddings(_one_node_graph())


def test_embed_error_with_ignore_errors_is_marked_degraded(
    config: Config, mocker, caplog
) -> None:
    config.processing.ignore_errors = True
    embedder = _failing_embedder(config, mocker)
    with caplog.at_level(logging.ERROR):
        out = embedder.generate_embeddings(_one_node_graph())
    # No random substitute: an empty result the layout treats as degraded.
    assert out.nodes == [] and out.embeddings == {}
    assert any("DEGRADED" in r.getMessage() for r in caplog.records)


def test_successful_embeddings_are_not_empty(embedder) -> None:
    assert embedder.generate_embeddings(_one_node_graph()).embeddings


def test_unsupported_model_raises(config: Config) -> None:
    class _NoInfoFactory(_FakeEmbeddingFactory):
        def get_model_info(self, model_id):  # noqa: ANN001
            return None

    with pytest.raises(ValueError, match="Unsupported Bedrock model"):
        BedrockNodeEmbedder(config=config, embedding_factory=_NoInfoFactory())


def test_yaml_string_model_id_is_coerced_to_the_enum() -> None:
    # The documented YAML form is a plain string; it used to reach the embedder
    # as str and fail on '.value', silently dropping the visualization.
    from unified_kg_rag.domain.models import EmbeddingModelId

    config = Config.model_validate(
        {
            "graph": {
                "visualization": {
                    "embeddings": {"bedrock_model_id": "amazon.titan-embed-text-v1"}
                }
            }
        }
    )
    model_id = config.graph.visualization.embeddings["bedrock_model_id"]
    assert model_id is EmbeddingModelId.TITAN_EMBED_V1
    BedrockNodeEmbedder(config=config, embedding_factory=_FakeEmbeddingFactory())


def test_unknown_model_id_is_rejected_at_config_load() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="bedrock_model_id"):
        Config.model_validate(
            {"graph": {"visualization": {"embeddings": {"bedrock_model_id": "nope"}}}}
        )
