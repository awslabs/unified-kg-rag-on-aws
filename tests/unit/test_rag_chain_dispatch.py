# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GraphRAGChain strategy/retriever dispatch (role-based wiring).

These cover the consumer side of the registry refactor: that
``_get_strategy_instance`` builds exactly the retrievers each strategy declares
via its ``StrategySpec.required_roles`` and injects them keyed by role value.
``_get_retriever`` is stubbed so no AWS clients are created.
"""

from __future__ import annotations

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain
from unified_kg_rag.domain.models import Config, RetrieverRole, SearchStrategy
from unified_kg_rag.domain.retrieval.strategy_registry import (
    get_strategy_spec,
    registered_strategies,
)

pytestmark = pytest.mark.unit


class _SentinelRetriever:
    """Stand-in for a retriever, tagged by the role it was requested for."""

    def __init__(self, role: RetrieverRole) -> None:
        self.role = role


@pytest.fixture
def chain(config: Config, monkeypatch: pytest.MonkeyPatch) -> GraphRAGChain:
    instance = GraphRAGChain(config=config)
    monkeypatch.setattr(
        instance, "_get_retriever", lambda role: _SentinelRetriever(role)
    )
    return instance


@pytest.mark.parametrize("strategy", list(registered_strategies()))
def test_built_retrievers_match_spec(
    chain: GraphRAGChain, strategy: SearchStrategy
) -> None:
    spec = get_strategy_spec(strategy)
    instance = chain._get_strategy_instance(strategy)

    expected_keys = {role.value for role in spec.required_roles}
    assert set(instance.retrievers.keys()) == expected_keys
    for role in spec.required_roles:
        retriever = instance.retrievers[role.value]
        assert retriever is not None
        assert retriever.role is role


def test_simple_strategy_gets_only_document_role(chain: GraphRAGChain) -> None:
    instance = chain._get_strategy_instance(SearchStrategy.SIMPLE)
    assert set(instance.retrievers.keys()) == {RetrieverRole.DOCUMENT.value}
    # The base-class role accessor resolves it; graph role is absent.
    assert instance.document_retriever is not None
    assert instance.graph_retriever is None


def test_graph_strategy_gets_both_roles(chain: GraphRAGChain) -> None:
    instance = chain._get_strategy_instance(SearchStrategy.LOCAL)
    assert set(instance.retrievers.keys()) == {
        RetrieverRole.DOCUMENT.value,
        RetrieverRole.GRAPH.value,
    }
    assert instance.graph_retriever is not None
    assert instance.document_retriever is not None


def test_instance_type_matches_registered_class(chain: GraphRAGChain) -> None:
    for strategy in registered_strategies():
        spec = get_strategy_spec(strategy)
        instance = chain._get_strategy_instance(strategy)
        assert isinstance(instance, spec.strategy_class)


# --- Provider/backend injection seams (hexagonal — custom/local backends) ----


class _FakeModelFactory:
    """A minimal LLMFactoryPort stand-in (structural typing — no Bedrock)."""

    def get_model(self, model_id, **kwargs):  # noqa: ANN001, ANN003
        return object()

    def get_model_info(self, model_id):  # noqa: ANN001
        return None


def test_injected_model_factory_replaces_bedrock(config: Config) -> None:
    fake = _FakeModelFactory()
    chain = GraphRAGChain(config=config, model_factory=fake)
    # The injected factory is used verbatim instead of constructing Bedrock.
    assert chain.factory is fake


def test_injected_retriever_builder_overrides_role(config: Config) -> None:
    sentinel = _SentinelRetriever(RetrieverRole.GRAPH)
    chain = GraphRAGChain(
        config=config,
        retriever_builders={RetrieverRole.GRAPH: lambda: sentinel},
    )
    # The custom builder wins over the default Neptune builder for that role.
    assert chain._get_retriever(RetrieverRole.GRAPH) is sentinel


def test_default_backends_unchanged_without_injection(config: Config) -> None:
    # No injection -> AWS defaults remain (Bedrock factory instance present).
    chain = GraphRAGChain(config=config)
    assert chain.factory is not None
    assert chain._retriever_builders_override == {}
