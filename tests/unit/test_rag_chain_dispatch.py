# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for GraphRAGChain strategy/retriever dispatch (M1 registry wiring).

These cover the consumer side of the registry refactor: that
``_get_strategy_instance`` builds exactly the retrievers each strategy declares
via its ``StrategySpec.required_retrievers`` and injects them into the strategy
instance. ``_get_retriever`` is stubbed so no AWS clients are created.
"""
from __future__ import annotations

import pytest

import aws_graphrag.retrieval.search_strategies  # noqa: F401  (registers strategies)
from aws_graphrag.models import Config, RetrieverType, SearchStrategy
from aws_graphrag.retrieval.rag_chain import GraphRAGChain
from aws_graphrag.retrieval.strategy_registry import (
    get_strategy_spec,
    registered_strategies,
)

pytestmark = pytest.mark.unit


class _SentinelRetriever:
    """Stand-in for a retriever, tagged by the type it was requested for."""

    def __init__(self, retriever_type: RetrieverType) -> None:
        self.retriever_type = retriever_type


@pytest.fixture
def chain(config: Config, monkeypatch: pytest.MonkeyPatch) -> GraphRAGChain:
    instance = GraphRAGChain(config=config)
    monkeypatch.setattr(
        instance,
        "_get_retriever",
        lambda retriever_type: _SentinelRetriever(retriever_type),
    )
    return instance


@pytest.mark.parametrize("strategy", list(registered_strategies()))
def test_built_retrievers_match_spec(
    chain: GraphRAGChain, strategy: SearchStrategy
) -> None:
    spec = get_strategy_spec(strategy)
    instance = chain._get_strategy_instance(strategy)

    expected_keys = {rt.value for rt in spec.required_retrievers}
    assert set(instance.retrievers.keys()) == expected_keys
    # Every injected retriever is non-None and tagged with the right type.
    for rt in spec.required_retrievers:
        retriever = instance.retrievers[rt.value]
        assert retriever is not None
        assert retriever.retriever_type is rt


def test_simple_strategy_gets_only_opensearch(chain: GraphRAGChain) -> None:
    instance = chain._get_strategy_instance(SearchStrategy.SIMPLE)
    assert set(instance.retrievers.keys()) == {RetrieverType.OPENSEARCH.value}


def test_graph_strategy_gets_both_retrievers(chain: GraphRAGChain) -> None:
    instance = chain._get_strategy_instance(SearchStrategy.LOCAL)
    assert set(instance.retrievers.keys()) == {
        RetrieverType.OPENSEARCH.value,
        RetrieverType.NEPTUNE.value,
    }


def test_instance_type_matches_registered_class(chain: GraphRAGChain) -> None:
    for strategy in registered_strategies():
        spec = get_strategy_spec(strategy)
        instance = chain._get_strategy_instance(strategy)
        assert isinstance(instance, spec.strategy_class)
