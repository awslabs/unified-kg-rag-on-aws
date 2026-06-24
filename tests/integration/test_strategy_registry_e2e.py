# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end search-strategy registry tests (AWS-free).

Asserts every registered strategy is resolvable with valid retriever roles and
that a strategy runs end to end through the registry against fake retrievers —
covering the role-based injection seam without Bedrock/Neptune/OpenSearch.
"""

from __future__ import annotations

import pytest

import aws_graphrag.adapters.search_strategies  # noqa: F401  (triggers registration)
from aws_graphrag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from aws_graphrag.domain.retrieval.strategy_registry import (
    get_strategy_spec,
    registered_strategies,
)

pytestmark = pytest.mark.integration

_VALID_ROLES = {RetrieverRole.DOCUMENT, RetrieverRole.GRAPH}


class FakeRetriever:
    def __init__(self, tag: str) -> None:
        self.tag = tag

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                content=f"{self.tag} hit",
                score=1.0,
                source=f"{self.tag}-1",
                retriever_type=self.tag,
                metadata={"id": f"{self.tag}-id"},
            )
        ]


def test_all_registered_strategies_have_valid_specs() -> None:
    strategies = registered_strategies()
    assert len(strategies) >= 7  # simple/local/global/drift + mix/hybrid/naive
    for strategy in strategies:
        spec = get_strategy_spec(strategy)
        assert spec.strategy_class is not None
        assert spec.required_roles, f"{strategy} declares no required roles"
        assert set(spec.required_roles) <= _VALID_ROLES


def test_every_non_auto_enum_value_is_registered() -> None:
    registered = {s.value for s in registered_strategies()}
    for member in SearchStrategy:
        if member is SearchStrategy.AUTO:
            continue  # AUTO is an LLM router, not a concrete strategy
        assert member.value in registered, f"{member.value} not registered"


async def test_simple_strategy_runs_through_registry() -> None:
    config = Config()
    config.search.reranking.enabled = False
    spec = get_strategy_spec(SearchStrategy.SIMPLE)

    # Inject only the roles the strategy declares it needs.
    retrievers = {role.value: FakeRetriever(role.value) for role in spec.required_roles}
    strategy = spec.strategy_class(config=config, retrievers=retrievers)
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None: [
            r for results in results_dict.values() for r in results
        ]
    )

    result = await strategy.asearch(SearchQuery(query="what is graphrag?"))
    assert result.search_strategy == "simple_search"
    assert len(result.results) >= 1
