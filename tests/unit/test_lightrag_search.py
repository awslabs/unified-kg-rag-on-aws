# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for the LightRAG dual-level search strategy (M3).

Uses fake retrievers (no AWS) and a stubbed HybridScorer to assert that each
mode (naive / hybrid / mix) queries the correct indices with the correct
keyword lists, fusing through the shared scorer.
"""

from __future__ import annotations

import pytest

import aws_graphrag.adapters.search_strategies  # noqa: F401
from aws_graphrag.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from aws_graphrag.retrieval.strategy_registry import get_strategy_spec

pytestmark = pytest.mark.unit


class FakeRetriever:
    """Records the index_prefixes/query of each aretrieve call."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.calls: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        prefix = query.index_prefixes[0] if query.index_prefixes else self.tag
        return [
            RetrievalResult(
                content=f"{prefix} result",
                score=1.0,
                source=f"{prefix}-1",
                retriever_type=self.tag,
                metadata={"id": f"{prefix}-id"},
            )
        ]


def _make_strategy(config: Config):
    spec = get_strategy_spec(SearchStrategy.MIX)
    os_r, neptune_r = FakeRetriever("document"), FakeRetriever("graph")
    strategy = spec.strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: os_r,
            RetrieverRole.GRAPH.value: neptune_r,
        },
    )
    # Stub the shared scorer to just flatten the source dict (avoid Bedrock).
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None: [
            r for results in results_dict.values() for r in results
        ]
    )
    return strategy, os_r, neptune_r


def _query(mode: SearchStrategy, **kw) -> SearchQuery:
    return SearchQuery(query="q", metadata={"lightrag_mode": mode.value}, **kw)


async def test_naive_mode_queries_only_text_units(config: Config) -> None:
    strategy, os_r, neptune_r = _make_strategy(config)
    result = await strategy.asearch(
        _query(SearchStrategy.NAIVE, ll_keywords=["x"], hl_keywords=["y"])
    )
    # Naive ignores keywords/graph; one text-units retrieval, no Neptune.
    prefixes = [q.index_prefixes[0] for q in os_r.calls]
    assert prefixes == [config.indexing.opensearch.text_units_index_prefix]
    assert neptune_r.calls == []
    assert result.search_strategy == "lightrag_naive"


async def test_hybrid_mode_uses_entities_and_relationships(config: Config) -> None:
    strategy, os_r, neptune_r = _make_strategy(config)
    await strategy.asearch(
        _query(SearchStrategy.HYBRID, ll_keywords=["alice"], hl_keywords=["theme"])
    )
    prefixes = {q.index_prefixes[0] for q in os_r.calls}
    assert config.indexing.opensearch.entities_index_prefix in prefixes
    assert config.indexing.opensearch.relationships_index_prefix in prefixes
    # text-units (naive blend) NOT queried in hybrid mode.
    assert config.indexing.opensearch.text_units_index_prefix not in prefixes
    # Entity hits seed a Neptune expansion.
    assert len(neptune_r.calls) == 1


async def test_ll_keywords_go_to_entities_index(config: Config) -> None:
    strategy, os_r, _ = _make_strategy(config)
    await strategy.asearch(_query(SearchStrategy.HYBRID, ll_keywords=["alice", "bob"]))
    entity_calls = [
        q
        for q in os_r.calls
        if q.index_prefixes[0] == config.indexing.opensearch.entities_index_prefix
    ]
    assert entity_calls and "alice, bob" == entity_calls[0].query


async def test_mix_mode_blends_chunks(config: Config) -> None:
    strategy, os_r, neptune_r = _make_strategy(config)
    await strategy.asearch(
        _query(SearchStrategy.MIX, ll_keywords=["x"], hl_keywords=["y"])
    )
    prefixes = {q.index_prefixes[0] for q in os_r.calls}
    # mix = hybrid (entities+relationships) PLUS naive chunks.
    assert {
        config.indexing.opensearch.entities_index_prefix,
        config.indexing.opensearch.relationships_index_prefix,
        config.indexing.opensearch.text_units_index_prefix,
    } <= prefixes


async def test_empty_keywords_fall_back_to_raw_query(config: Config) -> None:
    strategy, os_r, neptune_r = _make_strategy(config)
    # Short hybrid query with no extracted keywords -> raw query forced as an
    # ll_keyword (LightRAG behavior), so entities ARE queried (no total miss).
    await strategy.asearch(_query(SearchStrategy.HYBRID))
    entity_calls = [
        q
        for q in os_r.calls
        if q.index_prefixes[0] == config.indexing.opensearch.entities_index_prefix
    ]
    assert entity_calls and entity_calls[0].query == "q"


async def test_long_empty_keyword_query_does_not_fall_back(config: Config) -> None:
    strategy, os_r, neptune_r = _make_strategy(config)
    # Exceeds config.search.lightrag_search.raw_query_fallback_max_len (default 50).
    long_query = "x" * 60
    await strategy.asearch(
        SearchQuery(query=long_query, metadata={"lightrag_mode": "hybrid"})
    )
    # No fallback -> no graph retrieval for a long keyword-less query.
    assert os_r.calls == []
    assert neptune_r.calls == []
