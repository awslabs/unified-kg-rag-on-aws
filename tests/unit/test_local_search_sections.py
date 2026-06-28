# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for local search's community-report + relationship sections.

MS GraphRAG local search builds context from entities + the community reports
those entities belong to + in-network relationships + text units. These tests
assert that local search now enriches its entity/text-unit core with a
community-report section (always, on the GraphRAG path) and a relationship
section (gated on the relationship vector index being built). Fake retrievers
stand in for OpenSearch / Neptune and the shared HybridScorer is stubbed to
flatten results, so Bedrock is never touched.
"""

from __future__ import annotations

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.domain.retrieval.strategy_registry import get_strategy_spec

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
                metadata={"id": f"{prefix}-id", "text_unit_ids": []},
            )
        ]


def _make_strategy(config: Config):
    spec = get_strategy_spec(SearchStrategy.LOCAL)
    os_r, neptune_r = FakeRetriever("document"), FakeRetriever("graph")
    strategy = spec.strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: os_r,
            RetrieverRole.GRAPH.value: neptune_r,
        },
    )
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None: [
            r for results in results_dict.values() for r in results
        ]
    )
    return strategy, os_r, neptune_r


def _query(**kw) -> SearchQuery:
    return SearchQuery(query="who founded acme", entity_focus=["Acme"], **kw)


def _doc_index_prefixes(os_r: FakeRetriever) -> list[str]:
    return [q.index_prefixes[0] for q in os_r.calls if q.index_prefixes]


async def test_community_reports_section_queried(config: Config) -> None:
    strategy, os_r, _ = _make_strategy(config)

    await strategy.asearch(_query())

    reports_prefix = config.indexing.opensearch.community_reports_index_prefix
    assert reports_prefix in _doc_index_prefixes(os_r)
    # The report query uses the entity focus, mirroring entity retrieval.
    report_calls = [q for q in os_r.calls if q.index_prefixes == [reports_prefix]]
    assert report_calls and report_calls[0].query == "Acme"


async def test_relationships_section_queried_when_vector_index_built(
    config: Config,
) -> None:
    config.indexing.opensearch.build_relationship_vector_index = True
    strategy, os_r, _ = _make_strategy(config)

    await strategy.asearch(_query())

    rel_prefix = config.indexing.opensearch.relationships_index_prefix
    assert rel_prefix in _doc_index_prefixes(os_r)


async def test_no_relationship_query_when_vector_index_disabled(
    config: Config,
) -> None:
    # GraphRAG-only deployment: relationship VECTOR index absent -> never queried.
    config.indexing.opensearch.build_relationship_vector_index = False
    strategy, os_r, _ = _make_strategy(config)

    await strategy.asearch(_query())

    rel_prefix = config.indexing.opensearch.relationships_index_prefix
    assert rel_prefix not in _doc_index_prefixes(os_r)


async def test_sections_fall_back_to_query_text_without_entity_focus(
    config: Config,
) -> None:
    config.indexing.opensearch.build_relationship_vector_index = True
    strategy, os_r, _ = _make_strategy(config)

    await strategy.asearch(SearchQuery(query="raw text", entity_focus=[]))

    reports_prefix = config.indexing.opensearch.community_reports_index_prefix
    rel_prefix = config.indexing.opensearch.relationships_index_prefix
    report_calls = [q for q in os_r.calls if q.index_prefixes == [reports_prefix]]
    rel_calls = [q for q in os_r.calls if q.index_prefixes == [rel_prefix]]
    assert report_calls and report_calls[0].query == "raw text"
    assert rel_calls and rel_calls[0].query == "raw text"
