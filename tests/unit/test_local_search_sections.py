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
        lambda results_dict, top_k, retrieval_multiplier=1, query=None, **_kw: [
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


def test_per_type_quota_is_configured_shares_of_top_k(config: Config) -> None:
    # The fusion quota is config, not constants: MS local's proportional context
    # assembly is expressed as per-type multipliers of top_k with a floor, so a
    # deployment that rebalances the shares must move the quota.
    strategy, _, _ = _make_strategy(config)
    quota_config = config.search.local_search.type_quota

    quota = strategy._per_type_quota(top_k=40)
    assert quota["text"] == int(quota_config.text_multiplier * 40)
    assert quota["entity"] == int(quota_config.entity_multiplier * 40)
    assert quota["relationship"] == int(quota_config.relationship_multiplier * 40)
    assert quota["community"] == int(quota_config.community_multiplier * 40)
    assert quota["claim"] == int(quota_config.claim_multiplier * 40)

    quota_config.community_multiplier = 2.0
    assert strategy._per_type_quota(top_k=40)["community"] == 80


def test_per_type_quota_floors_hold_at_a_tiny_top_k(config: Config) -> None:
    # At top_k=1 the multipliers alone would reserve 0-2 slots, which starves the
    # KG sections the multi-hop bridge items live in; the floors are what keep each
    # section represented.
    strategy, _, _ = _make_strategy(config)
    quota_config = config.search.local_search.type_quota

    quota = strategy._per_type_quota(top_k=1)
    assert quota["text"] == quota_config.text_floor
    assert quota["entity"] == quota_config.entity_floor
    assert quota["relationship"] == quota_config.relationship_floor
    assert quota["community"] == quota_config.community_floor
    assert quota["claim"] == quota_config.claim_floor


class LineageRetriever(FakeRetriever):
    """FakeRetriever whose hits cite one text unit, so the chunk fetch runs."""

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        results = await super().aretrieve(query)
        for result in results:
            result.metadata["text_unit_ids"] = ["chunk-1"]
        return results


def _make_lineage_strategy(config: Config):
    strategy, _, _ = _make_strategy(config)
    os_r, neptune_r = LineageRetriever("document"), LineageRetriever("graph")
    strategy.retrievers = {
        RetrieverRole.DOCUMENT.value: os_r,
        RetrieverRole.GRAPH.value: neptune_r,
    }
    return strategy, os_r, neptune_r


async def test_caller_filters_reach_every_sub_query(config: Config) -> None:
    # Each section builds a fresh SearchQuery; before the fix none carried
    # query.filters, so `--filters` was silently dropped on the local path.
    config.processing.claim_extraction.enabled = True
    config.indexing.opensearch.build_relationship_vector_index = True
    strategy, os_r, neptune_r = _make_lineage_strategy(config)
    filters = {"doc_type": "contract"}
    query = _query(filters=dict(filters))

    await strategy.asearch(query)

    os_cfg = config.indexing.opensearch
    assert set(_doc_index_prefixes(os_r)) == {
        os_cfg.entities_index_prefix,
        os_cfg.text_units_index_prefix,
        os_cfg.community_reports_index_prefix,
        os_cfg.relationships_index_prefix,
        os_cfg.claims_index_prefix,
    }
    for call in [*os_r.calls, *neptune_r.calls]:
        assert call.filters is not None
        assert call.filters["doc_type"] == "contract"
    # The chunk fetch-by-id keeps its id scope alongside the caller filters.
    chunk_calls = [
        q for q in os_r.calls if q.index_prefixes == [os_cfg.text_units_index_prefix]
    ]
    assert chunk_calls and chunk_calls[0].filters == {
        "doc_type": "contract",
        "id": ["chunk-1"],
    }
    assert query.filters == filters  # caller's dict not mutated


async def test_unfiltered_query_sends_no_filters(config: Config) -> None:
    strategy, os_r, _ = _make_lineage_strategy(config)

    await strategy.asearch(_query())

    os_cfg = config.indexing.opensearch
    for call in os_r.calls:
        if call.index_prefixes == [os_cfg.text_units_index_prefix]:
            assert call.filters == {"id": ["chunk-1"]}
        else:
            assert call.filters is None


async def test_no_entity_focus_falls_back_to_raw_query(config: Config) -> None:
    # With no extracted entities the candidate lookup used to return [] and local
    # search produced no entities and no chunks; it now maps the raw query onto
    # the entities index like the other sections already do.
    strategy, os_r, neptune_r = _make_lineage_strategy(config)
    query = SearchQuery(query="who founded acme", entity_focus=[], top_k=7)

    result = await strategy.asearch(query)

    os_cfg = config.indexing.opensearch
    entity_calls = [
        q for q in os_r.calls if q.index_prefixes == [os_cfg.entities_index_prefix]
    ]
    assert len(entity_calls) == 1
    assert entity_calls[0].query == "who founded acme"
    assert entity_calls[0].top_k == 7
    assert len(neptune_r.calls) == 1  # graph expansion is seeded
    assert result.metadata["text_unit_count"] == 1


class _SeedOnlyRetriever(FakeRetriever):
    """Entity hits cite a chunk; graph expansion returns nothing."""

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        prefix = query.index_prefixes[0] if query.index_prefixes else self.tag
        if self.tag == "graph" or "entities" not in prefix:
            return []
        return [
            RetrievalResult(
                content="Entity: Acme",
                score=3.0,
                source="acme-id",
                retriever_type="document",
                metadata={"id": "acme-id", "text_unit_ids": ["chunk-acme"]},
            )
        ]


async def test_matched_entity_chunks_are_fetched_without_graph_hits(
    config: Config,
) -> None:
    # The matched entities' own chunks must reach the text-unit fetch even when
    # the Neptune expansion returns no nodes for them.
    strategy, _, _ = _make_strategy(config)
    os_r, neptune_r = _SeedOnlyRetriever("document"), _SeedOnlyRetriever("graph")
    strategy.retrievers = {
        RetrieverRole.DOCUMENT.value: os_r,
        RetrieverRole.GRAPH.value: neptune_r,
    }
    await strategy.asearch(_query())
    text_unit_prefix = config.indexing.opensearch.text_units_index_prefix
    fetches = [q for q in os_r.calls if q.index_prefixes == [text_unit_prefix]]
    assert fetches, "expected a text-unit fetch"
    assert fetches[0].filters is not None
    assert fetches[0].filters["id"] == ["chunk-acme"]
