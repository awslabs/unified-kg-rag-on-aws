# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for local search's claims (covariate) retrieval (AWS-free).

MS GraphRAG injects covariates into local-search context. These tests assert
the gated behavior: when claim extraction is ON, local search queries the
claims index and merges the hits into its context; when OFF (the default), no
claims query is issued at all (zero behavior change). Fake retrievers stand in
for OpenSearch / Neptune and the shared HybridScorer is stubbed to flatten
results, so Bedrock is never touched.
"""

from __future__ import annotations

import pytest

import aws_graphrag.adapters.search_strategies  # noqa: F401
from aws_graphrag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from aws_graphrag.domain.retrieval.strategy_registry import get_strategy_spec

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
    # Stub the shared scorer to flatten the source dict (avoid Bedrock).
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None: [
            r for results in results_dict.values() for r in results
        ]
    )
    return strategy, os_r, neptune_r


def _query(**kw) -> SearchQuery:
    return SearchQuery(query="what is the claim", entity_focus=["Alice"], **kw)


def _doc_index_prefixes(os_r: FakeRetriever) -> list[str]:
    return [q.index_prefixes[0] for q in os_r.calls if q.index_prefixes]


async def test_claims_queried_when_extraction_enabled(config: Config) -> None:
    config.processing.claim_extraction.enabled = True
    strategy, os_r, _ = _make_strategy(config)

    result = await strategy.asearch(_query())

    claims_prefix = config.indexing.opensearch.claims_index_prefix
    assert claims_prefix in _doc_index_prefixes(os_r)
    # The claims query uses the entity focus, mirroring entity retrieval.
    claim_calls = [q for q in os_r.calls if q.index_prefixes == [claims_prefix]]
    assert claim_calls and claim_calls[0].query == "Alice"
    # Claim hits flow into the assembled context.
    assert any(r.retriever_type == "document" for r in result.results)


async def test_no_claims_query_when_extraction_disabled(config: Config) -> None:
    # Default: claim extraction OFF -> the claims index is never queried.
    assert config.processing.claim_extraction.enabled is False
    strategy, os_r, _ = _make_strategy(config)

    await strategy.asearch(_query())

    claims_prefix = config.indexing.opensearch.claims_index_prefix
    assert claims_prefix not in _doc_index_prefixes(os_r)


async def test_claims_fall_back_to_query_text_without_entity_focus(
    config: Config,
) -> None:
    config.processing.claim_extraction.enabled = True
    strategy, os_r, _ = _make_strategy(config)

    await strategy.asearch(SearchQuery(query="raw claim text", entity_focus=[]))

    claims_prefix = config.indexing.opensearch.claims_index_prefix
    claim_calls = [q for q in os_r.calls if q.index_prefixes == [claims_prefix]]
    assert claim_calls and claim_calls[0].query == "raw claim text"
