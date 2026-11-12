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

import unified_kg_rag.adapters.search_strategies  # noqa: F401
from tests.fixtures.fakes.retrievers import FakeRetriever
from unified_kg_rag.adapters.retrieval.strategy_registry import get_strategy_spec
from unified_kg_rag.domain.models import (
    Config,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)

pytestmark = pytest.mark.unit


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
        lambda results_dict, top_k, retrieval_multiplier=1, query=None, **_kw: [
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
    assert any(r.source == f"{claims_prefix}-1" for r in result.results)


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


async def test_a_namespace_without_a_claims_index_is_not_searched(
    config: Config,
) -> None:
    # Regression (rv8 retrieval low): a namespace whose claim extraction
    # found no claim has no claims index, and every local query searched it
    # and WARNed "Index not found".
    config.processing.claim_extraction.enabled = True
    strategy, os_r, _ = _make_strategy(config)
    asked: list[tuple[str, str | None]] = []
    held = {"exists": False}

    async def has_index(prefix: str, suffix: str | None) -> bool | None:
        asked.append((prefix, suffix))
        return held["exists"]

    os_r.ahas_index = has_index  # type: ignore[method-assign]
    claims_prefix = config.indexing.opensearch.claims_index_prefix

    await strategy.asearch(_query())
    assert claims_prefix not in _doc_index_prefixes(os_r)
    # Once it exists it is searched, and asked for no more.
    held["exists"] = True
    await strategy.asearch(_query())
    await strategy.asearch(_query())
    claim_calls = [q for q in os_r.calls if q.index_prefixes == [claims_prefix]]
    assert len(claim_calls) == 2
    assert [p for p, _ in asked].count(claims_prefix) == 2
