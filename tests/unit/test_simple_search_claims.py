# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for simple search's claims-index gating (AWS-free).

Simple search sweeps every index by default. These tests assert that the
claims index is included in the sweep only when claim extraction is enabled,
and excluded otherwise, while caller-pinned index_prefixes are passed through
untouched. The shared HybridScorer is stubbed to avoid Bedrock.
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
    def __init__(self) -> None:
        self.calls: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        return [
            RetrievalResult(
                content="result", score=1.0, source="s-1", retriever_type="document"
            )
        ]


def _make_strategy(config: Config):
    spec = get_strategy_spec(SearchStrategy.SIMPLE)
    os_r = FakeRetriever()
    strategy = spec.strategy_class(
        config=config,
        retrievers={RetrieverRole.DOCUMENT.value: os_r},
    )
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None: [
            r for results in results_dict.values() for r in results
        ]
    )
    return strategy, os_r


async def test_claims_excluded_from_sweep_when_disabled(config: Config) -> None:
    # Default: claim extraction OFF -> claims index dropped from the all-index
    # sweep; the search is pinned to the non-claims indexes.
    assert config.processing.claim_extraction.enabled is False
    strategy, os_r = _make_strategy(config)

    await strategy.asearch(SearchQuery(query="q"))

    o = config.indexing.opensearch
    assert len(os_r.calls) == 1
    prefixes = os_r.calls[0].index_prefixes
    assert prefixes is not None
    assert o.claims_index_prefix not in prefixes
    assert o.text_units_index_prefix in prefixes


async def test_claims_included_in_sweep_when_enabled(config: Config) -> None:
    # Claim extraction ON -> default all-index sweep (index_prefixes left None,
    # which the retriever normalizes to all mappings, incl. claims).
    config.processing.claim_extraction.enabled = True
    strategy, os_r = _make_strategy(config)

    await strategy.asearch(SearchQuery(query="q"))

    assert len(os_r.calls) == 1
    assert os_r.calls[0].index_prefixes is None


async def test_caller_pinned_prefixes_passed_through(config: Config) -> None:
    # When the caller pins index_prefixes, the gate never rewrites them.
    strategy, os_r = _make_strategy(config)
    pinned = [config.indexing.opensearch.entities_index_prefix]

    await strategy.asearch(SearchQuery(query="q", index_prefixes=pinned))

    assert os_r.calls[0].index_prefixes == pinned
