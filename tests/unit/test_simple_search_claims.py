# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for simple search's sweep (AWS-free).

Simple search searches every index the pipeline builds, one search (and one
fusion bucket) per index. These tests assert that the claims index is
included in the sweep only when claim extraction is enabled, and excluded
otherwise, that the summary tree joins it when RAPTOR is built, and that
caller-pinned index_prefixes are searched as given. The shared HybridScorer
is stubbed to avoid Bedrock.
"""

from __future__ import annotations

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401
from unified_kg_rag.adapters.retrieval.strategy_registry import get_strategy_spec
from unified_kg_rag.domain.models import (
    Config,
    Methodology,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.domain.retrieval.index_prefixes import all_index_prefixes

pytestmark = pytest.mark.unit


class FakeRetriever:
    def __init__(self, has_claims: bool | None = None) -> None:
        self.calls: list[SearchQuery] = []
        self.has_claims = has_claims
        self.asked: list[tuple[str, str | None]] = []

    async def ahas_index(self, prefix: str, suffix: str | None) -> bool | None:
        self.asked.append((prefix, suffix))
        return self.has_claims

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        return [
            RetrievalResult(
                content="result", score=1.0, source="s-1", retriever_type="document"
            )
        ]


def _swept(retriever: FakeRetriever) -> list[str]:
    """The index each search asked for, in order (one per search)."""
    swept = []
    for call in retriever.calls:
        assert isinstance(call.index_prefixes, list)
        assert len(call.index_prefixes) == 1
        swept.extend(call.index_prefixes)
    return swept


def _make_strategy(config: Config, has_claims: bool | None = None):
    spec = get_strategy_spec(SearchStrategy.SIMPLE)
    os_r = FakeRetriever(has_claims)
    strategy = spec.strategy_class(
        config=config,
        retrievers={RetrieverRole.DOCUMENT.value: os_r},
    )
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None, **_: [
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
    prefixes = _swept(os_r)
    assert o.claims_index_prefix not in prefixes
    assert o.text_units_index_prefix in prefixes


async def test_claims_included_in_sweep_when_enabled(config: Config) -> None:
    # Claim extraction ON -> every index the pipeline builds, claims too.
    config.processing.claim_extraction.enabled = True
    strategy, os_r = _make_strategy(config)

    await strategy.asearch(SearchQuery(query="q"))

    assert _swept(os_r) == all_index_prefixes(config)


async def test_claims_excluded_when_no_claim_was_extracted(config: Config) -> None:
    # Regression (rv9 query L9): a claim extraction that found no claim
    # built no claims index, and every simple query WARNed "Index not found".
    config.processing.claim_extraction.enabled = True
    strategy, os_r = _make_strategy(config, has_claims=False)

    await strategy.asearch(SearchQuery(query="q"))

    o = config.indexing.opensearch
    prefixes = _swept(os_r)
    assert o.claims_index_prefix not in prefixes
    assert o.text_units_index_prefix in prefixes
    # Once found, the index is not asked for again.
    os_r.has_claims = True
    await strategy.asearch(SearchQuery(query="q"))
    os_r.calls.clear()
    await strategy.asearch(SearchQuery(query="q"))
    assert _swept(os_r) == all_index_prefixes(config)
    assert len(os_r.asked) == 2


async def test_relationship_and_reports_excluded_when_not_built(
    config: Config,
) -> None:
    # Claims ON but no relationship vector index / community reports (a
    # graphrag-only config) -> the sweep is pinned to the indices the
    # pipeline actually builds.
    config.indexing.methodologies = [Methodology.GRAPHRAG]
    config.processing.claim_extraction.enabled = True
    config.indexing.opensearch.build_relationship_vector_index = False
    config.graph.community_detection.report_generation.enabled = False
    strategy, os_r = _make_strategy(config)

    await strategy.asearch(SearchQuery(query="q"))

    o = config.indexing.opensearch
    assert _swept(os_r) == [
        o.text_units_index_prefix,
        o.entities_index_prefix,
        o.claims_index_prefix,
    ]


async def test_caller_pinned_prefixes_passed_through(config: Config) -> None:
    # When the caller pins index_prefixes, the gate never rewrites them.
    strategy, os_r = _make_strategy(config)
    o = config.indexing.opensearch
    pinned = [o.entities_index_prefix, o.claims_index_prefix]

    await strategy.asearch(SearchQuery(query="q", index_prefixes=pinned))

    assert _swept(os_r) == pinned


@pytest.mark.parametrize("kinds", ["document", ["seed_graph"]])
async def test_reports_dropped_under_a_single_kind_filter(
    config: Config, kinds: str | list[str]
) -> None:
    # A community report carries no source kind, so a filter that keeps one
    # kind drops the reports index from the sweep (local and DRIFT read no
    # report then either).
    strategy, os_r = _make_strategy(config)
    o = config.indexing.opensearch

    await strategy.asearch(SearchQuery(query="q", filters={"source_kinds": kinds}))

    prefixes = _swept(os_r)
    assert o.community_reports_index_prefix not in prefixes
    assert o.text_units_index_prefix in prefixes
    assert o.entities_index_prefix in prefixes


async def test_reports_kept_when_the_filter_keeps_both_kinds(config: Config) -> None:
    strategy, os_r = _make_strategy(config)
    filters = {"source_kinds": ["document", "seed_graph"]}

    await strategy.asearch(SearchQuery(query="q", filters=filters))

    reports = config.indexing.opensearch.community_reports_index_prefix
    assert reports in _swept(os_r)


async def test_pinned_reports_dropped_under_a_single_kind_filter(
    config: Config,
) -> None:
    strategy, os_r = _make_strategy(config)
    o = config.indexing.opensearch
    filters = {"source_kinds": "document"}

    await strategy.asearch(
        SearchQuery(
            query="q",
            index_prefixes=[o.entities_index_prefix, o.community_reports_index_prefix],
            filters=filters,
        )
    )
    assert _swept(os_r) == [o.entities_index_prefix]

    # Pinned to the reports alone: nothing to search, not every index.
    out = await strategy.asearch(
        SearchQuery(
            query="q", index_prefixes=o.community_reports_index_prefix, filters=filters
        )
    )
    assert len(os_r.calls) == 1
    assert out.results == []


async def test_the_reports_skip_is_logged_only_when_the_sweep_held_them(
    config: Config, caplog: pytest.LogCaptureFixture
) -> None:
    strategy, os_r = _make_strategy(config)
    o = config.indexing.opensearch
    filters = {"source_kinds": "document"}

    with caplog.at_level("INFO"):
        await strategy.asearch(
            SearchQuery(
                query="q", index_prefixes=[o.entities_index_prefix], filters=filters
            )
        )
    assert os_r.calls[0].index_prefixes == [o.entities_index_prefix]
    assert "Community reports skipped" not in caplog.text

    with caplog.at_level("INFO"):
        await strategy.asearch(SearchQuery(query="q", filters=filters))
    assert "Community reports skipped" in caplog.text


class _CuttingRetriever(FakeRetriever):
    """Five hits per index asked for, interleaved and cut to
    ``top_k * retrieval_multiplier`` as the document retriever cuts."""

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        prefixes = query.index_prefixes
        assert isinstance(prefixes, list)
        hits = [
            RetrievalResult(
                content=f"{prefix} {rank}",
                score=1.0 / (rank + 1),
                source=f"{prefix}-{rank}",
                retriever_type="document",
            )
            for rank in range(5)
            for prefix in prefixes
        ]
        return hits[: query.top_k * query.retrieval_multiplier]


async def test_each_index_brings_its_own_candidate_pool(config: Config) -> None:
    # Regression (x5, 2026-11-08): one search over every index kept
    # top_k x retrieval_multiplier hits in all (2-3 per index) before fusion
    # and the reranker saw them, and simple lost to naive. Each index now
    # brings that many, one fusion bucket each; the union is cut to top_k.
    config.processing.claim_extraction.enabled = True
    strategy = get_strategy_spec(SearchStrategy.SIMPLE).strategy_class(
        config=config,
        retrievers={RetrieverRole.DOCUMENT.value: _CuttingRetriever()},
    )
    fused: dict[str, list[RetrievalResult]] = {}

    def fuse(results_dict, top_k, retrieval_multiplier=1, query=None, **_):
        fused.update(results_dict)
        return [r for results in results_dict.values() for r in results][:top_k]

    strategy.hybrid_scorer.fuse_and_rerank_results = fuse  # type: ignore[method-assign]

    out = await strategy.asearch(
        SearchQuery(query="q", top_k=2, retrieval_multiplier=2)
    )

    assert sorted(fused) == [
        "simple_claims",
        "simple_community_reports",
        "simple_entities",
        "simple_relationships",
        "simple_text_units",
    ]
    assert all(len(hits) == 4 for hits in fused.values())
    assert len(out.results) == 2
    assert out.metadata["sources"] == dict.fromkeys(fused, 4)


async def test_the_summary_tree_joins_the_sweep_where_it_is_built(
    config: Config,
) -> None:
    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.RAPTOR]
    strategy, os_r = _make_strategy(config)
    o = config.indexing.opensearch

    await strategy.asearch(SearchQuery(query="q"))

    assert _swept(os_r)[-1] == o.summary_nodes_index_prefix
    assert o.summary_nodes_index_prefix in strategy.filtered_index_prefixes(
        SearchStrategy.SIMPLE
    )
    # Not held by the queried namespace: not searched.
    os_r.calls.clear()
    await strategy.asearch(
        SearchQuery(query="q", metadata={"held_artifacts": ["text_units"]})
    )
    assert _swept(os_r) == [o.text_units_index_prefix]
