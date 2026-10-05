# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy-layer retrieval error visibility (AWS-free).

Regression: the retrievers re-raise clearly-fatal errors (auth/credentials/
endpoint/connection) instead of masking them as "no results" (see
``test_retriever_error_visibility``). But each search strategy wrapped its
retriever calls in a broad ``except Exception: return {}/[]``, which
*re-swallowed* those fatal errors and defeated the retriever-layer guard. The
strategies now re-raise ``is_fatal_retrieval_error`` errors and degrade to an
empty result only on genuinely-transient failures.
"""

from __future__ import annotations

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.domain.retrieval.strategy_registry import get_strategy_spec
from unified_kg_rag.shared import AWSServiceError

pytestmark = pytest.mark.unit

_FATAL = AWSServiceError("Cannot get AWS credentials for OpenSearch IAM.")
_TRANSIENT = AWSServiceError("Read timed out")


class _RaisingRetriever:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def aretrieve(self, query: SearchQuery) -> list:
        raise self.exc


def _simple_strategy(config: Config, exc: Exception):
    spec = get_strategy_spec(SearchStrategy.SIMPLE)
    strategy = spec.strategy_class(
        config=config,
        retrievers={RetrieverRole.DOCUMENT.value: _RaisingRetriever(exc)},
    )
    # Stub the Bedrock-backed fuser so a transient degrade path returns cleanly.
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None: [
            r for results in results_dict.values() for r in results
        ]
    )
    return strategy


async def test_simple_strategy_reraises_fatal(config: Config) -> None:
    strategy = _simple_strategy(config, _FATAL)
    with pytest.raises(AWSServiceError, match="credentials"):
        await strategy.asearch(SearchQuery(query="q"))


async def test_simple_strategy_degrades_on_transient(config: Config) -> None:
    strategy = _simple_strategy(config, _TRANSIENT)
    # Transient error is swallowed into an empty result set, not raised.
    result = await strategy.asearch(SearchQuery(query="q"))
    assert result.results == []


# --- LightRAG / local: graph expansion + linked-chunk paths -----------------
#
# LightRAG's `_expand_via_graph` and `_retrieve_linked_chunks` caught every
# exception without the fatal re-raise, so a Neptune AccessDenied or a broken
# OpenSearch endpoint on those paths was reported as "no results".

_NEPTUNE_FATAL = AWSServiceError(
    "AccessDeniedException: not authorized to perform neptune-db:ReadDataViaQuery"
)


class _OkRetriever:
    """Returns one entity hit citing one chunk, so expansion + chunk fetch run."""

    def __init__(self) -> None:
        self.calls: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        return [
            RetrievalResult(
                content="Vendor supplies Buyer",
                score=1.0,
                source="entity-1",
                retriever_type="document",
                metadata={"id": "entity-1", "text_unit_ids": ["chunk-1"]},
            )
        ]


class _FailOnIdFetch(_OkRetriever):
    """Answers keyword queries but raises on the fetch-by-chunk-id lookup."""

    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self.exc = exc

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        if query.filters and query.filters.get("id") == ["chunk-1"]:
            raise self.exc
        return await super().aretrieve(query)


def _strategy(config: Config, mode: SearchStrategy, document, graph):
    spec = get_strategy_spec(mode)
    strategy = spec.strategy_class(
        config=config,
        retrievers={
            RetrieverRole.DOCUMENT.value: document,
            RetrieverRole.GRAPH.value: graph,
        },
    )
    strategy.hybrid_scorer.fuse_and_rerank_results = (  # type: ignore[method-assign]
        lambda results_dict, top_k, retrieval_multiplier=1, query=None, **_kw: [
            r for results in results_dict.values() for r in results
        ]
    )
    return strategy


def _lightrag_query(mode: SearchStrategy) -> SearchQuery:
    return SearchQuery(
        query="q",
        ll_keywords=["Vendor"],
        hl_keywords=["supply"],
        metadata={"lightrag_mode": mode.value},
    )


@pytest.mark.parametrize("mode", [SearchStrategy.MIX, SearchStrategy.HYBRID])
async def test_lightrag_graph_expansion_reraises_fatal(
    config: Config, mode: SearchStrategy
) -> None:
    strategy = _strategy(
        config, mode, _OkRetriever(), _RaisingRetriever(_NEPTUNE_FATAL)
    )
    with pytest.raises(AWSServiceError, match="AccessDenied"):
        await strategy.asearch(_lightrag_query(mode))


async def test_lightrag_linked_chunk_fetch_reraises_fatal(config: Config) -> None:
    strategy = _strategy(
        config, SearchStrategy.MIX, _FailOnIdFetch(_FATAL), _OkRetriever()
    )
    with pytest.raises(AWSServiceError, match="credentials"):
        await strategy.asearch(_lightrag_query(SearchStrategy.MIX))


async def test_lightrag_degrades_on_transient(config: Config) -> None:
    document = _FailOnIdFetch(_TRANSIENT)
    strategy = _strategy(
        config, SearchStrategy.MIX, document, _RaisingRetriever(_TRANSIENT)
    )
    result = await strategy.asearch(_lightrag_query(SearchStrategy.MIX))
    # Expansion and linked chunks degrade to nothing; keyword hits survive.
    assert result.results
    assert result.metadata["sources"].get("graph_entities") is None
    assert "lightrag_linked_chunks" not in result.metadata["sources"]


async def test_local_graph_expansion_reraises_fatal(config: Config) -> None:
    strategy = _strategy(
        config,
        SearchStrategy.LOCAL,
        _OkRetriever(),
        _RaisingRetriever(_NEPTUNE_FATAL),
    )
    with pytest.raises(AWSServiceError, match="AccessDenied"):
        await strategy.asearch(SearchQuery(query="q", entity_focus=["Vendor"]))


async def test_local_graph_expansion_degrades_on_transient(config: Config) -> None:
    strategy = _strategy(
        config, SearchStrategy.LOCAL, _OkRetriever(), _RaisingRetriever(_TRANSIENT)
    )
    result = await strategy.asearch(SearchQuery(query="q", entity_focus=["Vendor"]))
    assert result.metadata["expanded_entity_count"] == 0
