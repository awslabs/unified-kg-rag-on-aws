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
