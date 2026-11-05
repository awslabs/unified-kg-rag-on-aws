# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property-based tests for fusion invariants (HybridScorer RRF)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer
from unified_kg_rag.domain.models import Config, RetrievalResult

pytestmark = pytest.mark.property


def _scorer() -> HybridScorer:
    config = Config()
    config.providers.rerank.kind = "none"
    return HybridScorer(config)


def _result(content: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        content=content, score=score, source=content, retriever_type="t"
    )


@given(st.integers(min_value=2, max_value=30))
def test_rrf_is_rank_monotonic(n: int) -> None:
    """A single ranked list fused by RRF preserves the ranking: earlier items
    get strictly higher scores (1/(k+rank) is strictly decreasing in rank)."""
    scorer = _scorer()
    ranked = [_result(f"item{i}", score=1.0 - i / n) for i in range(n)]
    fused = scorer._reciprocal_rank_fusion({"src": ranked})
    by_content = {r.content: r.score for r in fused}
    scores_in_rank_order = [by_content[f"item{i}"] for i in range(n)]
    assert all(
        a > b
        for a, b in zip(scores_in_rank_order, scores_in_rank_order[1:], strict=False)
    )
