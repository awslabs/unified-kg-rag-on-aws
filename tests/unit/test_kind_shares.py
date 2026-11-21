# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A mixed-kind result list cut to a bound, each kind keeping its share."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    ProcessedQuery,
)
from unified_kg_rag.domain.models import (
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from unified_kg_rag.domain.retrieval.kind_shares import apportion, cut_by_kind

pytestmark = pytest.mark.unit


def test_apportion_splits_the_seats_by_largest_remainder() -> None:
    assert apportion(
        {"text": 4, "entity": 4, "relationship": 4, "community": 1}, 5
    ) == {
        # 20/13 = 1.54 each, 0.38: the two seats left go to the larger
        # remainders, tied, then in the order given (the strategy's rank).
        "text": 2,
        "entity": 2,
        "relationship": 1,
        "community": 0,
    }
    assert apportion({"text": 20, "entity": 40, "relationship": 40}, 10) == {
        "text": 2,
        "entity": 4,
        "relationship": 4,
    }
    assert apportion({"text": 3, "entity": 1}, 10) == {"text": 3, "entity": 1}
    assert apportion({"text": 3}, 0) == {"text": 0}


@given(
    st.dictionaries(st.sampled_from("abcdef"), st.integers(0, 50), min_size=1),
    st.integers(0, 120),
)
def test_apportion_fills_the_bound_without_exceeding_a_count(
    counts: dict[str, int], seats: int
) -> None:
    given_seats = apportion(counts, seats)
    assert sum(given_seats.values()) == min(seats, sum(counts.values()))
    total = sum(counts.values())
    for kind, count in counts.items():
        assert 0 <= given_seats[kind] <= count
        if seats < total:
            # Quota property: within one seat of the exact share.
            assert abs(given_seats[kind] - seats * count / total) < 1


def _result(kind: str, score: float, n: int) -> RetrievalResult:
    return RetrievalResult(
        content=f"{kind} {n}", score=score, source=f"{kind}-{n}", retriever_type=kind
    )


def _mixed() -> list[RetrievalResult]:
    # The order the hybrid scorer returns with only text reranked: every
    # chunk's rerank score (0-1) above every graph record's fusion score.
    texts = [_result("text", 0.9 - 0.1 * i, i) for i in range(6)]
    entities = [_result("entity", 0.0164 - 0.0001 * i, i) for i in range(6)]
    relations = [_result("relationship", 0.01635 - 0.0001 * i, i) for i in range(6)]
    ranked = texts + entities + relations
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked


def test_cut_by_kind_keeps_each_kinds_best_in_order() -> None:
    ranked = _mixed()
    kept = cut_by_kind(ranked, 6, lambda r: r.retriever_type)
    assert [r.source for r in kept] == [
        "text-0",
        "text-1",
        "entity-0",
        "relationship-0",
        "entity-1",
        "relationship-1",
    ]
    assert cut_by_kind(ranked, 50, lambda r: r.retriever_type) == ranked


def test_a_searchs_cut_keeps_the_graph_beside_reranked_chunks() -> None:
    # R2B-1: the cut took the first top_k of that list: chunks only.
    sr = SearchResult(
        query=SearchQuery(query="q"),
        results=_mixed(),
        total_results=18,
        search_strategy="local",
        processing_time=0.0,
    )
    state = {
        "search_results": sr,
        "resolved_strategy": SearchStrategy.LOCAL,
        "start_time": 0.0,
        "processed_query": ProcessedQuery(original_query="q", final_query="q"),
        "top_k": 3,
        "retrieval_multiplier": 2,
    }
    out = GraphRAGChain._format_search_output_step(state)
    kinds = [r.retriever_type for r in out.search_results.results]
    assert sorted(kinds) == ["entity", "entity", "relationship", "relationship"] + [
        "text",
        "text",
    ]
    assert out.search_results.total_results == 6
    assert out.metadata["total_results"] == 6
