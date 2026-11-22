# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A mixed-kind result list cut to a bound, each kind keeping its share."""

from __future__ import annotations

from fractions import Fraction

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


def test_apportion_splits_the_seats_by_sainte_lague() -> None:
    assert apportion(
        {"text": 4, "entity": 4, "relationship": 4, "community": 1}, 5
    ) == {
        # Quotients 4, 4, 4, 1, then 4/3 each: the tied seats go in the
        # order given (the strategy's rank).
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


def test_apportion_breaks_a_tie_in_exact_arithmetic() -> None:
    # R3B-3: largest remainder in floats read 4*2/6 - 1 as 0.33333333333333326,
    # below text's and entity's 1/3, so text took relationship's seat.
    assert apportion({"text": 1, "entity": 1, "relationship": 4}, 2) == {
        "text": 0,
        "entity": 0,
        "relationship": 2,
    }
    # The second seat ties (9/3 = 3/1): it goes to the kind ranked first.
    assert apportion({"text": 9, "entity": 3}, 2) == {"text": 2, "entity": 0}
    assert apportion({"entity": 3, "text": 9}, 2) == {"entity": 1, "text": 1}
    assert apportion({"entity": 3, "text": 9}, 3) == {"entity": 1, "text": 2}


def test_a_larger_bound_never_takes_a_kinds_seat() -> None:
    # R3B-3: largest remainder (the Alabama paradox) gave text 1 of 3 seats
    # and 0 of 4, so kg_search at top_k 4 dropped the chunk top_k 3 returned.
    counts = {"text": 1, "entity": 3, "relationship": 3}
    assert apportion(counts, 3) == {"text": 1, "entity": 1, "relationship": 1}
    assert apportion(counts, 4) == {"text": 1, "entity": 2, "relationship": 1}


def _sainte_lague(counts: dict[str, int], seats: int) -> dict[str, int]:
    """The same method as a priority list: every quotient count / (2j + 1)
    of every kind, in exact fractions, largest first (a tie: the kind first
    in ``counts``, then its own earlier quotient), the first ``seats``."""
    order = {kind: i for i, kind in enumerate(counts)}
    quotients = sorted(
        (
            (-Fraction(count, 2 * j + 1), order[kind], j, kind)
            for kind, count in counts.items()
            for j in range(count)
        ),
    )
    given = dict.fromkeys(counts, 0)
    for *_, kind in quotients[: max(seats, 0)]:
        given[kind] += 1
    return given


_COUNTS = st.dictionaries(st.sampled_from("abcdef"), st.integers(0, 50), min_size=1)


@given(_COUNTS, st.integers(-2, 120))
def test_apportion_gives_each_kind_its_exact_seats(
    counts: dict[str, int], seats: int
) -> None:
    given_seats = apportion(counts, seats)
    assert given_seats == _sainte_lague(counts, seats)
    assert sum(given_seats.values()) == min(max(seats, 0), sum(counts.values()))
    for kind, count in counts.items():
        assert 0 <= given_seats[kind] <= count


@given(_COUNTS, st.integers(0, 120))
def test_apportion_is_house_monotone(counts: dict[str, int], seats: int) -> None:
    fewer, more = apportion(counts, seats), apportion(counts, seats + 1)
    assert all(more[kind] >= fewer[kind] for kind in counts)


@given(
    st.lists(st.sampled_from(["text", "entity", "relationship", "summary"])),
    st.integers(0, 40),
)
def test_a_larger_bound_keeps_every_record_a_smaller_one_kept(
    kinds: list[str], bound: int
) -> None:
    items = list(enumerate(kinds))
    smaller = cut_by_kind(items, bound, lambda item: item[1])
    larger = cut_by_kind(items, bound + 1, lambda item: item[1])
    assert set(smaller) <= set(larger)
    assert len(larger) == min(bound + 1, len(items))
    assert larger == sorted(larger)  # the ranked order


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
