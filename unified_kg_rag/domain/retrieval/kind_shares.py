# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ranked list of mixed score scales cut to a bound, each scale keeping
its share.

A strategy's result list mixes records whose scores are on different
scales: a reranked record carries the reranker's relevance (0-1 for a
cross-encoder or Cohere, a logit for some), an unreranked entity or
relationship its fusion score (about 1/61 under RRF). One cut by score
across that list keeps whichever scale is larger. The records the reranker
scored together share one scale, whatever their kinds (``tree`` reranks
its summaries and passages in one pool, RAPTOR's collapsed tree); the
others are sized by the strategy's per-type quota (``search.local_search.
type_quota``, LightRAG's stream floors, ppr's passages plus facts), one
group per kind. :func:`cut_by_kind` scales the composition of those groups
(:func:`score_group`) down to the bound, so the shorter list has the same
mix, and compares scores only within a group, where they share a scale.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from unified_kg_rag.domain.models import RetrievalResult

__all__ = ["RERANKED", "apportion", "cut_by_kind", "score_group"]

T = TypeVar("T")

# The metadata flag the reranker sets on each record it scored, and the
# group of every record so flagged: one comparator across their kinds.
RERANKED = "reranked"


def score_group(result: RetrievalResult) -> str:
    """The group whose scores ``result``'s compares with: the reranker's
    (:data:`RERANKED` in its metadata, whatever its kind), else its kind's
    (``retriever_type``), the scale its fusion and its type's quota gave
    it."""
    if (result.metadata or {}).get(RERANKED):
        return RERANKED
    return str(result.retriever_type)


def _outranks(count: int, given: int, other_count: int, other_given: int) -> bool:
    """Whether ``count / sqrt(given (given + 1))`` exceeds the other kind's,
    squared and cross-multiplied in exact integers: a kind without a seat
    (``given`` 0, an infinite priority) outranks every kind with one, and
    two without a seat tie."""
    return count * count * other_given * (other_given + 1) > (
        other_count * other_count * given * (given + 1)
    )


def apportion(counts: Mapping[str, int], seats: int) -> dict[str, int]:
    """``seats`` split across ``counts``' kinds in proportion to their
    counts, by the Huntington-Hill divisor method: each seat in turn goes
    to the kind with the largest priority ``count / sqrt(given * (given +
    1))`` among those with a record left, compared squared and
    cross-multiplied in exact integers (no float square root); a tie goes
    to the kind first in ``counts`` (the first ranked, for
    :func:`cut_by_kind`). A kind's priority is infinite until it holds a
    seat, so whenever ``seats`` reaches the number of kinds with records,
    each of them gets one: the cut exists so the graph records survive
    beside the reranked chunks that outnumber them (Sainte-Lague rounded a
    relationship's 0.4 of a seat to none at ``top_k`` 2). Seats are handed
    out one by one, so a larger ``seats`` never takes one from a kind
    (house monotone, which largest remainder is not: ``{1, 3, 3}`` gave the
    first kind 1 of 3 seats and 0 of 4). The seats sum to ``min(seats,
    sum(counts))`` and no kind gets more than its count."""
    given = dict.fromkeys(counts, 0)
    for _ in range(min(max(seats, 0), sum(counts.values()))):
        best: str | None = None
        for kind, count in counts.items():
            if given[kind] >= count:
                continue
            if best is None or _outranks(count, given[kind], counts[best], given[best]):
                best = kind
        assert best is not None  # a kind has a record left below the sum
        given[best] += 1
    return given


def cut_by_kind(items: Sequence[T], bound: int, kind: Callable[[T], str]) -> list[T]:
    """``items`` (ranked) cut to at most ``bound``: each ``kind`` (a group
    of items whose scores compare, :func:`score_group`) keeps its share of
    the bound (:func:`apportion`; a tied seat goes to the kind ranked
    first), its own best-ranked items, and the kept items stay in
    ``items``' order. A larger ``bound`` keeps every item a smaller one
    kept."""
    if len(items) <= bound:
        return list(items)
    seats = apportion(Counter(kind(item) for item in items), bound)
    kept: list[T] = []
    for item in items:
        k = kind(item)
        if seats[k] > 0:
            seats[k] -= 1
            kept.append(item)
    return kept
