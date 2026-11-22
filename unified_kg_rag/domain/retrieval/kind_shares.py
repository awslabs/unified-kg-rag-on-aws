# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ranked list of mixed kinds cut to a bound, each kind keeping its share.

A strategy's result list mixes kinds whose scores are on different scales:
a reranked text chunk carries the reranker's relevance (0-1 for a
cross-encoder or Cohere, a logit for some), an entity or relationship its
fusion score (about 1/61 under RRF). One cut by score across that list
keeps whichever scale is larger. The strategies already size each kind
with a per-type quota (``search.local_search.type_quota``, LightRAG's
stream floors, ppr's passages plus facts) and diversify each kind within
its own quota; :func:`cut_by_kind` scales the composition those quotas
produced down to the bound, so the shorter list has the same mix, and
compares scores only within a kind, where they share a scale.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

__all__ = ["apportion", "cut_by_kind"]

T = TypeVar("T")


def apportion(counts: Mapping[str, int], seats: int) -> dict[str, int]:
    """``seats`` split across ``counts``' kinds in proportion to their
    counts, by the Sainte-Lague (Webster) divisor method: each seat in turn
    goes to the kind with the largest quotient ``count / (2 * given + 1)``
    among those with a record left, compared in exact integers; a tie goes
    to the kind first in ``counts`` (the first ranked, for
    :func:`cut_by_kind`). Seats are handed out one by one, so a larger
    ``seats`` never takes one from a kind (house monotone, which largest
    remainder is not: ``{1, 3, 3}`` gave the first kind 1 of 3 seats and 0
    of 4). Of the divisor methods it is the one biased toward neither large
    nor small kinds; D'Hondt favours the largest, here the reranked chunks
    the cut exists to keep from crowding out the graph. The seats sum to
    ``min(seats, sum(counts))`` and no kind gets more than its count."""
    given = dict.fromkeys(counts, 0)
    for _ in range(min(max(seats, 0), sum(counts.values()))):
        best: str | None = None
        for kind, count in counts.items():
            if given[kind] >= count:
                continue
            # count / (2g + 1) > best's, cross-multiplied: no float ties.
            if best is None or count * (2 * given[best] + 1) > counts[best] * (
                2 * given[kind] + 1
            ):
                best = kind
        assert best is not None  # a kind has a record left below the sum
        given[best] += 1
    return given


def cut_by_kind(items: Sequence[T], bound: int, kind: Callable[[T], str]) -> list[T]:
    """``items`` (ranked) cut to at most ``bound``: each kind keeps its
    share of the bound (:func:`apportion`; a tied seat goes to the kind
    ranked first), its own best-ranked items, and the kept items stay in
    ``items``' order."""
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
