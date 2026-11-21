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

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

__all__ = ["apportion", "cut_by_kind"]

T = TypeVar("T")


def apportion(counts: Mapping[str, int], seats: int) -> dict[str, int]:
    """``seats`` split across ``counts``' kinds in proportion to their
    counts, by largest remainder (Hamilton's method): each kind first gets
    the whole part of its exact share, then the seats left go one each to
    the largest remainders (ties: the larger count, then the kind first in
    ``counts``). The seats sum to ``min(seats, sum(counts))`` and no kind
    gets more than its count."""
    total = sum(counts.values())
    if seats >= total:
        return dict(counts)
    if seats <= 0 or total <= 0:
        return dict.fromkeys(counts, 0)
    exact = {kind: seats * count / total for kind, count in counts.items()}
    given = {kind: math.floor(share) for kind, share in exact.items()}
    left = seats - sum(given.values())
    position = {kind: i for i, kind in enumerate(counts)}
    order = sorted(
        counts,
        key=lambda kind: (-(exact[kind] - given[kind]), -counts[kind], position[kind]),
    )
    for kind in order[:left]:
        given[kind] += 1
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
