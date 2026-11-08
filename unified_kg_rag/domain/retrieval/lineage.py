# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Merging the lineages (text unit ids) of several retrieved items into one.

Every strategy that names "the units these graph items rest on" merges the
items' ``text_unit_ids`` with :func:`merge`, in one of two orders:

- ``first_seen``: each unit once, in the order the items name them (the
  lineage of a synthesized result: global search's key point, DRIFT's
  intermediate answer, written from the items in their order);
- ``support``: the units most items cite first, then those the best-scored
  item cites, then first seen (local search's text units, the structured
  strategy's query result and passages), so a cap keeps the units most of
  the evidence points to. MS GraphRAG local orders its candidate units by
  the rank of the first selected entity citing them, then by how many of
  its relationships they state; this ranks by graph support instead.

An item citing a unit twice supports it once.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

__all__ = ["LineageOrder", "merge"]

LineageOrder = Literal["first_seen", "support"]


def merge(
    lineages: Iterable[Sequence[str]],
    *,
    order: LineageOrder = "first_seen",
    scores: Sequence[float] | None = None,
) -> list[str]:
    """The units of ``lineages`` (one list of unit ids per item) each once,
    in ``order`` (module docstring); ``scores`` gives each item's score, in
    ``lineages`` order (``support`` breaks count ties by the best one)."""
    support: dict[str, int] = {}
    best: dict[str, float] = {}
    for index, lineage in enumerate(lineages):
        score = (
            float(scores[index]) if scores is not None and index < len(scores) else 0.0
        )
        for unit in dict.fromkeys(lineage):
            support[unit] = support.get(unit, 0) + 1
            if score > best.get(unit, float("-inf")):
                best[unit] = score
    if order == "first_seen":
        return list(support)
    seen = {unit: index for index, unit in enumerate(support)}
    return sorted(support, key=lambda unit: (-support[unit], -best[unit], seen[unit]))
