# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Relationship weight = sum of the extracted strengths of its text units.

Microsoft GraphRAG sums the LLM ``relationship_strength`` of every extracted
instance of an edge. A plain sum cannot be maintained incrementally: a re-applied
delta would add its strengths again, and removing a document could not take
its share back out. So each edge keeps its strength per supporting text unit
(``attributes["text_unit_weights"]``, parallel to ``text_unit_ids``) and its
weight is the sum of that map.

- Full build: instances of one edge are combined with :func:`sum_weights`; two
  instances from the same text unit add up, as in MS GraphRAG.
- Incremental merge: :func:`overlay_weights` lets the delta's entry for a text
  unit replace the stored one. A text unit is extracted in one run only, so a
  repeated entry is a re-application or a re-extraction, never new evidence.
- Removal: dropping a text unit drops its entry.

Both paths therefore give the same weight for the same set of text units. An
edge without the stored map (written before it existed, or a fresh single
extraction) splits its current weight evenly over its text units, so its
weight stays as it is until those text units are re-extracted.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from unified_kg_rag.domain.models import Relationship

TEXT_UNIT_WEIGHTS_ATTRIBUTE = "text_unit_weights"


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def text_unit_weights(rel: Relationship) -> dict[str, float]:
    """The edge's strength per supporting text unit (see the module docstring)."""
    text_units = _dedupe(rel.text_unit_ids or [])
    if not text_units:
        return {}
    stored = (rel.attributes or {}).get(TEXT_UNIT_WEIGHTS_ATTRIBUTE)
    if (
        isinstance(stored, list)
        and len(stored) == len(text_units)
        and all(isinstance(v, int | float) and not isinstance(v, bool) for v in stored)
    ):
        return {tu: float(w) for tu, w in zip(text_units, stored, strict=True)}
    total = rel.weight if rel.weight is not None else 1.0
    share = total / len(text_units)
    return dict.fromkeys(text_units, share)


def sum_weights(weight_maps: Iterable[dict[str, float]]) -> dict[str, float]:
    """Combine instances of one edge: strengths for the same text unit add up."""
    combined: dict[str, float] = {}
    for weights in weight_maps:
        for text_unit, weight in weights.items():
            combined[text_unit] = combined.get(text_unit, 0.0) + weight
    return combined


def overlay_weights(
    stored: dict[str, float], delta: dict[str, float]
) -> dict[str, float]:
    """Merge a delta into a stored edge: the delta's entry for a text unit wins."""
    return {**stored, **delta}


def apply_text_unit_weights(rel: Relationship, weights: dict[str, float]) -> None:
    """Set ``rel``'s text units, stored strengths and weight from ``weights``.

    Without text units the edge keeps its own weight and lineage.
    """
    if not weights:
        return
    rel.text_unit_ids = list(weights)
    rel.attributes = {
        **(rel.attributes or {}),
        TEXT_UNIT_WEIGHTS_ATTRIBUTE: list(weights.values()),
    }
    # fsum is exactly rounded, so the weight does not depend on the order the
    # text units were merged in.
    rel.weight = math.fsum(weights.values())
