# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid-query score fusion as the OpenSearch neural-search plugin does it.

``OpenSearchIndexer.initialize`` writes a search pipeline with a
``normalization-processor`` (``min_max`` normalization, ``arithmetic_mean``
combination, ``search.hybrid.lexical_weight`` / ``vector_weight``). An engine
that answers ``hybrid`` queries itself uses ``combine_min_max`` so its scores
match that pipeline's:

* each sub-query's scores are min-max normalized over that sub-query's own
  hits: ``(score - min) / (max - min)``; a normalized 0 becomes
  ``MIN_SCORE`` (0.001), and when ``max == min`` every hit scores 1.0;
* the combined score is the weighted arithmetic mean over *every* sub-query: a
  document a sub-query did not return scores 0 there, and that sub-query's
  weight still counts in the denominator.

Because of ``MIN_SCORE``, a hit just above its sub-query's minimum can
normalize below the minimum hit itself (0.0001 < 0.001); this mirrors the
plugin rather than correcting it.
"""

MIN_SCORE = 0.001
SINGLE_RESULT_SCORE = 1.0


def _normalize(hits: list[tuple[str, float]]) -> dict[str, float]:
    """Min-max normalized score per id; the first hit of a repeated id wins."""
    scores: dict[str, float] = {}
    for doc_id, score in hits:
        scores.setdefault(doc_id, score)
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return dict.fromkeys(scores, SINGLE_RESULT_SCORE)
    span = high - low
    return {
        doc_id: (score - low) / span if score != low else MIN_SCORE
        for doc_id, score in scores.items()
    }


def combine_min_max(
    sub_results: list[list[tuple[str, float]]], weights: list[float]
) -> list[tuple[str, float]]:
    """Fuse per-sub-query ``(id, score)`` hits into one ranking, best first.

    Args:
        sub_results: The hits of each sub-query of the hybrid query, in the
            order the query lists them (lexical first, then vector).
        weights: One non-negative weight per sub-query.

    Returns:
        ``(id, combined score)`` pairs sorted by score, highest first. Ties
        keep the order the ids first appear in (sub-query by sub-query, hit by
        hit). Every combined score is in ``[0, 1]``; all weights 0 gives 0.

    Raises:
        ValueError: ``weights`` does not have one entry per sub-query, or one
            is negative.
    """
    if len(weights) != len(sub_results):
        raise ValueError(
            f"Expected {len(sub_results)} weights (one per sub-query), "
            f"got {len(weights)}"
        )
    if any(weight < 0 for weight in weights):
        raise ValueError(f"Hybrid weights must be non-negative: {weights}")
    normalized = [_normalize(hits) for hits in sub_results]
    total_weight = sum(weights)
    order: dict[str, None] = {}
    for scores in normalized:
        order.update(dict.fromkeys(scores))
    combined = [
        (
            doc_id,
            (
                sum(
                    weight * scores.get(doc_id, 0.0)
                    for weight, scores in zip(weights, normalized, strict=True)
                )
                / total_weight
                if total_weight > 0
                else 0.0
            ),
        )
        for doc_id in order
    ]
    # sorted() is stable: equal scores keep their first-appearance order.
    return sorted(combined, key=lambda pair: pair[1], reverse=True)
