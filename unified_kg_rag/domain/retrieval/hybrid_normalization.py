# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid score fusion as the OpenSearch neural-search plugin defines it.

The ``normalization-processor`` of a search pipeline with ``min_max``
normalization and ``arithmetic_mean`` combination. ``OpenSearchRetriever``
fuses the hits of a hybrid search's lexical and kNN searches with
``combine_min_max`` (weights ``search.hybrid.lexical_weight`` /
``vector_weight``), on every engine:

* each sub-query's scores are min-max normalized over that sub-query's own
  hits: ``(score - min) / (max - min)``, at least ``MIN_SCORE`` (0.001), and
  when ``max == min`` every hit scores 1.0;
* the combined score is the weighted arithmetic mean over *every* sub-query: a
  document a sub-query did not return scores 0 there, and that sub-query's
  weight still counts in the denominator.

The plugin raises only an exact 0 to ``MIN_SCORE``, so a hit just above its
sub-query's minimum normalized below the minimum hit itself (0.0001 < 0.001)
and lost its rank to it. The fusion runs client-side, so it floors every
normalized score instead: the order within a sub-query is kept.
"""

MIN_SCORE = 0.001
SINGLE_RESULT_SCORE = 1.0


def _normalize(hits: list[tuple[str, float]]) -> dict[str, tuple[float, float]]:
    """Min-max normalized score per id, floored at ``MIN_SCORE``, beside the
    normalized score before the floor; the first hit of a repeated id wins."""
    scores: dict[str, float] = {}
    for doc_id, score in hits:
        scores.setdefault(doc_id, score)
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if high == low:
        return dict.fromkeys(scores, (SINGLE_RESULT_SCORE, SINGLE_RESULT_SCORE))
    span = high - low
    normalized = {doc_id: (score - low) / span for doc_id, score in scores.items()}
    return {
        doc_id: (max(value, MIN_SCORE), value) for doc_id, value in normalized.items()
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
        ``(id, combined score)`` pairs sorted by score, highest first. Equal
        scores rank by their score before the ``MIN_SCORE`` floor (so the
        order within a sub-query is kept), then by id: the order the hits
        came in (a store's index order on a tie) never decides a cut through
        a tie, so an index rebuilt in another order, and every store, rank
        alike. Every combined score is in ``[0, 1]``; all weights 0 gives 0.

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

    def mean(doc_id: str, which: int) -> float:
        if total_weight <= 0:
            return 0.0
        return (
            sum(
                weight * scores[doc_id][which]
                for weight, scores in zip(weights, normalized, strict=True)
                if doc_id in scores
            )
            / total_weight
        )

    combined = {doc_id: (mean(doc_id, 0), mean(doc_id, 1)) for doc_id in order}
    ranked = sorted(
        combined,
        key=lambda doc_id: (-combined[doc_id][0], -combined[doc_id][1], doc_id),
    )
    return [(doc_id, combined[doc_id][0]) for doc_id in ranked]
