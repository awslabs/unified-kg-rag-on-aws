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


# --- several buckets ----------------------------------------------------------

_IDS = [f"item{i}" for i in range(8)]
# A bucket: a ranked list of ids (repeats allowed: a bucket a strategy
# assembles from several calls can repeat one).
_bucket = st.lists(st.sampled_from(_IDS), min_size=1, max_size=8)
_buckets = st.dictionaries(
    st.sampled_from(["text", "entities", "relationships", "graph", "reports"]),
    _bucket,
    min_size=1,
    max_size=5,
)
_weights = st.dictionaries(
    st.sampled_from(["text", "entities", "relationships", "graph", "reports"]),
    st.floats(min_value=0.1, max_value=3.0),
    max_size=5,
)


def _fuse(
    buckets: dict[str, list[str]], weights: dict[str, float] | None = None
) -> dict[str, float]:
    scorer = _scorer()
    if weights is not None:
        scorer.fusion_config = scorer.fusion_config.model_copy(
            update={"fusion_weights": weights}
        )
    fused = scorer._reciprocal_rank_fusion(
        {name: [_result(i, score=1.0) for i in ids] for name, ids in buckets.items()}
    )
    return {r.source: r.score for r in fused}


@given(_buckets, _weights)
def test_rrf_sums_one_weighted_term_per_bucket(buckets, weights) -> None:
    """An item's score is the sum, over the buckets it appears in, of the
    bucket's weight / (k + its first rank there)."""
    k = Config().search.fusion.rrf_k
    expected: dict[str, float] = {}
    for name, ids in buckets.items():
        for rank, item in enumerate(dict.fromkeys(ids), 1):
            expected[item] = expected.get(item, 0.0) + weights.get(name, 1.0) / (
                k + rank
            )
    fused = _fuse(buckets, weights)
    assert fused.keys() == expected.keys()
    for item, score in expected.items():
        assert fused[item] == pytest.approx(score)


@given(_buckets, st.randoms(use_true_random=False))
def test_rrf_does_not_depend_on_the_bucket_order(buckets, random) -> None:
    names = list(buckets)
    random.shuffle(names)
    shuffled = {name: buckets[name] for name in names}
    assert _fuse(shuffled) == pytest.approx(_fuse(buckets))


@given(_buckets, st.sampled_from(_IDS))
def test_rrf_never_lowers_an_item_moved_up_a_bucket(buckets, item) -> None:
    """Moving an item to the top of one bucket (the other buckets as they
    were) never lowers its fused score, and adding it to a bucket that lacked
    it raises it: agreement between streams is rewarded."""
    before = _fuse(buckets).get(item, 0.0)
    for name, ids in buckets.items():
        moved = {**buckets, name: [item, *(i for i in ids if i != item)]}
        after = _fuse(moved)[item]
        if item in ids:
            assert after >= before - 1e-12
        else:
            assert after > before


# --- MMR diversity cut ----------------------------------------------------------

_words = st.lists(
    st.sampled_from(["vendor", "depot", "widget", "buyer", "공급", "倉庫", "部品"]),
    min_size=1,
    max_size=5,
)


def _candidates(contents: list[list[str]]) -> list[RetrievalResult]:
    # Distinct, evenly spaced scores: rank i scores 1 - i / n.
    n = len(contents)
    return [
        RetrievalResult(
            content=" ".join(words),
            score=1.0 - i / n,
            source=f"item{i}",
            retriever_type="t",
        )
        for i, words in enumerate(contents)
    ]


@given(st.lists(_words, min_size=2, max_size=12), st.integers(1, 6))
def test_mmr_at_lambda_one_is_the_identity(contents, top_k) -> None:
    """``diversity_lambda: 1`` is pure relevance: the candidates come back
    as they were, in their order (the caller cuts them)."""
    scorer = _scorer()
    scorer.fusion_config = scorer.fusion_config.model_copy(
        update={"diversity_lambda": 1.0}
    )
    results = _candidates(contents)[::-1]
    kept = scorer._apply_diversity_filtering(results, top_k)
    assert [r.source for r in kept] == [r.source for r in results]


@given(st.lists(_words, min_size=2, max_size=12), st.integers(1, 6))
def test_mmr_with_a_negligible_penalty_keeps_the_most_relevant(contents, top_k) -> None:
    """A penalty weight below every relevance gap cannot reorder: the cut is
    the ``top_k`` best by score, whatever the redundancy (CJK terms too)."""
    n = len(contents)
    scorer = _scorer()
    # Normalized relevance gaps are 1 / (n - 1); the penalty is at most
    # (1 - lambda) * 1.
    lam = 1.0 - 0.5 / n
    scorer.fusion_config = scorer.fusion_config.model_copy(
        update={"diversity_lambda": lam}
    )
    results = _candidates(contents)
    kept = scorer._apply_diversity_filtering(results[::-1], top_k)
    assert [r.source for r in kept] == [r.source for r in results[: min(top_k, n)]]
