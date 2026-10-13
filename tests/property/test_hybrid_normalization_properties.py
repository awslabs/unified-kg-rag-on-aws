# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariants of ``combine_min_max`` (OpenSearch hybrid min-max fusion).

Raw scores are drawn as integers in [0, 100]: any normalized score other than
the floored minimum is then at least 0.01, above ``MIN_SCORE``, so the
plugin's flooring quirk (see the module docstring) cannot reorder hits.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from unified_kg_rag.domain.retrieval.hybrid_normalization import combine_min_max

pytestmark = pytest.mark.property

_ids = st.sampled_from([f"d{i}" for i in range(12)])
_hits = st.lists(
    st.tuples(_ids, st.integers(min_value=0, max_value=100).map(float)),
    max_size=10,
    unique_by=lambda hit: hit[0],
)
_weight = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


@given(lexical=_hits, vector=_hits, w=_weight)
def test_scores_are_in_unit_range_and_sorted(lexical, vector, w) -> None:
    fused = combine_min_max([lexical, vector], [w, 1.0 - w])
    scores = [score for _, score in fused]
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores == sorted(scores, reverse=True)
    # Every hit of either sub-query appears exactly once.
    ids = [doc_id for doc_id, _ in fused]
    assert len(ids) == len(set(ids))
    assert set(ids) == {d for d, _ in lexical} | {d for d, _ in vector}


@given(lexical=_hits, vector=_hits, w=_weight, data=st.data())
def test_raising_a_hit_score_never_lowers_its_combined_score(
    lexical, vector, w, data
) -> None:
    if not lexical:
        return
    index = data.draw(st.integers(min_value=0, max_value=len(lexical) - 1))
    raise_by = data.draw(st.integers(min_value=1, max_value=50))
    doc_id, score = lexical[index]
    raised = list(lexical)
    raised[index] = (doc_id, score + raise_by)
    weights = [w, 1.0 - w]
    before = dict(combine_min_max([lexical, vector], weights))[doc_id]
    after = dict(combine_min_max([raised, vector], weights))[doc_id]
    assert after >= before - 1e-12


@given(lexical=_hits, vector=_hits)
def test_lexical_weight_one_reproduces_lexical_order(lexical, vector) -> None:
    ranked = sorted(lexical, key=lambda hit: hit[1], reverse=True)
    fused = combine_min_max([ranked, vector], [1.0, 0.0])
    lexical_ids = {d for d, _ in ranked}
    # Lexical hits come first, in lexical order; vector-only hits score 0.
    assert [d for d, _ in fused[: len(ranked)]] == [d for d, _ in ranked]
    assert all(score == 0.0 for d, score in fused if d not in lexical_ids)
