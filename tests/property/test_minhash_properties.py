# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""FuzzyMatcher's shared-permutation MinHash matches a per-name build.

The matcher used to build a fresh ``MinHash(num_perm)`` per name and per
query, regenerating the same seeded permutations every time. It now shares
them; the signatures, and so every match and score, must be unchanged.
"""

from __future__ import annotations

import numpy as np
import pytest
from datasketch import MinHash
from hypothesis import given, settings
from hypothesis import strategies as st

from unified_kg_rag.domain.ingestion.base_resolver import FuzzyMatcher
from unified_kg_rag.domain.models.config import ResolutionMethod
from unified_kg_rag.shared.utils import entity_key

pytestmark = pytest.mark.property


def _reference_minhash(text: str, num_perm: int, n_grams: int = 3) -> MinHash:
    minhash = MinHash(num_perm=num_perm)
    normalized = entity_key(text)
    if len(normalized) < n_grams:
        shingles = {normalized}
    else:
        shingles = {
            normalized[i : i + n_grams] for i in range(len(normalized) - n_grams + 1)
        }
    for shingle in shingles:
        minhash.update(shingle.encode("utf8"))
    return minhash


_NAMES = st.text(
    alphabet=st.sampled_from(list("abcdeAB 12-가나漢.&")), min_size=0, max_size=24
)


@settings(max_examples=200, deadline=None)
@given(_NAMES, st.sampled_from([16, 64, 128]))
def test_signature_matches_a_fresh_minhash(name: str, num_perm: int) -> None:
    shared = FuzzyMatcher._create_minhash(name, num_perm)
    assert np.array_equal(
        shared.hashvalues, _reference_minhash(name, num_perm).hashvalues
    )


@settings(max_examples=50, deadline=None)
@given(st.lists(_NAMES, min_size=1, max_size=20), _NAMES)
def test_matches_and_scores_are_unchanged(candidates: list[str], query: str) -> None:
    matcher = FuzzyMatcher(
        candidates,
        resolution_method=ResolutionMethod.MINHASH,
        similarity_threshold=0.5,
        minhash_permutations=64,
    )
    reference = _reference_minhash(query, 64)
    expected = sorted(
        (str(c), reference.jaccard(_reference_minhash(str(c), 64)))
        for c in matcher.lsh.query(reference)
    )
    assert sorted(matcher._find_all_lsh_matches(query)) == [
        m for m in expected if m[1] >= 0.5
    ]
