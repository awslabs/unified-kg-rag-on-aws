# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The compiled-regex token estimate matches the per-character reference.

``estimate_token_count`` is the only counter for models without CountTokens
(the default Claude 5.5 models), so it runs over every context section of
every query; it was rewritten from a per-character Python loop to one regex
scan. The reference below is the original loop, kept to pin the results.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from unified_kg_rag.adapters.aws.token_counter import (
    _DENSE_SCRIPT_RANGES,
    estimate_token_count,
)

pytestmark = pytest.mark.property


def _reference_estimate(text: str) -> int:
    if not text:
        return 0
    dense_chars = sum(
        1 for ch in text if any(lo <= ord(ch) <= hi for lo, hi in _DENSE_SCRIPT_RANGES)
    )
    char_estimate = dense_chars + (len(text) - dense_chars) // 4
    return max(len(text.split()), char_estimate, 1)


# Every range edge and its neighbours, so off-by-one bounds are exercised.
_EDGES = sorted(
    {
        chr(cp)
        for lo, hi in _DENSE_SCRIPT_RANGES
        for cp in (lo - 1, lo, lo + 1, hi - 1, hi, hi + 1)
    }
)
_MULTILINGUAL = st.text(
    alphabet=st.one_of(
        st.characters(),  # any codepoint outside the surrogate block
        st.sampled_from(_EDGES),
        st.sampled_from(list("가나다 漢字 ひらがな カタカナ ＡＢ 。、 abc \t\n")),
    ),
    max_size=400,
)


@settings(max_examples=500, deadline=None)
@given(_MULTILINGUAL)
def test_estimate_matches_reference(text: str) -> None:
    assert estimate_token_count(text) == _reference_estimate(text)


def test_estimate_counts_each_range_edge() -> None:
    for lo, hi in _DENSE_SCRIPT_RANGES:
        for cp in (lo - 1, lo, hi, hi + 1):
            text = chr(cp) * 8
            assert estimate_token_count(text) == _reference_estimate(text), hex(cp)
