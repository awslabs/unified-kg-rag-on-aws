# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The compiled-regex token estimate matches the per-character reference.

``estimate_token_count`` is the only counter for models without CountTokens
(the default Claude 5.5 models), so it runs over every context section of
every query; it was rewritten from a per-character Python loop to one regex
scan. The reference below is the original loop over the original table of
dense-script ranges, both kept to pin the results: the table now lives in
``shared.utils.scripts`` beside the letter blocks, and a change to what
the estimate counts is a measured change of its own.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.shared.utils.scripts import DENSE_SCRIPT_TOKEN_RANGES

pytestmark = pytest.mark.property

_DENSE_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x2E80, 0x2FDF),  # CJK radicals / Kangxi
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),  # Half/full-width forms
)


def test_the_estimate_counts_the_original_ranges() -> None:
    assert DENSE_SCRIPT_TOKEN_RANGES == _DENSE_SCRIPT_RANGES


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
