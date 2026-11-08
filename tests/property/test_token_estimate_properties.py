# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The compiled-regex token estimate matches a per-character reference.

``estimate_token_count`` is the only counter for models without CountTokens
(the default Claude 5.5 models), so it runs over every context section of
every query; it scans with one compiled class per cost instead of a
per-character Python loop. The reference below is that loop over the
script table (``shared.utils.scripts``): each character costs its script's
twelfths of a token, a CJK symbol or full-width form one token.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from unified_kg_rag.adapters.llm.token_estimate import (
    estimate_token_count,
    token_units,
)
from unified_kg_rag.shared.utils.scripts import (
    DENSE_SCRIPT_TOKEN_RANGES,
    SCRIPTS,
    script_of,
)

pytestmark = pytest.mark.property

# The ranges beside the dense letters that count one token a character.
_SYMBOL_RANGES: tuple[tuple[int, int], ...] = (
    (0x2E80, 0x2FDF),  # CJK radicals / Kangxi
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0xFF00, 0xFFEF),  # Half/full-width forms
)


def _reference_units(text: str) -> int:
    units = 0
    for ch in text:
        if any(lo <= ord(ch) <= hi for lo, hi in _SYMBOL_RANGES):
            units += 12
        elif (script := script_of(ch)) is not None:
            units += round(12 / script.chars_per_token)
        else:
            units += 3
    return units


def _reference_estimate(text: str) -> int:
    if not text:
        return 0
    return max(len(text.split()), _reference_units(text) // 12, 1)


def test_every_rate_is_a_whole_number_of_twelfths() -> None:
    for script in SCRIPTS:
        assert (12 / script.chars_per_token).is_integer(), script.name
        # The chunk-size bound (chunk_length.max_chars) holds for rates of
        # at most four characters a token.
        assert script.chars_per_token <= 4


def test_the_dense_token_ranges_are_the_dense_letters_and_the_symbols() -> None:
    for lo, hi in DENSE_SCRIPT_TOKEN_RANGES:
        for cp in (lo, hi):
            assert _reference_units(chr(cp)) == 12, hex(cp)


# Every block edge and its neighbours, so off-by-one bounds are exercised.
_EDGES = sorted(
    {
        chr(cp)
        for lo, hi in (
            *(span for s in SCRIPTS for span in s.ranges),
            *_SYMBOL_RANGES,
        )
        for cp in (lo - 1, lo, lo + 1, hi - 1, hi, hi + 1)
    }
)
_MULTILINGUAL = st.text(
    alphabet=st.one_of(
        st.characters(),  # any codepoint outside the surrogate block
        st.sampled_from(_EDGES),
        st.sampled_from(
            list("가나다 漢字 ひらがな カタカナ ＡＢ 。、 abc Жд ש ع ह ก \t\n")
        ),
    ),
    max_size=400,
)


@settings(max_examples=500, deadline=None)
@given(_MULTILINGUAL)
def test_estimate_matches_reference(text: str) -> None:
    assert estimate_token_count(text) == _reference_estimate(text)


@settings(max_examples=200, deadline=None)
@given(_MULTILINGUAL, _MULTILINGUAL)
def test_units_are_additive(first: str, second: str) -> None:
    assert token_units(first + second) == token_units(first) + token_units(second)


def test_estimate_counts_each_block_edge() -> None:
    for ch in _EDGES:
        text = ch * 8
        assert estimate_token_count(text) == _reference_estimate(text), hex(ord(ch))
