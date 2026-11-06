# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unicode script helpers for Han, Hangul and Kana text."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from unified_kg_rag.shared.utils import scripts
from unified_kg_rag.shared.utils.scripts import (
    drop_dense_script_spaces,
    has_dense_script,
    is_dense_script_char,
    is_hangul_syllable,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("ch", ["가", "힣", "ㄱ", "あ", "カ", "東", "𠀋"])
def test_dense_script_letters(ch: str) -> None:
    assert is_dense_script_char(ch)


@pytest.mark.parametrize("ch", ["a", "Ä", "1", "。", "，", "Ａ", " "])
def test_not_dense_script_letters(ch: str) -> None:
    assert not is_dense_script_char(ch)


def test_has_dense_script() -> None:
    assert has_dense_script("Acme 코리아")
    assert not has_dense_script("Acme Korea")


def _in_letter_ranges(ch: str) -> bool:
    return any(lo <= ord(ch) <= hi for lo, hi in scripts._DENSE_SCRIPT_LETTER_RANGES)


def test_dense_script_char_matches_the_letter_ranges_at_every_code_point() -> None:
    # The compiled class answers as the range table it is built from.
    differing = [
        cp
        for cp in range(0x110000)
        if is_dense_script_char(chr(cp)) != _in_letter_ranges(chr(cp))
    ]
    assert differing == []


@given(st.text(alphabet=st.characters(codec=None), max_size=40))
def test_has_dense_script_is_any_dense_letter(text: str) -> None:
    assert has_dense_script(text) == any(_in_letter_ranges(ch) for ch in text)


def test_dense_script_char_takes_one_character() -> None:
    with pytest.raises(TypeError):
        is_dense_script_char("가나")


def test_is_hangul_syllable() -> None:
    assert is_hangul_syllable("한")
    assert not is_hangul_syllable("ㄱ")
    assert not is_hangul_syllable("東")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("가나다 상사", "가나다상사"),
        ("3억  원", "3억원"),
        ("Acme 코리아", "Acme코리아"),
        ("Acme   Corp", "Acme Corp"),
        ("青空 海運", "青空海運"),
    ],
)
def test_drop_dense_script_spaces(text: str, expected: str) -> None:
    assert drop_dense_script_spaces(text) == expected
