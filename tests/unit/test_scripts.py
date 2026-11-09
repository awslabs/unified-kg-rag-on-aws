# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The script table and its helpers for Han, Hangul and Kana text."""

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
    script_named,
    script_of,
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
    dense = scripts._ranges_of(scripts.Spacing.DENSE)
    return any(lo <= ord(ch) <= hi for lo, hi in dense)


@pytest.mark.slow
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


def test_script_blocks_are_disjoint_and_sorted_once() -> None:
    spans = sorted(span for script in scripts.SCRIPTS for span in script.ranges)
    assert all(lo <= hi for lo, hi in spans)
    assert all(prev[1] < nxt[0] for prev, nxt in zip(spans, spans[1:], strict=False))


@pytest.mark.parametrize(
    ("ch", "name", "spacing"),
    [
        ("a", "latin", scripts.Spacing.SPACED),
        ("ệ", "latin", scripts.Spacing.SPACED),
        ("Ж", "cyrillic", scripts.Spacing.SPACED),
        ("ש", "hebrew", scripts.Spacing.SPACED),
        ("ع", "arabic", scripts.Spacing.SPACED),
        ("ह", "devanagari", scripts.Spacing.SPACED),
        ("ก", "thai", scripts.Spacing.SPACELESS),
        ("ລ", "lao", scripts.Spacing.SPACELESS),
        ("က", "myanmar", scripts.Spacing.SPACELESS),
        ("ក", "khmer", scripts.Spacing.SPACELESS),
        ("한", "hangul", scripts.Spacing.DENSE),
        ("ㄱ", "hangul", scripts.Spacing.DENSE),
        ("カ", "kana", scripts.Spacing.DENSE),
        ("東", "han", scripts.Spacing.DENSE),
        ("𠮷", "han", scripts.Spacing.DENSE),
    ],
)
def test_script_of(ch: str, name: str, spacing: scripts.Spacing) -> None:
    script = script_of(ch)
    assert script is not None
    assert (script.name, script.spacing) == (name, spacing)
    assert script_named(name) is script


@pytest.mark.parametrize("ch", ["1", " ", "。", "，", "$", "\u0301", "😀"])
def test_script_of_non_letters_is_none(ch: str) -> None:
    assert script_of(ch) is None


def test_letter_class_matches_the_script_of_each_letter() -> None:
    import re

    hangul = re.compile(f"[{scripts.letter_class('hangul')}]")
    assert hangul.match("한") and hangul.match("ㅎ")
    assert not hangul.match("カ")


def test_dense_letters_are_the_dense_scripts_letters() -> None:
    # is_dense_script_char and script_of agree on every dense block edge.
    for script in scripts.SCRIPTS:
        for lo, hi in script.ranges:
            for cp in (lo, hi):
                dense = script.spacing is scripts.Spacing.DENSE
                assert is_dense_script_char(chr(cp)) is dense, hex(cp)


def test_one_latin_token_rate_and_one_hangul_predicate() -> None:
    # The chunk-size bound, the token estimate and the script table read one
    # rate; the evaluators' Hangul test is the table's (syllables and jamo).
    from unified_kg_rag.adapters.llm import token_estimate
    from unified_kg_rag.domain.ingestion import chunk_length
    from unified_kg_rag.shared.utils import scripts

    assert chunk_length.LATIN_CHARS_PER_TOKEN is scripts.LATIN_CHARS_PER_TOKEN
    assert token_estimate.LATIN_CHARS_PER_TOKEN is scripts.LATIN_CHARS_PER_TOKEN
    assert scripts.script_named("latin").chars_per_token == 4
    assert [scripts.is_hangul(c) for c in "한ㄱᄀA漢"] == [True] * 3 + [False] * 2
    assert [scripts.is_hangul_syllable(c) for c in "한ㄱ"] == [True, False]


@pytest.mark.parametrize(
    ("text", "folded"),
    [
        ("Café Nguyễn", "Cafe Nguyen"),
        ("Đặng ĐỖ", "Dang DO"),
        ("Łódź", "Lodz"),
        ("Yıldız İnönü", "Yildiz Inonu"),
        ("Søren Ærø Œuvre þing Ħamrun", "Soren AEro OEuvre thing Hamrun"),
        # Other scripts keep their marks and letters.
        ("ガ हिन्दी Ёлка", "ガ हिन्दी Ёлка"),
    ],
)
def test_without_latin_accents_folds_every_latin_mark(text: str, folded: str) -> None:
    # Regression (rv11 query M3): one fold, linking's and grounding's, for
    # the decomposing accents and the letters no decomposition takes a mark
    # from.
    assert scripts.without_latin_accents(text) == folded
