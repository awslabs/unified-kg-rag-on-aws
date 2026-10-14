# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyzers of the embedded search engine (``adapters.local.search_engine``)."""

from __future__ import annotations

import logging
import random
import re
import sys

import pytest

from unified_kg_rag.adapters.local.search_engine import analyzers
from unified_kg_rag.adapters.local.search_engine.analyzers import (
    english_analyzer,
    get_analyzer,
    register_analyzer,
    standard_analyzer,
)
from unified_kg_rag.shared.utils.scripts import is_dense_script_char

pytestmark = pytest.mark.unit


def test_korean_runs_become_overlapping_bigrams() -> None:
    assert standard_analyzer("홍길동으로부터") == [
        "홍길",
        "길동",
        "동으",
        "으로",
        "로부",
        "부터",
    ]
    # So a word matches inside a longer run: every bigram of the query is
    # one of the document's.
    assert set(standard_analyzer("홍길동")) <= set(standard_analyzer("홍길동으로부터"))


def test_japanese_and_chinese_bigrams() -> None:
    assert standard_analyzer("東京都に住む") == ["東京", "京都", "都に", "に住", "住む"]
    assert standard_analyzer("カタカナ") == ["カタ", "タカ", "カナ"]
    assert standard_analyzer("北京大学") == ["北京", "京大", "大学"]


def test_single_dense_letter_is_a_unigram() -> None:
    assert standard_analyzer("2년 후") == ["2", "년", "후"]


def test_mixed_scripts_split_at_script_boundaries() -> None:
    assert standard_analyzer("Vendor가나다 ships 3 Widgets, 東京") == [
        "vendor",
        "가나",
        "나다",
        "ships",
        "3",
        "widgets",
        "東京",
    ]


@pytest.mark.parametrize("name", ["standard", "nori", "kuromoji", "smartcn"])
def test_cjk_analyzer_names_share_the_bigram_analyzer(name: str) -> None:
    assert get_analyzer(name) is standard_analyzer


def test_unknown_analyzer_falls_back_to_standard_and_warns_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        assert get_analyzer("no-such-analyzer-x") is standard_analyzer
        assert get_analyzer("no-such-analyzer-x") is standard_analyzer
    assert caplog.text.count("no-such-analyzer-x") == 1


def test_register_analyzer_adds_a_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(analyzers, "_ANALYZERS", dict(analyzers._ANALYZERS))

    @register_analyzer("whitespace")
    def whitespace(text: str) -> list[str]:
        return text.split()

    assert get_analyzer("whitespace")("A b") == ["A", "b"]


def test_english_strips_possessives_and_stop_words_and_stems() -> None:
    pytest.importorskip("snowballstemmer")
    assert english_analyzer("The Vendor's suppliers are shipping widgets") == [
        "vendor",
        "supplier",
        "ship",
        "widget",
    ]
    # Dense-script text still becomes bigrams.
    assert english_analyzer("supplies 공급계약") == ["suppli", "공급", "급계", "계약"]


def test_english_stop_words_are_lucenes() -> None:
    # OpenSearch's ``english`` analyzer drops exactly these (``_english_``).
    assert len(analyzers.ENGLISH_STOP_WORDS) == 33
    assert {"the", "a", "is", "with", "their"} <= analyzers.ENGLISH_STOP_WORDS
    pytest.importorskip("snowballstemmer")
    assert english_analyzer("interest bill amount") == ["interest", "bill", "amount"]
    assert english_analyzer("fire system") == ["fire", "system"]
    assert english_analyzer("computer name detail") == ["comput", "name", "detail"]


def test_english_without_snowballstemmer_does_not_stem(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "snowballstemmer", None)  # import fails
    monkeypatch.setattr(analyzers, "_warned", set())
    analyzers._porter_stem.cache_clear()
    try:
        with caplog.at_level(logging.WARNING):
            assert english_analyzer("The vendors are shipping") == [
                "vendors",
                "shipping",
            ]
            english_analyzer("again")
        assert caplog.text.count("snowballstemmer is not installed") == 1
    finally:
        analyzers._porter_stem.cache_clear()


def _per_character_tokens(text: str) -> list[str]:
    """The ``standard`` tokens by a per-character script test (reference)."""
    tokens: list[str] = []
    for word in re.findall(r"\w+", text.lower()):
        start = 0
        for i in range(1, len(word) + 1):
            if i == len(word) or is_dense_script_char(word[i]) != is_dense_script_char(
                word[start]
            ):
                run = word[start:i]
                if is_dense_script_char(run[0]):
                    tokens.extend(
                        [run]
                        if len(run) == 1
                        else [run[i : i + 2] for i in range(len(run) - 1)]
                    )
                else:
                    tokens.append(run)
                start = i
    return tokens


def test_dense_run_split_matches_a_per_character_scan() -> None:
    rng = random.Random(3)
    words = [
        "Vendor",
        "ships",
        "3",
        "홍길동으로부터",
        "AI기반",
        "東京都に住む",
        "カタカナx",
        "北京大学",
        "2년",
        "x東y",
        "가",
        "ㄱㄴ",
        "\U00020000\U00020001b",
        "été",
        "under_score",
    ]
    for _ in range(500):
        text = " ".join(rng.choices(words, k=rng.randint(1, 12)))
        assert standard_analyzer(text) == _per_character_tokens(text)


def test_english_stems_each_distinct_word_once(monkeypatch: pytest.MonkeyPatch) -> None:
    snowballstemmer = pytest.importorskip("snowballstemmer")
    calls: list[str] = []
    real = snowballstemmer.stemmer

    def counting(name: str) -> object:
        stemmer = real(name)

        class Counting:
            def stemWord(self, word: str) -> str:  # noqa: N802
                calls.append(word)
                return str(stemmer.stemWord(word))

        return Counting()

    monkeypatch.setattr(snowballstemmer, "stemmer", counting)
    analyzers._porter_stem.cache_clear()
    try:
        assert english_analyzer("shipping widgets shipping widgets") == [
            "ship",
            "widget",
            "ship",
            "widget",
        ]
        assert sorted(calls) == ["shipping", "widgets"]
    finally:
        analyzers._porter_stem.cache_clear()


def test_in_word_punctuation_joins_like_opensearch() -> None:
    # OpenSearch 2.13's _analyze output for the same text.
    assert standard_analyzer("don't 1,000.50 U.S.A. e-mail a:b 12:30 3.2.1(a)") == [
        "don't",
        "1,000.50",
        "u.s.a",
        "e",
        "mail",
        "a:b",
        "12",
        "30",
        "3.2.1",
        "a",
    ]
    # Not across a letter-digit boundary, nor into a dense-script run.
    assert standard_analyzer("a1.b 1.a 가.나") == ["a1", "b", "1", "a", "가", "나"]


# OpenSearch 2.13's _analyze output (standard analyzer) for the same text.
_OPENSEARCH_TOKENS = {
    # Combining marks stay in their word (Python's \w has none of them).
    "हिन्दी भाषा में अनुबंध": ["हिन्दी", "भाषा", "में", "अनुबंध"],
    "বাংলা ভাষায় চুক্তি": ["বাংলা", "ভাষায়", "চুক্তি"],
    "தமிழ் ஒப்பந்தம்": ["தமிழ்", "ஒப்பந்தம்"],
    "العَرَبِيَّة الْعَقْد": ["العَرَبِيَّة", "الْعَقْد"],
    "עִבְרִית חוֹזֶה": ["עִבְרִית", "חוֹזֶה"],
    "café résumé": ["café", "résumé"],
    "́abc x ́ y": ["abc", "x", "y"],
    # Spaceless scripts: a whole run, apart from other scripts and digits.
    "ภาษาไทย สัญญาที่ลงนาม": ["ภาษาไทย", "สัญญาที่ลงนาม"],
    "မြန်မာဘာသာ ខ្មែរ": ["မြန်မာဘာသာ", "ខ្មែរ"],
    "ไทยabc ไทย๑๒ a_ไทย ไทย_ไทย": [
        "ไทย",
        "abc",
        "ไทย",
        "๑๒",
        "a_",
        "ไทย",
        "ไทย",
        "ไทย",
    ],
    # In-word punctuation of other scripts and widths.
    "a：b a‧b ab︓cd ١٬٠٠٠ 12؍34 a٬b": [
        "a：b",
        "a‧b",
        "ab︓cd",
        "١٬٠٠٠",
        "12؍34",
        "a",
        "b",
    ],
    "צה\"ל ג'ורג' ab\"cd": ['צה"ל', "ג'ורג'", "ab", "cd"],
    # Format characters join, but a zero width space separates; a connector
    # (or a narrow no-break space) joins but is no word on its own.
    "co­op a‍b a​b 1 000 _a_ __ foo‿bar": [
        "co­op",
        "a‍b",
        "a",
        "b",
        "1 000",
        "_a_",
        "foo‿bar",
    ],
    # Symbols are no word characters, but symbols that are letters are.
    "x² H₂O ½ ① Ⓐb": ["x", "h", "o", "ⓐb"],
}


@pytest.mark.parametrize("text", list(_OPENSEARCH_TOKENS))
def test_words_split_like_opensearch_in_every_script(text: str) -> None:
    assert standard_analyzer(text) == _OPENSEARCH_TOKENS[text]


def test_text_beyond_the_basic_multilingual_plane_splits_the_same() -> None:
    # Such text takes the slower pattern with the ranges above U+FFFF.
    for text, expected in _OPENSEARCH_TOKENS.items():
        assert standard_analyzer(f"{text} 𝐀b \U00020000") == [
            *expected,
            "𝐀b",
            "\U00020000",
        ]
    assert standard_analyzer("a\U000e0100b 🄰b 𑄃𑄨") == ["a\U000e0100b", "🄰b", "𑄃𑄨"]
