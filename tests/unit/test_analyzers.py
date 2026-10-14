# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyzers of the embedded search engine (``adapters.local.search_engine``)."""

from __future__ import annotations

import logging
import sys

import pytest

from unified_kg_rag.adapters.local.search_engine import analyzers
from unified_kg_rag.adapters.local.search_engine.analyzers import (
    english_analyzer,
    get_analyzer,
    register_analyzer,
    standard_analyzer,
)

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
