# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The language registry (``domain.languages``): which language a text's
script decides."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.languages import dense_language
from unified_kg_rag.domain.models import LanguageCode
from unified_kg_rag.shared.utils.scripts import ScriptCounts, count_letters

pytestmark = pytest.mark.unit

KO, JA, ZH = LanguageCode.KO, LanguageCode.JA, LanguageCode.ZH


@pytest.mark.parametrize(
    ("hangul", "kana", "han", "expected"),
    [
        (10, 0, 0, KO),
        # Korean with Hanja: a tenth Hangul is Korean.
        (1, 0, 9, KO),
        (1, 0, 10, ZH),
        (0, 1, 9, JA),
        (0, 0, 10, ZH),
        # A Korean word in Japanese text: more Kana than Hangul.
        (2, 7, 4, JA),
        (0, 0, 0, None),
    ],
)
def test_dense_language_by_the_hangul_and_kana_shares(
    hangul: int, kana: int, han: int, expected: LanguageCode | None
) -> None:
    counts = ScriptCounts({"hangul": hangul, "kana": kana, "han": han, "latin": 50})
    assert dense_language(counts) is expected


def test_han_alone_is_japanese_on_a_japanese_index() -> None:
    counts = count_letters(["東京本社"])
    assert dense_language(counts) is ZH
    assert dense_language(counts, JA) is JA
    assert dense_language(counts, KO) is ZH


def test_the_dominant_script_weighs_letters_by_their_token_rate() -> None:
    # Nine Latin letters weigh about two Hangul syllables.
    assert count_letters(["AWS Lambda의 실행 시간"]).dominant() == "dense"
    assert count_letters(["What does 계약 mean?"]).dominant() == "latin"
    assert count_letters(["Кто поставщик Acme?"]).dominant() == "cyrillic"
    assert count_letters(["123 ..."]).dominant() is None
