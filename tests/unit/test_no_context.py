# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fixed no-context reply is written in the query's language."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import LanguageCode
from unified_kg_rag.domain.retrieval import no_context
from unified_kg_rag.domain.retrieval.no_context import (
    NO_CONTEXT_ANSWER,
    NO_CONTEXT_ANSWERS,
    answer_language,
    is_no_context_answer,
    no_context_answer,
)

pytestmark = pytest.mark.unit

EN, KO = LanguageCode.EN, LanguageCode.KO


def test_every_language_has_its_own_reply() -> None:
    assert set(NO_CONTEXT_ANSWERS) == set(LanguageCode)
    assert len(set(NO_CONTEXT_ANSWERS.values())) == len(LanguageCode)
    # The English reply is the one earlier releases gave.
    assert NO_CONTEXT_ANSWER == (
        "I could not find relevant information in the available data to answer "
        "this question."
    )


def test_the_function_word_lists_are_disjoint() -> None:
    lists = list(no_context._FUNCTION_WORDS.values())
    for i, words in enumerate(lists):
        for other in lists[i + 1 :]:
            assert not words & other


@pytest.mark.parametrize(
    ("query", "language"),
    [
        ("Who supplies the Buyer?", EN),
        ("벤더는 누구에게 공급합니까?", KO),
        # Mixed with Latin names, still Korean.
        ("Vendor의 납품 기한은?", KO),
        ("ベンダーの納期はいつですか?", LanguageCode.JA),
        ("供应商的交货期限是什么?", LanguageCode.ZH),
        ("Кто поставляет товар покупателю?", LanguageCode.RU),
        ("Wer ist der Lieferant und was liefert er?", LanguageCode.DE),
        ("¿Quién es el proveedor del comprador?", LanguageCode.ES),
        ("Qui est le fournisseur et quand livre-t-il?", LanguageCode.FR),
        ("Chi è il fornitore della merce?", LanguageCode.IT),
        ("Quem é o fornecedor do comprador?", LanguageCode.PT),
        # The script most of the letters are in decides, not the first found.
        ("What does 계약 mean in the Vendor contract?", EN),
        ("AWS Lambda 함수의 최대 실행 시간은 얼마입니까?", KO),
        ("Acme社の売上高(매출)はいくらですか？", LanguageCode.JA),
        ("供給者는 購買者에게 部品을 供給한다", KO),
    ],
)
def test_the_reply_follows_the_query_language(
    query: str, language: LanguageCode
) -> None:
    assert answer_language(query, EN) is language
    assert no_context_answer(query, EN) == NO_CONTEXT_ANSWERS[language]


def test_an_undecided_query_gets_the_index_language_in_its_script() -> None:
    # Only names: the index language, when it is written in Latin script.
    assert answer_language("Vendor GmbH", LanguageCode.DE) is LanguageCode.DE
    # A Korean index's reply is not given to a Latin-script query.
    assert answer_language("Vendor GmbH", KO) is EN
    # Kanji without kana: Japanese on a Japanese index, else Chinese.
    assert answer_language("東京本社", LanguageCode.JA) is LanguageCode.JA
    assert answer_language("東京本社", EN) is LanguageCode.ZH
    # An unknown code falls back to English.
    assert answer_language("Vendor", "xx") is EN


def test_every_reply_is_recognised() -> None:
    for reply in NO_CONTEXT_ANSWERS.values():
        assert is_no_context_answer(f" {reply}\n")
    assert not is_no_context_answer("Vendor supplies Buyer.")
