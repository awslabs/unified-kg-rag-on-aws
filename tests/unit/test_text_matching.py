# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared deterministic text matcher (AWS-free)."""

from __future__ import annotations

import pytest

from unified_kg_rag.evaluation.text_matching import (
    is_spaceless_script,
    phrase_in_text,
    tokenize,
)

pytestmark = pytest.mark.unit


class TestTokenize:
    def test_squad_normalized_tokens(self) -> None:
        assert tokenize("The Vendor, Inc. ships 1,000 parts!") == [
            "vendor",
            "inc",
            "ships",
            "1000",
            "parts",
        ]


class TestPhraseInText:
    @pytest.mark.parametrize(
        ("phrase", "text"),
        [
            ("Acme", "owned by Acme, Inc."),
            ("works at", "she works at acme"),
            ("net 30 days", "Payment: Net-30 days? no, net 30 days."),
            ("1000", "a fee of USD 1,000 applies"),
            ("1.5 million", "about 1.5 million users"),
            ("C++", "written in C++."),
            ("서울 특별시", "본사는 서울 특별시에 있다"),  # particle on last word
            ("서울", "본사는 서울은 아니다"),  # single word: substring
            ("東京電力", "本契約は東京電力と締結された"),
            # Latin gold answers followed by an attached Korean particle
            ("AWS", "AWS는 클라우드 사업자이다"),
            ("2024년", "계약은 2024년에 체결되었다"),
            ("Acme Corp", "공급자는 Acme Corp입니다"),
            # spacing variants of Korean phrases, in both directions
            ("가나다 상사", "공급자는 가나다상사이다"),
            ("가나다상사", "공급자는 가나다 상사이다"),
            ("3억 원", "계약 금액은 3억원이다"),
            ("3억원", "계약 금액은 3억 원이다"),
            ("2년", "보증 기간은 2년이다"),
            ("二年", "保証期間は二年である"),
            # a bare number followed by a Korean unit (and a particle) keeps
            # its value
            ("2024", "계약은 2024년에 체결되었다"),
            ("2", "기간은 2개월이다"),
            ("3", "참석자는 3명이었다"),
            ("5000", "수수료는 5,000원이다"),
            ("12", "총 12회"),
            # separators and dashes
            ("2,000", "a fee of 2000 applies"),
            ("1990", "from 1990-1995"),
            ("1995", "from 1990–1995"),
            ("5 year", "a 5-year term"),
            ("5-year", "a 5 year term"),
            ("-5", "the balance is -5 units"),
            ("-5", "the balance is (−5) units"),
            ("3.5%", "a rate of 3.5% applies"),
        ],
    )
    def test_match(self, phrase: str, text: str) -> None:
        assert phrase_in_text(phrase, text) is True

    @pytest.mark.parametrize(
        ("phrase", "text"),
        [
            ("AI", "the airport is busy"),
            ("at works", "she works at acme"),
            ("서울 특별시", "서울에 있는 특별시"),  # words not contiguous
            ("서울 특별시", "서울 특별시청사관리본부"),  # suffix longer than a particle
            ("", "anything"),
            ("the", "the vendor"),  # normalizes to nothing
            # a number inside a longer number is a different number
            ("2년", "보증 기간은 12년이다"),
            ("24개월", "보증 기간은 124개월이다"),
            ("二年", "保証期間は十二年である"),
            ("3억 원", "계약 금액은 13억 원이다"),
            ("3억원", "계약 금액은 13억원이다"),
            ("AWS", "XAWS는 별개 회사이다"),
            ("2024", "20245년에"),  # digits continue the token
            # a magnitude numeral after a bare number changes its value
            ("3", "금액은 3억이다"),
            ("3", "금액은 3억원이다"),
            ("5", "금액은 5천만 원이다"),
            ("2", "인원은 2만 명이다"),
            ("2년", "변동폭은 -2년이다"),
            # sign, ranges and percent signs are part of the value
            ("5", "the balance is -5 units"),
            ("-5", "the balance is 5 units"),
            ("19901995", "from 1990-1995"),
            ("3.5%", "about 3.5 million users"),
            ("3.5%", "a rate of 35% applies"),
            ("3.5", "a rate of 3.5% applies"),
            ("1.5", "about 15 users"),
            # a decimal point or comma between digits is part of the number
            ("15 million", "about 1.5 million users"),
            ("5", "rate of 1.5 percent"),
            ("2년", "기간은 1.2년이다"),
            ("C", "written in C++"),
            ("AWS", "AWS클라우드서비스는"),  # more than a particle
        ],
    )
    def test_no_match(self, phrase: str, text: str) -> None:
        assert phrase_in_text(phrase, text) is False

    def test_ranges_and_compounds_tokenize_apart(self) -> None:
        assert tokenize("1990-1995") == ["1990", "1995"]
        assert tokenize("5-year, 3.5% and -2") == ["5", "year", "3.5%", "and", "-2"]

    def test_spaceless_script_detection(self) -> None:
        assert is_spaceless_script("東京電力")
        assert is_spaceless_script("서울")
        assert not is_spaceless_script("서울 특별시")
        assert not is_spaceless_script("Acme")
