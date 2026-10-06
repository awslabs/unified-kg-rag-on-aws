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
            ("서울 특별시", "본사는 서울 특별시에 있다"),  # particle on last word
            ("서울", "본사는 서울은 아니다"),  # single word: substring
            ("東京電力", "本契約は東京電力と締結された"),
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
        ],
    )
    def test_no_match(self, phrase: str, text: str) -> None:
        assert phrase_in_text(phrase, text) is False

    def test_spaceless_script_detection(self) -> None:
        assert is_spaceless_script("東京電力")
        assert is_spaceless_script("서울")
        assert not is_spaceless_script("서울 특별시")
        assert not is_spaceless_script("Acme")
