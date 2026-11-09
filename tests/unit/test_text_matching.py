# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared deterministic text matcher (AWS-free)."""

from __future__ import annotations

import time

import pytest

from unified_kg_rag.evaluation.text_matching import (
    is_spaceless_script,
    leading_number,
    phrase_in_text,
    strip_possessives,
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


class TestLoneLetters:
    @pytest.mark.parametrize("text", ["Type-A", "TypeA", "type-a", "TYPE-a"])
    def test_dashed_forms_are_one_token_in_either_case(self, text: str) -> None:
        assert tokenize(text) == ["typea"]

    @pytest.mark.parametrize(
        ("phrase", "text", "expected"),
        [
            ("Type-A", "type-a is used", True),
            ("type-a", "Type-A is used", True),
            ("Type-A", "Type-B is used", False),
            ("Plan B", "switch to plan B.", True),
            ("vendor", "the contract a vendor signed", True),
            ("C++", "written in C++ and C#", True),
            # A space before a lone capital is not deleted: the letter is a
            # word (or an article) of its own, as in lowercase text.
            ("yes", "Yes I agree with that", True),
            ("Vitamin A", "low vitamin a levels", True),
            ("Plan B", "they chose plan b", True),
            ("Vendor", "the Vendor I contacted", True),
        ],
    )
    def test_matching(self, phrase: str, text: str, expected: bool) -> None:
        assert phrase_in_text(phrase, text) is expected

    def test_a_spaced_lone_capital_stays_its_own_word(self) -> None:
        assert tokenize("Yes I agree") == ["yes", "i", "agree"]
        assert tokenize("Vitamin A") == tokenize("vitamin a")

    @pytest.mark.perf
    def test_many_dashes_normalize_in_linear_time(self) -> None:
        # Each dash once sliced the whole text: 4,000 dashes in 70 KB took
        # seconds. Linear, it takes milliseconds; the bound only catches a
        # quadratic regression. CPU time: a busy runner delays the process,
        # it does not add to it.
        text = "word-" * 4_000 + "x" * 50_000
        started = time.process_time()
        tokens = tokenize(text)
        assert time.process_time() - started < 1.0
        assert tokens.count("word") == 4_000

    def test_many_dashes_normalize_to_words(self) -> None:
        # The timing check's input, on every run: its result.
        tokens = tokenize("word-" * 4_000 + "x" * 50_000)
        assert tokens.count("word") == 4_000


class TestPossessives:
    @pytest.mark.parametrize(
        ("phrase", "text"),
        [
            ("Depot", "Depot's widgets ship today."),
            ("Depot", "Depot’s widgets ship today."),  # typographic apostrophe
            ("Depot", "DEPOT'S widgets"),
            ("Depot", "Depot＇s widgets"),  # fullwidth, folded by NFKC
            ("Vendor's depot", "the Vendor’s depot"),
            ("Vendor's depot", "the Vendor depot"),
        ],
    )
    def test_a_possessive_matches_its_word(self, phrase: str, text: str) -> None:
        # Before, punctuation removal left "depots", which "depot" missed.
        assert phrase_in_text(phrase, text)

    @pytest.mark.parametrize(
        ("phrase", "text"), [("Depot", "Depots store widgets."), ("Depot", "Depotsy")]
    )
    def test_a_plural_is_not_a_possessive(self, phrase: str, text: str) -> None:
        assert not phrase_in_text(phrase, text)

    def test_only_the_matching_path_drops_them(self) -> None:
        assert strip_possessives("Depot's and Mill’s") == "Depot and Mill"
        # Exact match and token F1 keep the official normalization.
        assert tokenize("Depot's") == ["depots"]


class TestLatinInsideChineseAndJapanese:
    """Chinese and Japanese put no space around a Latin word or a number."""

    @pytest.mark.parametrize(
        ("phrase", "text"),
        [
            ("AWS", "AWSは東京にデータセンターがあります。"),
            ("AWS", "答案是AWS公司。"),
            ("Acme Corp", "供应商是Acme Corp，位于上海。"),
            ("Acme Corp", "ベンダーはAcme Corpです。"),
            ("2024", "契約は2024年に締結されました。"),
            ("2024", "合同于2024年签订。"),
            ("C++", "用C++编写"),
        ],
    )
    def test_match(self, phrase: str, text: str) -> None:
        assert phrase_in_text(phrase, text) is True

    @pytest.mark.parametrize(
        ("phrase", "text"),
        [
            ("AI", "airport附近"),  # a Latin letter continues the word
            ("2", "合同期限12年"),  # a digit continues the number
            ("1.5", "增长11.5倍"),
            ("3", "金额为3億円"),  # a Han numeral after a digit scales it
            ("Corp", "AcmeCorp的"),
            # Korean keeps its token rules: a Hangul neighbour is judged there
            ("AWS", "AWS연구소에서"),
            ("2024", "2024억"),
            ("AI", "AI(인공지능)"),
        ],
    )
    def test_no_match(self, phrase: str, text: str) -> None:
        assert phrase_in_text(phrase, text) is False


class TestKoreanCounters:
    @pytest.mark.parametrize(
        "text",
        [
            "벤더는 3곳입니다.",
            "3군데에 공급합니다",
            "3가지 제품",
            "3개사가 공급합니다",
        ],
    )
    def test_a_count_keeps_its_value_before_a_counter(self, text: str) -> None:
        assert phrase_in_text("3", text) is True


class TestLeadingNumber:
    @pytest.mark.parametrize(
        ("text", "number"),
        [
            ("Three vendors supply it. Two of them are local.", 3),
            ("It has 3 vendors (one with 2 products).", 3),
            ("twenty-one vendors", 21),
            ("It supplies no buyers.", 0),
            ("None.", 0),
            ("The largest has 4,486 employees.", 4486),
            ("# 2 headings\nIt has 5 vendors.", 5),
            ("세 곳의 벤더가 공급합니다.", 3),
            ("총 4곳이며, 그중 3곳은 국내입니다.", 4),
            ("열두 개사가 공급합니다", 12),
            ("공급하는 벤더가 없습니다", 0),
            ("ベンダーは三社です。", 3),
            ("二十三件です", 23),
            ("ベンダーは3社です", 3),
        ],
    )
    def test_the_first_number_stated(self, text: str, number: float) -> None:
        assert leading_number(text) == number

    @pytest.mark.parametrize(
        ("text", "number"),
        [
            # Regression (rv9 query L13): 台 and 대 were no counters, 万 and
            # 만 no multipliers.
            ("十二台を不合格とした。", 12),
            ("十二台不合格。", 12),
            ("两千台", 2000),
            ("오리온 랩스는 열두 대를 불합격 처리했다.", 12),
            ("一万二千台", 12000),
            ("十二万台", 120_000),
            ("一億二千万人", 120_000_000),
            ("만 이천 대", 12000),
            ("십이 대가 불합격이었다.", 12),
            # Digits a multiplier scales.
            ("1만 2천 대", 12000),
            ("3万社", 30000),
            ("1천만 명", 10**7),
            # Regression (rv10 query M2): a decimal or thousands separator
            # before a multiplier dropped the number (and the next was
            # read); 零 holds an empty place.
            ("约1.5万人", 15000),
            ("1.5万人です", 15000),
            ("3,000万円", 30_000_000),
            ("1.5억 원", 150_000_000),
            ("1,200만 명", 12_000_000),
            ("약 1.5만 명의 직원과 3개 공장", 15000),
            ("一千零一家", 1001),
            ("两千零二十四家", 2024),
            ("三千零五十家", 3050),
            # Low: 円 (and 元) count a sum.
            ("一億円", 100_000_000),
            ("三千万元", 30_000_000),
        ],
    )
    def test_cjk_counters_and_multipliers(self, text: str, number: float) -> None:
        assert leading_number(text) == number

    @pytest.mark.parametrize(
        "text",
        [
            # A Sino-Korean digit alone is another word ("이 곳": this place),
            # and Han digits without a counter are no count (a year).
            "이 곳은 넓다",
            "二〇二四年",
        ],
    )
    def test_cjk_digits_that_state_no_count(self, text: str) -> None:
        assert leading_number(text) is None

    @pytest.mark.parametrize(
        ("text", "number"),
        [
            # List enumerators and citation markers count nothing.
            ("1. Acme\n2. Beta\n3. Gamma\nIn total, three vendors.", 3),
            ("(1) Acme\n(2) Beta\nTwo vendors.", 2),
            ("Acme [1], Beta [2] and Gamma [3]: three vendors.", 3),
            # Years.
            ("As of 2024, three vendors supply Buyer A.", 3),
            ("In March 2024 it had 7 buyers.", 7),
            ("2024년 기준 3곳입니다.", 3),
            ("2024年時点で3社です。", 3),
            ("It has 2003 employees.", 2003),
            # Ordinals.
            ("공급 벤더는 두 번째로 많은 네 곳입니다.", 4),
            ("2번째로 많은 3곳입니다.", 3),
            ("三つ目の供給元を含め、五社です。", 5),
            ("3番目に多いのは4社です。", 4),
            ("第三家供应商之后，共有五家。", 5),
            # "no" and a negated number that modify something else.
            ("No single vendor dominates; three vendors supply Buyer A.", 3),
            ("No, three vendors supply it.", 3),
            ("No more than 5 vendors.", 5),
            ("One of them, Acme, is the largest; in total four vendors.", 4),
            ("Not one but three vendors.", 3),
            ("Buyer Aへの供給元は一つではなく、三社あります。", 3),
            ("不是一家，而是三家。", 3),
            ("한 곳이 아니라 세 곳입니다.", 3),
            ("문제없습니다. 공급사는 세 곳입니다.", 3),
            # Zero, stated with a negation.
            ("Zortal does not supply any buyers.", 0),
            ("Zortal doesn't have any buyers.", 0),
            ("한 곳도 없습니다.", 0),
            ("一社もありません。", 0),
            ("一家也没有。", 0),
        ],
    )
    def test_numbers_that_count_nothing_are_skipped(
        self, text: str, number: float
    ) -> None:
        assert leading_number(text) == number

    @pytest.mark.parametrize(
        "text",
        [
            "I could not find Zortal13 Holdings.",  # a number inside a word
            "네, 공급합니다.",  # "네" is yes, not four, without a counter
            "The 3rd vendor",
            "1. Acme\n2. Beta",  # enumerators only
            "As of 2024, it is unknown.",
            "No, it does not.",
        ],
    )
    def test_none(self, text: str) -> None:
        assert leading_number(text) is None

    @pytest.mark.parametrize(
        ("text", "number"),
        [
            # A partitive states its part; the whole it is taken from counts
            # nothing ("of the three", "三家中", "세 곳 중", "三社のうち").
            ("Only one of the three vendors supplies X.", 1),
            ("One of the vendors, Acme, supplies X.", 1),
            ("Of the three vendors, only one supplies X.", 1),
            ("It has a total of three vendors.", 3),
            ("三家中只有一家。", 1),
            ("三家供应商中，只有一家供货。", 1),
            ("三家中国供应商。", 3),
            ("세 곳 중 한 곳만 공급합니다.", 1),
            ("세 곳 중에서 두 곳이 국내입니다.", 2),
            ("三社のうち一社だけです。", 1),
            # "Not one" and "not a single" state zero, "not one but" negates.
            ("Not one vendor supplies X.", 0),
            ("Not a single vendor supplies X.", 0),
            ("Not one but three vendors.", 3),
            # 两/兩 is two.
            ("共有两家供应商。", 2),
            ("共有兩家供應商。", 2),
            ("两个", 2),
            # Zero stated by a Chinese or Japanese verb of absence.
            ("没有供应商。", 0),
            ("沒有供應商。", 0),
            ("サプライヤーはいません。", 0),
            ("供給元はありません。", 0),
            # A space before a Han counter.
            ("三 社", 3),
            ("三 家供应商", 3),
            # English scale words.
            ("three hundred vendors", 300),
            ("two thousand", 2000),
            ("three hundred and twelve buyers", 312),
            ("two thousand five hundred employees", 2500),
            ("twenty-one hundred", 2100),
        ],
    )
    def test_partitives_negated_zero_and_scales(self, text: str, number: float) -> None:
        assert leading_number(text) == number

    @pytest.mark.parametrize(
        "text",
        [
            "Aではありません。",  # a negated copula, not absence
            "問題ありません。",  # "no problem"
            "没有问题。",  # "no problem"
            "One of them, Acme, is the largest.",  # an anaphoric partitive
            "特に問題はありません。",  # "no particular problem"
            "Acme is not one of the suppliers.",  # a negated partitive
        ],
    )
    def test_idioms_of_absence_count_nothing(self, text: str) -> None:
        assert leading_number(text) is None
