# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared deterministic text matcher (AWS-free)."""

from __future__ import annotations

import time

import pytest

from unified_kg_rag.evaluation.text_matching import (
    is_spaceless_script,
    number_value,
    phrase_in_text,
    stated_numbers,
    strip_possessives,
    tokenize,
    tokenized_by_script,
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
        assert is_spaceless_script("กรุงเทพ")
        assert is_spaceless_script("ພາສາລາວ")
        assert not is_spaceless_script("ירושלים")

    def test_the_scripts_token_f1_reads_by_script(self) -> None:
        # The dataset flag the scoring check reads: the tokenizations
        # differ on these scripts' text only.
        for text in ("東京", "서울", "กรุงเทพ", "ירושלים", "القاهرة"):
            assert tokenized_by_script(text), text
        for text in ("Acme Corp", "Москва", "दिल्ली", "Hà Nội"):
            assert not tokenized_by_script(text), text


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


class TestStatedNumbers:
    """``stated_numbers``: every number a text writes, in any position, 0
    for a statement of none; which one is the count is not read."""

    @pytest.mark.parametrize(
        ("text", "numbers"),
        [
            ("Three vendors supply it. Two of them are local.", {2, 3}),
            ("It has 3 vendors (one with 2 products).", {1, 2, 3}),
            ("twenty-one vendors", {21}),
            ("It supplies no buyers.", {0}),
            ("None.", {0}),
            ("The largest has 4,486 employees.", {4486}),
            ("# 2 headings\nIt has 5 vendors.", {5}),
            ("세 곳의 벤더가 공급합니다.", {3}),
            ("총 4곳이며, 그중 3곳은 국내입니다.", {3, 4}),
            ("열두 개사가 공급합니다", {12}),
            ("공급하는 벤더가 없습니다", {0}),
            ("ベンダーは三社です。", {3}),
            ("二十三件です", {23}),
            ("ベンダーは3社です", {3}),
            # Every number counts, wherever it stands: a partitive's whole,
            # a negated one, "one of them", "no" answering yes or no.
            ("Only one of the three vendors supplies X.", {1, 3}),
            ("One of the vendors, Acme, supplies X.", {1}),
            ("Of the three vendors, only one supplies X.", {1, 3}),
            ("It has a total of three vendors.", {3}),
            ("三家中只有一家。", {1, 3}),
            ("三家供应商中，只有一家供货。", {1, 3}),
            ("三家中国供应商。", {3}),
            ("세 곳 중 한 곳만 공급합니다.", {1, 3}),
            ("세 곳 중에서 두 곳이 국내입니다.", {2, 3}),
            ("三社のうち一社だけです。", {1, 3}),
            ("Not one vendor supplies X.", {0}),
            ("Not a single vendor supplies X.", {0}),
            ("Not one but three vendors.", {1, 3}),
            ("共有两家供应商。", {2}),
            ("共有兩家供應商。", {2}),
            ("两个", {2}),
            ("没有供应商。", {0}),
            ("沒有供應商。", {0}),
            ("サプライヤーはいません。", {0}),
            ("供給元はありません。", {0}),
            ("三 社", {3}),
            ("三 家供应商", {3}),
            ("three hundred vendors", {300}),
            ("two thousand", {2000}),
            ("three hundred and twelve buyers", {312}),
            ("two thousand five hundred employees", {2500}),
            ("twenty-one hundred", {2100}),
            # A hundreds place filled twice is no one number.
            ("three hundred hundred vendors", set()),
            ("trois cents cents", set()),
            ("As of 2024, it is unknown.", {2024}),
            ("No, it does not.", {0}),
            ("One of them, Acme, is the largest.", {1}),
            ("Acme is not one of the suppliers.", {1}),
        ],
    )
    def test_every_number_stated(self, text: str, numbers: set[float]) -> None:
        assert stated_numbers(text) == numbers

    @pytest.mark.parametrize(
        ("text", "numbers"),
        [
            # Regression (rv9 query L13): 台 and 대 were no counters, 万 and
            # 만 no multipliers; (rv10 query M2) a decimal or thousands
            # separator before a multiplier; 零 holds an empty place; 円 and
            # 元 count a sum.
            ("十二台を不合格とした。", {12}),
            ("十二台不合格。", {12}),
            ("两千台", {2000}),
            ("오리온 랩스는 열두 대를 불합격 처리했다.", {12}),
            ("一万二千台", {12_000}),
            ("十二万台", {120_000}),
            ("一億二千万人", {120_000_000}),
            ("만 이천 대", {12_000}),
            ("십이 대가 불합격이었다.", {12}),
            ("1만 2천 대", {12_000}),
            ("3万社", {30_000}),
            ("1천만 명", {10_000_000}),
            ("约1.5万人", {15_000}),
            ("1.5万人です", {15_000}),
            ("3,000万円", {30_000_000}),
            ("1.5억 원", {150_000_000}),
            ("1,200만 명", {12_000_000}),
            ("약 1.5만 명의 직원과 3개 공장", {3, 15_000}),
            ("一千零一家", {1001}),
            ("两千零二十四家", {2024}),
            ("三千零五十家", {3050}),
            ("一億円", {100_000_000}),
            ("三千万元", {30_000_000}),
            # Japanese 兆 (10^12) scales as 億 does.
            ("一兆二千億円", {1_200_000_000_000}),
            ("3兆円", {3_000_000_000_000}),
            # A sum in words, in the financial forms (大写, 大字).
            ("人民币壹拾万元", {100_000}),
            ("叁仟伍佰元", {3500}),
            ("金壱萬円", {10_000}),
            ("金参拾萬円", {300_000}),
            ("壹佰贰拾元", {120}),
            ("伍元", {5}),
            ("壹拾 万元", {100_000}),
            ("叁佰元伍角", {300}),
            ("壹萬2千円", {12_000}),
            ("金参百円", {300}),
            # 圆, the 大写 form of 元, counts a sum only.
            ("人民币叁佰圆整", {300}),
            ("人民币壹拾万圆整", {100_000}),
            ("五百圆", {500}),
        ],
    )
    def test_cjk_counters_and_multipliers(self, text: str, numbers: set[float]) -> None:
        assert stated_numbers(text) == numbers

    @pytest.mark.parametrize(
        ("text", "numbers"),
        [
            # Regression (rv37 matching M1): a comma before a run sets a
            # clause off, NFKC folding "，" to "," or written ASCII; only a
            # digit with a separator continues a number ("1.5万", "3,000万").
            ("是的，五家供应商。", {5}),
            ("据报道，三千元。", {3000}),
            ("截至2023年，十二家公司", {12, 2023}),
            ("每人，五十元", {50}),
            ("据报道，５万元。", {50_000}),
            ("合同金额,三万元", {30_000}),
            ("，叁佰圆整", {300}),
            ("はい,三社です。", {3}),
            ("没有，三家都未参与。", {0, 3}),
            ("1,5万", set()),
        ],
    )
    def test_a_comma_setting_a_clause_off_starts_a_run(
        self, text: str, numbers: set[float]
    ) -> None:
        assert stated_numbers(text) == numbers

    @pytest.mark.parametrize(
        ("text", "numbers"),
        [
            # A Han decimal point before 万 or 億 (zh numeral_decimal_points).
            ("一点五万元", {15_000}),
            ("十二点五万元", {125_000}),
            ("投资额为一点二亿元", {120_000_000}),
            ("一點五萬元", {15_000}),
            # Before a smaller multiplier it is another word: a time, a
            # little.
            ("三点五十分", set()),
            ("便宜一点五千元", {5000}),
        ],
    )
    def test_a_han_decimal_point_before_a_section_multiplier(
        self, text: str, numbers: set[float]
    ) -> None:
        assert stated_numbers(text) == numbers

    @pytest.mark.parametrize(
        ("text", "numbers"),
        [
            # Regression (rv35 index M1): a form of a sum in words is a
            # letter of another word outside a sum, even before a counter.
            ("大陆人", set()),
            ("陆家嘴", set()),
            ("登陆台湾", set()),
            ("队伍人数", set()),
            ("收拾家务", set()),
            ("放肆人", set()),
            ("人參家", set()),
            ("古参社員が担当します", set()),
            ("持参人払いの小切手", set()),
            ("降参人", set()),
            ("拾人", set()),
            ("没有大陆人参加", {0}),
            # An ordinary numeral beside one still counts.
            ("人参三人", {3}),
            ("陆家嘴有三家银行", {3}),
            # Regression (rv36 matching M1): so does one before a currency
            # counter; the run read as a sum first ("参二百", "伍五千",
            # "拾五千") is none.
            ("人参二百円", {200}),
            ("队伍五千元", {5000}),
            ("新参三千円", {3000}),
            ("收拾五千元", {5000}),
            ("人參一百元一斤", {100}),
            ("大陆三万元起", {30_000}),
            ("収拾100円", {100}),
            # A currency counter starting a word after a lone form makes
            # no sum.
            ("大陆元首", set()),
            ("登陆元宇宙", set()),
            ("队伍元气", set()),
            ("古参円熟", set()),
            ("人参二百圆", {200}),
            ("大陆圆满", set()),
            ("终于一圆梦想", set()),
            ("两圆相交", set()),
        ],
    )
    def test_a_sum_form_outside_a_sum_is_no_number(
        self, text: str, numbers: set[float]
    ) -> None:
        assert stated_numbers(text) == numbers

    @pytest.mark.parametrize(
        ("text", "numbers"),
        [
            # List enumerators, citation markers and ordinals are no
            # numbers; a year, "no" and a negated number are.
            ("1. Acme\n2. Beta\n3. Gamma\nIn total, three vendors.", {3}),
            ("(1) Acme\n(2) Beta\nTwo vendors.", {2}),
            ("Acme [1], Beta [2] and Gamma [3]: three vendors.", {3}),
            ("As of 2024, three vendors supply Buyer A.", {3, 2024}),
            ("In March 2024 it had 7 buyers.", {7, 2024}),
            ("2024년 기준 3곳입니다.", {3, 2024}),
            ("2024年時点で3社です。", {3, 2024}),
            ("It has 2003 employees.", {2003}),
            ("공급 벤더는 두 번째로 많은 네 곳입니다.", {4}),
            ("2번째로 많은 3곳입니다.", {3}),
            ("三つ目の供給元を含め、五社です。", {5}),
            ("3番目に多いのは4社です。", {4}),
            ("第三家供应商之后，共有五家。", {5}),
            ("No single vendor dominates; three vendors supply Buyer A.", {0, 3}),
            ("No, three vendors supply it.", {0, 3}),
            ("No more than 5 vendors.", {0, 5}),
            ("One of them, Acme, is the largest; in total four vendors.", {1, 4}),
            ("Not one but three vendors.", {1, 3}),
            ("Buyer Aへの供給元は一つではなく、三社あります。", {1, 3}),
            ("不是一家，而是三家。", {1, 3}),
            ("한 곳이 아니라 세 곳입니다.", {1, 3}),
            ("문제없습니다. 공급사는 세 곳입니다.", {3}),
            ("Zortal does not supply any buyers.", {0}),
            ("Zortal doesn't have any buyers.", {0}),
            ("한 곳도 없습니다.", {0}),
            ("一社もありません。", {0}),
            ("一家也没有。", {0}),
        ],
    )
    def test_markup_and_ordinals_state_no_number(
        self, text: str, numbers: set[float]
    ) -> None:
        assert stated_numbers(text) == numbers

    @pytest.mark.parametrize(
        "text",
        [
            "이 곳은 넓다",
            "二〇二四年",
            "I could not find Zortal13 Holdings.",
            "네, 공급합니다.",
            "The 3rd vendor",
            "1. Acme\n2. Beta",
            "Aではありません。",
            "問題ありません。",
            "没有问题。",
            "特に問題はありません。",
        ],
    )
    def test_none(self, text: str) -> None:
        # A number inside a word, "네" (yes) without a counter, a
        # Sino-Korean digit alone ("이 곳": this place), Han digits without
        # a counter, enumerators only, and idioms of absence ("no
        # problem", a negated copula).
        assert stated_numbers(text) == set()

    def test_a_dash_after_a_number_joins_a_range(self) -> None:
        assert stated_numbers("1990-1995") == {1990, 1995}
        assert stated_numbers("3-5 vendors") == {3, 5}
        assert stated_numbers("It fell to -5") == {-5}
        assert stated_numbers("A-1 grade") == set()

    def test_a_unicode_minus_is_a_minus(self) -> None:
        # U+2212, as normalize_answer folds it; a dash between two numbers
        # still joins a range.
        assert stated_numbers("It fell to \u22125") == {-5}
        assert stated_numbers("(\u22125)") == {-5}
        assert stated_numbers("1990\u20131995") == {1990, 1995}
        assert stated_numbers("It fell to \u20135") == {-5}

    @pytest.mark.parametrize("dash", ["\u2014", "\u2015"])
    def test_a_dash_never_written_as_a_minus_is_none(self, dash: str) -> None:
        # An em-dash or horizontal bar sets a number off ("3 vendors —2 in
        # Seoul"); it is no minus sign.
        assert stated_numbers(f"3 vendors {dash}2 in Seoul") == {3, 2}
        assert stated_numbers(f"({dash}5)") == {5}


class TestNumberValue:
    """``number_value``: the one number a gold states, else ``None``."""

    @pytest.mark.parametrize(
        ("gold", "value"),
        [
            # One number.
            ("3", 3),
            ("4,486", 4486),
            ("three", 3),
            ("none", 0),
            ("0", 0),
            ("-5", -5),
            ("\u22125", -5),
            # No number.
            ("", None),
            ("Acme", None),
            # Several: the gold is a phrase, not its first or least number.
            ("1990-1995", None),
            ("2 or 3", None),
            ("Jan 5, 2024", None),
            ("3 (as of 2024)", None),
        ],
    )
    def test_the_one_number_a_gold_states(self, gold: str, value: float | None) -> None:
        assert number_value(gold) == value
