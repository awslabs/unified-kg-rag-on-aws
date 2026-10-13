# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the deterministic answer-match evaluator (AWS-free)."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import (
    Config,
    EvaluationQuery,
    EvaluationResult,
    EvaluatorType,
)
from unified_kg_rag.evaluation import AnswerMatchEvaluator, EvaluationManager
from unified_kg_rag.evaluation.answer_match_evaluator import (
    answer_contains,
    exact_match,
    normalize_answer,
    token_f1,
)

pytestmark = pytest.mark.unit


class TestNormalization:
    def test_squad_normalization(self) -> None:
        assert normalize_answer("The  Vendor, Inc.!") == "vendor inc"

    def test_unicode_punctuation_removed(self) -> None:
        assert normalize_answer("「공급사」는…") == "공급사는"

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [
            ("1,000", "1000"),  # punctuation deleted, not replaced by a space
            ("USD 1,000.", "usd 1000"),
            ("\uff11\uff10\uff10\uff10", "1000"),  # fullwidth digits (NFKC)
            ("Vendor\u2019s", "Vendors"),
            ("the U.S.", "US"),
            ("2,000", "2000"),
            ("a 5-year term", "5 year term"),
            ("3.5%", "3.5 %"),
        ],
    )
    def test_squad_equivalences(self, prediction: str, reference: str) -> None:
        assert exact_match(prediction, reference) == 1.0

    @pytest.mark.parametrize(
        ("text", "normalized"),
        [
            ("3.5 days", "3.5 days"),  # decimal point kept: 3.5 is not 35
            ("1,000,000.", "1000000"),  # thousands separators dropped
            ("1,5", "1,5"),  # a decimal comma is kept
            ("C++ and C#!", "c++ and c#"),
            ("#1 vendor", "1 vendor"),
            ("A", "a"),  # only articles: kept rather than emptied
            ("The.", "the"),
        ],
    )
    def test_meaningful_punctuation_kept(self, text: str, normalized: str) -> None:
        assert normalize_answer(text) == normalized

    @pytest.mark.parametrize(
        ("prediction", "reference"),
        [
            ("about 1.5 million users", "15 million"),
            ("The rate is 3.5.", "35 percent"),
            ("Written in C++.", "C"),
            ("The balance is -5.", "5"),
            ("A rate of 3.5% applies.", "3.5 million"),
            ("A rate of 3.5% applies.", "35%"),
            ("From 1990-1995.", "19901995"),
            ("금액은 3억이다.", "3"),
        ],
    )
    def test_different_numbers_and_names_not_contained(
        self, prediction: str, reference: str
    ) -> None:
        assert answer_contains(prediction, reference) == 0.0
        assert exact_match(prediction, reference) == 0.0

    @pytest.mark.parametrize(
        ("prediction", "reference", "expected"),
        [
            ("Yes, the Buyer may terminate.", "yes", 1.0),
            ("No. There is no such clause.", "no", 1.0),
            # the gold appears, but the answer's polarity is the opposite
            ("Yes, although no notice is required.", "no", 0.0),
            # "there is no X" is a determiner, not the answer
            ("There is no clause. Yes, it was signed.", "yes", 1.0),
            ("There is no evidence. So, yes.", "yes", 1.0),
            ("There is no evidence. So, yes.", "no", 0.0),
            ("The contract does not say.", "no", 0.0),
            ("Based on the context, no.", "No", 1.0),
            ("No, the Buyer may not.", "no", 1.0),
            ("**No**", "no", 1.0),
            # a markdown heading before the answer, or an abbreviation in it
            ("## Summary\nYes, it is.", "yes", 1.0),
            ("## Summary\nNo.\nThe clause was removed.", "no", 1.0),
            ("The U.S. law says yes.", "yes", 1.0),
            ("Clause No. 5 says yes.", "yes", 1.0),
            ("Clause No. 5 says yes.", "no", 0.0),
        ],
    )
    def test_yes_no_gold_scored_by_leading_polarity(
        self, prediction: str, reference: str, expected: float
    ) -> None:
        assert answer_contains(prediction, reference) == expected

    @pytest.mark.parametrize(
        ("prediction", "reference", "expected"),
        [
            # the gold word appears, but the answer is the opposite polarity
            ("아니요, 계약서에는 예외 조항은 없습니다.", "예", 0.0),
            ("아니요, 계약서에는 예외 조항은 없습니다.", "아니요", 1.0),
            ("예, 해지할 수 있습니다.", "예", 1.0),
            ("네, 맞습니다.", "예", 1.0),
            ("아니요, 없습니다.", "아니오", 1.0),
            # inside a longer word, or "네" the numeral, it is no answer
            ("예를 들어 계약서에는 그런 조항이 없습니다.", "예", 0.0),
            ("네트워크 장비 공급 계약입니다.", "네", 0.0),
            ("네 가지 경우가 있습니다. 아니요.", "네", 0.0),
            ("아니요라고 할 수 없습니다. 예, 맞습니다.", "아니요", 0.0),
            ("不是，没有这样的条款。", "是", 0.0),
            ("是的，买方可以终止。", "是", 1.0),
            ("这是一个条款。", "否", 0.0),
            ("不是的，没有。", "否", 1.0),
            ("いいえ、ありません。", "はい", 0.0),
            ("はい、あります。", "はい", 1.0),
            ("いいえ、ありません。", "いいえ", 1.0),
        ],
    )
    def test_cjk_yes_no_gold_scored_by_leading_polarity(
        self, prediction: str, reference: str, expected: float
    ) -> None:
        assert answer_contains(prediction, reference) == expected

    def test_korean_number_with_a_unit_is_contained(self) -> None:
        assert answer_contains("계약은 2024년에 체결되었다", "2024") == 1.0
        assert answer_contains("계약 금액은 3억 원이다", "3") == 0.0

    def test_range_is_two_numbers(self) -> None:
        assert token_f1("1990-1995", "1990 1995") == 1.0
        assert answer_contains("It ran from 1990-1995.", "1995") == 1.0

    def test_answer_contains_korean_particle(self) -> None:
        assert (
            answer_contains("공급사의 본사는 서울은 아니고 부산입니다.", "부산") == 1.0
        )
        assert answer_contains("본사는 서울 특별시에 있습니다.", "서울 특별시") == 1.0
        assert answer_contains("본사는 부산에 있습니다.", "서울") == 0.0

    def test_articles_only_as_whole_words(self) -> None:
        assert normalize_answer("Theater an Arena") == "theater arena"

    def test_exact_match_ignores_case_articles_punctuation(self) -> None:
        assert exact_match("the Buyer.", "Buyer") == 1.0
        assert exact_match("Buyer Ltd", "Buyer") == 0.0

    def test_token_f1(self) -> None:
        # pred tokens {vendor, ships, parts}, ref {vendor, ships, tools}
        assert token_f1("Vendor ships parts", "vendor ships tools") == pytest.approx(
            2 / 3
        )
        assert token_f1("unrelated", "vendor") == 0.0
        assert token_f1("", "") == 1.0
        assert token_f1("a", "vendor") == 0.0


def _report(config: Config, answer: str, ground_truth: str, metadata: dict):
    evaluator = AnswerMatchEvaluator(config)
    result = EvaluationResult(
        query_id="q",
        question="?",
        generated_answer=answer,
        ground_truth=ground_truth,
        metadata=metadata,
    )
    return evaluator.evaluate_single(
        EvaluationQuery(query_id="q", question="?"), result, ground_truth
    )


class TestEvaluator:
    def test_max_over_aliases(self, config: Config) -> None:
        report = _report(
            config,
            "Net 30 days",
            "thirty days",
            {"answer_aliases": ["net 30 days", "30 days"]},
        )
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values == {"answer_contains": 1.0, "exact_match": 1.0, "token_f1": 1.0}

    def test_single_string_alias_accepted(self, config: Config) -> None:
        report = _report(config, "USD 500", "", {"answer_aliases": "usd 500"})
        assert {m.metric_type.value: m.value for m in report.metrics} == {
            "answer_contains": 1.0,
            "exact_match": 1.0,
            "token_f1": 1.0,
        }

    def test_no_reference_skipped(self, config: Config) -> None:
        report = _report(config, "anything", "", {})
        assert report.metrics == []
        assert set(report.metadata["skipped_metrics"]) == {
            "answer_contains",
            "exact_match",
            "token_f1",
        }

    def test_article_only_gold_is_scored(self, config: Config) -> None:
        report = _report(config, "A.", "A", {})
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values == {"answer_contains": 1.0, "exact_match": 1.0, "token_f1": 1.0}

    def test_long_answer_contains_gold(self, config: Config) -> None:
        report = _report(
            config, "Payment is due within net 30 days of invoice.", "Net 30 days", {}
        )
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values["answer_contains"] == 1.0
        assert values["exact_match"] == 0.0

    def test_contains_matches_alias(self, config: Config) -> None:
        report = _report(
            config,
            "The Buyer pays USD 1,000.",
            "one thousand",
            {"answer_aliases": ["1000"]},
        )
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values["answer_contains"] == 1.0

    def test_contains_miss(self, config: Config) -> None:
        report = _report(config, "The Vendor ships parts.", "Buyer", {})
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values["answer_contains"] == 0.0

    def test_resolver_maps_type(self) -> None:
        assert (
            EvaluationManager._resolve_evaluator_class(EvaluatorType.ANSWER_MATCH)
            is AnswerMatchEvaluator
        )
