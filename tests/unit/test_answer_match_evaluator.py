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
    ModelRole,
)
from unified_kg_rag.evaluation import (
    AnswerMatchEvaluator,
    EvaluationManager,
    answer_match_evaluator,
)
from unified_kg_rag.evaluation.answer_match_evaluator import (
    answer_contains,
    exact_match,
    first_answer_line,
    normalize_answer,
    token_f1,
)
from unified_kg_rag.shared import EvaluationException

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
            # options named with "/" are not the answer
            ("Yes/No: No", "no", 1.0),
            ("Yes/No: No", "yes", 0.0),
            ("Answer (yes / no): yes", "yes", 1.0),
            # answering punctuation, a dash and the line end still answer
            ("Answer: No", "no", 1.0),
            ("No—the Buyer pays.", "no", 1.0),
            ("1. No, the Buyer pays.", "no", 1.0),
            ("| Warranty? | Yes |", "yes", 1.0),
            ("No warranty is provided.", "no", 0.0),
            # any punctuation after "no" answers, markdown emphasis included
            ("No: the contract does not allow it.", "no", 1.0),
            ("**No**: the clause forbids it.", "no", 1.0),
            ("No (see Clause 5).", "no", 1.0),
            ("No - it does not.", "no", 1.0),
            ("No: Yes it is", "no", 1.0),
            ("No: Yes it is", "yes", 0.0),
            # emphasis or an opening quote after "no" starts the next word
            ("There is no **cap** on fees. Yes, the Vendor may.", "yes", 1.0),
            ("There is **no** penalty clause. Yes, the Buyer may.", "yes", 1.0),
            ('There is no "cap" on fees. Yes, the Vendor may.', "yes", 1.0),
            ("There is no `cap` field. Yes, it is allowed.", "yes", 1.0),
            ("There is no \u201ccap\u201d on fees. Yes.", "yes", 1.0),
            ('He said "no".', "no", 1.0),
            ("\u201cNo\u201d, it is not.", "no", 1.0),
            ("| Question | Answer |\n|---|---|\n| Warranty? | **No** |", "no", 1.0),
            ("Nope, the Buyer pays.", "no", 1.0),
            # a hyphen joins "no" to the next word, the non-breaking one too
            ("No-cap clause. Yes, it applies.", "yes", 1.0),
            ("No\u2011cap clause. Yes, it applies.", "yes", 1.0),
            ("No\u2010cap clause. Yes, it applies.", "yes", 1.0),
            # known limit: other punctuation before a word still reads as "no"
            ("There is no (explicit) cap; yes, it may.", "no", 1.0),
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
            # "예:" / "예)" abbreviate "for example", not "yes"
            ("계약은 자동 갱신되지 않습니다 (예: 만료 시 종료).", "예", 0.0),
            (
                "예: 공급자가 위반한 경우 해지됩니다. 아니요, 갱신은 없습니다.",
                "아니요",
                1.0,
            ),
            ("예) 2024년 계약서. 아니요.", "아니요", 1.0),
            ("질문 (네/아니요): 아니요", "아니요", 1.0),
            ("질문 (네/아니요): 아니요", "네", 0.0),
            ("질문 2 (네/아니요): 네", "네", 1.0),
            # non-misfires: still answers
            ("**네**, 맞습니다.", "예", 1.0),
            ("답변: 예", "예", 1.0),
            ("1. 예, 구매자가 부담합니다.", "예", 1.0),
            ("예외적으로, 아니요.", "아니요", 1.0),
            ("根据合同第3条，是。", "是", 1.0),
            ("合同规定：是", "是", 1.0),
            ("不是。卖方不承担运费。", "否", 1.0),
            ("아닙니다. 상한이 없습니다.", "아니요", 1.0),
            ("아닙니다. 상한이 없습니다.", "예", 0.0),
            ("没有，合同里没有上限。", "否", 1.0),
            ("没有，合同里没有上限。", "是", 0.0),
            # inside a longer word "没有" is no answer ("有没有": whether)
            ("合同有没有上限？是的，有。", "是", 1.0),
            ("否则，买方可以终止合同。", "否", 0.0),
            ("倉庫にはいります。", "はい", 0.0),
            # "아니요"/"아니오" have no second meaning: any punctuation answers
            ("아니요: 계약서에는 없습니다.", "아니요", 1.0),
            ("**아니요**: 없습니다", "아니요", 1.0),
            ("아니오 (계약 제5조).", "아니오", 1.0),
            ("아니요 계약서에는 그런 내용이 없습니다.", "아니요", 1.0),
            ("**네**: 맞습니다.", "네", 1.0),
            # emphasis or an opening quote after the numeral "네" starts a word
            ("계약에는 네 **가지** 조건이 있습니다. 아니요.", "아니요", 1.0),
            ('네 "가지" 조건이 있습니다. 아니요.', "아니요", 1.0),
            ("- **아니요** — 없습니다.", "아니요", 1.0),
            ("「いいえ」、買主は支払いません。", "いいえ", 1.0),
            ("**是的**，卖方承担。", "是", 1.0),
            ("答案：否。", "否", 1.0),
            ("回答:不是。", "否", 1.0),
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


class TestScriptAwareTokenF1:
    """Regression: tokens were whitespace words, so a Chinese or Japanese
    answer was one token (F1 0 unless exact) and a Korean word with a
    particle never matched its gold word."""

    @pytest.mark.parametrize(
        ("prediction", "reference", "expected"),
        [
            # 3 of 5 characters / 3 of 3: P 0.6, R 1.
            ("東京都です。", "東京都", 0.75),
            ("北京市是中国的首都", "北京市", 0.5),
            # "서울은" counts as "서울"; then 2 of 10 syllables.
            ("서울은 한국의 수도입니다.", "서울", 2 * 0.2 / 1.2),
            ("서울은", "서울", 1.0),
            ("2024年に締結", "2024年", 2 * 0.4 / 1.4),
            ("대금은 3억 원입니다", "3억 원", 2 * 0.5 / 1.5),
            ("東京", "大阪", 0.0),
            ("The capital is Tokyo.", "Tokyo", 0.5),
        ],
    )
    def test_dense_scripts_score_by_character(
        self, prediction: str, reference: str, expected: float
    ) -> None:
        assert token_f1(prediction, reference) == pytest.approx(expected)

    def test_the_official_normalizations_keep_whitespace_tokens(
        self, config: Config
    ) -> None:
        assert token_f1("東京都です", "東京都", script_aware=False) == 0.0
        config.evaluation.answer_normalization = "official_squad"
        report = _report(config, "東京都です", "東京都", {})
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values["token_f1"] == 0.0
        assert report.metadata["token_f1_tokens"] == "whitespace"
        config.evaluation.answer_normalization = "default"
        report = _report(config, "東京都です", "東京都", {})
        values = {m.metric_type.value: m.value for m in report.metrics}
        assert values["token_f1"] == pytest.approx(2 * 0.6 / 1.6)
        assert report.metadata["token_f1_tokens"] == "script_aware"


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


# The answer prompt's shape: a one-sentence answer, then its support.
_DEPOT_ANSWER = (
    "## Answer\n\nDepot stores widgets for Vendor.\n"
    "Support: the context links Depot and Vendor through widget flows."
)


def _values(report) -> dict[str, float]:
    return {m.metric_type.value: m.value for m in report.metrics}


class TestAnswerSpan:
    def test_first_line_skips_headings_and_blank_lines(self) -> None:
        assert first_answer_line(_DEPOT_ANSWER) == "Depot stores widgets for Vendor."
        assert first_answer_line("# Only a heading") == "# Only a heading"

    @pytest.mark.parametrize(("span", "f1"), [("full", 1 / 8), ("first_line", 1 / 3)])
    def test_em_and_f1_read_the_configured_span(
        self, config: Config, span: str, f1: float
    ) -> None:
        # Before, EM/F1 always read the whole answer: the supporting
        # sentence diluted F1 to 1/8 for a correct answer.
        config.evaluation.answer_span = span
        report = _report(config, _DEPOT_ANSWER, "Depot", {})
        values = _values(report)
        assert values["token_f1"] == pytest.approx(f1)
        assert values["exact_match"] == 0.0
        # answer_contains always reads the whole answer.
        assert values["answer_contains"] == 1.0
        assert report.metadata["answer_span"] == span

    def test_the_default_is_first_line(self) -> None:
        assert Config().evaluation.answer_span == "first_line"


class _SpanChain:
    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply
        self.calls: list[dict] = []

    async def ainvoke(self, inputs: dict) -> str:
        self.calls.append(inputs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _extracting(mocker, config: Config, reply: str | Exception):
    config.evaluation.answer_span = "extracted"
    chain = _SpanChain(reply)
    setup = mocker.patch.object(
        answer_match_evaluator, "setup_chain", return_value=chain
    )
    mocker.patch.object(answer_match_evaluator.Providers, "resolve")
    evaluator = AnswerMatchEvaluator(config)
    assert setup.call_args.args[1] == config.model_id(ModelRole.ANSWER_SPAN)
    return evaluator, chain


async def _ascore(evaluator, answer: str, gold: str, question: str = "Who?"):
    result = EvaluationResult(
        query_id="q", question=question, generated_answer=answer, ground_truth=gold
    )
    return await evaluator.aevaluate_single(
        EvaluationQuery(query_id="q", question=question), result, gold
    )


class TestExtractedSpan:
    @pytest.mark.parametrize("span", ["first_line", "extracted"])
    async def test_a_decline_scores_zero_without_a_span_call(
        self, mocker, config, span
    ) -> None:
        # The decline names the gold answer; before, answer_contains gave it 1.
        evaluator, chain = _extracting(mocker, config, "Depot")
        evaluator.answer_span = span
        result = EvaluationResult(
            query_id="q",
            question="Who?",
            generated_answer="The context names Depot but not who runs it.",
            ground_truth="Depot",
            abstained=True,
        )
        report = await evaluator.aevaluate_single(
            EvaluationQuery(query_id="q", question="Who?"), result, "Depot"
        )
        assert _values(report) == {
            "answer_contains": 0.0,
            "exact_match": 0.0,
            "token_f1": 0.0,
        }
        assert report.metadata["abstained"] is True
        assert chain.calls == []

    async def test_a_short_span_scores_like_the_papers(self, mocker, config) -> None:
        evaluator, chain = _extracting(mocker, config, '"Depot"\n')
        report = await _ascore(evaluator, _DEPOT_ANSWER, "Depot")
        assert _values(report)["exact_match"] == 1.0
        assert _values(report)["token_f1"] == 1.0
        assert report.metadata["scored_span"] == "Depot"
        assert chain.calls == [{"question": "Who?", "answer": _DEPOT_ANSWER}]

    async def test_spans_are_cached_per_question_and_answer(
        self, mocker, config
    ) -> None:
        evaluator, chain = _extracting(mocker, config, "Depot")
        await _ascore(evaluator, _DEPOT_ANSWER, "Depot")
        await _ascore(evaluator, _DEPOT_ANSWER, "Depot")
        assert len(chain.calls) == 1
        await _ascore(evaluator, _DEPOT_ANSWER, "Depot", question="Where?")
        assert len(chain.calls) == 2

    @pytest.mark.parametrize(
        "reply", [RuntimeError("throttled"), "", "Harbor"]  # Harbor: not in it
    )
    async def test_a_failed_extraction_falls_back_to_the_first_line(
        self, mocker, config, reply
    ) -> None:
        evaluator, _ = _extracting(mocker, config, reply)
        report = await _ascore(evaluator, _DEPOT_ANSWER, "Depot")
        assert _values(report)["token_f1"] == pytest.approx(1 / 3)
        assert report.metadata["scored_span"] == "Depot stores widgets for Vendor."
        assert report.metadata["answer_span_fallback"]

    async def test_no_answer_scores_an_empty_span(self, mocker, config) -> None:
        evaluator, _ = _extracting(mocker, config, "NO_ANSWER")
        report = await _ascore(evaluator, "The context does not say.", "Depot")
        assert _values(report)["token_f1"] == 0.0
        assert report.metadata["scored_span"] == ""
        assert "answer_span_fallback" not in report.metadata

    async def test_a_yes_no_span_need_not_be_verbatim(self, mocker, config) -> None:
        evaluator, _ = _extracting(mocker, config, "no")
        report = await _ascore(evaluator, "The clause does not apply.", "No")
        assert _values(report)["exact_match"] == 1.0

    async def test_no_reference_makes_no_model_call(self, mocker, config) -> None:
        evaluator, chain = _extracting(mocker, config, "Depot")
        report = await _ascore(evaluator, _DEPOT_ANSWER, "")
        assert report.metrics == [] and chain.calls == []

    def test_the_sync_path_refuses_a_model_call(self, mocker, config) -> None:
        evaluator, _ = _extracting(mocker, config, "Depot")
        with pytest.raises(EvaluationException, match="aevaluate_single"):
            evaluator.evaluate_single(
                EvaluationQuery(query_id="q", question="?"),
                EvaluationResult(
                    query_id="q", question="?", generated_answer="x", ground_truth=""
                ),
                "Depot",
            )

    def test_other_spans_build_no_model(self, mocker, config) -> None:
        setup = mocker.patch.object(answer_match_evaluator, "setup_chain")
        AnswerMatchEvaluator(config)
        setup.assert_not_called()


class TestHotpotQARule:
    @pytest.mark.parametrize(
        ("mode", "f1"), [("official_squad", 0.5), ("hotpotqa", 0.0)]
    )
    def test_yes_no_gets_no_partial_credit(
        self, config: Config, mode: str, f1: float
    ) -> None:
        config.evaluation.answer_normalization = mode
        report = _report(config, "No, it is", "no", {})
        assert _values(report)["token_f1"] == pytest.approx(f1)

    def test_equal_yes_no_still_scores(self, config: Config) -> None:
        config.evaluation.answer_normalization = "hotpotqa"
        assert _values(_report(config, "Yes.", "yes", {}))["token_f1"] == 1.0
        # Other golds keep partial credit, with the official normalization.
        report = _report(config, "Depot and Mill", "Depot", {})
        assert _values(report)["token_f1"] == pytest.approx(0.5)


def test_answer_contains_reads_through_a_possessive() -> None:
    assert answer_contains("Depot's crew stores the widgets.", "Depot") == 1.0
    assert answer_contains("Depots store the widgets.", "Depot") == 0.0
