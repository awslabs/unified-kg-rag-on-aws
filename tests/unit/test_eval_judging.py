# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Length-controlled pairwise judging, the assertion evaluator and the
closed-book arm, with fake models (no AWS)."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from unified_kg_rag.adapters.evaluators.base import (
    FAILED_METRICS_KEY,
    SKIPPED_METRICS_KEY,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.cli import run_evaluation
from unified_kg_rag.domain.models import (
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
    ModelRole,
)
from unified_kg_rag.domain.prompts import (
    AssertionJudgementPrompt,
    PairwiseDifferentialOverallPrompt,
    PairwiseDifferentialPrompt,
    PairwiseOverlapPrompt,
)
from unified_kg_rag.evaluation import assertion_evaluator, pairwise
from unified_kg_rag.evaluation.closed_book import CLOSED_BOOK_ARM, closed_book_chain
from unified_kg_rag.evaluation.comparison import paired_comparison
from unified_kg_rag.evaluation.evaluation_manager import EvaluationManager
from unified_kg_rag.evaluation.gates import gate_breaches, parse_gate
from unified_kg_rag.evaluation.report import comparison_markdown
from unified_kg_rag.shared import EvaluationException

pytestmark = pytest.mark.unit


def _result(query_id: str, answer: str, error: bool = False) -> EvaluationResult:
    return EvaluationResult(
        query_id=query_id,
        question=f"question {query_id}?",
        generated_answer=answer,
        ground_truth="",
        error=error,
    )


# --- Length buckets -------------------------------------------------------


class TestLengthBuckets:
    @pytest.mark.parametrize(
        ("ratio", "bucket"),
        [
            (0.5, "<0.67x"),
            (0.8, "0.67-0.9x"),
            (0.9, "0.9-1.1x"),
            (1.0, "0.9-1.1x"),
            (1.1, "0.9-1.1x"),
            (1.2, "1.1-1.5x"),
            (1.5, ">1.5x"),
            (9.0, ">1.5x"),
        ],
    )
    def test_ratio_buckets(self, ratio: float, bucket: str) -> None:
        assert pairwise.length_bucket(ratio) == bucket

    @pytest.mark.parametrize(
        ("short", "long"),
        [
            (
                "서울입니다.",
                "수도는 서울이며 한강을 따라 넓게 자리하고 인구가 가장 많습니다.",
            ),
            (
                "東京です。",
                "首都は東京で、人口が最も多く、経済と文化の中心でもあります。",
            ),
            ("北京。", "首都是北京，它是政治和文化中心，人口众多，历史悠久。"),
        ],
        ids=["ko", "ja", "zh"],
    )
    def test_lengths_are_estimated_tokens_not_whitespace_words(
        self, short: str, long: str
    ) -> None:
        # In whitespace words the Japanese and Chinese pairs are 1 vs 1: the
        # same "length". The script-aware estimate sees the longer answer.
        lengths = pairwise.answer_lengths(short, long)
        assert lengths["candidate_tokens"] > 2 * lengths["baseline_tokens"]
        assert lengths["bucket"] == ">1.5x"
        assert 0 < lengths["relative_difference"] < 1

    def test_summary_reports_win_rates_per_bucket_and_correlation(self) -> None:
        lengths = {
            "a": pairwise.answer_lengths("x " * 10, "x " * 30),  # longer
            "b": pairwise.answer_lengths("x " * 10, "x " * 30),
            "c": pairwise.answer_lengths("x " * 30, "x " * 10),  # shorter
            "d": pairwise.answer_lengths("x " * 10, "x " * 10),  # similar
        }
        outcomes = {
            "a": {"comprehensiveness": "candidate"},
            "b": {"comprehensiveness": "candidate"},
            "c": {"comprehensiveness": "baseline"},
            "d": {"comprehensiveness": "tie"},
        }
        summary = pairwise.length_summary(lengths, outcomes, ["comprehensiveness"])
        buckets = summary["buckets"]
        assert buckets[">1.5x"]["pairs"] == 2
        assert buckets[">1.5x"]["criteria"]["comprehensiveness"][
            "candidate_win_rate"
        ] == pytest.approx(1.0)
        assert buckets["<0.67x"]["criteria"]["comprehensiveness"][
            "candidate_win_rate"
        ] == pytest.approx(0.0)
        assert buckets["0.9-1.1x"]["criteria"]["comprehensiveness"][
            "candidate_win_rate"
        ] == pytest.approx(0.5)
        assert buckets["1.1-1.5x"]["pairs"] == 0
        # The judge picks the longer answer: a strong positive correlation.
        assert summary["score_correlation"]["comprehensiveness"] > 0.9
        assert summary["unit"] == "estimated_tokens"


# --- Pairwise judge ---------------------------------------------------------


class _Judge:
    """Prefers the longer answer (standard) or the longer unique part."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(inputs)
        first = inputs.get("answer1", inputs.get("unique1", ""))
        second = inputs.get("answer2", inputs.get("unique2", ""))
        winner = "1" if len(first) > len(second) else "2"
        return {c: {"winner": winner} for c in pairwise.CRITERIA}


class _Overlap:
    def __init__(self, reply: Any) -> None:
        self.reply = reply
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, inputs: dict[str, Any]) -> Any:
        self.calls.append(inputs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply(inputs) if callable(self.reply) else self.reply


def _differential_judge(mocker, overlap: _Overlap) -> pairwise.PairwiseJudge:
    judge = _Judge()
    setup = mocker.patch.object(pairwise, "setup_chain", side_effect=[judge, overlap])
    mocker.patch.object(pairwise.Providers, "resolve")
    config = Config()
    config.evaluation.pairwise_mode = "differential"
    built = pairwise.PairwiseJudge(config)
    assert setup.call_args_list[0].args[2] is PairwiseDifferentialPrompt
    assert setup.call_args_list[1].args[2] is PairwiseOverlapPrompt
    assert setup.call_args_list[1].args[1] == config.model_id(
        ModelRole.PAIRWISE_OVERLAP
    )
    return built


class TestPairwiseLength:
    def test_standard_mode_reports_length_buckets(self, mocker) -> None:
        mocker.patch.object(pairwise, "setup_chain", return_value=_Judge())
        mocker.patch.object(pairwise.Providers, "resolve")
        judge = pairwise.PairwiseJudge(Config())
        out = asyncio.run(
            judge.acompare(
                [_result("a", "short"), _result("b", "a much longer answer here")],
                [_result("a", "a much longer answer here"), _result("b", "short")],
            )
        )
        assert out["mode"] == "standard"
        length = out["length"]
        assert length["pairs"] == 2
        assert length["buckets"][">1.5x"]["criteria"]["comprehensiveness"][
            "candidate_win_rate"
        ] == pytest.approx(1.0)
        assert length["buckets"]["<0.67x"]["criteria"]["comprehensiveness"][
            "candidate_win_rate"
        ] == pytest.approx(0.0)
        assert set(length["per_query"]) == {"a", "b"}

    def test_a_pair_decided_by_a_failed_answer_has_no_length(self, mocker) -> None:
        mocker.patch.object(pairwise, "setup_chain", return_value=_Judge())
        mocker.patch.object(pairwise.Providers, "resolve")
        judge = pairwise.PairwiseJudge(Config())
        out = asyncio.run(
            judge.acompare(
                [_result("a", "x"), _result("b", "y")],
                [_result("a", "longer x"), _result("b", "", error=True)],
            )
        )
        assert out["candidate_errors"] == 1
        assert set(out["length"]["per_query"]) == {"a"}


class TestDifferentialJudge:
    def test_the_judge_sees_only_unique_statements_in_both_orders(self, mocker) -> None:
        overlap = _Overlap(
            {
                "shared": ["Vendor supplies widgets."],
                "unique_1": ["Delivery is weekly."],
                "unique_2": ["Penalty is 100 credits per day.", "Buyer may cancel."],
            }
        )
        judge = _differential_judge(mocker, overlap)
        out = asyncio.run(
            judge.acompare(
                [_result("a", "Vendor supplies widgets. Delivery is weekly.")],
                [_result("a", "Vendor supplies widgets. Penalty ... cancel.")],
            )
        )
        calls = judge.chain.calls
        assert len(calls) == 2 and len(overlap.calls) == 1
        # The shared statement is common ground in both orders; the unique
        # parts swap places.
        assert all("Vendor supplies widgets." in c["shared"] for c in calls)
        assert "Delivery is weekly." in calls[0]["unique1"]
        assert "Delivery is weekly." in calls[1]["unique2"]
        assert all("answer1" not in c for c in calls)
        assert out["mode"] == "differential"
        assert out["per_query"]["a"]["comprehensiveness"] == "candidate"
        assert out["position_consistency"] == 1.0
        assert out["identical_pairs"] == 0
        split = out["differential"]["per_query"]["a"]
        assert split["shared"] == ["Vendor supplies widgets."]
        assert out["length"]["per_query"]["a"]["candidate_unique_tokens"] > 0

    def test_no_unique_statement_is_a_tie_without_a_judge_call(self, mocker) -> None:
        overlap = _Overlap({"shared": ["Same."], "unique_1": [], "unique_2": [" "]})
        judge = _differential_judge(mocker, overlap)
        out = asyncio.run(
            judge.acompare([_result("a", "Same.")], [_result("a", "Same!")])
        )
        assert judge.chain.calls == []
        assert out["identical_pairs"] == 1
        assert out["judged"] == 1
        assert set(out["per_query"]["a"].values()) == {"tie"}
        # No judge call, so no order agreement to report.
        assert out["position_consistency"] is None
        assert out["criteria"]["comprehensiveness"]["candidate_win_rate"] == 0.5

    @pytest.mark.parametrize(
        "reply",
        [
            {"shared": [], "unique_1": ["x"]},  # a missing key
            {"shared": "x", "unique_1": [], "unique_2": []},  # not a list
            RuntimeError("throttled"),
        ],
        ids=["missing", "not-a-list", "call-failed"],
    )
    def test_a_failed_split_fails_the_pair_not_a_tie(self, mocker, reply) -> None:
        judge = _differential_judge(mocker, _Overlap(reply))
        out = asyncio.run(judge.acompare([_result("a", "x")], [_result("a", "y")]))
        assert (out["failed"], out["judged"]) == (1, 0)
        assert out["criteria"]["comprehensiveness"]["candidate_win_rate"] is None

    def test_overall_uses_the_differential_overall_prompt(self, mocker) -> None:
        setup = mocker.patch.object(pairwise, "setup_chain", return_value=_Judge())
        mocker.patch.object(pairwise.Providers, "resolve")
        config = Config()
        config.evaluation.pairwise_mode = "differential"
        config.evaluation.pairwise_overall = True
        judge = pairwise.PairwiseJudge(config)
        assert setup.call_args_list[0].args[2] is PairwiseDifferentialOverallPrompt
        assert judge.criteria[-1] == "overall"

    @pytest.mark.parametrize(
        ("one", "two"),
        [
            ("서울은 수도다. 인구가 많다.", "서울은 수도다. 한강이 흐른다."),
            ("東京は首都です。人口が多い。", "東京は首都です。富士山が見える。"),
        ],
        ids=["ko", "ja"],
    )
    def test_statements_stay_in_the_answers_language(
        self, mocker, one: str, two: str
    ) -> None:
        def sentences(text: str) -> list[str]:
            return [s for s in re.split(r"(?<=[.。])\s*", text) if s]

        def split(inputs: dict[str, Any]) -> dict[str, Any]:
            first, second = sentences(inputs["answer1"]), sentences(inputs["answer2"])
            return {
                "shared": [s for s in first if s in second],
                "unique_1": [s for s in first if s not in second],
                "unique_2": [s for s in second if s not in first],
            }

        judge = _differential_judge(mocker, _Overlap(split))
        out = asyncio.run(judge.acompare([_result("a", one)], [_result("a", two)]))
        assert out["judged"] == 1
        assert judge.chain.calls[0]["shared"] == f"- {sentences(one)[0]}"
        assert sentences(two)[1] in judge.chain.calls[0]["unique2"]
        prompt = PairwiseOverlapPrompt.resolve().system_prompt_template
        assert "never translate" in prompt


# --- Assertion evaluator -----------------------------------------------------


class _AssertionJudge:
    """Labels each shown assertion by a rule on its text."""

    def __init__(self, rule, fail_trials: set[int] | None = None) -> None:
        self.rule = rule
        self.fail_trials = fail_trials or set()
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, inputs: dict[str, Any]) -> Any:
        trial = len(self.calls)
        self.calls.append(inputs)
        if trial in self.fail_trials:
            raise RuntimeError("judge timeout")
        verdicts = []
        for line in inputs["assertions"].splitlines():
            number, text = line.split(". ", 1)
            verdicts.append({"id": int(number), "verdict": self.rule(text, trial)})
        return {"verdicts": verdicts}


def _evaluator(
    mocker, judge, trials: int = 3
) -> assertion_evaluator.AssertionEvaluator:
    mocker.patch.object(assertion_evaluator, "setup_chain", return_value=judge)
    mocker.patch.object(assertion_evaluator.Providers, "resolve")
    config = Config()
    config.evaluation.assertion_trials = trials
    return assertion_evaluator.AssertionEvaluator(config)


def _scored(result: EvaluationResult, assertions: list[Any]) -> EvaluationResult:
    result.metadata["assertions"] = assertions
    return result


def _score(evaluator, result) -> EvaluationReport:
    query = EvaluationQuery(query_id=result.query_id, question=result.question)
    return asyncio.run(evaluator.aevaluate_single(query, result, ""))


def _metrics(report: EvaluationReport) -> dict[str, float]:
    return {m.metric_type.value: m.value for m in report.metrics}


class TestAssertionParsing:
    def test_verdicts_need_every_id_once_with_a_known_label(self) -> None:
        good = {
            "verdicts": [
                {"id": "2", "verdict": "Absent"},
                {"id": 1, "verdict": "supported"},
            ]
        }
        assert assertion_evaluator.parse_verdicts(good, 2) == ["supported", "absent"]
        for bad, match in [
            ({"verdicts": [{"id": 1, "verdict": "supported"}]}, "no label"),
            (
                {
                    "verdicts": [
                        {"id": 1, "verdict": "maybe"},
                        {"id": 2, "verdict": "absent"},
                    ]
                },
                "label",
            ),
            ({"verdicts": [{"id": 3, "verdict": "absent"}]}, "not 1..2"),
            ({"verdicts": [{"id": 1, "verdict": "absent"}] * 2}, "twice"),
            ({"answer": "yes"}, "no 'verdicts'"),
        ]:
            with pytest.raises(EvaluationException, match=match):
                assertion_evaluator.parse_verdicts(bad, 2)

    def test_trial_orders_are_seeded_permutations(self) -> None:
        order = assertion_evaluator.trial_order("q1", 0, 6)
        assert sorted(order) == list(range(6))
        assert order == assertion_evaluator.trial_order("q1", 0, 6)
        others = {tuple(assertion_evaluator.trial_order("q1", t, 6)) for t in range(5)}
        assert len(others) > 1

    def test_a_label_needs_more_than_half_the_trials(self) -> None:
        kinds = ["positive", "positive", "negative"]
        trials = [
            ["supported", "contradicted", "supported"],
            ["supported", "absent", "absent"],
        ]
        out = assertion_evaluator.aggregate_trials(kinds, trials)
        # 2/2 supported -> passes; 1/2 contradicted is not more than half.
        assert out["assertion_recall"] == pytest.approx(0.5)
        assert out["assertion_contradiction_rate"] == 0.0
        assert out["negative_assertion_violation_rate"] == 0.0
        out = assertion_evaluator.aggregate_trials(["positive"], [["absent"]] * 3)
        assert out["negative_assertion_violation_rate"] is None

    def test_items_carry_strings_or_objects_and_refuse_bad_ones(
        self, tmp_path: Path
    ) -> None:
        data = tmp_path / "eval.json"
        data.write_text(
            json.dumps(
                [
                    {
                        "id": "q1",
                        "question": "벤더는 무엇을 공급합니까?",
                        "assertions": [
                            "벤더는 위젯을 공급한다",
                            {"text": "벤더는 무료로 공급한다", "kind": "negative"},
                        ],
                    }
                ]
            ),
            encoding="utf-8",
        )
        _, truths = EvaluationManager.load_data(data)
        assert [(a.text, a.kind) for a in truths[0].assertions] == [
            ("벤더는 위젯을 공급한다", "positive"),
            ("벤더는 무료로 공급한다", "negative"),
        ]
        for bad in ([{"text": " "}], [{"text": "x", "kind": "maybe"}], "x", [3]):
            data.write_text(json.dumps([{"question": "q", "assertions": bad}]))
            with pytest.raises(EvaluationException, match="'assertions'"):
                EvaluationManager.load_data(data)


class TestAssertionEvaluator:
    def test_recall_contradiction_and_violation_per_query(self, mocker) -> None:
        def rule(text: str, trial: int) -> str:
            if "weekly" in text:
                return "contradicted"
            if "free" in text:
                return "supported"  # the answer states the negative one
            return "supported" if "widgets" in text else "absent"

        judge = _AssertionJudge(rule)
        evaluator = _evaluator(mocker, judge)
        result = _scored(
            _result("q1", "Vendor supplies widgets for free, monthly."),
            [
                "Vendor supplies widgets.",
                "Delivery is weekly.",
                "Buyer pays in advance.",
                {"text": "Supply is free.", "kind": "negative"},
            ],
        )
        report = _score(evaluator, result)
        assert _metrics(report) == {
            "assertion_recall": pytest.approx(1 / 3),
            "assertion_contradiction_rate": pytest.approx(1 / 3),
            "negative_assertion_violation_rate": 1.0,
        }
        assert len(judge.calls) == 3
        assert report.metadata["failed_trials"] == 0
        # Every assertion keeps its own label whatever order it was shown in.
        labels = {
            entry["text"]: max(entry["shares"], key=entry["shares"].get)
            for entry in report.metadata["assertion_labels"]
        }
        assert labels == {
            "Vendor supplies widgets.": "supported",
            "Delivery is weekly.": "contradicted",
            "Buyer pays in advance.": "absent",
            "Supply is free.": "supported",
        }
        # Each trial shows its seeded order; the labels map back (else the
        # text-keyed rule above would score other assertions).
        texts = [
            a["text"] if isinstance(a, dict) else a
            for a in result.metadata["assertions"]
        ]
        for trial, call in enumerate(judge.calls):
            order = assertion_evaluator.trial_order("q1", trial, len(texts))
            assert call["assertions"].splitlines() == [
                f"{n}. {texts[i]}" for n, i in enumerate(order, start=1)
            ]
        assert report.evaluator_type is EvaluatorType.ASSERTION

    def test_labels_map_back_from_the_shuffled_order(self, mocker) -> None:
        verdicts = {"widgets": "supported", "weekly": "contradicted", "cash": "absent"}

        def rule(text: str, trial: int) -> str:
            return next(v for k, v in verdicts.items() if k in text)

        texts = ["About widgets.", "About weekly.", "About cash."]
        order = assertion_evaluator.trial_order("q-map", 0, 3)
        assert order != [0, 1, 2]  # the precondition: a shuffled order
        evaluator = _evaluator(mocker, _AssertionJudge(rule), trials=1)
        report = _score(evaluator, _scored(_result("q-map", "a"), texts))
        assert [
            max(e["shares"], key=e["shares"].get)
            for e in report.metadata["assertion_labels"]
        ] == ["supported", "contradicted", "absent"]

    def test_failed_trials_are_left_out_and_all_failed_is_failed_not_zero(
        self, mocker
    ) -> None:
        evaluator = _evaluator(
            mocker, _AssertionJudge(lambda t, n: "supported", fail_trials={0})
        )
        report = _score(evaluator, _scored(_result("q", "a"), ["x", "y"]))
        assert _metrics(report)["assertion_recall"] == 1.0
        assert report.metadata["failed_trials"] == 1
        evaluator = _evaluator(
            mocker, _AssertionJudge(lambda t, n: "supported", fail_trials={0, 1, 2})
        )
        report = _score(evaluator, _scored(_result("q", "a"), ["x"]))
        assert report.metrics == []
        failed = report.metadata[FAILED_METRICS_KEY]
        assert set(failed) == {"assertion_recall", "assertion_contradiction_rate"}
        assert report.metadata[SKIPPED_METRICS_KEY] == {
            "negative_assertion_violation_rate": "no_negative_assertions"
        }

    def test_a_malformed_reply_fails_its_trial(self, mocker) -> None:
        class _Partial(_AssertionJudge):
            async def ainvoke(self, inputs):
                self.calls.append(inputs)
                return {"verdicts": [{"id": 1, "verdict": "supported"}]}

        evaluator = _evaluator(mocker, _Partial(None), trials=2)
        report = _score(evaluator, _scored(_result("q", "a"), ["x", "y"]))
        assert report.metrics == []
        assert "no label" in report.metadata[FAILED_METRICS_KEY]["assertion_recall"]

    def test_an_item_without_assertions_makes_no_call(self, mocker) -> None:
        judge = _AssertionJudge(lambda t, n: "supported")
        report = _score(_evaluator(mocker, judge), _result("q", "a"))
        assert judge.calls == []
        assert set(report.metadata[SKIPPED_METRICS_KEY].values()) == {"no_assertions"}

    @pytest.mark.parametrize(
        ("answer", "assertion"),
        [
            ("벤더는 위젯을 공급한다.", "벤더는 위젯을 공급한다"),
            ("ベンダーはウィジェットを供給する。", "ベンダーはウィジェットを供給する"),
            ("供应商供应小部件。", "供应商供应小部件"),
        ],
        ids=["ko", "ja", "zh"],
    )
    def test_cjk_items_are_judged_and_parsed_alike(
        self, mocker, answer: str, assertion: str
    ) -> None:
        judge = _AssertionJudge(lambda text, n: "supported")
        report = _score(
            _evaluator(mocker, judge), _scored(_result("q", answer), [assertion])
        )
        assert _metrics(report)["assertion_recall"] == 1.0
        assert assertion in judge.calls[0]["assertions"]
        assert judge.calls[0]["answer"] == answer

    def test_the_manager_resolves_and_threads_the_assertions(
        self, mocker, tmp_path: Path
    ) -> None:
        assert (
            EvaluationManager._resolve_evaluator_class(EvaluatorType.ASSERTION)
            is assertion_evaluator.AssertionEvaluator
        )
        judge = _AssertionJudge(
            lambda t, n: "supported" if "widgets" in t else "absent"
        )
        mocker.patch.object(assertion_evaluator, "setup_chain", return_value=judge)
        config = Config()
        config.evaluation.enabled_evaluators = [EvaluatorType.ASSERTION]
        config.evaluation.assertion_trials = 1

        class _Chain:
            providers = Providers(
                config,
                llm_factory=ScriptedLLMFactory(lambda s, h: ""),
                embedding_factory=HashingEmbeddingFactory(),
            )

            async def ainvoke(self, inputs, config=None):
                meta = {"search_strategy": "local", "processing_time": 0.1}
                return {"answer": "widgets", "sources": [], "metadata": meta}

        data = tmp_path / "eval.json"
        data.write_text(
            json.dumps(
                [{"id": "q1", "question": "What?", "assertions": ["widgets", "price"]}]
            )
        )
        manager = EvaluationManager(config, rag_chain=_Chain())  # type: ignore[arg-type]
        queries, truths = manager.load_data(data)
        _, reports, summary = asyncio.run(manager.evaluate_dataset(queries, truths))
        assert summary.metric_statistics["assertion_recall"]["mean"] == 0.5
        assert summary.metric_outcomes["assertion"]["assertion_recall"]["scored"] == 1
        manifest = summary.run_manifest
        assert manifest["assertion_scoring"] == {"trials": 1}
        assert manifest["models"]["evaluation_judge"] is not None


# --- Comparison and gates ----------------------------------------------------


def _report(query_id: str, **metrics: float) -> EvaluationReport:
    return EvaluationReport(
        query_id=query_id,
        evaluator_type=EvaluatorType.ASSERTION,
        metrics=[
            EvaluationMetric(metric_type=EvaluationMetricType(k), value=v)
            for k, v in metrics.items()
        ],
    )


class TestAssertionComparison:
    def test_a_failed_answer_is_zero_recall_but_unpaired_on_the_rates(self) -> None:
        base = (
            [
                _report(
                    "a", assertion_recall=1.0, negative_assertion_violation_rate=0.5
                ),
                _report(
                    "b", assertion_recall=0.5, negative_assertion_violation_rate=0.5
                ),
            ],
            [_result("a", "x"), _result("b", "y")],
        )
        other = (
            [_report("a", assertion_recall=1.0, negative_assertion_violation_rate=1.0)],
            [_result("a", "x"), _result("b", "", error=True)],
        )
        out = paired_comparison({"local": [base], "global": [other]})
        deltas = out["deltas_vs_baseline"]["global"]
        recall = deltas["assertion/assertion_recall"]
        # The failed answer counts 0 on recall ...
        assert recall["pairs"] == 2 and recall["mean_delta"] == pytest.approx(-0.25)
        # ... but a crash is no perfect violation rate of 0: unpaired there.
        violation = deltas["assertion/negative_assertion_violation_rate"]
        assert (violation["pairs"], violation["unpaired"]) == (1, 1)
        assert violation["mean_delta"] == pytest.approx(0.5)

    def test_gates_read_assertion_recall_and_skip_the_closed_book_arm(self) -> None:
        reports = [_report(q, assertion_recall=1.0) for q in "abcdefghijkl"]
        results = [_result(q, "x") for q in "abcdefghijkl"]
        worse = [_report(q, assertion_recall=0.0) for q in "abcdefghijkl"]
        out = paired_comparison(
            {
                "local": [(reports, results)],
                "global": [(worse, results)],
                CLOSED_BOOK_ARM: [(worse, results)],
            }
        )
        out["closed_book_arms"] = [CLOSED_BOOK_ARM]
        gate = parse_gate("assertion/assertion_recall:ci_high<-0.1")
        breaches = gate_breaches(out, [gate])
        assert len(breaches) == 1 and breaches[0].startswith("global:")


# --- Closed-book arm --------------------------------------------------------


def _closed_book_providers(config: Config, seen: list[tuple[str, str]]) -> Providers:
    def respond(system: str, human: str) -> str:
        seen.append((system, human))
        return "Vendor.\nFrom general knowledge."

    return Providers(
        config,
        llm_factory=ScriptedLLMFactory(respond),
        embedding_factory=HashingEmbeddingFactory(),
    )


class TestClosedBook:
    def test_the_chain_answers_without_context_and_marks_it(self) -> None:
        config = Config()
        seen: list[tuple[str, str]] = []
        chain = closed_book_chain(config, _closed_book_providers(config, seen))
        out = asyncio.run(chain.ainvoke({"query": "누가 공급합니까?", "suffix": "x"}))
        assert out["answer"].startswith("Vendor.")
        assert out["sources"] == []
        meta = out["metadata"]
        assert (meta["search_strategy"], meta["retrieval"], meta["closed_book"]) == (
            CLOSED_BOOK_ARM,
            "none",
            True,
        )
        system, human = seen[0]
        assert "No" in system and "documents" in system
        assert "누가 공급합니까?" in human and "<context>" not in human

    def test_run_eval_adds_the_arm_and_marks_it_in_the_report(
        self, mocker, tmp_path: Path, capsys
    ) -> None:
        config = Config()
        config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
        mocker.patch.object(run_evaluation, "display_ascii_art")
        seen: list[tuple[str, str]] = []
        providers = _closed_book_providers(config, seen)
        rag_calls: list[dict[str, Any]] = []

        class _Chain:
            def __init__(self) -> None:
                self.providers = providers

            async def ainvoke(self, inputs, config=None):
                rag_calls.append(inputs)
                meta = {"search_strategy": "local", "processing_time": 0.1}
                return {"answer": "Vendor", "sources": [], "metadata": meta}

        data = tmp_path / "eval.json"
        data.write_text(
            json.dumps(
                [
                    {"id": "q1", "question": "Who ships?", "answer": "Vendor"},
                    {"id": "q2", "question": "誰が出荷しますか?", "answer": "Vendor"},
                ]
            )
        )
        out_dir = tmp_path / "out"
        args = run_evaluation.CommandLineInterface._setup_arguments().parse_args(
            [
                "--data-path",
                str(data),
                "--search-strategy",
                "local",
                "--closed-book",
                "--output-directory",
                str(out_dir),
                "--output-format",
                "json",
            ]
        )
        runner = run_evaluation.EvaluationRunner(args, _Chain(), config)  # type: ignore[arg-type]
        assert asyncio.run(runner.run()) == 0
        printed = json.loads(capsys.readouterr().out)
        comparison = printed["comparison"]
        assert list(comparison["strategies"]) == ["local", CLOSED_BOOK_ARM]
        assert comparison["closed_book_arms"] == [CLOSED_BOOK_ARM]
        # The RAG chain answered only the strategy's run (plus its warm-up);
        # the closed-book arm called the answer model once per question.
        assert len(seen) == 2
        assert all(c.get("search_strategy") == "local" for c in rag_calls)
        saved = list(out_dir.glob(f"evaluation_results_{CLOSED_BOOK_ARM}_*.json"))
        assert len(saved) == 1
        results = json.loads(saved[0].read_text())
        assert {r["metadata"]["retrieval"] for r in results} == {"none"}
        assert {r["search_strategy"] for r in results} == {CLOSED_BOOK_ARM}
        markdown = next(out_dir.glob("evaluation_comparison_*.md")).read_text()
        assert f"{CLOSED_BOOK_ARM} (no retrieval)" in markdown
        assert "answered without retrieval" in markdown

    def test_closed_book_with_saved_results_is_a_usage_error(self, tmp_path) -> None:
        args = run_evaluation.CommandLineInterface._setup_arguments().parse_args(
            ["--from-results", str(tmp_path), "--closed-book"]
        )
        with pytest.raises(SystemExit) as exit_info:
            run_evaluation.EvaluationRunner(args, None, Config())
        assert exit_info.value.code == 2


def test_the_markdown_report_has_the_length_table() -> None:
    comparison = {
        "baseline": "local",
        "baselines": {"global": "local"},
        "strategies": {"local": {"metrics": {}}, "global": {"metrics": {}}},
        "deltas_vs_baseline": {"global": {}},
        "pairwise_vs_baseline": {
            "global": {
                "mode": "differential",
                "judged": 2,
                "failed": 0,
                "position_consistency": 1.0,
                "criteria": {"comprehensiveness": {"candidate_win_rate": 0.75}},
                "length": pairwise.length_summary(
                    {
                        "a": pairwise.answer_lengths("x", "x x x x"),
                        "b": pairwise.answer_lengths("x x", "x x"),
                    },
                    {
                        "a": {"comprehensiveness": "candidate"},
                        "b": {"comprehensiveness": "tie"},
                    },
                    ["comprehensiveness"],
                ),
            }
        },
    }
    markdown = comparison_markdown(comparison)
    assert "differential mode" in markdown
    assert "Win rate by answer length" in markdown
    assert "| global vs local | >1.5x | 1 | 1.00 |" in markdown
    assert "| global vs local | 0.9-1.1x | 1 | 0.50 |" in markdown


def test_the_judge_prompts_declare_their_variables() -> None:
    for prompt in (
        PairwiseOverlapPrompt,
        PairwiseDifferentialPrompt,
        PairwiseDifferentialOverallPrompt,
        AssertionJudgementPrompt,
    ):
        resolved = prompt.resolve()
        for name in prompt.input_variables:
            assert "{" + name + "}" in (
                resolved.system_prompt_template + resolved.human_prompt_template
            )
