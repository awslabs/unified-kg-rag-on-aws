# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LangChainEvaluator (AWS-free).

The evaluator builds a Bedrock LLM and LangChain ``load_evaluator`` chains in
``_initialize_evaluator``. Both are patched out so no AWS/network call happens;
the real score/JSON parsing, rubric mapping, metric construction and error
handling are exercised against deterministic fake evaluators.
"""

from __future__ import annotations

import json

import pytest

import unified_kg_rag.evaluation  # noqa: F401  (resolves package import cycle)
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.evaluators import langchain_evaluator as lc_module
from unified_kg_rag.adapters.evaluators.langchain_evaluator import LangChainEvaluator
from unified_kg_rag.domain.models import (
    Config,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationResult,
    EvaluatorType,
)

pytestmark = pytest.mark.unit


class _FakeEvaluator:
    """Stands in for a LangChain evaluator object: returns a canned dict."""

    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.last_kwargs: dict | None = None

    def evaluate_strings(self, **kwargs):
        self.last_kwargs = kwargs
        return self.payload

    async def aevaluate_strings(self, **kwargs):
        self.last_kwargs = kwargs
        return self.payload


def _make_evaluator(mocker, *, langchain_metrics=None) -> LangChainEvaluator:
    """Build a LangChainEvaluator with Bedrock + load_evaluator stubbed."""
    mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    mocker.patch.object(lc_module, "load_evaluator", return_value=object())

    config = Config()
    if langchain_metrics is not None:
        config.evaluation.langchain_metrics = langchain_metrics
    return LangChainEvaluator(config=config)


def _query() -> EvaluationQuery:
    return EvaluationQuery(query_id="q1", question="Who founded Acme?")


def _result(answer: str = "Alice founded Acme.") -> EvaluationResult:
    return EvaluationResult(
        query_id="q1",
        question="Who founded Acme?",
        generated_answer=answer,
        ground_truth="",
    )


_PC = EvaluationMetricType.PARTIAL_CORRECTNESS
_C = EvaluationMetricType.CORRECTNESS


class TestJudgeScore:
    def test_partial_json_score_beats_langchain_word_heuristic(self) -> None:
        # LangChain's QA parser sees the last word "correct" and sets score=1;
        # the JSON score the custom prompt asked for is the real one.
        raw = '{"score": 0.2, "reasoning": "Only the city name is correct"}'
        result = {"reasoning": raw, "value": "CORRECT", "score": 1}
        assert LangChainEvaluator._judge_score(_PC, result) == (
            0.2,
            "Only the city name is correct",
        )

    def test_partial_fenced_json(self) -> None:
        fenced = '```json\n{"score": 0.5, "reasoning": "fenced"}\n```'
        assert LangChainEvaluator._judge_score(_PC, {"reasoning": fenced}) == (
            0.5,
            "fenced",
        )

    def test_partial_json_with_surrounding_text(self) -> None:
        raw = 'Here is my judgment: {"score": 0.6} done.'
        score, explanation = LangChainEvaluator._judge_score(_PC, {"reasoning": raw})
        assert score == 0.6
        assert explanation == raw  # no JSON reasoning -> raw text kept

    @pytest.mark.parametrize(
        "raw",
        [
            "It scored 2 of 5",  # used to become 2 -> clamped 1.0
            "Score: 8/10",  # used to become 8 -> clamped 1.0
            "no numbers here",  # used to become a real 0.0
            "0.75",  # bare number: not the JSON the prompt asked for
            '{"score": 8, "reasoning": "out of 10"}',  # out of range: no clamp
            '{"score": -0.1}',
            '{"score": "0.5"}',
            '{"score": true}',
            '{"score": NaN}',
            "",
        ],
    )
    def test_partial_without_valid_json_score_is_unscored(self, raw: str) -> None:
        with pytest.raises(lc_module.UnscoredJudgeOutputError):
            LangChainEvaluator._judge_score(_PC, {"reasoning": raw, "score": 1})

    def test_correctness_uses_langchain_verdict(self) -> None:
        assert LangChainEvaluator._judge_score(
            _C, {"score": 0, "value": "N", "reasoning": "wrong"}
        ) == (0.0, "wrong")

    def test_correctness_without_verdict_is_unscored(self) -> None:
        with pytest.raises(lc_module.UnscoredJudgeOutputError):
            LangChainEvaluator._judge_score(
                _C, {"score": None, "value": "maybe", "reasoning": "maybe"}
            )


class TestPrepareEvalArgs:
    def test_reference_added_when_required(self, mocker) -> None:
        ev = _make_evaluator(mocker)
        args = ev._prepare_eval_args(EvaluationMetricType.CORRECTNESS, "q", "a", "gt")
        assert args == {"input": "q", "prediction": "a", "reference": "gt"}

    def test_reference_omitted_when_not_required(self, mocker) -> None:
        ev = _make_evaluator(mocker)
        # Inject a metric mapping entry without requires_reference.
        ev.METRIC_MAPPING = {
            EvaluationMetricType.CORRECTNESS: {
                "type": ev.METRIC_MAPPING[EvaluationMetricType.CORRECTNESS]["type"]
            }
        }
        args = ev._prepare_eval_args(EvaluationMetricType.CORRECTNESS, "q", "a", "gt")
        assert args == {"input": "q", "prediction": "a"}


class TestInitialization:
    def test_only_supported_metrics_become_evaluators(self, mocker) -> None:
        ev = _make_evaluator(
            mocker,
            langchain_metrics=[
                EvaluationMetricType.CORRECTNESS,
                EvaluationMetricType.FAITHFULNESS,  # unsupported by LangChain
            ],
        )
        # FAITHFULNESS is skipped; only CORRECTNESS gets an evaluator.
        assert set(ev.evaluators) == {EvaluationMetricType.CORRECTNESS}

    def test_init_failure_raises_evaluation_exception(self, mocker) -> None:
        from unified_kg_rag.shared import EvaluationException

        mocker.patch.object(
            bedrock_module,
            "BedrockLanguageModelFactory",
            side_effect=RuntimeError("boom"),
        )
        with pytest.raises(EvaluationException):
            LangChainEvaluator(config=Config())


class TestEvaluateWithMetric:
    async def test_correctness_reads_score_field_directly(self, mocker) -> None:
        ev = _make_evaluator(mocker)
        fake = _FakeEvaluator({"score": 0.9, "reasoning": "great"})
        metric = await ev._aevaluate_with_metric(
            fake, EvaluationMetricType.CORRECTNESS, "q", "a", "gt"
        )
        assert metric.metric_type == EvaluationMetricType.CORRECTNESS
        assert metric.value == 0.9
        assert metric.explanation == "great"

    async def test_correctness_missing_score_is_unscored(self, mocker) -> None:
        ev = _make_evaluator(mocker)
        fake = _FakeEvaluator({"reasoning": "no score"})
        with pytest.raises(lc_module.UnscoredJudgeOutputError):
            await ev._aevaluate_with_metric(
                fake, EvaluationMetricType.CORRECTNESS, "q", "a", "gt"
            )

    async def test_partial_correctness_reads_json_from_reasoning(self, mocker) -> None:
        ev = _make_evaluator(mocker)
        fake = _FakeEvaluator(
            {"reasoning": json.dumps({"score": 0.75, "reasoning": "partial"})}
        )
        metric = await ev._aevaluate_with_metric(
            fake, EvaluationMetricType.PARTIAL_CORRECTNESS, "q", "a", "gt"
        )
        assert metric.value == 0.75
        assert metric.explanation == "partial"

    async def test_eval_args_passed_through(self, mocker) -> None:
        ev = _make_evaluator(mocker)
        fake = _FakeEvaluator({"score": 1.0})
        await ev._aevaluate_with_metric(
            fake, EvaluationMetricType.CORRECTNESS, "the q", "the a", "the gt"
        )
        assert fake.last_kwargs == {
            "input": "the q",
            "prediction": "the a",
            "reference": "the gt",
        }


class TestHandleEvaluationError:
    def test_returns_failure_reason_not_a_zero_metric(self) -> None:
        reason = LangChainEvaluator._handle_evaluation_error(
            EvaluationMetricType.CORRECTNESS, "q1", ValueError("nope")
        )
        assert isinstance(reason, str)
        assert "nope" in reason


async def _raise(**kwargs):
    raise RuntimeError("async boom")


class TestAevaluateSingle:
    async def test_gathers_async_metrics(self, mocker) -> None:
        ev = _make_evaluator(
            mocker, langchain_metrics=[EvaluationMetricType.CORRECTNESS]
        )
        ev.evaluators = {
            EvaluationMetricType.CORRECTNESS: _FakeEvaluator(
                {"score": 0.6, "reasoning": "ok"}
            )
        }
        report = await ev.aevaluate_single(_query(), _result(), ground_truth="gt")
        assert report.evaluator_type == EvaluatorType.LANGCHAIN
        assert [m.value for m in report.metrics] == [pytest.approx(0.6)]

    async def test_skips_metric_without_evaluator(self, mocker) -> None:
        ev = _make_evaluator(
            mocker, langchain_metrics=[EvaluationMetricType.CORRECTNESS]
        )
        ev.evaluators = {}  # nothing registered
        report = await ev.aevaluate_single(_query(), _result(), ground_truth="gt")
        assert report.metrics == []

    async def test_judge_error_is_failed_metric_not_raised(self, mocker) -> None:
        ev = _make_evaluator(
            mocker, langchain_metrics=[EvaluationMetricType.CORRECTNESS]
        )
        assert ev.config.processing.ignore_errors is False
        boom = _FakeEvaluator({})
        boom.aevaluate_strings = _raise
        ev.evaluators = {EvaluationMetricType.CORRECTNESS: boom}
        report = await ev.aevaluate_single(_query(), _result(), ground_truth="gt")
        # A judge failure is recorded as failed, NOT scored 0.0, and not raised.
        assert report.metrics == []
        assert "async boom" in report.metadata["failed_metrics"]["correctness"]

    async def test_failed_metric_does_not_drop_the_others(self, mocker) -> None:
        ev = _make_evaluator(mocker, langchain_metrics=[_C, _PC])
        boom = _FakeEvaluator({})
        boom.aevaluate_strings = _raise
        ev.evaluators = {
            _C: boom,
            _PC: _FakeEvaluator({"reasoning": '{"score": 0.8}'}),
        }
        report = await ev.aevaluate_single(_query(), _result(), ground_truth="gt")
        assert [m.metric_type for m in report.metrics] == [_PC]
        assert report.metrics[0].value == pytest.approx(0.8)
        assert set(report.metadata["failed_metrics"]) == {"correctness"}

    async def test_async_unparseable_judge_output_is_failed(self, mocker) -> None:
        ev = _make_evaluator(mocker, langchain_metrics=[_PC])
        ev.evaluators = {_PC: _FakeEvaluator({"reasoning": "scored 2 of 5"})}
        report = await ev.aevaluate_single(_query(), _result(), ground_truth="gt")
        assert report.metrics == []
        assert "partial_correctness" in report.metadata["failed_metrics"]

    async def test_async_empty_reference_skips(self, mocker) -> None:
        ev = _make_evaluator(mocker, langchain_metrics=[_C, _PC])
        judge = _FakeEvaluator({"score": 1.0})
        ev.evaluators = {_C: judge, _PC: judge}
        report = await ev.aevaluate_single(_query(), _result(), ground_truth="  ")
        # Both metrics are reference-based: skipped, judge never called.
        assert report.metrics == []
        assert judge.last_kwargs is None
        assert report.metadata["skipped_metrics"] == {
            "correctness": "empty_reference",
            "partial_correctness": "empty_reference",
        }
        assert "failed_metrics" not in report.metadata


class TestValidateConfig:
    def test_supported_metrics_valid(self, mocker) -> None:
        ev = _make_evaluator(
            mocker, langchain_metrics=[EvaluationMetricType.CORRECTNESS]
        )
        assert ev.validate_config() is True

    def test_unsupported_metric_invalid(self, mocker) -> None:
        ev = _make_evaluator(
            mocker, langchain_metrics=[EvaluationMetricType.CORRECTNESS]
        )
        ev.config.evaluation.langchain_metrics = [EvaluationMetricType.FAITHFULNESS]
        assert ev.validate_config() is False


def test_judge_model_requested_without_sampling_params(mocker) -> None:
    # The judge must not force temperature/top_p/top_k: the factory decides
    # per model capability (Claude 5 rejects them outright).
    _make_evaluator(mocker)
    factory = bedrock_module.BedrockLanguageModelFactory.return_value
    _, kwargs = factory.get_model.call_args
    assert not {"temperature", "top_p", "top_k"} & set(kwargs)


def test_judge_model_uses_configured_judge_effort(mocker) -> None:
    _make_evaluator(mocker)
    factory = bedrock_module.BedrockLanguageModelFactory.return_value
    _, kwargs = factory.get_model.call_args
    assert kwargs["effort"] == Config().evaluation.judge_effort == "low"


def test_cancelling_an_evaluator_batch_cancels_its_queries() -> None:
    # Before, the per-query judge tasks (awaited through as_completed) kept
    # running after the evaluation was cancelled.
    import asyncio

    from unified_kg_rag.adapters.evaluators.base import BaseGraphRAGEvaluator
    from unified_kg_rag.domain.models import (
        Config,
        EvaluationQuery,
        EvaluationResult,
        EvaluatorType,
    )

    finished: list[str] = []

    class _Slow(BaseGraphRAGEvaluator):
        async def aevaluate_single(self, query, result, ground_truth, **kwargs):
            await asyncio.sleep(0.5)
            finished.append(query.query_id)

    evaluator = _Slow(Config(), EvaluatorType.LANGCHAIN, show_progress=False)
    queries = [EvaluationQuery(query_id=f"q{i}", question="?") for i in range(4)]
    results = [
        EvaluationResult(
            query_id=q.query_id, question="?", generated_answer="a", ground_truth=""
        )
        for q in queries
    ]

    async def main() -> None:
        task = asyncio.create_task(
            evaluator.aevaluate_batch(queries, results, [""] * len(queries))
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.8)

    asyncio.run(main())
    assert finished == []
