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
    mocker.patch.object(lc_module, "BedrockLanguageModelFactory")
    mocker.patch.object(lc_module, "load_evaluator", return_value=object())
    mocker.patch.object(lc_module.boto3, "Session")

    config = Config()
    if langchain_metrics is not None:
        config.evaluation.langchain_metrics = langchain_metrics
    return LangChainEvaluator(config=config, rag_chain=None)


def _query() -> EvaluationQuery:
    return EvaluationQuery(query_id="q1", question="Who founded Acme?")


def _result(answer: str = "Alice founded Acme.") -> EvaluationResult:
    return EvaluationResult(
        query_id="q1",
        question="Who founded Acme?",
        generated_answer=answer,
        ground_truth="",
    )


class TestStripMarkdownCodeFence:
    def test_fenced_json_block_is_unwrapped(self) -> None:
        text = '```json\n{"score": 0.5}\n```'
        assert LangChainEvaluator._strip_markdown_code_fence(text) == '{"score": 0.5}'

    def test_bare_fence_is_unwrapped(self) -> None:
        text = "```\nhello\n```"
        assert LangChainEvaluator._strip_markdown_code_fence(text) == "hello"

    def test_no_fence_passes_through_stripped(self) -> None:
        assert LangChainEvaluator._strip_markdown_code_fence("  plain  ") == "plain"


class TestParseScore:
    def test_numeric_score_field(self) -> None:
        assert LangChainEvaluator._parse_score({"score": 0.7}) == 0.7

    def test_value_field_fallback(self) -> None:
        # No "score" key -> falls back to "value".
        assert LangChainEvaluator._parse_score({"value": 1}) == 1.0

    def test_json_embedded_in_reasoning(self) -> None:
        reasoning = '{"score": 0.42, "reasoning": "ok"}'
        assert LangChainEvaluator._parse_score({"reasoning": reasoning}) == 0.42

    def test_json_with_surrounding_text_via_brace_regex(self) -> None:
        reasoning = 'Here is my judgment: {"score": 0.6} done.'
        assert LangChainEvaluator._parse_score({"reasoning": reasoning}) == 0.6

    def test_first_number_fallback_regex(self) -> None:
        # Not valid JSON and no brace block -> first number is extracted.
        reasoning = "The score is 0.85 overall."
        assert LangChainEvaluator._parse_score({"reasoning": reasoning}) == 0.85

    def test_unparseable_returns_zero(self) -> None:
        assert LangChainEvaluator._parse_score({"reasoning": "no numbers here"}) == 0.0

    def test_empty_result_returns_zero(self) -> None:
        assert LangChainEvaluator._parse_score({}) == 0.0


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

        mocker.patch.object(lc_module.boto3, "Session")
        mocker.patch.object(
            lc_module,
            "BedrockLanguageModelFactory",
            side_effect=RuntimeError("boom"),
        )
        with pytest.raises(EvaluationException):
            LangChainEvaluator(config=Config(), rag_chain=None)


class TestEvaluateWithMetric:
    def test_correctness_reads_score_field_directly(self, mocker) -> None:
        ev = _make_evaluator(mocker)
        fake = _FakeEvaluator({"score": 0.9, "reasoning": "great"})
        metric = ev._evaluate_with_metric(
            fake, EvaluationMetricType.CORRECTNESS, "q", "a", "gt"
        )
        assert metric.metric_type == EvaluationMetricType.CORRECTNESS
        assert metric.value == 0.9
        assert metric.explanation == "great"

    def test_correctness_missing_score_defaults_zero(self, mocker) -> None:
        ev = _make_evaluator(mocker)
        fake = _FakeEvaluator({"reasoning": "no score"})
        metric = ev._evaluate_with_metric(
            fake, EvaluationMetricType.CORRECTNESS, "q", "a", "gt"
        )
        assert metric.value == 0.0

    def test_partial_correctness_uses_parse_score(self, mocker) -> None:
        ev = _make_evaluator(mocker)
        # No top-level score -> parse_score digs into reasoning JSON.
        fake = _FakeEvaluator(
            {"reasoning": json.dumps({"score": 0.75, "reasoning": "partial"})}
        )
        metric = ev._evaluate_with_metric(
            fake, EvaluationMetricType.PARTIAL_CORRECTNESS, "q", "a", "gt"
        )
        assert metric.value == 0.75
        # Reasoning JSON's "reasoning" field becomes the explanation.
        assert metric.explanation == "partial"

    def test_partial_correctness_fenced_reasoning_extracted(self, mocker) -> None:
        ev = _make_evaluator(mocker)
        fenced = '```json\n{"score": 0.5, "reasoning": "fenced reason"}\n```'
        fake = _FakeEvaluator({"score": 0.5, "reasoning": fenced})
        metric = ev._evaluate_with_metric(
            fake, EvaluationMetricType.PARTIAL_CORRECTNESS, "q", "a", "gt"
        )
        assert metric.value == 0.5
        assert metric.explanation == "fenced reason"

    def test_partial_correctness_plain_reasoning_kept(self, mocker) -> None:
        ev = _make_evaluator(mocker)
        fake = _FakeEvaluator({"score": 0.3, "reasoning": "just text"})
        metric = ev._evaluate_with_metric(
            fake, EvaluationMetricType.PARTIAL_CORRECTNESS, "q", "a", "gt"
        )
        # Not JSON -> explanation stays the raw reasoning.
        assert metric.explanation == "just text"

    def test_eval_args_passed_through(self, mocker) -> None:
        ev = _make_evaluator(mocker)
        fake = _FakeEvaluator({"score": 1.0})
        ev._evaluate_with_metric(
            fake, EvaluationMetricType.CORRECTNESS, "the q", "the a", "the gt"
        )
        assert fake.last_kwargs == {
            "input": "the q",
            "prediction": "the a",
            "reference": "the gt",
        }


class TestHandleEvaluationError:
    def test_error_metric_shape(self) -> None:
        metric = LangChainEvaluator._handle_evaluation_error(
            EvaluationMetricType.CORRECTNESS, "q1", ValueError("nope")
        )
        assert metric.value == 0.0
        assert metric.metadata == {"evaluation_error": True}
        assert "nope" in metric.explanation


class TestEvaluateSingle:
    def test_aggregates_metrics_and_overall_score(self, mocker) -> None:
        ev = _make_evaluator(
            mocker, langchain_metrics=[EvaluationMetricType.CORRECTNESS]
        )
        ev.evaluators = {
            EvaluationMetricType.CORRECTNESS: _FakeEvaluator(
                {"score": 0.8, "reasoning": "ok"}
            )
        }
        report = ev.evaluate_single(_query(), _result(), ground_truth="gt")
        assert report.evaluator_type == EvaluatorType.LANGCHAIN
        assert len(report.metrics) == 1
        assert report.overall_score == pytest.approx(0.8)

    def test_skips_metric_without_evaluator(self, mocker) -> None:
        ev = _make_evaluator(
            mocker, langchain_metrics=[EvaluationMetricType.CORRECTNESS]
        )
        ev.evaluators = {}  # nothing registered
        report = ev.evaluate_single(_query(), _result(), ground_truth="gt")
        assert report.metrics == []
        assert report.overall_score == 0.0

    def test_error_swallowed_when_ignore_errors(self, mocker) -> None:
        ev = _make_evaluator(
            mocker, langchain_metrics=[EvaluationMetricType.CORRECTNESS]
        )
        ev.ignore_errors = True
        boom = _FakeEvaluator({})
        boom.evaluate_strings = mocker.Mock(side_effect=RuntimeError("x"))
        ev.evaluators = {EvaluationMetricType.CORRECTNESS: boom}
        report = ev.evaluate_single(_query(), _result(), ground_truth="gt")
        assert report.metrics[0].metadata == {"evaluation_error": True}

    def test_error_reraised_when_not_ignore_errors(self, mocker) -> None:
        ev = _make_evaluator(
            mocker, langchain_metrics=[EvaluationMetricType.CORRECTNESS]
        )
        ev.ignore_errors = False
        boom = _FakeEvaluator({})
        boom.evaluate_strings = mocker.Mock(side_effect=RuntimeError("x"))
        ev.evaluators = {EvaluationMetricType.CORRECTNESS: boom}
        with pytest.raises(RuntimeError):
            ev.evaluate_single(_query(), _result(), ground_truth="gt")


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
        assert report.overall_score == pytest.approx(0.6)

    async def test_async_error_handled_when_ignore_errors(self, mocker) -> None:
        ev = _make_evaluator(
            mocker, langchain_metrics=[EvaluationMetricType.CORRECTNESS]
        )
        ev.ignore_errors = True
        boom = _FakeEvaluator({})

        async def _raise(**kwargs):
            raise RuntimeError("async boom")

        boom.aevaluate_strings = _raise
        ev.evaluators = {EvaluationMetricType.CORRECTNESS: boom}
        report = await ev.aevaluate_single(_query(), _result(), ground_truth="gt")
        assert report.metrics[0].metadata == {"evaluation_error": True}


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
