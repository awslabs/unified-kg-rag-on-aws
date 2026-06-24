# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for RagasEvaluator (AWS-free).

The constructor builds a boto session, an assumed-role session, a Bedrock
token counter and (in ``_initialize_evaluator``) Bedrock LLM/embedding models.
All of that is patched out. The real context truncation/token-budget logic,
metric selection/mapping, dataset preparation and DataFrame -> EvaluationMetric
parsing are exercised with a fake token counter and a stubbed ragas
``evaluate``.
"""

from __future__ import annotations

import pandas as pd
import pytest

import aws_graphrag.evaluation  # noqa: F401  (resolves package import cycle)
from aws_graphrag.adapters.evaluators import ragas_evaluator as rg_module
from aws_graphrag.adapters.evaluators.ragas_evaluator import RagasEvaluator
from aws_graphrag.domain.models import (
    Config,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationResult,
    EvaluatorType,
)

pytestmark = pytest.mark.unit


def _make_evaluator(mocker, *, ragas_metrics=None, max_context_tokens=8192):
    """Build a RagasEvaluator with all Bedrock/boto wiring patched out.

    Returns (evaluator, fake_token_counter).
    """
    mocker.patch.object(rg_module.boto3, "Session")
    mocker.patch.object(rg_module, "get_assumed_role_boto_session")
    mocker.patch.object(rg_module, "BedrockEmbeddingModelFactory")
    mocker.patch.object(rg_module, "BedrockLanguageModelFactory")

    fake_counter = mocker.Mock()
    # 1 token per whitespace-delimited word.
    fake_counter.count_tokens.side_effect = lambda text: len(text.split())
    fake_counter.truncate_to_token_limit.side_effect = lambda text, limit: (
        " ".join(text.split()[:limit]),
        limit,
    )
    mocker.patch.object(rg_module, "BedrockTokenCounter", return_value=fake_counter)

    config = Config()
    config.evaluation.max_context_tokens = max_context_tokens
    if ragas_metrics is not None:
        config.evaluation.ragas_metrics = ragas_metrics
    ev = RagasEvaluator(config=config, rag_chain=None, show_progress=False)
    return ev, fake_counter


def _query(qid="q1") -> EvaluationQuery:
    return EvaluationQuery(query_id=qid, question="Who founded Acme?")


def _result(contexts=None, qid="q1") -> EvaluationResult:
    return EvaluationResult(
        query_id=qid,
        question="Who founded Acme?",
        generated_answer="Alice",
        ground_truth="",
        retrieved_contexts=contexts or [],
    )


class TestTruncateContexts:
    def test_contexts_within_budget_kept_whole(self, mocker) -> None:
        ev, _ = _make_evaluator(mocker, max_context_tokens=1000)
        res = _result(contexts=["alpha beta", "gamma delta"])
        out = ev._truncate_contexts([res])
        assert out == [["alpha beta", "gamma delta"]]

    def test_overflowing_context_is_truncated_with_ellipsis(self, mocker) -> None:
        # BUFFER_TOKENS=128; pick a budget so the 2nd context overflows but
        # leaves > buffer remaining tokens to truncate into.
        ev, _ = _make_evaluator(mocker, max_context_tokens=200)
        first = " ".join(["w"] * 50)  # 50 tokens
        second = " ".join(["x"] * 100)  # would overflow
        out = ev._truncate_contexts([_result(contexts=[first, second])])
        contexts = out[0]
        # First fits whole; second is truncated and gets an ellipsis suffix.
        assert contexts[0] == first
        assert contexts[1].endswith("...")
        assert len(contexts) == 2

    def test_overflow_with_too_little_remaining_drops_context(self, mocker) -> None:
        # After the first context, remaining budget <= BUFFER_TOKENS so the
        # overflowing context is dropped entirely (no truncated fragment added).
        ev, _ = _make_evaluator(mocker, max_context_tokens=200)
        # first=72 fits exactly (0+72 == max-buffer == 72, not strictly over).
        # After it, remaining = 200-72 = 128, which is NOT > buffer(128) -> the
        # overflowing second context is dropped with no truncated fragment.
        first = " ".join(["w"] * 72)
        second = " ".join(["x"] * 50)
        out = ev._truncate_contexts([_result(contexts=[first, second])])
        assert out[0] == [first]

    def test_per_result_independence(self, mocker) -> None:
        ev, _ = _make_evaluator(mocker, max_context_tokens=1000)
        out = ev._truncate_contexts(
            [_result(contexts=["a b"]), _result(contexts=["c"])]
        )
        assert out == [["a b"], ["c"]]


class TestParseRagasReports:
    def test_maps_dataframe_columns_to_metrics(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker,
            ragas_metrics=[
                EvaluationMetricType.ANSWER_CORRECTNESS,
                EvaluationMetricType.FAITHFULNESS,
            ],
        )
        df = pd.DataFrame({"answer_correctness": [0.8], "faithfulness": [0.6]})
        reports = ev._parse_ragas_reports(df, [_query()], [_result()])
        assert len(reports) == 1
        report = reports[0]
        assert report.evaluator_type == EvaluatorType.RAGAS
        values = {m.metric_type: m.value for m in report.metrics}
        assert values == {
            EvaluationMetricType.ANSWER_CORRECTNESS: 0.8,
            EvaluationMetricType.FAITHFULNESS: 0.6,
        }
        assert report.overall_score == pytest.approx(0.7)

    def test_nan_value_becomes_zero(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.FAITHFULNESS]
        )
        df = pd.DataFrame({"faithfulness": [float("nan")]})
        reports = ev._parse_ragas_reports(df, [_query()], [_result()])
        assert reports[0].metrics[0].value == 0.0

    def test_metric_not_in_config_excluded(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.FAITHFULNESS]
        )
        # DataFrame has an extra column not enabled in config -> ignored.
        df = pd.DataFrame({"faithfulness": [0.5], "answer_correctness": [0.9]})
        reports = ev._parse_ragas_reports(df, [_query()], [_result()])
        types = {m.metric_type for m in reports[0].metrics}
        assert types == {EvaluationMetricType.FAITHFULNESS}

    def test_missing_column_skipped(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker,
            ragas_metrics=[
                EvaluationMetricType.FAITHFULNESS,
                EvaluationMetricType.ANSWER_RELEVANCY,
            ],
        )
        # Config wants two metrics but DataFrame only has one column.
        df = pd.DataFrame({"faithfulness": [0.5]})
        reports = ev._parse_ragas_reports(df, [_query()], [_result()])
        assert len(reports[0].metrics) == 1

    def test_no_matching_metrics_overall_zero(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.FAITHFULNESS]
        )
        df = pd.DataFrame({"unrelated": [0.5]})
        reports = ev._parse_ragas_reports(df, [_query()], [_result()])
        assert reports[0].metrics == []
        assert reports[0].overall_score == 0.0


class TestAevaluateBatch:
    async def test_no_valid_metrics_returns_empty(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.CORRECTNESS]
        )
        # CORRECTNESS is not a ragas metric -> metrics_to_use empty.
        out = await ev.aevaluate_batch([_query()], [_result()], [""])
        assert out == []

    async def test_builds_dataset_and_parses_results(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.FAITHFULNESS]
        )

        captured = {}

        class _FakeRagasResult:
            def to_pandas(self):
                return pd.DataFrame({"faithfulness": [0.9]})

        def fake_evaluate(*, dataset, metrics, **kwargs):
            captured["dataset"] = dataset
            captured["metrics"] = metrics
            return _FakeRagasResult()

        def fake_from_dict(d):
            captured["dataset_dict"] = d
            return d

        mocker.patch.object(rg_module, "evaluate", side_effect=fake_evaluate)
        mocker.patch.object(rg_module.Dataset, "from_dict", side_effect=fake_from_dict)

        reports = await ev.aevaluate_batch(
            [_query()], [_result(contexts=["ctx one"])], ["truth"]
        )
        # Dataset built from the right columns.
        dd = captured["dataset_dict"]
        assert dd["question"] == ["Who founded Acme?"]
        assert dd["answer"] == ["Alice"]
        assert dd["contexts"] == [["ctx one"]]
        assert dd["ground_truth"] == ["truth"]
        # Faithfulness metric object selected.
        assert captured["metrics"] == [
            rg_module.RagasEvaluator.RAGAS_METRICS[EvaluationMetricType.FAITHFULNESS]
        ]
        assert reports[0].metrics[0].value == 0.9

    async def test_failure_returns_empty_reports_when_not_ignore_errors(
        self, mocker
    ) -> None:
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.FAITHFULNESS]
        )
        ev.ignore_errors = False
        mocker.patch.object(
            rg_module, "evaluate", side_effect=RuntimeError("ragas down")
        )
        reports = await ev.aevaluate_batch([_query()], [_result()], [""])
        # One empty report per query, flagged as failed.
        assert len(reports) == 1
        assert reports[0].metadata.get("evaluation_failed") is True
        assert reports[0].metrics == []

    async def test_failure_reraises_when_ignore_errors_true(self, mocker) -> None:
        # NOTE: the source re-raises when ignore_errors is True (inverted-looking
        # guard); pin that real behaviour.
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.FAITHFULNESS]
        )
        ev.ignore_errors = True
        mocker.patch.object(
            rg_module, "evaluate", side_effect=RuntimeError("ragas down")
        )
        with pytest.raises(RuntimeError):
            await ev.aevaluate_batch([_query()], [_result()], [""])


class TestValidateConfig:
    def test_supported_metrics_valid(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.FAITHFULNESS]
        )
        assert ev.validate_config() is True

    def test_unsupported_metric_invalid(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.FAITHFULNESS]
        )
        ev.config.evaluation.ragas_metrics = [EvaluationMetricType.CORRECTNESS]
        assert ev.validate_config() is False


class TestInitFailure:
    def test_init_failure_raises_evaluation_exception(self, mocker) -> None:
        from aws_graphrag.shared import EvaluationException

        mocker.patch.object(rg_module.boto3, "Session")
        mocker.patch.object(rg_module, "get_assumed_role_boto_session")
        mocker.patch.object(rg_module, "BedrockTokenCounter")
        mocker.patch.object(rg_module, "BedrockEmbeddingModelFactory")
        mocker.patch.object(
            rg_module,
            "BedrockLanguageModelFactory",
            side_effect=RuntimeError("boom"),
        )
        with pytest.raises(EvaluationException):
            RagasEvaluator(config=Config(), rag_chain=None)
