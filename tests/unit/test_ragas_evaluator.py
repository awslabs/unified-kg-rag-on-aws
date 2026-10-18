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

import unified_kg_rag.evaluation  # noqa: F401  (resolves package import cycle)
from tests.fixtures.fakes.chat_models import fake_model_info
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.evaluators import ragas_evaluator as rg_module
from unified_kg_rag.adapters.evaluators.ragas_evaluator import RagasEvaluator
from unified_kg_rag.domain.models import (
    Config,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationResult,
    EvaluatorType,
    LanguageModelId,
)

pytestmark = pytest.mark.unit


def _make_evaluator(
    mocker,
    *,
    ragas_metrics=None,
    max_context_tokens=8192,
    evaluation_model_id=None,
    ragas_max_contexts=20,
    answer_context_tokens=1_000_000,
):
    """Build a RagasEvaluator with all Bedrock/boto wiring patched out.

    Returns (evaluator, fake_token_counter).
    """
    mocker.patch.object(bedrock_module, "get_assumed_role_boto_session")
    mocker.patch.object(bedrock_module, "BedrockEmbeddingModelFactory")
    llm_factory = mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
    llm_factory.return_value.get_model_info.side_effect = fake_model_info

    fake_counter = mocker.Mock()
    # 1 token per whitespace-delimited word.
    fake_counter.count_tokens.side_effect = lambda text: len(text.split())
    fake_counter.truncate_to_token_limit.side_effect = lambda text, limit: (
        " ".join(text.split()[:limit]),
        limit,
    )
    mocker.patch.object(
        bedrock_module, "BedrockTokenCounter", return_value=fake_counter
    )

    config = Config()
    config.evaluation.max_context_tokens = max_context_tokens
    config.search.token_manager.max_context_tokens = answer_context_tokens
    config.evaluation.ragas_max_contexts = ragas_max_contexts
    if ragas_metrics is not None:
        config.evaluation.ragas_metrics = ragas_metrics
    if evaluation_model_id is not None:
        config.models.roles.evaluation = evaluation_model_id
    ev = RagasEvaluator(config=config, show_progress=False)
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


def _contexts(ev, results, capped: bool = False) -> list[list[str]]:
    cap = ev.config.evaluation.ragas_max_contexts if capped else None
    return [contexts for contexts, _ in ev._select(ev._budgeted_contexts(results), cap)]


class TestTruncateContexts:
    def test_contexts_within_budget_kept_whole(self, mocker) -> None:
        ev, _ = _make_evaluator(mocker, max_context_tokens=1000)
        res = _result(contexts=["alpha beta", "gamma delta"])
        out = _contexts(ev, [res])
        assert out == [["alpha beta", "gamma delta"]]

    def test_overflowing_context_is_truncated_with_ellipsis(self, mocker) -> None:
        # BUFFER_TOKENS=128, so 72 tokens are usable: the 2nd context
        # overflows and is truncated into the 22 left.
        ev, _ = _make_evaluator(mocker, max_context_tokens=200)
        first = " ".join(["w"] * 50)  # 50 tokens
        second = " ".join(["x"] * 100)  # would overflow
        out = _contexts(ev, [_result(contexts=[first, second])])
        contexts = out[0]
        # First fits whole; second is truncated and gets an ellipsis suffix.
        assert contexts[0] == first
        assert contexts[1].endswith("...")
        assert len(contexts) == 2

    def test_overflow_with_too_little_remaining_drops_context(self, mocker) -> None:
        # first=72 fills the usable budget (max - buffer = 72) exactly, so
        # the overflowing second context is dropped with no truncated fragment.
        ev, _ = _make_evaluator(mocker, max_context_tokens=200)
        first = " ".join(["w"] * 72)
        second = " ".join(["x"] * 50)
        out = _contexts(ev, [_result(contexts=[first, second])])
        assert out[0] == [first]

    def test_per_result_independence(self, mocker) -> None:
        ev, _ = _make_evaluator(mocker, max_context_tokens=1000)
        out = _contexts(ev, [_result(contexts=["a b"]), _result(contexts=["c"])])
        assert out == [["a b"], ["c"]]

    def test_count_cap_keeps_top_ranked_contexts(self, mocker) -> None:
        ev, _ = _make_evaluator(mocker, ragas_max_contexts=2)
        out = _contexts(ev, [_result(contexts=["r1", "r2", "r3", "r4"])], capped=True)
        assert out == [["r1", "r2"]]

    def test_count_cap_none_disables(self, mocker) -> None:
        ev, _ = _make_evaluator(mocker, ragas_max_contexts=None)
        contexts = [f"c{i}" for i in range(50)]
        out = _contexts(ev, [_result(contexts=contexts)], capped=True)
        assert out == [contexts]

    def test_count_cap_applies_before_token_budget(self, mocker) -> None:
        # Without the count cap the oversized 3rd context would be truncated
        # into the leftover budget (400 - 60 > buffer) and kept as a fragment;
        # capped at 2, the selection keeps only the first two.
        ev, _ = _make_evaluator(mocker, max_context_tokens=400, ragas_max_contexts=2)
        first = " ".join(["a"] * 30)
        second = " ".join(["b"] * 30)
        third = " ".join(["c"] * 300)
        out = _contexts(ev, [_result(contexts=[first, second, third])], capped=True)
        assert out == [[first, second]]

    def test_token_budget_still_binds_under_count_cap(self, mocker) -> None:
        ev, _ = _make_evaluator(mocker, max_context_tokens=200, ragas_max_contexts=5)
        first = " ".join(["w"] * 72)
        out = _contexts(ev, [_result(contexts=[first, "x y", "z"])], capped=True)
        assert out == [[first]]

    def test_reports_tokens_seen(self, mocker) -> None:
        ev, _ = _make_evaluator(mocker, max_context_tokens=200)
        first = " ".join(["w"] * 50)
        second = " ".join(["x"] * 100)
        [(contexts, tokens)] = ev._select(
            ev._budgeted_contexts([_result(contexts=[first, second])])
        )
        # 50 whole + the second truncated into the 22 tokens left after the
        # buffer: the total stays within max_context_tokens - BUFFER_TOKENS.
        assert len(contexts) == 2 and tokens == 200 - ev.BUFFER_TOKENS

    def test_judge_budget_defaults_to_answer_budget(self, mocker) -> None:
        # The old fixed 8192 cut what the judges saw to a quarter of the
        # answer model's 30000-token context.
        assert Config().evaluation.max_context_tokens is None
        ev, _ = _make_evaluator(
            mocker, max_context_tokens=None, answer_context_tokens=200
        )
        assert ev._judge_context_budget == 200
        out = _contexts(ev, [_result(contexts=[" ".join(["w"] * 100)])])
        assert out[0][0].endswith("...")  # cut at 200 - BUFFER_TOKENS

    def test_no_cap_when_answer_budget_is_derived(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker, max_context_tokens=None, answer_context_tokens=None
        )
        assert ev._judge_context_budget is None
        contexts = [" ".join(["w"] * 5000), "x"]
        assert _contexts(ev, [_result(contexts=contexts)]) == [contexts]

    def test_judge_budget_below_answer_budget_warns(self, mocker, caplog) -> None:
        with caplog.at_level("WARNING"):
            ev, _ = _make_evaluator(
                mocker, max_context_tokens=8192, answer_context_tokens=30000
            )
        assert ev._judge_context_budget == 8192
        assert "less context than the answer model" in caplog.text

    def test_default_cap_is_twenty(self) -> None:
        assert Config().evaluation.ragas_max_contexts == 20


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

    def test_nan_value_is_skipped_not_zeroed(self, mocker) -> None:
        # NaN means "uncomputable" — it must be skipped, not coerced to 0.0
        # (which would conflate a measurement failure with a genuine zero and
        # bias the aggregate down).
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.FAITHFULNESS]
        )
        df = pd.DataFrame({"faithfulness": [float("nan")]})
        reports = ev._parse_ragas_reports(df, [_query()], [_result()])
        assert reports[0].metrics == []
        assert "faithfulness" in reports[0].metadata["failed_metrics"]

    def test_empty_reference_skips_reference_metrics(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker,
            ragas_metrics=[
                EvaluationMetricType.ANSWER_CORRECTNESS,
                EvaluationMetricType.CONTEXT_RECALL,
                EvaluationMetricType.FAITHFULNESS,
            ],
        )
        df = pd.DataFrame(
            {
                "answer_correctness": [0.0, 0.7],
                "context_recall": [0.0, 0.5],
                "faithfulness": [0.9, 0.9],
            }
        )
        reports = ev._parse_ragas_reports(
            df, [_query("q1"), _query("q2")], [_result(), _result()], ["", "truth"]
        )
        # Row without a reference: only the reference-free metric is scored.
        assert {m.metric_type for m in reports[0].metrics} == {
            EvaluationMetricType.FAITHFULNESS
        }
        assert reports[0].metadata["skipped_metrics"] == {
            "answer_correctness": "empty_reference",
            "context_recall": "empty_reference",
        }
        # Row with a reference: everything scored.
        assert len(reports[1].metrics) == 3
        assert "skipped_metrics" not in reports[1].metadata

    def test_nan_metric_not_scored_alongside_real_one(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker,
            ragas_metrics=[
                EvaluationMetricType.FAITHFULNESS,
                EvaluationMetricType.ANSWER_RELEVANCY,
            ],
        )
        df = pd.DataFrame({"faithfulness": [0.8], "answer_relevancy": [float("nan")]})
        reports = ev._parse_ragas_reports(df, [_query()], [_result()])
        assert len(reports[0].metrics) == 1
        assert reports[0].metrics[0].value == 0.8

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

    def test_no_matching_metrics_fail(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.FAITHFULNESS]
        )
        df = pd.DataFrame({"unrelated": [0.5]})
        reports = ev._parse_ragas_reports(df, [_query()], [_result()])
        assert reports[0].metrics == []
        assert "faithfulness" in reports[0].metadata["failed_metrics"]


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

    async def test_count_cap_applies_only_to_context_precision(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker,
            ragas_metrics=[
                EvaluationMetricType.CONTEXT_PRECISION,
                EvaluationMetricType.FAITHFULNESS,
            ],
            ragas_max_contexts=2,
        )
        runs: list[tuple[list, list]] = []

        class _FakeRagasResult:
            def __init__(self, columns: dict) -> None:
                self.columns = columns

            def to_pandas(self):
                return pd.DataFrame({**self.columns, "user_input": ["q"]})

        def fake_evaluate(*, dataset, metrics, **kwargs):
            runs.append((metrics, dataset["contexts"]))
            name = metrics[0].name
            return _FakeRagasResult({name: [0.5 if name == "faithfulness" else 0.25]})

        mocker.patch.object(rg_module, "evaluate", side_effect=fake_evaluate)
        mocker.patch.object(rg_module.Dataset, "from_dict", side_effect=lambda d: d)
        contexts = ["c1", "c2", "c3", "c4"]
        reports = await ev.aevaluate_batch(
            [_query()], [_result(contexts=contexts)], ["truth"]
        )
        metrics = rg_module.RagasEvaluator.RAGAS_METRICS
        assert runs == [
            ([metrics[EvaluationMetricType.FAITHFULNESS]], [contexts]),
            ([metrics[EvaluationMetricType.CONTEXT_PRECISION]], [["c1", "c2"]]),
        ]
        values = {m.metric_type.value: m.value for m in reports[0].metrics}
        assert values == {"context_precision": 0.25, "faithfulness": 0.5}
        md = reports[0].metadata
        assert md["judge_contexts"] == 4 and md["judge_context_tokens"] == 4
        assert md["context_precision_contexts"] == 2
        assert md["context_precision_context_tokens"] == 2

    async def test_each_context_counted_once_with_the_count_cap(self, mocker) -> None:
        # The capped context_precision run reuses the full run's counts; a
        # second walk recounted every context (a Bedrock CountTokens call
        # each once the counter's LRU cache is exceeded).
        ev, counter = _make_evaluator(
            mocker,
            ragas_metrics=[
                EvaluationMetricType.CONTEXT_PRECISION,
                EvaluationMetricType.FAITHFULNESS,
            ],
            ragas_max_contexts=2,
        )

        class _FakeRagasResult:
            def __init__(self, name: str) -> None:
                self.name = name

            def to_pandas(self):
                return pd.DataFrame({self.name: [1.0]})

        mocker.patch.object(
            rg_module,
            "evaluate",
            side_effect=lambda *, metrics, **_: _FakeRagasResult(metrics[0].name),
        )
        mocker.patch.object(rg_module.Dataset, "from_dict", side_effect=lambda d: d)
        await ev.aevaluate_batch(
            [_query()], [_result(contexts=["c1", "c2", "c3", "c4"])], ["truth"]
        )
        assert counter.count_tokens.call_count == 4

    async def test_single_run_when_cap_does_not_bind(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker,
            ragas_metrics=[
                EvaluationMetricType.CONTEXT_PRECISION,
                EvaluationMetricType.FAITHFULNESS,
            ],
            ragas_max_contexts=5,
        )

        class _FakeRagasResult:
            def to_pandas(self):
                return pd.DataFrame({"context_precision": [1.0], "faithfulness": [1.0]})

        evaluate = mocker.patch.object(
            rg_module, "evaluate", return_value=_FakeRagasResult()
        )
        mocker.patch.object(rg_module.Dataset, "from_dict", side_effect=lambda d: d)
        await ev.aevaluate_batch([_query()], [_result(contexts=["c1"])], ["truth"])
        assert evaluate.call_count == 1

    async def test_failure_returns_failed_reports(self, mocker) -> None:
        # Even with ignore_errors=False, a RAGAS failure after every answer is
        # generated becomes failed metrics instead of aborting the run.
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.FAITHFULNESS]
        )
        assert ev.config.processing.ignore_errors is False
        mocker.patch.object(
            rg_module, "evaluate", side_effect=RuntimeError("ragas down")
        )
        reports = await ev.aevaluate_batch([_query()], [_result()], [""])
        assert len(reports) == 1
        assert reports[0].metrics == []
        assert "ragas down" in reports[0].metadata["failed_metrics"]["faithfulness"]

    async def test_reference_metrics_not_run_when_no_row_has_reference(
        self, mocker
    ) -> None:
        ev, _ = _make_evaluator(
            mocker,
            ragas_metrics=[
                EvaluationMetricType.ANSWER_CORRECTNESS,
                EvaluationMetricType.FAITHFULNESS,
            ],
        )
        captured = {}

        class _FakeRagasResult:
            def to_pandas(self):
                return pd.DataFrame({"faithfulness": [0.9]})

        def fake_evaluate(*, dataset, metrics, **kwargs):
            captured["metrics"] = metrics
            return _FakeRagasResult()

        mocker.patch.object(rg_module, "evaluate", side_effect=fake_evaluate)
        mocker.patch.object(rg_module.Dataset, "from_dict", side_effect=lambda d: d)
        reports = await ev.aevaluate_batch([_query()], [_result()], [""])
        assert captured["metrics"] == [
            rg_module.RagasEvaluator.RAGAS_METRICS[EvaluationMetricType.FAITHFULNESS]
        ]
        assert reports[0].metadata["skipped_metrics"] == {
            "answer_correctness": "empty_reference"
        }

    async def test_rows_without_reference_run_without_reference_metrics(
        self, mocker
    ) -> None:
        # Judge calls for a reference metric on a row without a reference
        # would be thrown away, so those rows get their own run without it.
        ev, _ = _make_evaluator(
            mocker,
            ragas_metrics=[
                EvaluationMetricType.ANSWER_CORRECTNESS,
                EvaluationMetricType.FAITHFULNESS,
            ],
        )
        runs: list[tuple[list[str], list]] = []

        class _FakeRagasResult:
            def __init__(self, dataset, metrics) -> None:
                self.frame = pd.DataFrame(
                    {m.name: [0.5] * len(dataset["question"]) for m in metrics}
                )

            def to_pandas(self):
                return self.frame

        def fake_evaluate(*, dataset, metrics, **kwargs):
            runs.append((dataset["question"], metrics))
            return _FakeRagasResult(dataset, metrics)

        mocker.patch.object(rg_module, "evaluate", side_effect=fake_evaluate)
        mocker.patch.object(rg_module.Dataset, "from_dict", side_effect=lambda d: d)
        queries = [
            EvaluationQuery(query_id="no-ref", question="A?"),
            EvaluationQuery(query_id="ref", question="B?"),
        ]
        reports = await ev.aevaluate_batch(
            queries, [_result(), _result()], ["", "truth"]
        )
        metrics = rg_module.RagasEvaluator.RAGAS_METRICS
        assert runs == [
            (
                ["B?"],
                [
                    metrics[EvaluationMetricType.ANSWER_CORRECTNESS],
                    metrics[EvaluationMetricType.FAITHFULNESS],
                ],
            ),
            (["A?"], [metrics[EvaluationMetricType.FAITHFULNESS]]),
        ]
        assert [r.query_id for r in reports] == ["no-ref", "ref"]
        assert [m.metric_type.value for m in reports[0].metrics] == ["faithfulness"]
        assert reports[0].metadata["skipped_metrics"] == {
            "answer_correctness": "empty_reference"
        }
        assert {m.metric_type.value for m in reports[1].metrics} == {
            "answer_correctness",
            "faithfulness",
        }

    async def test_output_blocked_rows_skip_the_context_metrics(self, mocker) -> None:
        # Before, a blocked answer (its sources dropped) was judged on the
        # context metrics: precision/recall near 0 and faithfulness NaN, which
        # counted as failed toward --max-failure-rate.
        ev, _ = _make_evaluator(
            mocker,
            ragas_metrics=[
                EvaluationMetricType.ANSWER_RELEVANCY,
                EvaluationMetricType.CONTEXT_PRECISION,
                EvaluationMetricType.CONTEXT_RECALL,
                EvaluationMetricType.FAITHFULNESS,
            ],
        )
        runs: list[tuple[list[str], list[str]]] = []

        class _FakeRagasResult:
            def __init__(self, dataset, metrics) -> None:
                self.frame = pd.DataFrame(
                    {m.name: [0.5] * len(dataset["question"]) for m in metrics}
                )

            def to_pandas(self):
                return self.frame

        def fake_evaluate(*, dataset, metrics, **kwargs):
            runs.append((dataset["question"], sorted(m.name for m in metrics)))
            return _FakeRagasResult(dataset, metrics)

        mocker.patch.object(rg_module, "evaluate", side_effect=fake_evaluate)
        mocker.patch.object(rg_module.Dataset, "from_dict", side_effect=lambda d: d)
        blocked = _result(qid="blocked")
        blocked.guardrail_blocked = "output"
        queries = [
            EvaluationQuery(query_id="blocked", question="A?"),
            EvaluationQuery(query_id="ok", question="B?"),
        ]
        reports = await ev.aevaluate_batch(
            queries, [blocked, _result(qid="ok")], ["truth", "truth"]
        )
        assert runs[1] == (["A?"], ["answer_relevancy"])
        assert [m.metric_type.value for m in reports[0].metrics] == ["answer_relevancy"]
        assert reports[0].metadata["skipped_metrics"] == {
            "context_precision": "guardrail_blocked",
            "context_recall": "guardrail_blocked",
            "faithfulness": "guardrail_blocked",
        }
        assert "failed_metrics" not in reports[0].metadata
        assert len(reports[1].metrics) == 4

    async def test_failed_no_reference_run_fails_its_metrics_and_skips_reference(
        self, mocker
    ) -> None:
        ev, _ = _make_evaluator(
            mocker,
            ragas_metrics=[
                EvaluationMetricType.ANSWER_CORRECTNESS,
                EvaluationMetricType.FAITHFULNESS,
            ],
        )

        class _FakeRagasResult:
            def __init__(self, dataset, metrics) -> None:
                self.frame = pd.DataFrame(
                    {m.name: [0.5] * len(dataset["question"]) for m in metrics}
                )

            def to_pandas(self):
                return self.frame

        def fake_evaluate(*, dataset, metrics, **kwargs):
            if dataset["question"] == ["A?"]:
                raise RuntimeError("judge throttled")
            return _FakeRagasResult(dataset, metrics)

        mocker.patch.object(rg_module, "evaluate", side_effect=fake_evaluate)
        mocker.patch.object(rg_module.Dataset, "from_dict", side_effect=lambda d: d)
        queries = [
            EvaluationQuery(query_id="no-ref", question="A?"),
            EvaluationQuery(query_id="ref", question="B?"),
        ]
        no_ref, ref = await ev.aevaluate_batch(
            queries, [_result(), _result()], ["", "truth"]
        )
        assert no_ref.metrics == []
        assert set(no_ref.metadata["failed_metrics"]) == {"faithfulness"}
        assert "judge throttled" in no_ref.metadata["failed_metrics"]["faithfulness"]
        assert no_ref.metadata["skipped_metrics"] == {
            "answer_correctness": "empty_reference"
        }
        assert {m.metric_type.value: m.value for m in ref.metrics} == {
            "answer_correctness": 0.5,
            "faithfulness": 0.5,
        }
        assert "failed_metrics" not in ref.metadata

    @pytest.mark.parametrize("failing", ["faithfulness", "context_precision"])
    async def test_failed_split_run_keeps_the_other_runs_scores(
        self, mocker, failing: str
    ) -> None:
        ev, _ = _make_evaluator(
            mocker,
            ragas_metrics=[
                EvaluationMetricType.CONTEXT_PRECISION,
                EvaluationMetricType.FAITHFULNESS,
            ],
            ragas_max_contexts=1,
        )

        class _FakeRagasResult:
            def __init__(self, name: str) -> None:
                self.name = name

            def to_pandas(self):
                return pd.DataFrame({self.name: [0.75]})

        def fake_evaluate(*, dataset, metrics, **kwargs):
            name = metrics[0].name
            if name == failing:
                raise RuntimeError("judge timed out")
            return _FakeRagasResult(name)

        mocker.patch.object(rg_module, "evaluate", side_effect=fake_evaluate)
        mocker.patch.object(rg_module.Dataset, "from_dict", side_effect=lambda d: d)
        (report,) = await ev.aevaluate_batch(
            [_query()], [_result(contexts=["c1", "c2"])], ["truth"]
        )
        kept = ({"faithfulness", "context_precision"} - {failing}).pop()
        assert {m.metric_type.value: m.value for m in report.metrics} == {kept: 0.75}
        assert set(report.metadata["failed_metrics"]) == {failing}
        assert "judge timed out" in report.metadata["failed_metrics"][failing]

    async def test_only_reference_metrics_and_no_reference_skips_all(
        self, mocker
    ) -> None:
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.ANSWER_CORRECTNESS]
        )
        evaluate = mocker.patch.object(rg_module, "evaluate")
        reports = await ev.aevaluate_batch([_query()], [_result()], [""])
        evaluate.assert_not_called()
        assert reports[0].metrics == []
        assert reports[0].metadata["skipped_metrics"] == {
            "answer_correctness": "empty_reference"
        }


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
        from unified_kg_rag.shared import EvaluationException

        mocker.patch.object(bedrock_module, "get_assumed_role_boto_session")
        mocker.patch.object(bedrock_module, "BedrockTokenCounter")
        mocker.patch.object(bedrock_module, "BedrockEmbeddingModelFactory")
        mocker.patch.object(
            bedrock_module,
            "BedrockLanguageModelFactory",
            side_effect=RuntimeError("boom"),
        )
        with pytest.raises(EvaluationException):
            RagasEvaluator(config=Config())


class _TemperatureTrackingLLM:
    """Minimal LangChain-like judge exposing a mutable ``temperature``."""

    def __init__(self) -> None:
        self.temperature: float | None = None
        self.seen_temperatures: list[float | None] = []

    async def agenerate_prompt(self, prompts, **_kwargs):
        from langchain_core.outputs import Generation, LLMResult

        self.seen_temperatures.append(self.temperature)
        return LLMResult(generations=[[Generation(text="ok")] for _ in prompts])


class TestJudgeSamplingParams:
    def test_judge_bypasses_temperature_for_unsupported_model(self, mocker) -> None:
        # Default judge (Claude 5) rejects sampling params.
        ev, _ = _make_evaluator(mocker)
        assert ev.ragas_llm is not None
        assert ev.ragas_llm.bypass_temperature is True

    def test_judge_keeps_temperature_for_supporting_model(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker, evaluation_model_id=LanguageModelId.CLAUDE_V4_5_SONNET
        )
        assert ev.ragas_llm is not None
        assert ev.ragas_llm.bypass_temperature is False

    async def test_bypassed_wrapper_never_sets_temperature(self, mocker) -> None:
        from langchain_core.prompt_values import StringPromptValue

        ev, _ = _make_evaluator(mocker)
        assert ev.ragas_llm is not None
        judge = _TemperatureTrackingLLM()
        ev.ragas_llm.langchain_llm = judge  # type: ignore[assignment]

        await ev.ragas_llm.agenerate_text(StringPromptValue(text="q"))

        assert judge.seen_temperatures == [None]
        assert judge.temperature is None


class TestRunConfigAndJudgeEffort:
    def test_default_run_config_suits_thinking_judges(self, mocker) -> None:
        ev, _ = _make_evaluator(mocker)
        rc = ev._build_run_config()
        assert rc.timeout >= 300
        assert rc.timeout == ev.config.evaluation.ragas_timeout
        assert rc.max_workers == ev.config.evaluation.ragas_max_workers
        assert rc.max_retries == ev.config.evaluation.ragas_max_attempts

    def test_run_config_follows_evaluation_config(self, mocker) -> None:
        ev, _ = _make_evaluator(mocker)
        ev.config.evaluation.ragas_timeout = 900
        ev.config.evaluation.ragas_max_workers = 2
        ev.config.evaluation.ragas_max_attempts = 5
        rc = ev._build_run_config()
        assert (rc.timeout, rc.max_workers, rc.max_retries) == (900, 2, 5)

    async def test_evaluate_receives_run_config(self, mocker) -> None:
        ev, _ = _make_evaluator(
            mocker, ragas_metrics=[EvaluationMetricType.FAITHFULNESS]
        )
        ev.config.evaluation.ragas_timeout = 450
        captured = {}

        class _FakeRagasResult:
            def to_pandas(self):
                return pd.DataFrame({"faithfulness": [0.5]})

        def fake_evaluate(**kwargs):
            captured.update(kwargs)
            return _FakeRagasResult()

        mocker.patch.object(rg_module, "evaluate", side_effect=fake_evaluate)
        mocker.patch.object(rg_module.Dataset, "from_dict", side_effect=lambda d: d)

        await ev.aevaluate_batch([_query()], [_result(contexts=["c"])], ["t"])

        assert captured["llm"] is ev.ragas_llm

        assert captured["run_config"].timeout == 450

    def test_judge_model_gets_low_effort_by_default(self, mocker) -> None:
        _make_evaluator(mocker)
        factory = bedrock_module.BedrockLanguageModelFactory.return_value
        _, kwargs = factory.get_model.call_args
        assert kwargs["effort"] == "low"

    def test_null_judge_effort_inherits_bedrock_effort(self, mocker) -> None:
        mocker.patch.object(bedrock_module, "get_assumed_role_boto_session")
        mocker.patch.object(bedrock_module, "BedrockEmbeddingModelFactory")
        mocker.patch.object(bedrock_module, "BedrockLanguageModelFactory")
        mocker.patch.object(bedrock_module, "BedrockTokenCounter")
        config = Config()
        config.evaluation.judge_effort = None
        RagasEvaluator(config=config, show_progress=False)
        factory = bedrock_module.BedrockLanguageModelFactory.return_value
        _, kwargs = factory.get_model.call_args
        assert "effort" not in kwargs


@pytest.mark.parametrize(("preset", "tracked"), [(None, False), ("false", True)])
def test_ragas_telemetry_is_off_unless_the_user_opts_in(
    preset: str | None, tracked: bool
) -> None:
    # Old behaviour: ragas POSTed usage telemetry on every evaluation unless
    # the user knew to set RAGAS_DO_NOT_TRACK=true.
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if k != "RAGAS_DO_NOT_TRACK"}
    if preset is not None:
        env["RAGAS_DO_NOT_TRACK"] = preset
    code = (
        "import unified_kg_rag.evaluation\n"
        "import unified_kg_rag.adapters.evaluators.ragas_evaluator\n"
        "from ragas._analytics import do_not_track\n"
        "print(do_not_track())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=110,
    ).stdout.split()[-1]
    assert out == str(not tracked)
