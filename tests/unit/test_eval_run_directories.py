# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where ``run-eval`` writes its runs, and which saved runs it pools.

Every invocation wrote its runs flat into one directory, and
``--from-results`` pooled every run of a strategy found there as its
repeats: two unrelated invocations became one strategy's "repeats", with
exit 0. Each invocation now writes into a run subdirectory of its own, and
saved runs of one strategy from several invocations are refused unless
pooled explicitly (``--pool-runs``). All fakes, no AWS.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from unified_kg_rag.application.cli import run_evaluation
from unified_kg_rag.domain.models import (
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
    EvaluatorType,
)
from unified_kg_rag.evaluation.saved_results import (
    SavedRun,
    group_runs,
    load_saved_runs,
    pooling_errors,
)


class _Chain:
    async def ainvoke(self, inputs: dict[str, Any], config: Any = None) -> Any:
        meta = {"search_strategy": inputs.get("search_strategy") or "local"}
        return {"answer": "Vendor", "sources": [], "metadata": meta}


@pytest.fixture
def eval_config(mocker) -> Config:
    mocker.patch.object(run_evaluation, "display_ascii_art")
    config = Config()
    config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
    return config


def _dataset(tmp_path: Path) -> Path:
    data = tmp_path / "eval.json"
    data.write_text(
        json.dumps(
            [{"question": f"Who ships {i}?", "answer": "Vendor"} for i in range(12)]
        )
    )
    return data


async def _run_eval(config: Config, *argv: str) -> run_evaluation.EvaluationRunner:
    args = run_evaluation.CommandLineInterface().parser.parse_args(list(argv))
    runner = run_evaluation.EvaluationRunner(
        args, None if "--from-results" in argv else _Chain(), config
    )
    assert await runner.run() == 0
    return runner


async def test_each_invocation_writes_into_its_own_run_subdirectory(
    eval_config, tmp_path
) -> None:
    data, out = _dataset(tmp_path), tmp_path / "out"
    first = await _run_eval(
        eval_config, "--data-path", str(data), "--output-directory", str(out)
    )
    second = await _run_eval(
        eval_config, "--data-path", str(data), "--output-directory", str(out)
    )
    assert first.run_id != second.run_id
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [first.run_id, second.run_id]
    )
    (summary,) = (out / first.run_id).glob("evaluation_summary_*.json")
    saved = json.loads(summary.read_text())
    assert saved["run_manifest"]["run_id"] == first.run_id
    assert saved["configuration"]["outputs_directory"] == str(out / first.run_id)
    assert not list(out.glob("evaluation_*"))


async def test_run_subdirectories_off_writes_into_the_directory(
    eval_config, tmp_path
) -> None:
    eval_config.evaluation.run_subdirectories = False
    data, out = _dataset(tmp_path), tmp_path / "out"
    await _run_eval(
        eval_config, "--data-path", str(data), "--output-directory", str(out)
    )
    assert list(out.glob("evaluation_summary_*.json"))
    assert not [p for p in out.iterdir() if p.is_dir()]


async def test_from_results_refuses_one_strategy_from_two_invocations(
    eval_config, tmp_path
) -> None:
    # The reproduction: two unrelated runs of local in one directory were
    # pooled as its repeats, exit 0.
    data, out = _dataset(tmp_path), tmp_path / "out"
    common = ["--data-path", str(data), "--output-directory", str(out)]
    await _run_eval(eval_config, *common, "--search-strategy", "local,mix")
    await _run_eval(eval_config, *common, "--search-strategy", "local")
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        ["--from-results", str(out), "--output-directory", str(tmp_path / "o")]
    )
    runner = run_evaluation.EvaluationRunner(args, None, eval_config)
    with pytest.raises(
        run_evaluation.InvalidDatasetError,
        match=r"2 saved runs of local come from 2 or more run-eval invocations"
        r".*--pool-runs",
    ):
        await runner.run()
    # One invocation's subdirectory compares as it ran.
    (first,) = [
        p for p in out.iterdir() if len(list(p.glob("evaluation_results_*"))) == 2
    ]
    await _run_eval(
        eval_config,
        "--from-results",
        str(first),
        "--output-directory",
        str(tmp_path / "o"),
    )
    # Pooled explicitly: compared, the strategies' repeats differing.
    pooled = await _run_eval(
        eval_config,
        "--from-results",
        str(out),
        "--pool-runs",
        "--output-directory",
        str(tmp_path / "p"),
    )
    (comparison,) = (tmp_path / "p" / pooled.run_id).glob(
        "evaluation_comparison_*.json"
    )
    assert json.loads(comparison.read_text())["strategies"]["local"]["repeats"] == 2


async def test_baseline_results_refuse_pooled_invocations_before_answering(
    eval_config, tmp_path, mocker
) -> None:
    data, ci = _dataset(tmp_path), tmp_path / "ci"
    common = ["--data-path", str(data), "--output-directory", str(ci)]
    await _run_eval(eval_config, *common, "--search-strategy", "local")
    await _run_eval(eval_config, *common, "--search-strategy", "local")
    answer = mocker.spy(_Chain, "ainvoke")
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        [
            "--data-path",
            str(data),
            "--search-strategy",
            "local",
            "--baseline-results",
            str(ci),
            "--output-directory",
            str(tmp_path / "now"),
        ]
    )
    with pytest.raises(run_evaluation.InvalidDatasetError, match="--baseline-results"):
        await run_evaluation.EvaluationRunner(args, _Chain(), eval_config).run()
    answer.assert_not_called()


def _saved(strategy: str, stamp: str, **manifest: Any) -> SavedRun:
    result = EvaluationResult(
        query_id="q1", question="?", generated_answer="a", ground_truth=""
    )
    report = EvaluationReport(
        query_id="q1",
        evaluator_type=EvaluatorType.ANSWER_MATCH,
        metrics=[EvaluationMetric(metric_type=EvaluationMetricType.TOKEN_F1, value=1)],
        evaluation_time=datetime(2026, 1, 1),
    )
    summary = EvaluationSummary(
        total_queries=1,
        successful_evaluations=1,
        failed_evaluations=0,
        evaluation_start_time=datetime(2026, 1, 1),
        evaluation_end_time=datetime(2026, 1, 1),
        run_manifest=manifest,
    )
    return SavedRun(
        path=Path(f"evaluation_results_{strategy}_{stamp}.json"),
        strategy=strategy,
        stamp=stamp,
        results=[result],
        reports=[report],
        summary=summary,
    )


@pytest.mark.parametrize(
    ("manifests", "refused"),
    [
        # One invocation's three repeats.
        ([{"run_id": "r1"}] * 3, False),
        ([{"run_id": "r1"}, {"run_id": "r2"}], True),
        # A run saved before run_id was recorded next to one that has it.
        ([{"run_id": "r1"}, {}], True),
        # Saved before run_id: more runs than their --repeats.
        ([{"cli_args": {"repeats": 1}}] * 2, True),
        ([{"cli_args": {"repeats": 2}}] * 2, False),
        # Nothing to tell them apart by: pooled, as before.
        ([{}, {}], False),
    ],
)
def test_pooling_errors(manifests: list[dict[str, Any]], refused: bool) -> None:
    runs = [
        _saved("local", f"20260101_00000{i}_0000000{i}", **m)
        for i, m in enumerate(manifests)
    ]
    errors = pooling_errors(group_runs(runs))
    assert bool(errors) is refused


def test_a_directory_with_run_subdirectories_reads_them_all(tmp_path) -> None:
    out = tmp_path / "out"
    (out / "run_a").mkdir(parents=True)
    (out / "run_b").mkdir()
    (out / "notes").mkdir()
    for directory in ("run_a", "run_b"):
        run = _saved("local", "20260101_000000_00000000")
        stem = "local_20260101_000000_00000000"
        target = out / directory
        target.joinpath(f"evaluation_results_{stem}.json").write_text(
            json.dumps([r.model_dump(mode="json") for r in run.results])
        )
        target.joinpath(f"evaluation_reports_{stem}.json").write_text(
            json.dumps([r.model_dump(mode="json") for r in run.reports])
        )
    assert {run.path.parent.name for run in load_saved_runs([out])} == {
        "run_a",
        "run_b",
    }
