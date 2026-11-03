# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The comparison's cost block, regression gates and run-eval --from-results
(rebuilt from saved files, no model call)."""

from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
from datetime import datetime, timedelta
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
from unified_kg_rag.domain.models.config import ModelPrice
from unified_kg_rag.evaluation.comparison import (
    cost_comparison,
    paired_comparison,
)
from unified_kg_rag.evaluation.gates import gate_breaches, parse_gate
from unified_kg_rag.evaluation.saved_results import load_saved_runs

pytestmark = pytest.mark.unit

_SCORING = {"answer_span": "first_line", "answer_normalization": "default"}


def _usage(calls: int, tokens_in: int, tokens_out: int, model: str = "m-a") -> Any:
    counts = {
        "llm_calls": calls,
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "cache_hits": 0,
    }
    return {**counts, "by_role": {"query": counts}, "by_model": {model: counts}}


def _run(
    scores: dict[str, float],
    strategy: str,
    *,
    calls: int = 2,
    tokens: int = 100,
    latency: float = 1.0,
    usage: bool = True,
    model: str = "m-a",
) -> tuple[list[EvaluationReport], list[EvaluationResult]]:
    reports = [
        EvaluationReport(
            query_id=qid,
            evaluator_type=EvaluatorType.ANSWER_MATCH,
            metrics=[
                EvaluationMetric(metric_type=EvaluationMetricType.TOKEN_F1, value=v)
            ],
            evaluation_time=datetime(2026, 1, 1),
        )
        for qid, v in scores.items()
    ]
    results = [
        EvaluationResult(
            query_id=qid,
            question="?",
            generated_answer="a",
            ground_truth="",
            search_strategy=strategy,
            metadata={"search_strategy": strategy},
            response_time=latency + i,
            category="1hop" if i % 2 else "2hop",
            usage=_usage(calls, tokens + i, 10, model) if usage else None,
        )
        for i, qid in enumerate(scores)
    ]
    return reports, results


_BASE = {"q1": 0.8, "q2": 0.6, "q3": 0.9, "q4": 0.5}
_WORSE = {"q1": 0.4, "q2": 0.3, "q3": 0.5, "q4": 0.2}


class TestCostBlock:
    def test_per_query_distributions_and_paired_deltas(self) -> None:
        cost = cost_comparison(
            {
                "local": [_run(_BASE, "local", calls=2, tokens=100)],
                "mix": [_run(_BASE, "mix", calls=3, tokens=150, latency=2.0)],
            }
        )
        local = cost["strategies"]["local"]
        assert local["llm_calls_per_query"] == {"mean": 2.0, "p50": 2.0, "p95": 2.0}
        # input 100..103 + output 10 per answer.
        assert local["total_tokens_per_query"]["mean"] == pytest.approx(111.5)
        assert local["latency_seconds"]["p50"] == pytest.approx(2.5)
        assert local["latency_seconds"]["p95"] == pytest.approx(3.85)
        assert local["totals"]["llm_calls"] == 8
        assert local["usd_per_query"] is None and local["usd_total"] is None
        delta = cost["deltas_vs_baseline"]["mix"]
        assert delta["usage/llm_calls"]["mean_delta"] == 1.0
        assert delta["usage/total_tokens"]["mean_delta"] == 50.0
        assert delta["usage/total_tokens"]["ci_low"] == 50.0
        assert delta["latency/response_time_seconds"]["mean_delta"] == 1.0
        assert cost["warnings"] == []

    def test_prices_give_a_usd_estimate_and_unpriced_models_are_named(self) -> None:
        prices = {
            "m-a": ModelPrice(
                input_per_million_tokens=3.0, output_per_million_tokens=15.0
            )
        }
        cost = cost_comparison(
            {
                "local": [_run({"q1": 1.0}, "local", tokens=1000)],
                "mix": [_run({"q1": 1.0}, "mix", tokens=1000, model="m-b")],
            },
            prices=prices,
        )
        usd = 1000 * 3.0 / 1e6 + 10 * 15.0 / 1e6
        assert cost["strategies"]["local"]["usd_per_query"]["mean"] == pytest.approx(
            usd
        )
        assert cost["strategies"]["local"]["usd_total"] == pytest.approx(usd)
        assert cost["strategies"]["mix"]["usd_per_query"] is None
        assert cost["strategies"]["mix"]["usd_total"] is None
        assert "usage/cost_usd" not in cost["deltas_vs_baseline"]["mix"]
        (warning,) = cost["warnings"]
        assert "m-b" in warning and "evaluation.prices" in warning
        assert cost["prices"]["m-a"]["input_per_million_tokens"] == 3.0

    def test_results_saved_before_usage_have_latency_only(self) -> None:
        comparison = paired_comparison(
            {
                "local": [_run(_BASE, "local", usage=False)],
                "mix": [_run(_BASE, "mix", usage=False)],
            }
        )
        cost = comparison["cost"]
        assert cost["strategies"]["local"]["llm_calls_per_query"] is None
        assert list(cost["deltas_vs_baseline"]["mix"]) == [
            "latency/response_time_seconds"
        ]
        assert comparison["warnings"] == []
        # Usage on one side only is not paired, and says so.
        mixed = paired_comparison(
            {
                "local": [_run(_BASE, "local", usage=False)],
                "mix": [_run(_BASE, "mix")],
            }
        )
        (warning,) = mixed["warnings"]
        assert warning.startswith("No usage records in the results of local ")
        assert "usage/llm_calls" not in mixed["cost"]["deltas_vs_baseline"]["mix"]


class TestGates:
    def test_parse(self) -> None:
        gate = parse_gate("answer_match/token_f1:ci_high<-0.02")
        assert (gate.measure, gate.stat, gate.op, gate.value) == (
            "answer_match/token_f1",
            "ci_high",
            "<",
            -0.02,
        )
        assert parse_gate("usage/total_tokens:mean_delta >= 1e3").value == 1000.0
        for bad in ("token_f1<0", "a/b:median<0", "a/b:ci_high<", "a/b:ci_high=0"):
            with pytest.raises(ValueError):
                parse_gate(bad)

    def test_a_supported_regression_breaches_and_missing_data_fails(self) -> None:
        comparison = paired_comparison(
            {
                "local": [_run(_BASE, "local")],
                "mix": [_run(_WORSE, "mix", tokens=400)],
            }
        )
        f1 = parse_gate("answer_match/token_f1:ci_high<-0.02")
        tokens = parse_gate("usage/total_tokens:mean_delta>100")
        absent = parse_gate("retrieval/hit_at_k:ci_high<0")
        breaches = gate_breaches(comparison, [f1, tokens, absent])
        assert len(breaches) == 3
        assert breaches[0].startswith("mix: answer_match/token_f1 ci_high -")
        assert "usage/total_tokens mean_delta +300" in breaches[1]
        assert "no paired delta of 'retrieval/hit_at_k'" in breaches[2]
        assert (
            gate_breaches(comparison, [parse_gate("answer_match/token_f1:ci_low>0")])
            == []
        )


# --- run-eval --from-results / --baseline-results ------------------------------


def _save(
    directory: Path,
    strategy: str,
    run: tuple[list[EvaluationReport], list[EvaluationResult]],
    *,
    stamp: str,
    order: list[str],
    started: datetime,
    scoring: dict[str, Any] | None = None,
) -> None:
    """Write a run's files as EvaluationManager.save_results names them."""
    directory.mkdir(parents=True, exist_ok=True)
    reports, results = run
    stem = f"{strategy}_{stamp}"
    summary = EvaluationSummary(
        total_queries=len(results),
        successful_evaluations=len(results),
        failed_evaluations=0,
        evaluation_start_time=started,
        evaluation_end_time=started + timedelta(seconds=1),
        run_manifest={
            "cli_args": {"search_strategy": order},
            "answer_scoring": scoring or _SCORING,
            "dataset": {"content_sha256": "d" * 64},
            "models": {"answer_generation": "m-a"},
        },
    )
    for kind, data in (
        ("results", [r.model_dump() for r in results]),
        ("reports", [r.model_dump() for r in reports]),
        ("summary", summary.model_dump()),
    ):
        (directory / f"evaluation_{kind}_{stem}.json").write_text(
            json.dumps(data, default=str), encoding="utf-8"
        )


_T0 = datetime(2026, 1, 1, 9, 0, 0)


def _saved_comparison(directory: Path, candidate: dict[str, float]) -> None:
    order = ["local", "mix"]
    _save(
        directory,
        "local",
        _run(_BASE, "local"),
        stamp="20260101_090000_aaaaaaaa",
        order=order,
        started=_T0,
    )
    _save(
        directory,
        "mix",
        _run(candidate, "mix", tokens=300),
        stamp="20260101_090100_bbbbbbbb",
        order=order,
        started=_T0 + timedelta(minutes=1),
    )


def _cli(argv: list[str], config: Config | None = None) -> tuple[int, dict[str, Any]]:
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        [*argv, "--output-format", "json"]
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = asyncio.run(
            run_evaluation.EvaluationRunner(args, None, config or Config()).run()
        )
    out = buffer.getvalue()
    return code, json.loads(out[out.find("{") :])


def test_from_results_rebuilds_the_comparison_without_a_chain(tmp_path: Path) -> None:
    saved = tmp_path / "runs"
    _saved_comparison(saved, _WORSE)
    code, out = _cli(
        ["--from-results", str(saved), "--output-directory", str(tmp_path / "out")]
    )
    assert code == 0
    comparison = out["comparison"]
    expected = paired_comparison(
        {
            "local": [_run(_BASE, "local")],
            "mix": [_run(_WORSE, "mix", tokens=300)],
        },
        categories={"q1": "2hop", "q2": "1hop", "q3": "2hop", "q4": "1hop"},
    )
    assert comparison["baseline"] == "local"
    assert json.dumps(comparison["deltas_vs_baseline"], sort_keys=True) == json.dumps(
        expected["deltas_vs_baseline"], sort_keys=True
    )
    # Categories come back from the results.
    assert set(comparison["deltas_by_category"]["mix"]) == {"1hop", "2hop"}
    assert (
        comparison["cost"]["deltas_vs_baseline"]["mix"]["usage/input_tokens"][
            "mean_delta"
        ]
        == 200.0
    )
    assert comparison["sources"]["from_results"] == [str(saved)]
    assert [r["strategy"] for r in out["runs"]] == ["local", "mix"]
    # The comparison is exported next to its JSON.
    out_dir = tmp_path / "out"
    (csv_path,) = out_dir.glob("evaluation_comparison_*.csv")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    f1 = [r for r in rows if r["measure"] == "answer_match/token_f1"]
    assert {r["query_id"] for r in f1} == set(_BASE)
    q1 = next(r for r in f1 if r["query_id"] == "q1")
    assert (q1["baseline_value"], q1["value"], q1["delta"], q1["category"]) == (
        "0.8",
        "0.4",
        "-0.4",
        "2hop",
    )
    assert any(r["measure"] == "usage/llm_calls" for r in rows)
    (md,) = out_dir.glob("evaluation_comparison_*.md")
    text = md.read_text(encoding="utf-8")
    assert "# Paired comparison vs `local`" in text
    assert "## Cost per query (mean / p50 / p95)" in text


def test_a_regression_gate_exits_1_and_a_clean_run_passes(tmp_path: Path) -> None:
    regressed, clean = tmp_path / "regressed", tmp_path / "clean"
    _saved_comparison(regressed, _WORSE)
    _saved_comparison(clean, _BASE)
    gate = ["--fail-if", "answer_match/token_f1:ci_high<-0.02"]
    code, out = _cli(
        [
            "--from-results",
            str(regressed),
            *gate,
            "--output-directory",
            str(tmp_path / "o1"),
        ]
    )
    assert code == 1
    (failure,) = out["gate_failures"]
    assert failure.startswith("mix: answer_match/token_f1 ci_high")
    assert out["comparison"]["gates"]["conditions"] == [
        "answer_match/token_f1:ci_high<-0.02"
    ]
    code, out = _cli(
        [
            "--from-results",
            str(clean),
            *gate,
            "--output-directory",
            str(tmp_path / "o2"),
        ]
    )
    assert (code, out["gate_failures"]) == (0, [])


def test_strategy_order_and_selection(tmp_path: Path) -> None:
    saved = tmp_path / "runs"
    _saved_comparison(saved, _WORSE)
    code, out = _cli(
        [
            "--from-results",
            str(saved),
            "--search-strategy",
            "mix,local",
            "--output-directory",
            str(tmp_path / "o"),
        ]
    )
    assert (code, out["comparison"]["baseline"]) == (0, "mix")
    with pytest.raises(
        run_evaluation.InvalidDatasetError, match="no saved results for strategy naive"
    ):
        _cli(["--from-results", str(saved), "--search-strategy", "local,naive"])


def test_baseline_results_compare_and_refuse_other_answer_scoring(
    tmp_path: Path,
) -> None:
    earlier, later = tmp_path / "earlier", tmp_path / "later"
    _save(
        earlier,
        "local",
        _run(_BASE, "local"),
        stamp="20260101_090000_aaaaaaaa",
        order=["local"],
        started=_T0,
    )
    _save(
        later,
        "local",
        _run(_WORSE, "local"),
        stamp="20260102_090000_cccccccc",
        order=["local"],
        started=_T0 + timedelta(days=1),
    )
    code, out = _cli(
        [
            "--from-results",
            str(later),
            "--baseline-results",
            str(earlier),
            "--fail-if",
            "answer_match/token_f1:mean_delta<-0.1",
            "--output-directory",
            str(tmp_path / "o"),
        ]
    )
    comparison = out["comparison"]
    assert comparison["baseline"] == "baseline:local"
    assert list(comparison["deltas_vs_baseline"]) == ["local"]
    assert code == 1

    other = tmp_path / "other-span"
    _save(
        other,
        "local",
        _run(_WORSE, "local"),
        stamp="20260103_090000_dddddddd",
        order=["local"],
        started=_T0,
        scoring={"answer_span": "full", "answer_normalization": "default"},
    )
    with pytest.raises(
        run_evaluation.InvalidDatasetError, match="different answer_scoring"
    ):
        _cli(["--from-results", str(other), "--baseline-results", str(earlier)])


def test_saved_runs_need_their_reports(tmp_path: Path) -> None:
    saved = tmp_path / "runs"
    _saved_comparison(saved, _WORSE)
    next(saved.glob("evaluation_reports_mix_*")).unlink()
    with pytest.raises(Exception, match="has no reports file"):
        load_saved_runs([saved])
    with pytest.raises(Exception, match="holds no evaluation_results"):
        load_saved_runs([tmp_path])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--from-results", "d", "--repeats", "2"], "--repeats does not apply"),
        (
            ["--data-path", "x.json", "--fail-if", "a/b:ci_high<0"],
            "--fail-if needs a comparison",
        ),
    ],
)
def test_usage_errors(argv: list[str], message: str, capsys) -> None:
    args = run_evaluation.CommandLineInterface().parser.parse_args(argv)
    with pytest.raises(SystemExit) as exc:
        run_evaluation.EvaluationRunner(args, None, Config())
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--data-path", "x.json", "--from-results", "d"], "not allowed with"),
        ([], "one of the arguments --data-path --from-results is required"),
    ],
)
def test_one_input_is_required(argv: list[str], message: str, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        run_evaluation.CommandLineInterface().parser.parse_args(argv)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_from_results_rejudges_with_another_model(tmp_path: Path, monkeypatch) -> None:
    saved = tmp_path / "runs"
    _saved_comparison(saved, _WORSE)
    judged: list[str] = []

    class _Judge:
        def __init__(self, config: Config, providers: Any = None) -> None:
            judged.append(config.model_id(run_evaluation.ModelRole.EVALUATION))

        async def acompare(self, baseline: Any, candidate: Any) -> dict[str, Any]:
            return {
                "judged": len(candidate),
                "failed": 0,
                "criteria": {},
                "candidate_errors": 0,
                "baseline_errors": 0,
            }

    monkeypatch.setattr(run_evaluation, "PairwiseJudge", _Judge)
    code, out = _cli(
        [
            "--from-results",
            str(saved),
            "--pairwise-judge",
            "--judge-model",
            "vendor.judge-v2",
            "--output-directory",
            str(tmp_path / "o"),
        ]
    )
    assert code == 0 and judged == ["vendor.judge-v2"]
    comparison = out["comparison"]
    assert comparison["pairwise_judge_model"] == "vendor.judge-v2"
    assert comparison["pairwise_vs_baseline"]["mix"]["judged"] == 4
    # The saved answers came from m-a: not the judge, so no self-preference.
    assert not any("judge is the answer model" in w for w in comparison["warnings"])
