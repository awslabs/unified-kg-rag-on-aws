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
    LLMUsage,
)
from unified_kg_rag.domain.models.config import (
    DEFAULT_MODEL_ID,
    JUDGE_MODEL_ID,
    ModelPrice,
)
from unified_kg_rag.evaluation import gates as gates_module
from unified_kg_rag.evaluation.comparison import (
    adjust_for_multiplicity,
    answer_cost_usd,
    cost_comparison,
    paired_comparison,
)
from unified_kg_rag.evaluation.gates import gate_breaches, parse_gate
from unified_kg_rag.evaluation.report import comparison_markdown
from unified_kg_rag.evaluation.saved_results import (
    check_answer_scoring,
    load_saved_runs,
    rerank_unmetered,
)
from unified_kg_rag.shared import EvaluationException

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


@pytest.fixture
def small_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gates read the 4-query fixtures below (a gate needs 10 pairs)."""
    monkeypatch.setattr(gates_module, "MIN_GATE_PAIRS", 1)


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

    def test_cache_tokens_are_priced_at_their_own_rates(self) -> None:
        # input_tokens includes the cache reads and writes; they were priced
        # at the input rate, over-stating a cached run's cost.
        from unified_kg_rag.domain.models import LLMUsage

        counts = {
            "llm_calls": 1,
            "input_tokens": 1_000_000,
            "output_tokens": 100_000,
            "cache_read_input_tokens": 600_000,
            "cache_write_input_tokens": 100_000,
        }
        used = LLMUsage(**counts, by_model={"m-a": counts})
        rates = {"input_per_million_tokens": 2.0, "output_per_million_tokens": 10.0}
        cached = ModelPrice(
            **rates,
            cache_read_per_million_tokens=0.2,
            cache_write_per_million_tokens=2.5,
        )
        usd, unpriced = answer_cost_usd(used, {"m-a": cached})
        assert unpriced == set()
        assert usd == pytest.approx(0.3 * 2.0 + 0.6 * 0.2 + 0.1 * 2.5 + 0.1 * 10.0)
        # Without cache rates every input token is priced as input, as before.
        plain, _ = answer_cost_usd(used, {"m-a": ModelPrice(**rates)})
        assert plain == pytest.approx(1.0 * 2.0 + 0.1 * 10.0)

    def test_rerank_search_units_are_priced_and_an_unpriced_reranker_named(
        self,
    ) -> None:
        from unified_kg_rag.domain.models import LLMUsage

        llm = {"llm_calls": 1, "input_tokens": 1_000_000, "output_tokens": 0}
        rerank = {"rerank_calls": 3, "rerank_search_units": 4}
        used = LLMUsage(**llm, **rerank, by_model={"m-a": llm, "reranker": rerank})
        llm_price = ModelPrice(
            input_per_million_tokens=2.0, output_per_million_tokens=10.0
        )
        usd, unpriced = answer_cost_usd(
            used,
            {
                "m-a": llm_price,
                "reranker": ModelPrice(per_thousand_rerank_search_units=2.0),
            },
        )
        assert unpriced == set()
        assert usd == pytest.approx(2.0 + 4 * 2.0 / 1000)
        # A reranker without a price leaves the answer unpriced, as a model.
        assert answer_cost_usd(used, {"m-a": llm_price}) == (None, {"reranker"})

    def test_no_usd_delta_against_a_run_that_metered_no_rerank(self) -> None:
        """Regression: a baseline saved before rerank requests were metered
        priced none, so the USD delta credited the strategy with its
        rerank cost."""
        from unified_kg_rag.domain.models import LLMUsage

        prices = {
            "m-a": ModelPrice(
                input_per_million_tokens=3.0, output_per_million_tokens=15.0
            ),
            "reranker": ModelPrice(per_thousand_rerank_search_units=2.0),
        }
        old = _run({"q1": 1.0}, "local", tokens=1000)
        new = _run({"q1": 1.0}, "mix", tokens=1000)
        llm = new[1][0].usage.by_model["m-a"].model_dump()
        rerank = {"rerank_calls": 1, "rerank_search_units": 1}
        new[1][0].usage = LLMUsage(
            **{**llm, **rerank},
            by_model={"m-a": llm, "reranker": rerank},
        )
        runs = {"baseline:local": [old], "mix": [new]}
        both = cost_comparison(runs, prices=prices)
        assert "usage/cost_usd" in both["deltas_vs_baseline"]["mix"]
        cost = cost_comparison(runs, prices=prices, unmetered_rerank={"baseline:local"})
        assert "usage/cost_usd" not in cost["deltas_vs_baseline"]["mix"]
        # The other measures still pair, and each side keeps its own USD.
        assert "usage/llm_calls" in cost["deltas_vs_baseline"]["mix"]
        assert cost["strategies"]["mix"]["usd_total"] is not None
        (warning,) = cost["warnings"]
        assert "baseline:local" in warning and "usage_schema 2" in warning
        # Every side unmetered: like with like, no warning.
        same = cost_comparison(
            runs, prices=prices, unmetered_rerank={"baseline:local", "mix"}
        )
        assert "usage/cost_usd" in same["deltas_vs_baseline"]["mix"]
        assert same["warnings"] == []

    def test_runs_without_a_usage_schema_meter_no_rerank(self) -> None:
        assert rerank_unmetered(
            [
                ("old", {"answer_scoring": None}),
                ("v1", {"usage_schema": 1}),
                ("now", {"usage_schema": 2}),
                ("unknown", None),
            ]
        ) == {"old", "v1"}

    def test_a_price_entry_must_name_what_it_prices(self) -> None:
        from pydantic import ValidationError

        # A missing token price never costs $0: it is refused at load.
        for entry in (
            {},
            {"input_per_million_tokens": 1.0},
            {"cache_read_per_million_tokens": 0.1},
        ):
            with pytest.raises(ValidationError, match="model price sets"):
                ModelPrice(**entry)
        assert ModelPrice(input_per_million_tokens=1.0, output_per_million_tokens=2.0)
        assert ModelPrice(per_thousand_rerank_search_units=2.0)
        # A rerank-only entry does not price a language model's calls.
        from unified_kg_rag.domain.models import LLMUsage

        llm = {"llm_calls": 1, "input_tokens": 10, "output_tokens": 1}
        used = LLMUsage(**llm, by_model={"m-a": llm})
        rerank_only = {"m-a": ModelPrice(per_thousand_rerank_search_units=2.0)}
        assert answer_cost_usd(used, rerank_only) == (None, {"m-a"})

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

    def test_a_supported_regression_breaches_and_missing_data_fails(
        self, small_gates: None
    ) -> None:
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

    def test_a_delta_over_too_few_pairs_fails_every_gate(self) -> None:
        # One pair gives a zero-width interval at its own delta, so
        # ci_high<-0.02 passed silently on a single improved query.
        one = paired_comparison(
            {
                "local": [_run({"q1": 0.5}, "local")],
                "mix": [_run({"q1": 0.9}, "mix")],
            }
        )
        delta = one["deltas_vs_baseline"]["mix"]["answer_match/token_f1"]
        assert delta["ci_low"] == delta["ci_high"]
        gates = [
            parse_gate("answer_match/token_f1:ci_high<-0.02"),
            parse_gate("usage/total_tokens:mean_delta>500"),
        ]
        breaches = gate_breaches(one, gates)
        assert len(breaches) == 2
        assert "pairs 1 queries, fewer than the 10 a gate needs" in breaches[0]
        assert "'usage/total_tokens' delta pairs 1 queries" in breaches[1]
        ten = {f"q{i}": 0.5 for i in range(10)}
        enough = paired_comparison(
            {"local": [_run(ten, "local")], "mix": [_run(ten, "mix")]}
        )
        assert gate_breaches(enough, gates) == []

    @staticmethod
    def _per_strategy_baselines(**baseline_mix: Any) -> dict[str, Any]:
        """A --baseline-results comparison: mix pairs with baseline:mix, not
        with the first baseline (``baseline``), which is clean."""
        delta = {"pairs": 20, "mean_delta": 0.0, "ci_low": -0.1, "ci_high": 0.1}
        clean = {"answers_without_usage": 0, "failed_answers": 0}
        return {
            "baseline": "baseline:local",
            "baselines": {"local": "baseline:local", "mix": "baseline:mix"},
            "strategies": {
                "baseline:local": {"failed_metrics": {}},
                "baseline:mix": {
                    "failed_metrics": baseline_mix.pop("failed_metrics", {})
                },
                "mix": {"failed_metrics": {}},
            },
            "deltas_vs_baseline": {"mix": {"answer_match/token_f1": dict(delta)}},
            "cost": {
                "strategies": {
                    "baseline:local": dict(clean),
                    "baseline:mix": {**clean, **baseline_mix},
                    "mix": dict(clean),
                },
                "deltas_vs_baseline": {"mix": {"usage/total_tokens": dict(delta)}},
            },
        }

    def test_each_check_reads_the_strategys_own_baseline(self) -> None:
        # Each check read the strategy's baseline from "baselines", falling
        # back to the first one; reading the first one always passed here.
        tokens = [parse_gate("usage/total_tokens:mean_delta>1000")]
        f1 = [parse_gate("answer_match/token_f1:ci_high<-0.5")]
        unmeasured = self._per_strategy_baselines(answers_without_usage=3)
        (breach,) = gate_breaches(unmeasured, tokens)
        assert "answers without a value of 'usage/total_tokens' (baseline:mix 3)" in (
            breach
        )
        failed = self._per_strategy_baselines(failed_answers=2)
        (breach,) = gate_breaches(failed, tokens)
        assert "failed answers differ (baseline:mix 2, mix 0)" in breach
        scored = self._per_strategy_baselines(
            failed_metrics={"answer_match/token_f1": 2}
        )
        (breach,) = gate_breaches(scored, f1)
        assert "different numbers of answers (baseline:mix 2, mix 0)" in breach
        clean = self._per_strategy_baselines()
        assert gate_breaches(clean, [*tokens, *f1]) == []


def _judged(
    scores: dict[str, float | None],
) -> tuple[list[EvaluationReport], list[EvaluationResult]]:
    """One ragas run; a ``None`` score is a judge that failed on the query."""
    reports = [
        EvaluationReport(
            query_id=qid,
            evaluator_type=EvaluatorType.RAGAS,
            metrics=(
                []
                if value is None
                else [
                    EvaluationMetric(
                        metric_type=EvaluationMetricType.FAITHFULNESS, value=value
                    )
                ]
            ),
            metadata=(
                {"failed_metrics": {"faithfulness": "uncomputable (NaN)"}}
                if value is None
                else {}
            ),
        )
        for qid, value in scores.items()
    ]
    results = [
        EvaluationResult(
            query_id=qid, question="?", generated_answer="a", ground_truth="g"
        )
        for qid in scores
    ]
    return reports, results


class TestFailedScores:
    """A judge failure leaves the query out of the metric's delta."""

    def _comparison(self) -> dict[str, Any]:
        queries = [f"q{i}" for i in range(12)]
        # The judge fails on the four queries where mix would score 0.0.
        candidate = {q: (None if i < 4 else 0.7) for i, q in enumerate(queries)}
        return paired_comparison(
            {
                "local": [_judged(dict.fromkeys(queries, 0.6))],
                "mix": [_judged(candidate)],
            }
        )

    def test_counted_warned_and_fail_the_metric_gates(self) -> None:
        comparison = self._comparison()
        assert comparison["strategies"]["mix"]["failed_metrics"] == {
            "ragas/faithfulness": 4
        }
        assert comparison["strategies"]["local"]["failed_metrics"] == {}
        delta = comparison["deltas_vs_baseline"]["mix"]["ragas/faithfulness"]
        assert (delta["pairs"], delta["unpaired"]) == (8, 4)
        (warning,) = comparison["warnings"]
        assert warning.startswith(
            "ragas/faithfulness could not be scored on different numbers of "
            "answers (local 0, mix 4)"
        )
        # Before, both gates passed on the 8 queries the judge scored.
        for condition in (
            "ragas/faithfulness:ci_high<0",
            "ragas/faithfulness:mean_delta<0",
        ):
            (breach,) = gate_breaches(comparison, [parse_gate(condition)])
            assert "could not be scored on different numbers" in breach
        # A cost gate does not read the metric's failures.
        (cost,) = gate_breaches(
            comparison, [parse_gate("latency/response_time_seconds:mean_delta>1")]
        )
        assert "could not be scored" not in cost

    def test_the_tables_show_pairs_and_unpaired(self) -> None:
        text = comparison_markdown(self._comparison())
        assert "+0.100 [+0.100, +0.100] (8 pairs, 4 unpaired)" in text


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
    answer_model: str = "m-a",
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
            "models": {"answer_generation": answer_model},
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


def _saved_comparison(
    directory: Path, candidate: dict[str, float], answer_model: str = "m-a"
) -> None:
    order = ["local", "mix"]
    _save(
        directory,
        "local",
        _run(_BASE, "local"),
        stamp="20260101_090000_aaaaaaaa",
        order=order,
        started=_T0,
        answer_model=answer_model,
    )
    _save(
        directory,
        "mix",
        _run(candidate, "mix", tokens=300),
        stamp="20260101_090100_bbbbbbbb",
        order=order,
        started=_T0 + timedelta(minutes=1),
        answer_model=answer_model,
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
    expected = adjust_for_multiplicity(
        paired_comparison(
            {
                "local": [_run(_BASE, "local")],
                "mix": [_run(_WORSE, "mix", tokens=300)],
            },
            categories={"q1": "2hop", "q2": "1hop", "q3": "2hop", "q4": "1hop"},
        )
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
    (csv_path,) = out_dir.rglob("evaluation_comparison_*.csv")
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
    (md,) = out_dir.rglob("evaluation_comparison_*.md")
    text = md.read_text(encoding="utf-8")
    assert "# Paired comparison vs `local`" in text
    assert "## Cost per query (mean / p50 / p95)" in text


def test_a_regression_gate_exits_1_and_a_clean_run_passes(
    tmp_path: Path, small_gates: None
) -> None:
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


@pytest.mark.parametrize(
    ("question", "refused"),
    [("Who ships the widgets?", False), ("누가 위젯을 배송하나?", True)],
    ids=["english", "korean"],
)
async def test_a_baseline_is_checked_against_the_dataset_before_any_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, question: str, refused: bool
) -> None:
    """A baseline saved before token_f1_tokens was recorded (whitespace
    tokens) is refused only where the dataset holds Han, Hangul or Kana
    text, before any answer is generated."""
    earlier = tmp_path / "earlier"
    _save(
        earlier,
        "local",
        _run(_BASE, "local"),
        stamp="20260101_090000_aaaaaaaa",
        order=["local"],
        started=_T0,
    )
    data = tmp_path / "eval.json"
    data.write_text(
        json.dumps([{"id": "q1", "question": question, "answer": "Vendor"}]),
        encoding="utf-8",
    )
    answered: list[str] = []

    class _Chain:
        async def ainvoke(self, inputs: dict, config: Any = None) -> dict:
            answered.append(inputs["query"])
            meta = {"search_strategy": "local", "processing_time": 0.1}
            return {"answer": "Vendor", "sources": [], "metadata": meta}

    monkeypatch.setattr(run_evaluation, "display_ascii_art", lambda **_: None)
    config = Config()
    config.evaluation.enabled_evaluators = [EvaluatorType.ANSWER_MATCH]
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        [
            "--data-path",
            str(data),
            "--baseline-results",
            str(earlier),
            "--search-strategy",
            "local",
            "--output-directory",
            str(tmp_path / "out"),
        ]
    )
    runner = run_evaluation.EvaluationRunner(args, _Chain(), config)  # type: ignore[arg-type]
    if refused:
        with pytest.raises(
            run_evaluation.InvalidDatasetError, match="token_f1_tokens 'whitespace'"
        ):
            await runner.run()
        assert answered == []
    else:
        await runner.run()
        assert question in answered


def test_saved_runs_need_their_reports(tmp_path: Path) -> None:
    saved = tmp_path / "runs"
    _saved_comparison(saved, _WORSE)
    next(saved.rglob("evaluation_reports_mix_*")).unlink()
    with pytest.raises(Exception, match="has no reports file"):
        load_saved_runs([saved])
    empty = tmp_path / "empty"
    (empty / "run_1").mkdir(parents=True)
    with pytest.raises(Exception, match="holds no evaluation_results"):
        load_saved_runs([empty])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--from-results", "d", "--repeats", "2"], "--repeats does not apply"),
        (
            ["--from-results", "d", "--evaluators", "retrieval"],
            "--evaluators does not apply",
        ),
        (
            ["--from-results", "d", "--run-order", "sequential"],
            "--run-order does not apply",
        ),
        (
            ["--from-results", "d", "--repeat-extractions", "reuse"],
            "--repeat-extractions does not apply",
        ),
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


class _RecordingJudge:
    judged: list[str] = []

    def __init__(self, config: Config, providers: Any = None) -> None:
        type(self).judged.append(config.model_id(run_evaluation.ModelRole.EVALUATION))

    async def acompare(self, baseline: Any, candidate: Any) -> dict[str, Any]:
        return {
            "judged": len(candidate),
            "failed": 0,
            "criteria": {},
            "candidate_errors": 0,
            "baseline_errors": 0,
        }


@pytest.mark.parametrize("judge_wrote", ["candidates", "baseline"])
def test_the_default_judge_falls_back_from_the_saved_answer_model(
    tmp_path: Path, monkeypatch, judge_wrote: str
) -> None:
    # Saved answers came from the default judge's model: judging them with it
    # would favour them. A configured judge is kept (the test above).
    judge_model = f"global.{JUDGE_MODEL_ID}"
    saved, earlier = tmp_path / "runs", tmp_path / "earlier"
    args = ["--from-results", str(saved), "--pairwise-judge"]
    if judge_wrote == "candidates":
        _saved_comparison(saved, _WORSE, answer_model=judge_model)
    else:
        _saved_comparison(saved, _WORSE)
        _save(
            earlier,
            "local",
            _run(_BASE, "local"),
            stamp="20251231_090000_eeeeeeee",
            order=["local"],
            started=_T0 - timedelta(days=1),
            answer_model=judge_model,
        )
        args += ["--baseline-results", str(earlier)]
    monkeypatch.setattr(_RecordingJudge, "judged", [])
    monkeypatch.setattr(run_evaluation, "PairwiseJudge", _RecordingJudge)
    code, out = _cli([*args, "--output-directory", str(tmp_path / "o")])
    assert code == 0 and _RecordingJudge.judged == [DEFAULT_MODEL_ID]
    assert out["comparison"]["pairwise_judge_model"] == DEFAULT_MODEL_ID


def test_a_re_judge_without_the_pairwise_judge_names_no_judge(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # --from-results answers nothing, so the ragas/assertion judges score
    # nothing: no LLM judge is chosen or logged.
    import logging

    saved = tmp_path / "runs"
    _saved_comparison(saved, _WORSE)
    config = Config()
    config.evaluation.enabled_evaluators = [EvaluatorType.RAGAS]
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        ["--from-results", str(saved)]
    )
    with caplog.at_level(logging.INFO):
        run_evaluation.EvaluationRunner(args, None, config)
    assert not any("LLM judge" in r.getMessage() for r in caplog.records)


# --- per-strategy baselines, failed answers in cost, pooling checks ---------


def _saved_strategies(
    directory: Path, runs: dict[str, dict[str, float]], day: int = 0, **manifest: Any
) -> None:
    """Save one run per strategy, listed in that order."""
    order = list(runs)
    before = set(directory.rglob("evaluation_summary_*.json"))
    for i, (strategy, scores) in enumerate(runs.items()):
        _save(
            directory,
            strategy,
            _run(scores, strategy),
            stamp=f"2026010{day + 1}_09{i:02d}00_{'abcdef01'[i] * 8}",
            order=order,
            started=_T0 + timedelta(days=day, minutes=i),
        )
    if manifest:
        for path in set(directory.rglob("evaluation_summary_*.json")) - before:
            data = json.loads(path.read_text(encoding="utf-8"))
            data["run_manifest"].update(manifest)
            path.write_text(json.dumps(data), encoding="utf-8")


def test_baseline_results_pair_each_strategy_with_its_own(
    tmp_path: Path, small_gates: None
) -> None:
    # Before, every strategy was compared with the baseline's first one, so
    # a regression of mix against the earlier mix passed the gate.
    earlier, later = tmp_path / "earlier", tmp_path / "later"
    _saved_strategies(earlier, {"local": _BASE, "mix": _BASE})
    _saved_strategies(later, {"local": _BASE, "mix": _WORSE, "naive": _BASE}, day=1)
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
    assert comparison["baselines"] == {
        "local": "baseline:local",
        "mix": "baseline:mix",
        "naive": "baseline:local",
    }
    assert list(comparison["strategies"]) == [
        "baseline:local",
        "local",
        "baseline:mix",
        "mix",
        "naive",
    ]
    f1 = "answer_match/token_f1"
    assert comparison["deltas_vs_baseline"]["local"][f1]["mean_delta"] == 0.0
    assert comparison["deltas_vs_baseline"]["mix"][f1]["mean_delta"] == (
        pytest.approx(-0.35)
    )
    assert set(comparison["cost"]["deltas_vs_baseline"]) == {"local", "mix", "naive"}
    assert (code, out["gate_failures"]) == (
        1,
        ["mix: answer_match/token_f1 mean_delta -0.35 < -0.1"],
    )
    # naive has no earlier run: the first strategy's, and a warning.
    (warning,) = [w for w in comparison["warnings"] if "has no run of" in w]
    assert "naive" in warning and "baseline:local" in warning
    (md,) = (tmp_path / "o").rglob("evaluation_comparison_*.md")
    text = md.read_text(encoding="utf-8")
    assert "vs each strategy's baseline" in text
    assert "mix - baseline:mix [95% CI]" in text
    rows = list(
        csv.DictReader(next((tmp_path / "o").rglob("*.csv")).open(encoding="utf-8"))
    )
    assert {(r["strategy"], r["baseline"]) for r in rows} == {
        ("local", "baseline:local"),
        ("mix", "baseline:mix"),
        ("naive", "baseline:local"),
    }


def _failing_run(
    scores: dict[str, float], strategy: str, failed: dict[str, tuple[Any, Any]]
) -> tuple[list[EvaluationReport], list[EvaluationResult]]:
    """``_run`` with the answers of ``failed`` failed: (usage, latency) each."""
    reports, results = _run(scores, strategy)
    reports = [r for r in reports if r.query_id not in failed]
    for result in results:
        if result.query_id in failed:
            usage, latency = failed[result.query_id]
            result.error = True
            result.usage = LLMUsage.model_validate(usage) if usage else None
            result.response_time = latency
    return reports, results


class TestFailedAnswersInCost:
    def test_failed_answers_count_with_what_they_spent(self) -> None:
        # Before, a timed-out answer (usage and time spent) was left out, so a
        # strategy that timed out looked faster.
        timed_out = _failing_run(_BASE, "mix", {"q1": (_usage(2, 100, 10), 30.0)})
        cost = cost_comparison({"local": [_run(_BASE, "local")], "mix": [timed_out]})
        mix = cost["strategies"]["mix"]
        assert (mix["failed_answers"], mix["answers_without_latency"]) == (1, 0)
        latency = cost["deltas_vs_baseline"]["mix"]["latency/response_time_seconds"]
        assert latency["pairs"] == 4 and latency["mean_delta"] == pytest.approx(7.25)
        (warning,) = [w for w in cost["warnings"] if w.startswith("Failed answers")]
        assert "mix 1/4" in warning

    def test_answers_without_measures_are_warned_and_fail_the_cost_gates(
        self, small_gates: None
    ) -> None:
        # A failed answer saved before failed answers kept their measures.
        unmeasured = _failing_run(_BASE, "mix", {"q1": (None, None)})
        comparison = paired_comparison(
            {"local": [_run(_BASE, "local")], "mix": [unmeasured]}
        )
        (warning,) = [w for w in comparison["warnings"] if w.startswith("mix: ")]
        assert "1 without a usage record and 1 without a latency of 4" in warning
        gates = [
            parse_gate("usage/total_tokens:mean_delta>1000"),
            parse_gate("latency/response_time_seconds:mean_delta>1000"),
            parse_gate("answer_match/token_f1:mean_delta<-1"),
        ]
        breaches = gate_breaches(comparison, gates)
        assert len(breaches) == 2
        assert "answers without a value of 'usage/total_tokens' (mix 1)" in (
            breaches[0]
        )
        assert "'latency/response_time_seconds'" in breaches[1]

    def test_failed_answers_that_differ_fail_the_cost_gates(
        self, small_gates: None
    ) -> None:
        # A failed answer counts with what it spent before failing: a cost
        # delta between strategies that failed different numbers of answers
        # compares the failures too (a fast failure looks cheap).
        failed = _failing_run(_BASE, "mix", {"q1": (_usage(1, 10, 0), 0.1)})
        comparison = paired_comparison(
            {"local": [_run(_BASE, "local")], "mix": [failed]}
        )
        gates = [
            parse_gate("usage/total_tokens:mean_delta>1000"),
            parse_gate("latency/response_time_seconds:mean_delta>1000"),
            parse_gate("answer_match/token_f1:mean_delta<-1"),
        ]
        breaches = gate_breaches(comparison, gates)
        assert breaches == [
            "mix: --fail-if usage/total_tokens:mean_delta>1000: failed answers "
            "differ (local 0, mix 1), so the 'usage/total_tokens' delta compares "
            "the failures too",
            "mix: --fail-if latency/response_time_seconds:mean_delta>1000: failed "
            "answers differ (local 0, mix 1), so the "
            "'latency/response_time_seconds' delta compares the failures too",
        ]
        # The same failures on both sides compare like with like.
        both = paired_comparison(
            {
                "local": [
                    _failing_run(_BASE, "local", {"q1": (_usage(1, 10, 0), 0.1)})
                ],
                "mix": [failed],
            }
        )
        assert gate_breaches(both, gates) == []

    def test_repeats_and_cache_hits_that_differ_are_warned(self) -> None:
        def cached(run: Any, hits: int) -> Any:
            for result in run[1]:
                result.usage = result.usage.model_copy(update={"cache_hits": hits})
            return run

        cost = cost_comparison(
            {
                "baseline:local": [
                    cached(_run(_BASE, "local", calls=2), 0),
                    cached(_run(_BASE, "local", calls=1), 1),
                ],
                "local": [cached(_run(_BASE, "local", calls=2), 0)],
            }
        )
        repeats, hits = cost["warnings"]
        assert "different numbers of repeats (baseline:local 2, local 1)" in repeats
        assert "same --repeats" in repeats
        assert "Cache hits per answer differ (baseline:local 0.50, local 0.00)" in hits
        same = cost_comparison(
            {"local": [_run(_BASE, "local")], "mix": [_run(_BASE, "mix")]}
        )
        assert same["warnings"] == []


def test_extracted_span_fallbacks_are_warned_per_strategy() -> None:
    def spans(run: Any, fallbacks: int) -> Any:
        for i, report in enumerate(run[0]):
            report.metadata["answer_span"] = "extracted"
            if i < fallbacks:
                report.metadata["answer_span_fallback"] = "span model down"
        return run

    comparison = paired_comparison(
        {
            "local": [spans(_run(_BASE, "local"), 0)],
            "mix": [spans(_run(_BASE, "mix"), 4)],
        }
    )
    assert comparison["strategies"]["mix"]["answer_span_fallbacks"] == {
        "scored": 4,
        "fallbacks": 4,
    }
    per_strategy, rates = comparison["warnings"]
    assert per_strategy.startswith("mix: 4/4 extracted answer spans fell back")
    assert "fallback rates differ (local 0%, mix 100%)" in rates
    assert (
        "answer_span_fallbacks"
        not in paired_comparison(
            {"local": [_run(_BASE, "local")], "mix": [_run(_BASE, "mix")]}
        )["strategies"]["mix"]
    )


def test_a_failed_answer_is_not_filled_on_attributable_fraction() -> None:
    def with_attribution(run: Any) -> Any:
        reports, results = run
        reports.extend(
            EvaluationReport(
                query_id=r.query_id,
                evaluator_type=EvaluatorType.RETRIEVAL,
                metrics=[
                    EvaluationMetric(
                        metric_type=EvaluationMetricType.ATTRIBUTABLE_FRACTION,
                        value=1.0,
                    )
                ],
                evaluation_time=datetime(2026, 1, 1),
            )
            for r in results
            if not r.error
        )
        return reports, results

    comparison = paired_comparison(
        {
            "local": [with_attribution(_run(_BASE, "local"))],
            "mix": [
                with_attribution(
                    _failing_run(_BASE, "mix", {"q1": (_usage(1, 1, 1), 1.0)})
                )
            ],
        }
    )
    # A share of attributable sources, not a score: no 0 for an answer that
    # reported no source.
    delta = comparison["deltas_vs_baseline"]["mix"]["retrieval/attributable_fraction"]
    assert (delta["pairs"], delta["unpaired"], delta["mean_delta"]) == (3, 1, 0.0)


def test_from_results_warns_on_pooled_runs_with_other_settings(
    tmp_path: Path,
) -> None:
    saved = tmp_path / "runs"
    _saved_strategies(saved, {"local": _BASE, "naive": _BASE})
    # Later, local again with another --top-k into the same directory.
    _saved_strategies(
        saved,
        {"local": _BASE},
        day=1,
        cli_args={"search_strategy": ["local"], "top_k": 1},
    )
    # Not one run's repeats: refused (exit 2) unless pooled explicitly.
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        ["--from-results", str(saved)]
    )
    with pytest.raises(run_evaluation.InvalidDatasetError, match="--pool-runs"):
        run_evaluation.EvaluationRunner(args, None, Config())._saved_runs_input()
    code, out = _cli(
        [
            "--from-results",
            str(saved),
            "--pool-runs",
            "--output-directory",
            str(tmp_path / "o"),
        ]
    )
    warnings = out["comparison"]["warnings"]
    assert code == 0
    (pooled,) = [w for w in warnings if "pooled as its repeats" in w]
    assert "The 2 saved runs of local" in pooled and "top_k (null, 1)" in pooled
    assert any("different numbers of repeats (local 2, naive 1)" in w for w in warnings)


def test_saved_runs_without_a_summary_are_warned(tmp_path: Path) -> None:
    saved = tmp_path / "runs"
    _saved_comparison(saved, _WORSE)
    for path in saved.rglob("evaluation_summary_*.json"):
        path.unlink()
    code, out = _cli(
        ["--from-results", str(saved), "--output-directory", str(tmp_path / "o")]
    )
    (warning,) = [
        w for w in out["comparison"]["warnings"] if "no evaluation_summary" in w
    ]
    assert warning.startswith("--from-results: 2 saved run(s)")


def test_judge_model_without_the_pairwise_judge_is_warned(tmp_path: Path) -> None:
    saved = tmp_path / "runs"
    _saved_comparison(saved, _WORSE)
    code, out = _cli(
        [
            "--from-results",
            str(saved),
            "--judge-model",
            "vendor.judge-v2",
            "--output-directory",
            str(tmp_path / "o"),
        ]
    )
    assert code == 0
    assert any(
        w.startswith("--judge-model is given without --pairwise-judge")
        for w in out["comparison"]["warnings"]
    )


def test_output_directory_that_is_the_baseline_is_detected(tmp_path: Path) -> None:
    # A run saving into its baseline's directory grows that baseline: the
    # next run reading it pools this one in as a repeat.
    (tmp_path / "eval.json").write_text("[]", encoding="utf-8")
    (tmp_path / "ci").mkdir()
    args = run_evaluation.CommandLineInterface().parser.parse_args(
        [
            "--data-path",
            str(tmp_path / "eval.json"),
            "--baseline-results",
            str(tmp_path / "ci"),
            "--output-directory",
            str(tmp_path / "ci"),
        ]
    )
    config = Config()
    config.evaluation.run_subdirectories = False
    runner = run_evaluation.EvaluationRunner(args, None, config)
    assert runner._output_is_baseline()
    runner.args.output_directory = str(tmp_path / "elsewhere")
    assert not runner._output_is_baseline()
    # Into a run subdirectory of its own (the default), it grows nothing.
    runner.args.output_directory = str(tmp_path / "ci")
    runner.config.evaluation.run_subdirectories = True
    assert not runner._output_is_baseline()


def test_markdown_lists_the_pairwise_judge(tmp_path: Path, monkeypatch) -> None:
    saved = tmp_path / "runs"
    _saved_comparison(saved, _WORSE)

    class _Judge:
        def __init__(self, config: Config, providers: Any = None) -> None:
            pass

        async def acompare(self, baseline: Any, candidate: Any) -> dict[str, Any]:
            return {
                "judged": 4,
                "failed": 0,
                "candidate_errors": 0,
                "baseline_errors": 0,
                "position_consistency": 0.75,
                "criteria": {
                    "directness": {
                        "candidate_win_rate": 0.25,
                        "ci_low": 0.0,
                        "ci_high": 0.5,
                    }
                },
            }

    monkeypatch.setattr(run_evaluation, "PairwiseJudge", _Judge)
    code, _ = _cli(
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
    (md,) = (tmp_path / "o").rglob("evaluation_comparison_*.md")
    text = md.read_text(encoding="utf-8")
    assert code == 0 and "## Pairwise judge" in text
    assert "| mix vs local | 4 | 0 | 75% | 0.25 [0.00, 0.50] |" in text


def test_the_tokenization_is_compared_only_on_dense_script_datasets() -> None:
    """Regression: an English-only dataset was refused against a run
    saved before token_f1_tokens was recorded, although every tokenization
    scores Latin text alike."""
    old = {"answer_span": "first_line", "answer_normalization": "default"}
    now = {**old, "token_f1_tokens": "script_aware_v2"}
    pair = [("before", {"answer_scoring": old}), ("now", {"answer_scoring": now})]
    assert check_answer_scoring(pair, dense_script=False) == []
    with pytest.raises(EvaluationException, match="token_f1_tokens"):
        check_answer_scoring(pair, dense_script=True)
    # Without the caller's dataset, what the manifests record.
    english = {"dense_script": False}
    recorded = [
        ("before", {"answer_scoring": old}),
        ("now", {"answer_scoring": now, "dataset": english}),
    ]
    assert check_answer_scoring(recorded) == []
    korean = [
        recorded[0],
        ("now", {"answer_scoring": now, "dataset": {"dense_script": True}}),
    ]
    with pytest.raises(EvaluationException):
        check_answer_scoring(korean)
    # The other settings are compared whatever the dataset.
    squad = {**old, "answer_normalization": "official_squad"}
    with pytest.raises(EvaluationException):
        check_answer_scoring(
            [("before", {"answer_scoring": squad}), recorded[1]], dense_script=False
        )


def test_a_run_of_the_earlier_korean_particle_rule_is_not_compared() -> None:
    """``script_aware`` dropped any one to three Hangul syllables after a
    gold word: its Korean token F1 differs from a run of now."""
    scoring = {"answer_span": "first_line", "answer_normalization": "default"}
    with pytest.raises(EvaluationException, match="'script_aware'"):
        check_answer_scoring(
            [
                (
                    "before",
                    {"answer_scoring": {**scoring, "token_f1_tokens": "script_aware"}},
                ),
                (
                    "now",
                    {
                        "answer_scoring": {
                            **scoring,
                            "token_f1_tokens": "script_aware_v2",
                        }
                    },
                ),
            ]
        )


def test_runs_saved_before_script_aware_tokens_are_not_compared() -> None:
    """A run saved before token_f1_tokens was recorded tokenized by
    whitespace: comparable with an official-normalization run of now, not
    with a default-normalization one."""
    old = {"answer_span": "first_line", "answer_normalization": "default"}
    now = {**old, "token_f1_tokens": "script_aware_v2"}
    with pytest.raises(EvaluationException, match="token_f1_tokens 'whitespace'"):
        check_answer_scoring(
            [("before", {"answer_scoring": old}), ("now", {"answer_scoring": now})]
        )
    squad = {"answer_span": "first_line", "answer_normalization": "official_squad"}
    assert (
        check_answer_scoring(
            [
                ("before", {"answer_scoring": squad}),
                ("now", {"answer_scoring": {**squad, "token_f1_tokens": "whitespace"}}),
            ]
        )
        == []
    )
