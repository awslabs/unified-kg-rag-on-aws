# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multiplicity in a paired comparison: sign-flip p-values, Holm families
and the gates that read them.

A comparison reads many deltas, and each unadjusted 95% interval excludes 0
on noise in 5% of runs, so over ten metrics a clean change looks
"supported" somewhere in 30-40% of runs. The null simulations below
check that the p-value holds its level and that Holm holds the family-wise
rate, on the comparison's own code path. Deterministic (seeded), LLM-free.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from unified_kg_rag.domain.models import (
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
)
from unified_kg_rag.evaluation.comparison import (
    DEFAULT_PRIMARY_METRIC,
    adjust_for_multiplicity,
    bootstrap_ci,
    holm_adjusted,
    paired_comparison,
    sign_flip_p_value,
)
from unified_kg_rag.evaluation.gates import gate_breaches, parse_gate
from unified_kg_rag.evaluation.report import comparison_markdown, delta_cell

pytestmark = pytest.mark.unit

ALPHA = 0.05
_METRICS = [
    (EvaluatorType.ANSWER_MATCH, EvaluationMetricType.ANSWER_CONTAINS),
    (EvaluatorType.ANSWER_MATCH, EvaluationMetricType.EXACT_MATCH),
    (EvaluatorType.ANSWER_MATCH, EvaluationMetricType.TOKEN_F1),
    (EvaluatorType.RETRIEVAL, EvaluationMetricType.HIT_AT_K),
    (EvaluatorType.RETRIEVAL, EvaluationMetricType.RECALL_AT_K),
    (EvaluatorType.RETRIEVAL, EvaluationMetricType.MRR),
    (EvaluatorType.GRAPH_AWARE, EvaluationMetricType.ENTITY_COVERAGE),
    (EvaluatorType.GRAPH_AWARE, EvaluationMetricType.RELATIONSHIP_COVERAGE),
]


def _run(scores: np.ndarray, strategy: str, latency: np.ndarray | None = None):
    """One run: ``scores[q, m]`` of query q on metric m of ``_METRICS``."""
    reports = [
        EvaluationReport(
            query_id=f"q{q}",
            evaluator_type=evaluator,
            metrics=[EvaluationMetric(metric_type=metric, value=float(scores[q, m]))],
            evaluation_time=datetime(2026, 1, 1),
        )
        for q in range(scores.shape[0])
        for m, (evaluator, metric) in enumerate(_METRICS)
    ]
    results = [
        EvaluationResult(
            query_id=f"q{q}",
            question="?",
            generated_answer="a",
            ground_truth="",
            search_strategy=strategy,
            response_time=None if latency is None else float(latency[q]),
        )
        for q in range(scores.shape[0])
    ]
    return reports, results


class TestSignFlip:
    def test_exact_on_few_pairs(self) -> None:
        # Five wins of five: the exact two-sided sign test's 2 * 0.5**5.
        assert sign_flip_p_value([1.0] * 5, resamples=2000, seed=0) == 0.0625
        assert sign_flip_p_value([1.0, -1.0], resamples=2000, seed=0) == 1.0
        # Ties carry no sign; all ties is no evidence at all.
        assert sign_flip_p_value([0.0, 0.0, 0.5], resamples=2000, seed=0) == 1.0
        assert sign_flip_p_value([0.0] * 4, resamples=2000, seed=0) == 1.0

    def test_sampled_on_many_pairs_and_seeded(self) -> None:
        deltas = [0.1] * 30 + [-0.1] * 10
        p = sign_flip_p_value(deltas, resamples=2000, seed=0)
        assert p == sign_flip_p_value(deltas, resamples=2000, seed=0)
        # 30 of 40 one way: the binomial two-sided p is about 0.0022.
        assert 0.0 < p < 0.01

    def test_holds_its_level_on_null_deltas(self) -> None:
        # Two arms of one configuration: 0/1 scores of the same rate.
        rng = np.random.default_rng(1)
        rejected = 0
        trials = 400
        for _ in range(trials):
            a = rng.random(200) < 0.5
            b = rng.random(200) < 0.5
            deltas = (b.astype(float) - a.astype(float)).tolist()
            rejected += sign_flip_p_value(deltas, resamples=1000, seed=0) < ALPHA
        # Nominal 5%; binomial sd over 400 trials is about 1.1%.
        assert 0.02 <= rejected / trials <= 0.08

    def test_holds_its_level_on_ten_pairs(self) -> None:
        # Where the percentile bootstrap breached on noise in 8% of runs.
        rng = np.random.default_rng(2)
        trials = 600
        rejected = sum(
            sign_flip_p_value(
                (rng.integers(0, 2, 10) - rng.integers(0, 2, 10)).tolist(),
                resamples=2000,
                seed=0,
            )
            < ALPHA
            for _ in range(trials)
        )
        assert rejected / trials <= 0.06


class TestHolm:
    def test_step_down_values(self) -> None:
        assert holm_adjusted([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
        assert holm_adjusted([0.5, 0.001]) == pytest.approx([0.5, 0.002])
        assert holm_adjusted([0.9, 0.8]) == pytest.approx([1.0, 1.0])
        assert holm_adjusted([]) == []

    def test_holds_the_family_wise_rate_where_unadjusted_tests_do_not(self) -> None:
        rng = np.random.default_rng(3)
        trials, family = 300, 10
        raw_any = holm_any = 0
        for _ in range(trials):
            p = [
                sign_flip_p_value(
                    (rng.integers(0, 2, 150) - rng.integers(0, 2, 150)).tolist(),
                    resamples=500,
                    seed=0,
                )
                for _ in range(family)
            ]
            raw_any += min(p) < ALPHA
            holm_any += min(holm_adjusted(p)) < ALPHA
        # Ten null tests: some unadjusted one is "significant" in ~40%.
        assert raw_any / trials > 0.25
        assert holm_any / trials <= 0.08


def _null_comparison(rng: np.random.Generator, queries: int, **kwargs):
    """Two strategies of one configuration: every metric's scores drawn from
    the same distribution per query (query difficulty shared, as in a real
    A/A pair), and latency too."""
    difficulty = rng.random((queries, len(_METRICS)))
    a = (rng.random((queries, len(_METRICS))) < difficulty).astype(float)
    b = (rng.random((queries, len(_METRICS))) < difficulty).astype(float)
    latency = 1.0 + rng.random(queries)
    runs = {
        "local": [_run(a, "local", latency + rng.normal(0, 0.1, queries))],
        "mix": [_run(b, "mix", latency + rng.normal(0, 0.1, queries))],
    }
    return paired_comparison(runs, resamples=500, **kwargs)


class TestNullComparisons:
    def test_false_positives_on_the_comparison_path(self) -> None:
        # The comparison as run-eval builds it, on A/A pairs (fewer trials
        # than the p-value checks above: each builds a whole comparison).
        rng = np.random.default_rng(4)
        trials = 60
        primary_hits = secondary_any = secondary_raw_any = 0
        for _ in range(trials):
            comparison = adjust_for_multiplicity(_null_comparison(rng, 100))
            deltas = comparison["deltas_vs_baseline"]["mix"]
            primary_hits += deltas[DEFAULT_PRIMARY_METRIC]["adjusted_p_value"] < ALPHA
            secondary = [d for d in deltas.values() if d.get("family") == "secondary"]
            assert len(secondary) == len(_METRICS) - 1
            secondary_any += min(d["adjusted_p_value"] for d in secondary) < ALPHA
            secondary_raw_any += any(
                d["ci_low"] > 0 or d["ci_high"] < 0 for d in secondary
            )
        assert primary_hits / trials <= 0.10  # nominal 5%, sd ~2.8%
        assert secondary_any / trials <= 0.10  # family-wise, nominal 5%
        # Unadjusted, seven metrics' intervals exclude 0 somewhere ~30%.
        assert secondary_raw_any / trials > 0.15

    def test_a_real_difference_is_still_found(self) -> None:
        rng = np.random.default_rng(5)
        queries = 200
        a = (rng.random((queries, len(_METRICS))) < 0.4).astype(float)
        b = a.copy()
        # The candidate answers 15% more queries on the primary metric.
        flip = (a[:, 0] == 0) & (rng.random(queries) < 0.25)
        b[flip, 0] = 1.0
        comparison = adjust_for_multiplicity(
            paired_comparison(
                {"local": [_run(a, "local")], "mix": [_run(b, "mix")]},
                resamples=1000,
            )
        )
        primary = comparison["deltas_vs_baseline"]["mix"][DEFAULT_PRIMARY_METRIC]
        assert primary["family"] == "primary"
        assert primary["adjusted_p_value"] < 0.001
        assert primary["adjusted_ci_low"] > 0


def _comparison_with(strategies: list[str]) -> dict:
    rng = np.random.default_rng(6)
    base = (rng.random((40, len(_METRICS))) < 0.5).astype(float)
    runs = {
        name: [_run(base if i == 0 else 1 - base, name)]
        for i, name in enumerate(strategies)
    }
    return paired_comparison(runs, resamples=500)


class TestFamilies:
    def test_primary_is_adjusted_over_strategies_only(self) -> None:
        comparison = adjust_for_multiplicity(_comparison_with(["a", "b", "c"]))
        multiplicity = comparison["multiplicity"]
        assert multiplicity["primary_metric"] == DEFAULT_PRIMARY_METRIC
        families = multiplicity["families"]
        assert families["primary"]["tests"] == 2
        assert families["secondary"]["tests"] == 2 * (len(_METRICS) - 1)
        assert families["primary"]["interval_confidence"] == pytest.approx(0.975)
        for strategy in ("b", "c"):
            delta = comparison["deltas_vs_baseline"][strategy][DEFAULT_PRIMARY_METRIC]
            # Wider than the unadjusted interval, from the same per-query deltas.
            assert delta["adjusted_ci_low"] <= delta["ci_low"]
            assert delta["adjusted_ci_high"] >= delta["ci_high"]
            low, high = bootstrap_ci(
                list(delta["per_query"].values()),
                resamples=500,
                confidence=0.975,
                seed=0,
            )
            assert (delta["adjusted_ci_low"], delta["adjusted_ci_high"]) == (low, high)

    def test_a_declared_primary_metric_and_an_excluded_arm(self) -> None:
        comparison = adjust_for_multiplicity(
            _comparison_with(["a", "b", "closed_book"]),
            primary_metric="retrieval/mrr",
            exclude=["closed_book"],
        )
        assert comparison["multiplicity"]["primary_metric"] == "retrieval/mrr"
        assert comparison["multiplicity"]["unadjusted"] == ["closed_book"]
        assert comparison["multiplicity"]["families"]["primary"]["tests"] == 1
        b = comparison["deltas_vs_baseline"]["b"]
        assert b["retrieval/mrr"]["family"] == "primary"
        # One primary test: no adjustment.
        assert b["retrieval/mrr"]["adjusted_p_value"] == b["retrieval/mrr"]["p_value"]
        assert b[DEFAULT_PRIMARY_METRIC]["family"] == "secondary"
        closed = comparison["deltas_vs_baseline"]["closed_book"]
        assert all("adjusted_p_value" not in d for d in closed.values())

    def test_a_declared_primary_metric_with_no_delta_is_warned_about(self) -> None:
        comparison = adjust_for_multiplicity(
            _comparison_with(["a", "b"]), primary_metric="ragas/answer_correctness"
        )
        assert comparison["multiplicity"]["primary_metric"] is None
        assert "primary" not in comparison["multiplicity"]["families"]
        assert any("ragas/answer_correctness" in w for w in comparison["warnings"])
        # The default one missing is no warning (answers not scored).
        quiet = _comparison_with(["a", "b"])
        for delta in quiet["deltas_vs_baseline"]["b"].values():
            delta.pop("per_query")
        adjust_for_multiplicity(quiet)
        assert not any("primary metric" in w for w in quiet["warnings"])

    def test_cost_measures_are_their_own_family(self) -> None:
        rng = np.random.default_rng(7)
        comparison = adjust_for_multiplicity(_null_comparison(rng, 50))
        latency = comparison["cost"]["deltas_vs_baseline"]["mix"][
            "latency/response_time_seconds"
        ]
        assert latency["family"] == "cost"
        assert comparison["multiplicity"]["families"]["cost"]["tests"] == 1


class TestGatesAndReport:
    def test_gates_read_the_adjusted_statistics(self) -> None:
        comparison = adjust_for_multiplicity(_comparison_with(["a", "b"]))
        delta = comparison["deltas_vs_baseline"]["b"][DEFAULT_PRIMARY_METRIC]
        gate = parse_gate(f"{DEFAULT_PRIMARY_METRIC}:adjusted_ci_high<1")
        assert gate_breaches(comparison, [gate]) == [
            f"b: {DEFAULT_PRIMARY_METRIC} adjusted_ci_high "
            f"{delta['adjusted_ci_high']:+.4g} < 1"
        ]
        assert not gate_breaches(
            comparison, [parse_gate(f"{DEFAULT_PRIMARY_METRIC}:adjusted_p_value>0.5")]
        )

    def test_a_gate_on_a_missing_adjusted_statistic_fails(self) -> None:
        comparison = _comparison_with(["a", "b"])  # not adjusted
        (breach,) = gate_breaches(
            comparison, [parse_gate(f"{DEFAULT_PRIMARY_METRIC}:adjusted_p_value<0.05")]
        )
        assert "has no adjusted_p_value" in breach

    def test_unknown_statistic_is_refused(self) -> None:
        with pytest.raises(ValueError, match="adjusted_ci_high"):
            parse_gate("a/b:holm<0")

    def test_tables_show_the_holm_p_value_and_the_primary_metric(self) -> None:
        comparison = adjust_for_multiplicity(_comparison_with(["a", "b"]))
        delta = comparison["deltas_vs_baseline"]["b"][DEFAULT_PRIMARY_METRIC]
        assert f"Holm p {delta['adjusted_p_value']:.3g}" in delta_cell(delta)
        markdown = comparison_markdown(comparison)
        assert f"`{DEFAULT_PRIMARY_METRIC}` (primary)" in markdown
        assert "Primary metric:" in markdown and "secondary 7" in markdown
