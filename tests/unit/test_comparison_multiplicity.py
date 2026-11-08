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
    baseline_comparison,
    bootstrap_ci,
    bootstrap_cis,
    closed_book_strata,
    family_resamples,
    holm_adjusted,
    paired_comparison,
    sign_flip_p_value,
    sign_flip_p_values,
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


def _reference_percentile_ci(
    deltas: list[float], *, resamples: int, confidence: float, seed: int
) -> tuple[float, float]:
    """The percentile bootstrap written out with numpy: ``resamples`` rows
    of query indices drawn from the seed, each row's mean, and the
    ``(1 - confidence) / 2`` tails by ``np.percentile``."""
    data = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(resamples, data.size))].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


class TestBootstrapPercentiles:
    """The intervals against an independent numpy percentile bootstrap, and
    the small samples and counts it special-cases."""

    @pytest.mark.parametrize("confidence", [0.95, 0.9, 0.99])
    @pytest.mark.parametrize("size", [3, 17, 60])
    def test_matches_np_percentile(self, size: int, confidence: float) -> None:
        deltas = np.random.default_rng(size).normal(0.1, 1, size).tolist()
        kwargs = {"resamples": 3000, "confidence": confidence, "seed": 4}
        assert bootstrap_ci(deltas, **kwargs) == pytest.approx(
            _reference_percentile_ci(deltas, **kwargs)
        )

    def test_the_interval_is_wider_at_higher_confidence(self) -> None:
        deltas = np.random.default_rng(1).normal(0, 1, 40).tolist()
        narrow = bootstrap_ci(deltas, resamples=4000, confidence=0.8, seed=0)
        wide = bootstrap_ci(deltas, resamples=4000, confidence=0.99, seed=0)
        assert wide[0] < narrow[0] < narrow[1] < wide[1]

    def test_one_query_is_its_own_interval(self) -> None:
        assert bootstrap_ci([0.7], resamples=2000, confidence=0.95, seed=0) == (
            0.7,
            0.7,
        )

    def test_two_queries_are_resampled(self) -> None:
        # Resampled means of two values are a, (a+b)/2 or b, each end with
        # probability 1/4: the 95% interval spans both values.
        assert bootstrap_ci([0.0, 1.0], resamples=2000, confidence=0.95, seed=0) == (
            0.0,
            1.0,
        )

    def test_wins_ties_and_losses_count_each_sign(self) -> None:
        delta = paired_comparison(
            {
                "a": [_run(np.array([[0.0] * len(_METRICS)] * 5), "a")],
                "b": [
                    _run(np.array([[v] * len(_METRICS) for v in (1, 0, 0, 0, 1)]), "b")
                ],
            },
            resamples=200,
        )["deltas_vs_baseline"]["b"][DEFAULT_PRIMARY_METRIC]
        assert (delta["wins"], delta["ties"], delta["losses"]) == (2, 3, 0)
        losing = paired_comparison(
            {
                "a": [_run(np.array([[1.0] * len(_METRICS)] * 4), "a")],
                "b": [_run(np.array([[v] * len(_METRICS) for v in (0, 1, 0, 1)]), "b")],
            },
            resamples=200,
        )["deltas_vs_baseline"]["b"][DEFAULT_PRIMARY_METRIC]
        assert (losing["wins"], losing["ties"], losing["losses"]) == (0, 2, 2)

    def test_the_precision_delta_interval_is_a_percentile_interval(self) -> None:
        from unified_kg_rag.evaluation.comparison import _precision_delta

        rng = np.random.default_rng(2)
        queries = [f"q{i}" for i in range(30)]
        unanswerable = set(queries[:10])
        one = {q: float(rng.random() < 0.4) for q in queries}
        two = {q: float(rng.random() < 0.6) for q in queries}
        one["q0"] = two["q0"] = 1.0  # both decline something
        kwargs = {"resamples": 3000, "confidence": 0.9, "seed": 5}
        delta = _precision_delta((one, unanswerable), (two, unanswerable), **kwargs)
        assert delta is not None
        # Written out: resample the queries (in sorted order, as paired),
        # recompute both precisions.
        queries = sorted(queries)
        a = np.asarray([one[q] for q in queries])
        b = np.asarray([two[q] for q in queries])
        hit = np.asarray([q in unanswerable for q in queries], dtype=float)
        index = np.random.default_rng(5).integers(0, 30, size=(3000, 30))
        valid = (a[index].sum(axis=1) > 0) & (b[index].sum(axis=1) > 0)
        diffs = (b * hit)[index].sum(axis=1)[valid] / b[index].sum(axis=1)[valid] - (
            a * hit
        )[index].sum(axis=1)[valid] / a[index].sum(axis=1)[valid]
        low, high = np.percentile(diffs, [5, 95])
        assert (delta["ci_low"], delta["ci_high"]) == pytest.approx((low, high))
        assert delta["mean_delta"] == pytest.approx(
            (b * hit).sum() / b.sum() - (a * hit).sum() / a.sum()
        )


class TestBatched:
    """A family's p-values and intervals are drawn together, one draw per
    size for all its tests: exactly what one draw per test gives."""

    def test_the_same_values_as_one_test_at_a_time(self) -> None:
        rng = np.random.default_rng(3)
        samples = [
            rng.normal(0.2, 1, 40).tolist(),
            rng.normal(0.0, 1, 40).tolist(),
            (rng.random(40) < 0.5).astype(float).tolist(),  # fewer non-zero
            [1.0, 0.0, -1.0, 1.0, 1.0],  # exact sign patterns
            [0.0, 0.0],  # no non-zero delta
            [0.7],  # one query
            rng.normal(0.5, 1, 25).tolist(),
        ]
        kwargs = {"resamples": 9000, "seed": 7}
        assert sign_flip_p_values(samples, **kwargs) == [
            sign_flip_p_value(s, **kwargs) for s in samples
        ]
        assert bootstrap_cis(samples, confidence=0.99, **kwargs) == [
            bootstrap_ci(s, confidence=0.99, **kwargs) for s in samples
        ]

    def test_held_means_are_bounded_without_changing_the_intervals(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A family of m tests resamples ~400 m times each: holding every
        # test's means at once grows with m squared. Past the budget the
        # tests of one size go a group at a time, drawing the same indices.
        from unified_kg_rag.evaluation import comparison

        rng = np.random.default_rng(5)
        samples = [rng.normal(0.1, 1, 30).tolist() for _ in range(7)]
        kwargs = {"resamples": 3000, "seed": 11, "confidence": 0.95}
        whole = bootstrap_cis(samples, **kwargs)
        monkeypatch.setattr(comparison, "_BOOTSTRAP_HELD_MEANS", 2 * 3000)
        draws: list[int] = []
        real_rng = np.random.default_rng

        def counting_rng(seed):  # type: ignore[no-untyped-def]
            draws.append(seed)
            return real_rng(seed)

        monkeypatch.setattr(comparison.np.random, "default_rng", counting_rng)
        assert bootstrap_cis(samples, **kwargs) == whole
        # Seven tests of one size, two at a time: four groups, four draws.
        assert len(draws) == 4


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
            # Read from the family's resamples: 20 x 2 tests / 0.05.
            assert families["primary"]["resamples"] == 800
            low, high = bootstrap_ci(
                list(delta["per_query"].values()),
                resamples=800,
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


def _pairwise(scores: list[float]) -> dict:
    """A pairwise_vs_baseline entry of one criterion with these per-query
    candidate scores (its p-value as the judge computes it)."""
    per_query = {f"q{i}": v for i, v in enumerate(scores)}
    return {
        "criteria": {
            "comprehensiveness": {
                "candidate_win_rate": float(np.mean(scores)),
                "p_value": sign_flip_p_value(
                    [v - 0.5 for v in scores], resamples=2000, seed=0
                ),
                "per_query": per_query,
            }
        }
    }


class TestPairwiseAndArmFamilies:
    def test_pairwise_criteria_are_a_family_tested_against_one_half(self) -> None:
        comparison = _comparison_with(["a", "b", "c", "d"])
        comparison["pairwise_vs_baseline"] = {
            "b": _pairwise([1.0] * 30 + [0.5] * 10),
            "c": _pairwise([0.0] * 30 + [0.5] * 10),
            "d": _pairwise([1.0, 0.0] * 20),
        }
        adjust_for_multiplicity(comparison)
        families = comparison["multiplicity"]["families"]
        assert families["pairwise"]["tests"] == 3
        b, c, d = (
            comparison["pairwise_vs_baseline"][s]["criteria"]["comprehensiveness"]
            for s in ("b", "c", "d")
        )
        assert b["family"] == c["family"] == d["family"] == "pairwise"
        # Thirty wins (or losses) and ten ties: far from 0.5 either way. Read
        # again from the family's 1,200 resamples (20 x 3 tests / 0.05, more
        # than the comparison's 500): the Monte Carlo floor 1 / 1201, Holm
        # over the three tests.
        assert families["pairwise"]["resamples"] == 1200
        for strong in (b, c):
            assert strong["p_value_resamples"] == 1200
            assert strong["p_value"] == pytest.approx(1 / 1201)
            assert strong["adjusted_p_value"] == pytest.approx(3 / 1201)
        # Even wins and losses: no difference from 0.5.
        assert d["adjusted_p_value"] == pytest.approx(1.0)
        # The interval is of the win rate itself.
        assert d["adjusted_ci_low"] < 0.5 < d["adjusted_ci_high"]
        assert b["adjusted_ci_low"] > 0.5 > c["adjusted_ci_high"]

    def test_aa_and_closed_book_arms_are_left_out_and_arms_are_their_own(
        self,
    ) -> None:
        labels = ["local", "mix", "local@aa", "local@rerank_off", "closed_book"]
        comparison = _comparison_with(labels)
        comparison["pairwise_vs_baseline"] = {
            label: _pairwise([1.0, 0.5, 0.0, 1.0] * 5) for label in labels[1:]
        }
        adjust_for_multiplicity(
            comparison,
            exclude=["closed_book", "local@aa"],
            arms=["local@aa", "local@rerank_off"],
        )
        families = comparison["multiplicity"]["families"]
        assert comparison["multiplicity"]["unadjusted"] == ["closed_book", "local@aa"]
        # One strategy against the first, one arm against its base run.
        for kind in ("primary", "pairwise"):
            assert families[kind]["tests"] == 1
            assert families[f"arm_{kind}"]["tests"] == 1
        deltas = comparison["deltas_vs_baseline"]
        assert deltas["mix"][DEFAULT_PRIMARY_METRIC]["family"] == "primary"
        assert (
            deltas["local@rerank_off"][DEFAULT_PRIMARY_METRIC]["family"]
            == "arm_primary"
        )
        assert deltas["local@rerank_off"]["retrieval/mrr"]["family"] == "arm_secondary"
        for excluded in ("local@aa", "closed_book"):
            assert all("family" not in d for d in deltas[excluded].values())
            judged = comparison["pairwise_vs_baseline"][excluded]["criteria"]
            assert "adjusted_p_value" not in judged["comprehensiveness"]

    def test_a_pairwise_criterion_without_a_decided_pair_is_no_test(self) -> None:
        comparison = _comparison_with(["a", "b"])
        comparison["pairwise_vs_baseline"] = {
            "b": {
                "criteria": {
                    "comprehensiveness": {
                        "candidate_win_rate": None,
                        "p_value": None,
                        "per_query": {},
                    }
                }
            }
        }
        adjust_for_multiplicity(comparison)
        assert "pairwise" not in comparison["multiplicity"]["families"]


class TestClosedBookStrata:
    def _runs(self) -> dict:
        # q0-q3 the answer model knows without retrieval (two repeats: q3
        # once of twice, still known), q4-q7 it does not.
        n = len(_METRICS)
        closed = [
            _run(np.array([[v] * n for v in (1, 1, 1, 1, 0, 0, 0, 0)]), "closed_book"),
            _run(np.array([[v] * n for v in (1, 1, 1, 0, 0, 0, 0, 1)]), "closed_book"),
        ]
        mix = np.array([[v] * n for v in (1, 1, 1, 1, 0, 0, 0, 0)])
        local = np.array([[v] * n for v in (1, 1, 1, 0, 1, 1, 1, 1)])
        return {
            "mix": [_run(mix, "mix")],
            "local": [_run(local, "local")],
            "closed_book": closed,
        }

    def test_known_is_at_least_half_the_repeats_right(self) -> None:
        strata = closed_book_strata(self._runs()["closed_book"], DEFAULT_PRIMARY_METRIC)
        assert strata == {
            **{f"q{i}": "known" for i in range(4)},
            "q4": "unknown",
            "q5": "unknown",
            "q6": "unknown",
            "q7": "known",
        }
        assert closed_book_strata(self._runs()["closed_book"], "ragas/x") == {}

    def test_deltas_by_closed_book_split_the_paired_deltas(self) -> None:
        runs = self._runs()
        strata = closed_book_strata(runs["closed_book"], DEFAULT_PRIMARY_METRIC)
        comparison = adjust_for_multiplicity(
            paired_comparison(runs, resamples=500, closed_book=strata),
            exclude=["closed_book"],
        )
        local = comparison["deltas_by_closed_book"]["local"]
        # Retrieval adds the three answers the model does not know.
        assert local["unknown"][DEFAULT_PRIMARY_METRIC]["mean_delta"] == 1.0
        assert local["unknown"][DEFAULT_PRIMARY_METRIC]["pairs"] == 3
        # On the known ones it loses q3 and wins q7.
        assert local["known"][DEFAULT_PRIMARY_METRIC]["mean_delta"] == 0.0
        assert local["known"][DEFAULT_PRIMARY_METRIC]["wins"] == 1
        # Exploratory: no Holm value.
        assert all(
            "adjusted_p_value" not in d
            for stratum in local.values()
            for d in stratum.values()
        )
        assert "deltas_by_closed_book" not in paired_comparison(runs, resamples=200)
        # Each strategy against its own baseline (--arm, --baseline-results).
        merged = baseline_comparison(
            {"mix": runs["mix"]},
            {"local": runs["local"], "closed_book": runs["closed_book"]},
            {"local": "mix", "closed_book": "mix"},
            resamples=500,
            closed_book=strata,
        )["deltas_by_closed_book"]
        assert merged["local"]["unknown"][DEFAULT_PRIMARY_METRIC]["mean_delta"] == 1.0
        assert set(merged) == {"local", "closed_book"}


class TestFamilyResamples:
    """A Monte Carlo sign-flip p-value is never below 1 / (1 + resamples), and
    Holm multiplies the smallest by the family's size: with 2,000 resamples a
    family of 100 or more tests could not reject at 0.05 however large the
    difference. The family reads enough resamples for its size instead."""

    def test_resamples_scale_with_the_family(self) -> None:
        assert family_resamples(2000, 1, 0.05) == 2000
        assert family_resamples(2000, 105, 0.05) == 42000
        assert family_resamples(50_000, 105, 0.05) == 50_000

    def test_a_large_family_still_rejects_a_large_difference(self) -> None:
        # 15 compared strategies x 7 secondary metrics = 105 tests; every
        # strategy recalls every query the baseline missed.
        rng = np.random.default_rng(8)
        base = (rng.random((40, len(_METRICS))) < 0.3).astype(float)
        runs = {"s0": [_run(base, "s0")]}
        for i in range(1, 16):
            better = base.copy()
            better[:, 4] = 1.0
            runs[f"s{i}"] = [_run(better, f"s{i}")]
        comparison = adjust_for_multiplicity(paired_comparison(runs, resamples=2000))
        families = comparison["multiplicity"]["families"]
        assert families["secondary"]["tests"] >= 100
        assert families["secondary"]["resamples"] >= 20 * 100 / ALPHA
        recall = comparison["deltas_vs_baseline"]["s1"]["retrieval/recall_at_k"]
        assert recall["mean_delta"] > 0.5
        # 1 / 2001 x 105 tests held it at 0.0525, above alpha.
        assert recall["p_value"] < 1 / 2001
        assert recall["p_value_resamples"] == families["secondary"]["resamples"]
        assert recall["adjusted_p_value"] < 0.01
        assert recall["adjusted_ci_low"] > 0

    @pytest.mark.slow
    @pytest.mark.timeout(600)
    def test_a_family_holds_its_level_with_scaled_resamples(self) -> None:
        # A/A comparisons of 6 strategies (35 secondary tests): an adjusted
        # p-value below alpha, or a simultaneous interval excluding 0, in
        # about alpha of the trials (the percentile interval holds its level
        # from about 100 queries; on 30 it excluded 0 in 15%).
        rng = np.random.default_rng(9)
        trials, hits, excluded = 8, 0, 0
        for _ in range(trials):
            base = (rng.random((100, len(_METRICS))) < 0.4).astype(float)
            runs = {
                f"s{i}": [
                    _run(
                        np.where(rng.random(base.shape) < 0.2, 1 - base, base),
                        f"s{i}",
                    )
                ]
                for i in range(6)
            }
            comparison = adjust_for_multiplicity(paired_comparison(runs, resamples=200))
            secondary = [
                d
                for deltas in comparison["deltas_vs_baseline"].values()
                for d in deltas.values()
                if d.get("family") == "secondary"
            ]
            hits += min(d["adjusted_p_value"] for d in secondary) < ALPHA
            excluded += any(
                d["adjusted_ci_low"] > 0 or d["adjusted_ci_high"] < 0 for d in secondary
            )
        assert hits <= 2
        assert excluded <= 2


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
