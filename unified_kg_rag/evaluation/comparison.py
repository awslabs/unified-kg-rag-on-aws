# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Paired comparison of evaluation runs over the same queries.

``run-eval --search-strategy a,b,c --repeats N`` answers and scores one
dataset once per strategy and repeat. Comparing their pooled means would
mix two sources of noise — which queries a strategy happened to do well on,
and run-to-run variance of the answer model — so the comparison is paired:
every metric is averaged per query over the repeats, the first strategy is
the baseline, and each other strategy is scored by its per-query deltas
against it. The mean delta comes with a paired bootstrap confidence
interval (queries resampled with replacement, a fixed seed, percentile
interval) and win/tie/loss counts; a CI that excludes 0 is a difference the
query sample supports. Deterministic and LLM-free.

A query whose answer generation failed has no reports, and dropping it
would let a strategy that crashes on the queries it would lose look as good
as one that answers them. So a failed answer scores 0 on every metric
another run scored that query on (``answer_length``, a diagnostic rather than
a score, excepted), and each strategy reports its failed answers;
``warnings`` names the strategies whose failure counts differ. Likewise each
strategy reports its guardrail-blocked answers (by blocked text), and
different counts are warned about too.

``auto`` is reported with the distribution of the strategies it routed to,
since its scores are a mix of theirs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from unified_kg_rag.domain.models import (
    EvaluationMetricType,
    EvaluationReport,
    EvaluationResult,
)
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

# One run: the reports and results of one strategy and repeat.
Run = tuple[list[EvaluationReport], list[EvaluationResult]]

# Not a score: a failed answer gets no value for it.
_DIAGNOSTICS = frozenset({EvaluationMetricType.ANSWER_LENGTH.value})


def _scored_metrics(runs: Mapping[str, Sequence[Run]]) -> dict[str, set[str]]:
    """Per query, the scores any run of any strategy reported for it."""
    scored: dict[str, set[str]] = defaultdict(set)
    for strategy_runs in runs.values():
        for reports, _ in strategy_runs:
            for report in reports:
                for metric in report.metrics:
                    if metric.metric_type.value not in _DIAGNOSTICS:
                        scored[report.query_id].add(
                            f"{report.evaluator_type.value}/{metric.metric_type.value}"
                        )
    return scored


def per_query_means(
    runs: Sequence[Run], failed_scores: Mapping[str, set[str]] | None = None
) -> dict[str, dict[str, float]]:
    """``{"evaluator/metric": {query_id: mean over the runs that scored it}}``.

    A query whose answer failed in a run counts 0 in that run for each metric
    ``failed_scores`` lists for it.
    """
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for reports, results in runs:
        for report in reports:
            for metric in report.metrics:
                key = f"{report.evaluator_type.value}/{metric.metric_type.value}"
                values[key][report.query_id].append(metric.value)
        for result in results:
            if result.error:
                for key in (failed_scores or {}).get(result.query_id, ()):
                    values[key][result.query_id].append(0.0)
    return {
        key: {qid: float(np.mean(v)) for qid, v in by_query.items()}
        for key, by_query in sorted(values.items())
    }


def bootstrap_ci(
    deltas: Sequence[float], *, resamples: int, confidence: float, seed: int
) -> tuple[float, float]:
    """Percentile CI of the mean of ``deltas`` (queries resampled)."""
    data = np.asarray(deltas, dtype=float)
    if data.size == 1:
        return float(data[0]), float(data[0])
    rng = np.random.default_rng(seed)
    means = data[rng.integers(0, data.size, size=(resamples, data.size))].mean(axis=1)
    tail = (1.0 - confidence) / 2 * 100
    low, high = np.percentile(means, [tail, 100 - tail])
    return float(low), float(high)


def paired_comparison(
    runs: Mapping[str, Sequence[Run]],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare strategies run over the same queries (the first is the baseline).

    ``runs`` maps each strategy, in the order given, to its repeats.
    """
    strategies = list(runs)
    baseline = strategies[0]
    scored = _scored_metrics(runs)
    means = {s: per_query_means(runs[s], scored) for s in strategies}
    summary: dict[str, Any] = {}
    for strategy in strategies:
        run_means = [per_query_means([run], scored) for run in runs[strategy]]
        metrics: dict[str, Any] = {}
        for key, by_query in means[strategy].items():
            per_repeat = [
                float(np.mean(list(m[key].values()))) for m in run_means if key in m
            ]
            metrics[key] = {
                "mean": float(np.mean(list(by_query.values()))),
                "queries": len(by_query),
                # Spread of the run means: the answer model's run-to-run noise.
                "repeat_stdev": (
                    float(np.std(per_repeat, ddof=1)) if len(per_repeat) > 1 else None
                ),
            }
        routed = Counter(
            result.search_strategy
            for _, results in runs[strategy]
            for result in results
            if result.search_strategy
        )
        failed = sorted(
            {
                result.query_id
                for _, results in runs[strategy]
                for result in results
                if result.error
            }
        )
        summary[strategy] = {
            "repeats": len(runs[strategy]),
            "routed_strategies": dict(sorted(routed.items())),
            # Answers whose generation failed (over every repeat), each
            # scored 0, and the queries they belong to.
            "failed_answers": sum(
                result.error for _, results in runs[strategy] for result in results
            ),
            "failed_queries": failed,
            # Answers the guardrail blocked (over every repeat), by blocked text.
            "guardrail_blocked_counts": dict(
                sorted(
                    Counter(
                        result.guardrail_blocked
                        for _, results in runs[strategy]
                        for result in results
                        if result.guardrail_blocked
                    ).items()
                )
            ),
            "metrics": metrics,
        }

    deltas: dict[str, Any] = {}
    for strategy in strategies[1:]:
        by_metric: dict[str, Any] = {}
        for key, base in means[baseline].items():
            other = means[strategy].get(key, {})
            paired = sorted(set(base) & set(other))
            if not paired:
                continue
            # Scored by one side only: a metric skipped there (no
            # expectation, no source of that type), never a failed answer.
            unpaired = len(set(base) ^ set(other))
            diff = [other[q] - base[q] for q in paired]
            low, high = bootstrap_ci(
                diff, resamples=resamples, confidence=confidence, seed=seed
            )
            by_metric[key] = {
                "pairs": len(paired),
                "unpaired": unpaired,
                "mean_delta": float(np.mean(diff)),
                "ci_low": low,
                "ci_high": high,
                "wins": sum(d > 0 for d in diff),
                "ties": sum(d == 0 for d in diff),
                "losses": sum(d < 0 for d in diff),
                "per_query": dict(zip(paired, diff, strict=True)),
            }
        deltas[strategy] = by_metric
    warnings = _failure_warnings(summary)
    for warning in warnings:
        logger.warning("%s", warning)
    return {
        "baseline": baseline,
        "strategies": summary,
        "deltas_vs_baseline": deltas,
        "warnings": warnings,
        "bootstrap": {
            "resamples": resamples,
            "confidence": confidence,
            "seed": seed,
        },
    }


def _failure_warnings(summary: Mapping[str, Any]) -> list[str]:
    """Warnings when the strategies failed, or had blocked, different numbers
    of answers."""
    warnings = []
    failed = {s: stats["failed_answers"] for s, stats in summary.items()}
    if len(set(failed.values())) > 1:
        counts = ", ".join(f"{s} {n}" for s, n in failed.items())
        warnings.append(
            f"Strategies failed different numbers of answers ({counts}); each "
            "failed answer scores 0, so the comparison reflects those failures."
        )
    blocked = {s: stats["guardrail_blocked_counts"] for s, stats in summary.items()}
    if len({tuple(c.items()) for c in blocked.values()}) > 1:
        counts = ", ".join(f"{s} {sum(c.values())}" for s, c in blocked.items())
        warnings.append(
            f"The guardrail blocked different numbers of answers ({counts}); "
            "a blocked answer is scored on its blocked message, so the "
            "comparison reflects those blocks."
        )
    return warnings
