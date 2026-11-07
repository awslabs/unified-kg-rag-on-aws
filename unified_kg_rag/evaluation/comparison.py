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
a score, and the lower-is-better assertion rates, on which 0 is the best
score, excepted) and that its own strategy scores on some query: a metric
the strategy never produces (``global`` reports no entities, so no
``retrieved_entity_coverage``) is not one it lost, and filling it would invent
a delta. A strategy that scored nothing at all is filled on every metric.
Each strategy reports its failed answers; ``warnings`` names the strategies
whose failure counts differ. Likewise each strategy reports its
guardrail-blocked answers (by blocked text), and different counts are warned
about too, as are runs whose exact match and token F1 read different answer
spans (``evaluation.answer_span``), and extracted spans that fell back to
the first line (``answer_span_fallbacks``).

With unanswerable items (``metadata.answerable: false``) the declines are
paired too: ``abstention/recall`` is per unanswerable query whether its
answer declined (a failed answer did not), a per-query mean like a metric;
``abstention/precision`` (declines on unanswerable queries over all
declines) is a ratio, so its delta's interval recomputes both precisions on
each resample of the queries (``_precision_delta``). Both are ``--fail-if``
measures.

Given each query's category, ``deltas_by_category`` repeats the deltas per
category. Its intervals are per category and uncorrected for the number of
categories and metrics compared: read them as exploratory.

``auto`` is reported with the distribution of the strategies it routed to,
since its scores are a mix of theirs.

``baseline_comparison`` compares each strategy with a baseline of its own
(``run-eval --baseline-results``: ``baseline:<s>`` against ``s``), one
paired comparison per pair, merged into one result of the same shape;
``baselines`` names each compared strategy's baseline.

``cost`` (``cost_comparison``) puts the answers' model use next to the
scores: per strategy the calls, tokens and latency per query (mean, p50,
p95 over every answer, a failed one with what it spent before it failed)
and, against the baseline, their paired per-query deltas with the same
bootstrap interval; with ``evaluation.prices`` a USD estimate. The comparison is a function of the saved results and reports
alone, so ``run-eval --from-results`` rebuilds it without a model call.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np

from unified_kg_rag.adapters.evaluators.base import FAILED_METRICS_KEY
from unified_kg_rag.domain.models import (
    EvaluationMetricType,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
    LLMUsage,
    LLMUsageCounts,
)
from unified_kg_rag.domain.models.config import ModelPrice
from unified_kg_rag.shared import get_logger

from .answer_match_evaluator import ANSWER_SPAN_KEY, SPAN_FALLBACK_KEY
from .iterative_statistics import iterative_statistics
from .structured_statistics import structured_statistics

logger = get_logger(__name__)

# One run: the reports and results of one strategy and repeat.
Run = tuple[list[EvaluationReport], list[EvaluationResult]]

# Not a score: a failed answer gets no value for it (an answer that failed
# reported no source, so it has no share of attributable ones either).
_DIAGNOSTICS = frozenset(
    {
        EvaluationMetricType.ANSWER_LENGTH.value,
        EvaluationMetricType.ATTRIBUTABLE_FRACTION.value,
    }
)


# Lower-is-better rates: a failed answer filled with 0 on them would score a
# crash as perfect, so it stays unpaired there instead.
_UNFILLED = _DIAGNOSTICS | {
    EvaluationMetricType.ASSERTION_CONTRADICTION_RATE.value,
    EvaluationMetricType.NEGATIVE_ASSERTION_VIOLATION_RATE.value,
}


def _strategy_failed_scores(
    runs: Mapping[str, Sequence[Run]], scored: Mapping[str, set[str]]
) -> dict[str, dict[str, set[str]]]:
    """Per strategy, the metrics a failed answer counts 0 on, per query.

    Those of ``scored`` the strategy itself scored at least once; all of
    them when it scored none (every answer failed).
    """
    failed: dict[str, dict[str, set[str]]] = {}
    for strategy, strategy_runs in runs.items():
        produced = set().union(*_scored_metrics({strategy: strategy_runs}).values())
        failed[strategy] = (
            {q: keys & produced for q, keys in scored.items()}
            if produced
            else dict(scored)
        )
    return failed


def _scored_metrics(runs: Mapping[str, Sequence[Run]]) -> dict[str, set[str]]:
    """Per query, the scores any run of any strategy reported for it."""
    scored: dict[str, set[str]] = defaultdict(set)
    for strategy_runs in runs.values():
        for reports, _ in strategy_runs:
            for report in reports:
                for metric in report.metrics:
                    if metric.metric_type.value not in _UNFILLED:
                        scored[report.query_id].add(
                            f"{report.evaluator_type.value}/{metric.metric_type.value}"
                        )
    return scored


# The decline measures of a dataset with unanswerable items
# (``metadata.answerable: false``): per query, whether the answer declined.
ABSTENTION_RECALL = "abstention/recall"
ABSTENTION_PRECISION = "abstention/precision"


def _unanswerable(result: EvaluationResult) -> bool:
    return result.metadata.get("answerable") is False


def _declined(result: EvaluationResult) -> bool:
    return bool(result.abstained) and not result.error


def per_query_means(
    runs: Sequence[Run], failed_scores: Mapping[str, set[str]] | None = None
) -> dict[str, dict[str, float]]:
    """``{"evaluator/metric": {query_id: mean over the runs that scored it}}``.

    A query whose answer failed in a run counts 0 in that run for each metric
    ``failed_scores`` lists for it. An unanswerable query also has
    ``abstention/recall``: 1 when its answer declined, else 0 (a failed
    answer is no decline).
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
            if _unanswerable(result):
                # Declined (1) or not (0); a failed answer is no decline.
                values[ABSTENTION_RECALL][result.query_id].append(
                    float(_declined(result))
                )
    return {
        key: {qid: float(np.mean(v)) for qid, v in by_query.items()}
        for key, by_query in sorted(values.items())
    }


# Resampled values drawn at once by ``bootstrap_ci`` before it draws them in
# chunks of rows (bounds its memory at the family-scaled resample counts).
_BOOTSTRAP_CHUNK_VALUES = 4_000_000


def bootstrap_ci(
    deltas: Sequence[float], *, resamples: int, confidence: float, seed: int
) -> tuple[float, float]:
    """Percentile CI of the mean of ``deltas`` (queries resampled)."""
    return bootstrap_cis(
        [deltas], resamples=resamples, confidence=confidence, seed=seed
    )[0]


def bootstrap_cis(
    samples: Sequence[Sequence[float]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> list[tuple[float, float]]:
    """``bootstrap_ci`` of each of ``samples``, the same intervals faster:
    samples of one size draw the same resampled indices from the seed, so
    each chunk of them is drawn once for all of that size (drawing them is
    most of the cost)."""
    out: list[tuple[float, float] | None] = [None] * len(samples)
    by_size: dict[int, list[int]] = defaultdict(list)
    arrays = [np.asarray(sample, dtype=float) for sample in samples]
    for index, data in enumerate(arrays):
        if data.size == 1:
            out[index] = (float(data[0]), float(data[0]))
        else:
            by_size[data.size].append(index)
    tail = (1.0 - confidence) / 2 * 100
    for size, members in by_size.items():
        rng = np.random.default_rng(seed)
        rows = max(1, _BOOTSTRAP_CHUNK_VALUES // size)
        means: dict[int, list[np.ndarray]] = {i: [] for i in members}
        for start in range(0, resamples, rows):
            drawn = rng.integers(0, size, size=(min(rows, resamples - start), size))
            for i in members:
                means[i].append(arrays[i][drawn].mean(axis=1))
        for i in members:
            low, high = np.percentile(np.concatenate(means[i]), [tail, 100 - tail])
            out[i] = (float(low), float(high))
    return [interval for interval in out if interval is not None]


# Resamples per test a family of m tests at level alpha needs: at least
# this many times m / alpha. The Monte Carlo sign-flip p-value is never below
# 1 / (1 + resamples), and Holm multiplies the smallest by m, so with 2,000
# resamples a family of 100 tests could not reject at 0.05 however large the
# difference; at 20 m / alpha the floor times m is alpha / 20. The family's
# Bonferroni interval reads the alpha / (2 m) tails, which then hold at least
# ten resampled means each instead of a fraction of one.
FAMILY_RESAMPLES_FACTOR = 20


def family_resamples(resamples: int, tests: int, alpha: float) -> int:
    """The resamples a family of ``tests`` tests at ``alpha`` reads its
    p-values and intervals from: ``resamples``, or
    ``FAMILY_RESAMPLES_FACTOR * tests / alpha`` when that is more."""
    if tests <= 0 or alpha <= 0:
        return resamples
    return max(resamples, math.ceil(FAMILY_RESAMPLES_FACTOR * tests / alpha))


# Rows of sign patterns drawn at a time by ``sign_flip_p_value`` (bounds its
# memory on large datasets).
_SIGN_FLIP_CHUNK = 256


def sign_flip_p_value(deltas: Sequence[float], *, resamples: int, seed: int) -> float:
    """Two-sided p-value of the mean of paired ``deltas`` under "no
    difference": a sign-flip randomization test.

    Under the null each query's delta is as likely negative as positive, so
    the observed sum of the non-zero deltas is compared with the sums under
    random signs: exactly over every sign pattern when there are at most
    ``log2(resamples)`` of them, else over ``resamples`` random patterns
    (``(1 + hits) / (1 + resamples)``, never 0). Ties (zero deltas) carry no
    sign. Unlike the percentile bootstrap it holds its level on few pairs
    (five wins of five: p = 0.0625, the exact sign test).
    """
    return sign_flip_p_values([deltas], resamples=resamples, seed=seed)[0]


def sign_flip_p_values(
    samples: Sequence[Sequence[float]], *, resamples: int, seed: int
) -> list[float]:
    """``sign_flip_p_value`` of each of ``samples``, the same p-values
    faster: samples with as many non-zero deltas draw the same sign
    patterns from the seed, so each chunk of them is drawn once for all of
    them (drawing them is most of the cost)."""
    out: list[float] = [1.0] * len(samples)
    by_size: dict[int, list[int]] = defaultdict(list)
    nonzeros: list[np.ndarray] = []
    observed: list[float] = []
    tolerances: list[float] = []
    for index, sample in enumerate(samples):
        data = np.asarray(sample, dtype=float)
        nonzero = data[data != 0]
        nonzeros.append(nonzero)
        observed.append(abs(float(nonzero.sum())))
        # Float sums of the same values in another order differ in the
        # last bits.
        tolerances.append(1e-9 * max(1.0, float(np.abs(nonzero).sum())))
        if nonzero.size == 0:
            continue
        if 2**nonzero.size <= resamples:
            patterns = (
                (np.arange(2**nonzero.size)[:, None] >> np.arange(nonzero.size)) & 1
            ) * 2.0 - 1.0
            sums = np.abs(patterns @ nonzero)
            out[index] = float(
                np.count_nonzero(sums >= observed[index] - tolerances[index])
                / sums.size
            )
        else:
            by_size[nonzero.size].append(index)
    for size, members in by_size.items():
        rng = np.random.default_rng(seed)
        hits = dict.fromkeys(members, 0)
        for start in range(0, resamples, _SIGN_FLIP_CHUNK):
            rows = min(_SIGN_FLIP_CHUNK, resamples - start)
            signs = rng.choice((-1.0, 1.0), size=(rows, size))
            for i in members:
                hits[i] += int(
                    np.count_nonzero(
                        np.abs(signs @ nonzeros[i]) >= observed[i] - tolerances[i]
                    )
                )
        for i in members:
            out[i] = (1 + hits[i]) / (1 + resamples)
    return out


def holm_adjusted(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjusted p-values, in the order given: the i-th
    smallest is ``max over j <= i of (m - j + 1) * p_(j)``, capped at 1.
    Rejecting where the adjusted value is below alpha keeps the chance of
    any false rejection in the family at alpha or less."""
    m = len(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(sorted(range(m), key=lambda i: p_values[i])):
        running = max(running, min(1.0, (m - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def _delta_stats(
    base: Mapping[str, float],
    other: Mapping[str, float],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any] | None:
    """Paired delta of ``other`` against ``base`` over the queries both
    scored; ``None`` when they share none."""
    paired = sorted(set(base) & set(other))
    if not paired:
        return None
    diff = [other[q] - base[q] for q in paired]
    low, high = bootstrap_ci(
        diff, resamples=resamples, confidence=confidence, seed=seed
    )
    return {
        "pairs": len(paired),
        # Scored by one side only: a metric skipped there (no expectation,
        # no source of that type, a metric the strategy never produces).
        "unpaired": len(set(base) ^ set(other)),
        "mean_delta": float(np.mean(diff)),
        "ci_low": low,
        "ci_high": high,
        # Unadjusted; adjust_for_multiplicity adds the family's Holm value.
        "p_value": sign_flip_p_value(diff, resamples=resamples, seed=seed),
        "wins": sum(d > 0 for d in diff),
        "ties": sum(d == 0 for d in diff),
        "losses": sum(d < 0 for d in diff),
        "per_query": dict(zip(paired, diff, strict=True)),
    }


def _decline_shares(
    strategy_runs: Sequence[Run],
) -> tuple[dict[str, float], set[str]]:
    """Per query, the share of the runs whose answer declined (a failed one
    did not), and the unanswerable queries."""
    declined: dict[str, list[float]] = defaultdict(list)
    unanswerable: set[str] = set()
    for _, results in strategy_runs:
        for result in results:
            declined[result.query_id].append(float(_declined(result)))
            if _unanswerable(result):
                unanswerable.add(result.query_id)
    return {q: float(np.mean(v)) for q, v in declined.items()}, unanswerable


def _precision(shares: Mapping[str, float], unanswerable: set[str]) -> float | None:
    """Declines on unanswerable queries over all declines (``None``: none)."""
    total = sum(shares.values())
    if not total:
        return None
    return sum(v for q, v in shares.items() if q in unanswerable) / total


def _precision_delta(
    base: tuple[dict[str, float], set[str]],
    other: tuple[dict[str, float], set[str]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
) -> dict[str, Any] | None:
    """Paired delta of the decline precision over the queries both answered.

    Precision is a ratio over the declines, not a mean over queries, so the
    interval resamples the queries (as ``bootstrap_ci``) and recomputes both
    sides' precision on each resample; a resample where one side declined
    nothing is left out. ``pairs`` counts the queries either side declined:
    the ones the ratio rests on. ``None`` when either side declined none.
    """
    paired = sorted(set(base[0]) & set(other[0]))
    unanswerable = base[1] | other[1]
    one = np.asarray([base[0][q] for q in paired], dtype=float)
    two = np.asarray([other[0][q] for q in paired], dtype=float)
    hit = np.asarray([q in unanswerable for q in paired], dtype=float)
    if not paired or not one.sum() or not two.sum():
        return None
    point = float((two * hit).sum() / two.sum() - (one * hit).sum() / one.sum())
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(paired), size=(resamples, len(paired)))
    one_total, two_total = one[index].sum(axis=1), two[index].sum(axis=1)
    valid = (one_total > 0) & (two_total > 0)
    diffs = (two * hit)[index].sum(axis=1)[valid] / two_total[valid] - (one * hit)[
        index
    ].sum(axis=1)[valid] / one_total[valid]
    tail = (1.0 - confidence) / 2 * 100
    low, high = np.percentile(diffs, [tail, 100 - tail]) if diffs.size else (point,) * 2
    return {
        "pairs": int(((one > 0) | (two > 0)).sum()),
        "unpaired": len(set(base[0]) ^ set(other[0])),
        "mean_delta": point,
        "ci_low": float(low),
        "ci_high": float(high),
    }


def paired_comparison(
    runs: Mapping[str, Sequence[Run]],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    categories: Mapping[str, str] | None = None,
    prices: Mapping[str, ModelPrice] | None = None,
    unmetered_rerank: Collection[str] = (),
) -> dict[str, Any]:
    """Compare strategies run over the same queries (the first is the baseline).

    ``runs`` maps each strategy, in the order given, to its repeats;
    ``categories`` (query id to category) adds ``deltas_by_category``;
    ``prices`` (model id to price) adds a USD estimate to ``cost``, with no
    USD delta between a strategy of ``unmetered_rerank`` (whose usage
    records count no rerank request) and one whose records count them.
    """
    comparison = _paired_comparison(
        runs,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
        categories=categories,
        prices=prices,
        unmetered_rerank=unmetered_rerank,
    )
    for warning in comparison["warnings"]:
        logger.warning("%s", warning)
    return comparison


def baseline_comparison(
    baselines: Mapping[str, Sequence[Run]],
    runs: Mapping[str, Sequence[Run]],
    pairs: Mapping[str, str],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    categories: Mapping[str, str] | None = None,
    prices: Mapping[str, ModelPrice] | None = None,
    unmetered_rerank: Collection[str] = (),
) -> dict[str, Any]:
    """Compare each strategy of ``runs`` with its own baseline.

    ``pairs`` maps each strategy to the label of its baseline in
    ``baselines``. Each pair is one ``paired_comparison``; the result has
    its shape, with every pair's baseline and strategy in ``strategies``,
    each strategy's deltas against its baseline, and ``baselines``
    (strategy to baseline label). ``baseline`` is the first pair's.
    """
    parts = {
        strategy: _paired_comparison(
            {pairs[strategy]: baselines[pairs[strategy]], strategy: strategy_runs},
            resamples=resamples,
            confidence=confidence,
            seed=seed,
            categories=categories,
            prices=prices,
            unmetered_rerank=unmetered_rerank,
        )
        for strategy, strategy_runs in runs.items()
    }
    first = next(iter(parts.values()))
    summary: dict[str, Any] = {}
    cost_strategies: dict[str, Any] = {}
    for strategy, part in parts.items():
        for name in (pairs[strategy], strategy):
            summary.setdefault(name, part["strategies"][name])
            cost_strategies.setdefault(name, part["cost"]["strategies"][name])
    comparison: dict[str, Any] = {
        "baseline": first["baseline"],
        "baselines": dict(pairs),
        "strategies": summary,
        "deltas_vs_baseline": {
            s: part["deltas_vs_baseline"][s] for s, part in parts.items()
        },
        "cost": {
            "strategies": cost_strategies,
            "deltas_vs_baseline": {
                s: part["cost"]["deltas_vs_baseline"][s] for s, part in parts.items()
            },
            "prices": first["cost"]["prices"],
        },
        "warnings": list(
            dict.fromkeys(w for part in parts.values() for w in part["warnings"])
        ),
        "bootstrap": first["bootstrap"],
    }
    if categories:
        comparison["deltas_by_category"] = {
            s: part["deltas_by_category"][s] for s, part in parts.items()
        }
    for warning in comparison["warnings"]:
        logger.warning("%s", warning)
    return comparison


def _paired_comparison(
    runs: Mapping[str, Sequence[Run]],
    *,
    resamples: int,
    confidence: float,
    seed: int,
    categories: Mapping[str, str] | None,
    prices: Mapping[str, ModelPrice] | None,
    unmetered_rerank: Collection[str] = (),
) -> dict[str, Any]:
    """``paired_comparison`` without logging its warnings."""
    strategies = list(runs)
    baseline = strategies[0]
    failed_scores = _strategy_failed_scores(runs, _scored_metrics(runs))
    means = {s: per_query_means(runs[s], failed_scores[s]) for s in strategies}
    declines = {s: _decline_shares(runs[s]) for s in strategies}
    summary: dict[str, Any] = {}
    for strategy in strategies:
        run_means = [
            per_query_means([run], failed_scores[strategy]) for run in runs[strategy]
        ]
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
            # The structured strategy's run records (over every repeat).
            "structured_statistics": structured_statistics(
                result for _, results in runs[strategy] for result in results
            ),
            # The iterative strategy's run records (over every repeat).
            "iterative_statistics": iterative_statistics(
                result for _, results in runs[strategy] for result in results
            ),
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
            # Scores an evaluator tried and could not compute (a judge call
            # that failed or returned NaN), over every repeat: such a query
            # is unpaired for that metric, not scored.
            "failed_metrics": _failed_metric_counts(runs[strategy]),
        }
        if declines[strategy][1]:
            per_repeat_precision = [
                p
                for p in (_precision(*_decline_shares([run])) for run in runs[strategy])
                if p is not None
            ]
            precision = _precision(*declines[strategy])
            if precision is not None:
                metrics[ABSTENTION_PRECISION] = {
                    "mean": precision,
                    # The queries the ratio rests on: those declined.
                    "queries": sum(v > 0 for v in declines[strategy][0].values()),
                    "repeat_stdev": (
                        float(np.std(per_repeat_precision, ddof=1))
                        if len(per_repeat_precision) > 1
                        else None
                    ),
                }
        fallbacks = _span_fallbacks(runs[strategy])
        if fallbacks is not None:
            # Extracted spans that fell back to the answer's first line.
            summary[strategy]["answer_span_fallbacks"] = fallbacks

    stats: dict[str, Any] = {
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
    }
    deltas: dict[str, Any] = {}
    for strategy in strategies[1:]:
        by_metric: dict[str, Any] = {}
        for key, base in means[baseline].items():
            delta = _delta_stats(base, means[strategy].get(key, {}), **stats)
            if delta is not None:
                by_metric[key] = delta
        if declines[baseline][1] or declines[strategy][1]:
            precision_delta = _precision_delta(
                declines[baseline], declines[strategy], **stats
            )
            if precision_delta is not None:
                by_metric[ABSTENTION_PRECISION] = precision_delta
        deltas[strategy] = by_metric
    cost = cost_comparison(
        runs, prices=prices, unmetered_rerank=unmetered_rerank, **stats
    )
    warnings = [
        *_failure_warnings(summary),
        *_span_warnings(runs),
        *_span_fallback_warnings(summary),
        *cost.pop("warnings"),
    ]
    comparison: dict[str, Any] = {
        "baseline": baseline,
        "baselines": dict.fromkeys(strategies[1:], baseline),
        "strategies": summary,
        "deltas_vs_baseline": deltas,
        "cost": cost,
        "warnings": warnings,
        "bootstrap": dict(stats),
    }
    if categories:
        comparison["deltas_by_category"] = _category_deltas(
            means, baseline, categories, stats
        )
    return comparison


def _category_deltas(
    means: Mapping[str, dict[str, dict[str, float]]],
    baseline: str,
    categories: Mapping[str, str],
    stats: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{strategy: {category: {metric: delta}}}`` over each category's queries."""
    by_category: dict[str, dict[str, dict[str, Any]]] = {}
    for strategy, strategy_means in list(means.items())[1:]:
        per_category: dict[str, dict[str, Any]] = {}
        for category in sorted(set(categories.values())):
            in_category = {q for q, c in categories.items() if c == category}
            by_metric: dict[str, Any] = {}
            for key, base in means[baseline].items():
                delta = _delta_stats(
                    {q: v for q, v in base.items() if q in in_category},
                    {
                        q: v
                        for q, v in strategy_means.get(key, {}).items()
                        if q in in_category
                    },
                    **stats,
                )
                if delta is not None:
                    by_metric[key] = delta
            if by_metric:
                per_category[category] = by_metric
        by_category[strategy] = per_category
    return by_category


def _span_warnings(runs: Mapping[str, Sequence[Run]]) -> list[str]:
    """A warning when the runs' exact match and token F1 read different
    answer spans (the ``answer_span`` their answer_match reports record)."""
    spans: dict[str, set[str]] = defaultdict(set)
    for strategy, strategy_runs in runs.items():
        for reports, _ in strategy_runs:
            for report in reports:
                span = report.metadata.get(ANSWER_SPAN_KEY)
                if span and report.evaluator_type is EvaluatorType.ANSWER_MATCH:
                    spans[str(span)].add(strategy)
    if len(spans) < 2:
        return []
    listed = "; ".join(
        f"{span}: {', '.join(sorted(s))}" for span, s in sorted(spans.items())
    )
    return [
        f"The runs scored exact_match and token_f1 on different answer spans "
        f"({listed}); those deltas compare the spans, not the strategies."
    ]


def _span_fallbacks(strategy_runs: Sequence[Run]) -> dict[str, int] | None:
    """The answer_match reports scored on an extracted span and those of them
    that fell back to the first line; ``None`` without extracted spans."""
    scored = [
        report
        for reports, _ in strategy_runs
        for report in reports
        if report.evaluator_type is EvaluatorType.ANSWER_MATCH
        and report.metrics
        and report.metadata.get(ANSWER_SPAN_KEY) == "extracted"
    ]
    if not scored:
        return None
    return {
        "scored": len(scored),
        "fallbacks": sum(SPAN_FALLBACK_KEY in r.metadata for r in scored),
    }


def _span_fallback_warnings(summary: Mapping[str, Any]) -> list[str]:
    """Warnings on extracted spans that fell back to the first line: per
    strategy that had any, and when the strategies' fallback rates differ."""
    rates = {
        s: stats["answer_span_fallbacks"]
        for s, stats in summary.items()
        if "answer_span_fallbacks" in stats
    }
    warnings = [
        f"{s}: {f['fallbacks']}/{f['scored']} extracted answer spans fell back "
        "to the answer's first line (the span model failed or gave no span "
        "found in the answer), so exact_match and token_f1 read the first "
        "line there."
        for s, f in rates.items()
        if f["fallbacks"]
    ]
    shares = {s: f["fallbacks"] / f["scored"] for s, f in rates.items()}
    if len(set(shares.values())) > 1:
        listed = ", ".join(f"{s} {share:.0%}" for s, share in shares.items())
        warnings.append(
            f"The strategies' extracted-span fallback rates differ ({listed}): "
            "their exact_match and token_f1 deltas compare the fallbacks as "
            "well as the strategies."
        )
    return warnings


def _failed_metric_counts(strategy_runs: Sequence[Run]) -> dict[str, int]:
    """``{"evaluator/metric": reports that list it as failed}`` over the runs."""
    counts: Counter[str] = Counter()
    for reports, _ in strategy_runs:
        for report in reports:
            for metric in report.metadata.get(FAILED_METRICS_KEY) or {}:
                counts[f"{report.evaluator_type.value}/{metric}"] += 1
    return dict(sorted(counts.items()))


def failed_metric_counts_differ(
    summary: Mapping[str, Any], strategies: Sequence[str], key: str
) -> dict[str, int]:
    """Per strategy, the failed scores of metric ``key``, when they differ
    between ``strategies``; else empty."""
    failed = {
        s: int(((summary.get(s) or {}).get("failed_metrics") or {}).get(key, 0))
        for s in dict.fromkeys(strategies)
    }
    return failed if len(set(failed.values())) > 1 else {}


def _failure_warnings(summary: Mapping[str, Any]) -> list[str]:
    """Warnings when the strategies failed, or had blocked, different numbers
    of answers, or failed to score a metric different numbers of times."""
    warnings = []
    failed = {s: stats["failed_answers"] for s, stats in summary.items()}
    if len(set(failed.values())) > 1:
        counts = ", ".join(f"{s} {n}" for s, n in failed.items())
        warnings.append(
            f"Strategies failed different numbers of answers ({counts}); each "
            "failed answer scores 0, so the comparison reflects those failures."
        )
    keys = sorted({k for stats in summary.values() for k in stats["failed_metrics"]})
    for key in keys:
        if differ := failed_metric_counts_differ(summary, list(summary), key):
            counts = ", ".join(f"{s} {n}" for s, n in differ.items())
            warnings.append(
                f"{key} could not be scored on different numbers of answers "
                f"({counts}); those queries are left out of its delta "
                "(unpaired), so a judge that fails where a strategy would "
                "lose hides the loss."
            )
    blocked = {s: stats["guardrail_blocked_counts"] for s, stats in summary.items()}
    if len({tuple(c.items()) for c in blocked.values()}) > 1:
        counts = ", ".join(f"{s} {sum(c.values())}" for s, c in blocked.items())
        warnings.append(
            f"The guardrail blocked different numbers of answers ({counts}); "
            "the answer metrics score a blocked answer on its blocked "
            "message, and an answer blocked on output has no retrieval or "
            "context score (unpaired), so the comparison reflects those "
            "blocks."
        )
    return warnings


# The per-answer cost measures, keyed as in cost.deltas_vs_baseline (and
# run-eval --fail-if): the usage counters and the answer latency.
_USAGE_MEASURES: dict[str, str] = {
    "usage/llm_calls": "llm_calls",
    "usage/input_tokens": "input_tokens",
    "usage/output_tokens": "output_tokens",
    "usage/total_tokens": "total_tokens",
}
LATENCY_MEASURE = "latency/response_time_seconds"
USD_MEASURE = "usage/cost_usd"
# What a run's usage records count (the run manifest's ``usage_schema``):
# 1, the model calls and tokens (a run saved before it was recorded); 2, the
# rerank requests too (``rerank_calls``, ``rerank_search_units``), which a
# USD estimate prices; 3, the search units counted as Bedrock bills them
# (a document over a chunk's token limit counts as several), where 2 counted
# one per 100 documents and undercounted long ones.
USAGE_SCHEMA = 3


def answer_cost_usd(
    usage: LLMUsage, prices: Mapping[str, ModelPrice]
) -> tuple[float | None, set[str]]:
    """The USD cost of one answer's model calls, and the models without a
    price (the cost is then ``None``)."""
    unpriced = {
        model
        for model, counts in usage.by_model.items()
        if (counts.llm_calls and _token_rates(prices.get(model)) is None)
        or (
            counts.rerank_calls
            and (
                model not in prices
                or prices[model].per_thousand_rerank_search_units is None
            )
        )
    }
    if unpriced:
        return None, unpriced
    usd = sum(
        _counts_usd(counts, prices[model])
        for model, counts in usage.by_model.items()
        if model in prices
    )
    return float(usd), set()


def _counts_usd(counts: LLMUsageCounts, price: ModelPrice) -> float:
    """The USD of one model's counts: ``input_tokens`` includes the cache
    reads and writes, each priced at its own rate when ``price`` has one;
    a rerank model's search units at their own price."""
    cached = counts.cache_read_input_tokens + counts.cache_write_input_tokens
    rate, output = _token_rates(price) or (0.0, 0.0)
    read = price.cache_read_per_million_tokens
    write = price.cache_write_per_million_tokens
    units = price.per_thousand_rerank_search_units or 0.0
    return (
        max(0, counts.input_tokens - cached) * rate
        + counts.cache_read_input_tokens * (rate if read is None else read)
        + counts.cache_write_input_tokens * (rate if write is None else write)
        + counts.output_tokens * output
    ) / 1e6 + counts.rerank_search_units * units / 1e3


def _token_rates(price: ModelPrice | None) -> tuple[float, float] | None:
    """A price's (input, output) USD per million tokens; ``None`` when it
    has none (a rerank model's, or no price)."""
    if price is None:
        return None
    rates = (price.input_per_million_tokens, price.output_per_million_tokens)
    if rates[0] is None or rates[1] is None:
        return None
    return rates[0], rates[1]


def _distribution(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    data = np.asarray(values, dtype=float)
    return {
        "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
    }


def _answer_measures(
    result: EvaluationResult, prices: Mapping[str, ModelPrice] | None
) -> tuple[dict[str, float], set[str]]:
    """One answer's cost measures (those it has) and its unpriced models."""
    measures: dict[str, float] = {}
    unpriced: set[str] = set()
    if result.usage is not None:
        for key, name in _USAGE_MEASURES.items():
            measures[key] = float(getattr(result.usage, name))
        if prices:
            usd, unpriced = answer_cost_usd(result.usage, prices)
            if usd is not None:
                measures[USD_MEASURE] = usd
    if result.response_time is not None:
        measures[LATENCY_MEASURE] = float(result.response_time)
    return measures, unpriced


# A difference in cache hits per answer above this is warned about.
_CACHE_HIT_TOLERANCE = 0.05


def cost_comparison(
    runs: Mapping[str, Sequence[Run]],
    *,
    prices: Mapping[str, ModelPrice] | None = None,
    unmetered_rerank: Collection[str] = (),
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """The comparison's ``cost`` block: per strategy, the answers' model calls,
    tokens, latency (and USD, with ``prices``) per query as mean / p50 / p95
    over every answer of every repeat, and against the first strategy the
    paired per-query deltas (each query's mean over its repeats) with a
    bootstrap interval, as for the metrics. Also ``warnings`` (popped by
    ``paired_comparison``): strategies without usage records next to some
    with them or with some answers unmeasured, failed answers in the
    measures, repeat counts or cache hits per answer that differ from the
    baseline's, unpriced models, and strategies whose usage records do not
    meter the rerank requests as billed (``unmetered_rerank``: saved before
    ``USAGE_SCHEMA`` 3) next to some that do: the USD delta between two
    such is left out, since one side's estimate prices no rerank.

    A failed answer counts with the usage and time it spent before it failed
    (``failed_answers``; ``EvaluationManager`` records both per answer). An
    answer without a usage record or a latency (results saved before usage
    accounting, or before failed answers kept theirs) has none of those
    measures: ``answers_without_usage`` / ``answers_without_latency``, and a
    ``--fail-if`` on such a measure fails (``gates``).
    """
    stats: dict[str, Any] = {
        "resamples": resamples,
        "confidence": confidence,
        "seed": seed,
    }
    strategies: dict[str, Any] = {}
    per_query: dict[str, dict[str, dict[str, float]]] = {}
    unpriced: set[str] = set()
    without_usage: list[str] = []
    for strategy, strategy_runs in runs.items():
        observed: dict[str, list[float]] = defaultdict(list)
        by_query: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        cache_hits = answers = failed = 0
        strategy_unpriced: set[str] = set()
        for _, results in strategy_runs:
            for result in results:
                answers += 1
                failed += result.error
                measures, missing = _answer_measures(result, prices)
                strategy_unpriced |= missing
                if result.usage is not None:
                    cache_hits += result.usage.cache_hits
                for key, value in measures.items():
                    observed[key].append(value)
                    by_query[key][result.query_id].append(value)
        unpriced |= strategy_unpriced
        with_usage = len(observed.get("usage/llm_calls", []))
        with_latency = len(observed.get(LATENCY_MEASURE, []))
        if not with_usage:
            without_usage.append(strategy)
        per_query[strategy] = {
            key: {q: float(np.mean(v)) for q, v in values.items()}
            for key, values in by_query.items()
        }
        totals = {
            name: int(np.sum(observed.get(key, [])))
            for key, name in _USAGE_MEASURES.items()
        }
        strategies[strategy] = {
            "answers": answers,
            "answers_with_usage": with_usage,
            "answers_without_usage": answers - with_usage,
            "answers_without_latency": answers - with_latency,
            # Counted in every measure with what they spent before failing.
            "failed_answers": failed,
            "repeats": len(strategy_runs),
            "cache_hits": cache_hits,
            "llm_calls_per_query": _distribution(observed.get("usage/llm_calls", [])),
            "input_tokens_per_query": _distribution(
                observed.get("usage/input_tokens", [])
            ),
            "output_tokens_per_query": _distribution(
                observed.get("usage/output_tokens", [])
            ),
            "total_tokens_per_query": _distribution(
                observed.get("usage/total_tokens", [])
            ),
            "latency_seconds": _distribution(observed.get(LATENCY_MEASURE, [])),
            # Over the answers every model of which has a price; the total
            # only when every answer's models have one.
            "usd_per_query": _distribution(observed.get(USD_MEASURE, [])),
            "usd_total": (
                float(np.sum(observed.get(USD_MEASURE, [])))
                if prices and not strategy_unpriced
                else None
            ),
            "totals": totals,
        }
    names = list(runs)
    unmetered = [s for s in names if s in unmetered_rerank]
    deltas: dict[str, Any] = {}
    for strategy in names[1:]:
        by_measure: dict[str, Any] = {}
        # The rerank requests priced on one side only.
        mixed = (strategy in unmetered) != (names[0] in unmetered)
        for key, base in per_query[names[0]].items():
            if mixed and key == USD_MEASURE:
                continue
            delta = _delta_stats(base, per_query[strategy].get(key, {}), **stats)
            if delta is not None:
                by_measure[key] = delta
        deltas[strategy] = by_measure
    warnings = []
    if prices and unmetered and len(unmetered) < len(names):
        warnings.append(
            f"The usage records of {', '.join(unmetered)} do not count rerank "
            f"search units as billed (saved before usage_schema {USAGE_SCHEMA}): "
            "their USD estimate leaves the rerank out or undercounts it, and "
            "no USD delta is given against a strategy whose records count "
            "them. Run them again to compare USD."
        )
    # Some strategies with usage and some without: their usage is not
    # paired (none with it at all is no warning, only no usage measures).
    if without_usage and len(without_usage) < len(names):
        warnings.append(
            f"No usage records in the results of {', '.join(without_usage)} "
            "(saved before usage accounting, or every answer failed): their "
            "usage is not compared."
        )
    warnings.extend(_measure_warnings(strategies))
    warnings.extend(_cache_warnings(strategies, names))
    if unpriced:
        warnings.append(
            "evaluation.prices has no price for "
            f"{', '.join(sorted(unpriced))}: the answers that used them have "
            "no USD estimate."
        )
    return {
        "strategies": strategies,
        "deltas_vs_baseline": deltas,
        "prices": (
            {model: price.model_dump() for model, price in sorted(prices.items())}
            if prices
            else None
        ),
        "warnings": warnings,
    }


def _measure_warnings(strategies: Mapping[str, Any]) -> list[str]:
    """Warnings on the answers a strategy's cost measures leave out, or
    count although they failed."""
    warnings = []
    for strategy, stats in strategies.items():
        # Some answers measured and some not (none at all is no measure of
        # that kind, and with usage the warning above).
        missing = [
            f"{stats[key]} without {what}"
            for key, what in (
                ("answers_without_usage", "a usage record"),
                ("answers_without_latency", "a latency"),
            )
            if 0 < stats[key] < stats["answers"]
        ]
        if missing:
            warnings.append(
                f"{strategy}: {' and '.join(missing)} of {stats['answers']} "
                "answers (results saved before failed answers kept their "
                "usage and time): its cost measures leave them out, and a "
                "--fail-if on usage/* or latency/* fails."
            )
    failed = {s: stats["failed_answers"] for s, stats in strategies.items()}
    if any(failed.values()):
        counts = ", ".join(
            f"{s} {n}/{strategies[s]['answers']}" for s, n in failed.items()
        )
        warnings.append(
            f"Failed answers count in the cost measures with the calls, "
            f"tokens and time they spent before failing ({counts}): a fast "
            "failure lowers a strategy's cost, a timeout raises its latency."
        )
    return warnings


def _cache_warnings(strategies: Mapping[str, Any], names: Sequence[str]) -> list[str]:
    """Warnings when a strategy ran a different number of repeats than the
    baseline, or hit the query extraction cache at another rate per answer:
    a later repeat that reuses the earlier ones' extractions
    (``evaluation.repeat_extractions: reuse``), or a run that shares the
    cache with another, makes the usage deltas measure the cache as well as
    the strategy."""
    if len(names) < 2:
        return []
    baseline = strategies[names[0]]
    warnings = []
    repeats = {s: strategies[s]["repeats"] for s in names}
    if len(set(repeats.values())) > 1:
        listed = ", ".join(f"{s} {n}" for s, n in repeats.items())
        warnings.append(
            f"The compared strategies ran different numbers of repeats "
            f"({listed}): their means average different numbers of runs, and "
            "a later repeat may reuse the query extractions of the earlier "
            "ones (cache_hits, with --repeat-extractions reuse), so the usage "
            "deltas reflect the repeat counts too. Compare runs made with the "
            "same --repeats."
        )

    def rate(stats: Mapping[str, Any]) -> float:
        return stats["cache_hits"] / stats["answers"] if stats["answers"] else 0.0

    differing = [
        s
        for s in names[1:]
        if abs(rate(strategies[s]) - rate(baseline)) > _CACHE_HIT_TOLERANCE
    ]
    if differing:
        listed = ", ".join(
            f"{s} {rate(strategies[s]):.2f}" for s in [names[0], *differing]
        )
        warnings.append(
            f"Cache hits per answer differ ({listed}): a cache hit spends no "
            "call or token, so the usage deltas reflect the query extraction "
            "cache (repeats, earlier runs in the process, or a strategy that "
            "extracts nothing) as well as the strategy. For cold-cache cost, "
            "set search.query_extraction_cache_size: 0."
        )
    return warnings


# The metric a comparison tests unadjusted when answers are scored and no
# other is declared (evaluation.primary_metric, run-eval --primary-metric):
# the answer_match headline, free and judge-free.
DEFAULT_PRIMARY_METRIC = "answer_match/answer_contains"


def adjust_for_multiplicity(
    comparison: dict[str, Any],
    *,
    primary_metric: str | None = None,
    exclude: Collection[str] = (),
) -> dict[str, Any]:
    """Add Holm-adjusted p-values and intervals to ``comparison``'s deltas.

    A comparison reads many deltas, and each 95% interval excludes 0 on
    noise alone in 5% of runs: over ten metrics a clean change shows a
    "supported" difference in 30-40% of runs. So the deltas fall into
    families, each adjusted over its tests (every compared strategy but
    ``exclude``, the closed-book arms, times its measures):

    - ``primary``: ``primary_metric`` (default ``DEFAULT_PRIMARY_METRIC``
      when a strategy scored it), the one metric declared in advance, so
      adjusted only over the compared strategies (with one, unadjusted);
    - ``secondary``: every other metric of ``deltas_vs_baseline`` with
      per-query deltas (``abstention/precision``, a ratio, has none);
    - ``cost``: the measures of ``cost.deltas_vs_baseline``.

    Each delta gets ``family``, ``adjusted_p_value`` (Holm over the family,
    from the sign-flip ``p_value``) and ``adjusted_ci_low`` /
    ``adjusted_ci_high``: the paired bootstrap interval at confidence
    ``1 - alpha / m`` (Bonferroni over the family's ``m`` tests, so
    simultaneous for the family and a little wider than Holm's decision
    needs: read ``adjusted_p_value`` for whether a difference counts, the
    interval for its size). Both read ``family_resamples`` resamples (at
    least ``FAMILY_RESAMPLES_FACTOR * m / alpha``): a p-value under alpha is
    drawn again from that many sign patterns (``p_value_resamples``), so the
    Monte Carlo floor ``1 / (1 + resamples)`` times ``m`` stays far below
    alpha, and the interval's ``alpha / (2 m)`` tails hold enough resampled
    means. ``deltas_by_category`` stays unadjusted (exploratory). Returns
    ``comparison``, with a ``multiplicity`` block (each family's ``tests``,
    ``interval_confidence`` and ``resamples``).
    """
    stats = comparison.get("bootstrap") or {}
    resamples = int(stats.get("resamples", 2000))
    confidence = float(stats.get("confidence", 0.95))
    seed = int(stats.get("seed", 0))
    alpha = 1.0 - confidence
    deltas = comparison.get("deltas_vs_baseline") or {}
    cost = (comparison.get("cost") or {}).get("deltas_vs_baseline") or {}
    compared = [s for s in deltas if s not in exclude]
    primary: str | None = primary_metric or DEFAULT_PRIMARY_METRIC
    if not any(primary in (deltas.get(s) or {}) for s in compared):
        if primary_metric is not None and compared:
            warning = (
                f"The primary metric '{primary}' has no paired delta in this "
                "comparison: every metric is tested as secondary (Holm over "
                "all of them)."
            )
            logger.warning("%s", warning)
            comparison.setdefault("warnings", []).append(warning)
        primary = None
    families: dict[str, list[dict[str, Any]]] = {
        "primary": [],
        "secondary": [],
        "cost": [],
    }
    for strategy in compared:
        for key, delta in (deltas.get(strategy) or {}).items():
            if "p_value" in delta and "per_query" in delta:
                families["primary" if key == primary else "secondary"].append(delta)
        for delta in (cost.get(strategy) or {}).values():
            if "p_value" in delta and "per_query" in delta:
                families["cost"].append(delta)
    summary: dict[str, Any] = {}
    for name, members in families.items():
        if not members:
            continue
        level = 1.0 - alpha / len(members)
        scaled = family_resamples(resamples, len(members), alpha)
        # Only a p-value under alpha can reject once adjusted (Holm's never
        # falls below the raw one): those are read again from enough sign
        # patterns that the Monte Carlo floor is no bound.
        reread = [
            d for d in members if scaled > resamples and float(d["p_value"]) < alpha
        ]
        p_values = sign_flip_p_values(
            [list(d["per_query"].values()) for d in reread],
            resamples=scaled,
            seed=seed,
        )
        for delta, p_value in zip(reread, p_values, strict=True):
            delta["p_value"] = p_value
            delta["p_value_resamples"] = scaled
        adjusted = holm_adjusted([float(d["p_value"]) for d in members])
        intervals = bootstrap_cis(
            [list(d["per_query"].values()) for d in members],
            resamples=scaled,
            confidence=level,
            seed=seed,
        )
        for delta, value, (low, high) in zip(members, adjusted, intervals, strict=True):
            delta.update(
                family=name,
                adjusted_p_value=value,
                adjusted_ci_low=low,
                adjusted_ci_high=high,
            )
        summary[name] = {
            "tests": len(members),
            "interval_confidence": level,
            "resamples": scaled,
        }
    comparison["multiplicity"] = {
        "method": "holm",
        "p_value": "sign_flip",
        "alpha": alpha,
        "primary_metric": primary,
        "families": summary,
        "unadjusted": sorted(set(deltas) - set(compared)),
    }
    return comparison


def per_query_rows(
    runs: Mapping[str, Sequence[Run]],
    categories: Mapping[str, str] | None = None,
    prices: Mapping[str, ModelPrice] | None = None,
) -> list[dict[str, Any]]:
    """The per-query values behind the comparison, one row per non-baseline
    strategy, measure and query: the baseline's and the strategy's value
    (each the mean over its repeats, a failed answer filled as in
    ``paired_comparison``) and their delta (``None`` when one side has no
    value). Measures are the metrics and the cost measures."""
    strategies = list(runs)
    failed_scores = _strategy_failed_scores(runs, _scored_metrics(runs))
    values: dict[str, dict[str, dict[str, float]]] = {}
    for strategy in strategies:
        values[strategy] = per_query_means(runs[strategy], failed_scores[strategy])
        by_query: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for _, results in runs[strategy]:
            for result in results:
                measures, _ = _answer_measures(result, prices)
                for key, value in measures.items():
                    by_query[key][result.query_id].append(value)
        for key, per_query in by_query.items():
            values[strategy][key] = {q: float(np.mean(v)) for q, v in per_query.items()}
    baseline = strategies[0]
    rows: list[dict[str, Any]] = []
    for strategy in strategies[1:]:
        for key in sorted(set(values[baseline]) | set(values[strategy])):
            base = values[baseline].get(key, {})
            other = values[strategy].get(key, {})
            for query_id in sorted(set(base) | set(other)):
                b, o = base.get(query_id), other.get(query_id)
                rows.append(
                    {
                        "strategy": strategy,
                        "baseline": baseline,
                        "measure": key,
                        "query_id": query_id,
                        "category": (categories or {}).get(query_id),
                        "baseline_value": b,
                        "value": o,
                        "delta": o - b if o is not None and b is not None else None,
                    }
                )
    return rows
