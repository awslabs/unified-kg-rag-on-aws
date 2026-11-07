# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression gates on a paired comparison (``run-eval --fail-if``).

A gate ``<measure>:<stat><op><value>`` reads one statistic of each
non-baseline strategy's paired delta against the baseline and fails the run
when the condition holds: ``answer_match/token_f1:ci_high<-0.02`` fails when
the interval of the token-F1 delta lies wholly below -0.02 (a supported
regression), ``usage/total_tokens:mean_delta>500`` when the answers cost 500
more tokens per query on average. ``<measure>`` is a key of
``deltas_vs_baseline`` (``<evaluator>/<metric>``, or ``abstention/recall``
and ``abstention/precision`` with unanswerable items) or of
``cost.deltas_vs_baseline`` (``usage/llm_calls``, ``usage/input_tokens``,
``usage/output_tokens``, ``usage/total_tokens``, ``usage/cost_usd``,
``latency/response_time_seconds``). A measure a strategy has no delta for
fails the gate too: a regression check must not pass on missing data. So
does a cost measure (``usage/*``, ``latency/*``) some answers of the
strategy or of its baseline have no value of (``cost.strategies.<s>
.answers_without_usage`` / ``answers_without_latency``): a delta over the
answers that have one can hide what the others cost. And so does a cost
measure when the strategy and its baseline failed different numbers of
answers (``cost.strategies.<s>.failed_answers``): a failed answer counts
with what it spent before failing, so the delta compares the failures too
(a fast failure looks cheap, a timeout slow).

The closed-book arm of ``run-eval --closed-book``
(``closed_book_arms``) is not gated: it answers without retrieval, so it
bounds the answer model's prior knowledge and is expected to score lower.

``<stat>`` is ``mean_delta``, ``ci_low`` or ``ci_high``, or a
multiplicity-adjusted one (``comparison.adjust_for_multiplicity``):
``adjusted_ci_low`` / ``adjusted_ci_high`` (the interval simultaneous over
the delta's family: the primary metric, the other metrics, or the cost
measures), ``p_value`` (the sign-flip test's, unadjusted) or
``adjusted_p_value`` (Holm over the family).
``answer_match/token_f1:adjusted_ci_high<-0.02`` fails only on a
regression that survives the many metrics the comparison reads. A delta
without the statistic (``abstention/precision`` has no adjusted ones) fails
the gate.

A metric gate fails when an evaluator could not score the metric on
different numbers of the strategy's and its baseline's answers
(``strategies.<s>.failed_metrics``, a judge call that failed or returned
NaN): those queries are unpaired, so a judge that fails exactly where a
strategy would lose hides the loss. And any gate fails on a delta over
fewer than ``MIN_GATE_PAIRS`` paired queries: its interval says nothing
(one pair gives a zero-width interval at that pair's delta).
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

from .comparison import failed_metric_counts_differ

STATS = (
    "mean_delta",
    "ci_low",
    "ci_high",
    "p_value",
    "adjusted_p_value",
    "adjusted_ci_low",
    "adjusted_ci_high",
)

# A gated delta over fewer paired queries fails the gate. The bootstrap
# percentile interval means little on a handful of queries: on 1 pair it has
# zero width, and on 5 pairs that all win it is [1, 1] although an exact
# sign test gives p = 2 * 0.5**5 = 0.06. 10 is a round floor at which a
# one-sided result survives that test (9 wins of 10: p = 0.02); real
# comparisons need a few hundred queries (docs/guides/evaluate.md, "When a
# difference counts"). The floor is not where the interval becomes exact:
# the percentile bootstrap is anti-conservative on few pairs. Simulated on
# null deltas (3,000 comparisons each, 2,000 resamples), the 95% interval
# excluded 0 in 8% of them at 10 pairs of 0/1-metric deltas (12% for
# heavy-tailed t3 deltas), 5% from 20-30 pairs (7% for t3 at 30-50), against
# the nominal 5%; a gate read on under ~30 pairs breaches on noise more often
# than its confidence says.
MIN_GATE_PAIRS = 10
_OPS: dict[str, Callable[[float, float], bool]] = {
    "<=": operator.le,
    ">=": operator.ge,
    "<": operator.lt,
    ">": operator.gt,
}
_GATE = re.compile(
    r"^(?P<measure>[^:\s]+):(?P<stat>[a-z_]+)\s*(?P<op><=|>=|<|>)\s*"
    r"(?P<value>[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)$"
)


class Gate(BaseModel):
    """One parsed ``--fail-if`` condition."""

    model_config = ConfigDict(frozen=True)

    measure: str
    stat: str
    op: str
    value: float

    def __str__(self) -> str:
        return f"{self.measure}:{self.stat}{self.op}{self.value:g}"


def parse_gate(text: str) -> Gate:
    """Parse ``<measure>:<stat><op><value>``.

    Raises:
        ValueError: The text is not a gate, or names an unknown statistic.
    """
    match = _GATE.match(text.strip())
    if match is None:
        raise ValueError(
            f"'{text}' is not <measure>:<stat><op><value> (e.g. "
            "answer_match/token_f1:ci_high<-0.02)"
        )
    if match.group("stat") not in STATS:
        raise ValueError(f"'{text}': the statistic must be one of {', '.join(STATS)}")
    return Gate(
        measure=match.group("measure"),
        stat=match.group("stat"),
        op=match.group("op"),
        value=float(match.group("value")),
    )


# The cost-block count of the answers without a value of a measure family.
_UNMEASURED = {
    "usage/": "answers_without_usage",
    "latency/": "answers_without_latency",
}


def _unmeasured(
    comparison: Mapping[str, Any], strategy: str, measure: str
) -> dict[str, int]:
    """Per side (the strategy and its baseline), the answers with no value
    of ``measure``; empty for a metric or when every answer has one."""
    key = next((k for p, k in _UNMEASURED.items() if measure.startswith(p)), None)
    if key is None:
        return {}
    cost = (comparison.get("cost") or {}).get("strategies") or {}
    baseline = (comparison.get("baselines") or {}).get(
        strategy, comparison.get("baseline")
    )
    sides = dict.fromkeys([baseline, strategy])
    return {
        str(side): int((cost.get(side) or {}).get(key) or 0)
        for side in sides
        if (cost.get(side) or {}).get(key)
    }


def _failures_differ(
    comparison: Mapping[str, Any], strategy: str, measure: str
) -> dict[str, int]:
    """Per side, the failed answers, when ``measure`` is a cost measure and
    the strategy and its baseline failed different numbers; else empty."""
    if not any(measure.startswith(prefix) for prefix in _UNMEASURED):
        return {}
    cost = (comparison.get("cost") or {}).get("strategies") or {}
    baseline = (comparison.get("baselines") or {}).get(
        strategy, comparison.get("baseline")
    )
    failed = {
        str(side): int((cost.get(side) or {}).get("failed_answers") or 0)
        for side in dict.fromkeys([baseline, strategy])
    }
    return failed if len(set(failed.values())) > 1 else {}


def _failed_scores_differ(
    comparison: Mapping[str, Any], strategy: str, measure: str
) -> dict[str, int]:
    """Per side, the answers ``measure`` could not be scored on, when it is
    a metric and the counts differ; else empty."""
    if any(measure.startswith(prefix) for prefix in _UNMEASURED):
        return {}
    baseline = (comparison.get("baselines") or {}).get(
        strategy, comparison.get("baseline")
    )
    return failed_metric_counts_differ(
        comparison.get("strategies") or {}, [str(baseline), strategy], measure
    )


def gate_breaches(comparison: Mapping[str, Any], gates: list[Gate]) -> list[str]:
    """Why ``comparison`` fails ``gates``: one line per strategy and gate
    that holds (or has no delta, or answers without the cost measure, to
    read, or failed answers that differ under a cost measure, or failed
    scores that differ under a metric, or too few pairs); empty when every
    gate passes."""
    deltas = comparison.get("deltas_vs_baseline") or {}
    cost = (comparison.get("cost") or {}).get("deltas_vs_baseline") or {}
    # A closed-book arm (run-eval --closed-book) bounds the answer model's
    # prior knowledge; it is expected to score lower and is not gated.
    ungated = set(comparison.get("closed_book_arms") or ())
    breaches = []
    for strategy in deltas:
        if strategy in ungated:
            continue
        for gate in gates:
            if unmeasured := _unmeasured(comparison, strategy, gate.measure):
                listed = ", ".join(f"{s} {n}" for s, n in unmeasured.items())
                breaches.append(
                    f"{strategy}: --fail-if {gate}: answers without a value "
                    f"of '{gate.measure}' ({listed}) to check"
                )
                continue
            if failed := _failures_differ(comparison, strategy, gate.measure):
                listed = ", ".join(f"{s} {n}" for s, n in failed.items())
                breaches.append(
                    f"{strategy}: --fail-if {gate}: failed answers differ "
                    f"({listed}), so the '{gate.measure}' delta compares the "
                    "failures too"
                )
                continue
            if failed := _failed_scores_differ(comparison, strategy, gate.measure):
                listed = ", ".join(f"{s} {n}" for s, n in failed.items())
                breaches.append(
                    f"{strategy}: --fail-if {gate}: '{gate.measure}' could not "
                    f"be scored on different numbers of answers ({listed}), so "
                    "its delta leaves out the queries one side failed to score"
                )
                continue
            delta = (deltas.get(strategy) or {}).get(gate.measure) or (
                cost.get(strategy) or {}
            ).get(gate.measure)
            if delta is None:
                breaches.append(
                    f"{strategy}: --fail-if {gate}: no paired delta of "
                    f"'{gate.measure}' to check"
                )
                continue
            pairs = int(delta.get("pairs") or 0)
            if pairs < MIN_GATE_PAIRS:
                breaches.append(
                    f"{strategy}: --fail-if {gate}: the '{gate.measure}' delta "
                    f"pairs {pairs} queries, fewer than the {MIN_GATE_PAIRS} a "
                    "gate needs to read its interval"
                )
                continue
            if delta.get(gate.stat) is None:
                breaches.append(
                    f"{strategy}: --fail-if {gate}: the '{gate.measure}' delta "
                    f"has no {gate.stat} (it is in no multiplicity family) to "
                    "check"
                )
                continue
            observed = float(delta[gate.stat])
            if _OPS[gate.op](observed, gate.value):
                breaches.append(
                    f"{strategy}: {gate.measure} {gate.stat} {observed:+.4g} "
                    f"{gate.op} {gate.value:g}"
                )
    return breaches
