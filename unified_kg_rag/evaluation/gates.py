# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression gates on a paired comparison (``run-eval --fail-if``).

A gate ``<measure>:<stat><op><value>`` reads one statistic of each
non-baseline strategy's paired delta against the baseline and fails the run
when the condition holds: ``answer_match/token_f1:ci_high<-0.02`` fails when
the interval of the token-F1 delta lies wholly below -0.02 (a supported
regression), ``usage/total_tokens:mean_delta>500`` when the answers cost 500
more tokens per query on average. ``<measure>`` is a key of
``deltas_vs_baseline`` (``<evaluator>/<metric>``) or of
``cost.deltas_vs_baseline`` (``usage/llm_calls``, ``usage/input_tokens``,
``usage/output_tokens``, ``usage/total_tokens``, ``usage/cost_usd``,
``latency/response_time_seconds``). A measure a strategy has no delta for
fails the gate too: a regression check must not pass on missing data. So
does a cost measure (``usage/*``, ``latency/*``) some answers of the
strategy or of its baseline have no value of (``cost.strategies.<s>
.answers_without_usage`` / ``answers_without_latency``): a delta over the
answers that have one can hide what the others cost.
"""

from __future__ import annotations

import operator
import re
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict

STATS = ("mean_delta", "ci_low", "ci_high")
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


def gate_breaches(comparison: Mapping[str, Any], gates: list[Gate]) -> list[str]:
    """Why ``comparison`` fails ``gates``: one line per strategy and gate
    that holds (or has no delta, or answers without the cost measure, to
    read); empty when every gate passes."""
    deltas = comparison.get("deltas_vs_baseline") or {}
    cost = (comparison.get("cost") or {}).get("deltas_vs_baseline") or {}
    breaches = []
    for strategy in deltas:
        for gate in gates:
            if unmeasured := _unmeasured(comparison, strategy, gate.measure):
                listed = ", ".join(f"{s} {n}" for s, n in unmeasured.items())
                breaches.append(
                    f"{strategy}: --fail-if {gate}: answers without a value "
                    f"of '{gate.measure}' ({listed}) to check"
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
            observed = float(delta[gate.stat])
            if _OPS[gate.op](observed, gate.value):
                breaches.append(
                    f"{strategy}: {gate.measure} {gate.stat} {observed:+.4g} "
                    f"{gate.op} {gate.value:g}"
                )
    return breaches
