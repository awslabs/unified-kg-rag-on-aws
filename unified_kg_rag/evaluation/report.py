# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A paired comparison as CSV (per query) and Markdown (summary tables).

``run-eval`` writes both next to ``evaluation_comparison_<stamp>.json``:
the CSV has one row per non-baseline strategy, measure and query
(``comparison.per_query_rows``), for a spreadsheet or a notebook; the
Markdown has the metric and cost tables, the gates and the warnings, for a
pull request or a CI job summary.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping, Sequence
from typing import Any

CSV_COLUMNS = (
    "strategy",
    "baseline",
    "measure",
    "query_id",
    "category",
    "baseline_value",
    "value",
    "delta",
)


def comparison_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_value(row.get(key)) for key in CSV_COLUMNS})
    return buffer.getvalue()


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.6g}"
    return "" if value is None else value


def _cell(text: Any) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _delta(delta: Mapping[str, Any] | None) -> str:
    if not delta:
        return "-"
    return (
        f"{delta['mean_delta']:+.3f} "
        f"[{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}]"
    )


def _dist(stats: Mapping[str, float] | None, digits: int = 1) -> str:
    if not stats:
        return "-"
    return " / ".join(f"{stats[k]:.{digits}f}" for k in ("mean", "p50", "p95"))


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    return [
        "| " + " | ".join(_cell(h) for h in header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
        *("| " + " | ".join(_cell(c) for c in row) + " |" for row in rows),
    ]


def comparison_markdown(comparison: Mapping[str, Any]) -> str:
    """The comparison's metric means and deltas, its cost block, the gates
    and the warnings, as Markdown."""
    baseline = comparison["baseline"]
    strategies = comparison["strategies"]
    others = list(strategies)[1:]
    confidence = comparison.get("bootstrap", {}).get("confidence", 0.95)
    lines = [f"# Paired comparison vs `{baseline}`", ""]
    rows = []
    for key in strategies[baseline]["metrics"]:
        row: list[Any] = [f"`{key}`"]
        for strategy in strategies:
            stats = strategies[strategy]["metrics"].get(key)
            row.append(f"{stats['mean']:.3f}" if stats else "-")
        for strategy in others:
            row.append(_delta(comparison["deltas_vs_baseline"][strategy].get(key)))
        rows.append(row)
    header = [
        "Metric",
        *strategies,
        *(f"{s} - {baseline} [{confidence:.0%} CI]" for s in others),
    ]
    lines += _table(header, rows)

    cost = comparison.get("cost") or {}
    if cost.get("strategies"):
        lines += [
            "",
            "## Cost per query (mean / p50 / p95)",
            "",
        ]
        cost_rows = []
        for strategy, stats in cost["strategies"].items():
            cost_rows.append(
                [
                    strategy,
                    _dist(stats.get("llm_calls_per_query")),
                    _dist(stats.get("input_tokens_per_query"), 0),
                    _dist(stats.get("output_tokens_per_query"), 0),
                    _dist(stats.get("latency_seconds"), 2),
                    _dist(stats.get("usd_per_query"), 5),
                ]
            )
        lines += _table(
            [
                "Strategy",
                "LLM calls",
                "Input tokens",
                "Output tokens",
                "Latency (s)",
                "USD",
            ],
            cost_rows,
        )
        measures = sorted(
            {m for d in cost.get("deltas_vs_baseline", {}).values() for m in d}
        )
        if measures and others:
            lines += ["", f"Paired cost deltas vs `{baseline}`:", ""]
            lines += _table(
                ["Measure", *others],
                [
                    [
                        f"`{measure}`",
                        *(
                            _delta(cost["deltas_vs_baseline"][s].get(measure))
                            for s in others
                        ),
                    ]
                    for measure in measures
                ],
            )

    gates = comparison.get("gates") or {}
    if gates.get("conditions"):
        lines += ["", "## Gates", ""]
        lines += [f"- `--fail-if {condition}`" for condition in gates["conditions"]]
        breaches = gates.get("breaches") or []
        lines += ["", f"**FAILED** ({len(breaches)}):" if breaches else "Passed."]
        lines += [f"- {_cell(b)}" for b in breaches]
    if comparison.get("warnings"):
        lines += ["", "## Warnings", ""]
        lines += [f"- {_cell(w)}" for w in comparison["warnings"]]
    return "\n".join(lines) + "\n"
