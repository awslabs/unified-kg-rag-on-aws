# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A paired comparison as CSV (per query) and Markdown (summary tables).

``run-eval`` writes both next to ``evaluation_comparison_<stamp>.json``:
the CSV has one row per non-baseline strategy, measure and query
(``comparison.per_query_rows``), for a spreadsheet or a notebook; the
Markdown has the metric, cost and pairwise-judge tables (with the win rates
per answer-length bucket), the gates and the warnings, for a pull request or
a CI job summary. A closed-book arm (``run-eval --closed-book``) is marked
"(no retrieval)".
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


def delta_cell(delta: Mapping[str, Any] | None) -> str:
    """A paired delta as ``mean [ci_low, ci_high] (pairs, unpaired, Holm
    p)``; the counts show how many queries the interval rests on and how
    many one side did not score (skipped, or failed to score), and the
    adjusted p-value whether the difference survives the comparison's many
    deltas (``adjust_for_multiplicity``)."""
    if not delta:
        return "-"
    counts = f"{delta['pairs']} pairs" if "pairs" in delta else ""
    if delta.get("unpaired"):
        counts += f", {delta['unpaired']} unpaired"
    if delta.get("adjusted_p_value") is not None:
        counts += f", Holm p {delta['adjusted_p_value']:.3g}"
    return (
        f"{delta['mean_delta']:+.3f} "
        f"[{delta['ci_low']:+.3f}, {delta['ci_high']:+.3f}]"
        + (f" ({counts})" if counts else "")
    )


def _win_rate(stats: Mapping[str, Any] | None) -> str:
    if not stats or stats.get("candidate_win_rate") is None:
        return "-"
    rate = f"{stats['candidate_win_rate']:.2f}"
    if stats.get("ci_low") is None:
        return rate
    return f"{rate} [{stats['ci_low']:.2f}, {stats['ci_high']:.2f}]"


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


def compared_pairs(comparison: Mapping[str, Any]) -> dict[str, str]:
    """Each compared strategy and its baseline (``baselines``; a comparison
    saved before it was recorded: the one ``baseline``)."""
    baselines = comparison.get("baselines") or {}
    return {
        strategy: baselines.get(strategy, comparison["baseline"])
        for strategy in comparison.get("deltas_vs_baseline") or {}
    }


def baseline_metric_keys(comparison: Mapping[str, Any]) -> list[str]:
    """The metrics the baselines scored, in their order."""
    strategies = comparison["strategies"]
    labels = dict.fromkeys(
        [comparison["baseline"], *compared_pairs(comparison).values()]
    )
    return list(
        dict.fromkeys(
            key
            for label in labels
            for key in strategies.get(label, {}).get("metrics", {})
        )
    )


def comparison_markdown(comparison: Mapping[str, Any]) -> str:
    """The comparison's metric means and deltas, its cost block, the
    pairwise judge's win rates, the gates and the warnings, as Markdown."""
    baseline = comparison["baseline"]
    strategies = comparison["strategies"]
    pairs = compared_pairs(comparison)
    others = list(pairs)
    confidence = comparison.get("bootstrap", {}).get("confidence", 0.95)
    title = (
        f"vs `{baseline}`"
        if len(set(pairs.values())) <= 1
        else "vs each strategy's baseline"
    )
    lines = [f"# Paired comparison {title}", ""]
    closed = set(comparison.get("closed_book_arms") or ())
    if closed:
        lines += [
            f"{', '.join(f'`{a}`' for a in sorted(closed))} answered without "
            "retrieval (the answer model alone): a bound on what the model knows "
            "without the corpus, not a strategy; its retrieval and context "
            "metrics score no sources, and gates skip it.",
            "",
        ]
    rows = []
    primary = (comparison.get("multiplicity") or {}).get("primary_metric")
    for key in baseline_metric_keys(comparison):
        row: list[Any] = [f"`{key}` (primary)" if key == primary else f"`{key}`"]
        for strategy in strategies:
            stats = strategies[strategy]["metrics"].get(key)
            row.append(f"{stats['mean']:.3f}" if stats else "-")
        for strategy in others:
            row.append(delta_cell(comparison["deltas_vs_baseline"][strategy].get(key)))
        rows.append(row)
    header = [
        "Metric",
        *(f"{s} (no retrieval)" if s in closed else s for s in strategies),
        *(f"{s} - {pairs[s]} [{confidence:.0%} CI]" for s in others),
    ]
    lines += _table(header, rows)
    lines += _multiplicity_lines(comparison)

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
                    stats.get("failed_answers", "-"),
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
                "Failed answers (included)",
            ],
            cost_rows,
        )
        measures = sorted(
            {m for d in cost.get("deltas_vs_baseline", {}).values() for m in d}
        )
        if measures and others:
            lines += ["", f"Paired cost deltas {title}:", ""]
            lines += _table(
                ["Measure", *(f"{s} - {pairs[s]}" for s in others)],
                [
                    [
                        f"`{measure}`",
                        *(
                            delta_cell(cost["deltas_vs_baseline"][s].get(measure))
                            for s in others
                        ),
                    ]
                    for measure in measures
                ],
            )

    pairwise = comparison.get("pairwise_vs_baseline") or {}
    if pairwise:
        criteria = list(
            dict.fromkeys(c for judged in pairwise.values() for c in judged["criteria"])
        )
        modes = sorted({str(j.get("mode") or "standard") for j in pairwise.values()})
        lines += [
            "",
            f"## Pairwise judge ({', '.join(modes)} mode; candidate win rate, "
            "ties 0.5)",
            "",
        ]
        lines += _table(
            [
                "Strategy",
                "Judged",
                "Judge failures",
                "Orders agreed",
                *criteria,
            ],
            [
                [
                    f"{strategy} vs {pairs.get(strategy, baseline)}",
                    judged.get("judged", "-"),
                    judged.get("failed", "-"),
                    (
                        f"{judged['position_consistency']:.0%}"
                        if judged.get("position_consistency") is not None
                        else "-"
                    ),
                    *(_win_rate((judged["criteria"] or {}).get(c)) for c in criteria),
                ]
                for strategy, judged in pairwise.items()
            ],
        )
        lines += _length_lines(pairwise, pairs, baseline, criteria)

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


def _multiplicity_lines(comparison: Mapping[str, Any]) -> list[str]:
    """How the deltas were adjusted for the many read
    (``comparison.adjust_for_multiplicity``)."""
    multiplicity = comparison.get("multiplicity") or {}
    families = multiplicity.get("families") or {}
    if not families:
        return []
    primary = multiplicity.get("primary_metric")
    listed = ", ".join(
        f"{name} {family['tests']} ({family['interval_confidence']:.2%} intervals)"
        for name, family in families.items()
    )
    return [
        "",
        (f"Primary metric: `{primary}`. " if primary else "No primary metric. ")
        + "Holm p is each delta's sign-flip p-value adjusted over its family "
        f"of tests ({listed}); a difference counts when it is below "
        f"{multiplicity.get('alpha', 0.05):g}. The intervals in the table are "
        "unadjusted (adjusted_ci_low/high in the JSON are simultaneous over "
        "the family).",
    ]


def _length_lines(
    pairwise: Mapping[str, Any],
    pairs: Mapping[str, str],
    baseline: str,
    criteria: Sequence[str],
) -> list[str]:
    """The pairwise win rates per answer-length bucket, and the correlation
    of the relative length difference with the pair's score."""
    rows: list[list[Any]] = []
    for strategy, judged in pairwise.items():
        length = judged.get("length") or {}
        for bucket, stats in (length.get("buckets") or {}).items():
            per_criterion = stats.get("criteria") or {}
            rows.append(
                [
                    f"{strategy} vs {pairs.get(strategy, baseline)}",
                    bucket,
                    stats.get("pairs", 0),
                    *(
                        (
                            f"{rate:.2f}"
                            if (
                                rate := (per_criterion.get(c) or {}).get(
                                    "candidate_win_rate"
                                )
                            )
                            is not None
                            else "-"
                        )
                        for c in criteria
                    ),
                ]
            )
        correlation = length.get("score_correlation") or {}
        if correlation:
            rows.append(
                [
                    f"{strategy} vs {pairs.get(strategy, baseline)}",
                    "correlation",
                    length.get("pairs", 0),
                    *(
                        (
                            f"{value:+.2f}"
                            if (value := correlation.get(c)) is not None
                            else "-"
                        )
                        for c in criteria
                    ),
                ]
            )
    if not rows:
        return []
    return [
        "",
        "Win rate by answer length (candidate / baseline, estimated tokens); "
        "a rate that rises with the ratio is a length effect. The correlation "
        "row is that of the relative length difference with the pair's score.",
        "",
        *_table(["Strategy", "Length ratio", "Pairs", *criteria], rows),
    ]
