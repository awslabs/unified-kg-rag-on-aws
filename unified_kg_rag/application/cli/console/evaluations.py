# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Evaluate page's files: ``run-eval`` datasets to pick, and the saved
runs (summaries, comparisons and per-question results) ``run-eval`` wrote,
read back as written."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from unified_kg_rag.application.cli.console.paths import Roots, walk_files
from unified_kg_rag.evaluation.comparison import (
    DEFAULT_PRIMARY_METRIC,
    per_query_means,
)
from unified_kg_rag.evaluation.saved_results import (
    SavedRun,
    group_runs,
    load_saved_runs,
)
from unified_kg_rag.shared.exceptions import DataProcessingError, EvaluationException

MAX_DATASET_BYTES = 50 * 1024 * 1024
MAX_RESULT_BYTES = 200 * 1024 * 1024
# Characters of each answer the per-question view shows.
MAX_ANSWER_CHARS = 2000
# Directories a dataset search skips: outputs, caches and stores.
_SKIPPED = {"cache", "outputs", "stores", "node_modules", "uploads"}


class EvaluationFileError(DataProcessingError):
    """A dataset or result file that cannot be read."""


def _read_json(path: Path, limit: int) -> Any:
    if path.stat().st_size > limit:
        raise EvaluationFileError(f"'{path.name}' is over {limit // 2**20} MiB")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise EvaluationFileError(f"'{path.name}' is not JSON: {e}") from e


def find_datasets(roots: Roots, name: str = "workspace") -> list[dict[str, Any]]:
    """JSON files under the workspace whose content is a list of objects
    with a ``question`` (a ``run-eval`` dataset), outside the output and
    cache directories."""
    root = roots.roots.get(name)
    if root is None or not root.path.is_dir():
        return []
    found = []
    for path in walk_files(root.path, limit=5000):
        if path.suffix.lower() != ".json":
            continue
        parts = path.relative_to(root.path).parts
        if (
            any(p in _SKIPPED for p in parts[:-1])
            or path.stat().st_size > MAX_DATASET_BYTES
        ):
            continue
        try:
            data = _read_json(path, MAX_DATASET_BYTES)
        except EvaluationFileError:
            continue
        if (
            isinstance(data, list)
            and data
            and all(isinstance(i, dict) and "question" in i for i in data[:20])
        ):
            ref = roots.ref(path)
            if ref:
                found.append({"ref": ref, "name": path.name, "items": len(data)})
    return found


def preview_dataset(path: Path) -> dict[str, Any]:
    """What a dataset holds: its items and which evaluators it feeds.

    Raises:
        EvaluationFileError: Not JSON, or not a list of objects with a
            ``question``.
    """
    data = _read_json(path, MAX_DATASET_BYTES)
    if not isinstance(data, list) or not all(
        isinstance(i, dict) and i.get("question") for i in data
    ):
        raise EvaluationFileError(
            f"'{path.name}' is not a run-eval dataset: a JSON list of objects, each with a question"
        )
    fields = {
        "answer": "answer_match",
        "reference_sources": "retrieval",
        "expected_entities": "graph_aware",
        "assertions": "assertion",
    }
    feeds = {
        evaluator: sum(1 for i in data if i.get(field))
        for field, evaluator in fields.items()
    }
    unanswerable = sum(
        1 for i in data if (i.get("metadata") or {}).get("answerable") is False
    )
    return {
        "items": len(data),
        "unanswerable": unanswerable,
        "feeds": feeds,
        "sample": [
            {
                "id": i.get("id") or i.get("query_id"),
                "question": str(i.get("question"))[:300],
                "answer": str(i.get("answer"))[:200] if i.get("answer") else None,
                "answerable": (i.get("metadata") or {}).get("answerable", True)
                is not False,
            }
            for i in data[:50]
        ],
    }


def find_runs(roots: Roots, name: str = "outputs") -> list[dict[str, Any]]:
    """Each directory under the evaluation outputs holding saved runs: its
    comparisons and summaries, newest first."""
    root = roots.roots.get(name)
    if root is None or not root.path.is_dir():
        return []
    groups: dict[Path, dict[str, list[Path]]] = defaultdict(
        lambda: {"summaries": [], "comparisons": []}
    )
    for path in walk_files(root.path, limit=20000):
        if path.suffix != ".json":
            continue
        if path.name.startswith("evaluation_summary_"):
            groups[path.parent]["summaries"].append(path)
        elif path.name.startswith("evaluation_comparison_"):
            groups[path.parent]["comparisons"].append(path)
    runs = []
    for directory, files in groups.items():
        ref = roots.ref(directory)
        if ref is None:
            continue
        newest = max(
            p.stat().st_mtime for p in files["summaries"] + files["comparisons"]
        )
        runs.append(
            {
                "ref": ref,
                "label": directory.name,
                "job_id": console_job_of(directory),
                "modified": newest,
                "summaries": len(files["summaries"]),
                "comparisons": [roots.ref(p) for p in sorted(files["comparisons"])],
                "strategies": sorted({_strategy_of(p) for p in files["summaries"]}),
            }
        )
    return sorted(runs, key=lambda r: r["modified"], reverse=True)


def console_job_of(directory: Path) -> str | None:
    """The console job that wrote a run directory: ``<outputs>/console/<job
    id>/`` itself, or the ``run_<stamp>/`` run-eval writes inside it."""
    for candidate in (directory, directory.parent):
        if candidate.parent.name == "console":
            return candidate.name
    return None


def _strategy_of(path: Path) -> str:
    """``local`` of ``evaluation_summary_local_20261107_092817_fa2911d9.json``."""
    stem = path.stem.removeprefix("evaluation_summary_")
    parts = stem.split("_")
    return "_".join(parts[:-3]) if len(parts) > 3 else stem


def read_run(directory: Path) -> dict[str, Any]:
    """A saved run directory: its newest comparison and every summary (the
    per-strategy fields the page shows).

    Raises:
        EvaluationFileError: A file that cannot be read.
    """
    comparisons = sorted(directory.glob("evaluation_comparison_*.json"))
    summaries = []
    for path in sorted(directory.glob("evaluation_summary_*.json")):
        data = _read_json(path, MAX_RESULT_BYTES)
        if not isinstance(data, dict):
            continue
        summaries.append(
            {
                "file": path.name,
                "strategy": _strategy_of(path),
                "total_queries": data.get("total_queries"),
                "successful_evaluations": data.get("successful_evaluations"),
                "failed_evaluations": data.get("failed_evaluations"),
                "average_response_time": data.get("average_response_time"),
                "metric_statistics": data.get("metric_statistics"),
                "metric_outcomes": data.get("metric_outcomes"),
                "abstention_statistics": data.get("abstention_statistics"),
                "answer_span_statistics": data.get("answer_span_statistics"),
                "usage": data.get("usage"),
                "rerank_status_counts": data.get("rerank_status_counts"),
                "guardrail_blocked_counts": data.get("guardrail_blocked_counts"),
                "started": data.get("evaluation_start_time"),
                "ended": data.get("evaluation_end_time"),
            }
        )
    comparison = _read_json(comparisons[-1], MAX_RESULT_BYTES) if comparisons else None
    if comparison is not None and isinstance(comparison, dict):
        # The manifest is long and the page does not show it whole.
        manifest = comparison.get("run_manifest") or {}
        comparison = {
            **comparison,
            "run_manifest": {
                k: manifest.get(k)
                for k in (
                    "created_at",
                    "package_version",
                    "git_sha",
                    "dataset",
                    "models",
                )
                if k in manifest
            },
        }
    return {
        "label": directory.name,
        "summaries": summaries,
        "comparison": comparison,
        "comparison_file": comparisons[-1].name if comparisons else None,
    }


def question_scores(
    directory: Path, comparison: dict[str, Any] | None
) -> dict[str, Any]:
    """Per question of a saved run: each strategy's score on one metric
    (its mean over the repeats that scored it; a failed answer has none and
    shows its error), its answer (the first repeat's) and how many of the
    repeats that answered it declined (``declined`` of ``repeats``), and
    whether the question is marked unanswerable (its answers are then read
    by whether they declined).

    The metric is the comparison's primary one, else
    ``answer_match/answer_contains`` when scored, else the first scored.
    Empty (``questions: []``) when the run saved no per-question results
    (``evaluation.save_detailed_results: false``).

    Raises:
        EvaluationFileError: A result file that cannot be read.
    """
    if not any(directory.glob("evaluation_results_*.json")):
        return {"metric": None, "strategies": [], "questions": []}
    try:
        grouped = group_runs(load_saved_runs([directory]))
    except EvaluationException as e:
        raise EvaluationFileError(str(e)) from e
    means = {
        strategy: per_query_means([run.run for run in saved])
        for strategy, saved in grouped.items()
    }
    scored = sorted({m for by_metric in means.values() for m in by_metric})
    primary = ((comparison or {}).get("multiplicity") or {}).get("primary_metric")
    metric = next(
        (m for m in (primary, DEFAULT_PRIMARY_METRIC) if m in scored),
        scored[0] if scored else None,
    )
    questions: dict[str, dict[str, Any]] = {}
    for strategy, saved in grouped.items():
        declines = _declines(saved)
        for result in saved[0].results:
            entry = questions.setdefault(
                result.query_id,
                {
                    "id": result.query_id,
                    "question": result.question,
                    "ground_truth": result.ground_truth,
                    # Marked metadata.answerable=false: a decline is right.
                    "unanswerable": result.metadata.get("answerable") is False,
                    "scores": {},
                    "answers": {},
                },
            )
            declined, answered = declines.get(result.query_id, (0, 0))
            entry["answers"][strategy] = {
                "text": _text(result.generated_answer),
                "error": result.error_message if result.error else None,
                # Over every repeat that answered it, as the scores are.
                "declined": declined,
                "repeats": answered,
            }
        for qid, value in (means[strategy].get(metric or "") or {}).items():
            if qid in questions:
                questions[qid]["scores"][strategy] = value
    return {
        "metric": metric,
        "strategies": list(grouped),
        "questions": list(questions.values()),
    }


def _declines(saved: list[SavedRun]) -> dict[str, tuple[int, int]]:
    """Per question id: how many of the repeats ``saved`` that answered it
    (without an error) declined, and how many answered it."""
    counts: dict[str, tuple[int, int]] = {}
    for run in saved:
        for result in run.results:
            if result.error:
                continue
            declined, answered = counts.get(result.query_id, (0, 0))
            counts[result.query_id] = (
                declined + int(bool(result.abstained)),
                answered + 1,
            )
    return counts


def _text(value: str | None) -> str:
    text = value or ""
    return text if len(text) <= MAX_ANSWER_CHARS else text[: MAX_ANSWER_CHARS - 1] + "…"
