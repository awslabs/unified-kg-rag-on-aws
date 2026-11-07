# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Evaluate page's files: ``run-eval`` datasets to pick, and the saved
runs (summaries and comparisons) ``run-eval`` wrote, read back as written."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from unified_kg_rag.application.cli.console.paths import Roots, walk_files
from unified_kg_rag.shared.exceptions import DataProcessingError

MAX_DATASET_BYTES = 50 * 1024 * 1024
MAX_RESULT_BYTES = 200 * 1024 * 1024
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
                "modified": newest,
                "summaries": len(files["summaries"]),
                "comparisons": [roots.ref(p) for p in sorted(files["comparisons"])],
                "strategies": sorted({_strategy_of(p) for p in files["summaries"]}),
            }
        )
    return sorted(runs, key=lambda r: r["modified"], reverse=True)


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
