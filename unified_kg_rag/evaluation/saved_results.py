# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read the result files ``run-eval`` saved, to rebuild a comparison offline.

``run-eval --from-results`` and ``--baseline-results`` point here: each run of
a strategy saved ``evaluation_results_<strategy>_<stamp>.json`` with its
``evaluation_reports_*`` (``evaluation.save_detailed_results``) and
``evaluation_summary_*`` (the run manifest) next to it. Grouped by strategy
and ordered by start time, the runs are what ``paired_comparison`` takes, so
a comparison rebuilt from the files of an uninterrupted run, or of an
interrupted one completed by a second run of the missing strategies, equals
the one the run printed. Two runs whose exact match and token F1 read the
answer differently (the manifest's ``answer_scoring``) are not comparable on
those metrics, and are refused.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from unified_kg_rag.domain.models import (
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
)
from unified_kg_rag.shared import EvaluationException, get_logger

from .comparison import Run

logger = get_logger(__name__)

# EvaluationManager.save_results: evaluation_<kind>_[<strategy>_]<stamp>.json.
_RESULTS_FILE = re.compile(
    r"^evaluation_results_(?:(?P<strategy>.+)_)?"
    r"(?P<stamp>\d{8}_\d{6}_[0-9a-f]{8})\.json$"
)


class SavedRun(BaseModel):
    """One saved run of one strategy: its results, reports and summary."""

    model_config = ConfigDict(frozen=True)

    path: Path
    strategy: str
    stamp: str
    results: list[EvaluationResult]
    reports: list[EvaluationReport]
    summary: EvaluationSummary | None = None

    @property
    def manifest(self) -> dict[str, Any] | None:
        return self.summary.run_manifest if self.summary else None

    @property
    def run(self) -> Run:
        return self.reports, self.results

    def _sort_key(self) -> tuple[datetime | None, str]:
        start = self.summary.evaluation_start_time if self.summary else None
        return (start or datetime.min, self.stamp)


def result_files(paths: Iterable[Path]) -> list[Path]:
    """The ``evaluation_results_*.json`` files of ``paths`` (directories, or
    result files themselves), each once.

    Raises:
        EvaluationException: A path is missing or not a result file, or
            none holds a result file.
    """
    files: dict[Path, None] = {}
    for path in paths:
        if path.is_dir():
            found = sorted(p for p in path.iterdir() if _RESULTS_FILE.match(p.name))
            if not found:
                raise EvaluationException(
                    f"'{path}' holds no evaluation_results_*.json file (they "
                    "are saved with evaluation.save_detailed_results: true)"
                )
            files.update(dict.fromkeys(p.resolve() for p in found))
        elif path.is_file() and _RESULTS_FILE.match(path.name):
            files[path.resolve()] = None
        elif path.exists():
            raise EvaluationException(
                f"'{path}' is not an evaluation_results_*.json file or a "
                "directory of them"
            )
        else:
            raise EvaluationException(f"'{path}' not found")
    return list(files)


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise EvaluationException(f"cannot read '{path}': {e}") from e


def load_saved_run(path: Path) -> SavedRun:
    """The run whose results ``path`` holds, with its reports and summary.

    Raises:
        EvaluationException: Its reports file is missing, or a file does not
            parse.
    """
    match = _RESULTS_FILE.match(path.name)
    if match is None:
        raise EvaluationException(f"'{path}' is not an evaluation_results file")
    reports_path = path.with_name(path.name.replace("_results_", "_reports_", 1))
    summary_path = path.with_name(path.name.replace("_results_", "_summary_", 1))
    if not reports_path.is_file():
        raise EvaluationException(
            f"'{path}' has no reports file '{reports_path.name}' next to it"
        )
    try:
        results = [EvaluationResult.model_validate(r) for r in _read(path)]
        reports = [EvaluationReport.model_validate(r) for r in _read(reports_path)]
        summary = (
            EvaluationSummary.model_validate(_read(summary_path))
            if summary_path.is_file()
            else None
        )
    except ValidationError as e:
        raise EvaluationException(f"'{path}': {e}") from e
    strategy = match.group("strategy") or _results_strategy(results)
    return SavedRun(
        path=path,
        strategy=strategy,
        stamp=match.group("stamp"),
        results=results,
        reports=reports,
        summary=summary,
    )


def _results_strategy(results: Sequence[EvaluationResult]) -> str:
    """The strategy the queries asked for, as save_results names its files."""
    strategies = {
        str(r.metadata.get("search_strategy") or r.search_strategy)
        for r in results
        if r.metadata.get("search_strategy") or r.search_strategy
    }
    return strategies.pop() if len(strategies) == 1 else "unknown"


def load_saved_runs(paths: Iterable[Path]) -> list[SavedRun]:
    """Every saved run under ``paths``, oldest first."""
    runs = [load_saved_run(path) for path in result_files(paths)]
    return sorted(runs, key=SavedRun._sort_key)


def strategy_order(saved: Sequence[SavedRun]) -> list[str]:
    """The strategies of ``saved`` in the order their oldest run listed them
    (its ``--search-strategy``: the first is the baseline), then any other
    in the order their first run started."""
    present = list(dict.fromkeys(run.strategy for run in saved))
    listed: list[str] = []
    for run in saved:
        args = (run.manifest or {}).get("cli_args") or {}
        names = args.get("search_strategy")
        if isinstance(names, list) and len(names) > 1:
            listed = [str(n) for n in names]
            break
    ordered = [s for s in listed if s in present]
    return [*ordered, *(s for s in present if s not in ordered)]


def group_runs(
    saved: Sequence[SavedRun], order: Sequence[str] | None = None
) -> dict[str, list[SavedRun]]:
    """``saved`` by strategy: those of ``order`` (else ``strategy_order``),
    each run list oldest first.

    Raises:
        EvaluationException: A strategy of ``order`` has no saved run.
    """
    by_strategy: dict[str, list[SavedRun]] = {}
    for run in saved:
        by_strategy.setdefault(run.strategy, []).append(run)
    names = list(order) if order else strategy_order(saved)
    missing = [s for s in names if s not in by_strategy]
    if missing:
        raise EvaluationException(
            f"no saved results for strategy {', '.join(missing)} (found: "
            f"{', '.join(by_strategy) or 'none'})"
        )
    dropped = [s for s in by_strategy if s not in names]
    if dropped:
        logger.info("Not comparing the saved runs of %s", ", ".join(dropped))
    return {s: by_strategy[s] for s in names}


_UNKNOWN = object()


def _answer_scoring(manifest: dict[str, Any] | None) -> Any:
    if manifest is None or "answer_scoring" not in manifest:
        return _UNKNOWN
    return manifest["answer_scoring"]


def check_answer_scoring(
    manifests: Sequence[tuple[str, dict[str, Any] | None]],
) -> list[str]:
    """Refuse runs whose exact match and token F1 read the answer
    differently (``answer_scoring``: ``evaluation.answer_span`` and
    ``answer_normalization``); returns warnings for runs that do not record
    it. ``manifests`` are ``(label, run manifest)`` pairs.

    Raises:
        EvaluationException: Two runs scored with answer_match record
            different settings.
    """
    scored: dict[str, list[str]] = {}
    unknown: list[str] = []
    for label, manifest in manifests:
        scoring = _answer_scoring(manifest)
        if scoring is _UNKNOWN:
            unknown.append(label)
        elif scoring is not None:  # None: answer_match was not enabled
            key = json.dumps(scoring, sort_keys=True)
            scored.setdefault(key, []).append(label)
    if len(scored) > 1:
        listed = "; ".join(
            f"{key} ({', '.join(dict.fromkeys(labels))})"
            for key, labels in scored.items()
        )
        raise EvaluationException(
            "the runs scored exact_match and token_f1 with different "
            f"answer_scoring settings ({listed}); they are not comparable. "
            "Re-run one side with the other's evaluation.answer_span and "
            "answer_normalization."
        )
    if unknown and scored:
        return [
            f"The runs of {', '.join(dict.fromkeys(unknown))} record no "
            "answer_scoring (no summary file, or saved before it was "
            "recorded): check they scored exact_match and token_f1 like the "
            "others."
        ]
    return []


def dataset_warnings(
    manifests: Sequence[tuple[str, dict[str, Any] | None]],
) -> list[str]:
    """A warning when the runs answered different datasets (manifest
    ``dataset.content_sha256``): only their shared query ids pair."""
    hashes: dict[str, list[str]] = {}
    for label, manifest in manifests:
        digest = ((manifest or {}).get("dataset") or {}).get("content_sha256")
        if digest:
            hashes.setdefault(str(digest), []).append(label)
    if len(hashes) < 2:
        return []
    listed = "; ".join(
        f"{digest[:12]} ({', '.join(dict.fromkeys(labels))})"
        for digest, labels in hashes.items()
    )
    return [
        f"The runs answered different datasets ({listed}); only the query ids "
        "they share are paired."
    ]
