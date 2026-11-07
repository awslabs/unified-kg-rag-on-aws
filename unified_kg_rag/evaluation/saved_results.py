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
from collections.abc import Iterable, Mapping, Sequence
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

from .answer_match_evaluator import SPAN_EFFORT_KEY, SPAN_MODEL_KEY, TOKENS_KEY
from .comparison import USAGE_SCHEMA, Run

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

    def _sort_key(self) -> tuple[datetime, int, str]:
        # The interleaved runs of one invocation share their start time:
        # the repeat the manifest records orders them (0: saved before it
        # was recorded), then the file's stamp.
        start = self.summary.evaluation_start_time if self.summary else None
        repeat = (self.manifest or {}).get("repeat")
        return (
            start or datetime.min,
            repeat if isinstance(repeat, int) else 0,
            self.stamp,
        )


def result_files(paths: Iterable[Path]) -> list[Path]:
    """The ``evaluation_results_*.json`` files of ``paths`` (directories, or
    result files themselves), each once. A directory's are its own and its
    subdirectories' (one level: the run subdirectories ``run-eval`` writes,
    ``evaluation.run_subdirectories``).

    Raises:
        EvaluationException: A path is missing or not a result file, or
            none holds a result file.
    """
    files: dict[Path, None] = {}
    for path in paths:
        if path.is_dir():
            found = sorted(
                p
                for directory in (
                    path,
                    *sorted(d for d in path.iterdir() if d.is_dir()),
                )
                for p in directory.iterdir()
                if _RESULTS_FILE.match(p.name)
            )
            if not found:
                raise EvaluationException(
                    f"'{path}' holds no evaluation_results_*.json file, nor "
                    "do its subdirectories (they are saved with "
                    "evaluation.save_detailed_results: true)"
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
        manifest = run.manifest or {}
        # A run with --arm records every run label in comparison order.
        compared = manifest.get("compared_runs")
        if isinstance(compared, list) and len(compared) > 1:
            listed = [str(n) for n in compared]
            break
        names = (manifest.get("cli_args") or {}).get("search_strategy")
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
    scoring = manifest["answer_scoring"]
    if isinstance(scoring, dict) and TOKENS_KEY not in scoring:
        # Saved before the tokenization was recorded: token F1 read
        # whitespace words then, under every normalization.
        scoring = {**scoring, TOKENS_KEY: "whitespace"}
    return scoring


def dataset_dense_script(
    manifests: Sequence[tuple[str, dict[str, Any] | None]],
) -> bool | None:
    """Whether the runs' dataset holds Han, Hangul or Kana text, as their
    manifests record it (``dataset.dense_script``): ``True`` if one does,
    ``False`` if every one that records it says not, ``None`` when none
    records it (saved before it was recorded)."""
    recorded = [
        flag
        for _, manifest in manifests
        if isinstance(
            flag := ((manifest or {}).get("dataset") or {}).get("dense_script"), bool
        )
    ]
    if not recorded:
        return None
    return any(recorded)


def check_answer_scoring(
    manifests: Sequence[tuple[str, dict[str, Any] | None]],
    *,
    dense_script: bool | None = None,
) -> list[str]:
    """Refuse runs whose exact match and token F1 read the answer
    differently (``answer_scoring``: ``evaluation.answer_span``,
    ``answer_normalization`` and the ``token_f1_tokens`` it implies; a run
    saved before that was recorded tokenized by whitespace); returns
    warnings for runs that do not record it. ``manifests`` are ``(label,
    run manifest)`` pairs.

    The tokenizations differ only on Han, Hangul and Kana text, so
    ``token_f1_tokens`` is left out of the comparison when the dataset
    holds none: ``dense_script`` (the caller's dataset), else what the
    manifests record (``dataset_dense_script``); unknown, it is compared.

    Raises:
        EvaluationException: Two runs scored with answer_match record
            different settings.
    """
    if dense_script is None:
        dense_script = dataset_dense_script(manifests)
    scorings = [(label, _answer_scoring(manifest)) for label, manifest in manifests]
    # An extracted span's model and effort, recorded since: compared only
    # when every extracted run records them.
    extracted = {
        label: SPAN_MODEL_KEY in sc
        for label, sc in scorings
        if isinstance(sc, dict) and sc.get("answer_span") == "extracted"
    }
    span_unrecorded = [label for label, known in extracted.items() if not known]
    drop = {TOKENS_KEY} if dense_script is False else set()
    if span_unrecorded and len(span_unrecorded) < len(extracted):
        drop |= {SPAN_MODEL_KEY, SPAN_EFFORT_KEY}
    scored: dict[str, list[str]] = {}
    unknown: list[str] = []
    for label, scoring in scorings:
        if scoring is _UNKNOWN:
            unknown.append(label)
        elif scoring is not None:  # None: answer_match was not enabled
            if isinstance(scoring, dict):
                scoring = {k: v for k, v in scoring.items() if k not in drop}
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
            "answer_normalization (a run that records token_f1_tokens "
            "'whitespace' under answer_normalization 'default' was scored "
            "before token F1 tokenized Chinese, Japanese and Korean by "
            "character, and one that records 'script_aware' before it "
            "dropped only Korean particles of their closed class: re-run "
            "it)."
        )
    warnings = []
    if SPAN_MODEL_KEY in drop:
        warnings.append(
            f"The runs of {', '.join(span_unrecorded)} record no answer-span "
            "model (saved before it was recorded): check their extracted "
            "spans came from the same models.roles.answer_span and effort as "
            "the others'."
        )
    if unknown and scored:
        warnings.append(
            f"The runs of {', '.join(dict.fromkeys(unknown))} record no "
            "answer_scoring (no summary file, or saved before it was "
            "recorded): check they scored exact_match and token_f1 like the "
            "others."
        )
    return warnings


def citation_warnings(
    manifests: Sequence[tuple[str, dict[str, Any] | None]],
) -> list[str]:
    """Warnings when the compared runs differ in ``search.citations.enabled``
    (the answer prompt asks for markers on one side only: the answers differ
    by the prompt as well as the strategy) or scored the citation metrics
    another way (``citations.scoring``; a run saved before it was recorded
    credited a graph citation whole)."""
    enabled: dict[str, list[str]] = {}
    # How the runs that may have scored citations did (one saved before
    # the block was recorded may have).
    scoring: dict[str, list[str]] = {}
    for label, manifest in manifests:
        if manifest is None:
            continue
        citations = manifest.get("citations")
        if not isinstance(citations, dict):
            scoring.setdefault("unrecorded", []).append(label)
            continue
        enabled.setdefault(str(citations.get("enabled")), []).append(label)
        if citations.get("enabled"):
            scoring.setdefault(str(citations.get("scoring")), []).append(label)
    warnings = []
    if len(enabled) > 1:
        listed = "; ".join(
            f"{'on' if value == 'True' else 'off'}: "
            f"{', '.join(dict.fromkeys(labels))}"
            for value, labels in enabled.items()
        )
        warnings.append(
            f"The compared runs differ in search.citations.enabled ({listed}): "
            "the answer prompt asks for citation markers on one side only, so "
            "the deltas measure that prompt as well as the strategies "
            "(answer_match, graph_aware and the pairwise judge read the "
            "answers without markers; the citation metrics exist on one side "
            "only)."
        )
    if len(scoring) > 1 and enabled.get("True"):
        listed = "; ".join(
            f"{value}: {', '.join(dict.fromkeys(labels))}"
            for value, labels in scoring.items()
        )
        warnings.append(
            f"The compared runs scored citation_precision and citation_recall "
            f"differently ({listed}; runs saved before 'file_weighted' "
            "credited a graph citation whole when any of its files matched): "
            "re-score them with run-eval --from-results --rescore."
        )
    return warnings


# The scoring models a run manifest records (``models``), as warnings name
# them.
_JUDGE_MODEL_KEYS = {
    "evaluation_judge": "LLM judge (models.roles.evaluation)",
    "judge_effort": "judge effort (evaluation.judge_effort)",
    "evaluation_embedding": "ragas embedding model (models.embedding)",
}


def judge_warnings(
    manifests: Sequence[tuple[str, dict[str, Any] | None]],
) -> list[str]:
    """Warnings when the compared runs were judged by different models: a
    different LLM judge, judge effort or ragas embedding model (manifest
    ``models``) scores the same answer differently, so the deltas of the
    judged metrics measure the judges as well as the runs. A run that
    records none (no LLM judge, or the embedding without ragas) is not
    compared."""
    warnings = []
    for key, what in _JUDGE_MODEL_KEYS.items():
        values: dict[str, list[str]] = {}
        for label, manifest in manifests:
            value = ((manifest or {}).get("models") or {}).get(key)
            if value is not None:
                values.setdefault(str(value), []).append(label)
        if len(values) > 1:
            listed = "; ".join(
                f"{value}: {', '.join(dict.fromkeys(labels))}"
                for value, labels in values.items()
            )
            warnings.append(
                f"The compared runs were scored with different {what} "
                f"({listed}): the deltas of the judged metrics measure the "
                "judges as well as the runs. Re-score them with one "
                "(run-eval --from-results --rescore, --judge-model)."
            )
    return warnings


def rerank_unmetered(
    manifests: Sequence[tuple[str, dict[str, Any] | None]],
) -> set[str]:
    """The labels of the runs whose usage records do not count rerank search
    units as billed: their manifest records a ``usage_schema`` below
    ``USAGE_SCHEMA`` (none before 2, too few for long documents before 3), or
    none (saved before it was recorded). A run without a manifest is not
    known, and not named."""
    return {
        label
        for label, manifest in manifests
        if manifest is not None
        and not (
            isinstance(schema := manifest.get("usage_schema"), int)
            and schema >= USAGE_SCHEMA
        )
    }


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


def interleaving_warnings(
    manifests: Sequence[tuple[str, dict[str, Any] | None]],
) -> list[str]:
    """A warning when the compared runs were not answered interleaved in one
    ``run-eval`` (one ``run_id`` whose manifests record ``run_order``
    ``interleaved``): runs answered one after another, in separate
    invocations (``--from-results`` of several, ``--baseline-results``) or
    before the order was recorded, differ in the time they ran at, so their
    latency deltas measure that time as well as the strategies. ``manifests``
    are ``(label, run manifest)`` pairs; one label compares nothing."""
    if len({label for label, _ in manifests}) < 2:
        return []
    invocations = {(manifest or {}).get("run_id") for _, manifest in manifests}
    orders = {(manifest or {}).get("run_order") for _, manifest in manifests}
    known = invocations - {None}
    if len(invocations) == 1 and known and orders == {"interleaved"}:
        return []
    if len(known) > 1:
        how = f"in {len(known)} separate run-eval invocations"
    elif None in invocations:
        how = "one after another (saved before run-eval interleaved them)"
    else:
        how = "one after another (--run-order sequential)"
    return [
        f"The compared runs were answered {how}, not interleaved in one run: "
        "their latency deltas (latency/response_time_seconds) also measure "
        "what changed between them (service load, warm connections and "
        "caches), and so may their usage deltas: two runs of one "
        "configuration answered one after the other differed by 0.5-5.8 s "
        "per query. Compare latency on runs answered together (one run-eval "
        "listing every strategy, --run-order interleaved)."
    ]


# The run settings the saved runs of one strategy must share to be pooled as
# its repeats: the query flags of run-eval, then manifest entries. The CLI
# query flags are not in retrieval_config_sha256 (a hash of the
# configuration), so they are compared here.
_POOLED_CLI_ARGS = (
    "suffix",
    "top_k",
    "search_type",
    "retrieval_multiplier",
)


def _pooled_settings(manifest: dict[str, Any]) -> dict[str, Any]:
    args = manifest.get("cli_args") or {}
    return {
        **{name: args.get(name) for name in _POOLED_CLI_ARGS},
        "retrieval_config_sha256": manifest.get("retrieval_config_sha256"),
        "git_sha": manifest.get("git_sha"),
        "index": manifest.get("index"),
        "answer model": (manifest.get("models") or {}).get("answer_generation"),
    }


def _shown(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= 24 else f"{text[:21]}..."


def _invocations(runs: Sequence[SavedRun]) -> int | None:
    """How many ``run-eval`` invocations answered ``runs``, when that is
    known to be more than one: distinct ``run_id``s, a run with one next to
    one without, or (runs saved before ``run_id`` was recorded) more runs
    than any of their invocations' ``--repeats``. ``None`` otherwise."""
    ids = {(run.manifest or {}).get("run_id") for run in runs}
    known = ids - {None}
    if len(known) > 1:
        return len(known)
    if known and None in ids:
        return 2
    if not known:
        repeats = [
            int(value)
            for run in runs
            if isinstance(
                value := ((run.manifest or {}).get("cli_args") or {}).get("repeats"),
                int,
            )
        ]
        if repeats and len(runs) > max(repeats):
            return 2
    return None


def pooling_errors(grouped: Mapping[str, Sequence[SavedRun]]) -> list[str]:
    """Why the saved runs of a strategy are not one run's repeats: they come
    from several ``run-eval`` invocations (``_invocations``), or differ in a
    setting that changes the answers (``pooling_warnings``). Pooled anyway
    (``run-eval --pool-runs``), they mix the invocations' time and
    settings into one strategy's means."""
    errors = []
    for strategy, runs in grouped.items():
        if (count := _invocations(runs)) is not None:
            errors.append(
                f"the {len(runs)} saved runs of {strategy} come from {count} or "
                "more run-eval invocations"
            )
    return [*errors, *pooling_warnings(grouped)]


def pooling_warnings(grouped: Mapping[str, Sequence[SavedRun]]) -> list[str]:
    """A warning per strategy whose saved runs, pooled as its repeats,
    differ in a setting that changes its answers (a query flag, the
    configuration hash, the commit, the index, the answer model). Runs
    without a summary are not compared (``summary_warnings``)."""
    warnings = []
    for strategy, runs in grouped.items():
        settings = [_pooled_settings(r.manifest) for r in runs if r.manifest]
        if len(settings) < 2:
            continue
        differing = [
            f"{name} ({', '.join(dict.fromkeys(_shown(s[name]) for s in settings))})"
            for name in settings[0]
            if len({json.dumps(s[name], sort_keys=True, default=str) for s in settings})
            > 1
        ]
        if differing:
            warnings.append(
                f"The {len(runs)} saved runs of {strategy} are pooled as its "
                f"repeats but differ in {'; '.join(differing)}: its means mix "
                "those settings. Keep runs of other settings in another "
                "directory, or list the result files to compare."
            )
    return warnings


def summary_warnings(saved: Sequence[SavedRun], label: str) -> list[str]:
    """A warning naming the saved runs without an ``evaluation_summary_*``
    file: their manifest is unknown, so their answer scoring, dataset and
    settings, and the judge's self-preference, are not checked."""
    missing = [run.path.name for run in saved if run.summary is None]
    if not missing:
        return []
    return [
        f"{label}: {len(missing)} saved run(s) have no evaluation_summary file "
        f"next to them ({', '.join(missing[:3])}{', ...' if len(missing) > 3 else ''}): "
        "their answer scoring, dataset, run settings and the pairwise judge's "
        "self-preference are not checked."
    ]
