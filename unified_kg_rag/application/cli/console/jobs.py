# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ingestion and evaluation jobs: the unchanged CLIs in child processes.

A job is ``run-ingestion`` or ``run-eval --output-format json`` started with
the console's directories. Stop sends SIGTERM, the stop the CLIs are built
for (the pipeline checkpoints its finished stages and exits 143; ``run-eval``
keeps the runs it finished); one still running after the CLIs' stop grace is
killed. Progress is read from the files the run writes (``stages``,
:func:`evaluation_progress`), the result from its JSON document on stdout.
One job per index namespace runs at a time (docs/design.md#101-the-console-run-console-opt-in).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from unified_kg_rag.application.cli.console import stages
from unified_kg_rag.application.cli.console.paths import PathRefused, walk_files
from unified_kg_rag.application.cli.console.store import ConsoleStore
from unified_kg_rag.domain.models import (
    Config,
    Constants,
    Methodology,
    SearchStrategy,
    validate_path_segment,
)
from unified_kg_rag.evaluation.gates import parse_gate
from unified_kg_rag.shared import ConfigurationError, get_logger

logger = get_logger(__name__)

# The CLIs end a stopped run within their stop grace (conventions:
# _STOP_GRACE_SECONDS, 100 s); a child still alive after this is killed.
STOP_GRACE_SECONDS = 110.0
# The log lines a job reports back.
LOG_TAIL_LINES = 80
MAX_REPEATS = 20
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_SLUG = re.compile(r"[^A-Za-z0-9._-]+")


class JobKind(str, Enum):
    INGEST = "ingest"
    EVALUATE = "evaluate"


class JobStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


_ACTIVE = (JobStatus.RUNNING, JobStatus.STOPPING)


class JobConflict(ConfigurationError):
    """Another job runs in the namespace (the request is refused)."""

    def __init__(self, message: str, job_id: str) -> None:
        super().__init__(message)
        self.job_id = job_id


@dataclass
class Job:
    """One job; ``record()`` is what the API and the store see."""

    id: str
    kind: JobKind
    namespace: str
    params: dict[str, Any]
    argv: list[str]
    env: dict[str, str]
    work_dir: Path
    log_path: Path
    created: float = field(default_factory=time.time)
    status: JobStatus = JobStatus.RUNNING
    started: float | None = None
    ended: float | None = None
    exit_code: int | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    pipeline_id: str | None = None
    output_directory: Path | None = None
    plan: list[str] = field(default_factory=list)
    total_runs: int = 0
    process: asyncio.subprocess.Process | None = None
    watcher: asyncio.Task[None] | None = None

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind.value,
            "namespace": self.namespace,
            "params": self.params,
            "command": [Path(self.argv[0]).name, *self.argv[1:]],
            "env": self.env,
            "created": self.created,
            "started": self.started,
            "ended": self.ended,
            "status": self.status.value,
            "exit_code": self.exit_code,
            "error": self.error,
            "result": self.result,
            "pipeline_id": self.pipeline_id,
            "output_directory": (
                str(self.output_directory) if self.output_directory else None
            ),
            "plan": self.plan,
            "total_runs": self.total_runs,
        }


def _cli(script: str, entry: str) -> list[str]:
    """The command of a console script of this environment: the script next
    to the interpreter, else its launcher entry through the interpreter."""
    beside = Path(sys.executable).with_name(script)
    if beside.is_file():
        return [str(beside)]
    return [
        sys.executable,
        "-c",
        f"import sys; from unified_kg_rag.application.cli.launch import {entry}; "
        f"sys.exit({entry}())",
    ]


def namespace_of(config: Config, index_value: str | None = None) -> str:
    return (
        index_value
        or config.processing.document_parsing.index_value
        or Constants.DEFAULT_SUFFIX.value
    )


def _pipeline_id(corpus: str, namespace: str) -> str:
    """A stable pipeline id per corpus and namespace, so a rerun of the same
    corpus resumes or runs incrementally on the same cache."""
    slug = _SLUG.sub("-", corpus).strip("-._")[:80] or "corpus"
    if not slug[0].isalnum():
        slug = f"c{slug}"
    suffix = "" if namespace == Constants.DEFAULT_SUFFIX.value else f"-{namespace}"
    return validate_path_segment(f"console-{slug}{suffix}"[:128], "pipeline_id")


class JobManager:
    """Starts, watches and stops the console's jobs."""

    def __init__(
        self,
        *,
        store: ConsoleStore,
        jobs_directory: Path,
        cache_directory: Path,
        evaluation_outputs: Path,
        config_path: Path | None,
        load_config: Callable[[], Config],
        on_ingest_end: Callable[[], None] | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.store = store
        self.jobs_directory = jobs_directory
        self.cache_directory = cache_directory
        self.evaluation_outputs = evaluation_outputs
        self.config_path = config_path
        self.load_config = load_config
        self.on_ingest_end = on_ingest_end
        self.cwd = cwd or Path.cwd()
        self.jobs: dict[str, Job] = {}

    # -- starting ---------------------------------------------------------

    def _check_free(self, namespace: str) -> None:
        for job in self.jobs.values():
            if job.namespace == namespace and job.status in _ACTIVE:
                raise JobConflict(
                    f"job {job.id} ({job.kind.value}) is running in namespace "
                    f"'{namespace}'; stop it or wait for it to finish",
                    job.id,
                )

    def _new_job(
        self,
        kind: JobKind,
        namespace: str,
        params: dict[str, Any],
        argv: list[str],
        env: dict[str, str],
    ) -> Job:
        job_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        work_dir = self.jobs_directory / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        return Job(
            id=job_id,
            kind=kind,
            namespace=namespace,
            params=params,
            argv=argv,
            env=env,
            work_dir=work_dir,
            log_path=work_dir / "job.log",
        )

    def _config_args(self) -> list[str]:
        return ["--config-path", str(self.config_path)] if self.config_path else []

    async def start_ingest(
        self,
        *,
        corpus: Path,
        corpus_ref: str,
        methodologies: list[str] | None = None,
        index_value: str | None = None,
        force_rebuild: bool = False,
        confirm_reset: bool = False,
        confirm_deletions: bool = False,
    ) -> Job:
        """Start ``run-ingestion`` over ``corpus`` (a resolved directory).

        Raises:
            JobConflict: A job runs in the namespace.
            ConfigurationError: An unknown methodology or a bad index value.
        """
        config = self.load_config()
        env: dict[str, str] = {}
        if methodologies:
            try:
                chosen = [Methodology(m).value for m in methodologies]
            except ValueError as e:
                raise ConfigurationError(f"methodologies: {e}") from e
            env["UKR__INDEXING__METHODOLOGIES"] = json.dumps(chosen)
            config = config.model_copy(deep=True)
            config.indexing.methodologies = [Methodology(m) for m in chosen]
        if index_value:
            try:
                index_value = validate_path_segment(index_value, "index value")
            except ValueError as e:
                raise ConfigurationError(str(e)) from e
            env["UKR__PROCESSING__DOCUMENT_PARSING__INDEX_VALUE"] = index_value
        namespace = namespace_of(config, index_value)
        self._check_free(namespace)
        if not await asyncio.to_thread(corpus.is_dir):
            raise PathRefused(f"'{corpus_ref}' is not a directory")
        pipeline_id = _pipeline_id(corpus_ref.partition(":")[2] or "corpus", namespace)
        argv = [
            *_cli("run-ingestion", "run_ingestion"),
            *self._config_args(),
            "--source-directory",
            str(corpus),
            "--cache-directory",
            str(self.cache_directory),
            "--pipeline-id",
            pipeline_id,
            "--output-format",
            "json",
        ]
        if force_rebuild:
            argv.append("--force-rebuild")
        if confirm_reset:
            argv.append("--confirm-reset")
        if confirm_deletions:
            argv.append("--confirm-deletions")
        job = self._new_job(
            JobKind.INGEST,
            namespace,
            {
                "corpus": corpus_ref,
                "methodologies": methodologies
                or [m.value for m in config.effective_methodologies],
                "index_value": index_value,
                "force_rebuild": force_rebuild,
                "confirm_reset": confirm_reset,
                "confirm_deletions": confirm_deletions,
            },
            argv,
            env,
        )
        job.pipeline_id = pipeline_id
        job.plan = stages.planned_stages(config)
        await self._launch(job)
        return job

    async def start_evaluate(
        self,
        *,
        dataset: Path,
        dataset_ref: str,
        strategies: list[str],
        repeats: int = 1,
        closed_book: bool = False,
        pairwise_judge: bool = False,
        fail_if: list[str] | None = None,
        max_failure_rate: float | None = None,
    ) -> Job:
        """Start ``run-eval`` over ``dataset`` (a resolved file).

        Raises:
            JobConflict: A job runs in the namespace.
            ConfigurationError: An unknown strategy, a bad gate or repeat count.
        """
        config = self.load_config()
        if not strategies:
            raise ConfigurationError("choose at least one strategy")
        try:
            chosen = [SearchStrategy(s).value for s in strategies]
        except ValueError as e:
            raise ConfigurationError(f"strategies: {e}") from e
        if len(set(chosen)) != len(chosen):
            raise ConfigurationError("a strategy is listed twice")
        if not 1 <= repeats <= MAX_REPEATS:
            raise ConfigurationError(f"repeats must be between 1 and {MAX_REPEATS}")
        gates = [g.strip() for g in fail_if or [] if g.strip()]
        for gate in gates:
            try:
                parse_gate(gate)
            except ValueError as e:
                raise ConfigurationError(f"gate '{gate}': {e}") from e
        if gates and len(chosen) < 2:
            raise ConfigurationError("a gate needs two or more strategies to compare")
        if pairwise_judge and len(chosen) + int(closed_book) < 2:
            raise ConfigurationError("the pairwise judge needs two or more strategies")
        if max_failure_rate is not None and not 0.0 <= max_failure_rate <= 1.0:
            raise ConfigurationError("the failure budget must be between 0 and 1")
        if not await asyncio.to_thread(dataset.is_file):
            raise PathRefused(f"'{dataset_ref}' is not a file")
        namespace = namespace_of(config)
        self._check_free(namespace)
        job = self._new_job(JobKind.EVALUATE, namespace, {}, [], {})
        output = self.evaluation_outputs / "console" / job.id
        output.mkdir(parents=True, exist_ok=True)
        argv = [
            *_cli("run-eval", "run_eval"),
            *self._config_args(),
            "--data-path",
            str(dataset),
            "--search-strategy",
            ",".join(chosen),
            "--repeats",
            str(repeats),
            "--output-directory",
            str(output),
            "--output-format",
            "json",
        ]
        if closed_book:
            argv.append("--closed-book")
        if pairwise_judge:
            argv.append("--pairwise-judge")
        for gate in gates:
            argv.extend(["--fail-if", gate])
        if max_failure_rate is not None:
            argv.extend(["--max-failure-rate", str(max_failure_rate)])
        job.argv = argv
        job.params = {
            "dataset": dataset_ref,
            "strategies": chosen,
            "repeats": repeats,
            "closed_book": closed_book,
            "pairwise_judge": pairwise_judge,
            "fail_if": gates,
            "max_failure_rate": max_failure_rate,
        }
        job.output_directory = output
        job.total_runs = (len(chosen) + int(closed_book)) * repeats
        await self._launch(job)
        return job

    async def _launch(self, job: Job) -> None:
        env = {
            **os.environ,
            **job.env,
            # Whole lines in the log, no colour codes, no progress bars.
            "PYTHONUNBUFFERED": "1",
            "NO_COLOR": "1",
            "COLUMNS": "120",
        }
        with job.log_path.open("ab") as log:
            job.process = await asyncio.create_subprocess_exec(
                *job.argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
                env=env,
                cwd=str(self.cwd),
                # Its own session: a Ctrl-C at the console's terminal does not
                # reach it; the console stops it (shutdown).
                start_new_session=True,
            )
        job.started = time.time()
        self.jobs[job.id] = job
        self.store.save_job(job.record())
        job.watcher = asyncio.create_task(self._watch(job), name=f"job-{job.id}")
        logger.info(
            "Started %s job %s (pid %s)", job.kind.value, job.id, job.process.pid
        )

    async def _watch(self, job: Job) -> None:
        assert job.process is not None and job.process.stdout is not None
        stdout = await job.process.stdout.read()
        code = await job.process.wait()
        job.ended = time.time()
        job.exit_code = code
        document = _last_json(stdout.decode("utf-8", errors="replace"))
        job.result = document
        stopped = job.status is JobStatus.STOPPING
        if code == 0:
            job.status = JobStatus.SUCCEEDED
        elif stopped or code in (
            128 + signal.SIGTERM,
            128 + signal.SIGINT,
            -signal.SIGTERM,
            -signal.SIGKILL,
        ):
            job.status = JobStatus.STOPPED
        else:
            job.status = JobStatus.FAILED
        if job.status is not JobStatus.SUCCEEDED:
            job.error = (document or {}).get("error") or _last_error(self.log_tail(job))
        self.store.save_job(job.record())
        logger.info(
            "%s job %s ended: %s (exit %s)",
            job.kind.value,
            job.id,
            job.status.value,
            code,
        )
        if job.kind is JobKind.INGEST and self.on_ingest_end is not None:
            try:
                self.on_ingest_end()
            except Exception as e:  # noqa: BLE001 - a refresh must not fail the job
                logger.warning("Refreshing after job %s failed: %s", job.id, e)

    # -- stopping ---------------------------------------------------------

    async def stop(self, job_id: str) -> Job:
        """Ask a running job to stop (SIGTERM); kill it after the grace.

        Raises:
            KeyError: No such job.
        """
        job = self.jobs[job_id]
        if job.status is JobStatus.RUNNING and job.process is not None:
            job.status = JobStatus.STOPPING
            self.store.save_job(job.record())
            with contextlib.suppress(ProcessLookupError):
                job.process.send_signal(signal.SIGTERM)
            asyncio.get_running_loop().call_later(STOP_GRACE_SECONDS, self._kill, job)
        return job

    def _kill(self, job: Job) -> None:
        if job.process is not None and job.process.returncode is None:
            logger.warning("Job %s outlived its stop grace; killing it", job.id)
            with contextlib.suppress(ProcessLookupError):
                job.process.kill()

    async def shutdown(self, grace_seconds: float = 30.0) -> None:
        """Stop every running job and wait (up to ``grace_seconds``) for them."""
        running = [j for j in self.jobs.values() if j.status in _ACTIVE]
        for job in running:
            await self.stop(job.id)
        watchers = [j.watcher for j in running if j.watcher is not None]
        if watchers:
            _, pending = await asyncio.wait(watchers, timeout=grace_seconds)
            for job in running:
                if job.watcher in pending:
                    self._kill(job)

    # -- reading ----------------------------------------------------------

    def get(self, job_id: str) -> dict[str, Any]:
        """A job's record with its progress and log tail (a finished job of
        an earlier console session from the store).

        Raises:
            KeyError: No such job.
        """
        job = self.jobs.get(job_id)
        if job is None:
            for record in self.store.jobs():
                if record["id"] == job_id:
                    return {**record, "progress": None, "log": []}
            raise KeyError(job_id)
        return {
            **job.record(),
            "progress": self.progress(job),
            "log": self.log_tail(job),
        }

    def all_jobs(self) -> list[dict[str, Any]]:
        live = {job.id: job for job in self.jobs.values()}
        records = [
            {**job.record(), "progress": self.progress(job)} for job in live.values()
        ]
        records.extend(r for r in self.store.jobs() if r["id"] not in live)
        return sorted(records, key=lambda r: r["created"], reverse=True)

    def active(self) -> list[dict[str, Any]]:
        return [j.record() for j in self.jobs.values() if j.status in _ACTIVE]

    def progress(self, job: Job) -> dict[str, Any]:
        running = job.status in _ACTIVE
        if job.kind is JobKind.INGEST:
            assert job.pipeline_id is not None
            metadata = stages.read_metadata(self.cache_directory, job.pipeline_id)
            started = (
                datetime.fromtimestamp(job.started).isoformat() if job.started else None
            )
            steps = stages.stage_progress(
                job.plan, metadata, running=running, started_after=started
            )
            done = sum(s["status"] in ("completed", "restored") for s in steps)
            return {"stages": steps, "done": done, "total": len(steps)}
        assert job.output_directory is not None
        return evaluation_progress(job.output_directory, job.total_runs)

    def log_tail(self, job: Job, lines: int = LOG_TAIL_LINES) -> list[str]:
        try:
            with job.log_path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - 64_000))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        cleaned = [_ANSI.sub("", line).rstrip() for line in text.splitlines()]
        return [line for line in cleaned if line.strip()][-lines:]


def evaluation_progress(output: Path, total: int) -> dict[str, Any]:
    """How many of an evaluation's runs (strategies x repeats) have saved
    their summary."""
    done = len(
        [
            p
            for p in walk_files(output, limit=1000)
            if p.name.startswith("evaluation_summary_")
        ]
    )
    return {"done": min(done, total) if total else done, "total": total}


def _last_json(text: str) -> dict[str, Any] | None:
    """The JSON document a ``--output-format json`` run printed (the last
    one on stdout)."""
    text = text.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except ValueError:
        start = text.rfind("\n{")
        try:
            value = json.loads(text[start + 1 :] if start >= 0 else text)
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def _last_error(lines: list[str]) -> str | None:
    for line in reversed(lines):
        if "error" in line.lower():
            return line.strip()
    return lines[-1].strip() if lines else None
