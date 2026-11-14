# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ingestion and evaluation jobs: the unchanged CLIs in child processes.

A job is ``run-ingestion`` or ``run-eval --output-format json`` started with
the console's directories. Stop sends SIGTERM, the stop the CLIs are built
for (the pipeline checkpoints its finished stages and exits 143; ``run-eval``
keeps the runs it finished); one still running after the CLIs' stop grace is
killed. Progress is read from what the run writes: an ingestion's stage
metadata (``stages``), an evaluation's answer progress lines in its log
(:class:`AnswerProgress`) and saved summaries (:func:`evaluation_progress`);
the result from its JSON document on stdout.
One job per index namespace runs at a time (docs/design.md#101-the-console-run-console-opt-in).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from unified_kg_rag.application.cli.console import stages
from unified_kg_rag.application.cli.console.paths import PathRefused, walk_files
from unified_kg_rag.application.cli.console.store import ConsoleStore
from unified_kg_rag.application.ingestion.pipeline_manager import INTEGRITY_QUESTIONS
from unified_kg_rag.domain.ingestion.methodologies import artifacts_to_build
from unified_kg_rag.domain.models import (
    Config,
    Methodology,
    SearchStrategy,
    validate_path_segment,
    validate_safe_name,
)
from unified_kg_rag.domain.models.methodology import METHODOLOGY_PROFILES
from unified_kg_rag.domain.retrieval.store_names import resolve_namespace
from unified_kg_rag.evaluation.gates import parse_gate
from unified_kg_rag.shared import InvalidArgumentError, get_logger
from unified_kg_rag.shared.logging import log_error
from unified_kg_rag.shared.utils.processes import process_gone, process_identity
from unified_kg_rag.shared.utils.validation import format_validation_errors

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
    # Left running by a console process that ended (see _mark_lost).
    LOST = "lost"


_ACTIVE = (JobStatus.RUNNING, JobStatus.STOPPING)


class JobConflict(InvalidArgumentError):
    """Another job runs in the namespace (the request is refused)."""

    def __init__(self, message: str, job_id: str) -> None:
        super().__init__(message)
        self.job_id = job_id


class NarrowingNotConfirmed(InvalidArgumentError):
    """An ingestion that builds fewer methodologies than its namespace
    holds, not confirmed (the request is refused)."""

    def __init__(self, namespace: str, dropped: list[str]) -> None:
        super().__init__(
            f"the index namespace '{namespace}' holds {', '.join(dropped)}, which "
            "this run does not build: it would stop serving them for every "
            "corpus in the namespace. Pick them too, or confirm the narrowing"
        )
        self.namespace = namespace
        self.dropped = dropped


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
    # An evaluation's answers so far, read from its log (AnswerProgress).
    answers: AnswerProgress | None = None
    # The question a failed ingestion stopped at, unanswered (its flag),
    # and for the integrity question its reason (IntegrityReason).
    confirmation: str | None = None
    confirmation_reason: str | None = None
    # The WARNING events of its log once it ended (_log_warnings): how many,
    # the last ones, and whether one advises rebuilding the namespace.
    warning_count: int = 0
    warnings: list[str] = field(default_factory=list)
    reset_advised: bool = False
    process: asyncio.subprocess.Process | None = None
    identity: str | None = None
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
            "confirmation": self.confirmation,
            "confirmation_reason": self.confirmation_reason,
            "warnings": {
                "count": self.warning_count,
                "lines": self.warnings,
                "reset_advised": self.reset_advised,
            },
            "pid": self.process.pid if self.process is not None else None,
            # Its pid with its start time (process_identity): a later console
            # tells it from a process that reused the pid.
            "process": self.identity,
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


def ingest_config(
    config: Config, methodologies: list[str] | None
) -> tuple[Config, list[str] | None]:
    """The config a console ingestion of ``methodologies`` (None: the
    config's) runs with, and the methodologies' values.

    Raises:
        InvalidArgumentError: An unknown methodology.
    """
    if not methodologies:
        return config, None
    try:
        chosen = [Methodology(m).value for m in methodologies]
    except ValueError as e:
        raise InvalidArgumentError(f"methodologies: {e}") from e
    try:
        # Validated whole: a selection a switch leaves nothing of is refused
        # here, as at load, not left with no methodology to build.
        updated = config.updated({"indexing.methodologies": chosen})
    except ValidationError as e:
        raise InvalidArgumentError(
            f"methodologies: {format_validation_errors('', e)}"
        ) from e
    return updated, chosen


def dropped_methodologies(held: list[str], config: Config) -> list[str]:
    """The ``held`` methodologies a run of ``config`` does not build the
    required artifacts of: after it the namespace no longer serves them.
    (One that reads only what another builds, as HippoRAG 2 reads
    LightRAG's, is not dropped.)"""
    built = artifacts_to_build(config)
    known = {m.value: m for m in Methodology}
    return [
        m
        for m in held
        if m in known and not METHODOLOGY_PROFILES[known[m]].required <= built
    ]


def _pipeline_id(corpus: Path, namespace: str) -> str:
    """A stable pipeline id per corpus directory and namespace, so a rerun of
    the same corpus resumes or runs incrementally on the same cache: a
    readable slug of the folder name and a hash of the resolved path and the
    namespace, so two corpora never share one (``uploads:docs`` and
    ``workspace:docs``, ``a b`` and ``a-b``, names in any script)."""
    resolved = str(corpus.resolve())
    digest = hashlib.sha256(f"{resolved}\0{namespace}".encode()).hexdigest()[:10]
    slug = _SLUG.sub("-", corpus.name).strip("-._")[:60]
    return validate_path_segment(
        f"console-{slug}-{digest}" if slug else f"console-{digest}", "pipeline_id"
    )


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
        read_held: Callable[[str], Awaitable[list[str] | None]] | None = None,
        cwd: Path | None = None,
    ) -> None:
        self.store = store
        self.jobs_directory = jobs_directory
        self.cache_directory = cache_directory
        self.evaluation_outputs = evaluation_outputs
        self.config_path = config_path
        self.load_config = load_config
        self.on_ingest_end = on_ingest_end
        # The methodologies a namespace's build record says it holds (None:
        # no record), for refusing an unconfirmed narrowing.
        self.read_held = read_held
        self.cwd = cwd or Path.cwd()
        self.jobs: dict[str, Job] = {}
        # Namespaces a request has claimed before its job is registered: two
        # requests at once must not both pass the check across an await.
        self._claimed: dict[str, str] = {}
        self._mark_lost()

    def _mark_lost(self) -> None:
        """Record the jobs an earlier console process left running as lost:
        it ended without seeing them end, so nothing watches them any more
        (their child may still run; the namespace lease guards the stores)."""
        for record in self.store.jobs(limit=None, statuses=[s.value for s in _ACTIVE]):
            pid = record.get("pid")
            identity = record.get("process")
            alive = isinstance(identity, str) and not process_gone(identity)
            record.update(
                status=JobStatus.LOST.value,
                ended=record.get("ended") or time.time(),
                error=(
                    "the console stopped while this job ran"
                    + (f"; its process {pid} still runs" if alive else "")
                ),
            )
            self.store.save_job(record)

    # -- starting ---------------------------------------------------------

    def _claim(self, namespace: str, job_id: str) -> None:
        """Claim ``namespace`` for ``job_id``, synchronously (no await
        between the check and the claim).

        Raises:
            JobConflict: A job runs in it, or another request claimed it.
        """
        for job in self.jobs.values():
            if job.namespace == namespace and job.status in _ACTIVE:
                raise JobConflict(
                    f"job {job.id} ({job.kind.value}) is running in namespace "
                    f"'{namespace}'; stop it or wait for it to finish",
                    job.id,
                )
        if (other := self._claimed.get(namespace)) is not None:
            raise JobConflict(
                f"job {other} is starting in namespace '{namespace}'", other
            )
        self._claimed[namespace] = job_id

    def _release(self, namespace: str, job_id: str) -> None:
        if self._claimed.get(namespace) == job_id:
            del self._claimed[namespace]

    @staticmethod
    def _new_id() -> str:
        return f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    def _new_job(
        self,
        job_id: str,
        kind: JobKind,
        namespace: str,
        params: dict[str, Any],
        argv: list[str],
        env: dict[str, str],
    ) -> Job:
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
        reset: bool = False,
        confirm_reset: bool = False,
        confirm_deletions: bool = False,
        confirm_integrity: bool = False,
        confirm_narrowing: bool = False,
    ) -> Job:
        """Start ``run-ingestion`` over ``corpus`` (a resolved directory);
        ``reset`` rebuilds the namespace from the whole corpus for this run
        only (``--reset``), as a warning about drift or re-clustering asks.

        A question the run may ask is answered only by its own flag:
        ``confirm_integrity`` (continue a pipeline id whose inputs changed),
        ``confirm_reset``, ``confirm_deletions``; unanswered, the run stops
        before it writes and the job's ``confirmation`` names the question.
        A run that builds fewer methodologies than the namespace's build
        record holds starts only with ``confirm_narrowing``: the namespace
        would stop serving the others, for every corpus in it.

        Raises:
            JobConflict: A job runs in the namespace.
            NarrowingNotConfirmed: It narrows the namespace, unconfirmed.
            InvalidArgumentError: An unknown methodology or a bad index value.
        """
        config, chosen = ingest_config(self.load_config(), methodologies)
        env: dict[str, str] = {}
        if chosen:
            env["UKR__INDEXING__METHODOLOGIES"] = json.dumps(chosen)
        if index_value:
            try:
                index_value = validate_safe_name(index_value, "index value")
            except ValueError as e:
                raise InvalidArgumentError(str(e)) from e
            env["UKR__PROCESSING__DOCUMENT_PARSING__INDEX_VALUE"] = index_value
        namespace = resolve_namespace(config, index_value)
        if not confirm_narrowing:
            await self._check_narrowing(namespace, config)
        job_id = self._new_id()
        self._claim(namespace, job_id)
        try:
            return await self._start_ingest(
                job_id,
                config,
                env,
                namespace,
                corpus,
                corpus_ref,
                methodologies,
                index_value,
                force_rebuild,
                reset,
                confirm_reset,
                confirm_deletions,
                confirm_integrity,
                confirm_narrowing,
            )
        finally:
            self._release(namespace, job_id)

    async def _check_narrowing(self, namespace: str, config: Config) -> None:
        """Raise ``NarrowingNotConfirmed`` if ``config`` builds fewer of the
        methodologies than ``namespace`` holds. A record that cannot be read
        is not a refusal: the run meets the store error itself."""
        if self.read_held is None:
            return
        try:
            held = await self.read_held(namespace)
        except Exception as e:  # noqa: BLE001 - the run reports a store error
            log_error(
                logger,
                logging.WARNING,
                "Reading what namespace '%s' holds failed: %s",
                namespace,
                error=e,
            )
            return
        if dropped := dropped_methodologies(held or [], config):
            raise NarrowingNotConfirmed(namespace, dropped)

    async def _start_ingest(
        self,
        job_id: str,
        config: Config,
        env: dict[str, str],
        namespace: str,
        corpus: Path,
        corpus_ref: str,
        methodologies: list[str] | None,
        index_value: str | None,
        force_rebuild: bool,
        reset: bool,
        confirm_reset: bool,
        confirm_deletions: bool,
        confirm_integrity: bool,
        confirm_narrowing: bool,
    ) -> Job:
        if not await asyncio.to_thread(corpus.is_dir):
            raise PathRefused(f"'{corpus_ref}' is not a directory")
        pipeline_id = _pipeline_id(corpus, namespace)
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
        if reset:
            argv.append("--reset")
        if confirm_reset:
            argv.append("--confirm-reset")
        if confirm_deletions:
            argv.append("--confirm-deletions")
        if confirm_integrity:
            argv.append("--confirm-integrity")
        job = self._new_job(
            job_id,
            JobKind.INGEST,
            namespace,
            {
                "corpus": corpus_ref,
                "methodologies": methodologies
                or [m.value for m in config.effective_methodologies],
                "index_value": index_value,
                "force_rebuild": force_rebuild,
                "reset": reset,
                "confirm_reset": confirm_reset,
                "confirm_deletions": confirm_deletions,
                "confirm_integrity": confirm_integrity,
                "confirm_narrowing": confirm_narrowing,
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
            InvalidArgumentError: An unknown strategy, a bad gate or repeat count.
        """
        config = self.load_config()
        if not strategies:
            raise InvalidArgumentError("choose at least one strategy")
        try:
            chosen = [SearchStrategy(s).value for s in strategies]
        except ValueError as e:
            raise InvalidArgumentError(f"strategies: {e}") from e
        if len(set(chosen)) != len(chosen):
            raise InvalidArgumentError("a strategy is listed twice")
        if not 1 <= repeats <= MAX_REPEATS:
            raise InvalidArgumentError(f"repeats must be between 1 and {MAX_REPEATS}")
        gates = [g.strip() for g in fail_if or [] if g.strip()]
        for gate in gates:
            try:
                parse_gate(gate)
            except ValueError as e:
                raise InvalidArgumentError(f"gate '{gate}': {e}") from e
        if gates and len(chosen) < 2:
            raise InvalidArgumentError("a gate needs two or more strategies to compare")
        if pairwise_judge and len(chosen) + int(closed_book) < 2:
            raise InvalidArgumentError(
                "the pairwise judge needs two or more strategies"
            )
        if max_failure_rate is not None and not 0.0 <= max_failure_rate <= 1.0:
            raise InvalidArgumentError("the failure budget must be between 0 and 1")
        if not await asyncio.to_thread(dataset.is_file):
            raise PathRefused(f"'{dataset_ref}' is not a file")
        namespace = resolve_namespace(config, None)
        job_id = self._new_id()
        self._claim(namespace, job_id)
        try:
            return await self._start_evaluate(
                job_id,
                namespace,
                dataset,
                dataset_ref,
                chosen,
                repeats,
                closed_book,
                pairwise_judge,
                gates,
                max_failure_rate,
            )
        finally:
            self._release(namespace, job_id)

    async def _start_evaluate(
        self,
        job_id: str,
        namespace: str,
        dataset: Path,
        dataset_ref: str,
        chosen: list[str],
        repeats: int,
        closed_book: bool,
        pairwise_judge: bool,
        gates: list[str],
        max_failure_rate: float | None,
    ) -> Job:
        config = self.load_config()
        job = self._new_job(job_id, JobKind.EVALUATE, namespace, {}, [], {})
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
        # As run-eval orders them (evaluation.run_order, one run: one at a
        # time): interleaved runs answer in one batch and are saved only
        # once every answer is in, so a stop keeps none of them.
        run_order = (
            str(config.evaluation.run_order) if job.total_runs > 1 else "sequential"
        )
        job.params["run_order"] = run_order
        job.answers = AnswerProgress(
            batches=1 if run_order == "interleaved" else job.total_runs
        )
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
        job.identity = process_identity(job.process.pid)
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
        if job.kind is JobKind.INGEST and self.on_ingest_end is not None:
            # With its status, before any await: no question meets the
            # replaced index through the old chain once the job reads ended.
            try:
                self.on_ingest_end()
            except Exception as e:  # noqa: BLE001 - a refresh must not fail the job
                log_error(
                    logger,
                    logging.WARNING,
                    "Refreshing after job %s failed: %s",
                    job.id,
                    error=e,
                )
        job.warning_count, job.warnings, job.reset_advised = await asyncio.to_thread(
            _log_warnings, job.log_path
        )
        if job.status is JobStatus.FAILED:
            job.error = (document or {}).get("error") or _last_error(self.log_tail(job))
            job.confirmation, job.confirmation_reason = confirmation_needed(job.error)
        self.store.save_job(job.record())
        logger.info(
            "%s job %s ended: %s (exit %s)",
            job.kind.value,
            job.id,
            job.status.value,
            code,
        )

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
            # The CLI's own stop (it stops its workers and checkpoints).
            with contextlib.suppress(ProcessLookupError):
                job.process.send_signal(signal.SIGTERM)
            asyncio.get_running_loop().call_later(STOP_GRACE_SECONDS, self._kill, job)
        return job

    def _kill(self, job: Job) -> None:
        """Kill a job that outlived its stop grace, with its process group
        (its own session): worker processes it started are not left behind."""
        if job.process is not None and job.process.returncode is None:
            logger.warning("Job %s outlived its stop grace; killing it", job.id)
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(job.process.pid, signal.SIGKILL)
            with contextlib.suppress(ProcessLookupError):
                job.process.kill()

    async def shutdown(self, grace_seconds: float = STOP_GRACE_SECONDS) -> None:
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
            if (record := self.store.job(job_id)) is None:
                raise KeyError(job_id)
            return {**record, "progress": None, "log": []}
        return {
            **job.record(),
            "progress": self.progress(job),
            "log": self.log_tail(job),
        }

    def all_jobs(
        self, *, since: float | None = None, limit: int | None = 200
    ) -> list[dict[str, Any]]:
        """This session's jobs and the stored ones (the newest ``limit``),
        created at or after ``since`` when given."""
        live = {
            job.id: job
            for job in self.jobs.values()
            if since is None or job.created >= since
        }
        records = [
            {**job.record(), "progress": self.progress(job)} for job in live.values()
        ]
        records.extend(
            r for r in self.store.jobs(limit=limit, since=since) if r["id"] not in live
        )
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
        progress = evaluation_progress(job.output_directory, job.total_runs)
        if job.answers is not None:
            job.answers.read(job.log_path)
            progress["answers"] = job.answers.view()
        return progress

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


# The flag each question run-ingestion asks names in its error, and the
# confirmation the console offers for it.
_CONFIRMATIONS = {
    "--confirm-integrity": "integrity",
    "--confirm-reset": "reset",
    "--confirm-deletions": "deletions",
}


def confirmation_needed(error: str | None) -> tuple[str | None, str | None]:
    """Which question a run stopped at unanswered, from its error, and for
    the integrity question its reason (``IntegrityReason``: the question
    names it), else ``None``."""
    if not error or "onfirmation required" not in error:
        return None, None
    for flag, name in _CONFIRMATIONS.items():
        if flag in error:
            reason = None
            if name == "integrity":
                reason = next(
                    (r for r, q in INTEGRITY_QUESTIONS.items() if q in error), None
                )
            return name, reason
    return None, None


# The answer batch's lines run-eval logs (BatchProcessor, ProgressLogger),
# in the text or the JSON log format alike.
_ANSWER_TASK = "Answer Generation"
_BATCH_START = re.compile(rf"Starting processing for '{_ANSWER_TASK}': (\d+) items")
_BATCH_PROGRESS = re.compile(rf"Progress '{_ANSWER_TASK}': (\d+)/(\d+) items")
_BATCH_DONE = re.compile(rf"Completed '{_ANSWER_TASK}': (\d+) results")


@dataclass
class AnswerProgress:
    """An evaluation's answers so far, from the answer batch progress lines
    of its log: each run answers in a batch of its own (sequential) or every
    run in one (interleaved), logged as it starts, every ~10% (or minute)
    and as it completes. ``batches`` is how many the job answers in. The
    log is read on from where the last read stopped."""

    batches: int
    offset: int = 0
    # Per batch seen: [answered, of].
    seen: list[list[int]] = field(default_factory=list)
    _partial: bytes = b""

    def read(self, log_path: Path) -> None:
        try:
            with log_path.open("rb") as handle:
                handle.seek(self.offset)
                data = handle.read()
        except OSError:
            return
        self.offset += len(data)
        *lines, self._partial = (self._partial + data).split(b"\n")
        for raw in lines:
            self.feed(raw.decode("utf-8", errors="replace"))

    def feed(self, line: str) -> None:
        if match := _BATCH_START.search(line):
            self.seen.append([0, int(match.group(1))])
        elif self.seen and (match := _BATCH_PROGRESS.search(line)):
            self.seen[-1] = [int(match.group(1)), int(match.group(2))]
        elif self.seen and _BATCH_DONE.search(line):
            self.seen[-1][0] = self.seen[-1][1]

    def view(self) -> dict[str, int] | None:
        """``done`` of ``total`` answers, the batches not started yet counted
        at the size of the last one (``None`` before the first starts, or
        with a log level that hides the lines)."""
        if not self.seen:
            return None
        done = sum(answered for answered, _ in self.seen)
        total = sum(of for _, of in self.seen)
        total += max(0, self.batches - len(self.seen)) * self.seen[-1][1]
        return {"done": min(done, total), "total": total}


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


# How many of a job's WARNING events its record keeps (the last ones).
WARNING_LINES = 20
# The characters the CLIs' banner and panels are drawn with.
_BOX = frozenset("│╭╮╰╯─┃━┏┓┗┛")


def _log_event(line: str) -> dict[str, Any] | None:
    """The structured log event a log line holds (the CLIs log JSON lines
    to their stderr, ``{"event": ..., "level": ...}``), or ``None``."""
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    if isinstance(event, dict) and isinstance(event.get("level"), str):
        return event
    return None


# How a run's warnings advise rebuilding the namespace (community drift,
# re-clustering due, ...): the setting ``--reset`` sets for one run, which
# the Ingest page offers as "Rebuild the namespace".
RESET_ADVICE = "indexing.reset: true"


def _log_warnings(log_path: Path) -> tuple[int, list[str], bool]:
    """How many WARNING events a job's log holds, the last
    :data:`WARNING_LINES` of them (their messages), and whether any of them
    advises a rebuild (:data:`RESET_ADVICE`)."""
    count, last, reset_advised = 0, [], False
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                event = _log_event(line)
                if event is not None and event["level"] == "warning":
                    count += 1
                    message = str(event.get("event", ""))
                    reset_advised = reset_advised or RESET_ADVICE in message
                    last.append(message)
                    del last[:-WARNING_LINES]
    except OSError:
        return 0, [], False
    return count, last, reset_advised


def _last_error(lines: list[str]) -> str | None:
    """A failed job's error, when its JSON status names none: the last
    ERROR (or CRITICAL) event of its log, else of its lines outside the
    banner's and panels' boxes and the structured events (a traceback's)
    the last that names an error, else the last, else ``None``."""
    for line in reversed(lines):
        event = _log_event(line)
        if event is not None and event["level"] in ("error", "critical"):
            return str(event.get("event", "")).strip() or None
    plain = [
        text
        for line in lines
        if (text := line.strip()) and text[0] not in _BOX and _log_event(line) is None
    ]
    named = [text for text in plain if "error" in text.lower()]
    found = named or plain
    return found[-1] if found else None
