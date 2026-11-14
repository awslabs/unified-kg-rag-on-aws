# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console's job manager and question service on their own: one job per
namespace under concurrent requests, jobs a crashed console left running,
pipeline ids, retired chains closed once idle, blocked answers recorded,
evaluation progress from the answer batch, what a namespace holds and an
ingestion that would narrow it."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from unified_kg_rag.application.cli.console import dashboard
from unified_kg_rag.application.cli.console import jobs as jobs_module
from unified_kg_rag.application.cli.console.ask import AskRequest, AskService
from unified_kg_rag.application.cli.console.jobs import (
    STOP_GRACE_SECONDS,
    JobConflict,
    JobManager,
    JobStatus,
    NarrowingNotConfirmed,
    _pipeline_id,
    confirmation_needed,
)
from unified_kg_rag.application.cli.console.store import ConsoleStore
from unified_kg_rag.application.ingestion.pipeline_manager import INTEGRITY_QUESTIONS
from unified_kg_rag.application.retrieval.rag_chain import ProcessedQuery, RAGOutput
from unified_kg_rag.domain.models import (
    Config,
    Methodology,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from unified_kg_rag.shared.exceptions import (
    INTERNAL_ERROR_MESSAGE,
    IndexNotFoundError,
    StoreConnectionError,
)

pytestmark = pytest.mark.unit


def _manager(tmp_path: Path, store: ConsoleStore | None = None) -> JobManager:
    return JobManager(
        store=store or ConsoleStore(tmp_path / "console.sqlite3"),
        jobs_directory=tmp_path / "jobs",
        cache_directory=tmp_path / "cache",
        evaluation_outputs=tmp_path / "outputs",
        config_path=None,
        load_config=Config,
    )


async def test_two_requests_at_once_start_one_job_per_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = _manager(tmp_path)
    launched: list[str] = []

    async def launch(job: Any) -> None:
        await asyncio.sleep(0.05)  # the subprocess start, an await
        job.status = JobStatus.RUNNING
        manager.jobs[job.id] = job
        launched.append(job.id)

    monkeypatch.setattr(manager, "_launch", launch)
    dataset = tmp_path / "d.json"
    dataset.write_text('[{"question": "q"}]')
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    results = await asyncio.gather(
        manager.start_evaluate(
            dataset=dataset, dataset_ref="w:d.json", strategies=["local"]
        ),
        manager.start_ingest(corpus=corpus, corpus_ref="w:corpus"),
        manager.start_evaluate(
            dataset=dataset, dataset_ref="w:d.json", strategies=["mix"]
        ),
        return_exceptions=True,
    )
    assert len(launched) == 1
    assert sum(isinstance(r, JobConflict) for r in results) == 2
    # The claim is released once the job is registered (or failed to start).
    assert manager._claimed == {}


async def test_a_job_that_fails_to_start_frees_its_namespace(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with pytest.raises(jobs_module.PathRefused):
        await manager.start_ingest(corpus=tmp_path / "missing", corpus_ref="w:missing")
    assert manager._claimed == {}


# A run-ingestion whose runner fails with a provider's text (an unexpected
# failure escaping the stages): the console's job reads its JSON status and
# keeps its stdout and stderr as the job's log.
_FAILING_INGEST = (
    "import sys\n"
    "from unified_kg_rag.application.cli import run_ingestion_pipeline as m\n"
    "def run(self):\n"
    "    raise RuntimeError('Quokkasecret provider reply: role/Zebrapassword')\n"
    "m.IngestionPipelineRunner.run = run\n"
    "m.main()\n"
)


async def test_an_unexpected_failure_of_a_job_shows_no_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression (rv29 U1): the run's last-resort handler masked the log
    # line but printed the text on stdout ("An unexpected error occurred:")
    # and in the JSON status' error, which the Ingest page showed.
    monkeypatch.setattr(
        jobs_module, "_cli", lambda *_: [sys.executable, "-c", _FAILING_INGEST]
    )
    manager = _manager(tmp_path)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    job = await manager.start_ingest(corpus=corpus, corpus_ref="w:corpus")
    assert job.watcher is not None
    await asyncio.wait_for(job.watcher, timeout=60)
    shown = manager.get(job.id)
    assert shown["status"] == "failed"
    # Named by its type, its text masked (conventions.shown_text).
    assert shown["error"] == f"RuntimeError: {INTERNAL_ERROR_MESSAGE}"
    tail = "\n".join(manager.log_tail(job))
    assert "RuntimeError" in tail  # its type and where it happened
    for text in (shown["error"], tail, job.log_path.read_text()):
        assert "Quokkasecret" not in text and "Zebrapassword" not in text


def test_jobs_a_stopped_console_left_running_are_marked_lost(tmp_path: Path) -> None:
    store = ConsoleStore(tmp_path / "console.sqlite3")
    for job_id, status in (("a", "running"), ("b", "stopping"), ("c", "succeeded")):
        store.save_job(
            {
                "id": job_id,
                "kind": "ingest",
                "namespace": "default",
                "created": time.time(),
                "status": status,
                "pid": 2**22 + 7,
            }
        )
    manager = _manager(tmp_path, store)
    by_id = {j["id"]: j for j in manager.all_jobs()}
    assert by_id["a"]["status"] == by_id["b"]["status"] == "lost"
    assert "stopped while this job ran" in by_id["a"]["error"]
    assert by_id["c"]["status"] == "succeeded"
    manager._claim("default", "new")  # the namespace is free again


@pytest.mark.skipif(
    not Path("/proc/self/stat").exists(), reason="needs /proc for start times"
)
def test_a_lost_job_s_process_is_told_from_one_that_reused_its_pid(
    tmp_path: Path,
) -> None:
    # Regression (rv25): _alive(pid) took any process holding the pid for
    # the job's, so a reused pid read "its process ... still runs".
    from unified_kg_rag.shared.utils.processes import process_identity

    me = process_identity()
    host, pid, started, namespace = me.split(":")
    reused = f"{host}:{pid}:{int(started) + 1}:{namespace}"
    store = ConsoleStore(tmp_path / "console.sqlite3")
    for job_id, identity in (("alive", me), ("reused", reused), ("old", None)):
        store.save_job(
            {
                "id": job_id,
                "kind": "ingest",
                "namespace": job_id,
                "created": time.time(),
                "status": "running",
                "pid": int(pid),
                **({"process": identity} if identity else {}),
            }
        )
    by_id = {j["id"]: j for j in _manager(tmp_path, store).all_jobs()}
    assert "still runs" in by_id["alive"]["error"]
    assert "still runs" not in by_id["reused"]["error"]
    # A record of an earlier release, without the identity: not judged.
    assert "still runs" not in by_id["old"]["error"]


def test_jobs_past_the_newest_200_are_found_by_id_and_marked_lost(
    tmp_path: Path,
) -> None:
    store = ConsoleStore(tmp_path / "console.sqlite3")
    now = time.time()
    store.save_job(
        {"id": "old", "kind": "ingest", "namespace": "n", "created": now - 900}
        | {"status": "running", "pid": 2**22 + 7}
    )
    for index in range(250):
        store.save_job(
            {"id": f"j{index}", "kind": "ingest", "namespace": "n"}
            | {"created": now - index, "status": "succeeded"}
        )
    manager = _manager(tmp_path, store)
    assert manager.get("old")["status"] == "lost"  # older than the newest 200
    with pytest.raises(KeyError):
        manager.get("missing")
    assert len(manager.all_jobs()) == 200
    assert len(manager.all_jobs(since=now - 99.5, limit=None)) == 100
    assert len(manager.all_jobs(since=0.0, limit=None)) == 251


def test_a_job_that_outlives_its_stop_grace_is_killed_with_its_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        jobs_module.os, "killpg", lambda pid, sig: killed.append((pid, sig))
    )

    class _Process:
        pid, returncode = 4321, None

        def kill(self) -> None:
            killed.append((self.pid, -1))

    manager = _manager(tmp_path)
    job = jobs_module.Job(
        id="j",
        kind=jobs_module.JobKind.INGEST,
        namespace="n",
        params={},
        argv=[],
        env={},
        work_dir=tmp_path,
        log_path=tmp_path / "log",
    )
    job.process = _Process()  # type: ignore[assignment]
    manager._kill(job)
    assert killed[0] == (4321, jobs_module.signal.SIGKILL)


def test_pipeline_ids_differ_per_folder_and_name(tmp_path: Path) -> None:
    folders = ["docs", "uploads/docs", "a b", "a-b", "계약서", "契約書", "合同"]
    paths = []
    for name in folders:
        path = tmp_path / name
        path.mkdir(parents=True)
        paths.append(path)
    ids = [_pipeline_id(p, "default") for p in paths]
    assert len(set(ids)) == len(ids)
    assert ids[0].startswith("console-docs-")
    assert _pipeline_id(paths[0], "default") == ids[0]  # stable
    assert _pipeline_id(paths[0], "tenant") != ids[0]


def test_the_question_a_run_stopped_at_is_named() -> None:
    for reason, question in INTEGRITY_QUESTIONS.items():
        error = f"Confirmation required ({question}); pass --confirm-integrity"
        assert confirmation_needed(error) == ("integrity", reason)
    # An error of a run before the reasons: the question, no reason.
    assert confirmation_needed(
        "Confirmation required (Continue despite integrity issues?); "
        "pass --confirm-integrity"
    ) == ("integrity", None)
    assert confirmation_needed(
        "Confirmation required (x); pass --confirm-deletions (or UKR_CONFIRM_DELETIONS=1)"
    ) == ("deletions", None)
    assert confirmation_needed("Pipeline failed at stage(s): indexing") == (None, None)
    assert confirmation_needed(None) == (None, None)


def test_shutdown_waits_the_documented_stop_grace() -> None:
    import inspect

    default = inspect.signature(JobManager.shutdown).parameters["grace_seconds"].default
    assert default == STOP_GRACE_SECONDS >= 100


# -- questions -----------------------------------------------------------------


def _output(**metadata: Any) -> RAGOutput:
    return RAGOutput(
        answer="Refused.",
        sources=[],
        search_results=SearchResult(
            query=SearchQuery(query="q"),
            results=[],
            total_results=0,
            search_strategy="local",
            processing_time=0.0,
        ),
        conversation_id=None,
        processed_query=ProcessedQuery(original_query="q", final_query="q"),
        metadata={"search_strategy": "local", **metadata},
    )


class _Chain:
    def __init__(self, output: RAGOutput, gate: asyncio.Event | None = None) -> None:
        self.output = output
        self.gate = gate
        self.closed = 0

    async def ainvoke(self, *_: Any, **__: Any) -> RAGOutput:
        if self.gate is not None:
            await self.gate.wait()
        return self.output

    async def aclose(self) -> None:
        self.closed += 1


async def _drain(service: AskService, query: str) -> list[dict[str, Any]]:
    return [e async for e in service.ask(AskRequest(query=query))]


async def test_a_replaced_chain_closes_once_its_question_finishes(
    tmp_path: Path,
) -> None:
    gate = asyncio.Event()
    chains = [_Chain(_output(), gate), _Chain(_output())]
    service = AskService(
        Config,
        ConsoleStore(tmp_path / "s.sqlite3"),
        chain_factory=lambda _c: chains.pop(0),
    )
    first = chains[0]
    pending = asyncio.create_task(_drain(service, "q1"))
    await asyncio.sleep(0.05)
    service.reset()  # an ingestion ended while q1 is answered
    await asyncio.sleep(0.05)
    assert first.closed == 0
    gate.set()
    await pending
    await asyncio.sleep(0.05)
    assert first.closed == 1
    assert service._retired == []
    second = chains[0]
    await _drain(service, "q2")
    service.reset()  # idle: closed at once
    await asyncio.sleep(0.05)
    assert second.closed == 1
    await service.close()


async def test_a_reset_during_a_chain_build_rebuilds_from_the_new_config(
    tmp_path: Path,
) -> None:
    import threading

    started, release = threading.Event(), threading.Event()
    configs = iter(["old", "new"])
    built: list[tuple[str, _Chain]] = []

    def factory(config: str) -> _Chain:
        chain = _Chain(_output())
        built.append((config, chain))
        if config == "old":  # a slow build the save lands during
            started.set()
            release.wait(5)
        return chain

    service = AskService(
        lambda: next(configs),  # type: ignore[arg-type,return-value]
        ConsoleStore(tmp_path / "s.sqlite3"),
        chain_factory=factory,  # type: ignore[arg-type]
    )
    pending = asyncio.create_task(service.chain())
    await asyncio.to_thread(started.wait, 5)
    service.reset()  # the config was saved while the old one was built
    release.set()
    chain, config = await pending
    assert config == "new" and chain is built[1][1]
    await asyncio.sleep(0.05)
    assert built[0][1].closed == 1  # the stale build is closed, not kept
    await service.close()


async def test_a_blocked_answer_is_recorded_as_blocked(tmp_path: Path) -> None:
    store = ConsoleStore(tmp_path / "s.sqlite3")
    chain = _Chain(_output(blocked=True, guardrail=[{"source": "INPUT"}]))
    service = AskService(Config, store, chain_factory=lambda _c: chain)
    events = await _drain(service, "q")
    assert events[-1]["kind"] == "result" and events[-1]["blocked"]
    assert store.queries()[0]["outcome"] == "blocked"
    await service.close()


def test_run_ingestion_has_a_flag_for_the_integrity_question_alone() -> None:
    from unified_kg_rag.application.cli.run_ingestion_pipeline import (
        CommandLineInterface,
    )

    parser = CommandLineInterface._setup_arguments()
    args = parser.parse_args(["--source-directory", ".", "--confirm-integrity"])
    assert args.confirm_integrity and not args.yes and not args.confirm_reset


async def test_an_ingest_job_can_rebuild_its_namespace_for_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``reset`` passes ``--reset`` (indexing.reset for this run only), so
    the warnings that ask for a rebuild need no Settings edit to undo."""
    from unified_kg_rag.application.cli.run_ingestion_pipeline import (
        CommandLineInterface,
    )

    manager = _manager(tmp_path)

    async def launch(job: Any) -> None:
        manager.jobs[job.id] = job

    monkeypatch.setattr(manager, "_launch", launch)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    parser = CommandLineInterface._setup_arguments()
    for reset in (True, False):
        job = await manager.start_ingest(
            corpus=corpus, corpus_ref="w:corpus", reset=reset
        )
        flags = job.argv[job.argv.index("--source-directory") :]
        assert parser.parse_args(flags).reset is reset
        assert job.params["reset"] is reset
        manager.jobs.clear()


class _StreamingChain(_Chain):
    """Streams its answer in chunks that split a citation marker."""

    def __init__(self, output: RAGOutput, chunks: list[str]) -> None:
        super().__init__(output)
        self.chunks = chunks
        self.inputs: list[Any] = []

    async def ainvoke(self, rag_input: Any, *_: Any, **kwargs: Any) -> RAGOutput:
        self.inputs.append(rag_input)
        for chunk in self.chunks:
            kwargs["on_answer_chunk"](chunk)
        return self.output


async def test_chunks_carry_the_markers_they_close(tmp_path: Path) -> None:
    chunks = ["Vendor ships [1", ", 2] and", " stores [3]", "."]
    output = _output()
    output.answer = "".join(chunks)
    chain = _StreamingChain(output, chunks)
    service = AskService(
        Config, ConsoleStore(tmp_path / "s.sqlite3"), chain_factory=lambda _c: chain
    )
    events = await _drain(service, "q")
    markers = [e["markers"] for e in events if e["kind"] == "chunk"]
    assert markers == [[], [1, 2], [3], []]
    assert "".join(e["text"] for e in events if e["kind"] == "chunk") == output.answer
    await service.close()


@pytest.mark.parametrize(
    ("requested", "ran", "streamed", "cites"),
    [
        ("local", "local", [[1, 2], [3]], True),
        # global and drift answers do not cite by default: no markers.
        ("global", "global", [[], []], False),
        ("drift", "drift", [[], []], False),
        # auto's route is known with its result: the page shows the
        # streamed text as plain text once the result says so.
        ("auto", "global", [[1, 2], [3]], False),
    ],
)
async def test_an_answer_shows_markers_only_when_its_strategy_cites(
    tmp_path: Path, requested: str, ran: str, streamed: list[list[int]], cites: bool
) -> None:
    chunks = ["Vendor ships [1, 2]", " and stores [3]"]
    output = _output(search_strategy=ran)
    output.answer = "".join(chunks)
    chain = _StreamingChain(output, chunks)
    service = AskService(
        Config, ConsoleStore(tmp_path / "s.sqlite3"), chain_factory=lambda _c: chain
    )
    events = [
        e
        async for e in service.ask(
            AskRequest(query="q", search_strategy=SearchStrategy(requested))
        )
    ]
    assert [e["markers"] for e in events if e["kind"] == "chunk"] == streamed
    assert events[-1]["kind"] == "result" and events[-1]["cites"] is cites
    await service.close()


async def test_a_follow_up_passes_the_page_s_conversation_as_history(
    tmp_path: Path,
) -> None:
    chain = _StreamingChain(_output(), [])
    service = AskService(
        Config, ConsoleStore(tmp_path / "s.sqlite3"), chain_factory=lambda _c: chain
    )
    turns = [
        {"role": "user", "content": "Who ships?"},
        {"role": "assistant", "content": "Vendor [1]."},
    ]
    request = AskRequest(query="Where to?", history=turns)
    _ = [e async for e in service.ask(request)]
    sent = chain.inputs[0]
    assert [t.model_dump() for t in sent.history] == turns
    assert not sent.use_memory and sent.conversation_id is None
    # A question without one sends none (not an empty conversation).
    _ = [e async for e in service.ask(AskRequest(query="q"))]
    assert chain.inputs[1].history is None
    await service.close()


def test_history_is_bounded_and_only_user_or_assistant() -> None:
    from pydantic import ValidationError

    from unified_kg_rag.application.cli.console.ask import MAX_HISTORY_TURNS

    with pytest.raises(ValidationError):
        AskRequest(query="q", history=[{"role": "system", "content": "x"}])
    with pytest.raises(ValidationError):
        AskRequest(
            query="q",
            history=[{"role": "user", "content": "x"}] * (MAX_HISTORY_TURNS + 1),
        )


def _log_line(event: str) -> str:
    # run-eval's JSON log format (--output-format json), as the console runs it.
    return f'{{"event": "{event}", "logger": "unified_kg_rag.adapters.llm.langchain", "level": "info"}}\n'


@pytest.mark.parametrize(
    ("run_order", "expected"),
    [
        # One batch of every run's answers; no run is saved before it ends.
        ("interleaved", {"done": 12, "total": 40}),
        # A batch per run: the first done (20), the second at 5, of 20 each.
        ("sequential", {"done": 25, "total": 40}),
    ],
)
async def test_evaluation_progress_counts_answers_from_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_order: str, expected: Any
) -> None:
    config = Config()
    config.evaluation.run_order = run_order  # type: ignore[assignment]
    manager = _manager(tmp_path)
    manager.load_config = lambda: config

    async def launch(job: Any) -> None:
        manager.jobs[job.id] = job

    monkeypatch.setattr(manager, "_launch", launch)
    dataset = tmp_path / "d.json"
    dataset.write_text('[{"question": "q"}]')
    job = await manager.start_evaluate(
        dataset=dataset, dataset_ref="workspace:d.json", strategies=["local", "mix"]
    )
    assert job.params["run_order"] == run_order
    before = manager.get(job.id)["progress"]
    assert before == {"done": 0, "total": 2, "answers": None}
    lines = (
        [
            "Starting processing for 'Answer Generation': 40 items, up to 5 at a time",
            "Progress 'Answer Generation': 4/40 items (10%), 2.00 items/s, ETA 18s",
            "Progress 'Answer Generation': 12/40 items (30%), 2.00 items/s, ETA 14s",
        ]
        if run_order == "interleaved"
        else [
            "Starting processing for 'Answer Generation': 20 items, up to 5 at a time",
            "Progress 'Answer Generation': 18/20 items (90%), 2.00 items/s, ETA 1s",
            "Completed 'Answer Generation': 20 results, 0 failed",
            "Starting processing for 'Answer Generation': 20 items, up to 5 at a time",
            "Progress 'Answer Generation': 5/20 items (25%), 2.00 items/s, ETA 8s",
        ]
    )
    with job.log_path.open("a", encoding="utf-8") as log:
        # Another batch's lines (a judge) are not answers.
        log.write(_log_line("Progress 'RAGAS Evaluation': 3/9 items (33%)"))
        for line in lines[:2]:
            log.write(_log_line(line))
    manager.get(job.id)  # read part of the log, then the rest
    with job.log_path.open("a", encoding="utf-8") as log:
        for line in lines[2:]:
            log.write(_log_line(line))
        log.write('{"event": "Progress \'Answer Gen')  # a line still being written
    progress = manager.get(job.id)["progress"]
    assert progress["answers"] == expected
    assert progress["done"] == 0  # no summary saved yet


# -- what a namespace holds, and narrowing it ------------------------------------


class _RecordChain(_Chain):
    """Reads a fixed build record (per suffix)."""

    def __init__(self, records: dict[str | None, Any]) -> None:
        super().__init__(_output())
        self.records = records

    async def aread_build_record(self, suffix: str | None) -> Any:
        return self.records.get(suffix)

    async def acheck_methodology(self, strategy: Any, suffix: str | None) -> Any:
        if suffix == "empty":
            raise IndexNotFoundError(f"No indices found for suffix '{suffix}'")
        return ()


async def test_held_methodologies_come_from_the_namespace_s_build_record(
    tmp_path: Path,
) -> None:
    from unified_kg_rag.domain.models.methodology import METHODOLOGY_PROFILES

    lightrag = sorted(
        a.value for a in METHODOLOGY_PROFILES[Methodology.LIGHTRAG].required
    )
    chain = _RecordChain({"tenant-a": {"artifacts": lightrag}, "old": {"mode": "x"}})
    service = AskService(
        Config, ConsoleStore(tmp_path / "s.sqlite3"), chain_factory=lambda _c: chain
    )
    held = await service.held_methodologies("tenant-a")
    assert held["namespace"] == "tenant-a" and held["recorded"] is True
    # LightRAG's artifacts (HippoRAG 2 reads a subset of them), not GraphRAG's.
    assert "lightrag" in held["held"] and "graphrag" not in held["held"]
    # No record, or one from before methodology profiles: nothing recorded,
    # and what the config builds stands in, as it does for a query.
    for suffix in (None, "old"):
        held = await service.held_methodologies(suffix)
        assert held["recorded"] is False
        assert held["held"] == ["graphrag", "lightrag", "hipporag"]
    assert (await service.held_methodologies(None))["namespace"] == "default"
    # Whether anything was ingested at all (rv25: the page read an empty
    # namespace from "recorded", which an index from before build records
    # lacks too).
    assert (await service.held_methodologies("old"))["indexed"] is True
    empty = await service.held_methodologies("empty")
    assert (empty["indexed"], empty["recorded"]) == (False, False)
    await service.close()


async def test_an_ingestion_that_drops_held_methodologies_needs_a_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # HippoRAG 2 is held on LightRAG's artifacts alone (no build of its own).
    held: dict[str, Any] = {"default": ["graphrag", "lightrag", "hipporag"]}
    asked: list[str] = []

    async def read_held(namespace: str) -> list[str] | None:
        asked.append(namespace)
        value = held.get(namespace)
        if isinstance(value, Exception):
            raise value
        return value

    manager = _manager(tmp_path)
    manager.read_held = read_held

    async def launch(job: Any) -> None:
        manager.jobs[job.id] = job
        job.status = JobStatus.SUCCEEDED  # free the namespace again

    monkeypatch.setattr(manager, "_launch", launch)
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    start = {"corpus": corpus, "corpus_ref": "workspace:corpus"}
    with pytest.raises(NarrowingNotConfirmed) as refused:
        await manager.start_ingest(**start, methodologies=["lightrag"])
    assert refused.value.dropped == ["graphrag"]
    assert refused.value.namespace == "default"
    assert "every corpus" in str(refused.value)
    job = await manager.start_ingest(
        **start, methodologies=["lightrag"], confirm_narrowing=True
    )
    assert job.params["confirm_narrowing"] is True
    # Building what it holds (or more) asks nothing; nor does leaving out a
    # methodology whose artifacts the run still builds.
    await manager.start_ingest(
        **start, methodologies=["graphrag", "lightrag", "raptor"]
    )
    await manager.start_ingest(**start)
    # The namespace the run writes is the one checked.
    await manager.start_ingest(
        **start, methodologies=["lightrag"], index_value="tenant-b"
    )
    assert asked[-1] == "tenant-b"
    # An unreadable record does not refuse: the run meets the store error.
    held["tenant-c"] = StoreConnectionError("down")
    await manager.start_ingest(
        **start, methodologies=["lightrag"], index_value="tenant-c"
    )


async def test_the_result_names_the_namespace_the_question_searched(
    tmp_path: Path,
) -> None:
    service = AskService(
        Config,
        ConsoleStore(tmp_path / "s.sqlite3"),
        chain_factory=lambda _c: _Chain(_output()),
    )
    events = [e async for e in service.ask(AskRequest(query="q", suffix="tenant-a"))]
    assert events[-1]["namespace"] == "tenant-a"
    events = [e async for e in service.ask(AskRequest(query="q"))]
    assert events[-1]["namespace"] == "default"
    await service.close()


async def test_the_question_s_span_names_the_namespace_it_searched(
    tmp_path: Path,
) -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from unified_kg_rag.application.observability import configure_observability
    from unified_kg_rag.domain.models import ObservabilityConfig
    from unified_kg_rag.shared import observability

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    configure_observability(ObservabilityConfig(enabled=True), tracer_provider=provider)
    service = AskService(
        Config,
        ConsoleStore(tmp_path / "s.sqlite3"),
        chain_factory=lambda _c: _Chain(_output()),
    )
    try:
        [e async for e in service.ask(AskRequest(query="q", suffix="tenant-a"))]
        [e async for e in service.ask(AskRequest(query="q"))]
    finally:
        await service.close()
        observability.uninstall()
    names = [
        dict(s.attributes or {}).get("kg_rag.namespace")
        for s in exporter.get_finished_spans()
        if s.name == "kg_rag.console.ask"
    ]
    assert names == ["tenant-a", "default"]


class _FailingChain(_Chain):
    def __init__(self, error: Exception) -> None:
        super().__init__(_output())
        self.error = error

    async def ainvoke(self, *_: Any, **__: Any) -> RAGOutput:
        raise self.error


async def test_a_missing_index_is_reported_as_such(tmp_path: Path) -> None:

    service = AskService(
        Config,
        ConsoleStore(tmp_path / "s.sqlite3"),
        chain_factory=lambda _c: _FailingChain(
            IndexNotFoundError("no index 'x'.", ingested=["tenant-a"])
        ),
    )
    events = await _drain(service, "q")
    assert events[-1]["kind"] == "error"
    assert events[-1]["error_kind"] == "index_not_found"
    # The console (one operator's) names the namespaces the store holds.
    assert events[-1]["message"].endswith("Suffixes ingested on this store: tenant-a.")
    # Recorded as refused, which the dashboard does not count as an error
    # (rv25: an empty namespace's first question was one).
    (row,) = service.store.queries()
    assert row["outcome"] == "refused"
    totals = dashboard.summarize([row], [], Config())["totals"]
    assert (totals["errors"], totals["refused"]) == (0, 1)
    await service.close()


async def test_an_unexpected_failure_shows_no_text_and_keeps_its_ids(
    tmp_path: Path,
) -> None:
    from unified_kg_rag.shared.exceptions import (
        INTERNAL_ERROR_MESSAGE,
        tag_query_failure,
    )

    error = RuntimeError("arn:aws:bedrock:region:000000000000:secret detail")
    tag_query_failure(error, "q-chain", "trace-1")
    service = AskService(
        Config,
        ConsoleStore(tmp_path / "s.sqlite3"),
        chain_factory=lambda _c: _FailingChain(error),
    )
    events = await _drain(service, "q")
    event = events[-1]
    assert (event["kind"], event["error_kind"]) == ("error", "failure")
    assert event["message"] == INTERNAL_ERROR_MESSAGE
    # The console's own id (the chain reuses it), and the trace's.
    assert event["query_id"] != "q-chain" and event["trace_id"] == "trace-1"
    (row,) = service.store.queries()
    assert row["trace_id"] == "trace-1"
    assert "secret" not in str(row)
    await service.close()


async def test_a_hidden_failure_is_logged_under_the_id_the_page_shows(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unexpected error's text is masked on the page, so the server log is
    where it can be found: a record carries the ``query_id`` the page shows
    (and only the error's digest at INFO)."""
    error = RuntimeError("arn:aws:bedrock:region:000000000000:secret detail")
    service = AskService(
        Config,
        ConsoleStore(tmp_path / "s.sqlite3"),
        chain_factory=lambda _c: _FailingChain(error),
    )
    with caplog.at_level(logging.INFO):
        events = await _drain(service, "q")
    query_id = events[-1]["query_id"]
    logged = [
        r
        for r in caplog.records
        if r.levelno >= logging.ERROR
        and (query_id in r.getMessage() or getattr(r, "query_id", None) == query_id)
    ]
    assert logged, [r.getMessage() for r in caplog.records]
    assert all("secret" not in r.getMessage() for r in caplog.records)
    await service.close()


@pytest.mark.parametrize(
    ("error", "logged"),
    [
        (StoreConnectionError("OpenSearch at localhost:9200 is down"), True),
        (IndexNotFoundError("No index found for suffix 'nosuch'"), False),
    ],
)
async def test_a_question_s_error_is_logged_as_mcp_logs_a_tool_s(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    logged: bool,
) -> None:
    # Regression (rv29): Ask logged only an unexpected failure, MCP every
    # error but a refusal and the deadline: an outage left no server line.
    service = AskService(
        Config,
        ConsoleStore(tmp_path / "s.sqlite3"),
        chain_factory=lambda _c: _FailingChain(error),
    )
    with caplog.at_level(logging.INFO):
        await _drain(service, "q")
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.ERROR and "Console question" in r.getMessage()
    ]
    assert bool(lines) is logged, lines
    if logged:
        assert str(error) in lines[0] and "[store_unreachable]" in lines[0]
    await service.close()


@pytest.mark.parametrize(
    ("error_class", "shown"),
    [("failure", False), ("store_unreachable", True), ("not_indexed", True)],
)
async def test_an_absorbed_failure_shows_text_by_the_raised_ones_rule(
    tmp_path: Path, error_class: str, shown: bool
) -> None:
    # Regression (rv27): under processing.ignore_errors the result event
    # and the history carried a failure's provider text (an ARN) that the
    # error event of a raised one masks.
    from unified_kg_rag.shared.exceptions import INTERNAL_ERROR_MESSAGE

    text = "AccessDenied for arn:aws:iam::111111111111:role/SECRET-ROLE-MARKER"
    output = _output(error=True, error_type="ResponseError", error_class=error_class)
    output.search_results.metadata["error"] = text
    service = AskService(
        Config,
        ConsoleStore(tmp_path / "s.sqlite3"),
        chain_factory=lambda _c: _Chain(output),
    )
    events = await _drain(service, "q")
    (result,) = [e for e in events if e["kind"] == "result"]
    assert result["error_kind"] == error_class
    assert result["error_message"] == (text if shown else INTERNAL_ERROR_MESSAGE)
    # The ids the page shows with it.
    (start,) = [e for e in events if e["kind"] == "start"]
    assert result["query_id"] == start["query_id"]
    (row,) = service.store.queries()
    assert ("SECRET-ROLE-MARKER" in str(row)) is shown
    await service.close()


async def test_a_strategy_s_reason_shows_no_unexpected_failure_s_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from unified_kg_rag.shared.exceptions import INTERNAL_ERROR_MESSAGE

    class Checking(_Chain):
        async def acheck_methodology(self, strategy: Any, suffix: Any) -> None:
            raise RuntimeError("provider said: SECRET-MARKER")

    service = AskService(
        Config,
        ConsoleStore(tmp_path / "s.sqlite3"),
        chain_factory=lambda _c: Checking(_output()),
    )
    with caplog.at_level("INFO"):
        views = await service.strategies()
    assert views and all(
        v["reason"] == INTERNAL_ERROR_MESSAGE and not v["available"] for v in views
    )
    # "The server log has the details": once per request, not per strategy,
    # without the text.
    logged = [r.getMessage() for r in caplog.records if r.levelno >= 40]
    assert len(logged) == 1 and "RuntimeError (len=" in logged[0], logged
    assert "SECRET-MARKER" not in logged[0]
    await service.close()


# The stores' rule (validate_safe_name): a name the console took but the
# config refuses started a job whose subprocess failed loading its config.
_UNSAFE_NAMESPACES = ["../etc", "a/b", "", "x" * 129, "a..b", "Docs", "docs.v2"]
_UNSAFE_NAMESPACES += ["acme*", "a,b"]


@pytest.mark.parametrize("suffix", _UNSAFE_NAMESPACES)
def test_a_question_s_suffix_must_be_a_safe_namespace_name(suffix: str) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AskRequest(query="q", suffix=suffix)
    assert AskRequest(query="q", suffix="tenant_a-1_v2").suffix == "tenant_a-1_v2"


@pytest.mark.parametrize("index_value", ["Docs", "docs.v2", "acme*"])
async def test_an_ingest_s_index_value_must_be_a_safe_namespace_name(
    tmp_path: Path, index_value: str
) -> None:
    from unified_kg_rag.shared import InvalidArgumentError

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    with pytest.raises(InvalidArgumentError, match="lowercase"):
        await _manager(tmp_path).start_ingest(
            corpus=corpus, corpus_ref="workspace:corpus", index_value=index_value
        )


class _Finished:
    """A job's process that printed ``document`` and exited ``code``."""

    def __init__(self, document: str, code: int = 0) -> None:
        self.returncode = code
        self.pid = 4242

        class _Out:
            async def read(self) -> bytes:
                return document.encode()

        self.stdout = _Out()

    async def wait(self) -> int:
        return self.returncode


def _ingest_job(manager: JobManager, tmp_path: Path, document: str, code: int = 0):
    job = jobs_module.Job(
        id="j1",
        kind=jobs_module.JobKind.INGEST,
        namespace="default",
        params={},
        argv=["run-ingestion"],
        env={},
        work_dir=tmp_path,
        log_path=tmp_path / "j1.log",
        pipeline_id="console-p1",
    )
    job.process = _Finished(document, code)  # type: ignore[assignment]
    manager.jobs[job.id] = job
    return job


async def test_an_ingestion_job_exports_no_graph_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rv25 perf 4: after every incremental job the console rebuilt the
    # whole graph from the stores, before saving the job. The Graph page
    # builds it when opened (graph_api.current_graph).
    from unified_kg_rag.application import bootstrap
    from unified_kg_rag.application.storage import visualization_export

    manager = _manager(tmp_path)
    built: list[Any] = []
    monkeypatch.setattr(bootstrap, "build_providers", lambda *a: object())
    monkeypatch.setattr(
        visualization_export,
        "current_visualization_data",
        lambda *a: built.append(a) or (tmp_path / "v.json", {}),
    )
    job = _ingest_job(
        manager, tmp_path, '{"status": "completed", "visualization_data": null}'
    )
    await manager._watch(job)
    assert job.status is JobStatus.SUCCEEDED
    assert job.result == {"status": "completed", "visualization_data": None}
    assert built == []


def test_the_rebuild_warnings_advise_the_reset_the_console_offers() -> None:
    """The warnings that ask for a rebuild (community drift, re-clustering
    due) name it as the console reads it (jobs.RESET_ADVICE)."""
    import inspect

    from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
    from unified_kg_rag.application.ingestion.pipeline_stages import IndexingStage

    for advising in (
        IncrementalIndexer._measure_drift,
        IndexingStage._check_community_counters,
    ):
        source = " ".join(inspect.getsource(advising).split())
        assert jobs_module.RESET_ADVICE in source.replace('" "', ""), advising


def _event(level: str, text: str) -> str:
    return json.dumps({"event": text, "logger": "x", "level": level})


async def test_a_job_records_its_warnings_and_a_stopped_one_no_error(
    tmp_path: Path,
) -> None:
    # rv24 ux 8 and quality 8: a run that logged WARNINGs showed a green
    # "Finished" with nothing else, and a stopped job's error was a banner
    # line ("│ Continue on Error: False │").
    manager = _manager(tmp_path)
    log = [
        "╭──────────────────╮",
        _event("info", "Starting"),
        *(_event("warning", f"warning {i}") for i in range(25)),
        "│ Continue on Error: False │",
        "╰──────────────────╯",
    ]
    job = _ingest_job(manager, tmp_path, '{"visualization_data": "x.json"}')
    job.log_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    await manager._watch(job)
    assert job.record()["warnings"]["count"] == 25
    assert job.record()["warnings"]["lines"] == [f"warning {i}" for i in range(5, 25)]

    assert job.record()["warnings"]["reset_advised"] is False

    # A warning that advises a rebuild, even one past the lines kept, points
    # the page at "Rebuild the namespace" (--reset).
    advised = _ingest_job(manager, tmp_path, '{"visualization_data": null}')
    advised.log_path.write_text(
        "\n".join(
            [
                _event("warning", "Community drift: ... indexing.reset: true."),
                *log,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    await manager._watch(advised)
    assert advised.record()["warnings"]["reset_advised"] is True

    stopped = _ingest_job(manager, tmp_path, '{"status": "interrupted"}', code=143)
    stopped.log_path.write_text("\n".join(log) + "\n", encoding="utf-8")
    await manager._watch(stopped)
    assert stopped.status is JobStatus.STOPPED and stopped.error is None


def test_a_failed_job_s_error_is_its_last_error_event_not_a_banner() -> None:
    banner = "│ Continue on Error: False │"
    assert (
        jobs_module._last_error(
            [_event("error", "Stage failed: boom"), _event("info", "x"), banner]
        )
        == "Stage failed: boom"
    )
    # No error event: the last plain line (a traceback's), never the banner.
    trace = "ValueError: bad value"
    assert jobs_module._last_error([trace, banner, "╰────╯"]) == trace
    assert jobs_module._last_error([_event("info", "x")]) is None
