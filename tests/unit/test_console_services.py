# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console's job manager and question service on their own: one job per
namespace under concurrent requests, jobs a crashed console left running,
pipeline ids, retired chains closed once idle, blocked answers recorded,
evaluation progress from the answer batch, what a namespace holds and an
ingestion that would narrow it."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

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
from unified_kg_rag.application.retrieval.rag_chain import ProcessedQuery, RAGOutput
from unified_kg_rag.domain.models import (
    Config,
    Methodology,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from unified_kg_rag.shared.exceptions import StoreConnectionError

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
    error = "Confirmation required (Continue despite integrity issues?); pass --confirm-integrity"
    assert confirmation_needed(error) == "integrity"
    assert (
        confirmation_needed(
            "Confirmation required (x); pass --confirm-deletions (or UKR_CONFIRM_DELETIONS=1)"
        )
        == "deletions"
    )
    assert confirmation_needed("Pipeline failed at stage(s): indexing") is None
    assert confirmation_needed(None) is None


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
    # No record, or one from before methodology profiles: nothing recorded.
    for suffix in (None, "old"):
        held = await service.held_methodologies(suffix)
        assert held["recorded"] is False and held["held"] == []
    assert (await service.held_methodologies(None))["namespace"] == "default"
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


class _FailingChain(_Chain):
    def __init__(self, error: Exception) -> None:
        super().__init__(_output())
        self.error = error

    async def ainvoke(self, *_: Any, **__: Any) -> RAGOutput:
        raise self.error


async def test_a_missing_index_is_reported_as_such(tmp_path: Path) -> None:
    from unified_kg_rag.shared.exceptions import IndexNotFoundError

    service = AskService(
        Config,
        ConsoleStore(tmp_path / "s.sqlite3"),
        chain_factory=lambda _c: _FailingChain(IndexNotFoundError("no index 'x'")),
    )
    events = await _drain(service, "q")
    assert events[-1]["kind"] == "error"
    assert events[-1]["error_kind"] == "index_not_found"
    await service.close()


@pytest.mark.parametrize("suffix", ["../etc", "a/b", "", "x" * 129, "-lead", "a..b"])
def test_a_question_s_suffix_must_be_a_safe_namespace_name(suffix: str) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AskRequest(query="q", suffix=suffix)
    assert AskRequest(query="q", suffix="tenant_a-1.v2").suffix == "tenant_a-1.v2"
