# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console's job manager and question service on their own: one job per
namespace under concurrent requests, jobs a crashed console left running,
pipeline ids, retired chains closed once idle, blocked answers recorded."""

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
    _pipeline_id,
    confirmation_needed,
)
from unified_kg_rag.application.cli.console.store import ConsoleStore
from unified_kg_rag.application.retrieval.rag_chain import ProcessedQuery, RAGOutput
from unified_kg_rag.domain.models import Config, SearchQuery, SearchResult

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
