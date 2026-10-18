# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-pipeline runs that die, stop or resume over one registry (AWS-free).

A run opens its processing attempts right after the diff and closes them
however it ends in the process, so a document that kills a stage before
indexing (here: chunking) is isolated and quarantined while the documents it
died with are indexed, a run stopped by SIGTERM counts nothing, and a run
resumed with a delta the registry no longer matches computes it again.
"""

from __future__ import annotations

import signal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from tests.integration.test_ingestion_stages import _config, _respond
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import (
    DocStatus,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.shared import PipelineExecutionError, PipelineInterrupted

pytestmark = pytest.mark.integration

_POISON = "POISON"


class _Died(BaseException):
    """The process died: no handler of the run gets to run."""


class _Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.source = tmp_path / "corpus"
        self.source.mkdir()
        (self.source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
        (self.source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
        (self.source / "poison.txt").write_text(f"{_POISON} Buyer pays the Depot.")
        self.store = FakeDocStatusStore()
        self.graph = FakeGraphStore()
        self.config = _config()
        self.config.processing.gleaning.enabled = False
        self.config.processing.claim_extraction.enabled = False
        self.vectors = FakeVectorStore(self.config.indexing.opensearch)
        self.extractions = 0
        # Raised by the graph store's entity writes: an error (an outage,
        # the run fails) or a stop request (SIGTERM during indexing).
        self.graph_down: BaseException | None = None
        upsert_entities = self.graph.upsert_entities

        def upsert(*args: Any, **kwargs: Any) -> Any:
            if self.graph_down is not None:
                raise self.graph_down
            return upsert_entities(*args, **kwargs)

        self.graph.upsert_entities = upsert  # type: ignore[method-assign]

    def _respond(self, system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system:
            self.extractions += 1
        return _respond(system, human)

    def pipeline(self) -> DataIngestionPipeline:
        providers = Providers(
            self.config,
            boto_session=MagicMock(),
            llm_factory=ScriptedLLMFactory(self._respond),
            embedding_factory=HashingEmbeddingFactory(),
            doc_status=self.store,
            vector_indexer_factory=lambda: self.vectors,
            graph_indexer_factory=lambda: self.graph,
        )
        pipeline = DataIngestionPipeline(
            self.config,
            PipelineConfig(
                stages_enabled=dict.fromkeys(PipelineStageType, True),
                local_directory=self.tmp_path / "cache",
            ),
            source_directory=self.source,
            providers=providers,
        )
        for stage in pipeline.stages:
            if isinstance(stage, ps.GraphResolutionStage):
                stage.resolver.entity_resolver.use_process_pool = False
                stage.resolver.relationship_resolver.use_process_pool = False
        return pipeline

    def run(
        self,
        pipeline_id: str,
        monkeypatch: pytest.MonkeyPatch,
        chunking_fails_with: type[BaseException] | None = None,
    ) -> PipelineContext | BaseException:
        """One run (a new process). With ``chunking_fails_with``, chunking a
        batch holding the poison document raises it: ``_Died`` is a process
        death (the run's handlers do not run), anything else unwinds."""
        pipeline = self.pipeline()
        original = ps.TextChunkingStage._execute_core

        def chunk(stage: Any, context: PipelineContext) -> Any:
            if chunking_fails_with is not None and any(
                _POISON in document.page_content for document in context.documents
            ):
                if chunking_fails_with is PipelineInterrupted:
                    raise PipelineInterrupted(signal.SIGTERM)
                raise chunking_fails_with()
            return original(stage, context)

        with monkeypatch.context() as patch:
            patch.setattr(ps.TextChunkingStage, "_execute_core", chunk)
            try:
                return pipeline.run(self.source, pipeline_id=pipeline_id)
            except (_Died, PipelineInterrupted) as stopped:
                return stopped
            finally:
                pipeline.close()

    def statuses(self) -> dict[str, DocStatus]:
        return {
            Path(record.file_path or "").name: record.status
            for record in self.store.list_all()
        }


def test_a_document_that_kills_chunking_is_isolated_across_resumed_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Old behaviour: the attempt was counted only by the indexing stage, so a
    # document that killed an earlier stage blocked every run for good.
    harness = _Harness(tmp_path)
    pipeline_id = "ingest-1"
    runs = 0
    while runs < 12:
        runs += 1
        outcome = harness.run(pipeline_id, monkeypatch, chunking_fails_with=_Died)
        if isinstance(outcome, BaseException):
            continue  # the re-run resumes the same pipeline
        assert outcome.status is PipelineStageStatus.COMPLETED
        if all(
            status is not DocStatus.PENDING for status in harness.statuses().values()
        ):
            break
        pipeline_id = f"ingest-{runs + 1}"  # a completed pipeline: a new one
    statuses = harness.statuses()
    assert statuses == {
        "contract.txt": DocStatus.PROCESSED,
        "logistics.txt": DocStatus.PROCESSED,
        "poison.txt": DocStatus.QUARANTINED,
    }
    assert runs <= 7  # 3 deaths, then bisection of 3 suspects
    names = {
        entity.name
        for entity in harness.graph.read_entities(sorted(harness.graph.ids("entities")))
    }
    assert {"Vendor", "Buyer", "Depot"} <= names


def test_a_run_that_fails_with_an_error_closes_its_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An error out of the run (not a failed stage) ends it in the process:
    # it must not count as a death of the documents it processed.
    harness = _Harness(tmp_path)

    def broken(*args: Any, **kwargs: Any) -> None:
        if harness.store.list_all():  # once the attempts are open
            raise RuntimeError("metrics sink broken")

    with monkeypatch.context() as patch:
        patch.setattr(DataIngestionPipeline, "_emit_stage_metrics", broken)
        with pytest.raises(PipelineExecutionError):
            harness.run("ingest-1", monkeypatch)
    records = harness.store.list_all()
    assert records
    assert {r.attempt_batch for r in records} == {None}

    outcome = harness.run("ingest-2", monkeypatch)
    assert isinstance(outcome, PipelineContext)
    assert {r.attempt_count for r in harness.store.list_all()} == {0}


@pytest.mark.parametrize(
    ("change", "poison_status"),
    [
        # Nothing changed, or only the extraction settings: deaths count per
        # content, so the reset still skips the document and keeps its record.
        ("none", DocStatus.QUARANTINED),
        ("settings", DocStatus.QUARANTINED),
        # An edit or a limit above its deaths retries it, as an incremental
        # run would (here it no longer kills chunking).
        ("content", DocStatus.PROCESSED),
        ("limit", DocStatus.PROCESSED),
    ],
)
def test_a_reset_skips_a_quarantined_document_and_keeps_its_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    poison_status: DocStatus,
) -> None:
    # Old behaviour: the reset processed every document, so the quarantined
    # one killed every reset (a reset opens no attempt, so it was never
    # quarantined again), and its record would have been cleared. With new
    # extraction settings, the reset still processed it and died every time.
    harness = _Harness(tmp_path)
    for run in range(1, 13):
        harness.run(f"ingest-{run}", monkeypatch, chunking_fails_with=_Died)
        if harness.statuses().get("poison.txt") is DocStatus.QUARANTINED:
            break
    assert harness.statuses()["poison.txt"] is DocStatus.QUARANTINED
    harness.config.indexing.reset = True
    if change == "settings":
        harness.config.processing.gleaning.enabled = True
    elif change == "content":
        (harness.source / "poison.txt").write_text("Buyer pays the Depot.")
    elif change == "limit":
        harness.config.indexing.max_document_attempts = 10
    # A retried document that still killed chunking would kill the reset.
    dies = None if poison_status is DocStatus.PROCESSED else _Died

    for run in range(2):
        reset = harness.run(f"reset-{run}", monkeypatch, chunking_fails_with=dies)
        assert isinstance(reset, PipelineContext)
        assert reset.status is PipelineStageStatus.COMPLETED
        assert harness.statuses() == {
            "contract.txt": DocStatus.PROCESSED,
            "logistics.txt": DocStatus.PROCESSED,
            "poison.txt": poison_status,
        }

    # Raising the limit retries it on an incremental run too.
    harness.config.indexing.reset = False
    harness.config.indexing.max_document_attempts = 10
    retried = harness.run("ingest-retry", monkeypatch)
    assert isinstance(retried, PipelineContext)
    assert harness.statuses()["poison.txt"] is DocStatus.PROCESSED


@pytest.mark.parametrize("stage", ["chunking", "indexing"])
def test_runs_stopped_by_sigterm_count_no_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    # Old behaviour: a run stopped during indexing (a task timeout, StopTask)
    # left its write-ahead records, counted like a crash.
    harness = _Harness(tmp_path)
    if stage == "indexing":
        harness.graph_down = PipelineInterrupted(signal.SIGTERM)
    for _ in range(4):
        outcome = harness.run(
            "ingest-1",
            monkeypatch,
            chunking_fails_with=(PipelineInterrupted if stage == "chunking" else None),
        )
        assert isinstance(outcome, PipelineInterrupted)
        records = harness.store.list_all()
        assert records
        assert {r.attempt_count for r in records} == {0}
    harness.graph_down = None

    outcome = harness.run("ingest-1", monkeypatch)

    assert isinstance(outcome, PipelineContext)
    assert set(harness.statuses().values()) == {DocStatus.PROCESSED}


def test_a_resume_after_another_run_committed_computes_the_delta_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    harness = _Harness(tmp_path)
    assert isinstance(
        harness.run("ingest-1", monkeypatch, chunking_fails_with=_Died), _Died
    )
    (harness.source / "poison.txt").unlink()
    other = harness.run("ingest-2", monkeypatch)
    assert isinstance(other, PipelineContext)
    (harness.source / "poison.txt").write_text(f"{_POISON} Buyer pays the Depot.")
    harness.extractions = 0

    with caplog.at_level("WARNING"):
        resumed = harness.run("ingest-1", monkeypatch)

    assert isinstance(resumed, PipelineContext)
    assert resumed.status is PipelineStageStatus.COMPLETED
    assert "resuming from 'document_loading'" in caplog.text
    # Only the poison document is new to the registry: the others, which
    # the other run committed, are not extracted again.
    assert harness.statuses() == {
        "contract.txt": DocStatus.PROCESSED,
        "logistics.txt": DocStatus.PROCESSED,
        "poison.txt": DocStatus.PROCESSED,
    }
    assert harness.extractions == 1


def test_runs_whose_indexing_fails_with_an_error_count_no_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Old behaviour: each run that failed on the outage left its write-ahead
    # records, counted as attempts, and the first healthy run quarantined
    # (and removed) the documents.
    harness = _Harness(tmp_path)
    (harness.source / "poison.txt").unlink()
    first = harness.run("ingest-0", monkeypatch)
    assert isinstance(first, PipelineContext)
    (harness.source / "contract.txt").write_text("Vendor supplies Depot and Buyer.")
    harness.graph_down = RuntimeError("graph store unavailable")
    for run in range(1, 4):
        failed = harness.run(f"ingest-{run}", monkeypatch)
        assert isinstance(failed, PipelineContext)
        assert failed.status is PipelineStageStatus.FAILED
    harness.graph_down = None

    healthy = harness.run("ingest-4", monkeypatch)

    assert isinstance(healthy, PipelineContext)
    assert healthy.status is PipelineStageStatus.COMPLETED
    assert set(harness.statuses().values()) == {DocStatus.PROCESSED}


def test_a_reset_that_could_not_drop_the_previous_graph_is_not_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Old behaviour: a rebuild whose graph writes partly failed (under
    # indexing.max_failure_rate) kept the previous graph beside the new one,
    # yet recorded the corpus and cleared the replaced documents' records:
    # nothing would ever remove their edges.
    harness = _Harness(tmp_path)
    (harness.source / "poison.txt").write_text("Buyer pays the Depot.")
    assert isinstance(harness.run("ingest-1", monkeypatch), PipelineContext)
    assert {"Buyer", "Depot"} in _edges(harness)
    (harness.source / "poison.txt").unlink()
    harness.config.indexing.reset = True
    harness.config.indexing.max_failure_rate = 0.9
    index_relationships = harness.graph.index_relationships

    def one_edge_fails(relationships: Any, *args: Any, **kwargs: Any) -> Any:
        stats = index_relationships(relationships, *args, **kwargs)
        if len(relationships) > 1:
            stats.successful_items -= 1
            stats.add_error("injected", ids=[relationships[-1].id])
        return stats

    monkeypatch.setattr(harness.graph, "index_relationships", one_edge_fails)
    reset = harness.run("ingest-2", monkeypatch)
    assert isinstance(reset, PipelineContext)
    assert reset.status is PipelineStageStatus.FAILED
    assert "neptune_sweep" in str(
        [r.error_message for r in reset.stage_results if r.stage_name == "indexing"]
    )
    # The replaced document's record is kept for the next run to remove.
    assert "poison.txt" in harness.statuses()

    monkeypatch.undo()
    harness.config.indexing.reset = False
    harness.run("ingest-3", monkeypatch)

    assert {"Buyer", "Depot"} not in _edges(harness)
    assert harness.statuses() == {
        "contract.txt": DocStatus.PROCESSED,
        "logistics.txt": DocStatus.PROCESSED,
    }


def _edges(harness: _Harness) -> list[set[str]]:
    graph = harness.graph
    return [
        {edge.source_name, edge.target_name}
        for edge in graph.read_relationships(sorted(graph.ids("relationships")))
    ]
