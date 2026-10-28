# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-runs, resumes, force rebuilds and phased runs of ``run-ingestion``.

The CLI runs in-process over a tiny synthetic corpus with a scripted chat
model and the in-memory stores and doc-status registry (AWS-free), the way an
operator re-runs it: same ``--pipeline-id``, no terminal on stdin.
"""

from __future__ import annotations

import io
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from tests.integration import test_ingestion_stages as stages_test
from unified_kg_rag.application import bootstrap
from unified_kg_rag.application.cli import run_ingestion_pipeline as cli
from unified_kg_rag.application.ingestion import pipeline as pipeline_module
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.domain.models import Config, DocStatus
from unified_kg_rag.shared.metrics import MetricsSink

pytestmark = pytest.mark.integration

# A run split into --enabled-stages windows, each in its own process (the
# Step Functions phases of earlier versions).
_PHASES = [
    ["document_parsing", "document_loading", "text_chunking", "translation"],
    [
        "graph_extraction",
        "gleaning",
        "graph_resolution",
        "claim_extraction",
        "claim_resolution",
    ],
    ["graph_analysis", "community_detection"],
    ["indexing"],
]


class _RecordingSink(MetricsSink):
    def __init__(self) -> None:
        self.emitted: list[dict[str, Any]] = []

    def emit(self, namespace, metrics, dimensions=None) -> None:  # type: ignore[no-untyped-def]
        self.emitted.append(dict(metrics))


class _Harness:
    """Runs the CLI against one set of fake stores across invocations."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.tmp_path = tmp_path
        self.source = tmp_path / "corpus"
        self.source.mkdir()
        (self.source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
        (self.source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
        self.config = stages_test._config()
        self.config.processing.claim_extraction.enabled = False
        self.config.aws.neptune.endpoint = "neptune.example.internal"
        self.config.aws.opensearch.endpoint = "opensearch.example.internal"
        self.registry = FakeDocStatusStore()
        self.graph = FakeGraphStore()
        self.vectors = FakeVectorStore(self.config.indexing.opensearch)
        self.llm_calls = 0
        # With the doc-status registry (incremental indexing) on.
        self.incremental = True
        self.sink = _RecordingSink()

        def respond(system: str, human: str) -> str:
            self.llm_calls += 1
            return stages_test._respond(system, human)

        def build_providers(config: Config, boto_session=None, **overrides):  # type: ignore[no-untyped-def]
            registry = {"doc_status": self.registry} if self.incremental else {}
            return bootstrap.build_providers(
                config,
                MagicMock(),
                llm_factory=ScriptedLLMFactory(respond),
                embedding_factory=HashingEmbeddingFactory(),
                graph_indexer_factory=lambda: self.graph,
                vector_indexer_factory=lambda: self.vectors,
                **registry,
            )

        monkeypatch.setattr(pipeline_module, "build_providers", build_providers)
        monkeypatch.setattr(cli, "load_cli_config", lambda _args: self.config)
        monkeypatch.setattr(cli, "setup_logging", lambda config, stream=None: None)
        monkeypatch.setattr(
            cli.IngestionPipelineRunner,
            "_build_metrics_sink",
            lambda runner: self.sink,
        )
        # No process pool in-process.
        original_init = ps.GraphResolutionStage.__init__

        def init(stage, *args, **kwargs):  # type: ignore[no-untyped-def]
            original_init(stage, *args, **kwargs)
            stage.resolver.entity_resolver.use_process_pool = False
            stage.resolver.relationship_resolver.use_process_pool = False

        monkeypatch.setattr(ps.GraphResolutionStage, "__init__", init)
        # No terminal: a confirmation prompt must not be asked.
        monkeypatch.setattr(sys, "stdin", io.StringIO(""))

    def run(self, *flags: str, cache: str = "cache") -> int:
        argv = [
            "run-ingestion",
            "--source-directory",
            str(self.source),
            "--cache-directory",
            str(self.tmp_path / cache),
            *flags,
        ]
        sys.argv = argv
        try:
            cli.main()
        except SystemExit as e:
            return e.code if isinstance(e.code, int) else 1
        return 0

    def metadata(self, pipeline_id: str, cache: str = "cache") -> dict[str, Any]:
        path = self.tmp_path / cache / pipeline_id / "pipeline_metadata.json"
        return json.loads(path.read_text())


@pytest.fixture
def harness(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Harness:
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    return _Harness(tmp_path, monkeypatch)


def test_rerun_with_the_same_pipeline_id_resumes_without_a_prompt(
    harness: _Harness,
) -> None:
    # Model calls show which stages re-run, so none is answered from the
    # response cache (see test_a_rerun_stage_answers_from_the_response_cache).
    harness.config.cache.llm_responses.enabled = False
    assert harness.run("--pipeline-id", "p1") == 0
    calls = harness.llm_calls

    # Same id, no terminal: the integrity check uses the run's own cache keys,
    # finds every stage cached and asks nothing; only indexing re-runs.
    assert harness.run("--pipeline-id", "p1") == 0
    assert harness.llm_calls == calls

    # An explicit, upper-case resume stage re-runs from there.
    assert harness.run("--pipeline-id", "p1", "--resume-from-stage", "GLEANING") == 0
    assert harness.llm_calls > calls


def test_a_rerun_stage_answers_from_the_response_cache(harness: _Harness) -> None:
    # Non-incremental, so the force rebuild re-extracts unchanged documents.
    harness.incremental = False
    assert harness.run("--pipeline-id", "p1") == 0
    calls = harness.llm_calls
    # Gleaning and the community reports run again on the same prompts.
    assert harness.run("--pipeline-id", "p1", "--resume-from-stage", "gleaning") == 0
    assert harness.llm_calls == calls
    # A force rebuild asks the model again.
    assert harness.run("--pipeline-id", "p1", "--force-rebuild") == 0
    assert harness.llm_calls > calls


def test_a_failed_integrity_check_without_a_terminal_needs_yes(
    harness: _Harness,
) -> None:
    assert harness.run("--pipeline-id", "p1") == 0
    # Editing the corpus changes every cache key: the completed stages are no
    # longer backed by the cache, and the operator has to confirm.
    (harness.source / "contract.txt").write_text("Vendor sells gadgets to Buyer.")
    assert harness.run("--pipeline-id", "p1") == 1
    assert harness.run("--pipeline-id", "p1", "--yes") == 0


def test_force_rebuild_rewrites_the_stage_cache(harness: _Harness) -> None:
    # A full (non-incremental) run, so the rebuild re-extracts every document
    # instead of skipping them as unchanged in the registry.
    harness.incremental = False
    assert harness.run("--pipeline-id", "p1") == 0
    index = harness.tmp_path / "cache" / "p1"
    before = {
        p.name: p.stat().st_mtime_ns for p in index.rglob("*") if "entities" in p.name
    }
    assert before
    calls = harness.llm_calls

    assert harness.run("--pipeline-id", "p1", "--force-rebuild") == 0
    assert harness.llm_calls > calls
    after = {
        p.name: p.stat().st_mtime_ns for p in index.rglob("*") if "entities" in p.name
    }
    assert all(after[name] > mtime for name, mtime in before.items())

    # The next plain run resumes from what the rebuild wrote.
    calls = harness.llm_calls
    assert harness.run("--pipeline-id", "p1") == 0
    assert harness.llm_calls == calls


@pytest.mark.parametrize(
    "stage_class", [ps.GraphExtractionStage, ps.CommunityDetectionStage]
)
def test_a_crashed_artifact_stage_never_commits_documents_processed(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, stage_class: type
) -> None:
    crash = {"on": True}
    original = stage_class.execute

    def execute(stage, context):  # type: ignore[no-untyped-def]
        if crash["on"]:
            raise RuntimeError("throttled beyond retries")
        return original(stage, context)

    monkeypatch.setattr(stage_class, "execute", execute)
    assert harness.run("--continue-on-error") == 1
    records = harness.registry.list_all()
    assert records and {r.status for r in records} == {DocStatus.FAILED}

    # The next run re-processes them instead of skipping them as unchanged.
    crash["on"] = False
    assert harness.run() == 0
    assert {r.status for r in harness.registry.list_all()} == {DocStatus.PROCESSED}
    assert harness.graph.ids("entities")
    assert harness.graph.ids("communities")


@pytest.mark.parametrize("stage_class", [ps.DocumentParsingStage, ps.TextChunkingStage])
def test_a_crashed_corpus_stage_skips_indexing(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch, stage_class: type
) -> None:
    # With no documents (or text units) a commit would read every stored
    # document as deleted, or record new ones processed with no content.
    assert harness.run() == 0
    entities = harness.graph.ids("entities")
    records = {(r.doc_id, r.status) for r in harness.registry.list_all()}
    assert entities and records

    def execute(stage, context):  # type: ignore[no-untyped-def]
        raise RuntimeError("parser crashed")

    monkeypatch.setattr(stage_class, "execute", execute)
    (harness.source / "new.txt").write_text("Buyer pays Depot.")
    assert harness.run("--continue-on-error") == 1
    assert harness.graph.ids("entities") == entities
    stored = {r.doc_id: r for r in harness.registry.list_all()}
    assert {(doc_id, stored[doc_id].status) for doc_id, _ in records} == records
    # Past document_loading, the new document has the record of the attempt
    # the run opened (and closed), with nothing indexed.
    new = [r for doc_id, r in stored.items() if (doc_id, r.status) not in records]
    if stage_class is not ps.DocumentParsingStage:
        assert [(r.status, r.text_unit_ids, r.attempt_batch) for r in new] == [
            (DocStatus.PENDING, [], None)
        ]


def _run_phases(harness: _Harness, pipeline_id: str, attempt: int) -> list[int]:
    codes = []
    for number, stages in enumerate(_PHASES):
        # Every phase runs in a fresh task: copy the shared checkpoint (S3 in
        # the deployment) into a cache directory of its own.
        cache = f"cache-{attempt}-{number}"
        shared = harness.tmp_path / "shared" / pipeline_id
        if shared.exists():
            shutil.copytree(shared, harness.tmp_path / cache / pipeline_id)
        code = harness.run(
            "--pipeline-id",
            pipeline_id,
            "--enabled-stages",
            ",".join(stages),
            cache=cache,
        )
        codes.append(code)
        shutil.rmtree(shared, ignore_errors=True)
        shutil.copytree(harness.tmp_path / cache / pipeline_id, shared)
        if code:
            break
    return codes


def test_a_failed_phased_run_resumes_at_the_failed_stage(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    fail = {"on": True}
    original = ps.IndexingStage.execute

    def execute(stage, context):  # type: ignore[no-untyped-def]
        if fail["on"]:
            raise RuntimeError("graph store unreachable")
        return original(stage, context)

    monkeypatch.setattr(ps.IndexingStage, "execute", execute)
    assert _run_phases(harness, "run-001", attempt=1) == [0, 0, 0, 1]
    calls = harness.llm_calls
    assert calls > 0
    emitted_first = list(harness.sink.emitted)

    # Re-running the execution with the same pipeline id: the earlier phases
    # find their stages completed and skip them (no LLM call); the index
    # phase runs the failed stage again.
    fail["on"] = False
    harness.sink.emitted.clear()
    assert _run_phases(harness, "run-001", attempt=2) == [0, 0, 0, 0]
    assert harness.llm_calls == calls
    assert {r.status for r in harness.registry.list_all()} == {DocStatus.PROCESSED}
    statuses = {
        r["stage_name"]: r["status"]
        for r in harness.metadata("run-001", cache="cache-2-3")["stage_results"]
    }
    assert set(statuses.values()) == {"completed"}

    # Each stage's counts are emitted when it finishes, so summed over the
    # first attempt's windows every count is reported once, and the retry
    # reports only what it ran.
    entity_counts = [
        m["total_entities_extracted"]
        for m in emitted_first
        if "total_entities_extracted" in m
    ]
    assert len(entity_counts) == 1
    assert len([m for m in harness.sink.emitted if "total_items_indexed" in m]) == 1
    assert all("total_entities_extracted" not in m for m in harness.sink.emitted)


def test_a_delta_whose_documents_name_no_entity_still_commits(
    harness: _Harness,
) -> None:
    # The only changed document no longer names an entity and the other one
    # is deleted: the run commits both instead of failing graph extraction
    # ("produced 0 outputs"), which would leave the deletion unapplied and
    # fail every later run the same way.
    assert harness.run() == 0
    assert harness.graph.ids("entities")
    (harness.source / "contract.txt").write_text("Nothing notable is written here.")
    (harness.source / "logistics.txt").unlink()

    assert harness.run() == 0
    records = harness.registry.list_all()
    assert [Path(r.file_path).name for r in records] == ["contract.txt"]
    assert {r.status for r in records} == {DocStatus.PROCESSED}
    assert not harness.graph.ids("entities")
    # The next run finds nothing to do.
    assert harness.run() == 0


@pytest.mark.parametrize(
    ("block", "stage"),
    [("gleaning", "gleaning"), ("claim_extraction", "claim_extraction")],
)
def test_a_stage_switched_on_for_an_existing_pipeline_id_runs(
    harness: _Harness, block: str, stage: str
) -> None:
    # The stage was off when the pipeline first ran, so it has no recorded
    # result; the re-run must start there rather than after the last
    # recorded stage.
    getattr(harness.config.processing, block).enabled = False
    assert harness.run("--pipeline-id", "p1") == 0
    assert stage not in {
        r["stage_name"] for r in harness.metadata("p1")["stage_results"]
    }

    getattr(harness.config.processing, block).enabled = True
    calls = harness.llm_calls
    # The later stages' cached outputs no longer match: confirm the resume.
    assert harness.run("--pipeline-id", "p1", "--yes") == 0
    statuses = {
        r["stage_name"]: r["status"] for r in harness.metadata("p1")["stage_results"]
    }
    assert statuses[stage] == "completed"
    assert harness.llm_calls > calls


def test_a_phase_after_a_failed_upstream_stage_does_not_run(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # graph_extraction failed; an indexing-only window (the next Step
    # Functions phase) must not index the empty upstream output.
    from unified_kg_rag.domain.models import PipelineConfig, PipelineStageType
    from unified_kg_rag.shared import PipelineExecutionError

    def execute(stage, context):  # type: ignore[no-untyped-def]
        raise RuntimeError("throttled beyond retries")

    monkeypatch.setattr(ps.GraphExtractionStage, "execute", execute)
    assert harness.run("--pipeline-id", "p1") == 1
    pipeline = pipeline_module.DataIngestionPipeline(
        harness.config,
        PipelineConfig(
            stages_enabled={
                stage: stage is PipelineStageType.INDEXING
                for stage in PipelineStageType
            },
            local_directory=harness.tmp_path / "cache",
        ),
        source_directory=harness.source,
    )
    with pytest.raises(PipelineExecutionError, match="upstream of this run"):
        pipeline.run(harness.source, pipeline_id="p1")
    assert not harness.graph.ids("entities")


def test_rerunning_a_stage_drops_the_recorded_results_after_it(
    harness: _Harness,
) -> None:
    harness.config.cache.llm_responses.enabled = False
    # Re-running gleaning alone makes the recorded later stages stale: they
    # leave the history, and the next run runs them again on its output.
    assert harness.run("--pipeline-id", "p1") == 0
    assert (
        harness.run(
            "--pipeline-id",
            "p1",
            "--resume-from-stage",
            "gleaning",
            "--enabled-stages",
            "gleaning",
        )
        == 0
    )
    names = [r["stage_name"] for r in harness.metadata("p1")["stage_results"]]
    assert names[-1] == "gleaning"

    calls = harness.llm_calls
    assert harness.run("--pipeline-id", "p1") == 0
    statuses = {
        r["stage_name"]: r["status"] for r in harness.metadata("p1")["stage_results"]
    }
    assert statuses["graph_resolution"] == statuses["indexing"] == "completed"
    assert harness.llm_calls > calls  # community reports regenerated


def test_a_stage_that_crashed_then_succeeds_commits_its_documents(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The crash recorded every text unit failed for graph_resolution, and the
    # resumed run restores that record; the stage's successful re-run must
    # replace it, or the documents stay FAILED.
    crash = {"on": True}
    original = ps.GraphResolutionStage.execute

    def execute(stage, context):  # type: ignore[no-untyped-def]
        if crash["on"]:
            raise RuntimeError("worker crashed")
        return original(stage, context)

    monkeypatch.setattr(ps.GraphResolutionStage, "execute", execute)
    assert harness.run("--pipeline-id", "p1", "--continue-on-error") == 1
    assert {r.status for r in harness.registry.list_all()} == {DocStatus.FAILED}

    crash["on"] = False
    assert harness.run("--pipeline-id", "p1", "--yes") == 0
    assert {r.status for r in harness.registry.list_all()} == {DocStatus.PROCESSED}


def test_a_failed_run_says_which_stage_failed(
    harness: _Harness,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    def execute(stage, context):  # type: ignore[no-untyped-def]
        raise RuntimeError("throttled beyond retries")

    monkeypatch.setattr(ps.GraphExtractionStage, "execute", execute)
    caplog.set_level(logging.INFO)
    assert harness.run() == 1
    assert "Error: Pipeline failed at stage(s): graph_extraction" in (
        capsys.readouterr().out
    )
    assert not any(
        "Pipeline execution completed successfully" in r.getMessage()
        for r in caplog.records
    )
    assert any(
        r.levelname == "WARNING" and "finished with failed stages" in r.getMessage()
        for r in caplog.records
    )


def _answer_badly_once(
    monkeypatch: pytest.MonkeyPatch, task: str, bad: str
) -> list[str]:
    """Answer ``task``'s first call for the logistics text with ``bad``.

    Returns the source texts of every ``task`` call, in order.
    """
    calls: list[str] = []
    respond = stages_test._respond

    def answer(system: str, human: str) -> str:
        if task not in system:
            return respond(system, human)
        text = stages_test._source_text(human)
        calls.append(text)
        if "Depot" in text and calls.count(text) == 1:
            return bad
        if task == "claim extraction specialist":
            return (
                "<claims><claim><subject>Depot</subject><object>Vendor</object>"
                "<claim_type>OBLIGATION</claim_type><claim_status>TRUE"
                "</claim_status><description>Depot stores for Vendor"
                f"</description><source_text>{text}</source_text></claim></claims>"
                if "Depot" in text
                else "<claims></claims>"
            )
        return respond(system, human)

    monkeypatch.setattr(stages_test, "_respond", answer)
    return calls


def test_a_rerun_asks_again_for_the_claims_a_bad_response_failed(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The response cache stores a response that does not parse. A rerun with
    # the same pipeline id must not replay it: the unit's claims would be
    # given up (PROCESSED without them) without asking the model.
    harness.config.processing.claim_extraction.enabled = True
    calls = _answer_badly_once(
        monkeypatch, "claim extraction specialist", "<claims><claim><subj"
    )
    assert harness.run("--pipeline-id", "p1") == 0
    status = {Path(r.file_path).name: r.status for r in harness.registry.list_all()}
    assert status["logistics.txt"] == DocStatus.FAILED
    assert not harness.vectors.ids("claims")
    asked = len(calls)

    assert harness.run("--pipeline-id", "p1") == 0
    assert [t for t in calls[asked:] if "Depot" in t]
    assert {r.status for r in harness.registry.list_all()} == {DocStatus.PROCESSED}
    assert harness.vectors.ids("claims")


def test_a_rerun_asks_again_for_the_graph_a_bad_response_failed(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Replaying the cached response would count the document's failure once
    # more (toward indexing.max_document_failures) without a model call.
    calls = _answer_badly_once(
        monkeypatch, "knowledge graph extraction expert", "garbage"
    )
    assert harness.run("--pipeline-id", "p1") == 0
    (failed,) = [r for r in harness.registry.list_all() if r.status is DocStatus.FAILED]
    assert failed.failure_count == 1
    asked = len(calls)

    assert harness.run("--pipeline-id", "p1") == 0
    assert [t for t in calls[asked:] if "Depot" in t]
    assert {r.status for r in harness.registry.list_all()} == {DocStatus.PROCESSED}
