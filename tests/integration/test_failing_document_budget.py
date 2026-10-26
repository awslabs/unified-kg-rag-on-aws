# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A document that always fails extraction, alone in an incremental delta.

The extraction stage fails on all of its units, which fails the stage and
stops the run before the indexing stage records the document FAILED. Its
consecutive-failure count must still grow when the error is the document's
own (an HTTP 400), so that after ``indexing.max_document_failures`` runs it
is skipped and the run applies the deletions of its delta; any other error
(a model outage, a credential, quota or configuration error, a bug) must not
count, so it never excludes a document.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from botocore.exceptions import NoCredentialsError, TokenRetrievalError

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_ingestion_stages import _source_text
from unified_kg_rag.adapters.local import models
from unified_kg_rag.application.cli.run_ingestion_pipeline import (
    IngestionPipelineRunner,
)
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.models import DocStatus, PipelineContext, PipelineStageStatus

pytestmark = pytest.mark.integration

_FAILING = "Zorbix and Vendor trade widgets."
_FILES = {
    "d0.txt": "Vendor and Buyer trade goods.",
    "d1.txt": "Depot and Carrier trade parts.",
}


def _runs(
    tmp: Path,
    status: int | Exception,
    pipeline_id: str | None = None,
    message: str = "rejected",
    contexts: list[PipelineContext] | None = None,
    reset: bool = False,
    beside: bool = False,
) -> list[tuple]:
    """Index d0, d1; add fz.txt, whose extraction the model answers with
    ``status`` (and ``message``), or fails with an exception ``status``, and
    delete d1; run four times (``indexing.reset``: rebuilding the whole
    corpus each time; ``beside``: a new document d2 added with fz, so the
    stage completes on d2 and the commit records fz). Each run's (status, fz's
    record (status, failure count), d1 still recorded); the last run's
    graph_extraction error."""
    config = e2e._config(tmp / "stores", tmp / "viz")
    config.indexing.max_document_failures = 2
    config.indexing.reset = reset
    config.processing.document_parsing.source_scope = "budget-corpus"
    server = FakeOllama(e2e._respond, "x", e2e.DIM)

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            messages = json.loads(request.content)["messages"]
            human = "".join(m["content"] for m in messages if m["role"] == "user")
            if "<input_text>" in human and "Zorbix" in _source_text(human):
                if isinstance(status, Exception):
                    raise status
                return httpx.Response(status, json={"error": message})
        return server(request)

    providers = e2e._providers(
        config, transport, e2e.FakeCrossEncoder("r")  # type: ignore[arg-type]
    )
    source = tmp / "corpus"
    source.mkdir(parents=True)
    for name, text in _FILES.items():
        (source / name).write_text(text)
    e2e._ingest(config, providers, source, tmp / "seed")
    (source / "fz.txt").write_text(_FAILING)
    if beside:
        (source / "d2.txt").write_text("Carrier and Buyer trade parts.")
    (source / "d1.txt").unlink()
    seen = []
    for run in range(4):
        cache = tmp / (pipeline_id or f"run{run}")
        if pipeline_id is None and cache.exists():
            shutil.rmtree(cache)
        context = e2e._ingest(config, providers, source, cache)
        if contexts is not None:
            contexts.append(context)
        records = {r.file_path: r for r in providers.build_doc_status().list_all()}
        fz = records["fz.txt"]
        seen.append(
            (
                context.status.value,
                (fz.status.value, fz.failure_count),
                "d1.txt" in records,
            )
        )
        errors = [
            r.error_message
            for r in context.stage_results
            if r.stage_name == "graph_extraction"
            and r.status == PipelineStageStatus.FAILED
        ]
    return [*seen, errors]


@pytest.mark.parametrize("pipeline_id", [None, "P1"])
def test_a_document_failing_for_its_own_error_is_excluded_and_deletions_apply(
    tmp_path: Path, pipeline_id: str | None
) -> None:
    contexts: list[PipelineContext] = []
    *seen, errors = _runs(tmp_path, 400, pipeline_id, contexts=contexts)
    failed = PipelineStageStatus.FAILED.value
    done = PipelineStageStatus.COMPLETED.value
    assert seen[:2] == [
        (failed, (DocStatus.FAILED.value, 1), True),
        (failed, (DocStatus.FAILED.value, 2), True),
    ], seen
    # Skipped from the third run on: the run completes and deletes d1.
    assert seen[2:] == [(done, (DocStatus.FAILED.value, 2), False)] * 2, seen
    assert errors == []
    # Each run lists the documents it recorded FAILED, not the ones a run
    # before it did (a resumed pipeline id restores that run's context).
    assert [len(c.recorded_failed_doc_ids) for c in contexts] == [1, 1, 0, 0]


def test_a_factory_without_an_item_classifier_counts_http_rejections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A custom factory without is_item_error: the HTTP 400 its client raises
    # still counts (looks_like_item_error), so the document is excluded and
    # the delta's deletion applies instead of failing every run.
    monkeypatch.setattr(models._LocalLanguageModelFactory, "is_item_error", None)
    *seen, _ = _runs(tmp_path, 400)
    assert [record for _, record, _ in seen] == [
        (DocStatus.FAILED.value, 1),
        *[(DocStatus.FAILED.value, 2)] * 3,
    ], seen
    assert seen[-1][2] is False, seen


@pytest.mark.parametrize("status", [503, 403, 404])
def test_a_model_outage_or_misconfiguration_counts_no_failure(
    tmp_path: Path, status: int
) -> None:
    # 503: the model is down; 403 (access denied) and 404 (no such model):
    # it is misconfigured. Neither is the document's fault: it stays
    # PENDING, uncounted, run after run.
    *seen, errors = _runs(tmp_path, status)
    assert {(status, record[1]) for status, record, _ in seen} == {
        (PipelineStageStatus.FAILED.value, 0)
    }, seen
    # The stage's error names the file, not only a text-unit id.
    assert errors and "fz.txt" in errors[0], errors


@pytest.mark.parametrize(
    "error",
    [
        # Expired or missing credentials: three runs and more count none.
        NoCredentialsError(),
        TokenRetrievalError(provider="sso", error_msg="Token has expired"),
        # A bug, or an error no classifier names.
        TypeError("unexpected keyword argument"),
        RuntimeError("model down"),
    ],
    ids=type,
)
def test_an_error_no_classifier_names_counts_no_failure(
    tmp_path: Path, error: Exception
) -> None:
    # Counted as the document's own, these excluded it for good after
    # max_document_failures runs, and with it a whole delta in an outage.
    *seen, errors = _runs(tmp_path, error)
    assert {(status, record[1]) for status, record, _ in seen} == {
        (PipelineStageStatus.FAILED.value, 0)
    }, seen
    assert errors and "fz.txt" in errors[0], errors


@pytest.mark.parametrize(
    "message",
    [
        "invalid model name",
        "model 'x' does not support tools",
        "unknown parameter: top_k",
        "max_tokens is too large",
    ],
)
def test_a_rejected_setting_counts_no_failure(tmp_path: Path, message: str) -> None:
    # An HTTP 400 for the model or a request setting fails every document
    # alike: the configuration is at fault, not the document.
    *seen, _ = _runs(tmp_path, 400, message=message)
    assert {record[1] for _, record, _ in seen} == {0}, seen


def test_a_whole_delta_outage_excludes_nothing_once_fixed(tmp_path: Path) -> None:
    # Credentials expired for three runs (indexing.max_document_failures):
    # every unit of every document fails. Counted, the documents were
    # excluded, and the fourth run, credentials renewed, completed with
    # nothing indexed.
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.processing.document_parsing.source_scope = "outage-corpus"
    down = {"on": True}

    def respond(system: str, human: str) -> str:
        if down["on"] and "extraction expert" in system:
            raise NoCredentialsError()
        return e2e._respond(system, human)

    providers = e2e._providers(
        config, FakeOllama(respond, "x", e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    for name, text in _FILES.items():
        (source / name).write_text(text)
    for run in range(config.indexing.max_document_failures):
        e2e._ingest(config, providers, source, tmp_path / f"down{run}")
        counts = {r.failure_count for r in providers.build_doc_status().list_all()}
        assert counts == {0}, (run, counts)
    down["on"] = False
    context = e2e._ingest(config, providers, source, tmp_path / "fixed")
    assert context.status == PipelineStageStatus.COMPLETED
    records = providers.build_doc_status().list_all()
    assert {r.status for r in records} == {DocStatus.PROCESSED}, records
    entities = e2e._stored(
        config, providers, config.indexing.opensearch.entities_index_prefix
    )
    assert entities


@pytest.mark.parametrize(
    ("status", "mode", "recorded", "counted"),
    [
        (503, "alone", 0, 0),
        (400, "alone", 1, 1),
        (503, "beside", 1, 0),
        (400, "beside", 1, 1),
        (503, "reset", 1, 0),
        (400, "reset", 1, 1),
    ],
)
def test_failed_documents_counts_the_documents_recorded_failed(
    tmp_path: Path, status: int, mode: str, recorded: int, counted: int
) -> None:
    # The JSON result counted the documents a stage failed on: a model
    # outage that recorded nothing (fz.txt stays PENDING) read as one
    # document recorded FAILED. Beside a new document, or in a reset
    # rebuild, the run completes past fz.txt's failed units and its commit
    # records fz.txt FAILED either way, counting the failure only when the
    # error is fz.txt's own (the 400), not the outage's.
    contexts: list[PipelineContext] = []
    seen = _runs(
        tmp_path,
        status,
        contexts=contexts,
        reset=mode == "reset",
        beside=mode == "beside",
    )
    runner = IngestionPipelineRunner.__new__(IngestionPipelineRunner)
    runner.pipeline = SimpleNamespace(executed_stages=frozenset())  # type: ignore[assignment]
    result = runner._json_result(contexts[0])
    assert result["failed_documents"] == recorded, result
    assert seen[0][1][1] == counted, seen
    if mode != "alone":
        assert seen[0][0] == PipelineStageStatus.COMPLETED.value, seen


_STAGES = pytest.mark.parametrize(
    "stage",
    [
        "knowledge graph extraction expert",
        "claim extraction specialist",
        "graph refinement specialist",
    ],
    ids=["graph", "claims", "gleaning"],
)


def _rejecting(tmp: Path, rejects: dict[str, Any]) -> tuple[Any, Any, Path]:
    """Embedded config (claims and gleaning on, max_document_failures 2) and
    providers whose model answers HTTP 400 "request rejected" to the requests
    of the stage whose system prompt contains ``rejects["stage"]`` and whose
    text contains ``rejects["text"]`` (every one when that is ``""``)."""
    config = e2e._config(tmp / "stores", tmp / "viz")
    config.indexing.max_document_failures = 2
    config.processing.claim_extraction.enabled = True
    config.processing.gleaning.enabled = True
    server = FakeOllama(e2e._respond, "x", e2e.DIM)

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat" and rejects["stage"]:
            messages = json.loads(request.content)["messages"]
            system = "".join(m["content"] for m in messages if m["role"] == "system")
            human = "".join(m["content"] for m in messages if m["role"] == "user")
            if rejects["stage"] in system and rejects["text"] in human:
                return httpx.Response(400, json={"error": "request rejected"})
        return server(request)

    providers = e2e._providers(
        config, transport, e2e.FakeCrossEncoder("r")  # type: ignore[arg-type]
    )
    return config, providers, tmp / "corpus"


def _counts(providers: Any) -> dict[str, int]:
    return {
        r.file_path: r.failure_count for r in providers.build_doc_status().list_all()
    }


@_STAGES
@pytest.mark.parametrize("files", [_FILES, {"d0.txt": _FILES["d0.txt"]}], ids=len)
def test_one_error_on_every_unit_of_a_delta_counts_for_no_document(
    tmp_path: Path, stage: str, files: dict[str, str]
) -> None:
    # The model answers every request of one stage, a known-good one too,
    # with one HTTP 400 no configuration pattern names: a setting it
    # rejects, not the documents' content. Counted in claim extraction or
    # gleaning (as it was), or for a delta of one document, three runs
    # excluded the documents for good.
    config, providers, source = _rejecting(tmp_path, {"stage": stage, "text": ""})
    source.mkdir()
    for name, text in files.items():
        (source / name).write_text(text)
    for run in range(3):
        e2e._ingest(config, providers, source, tmp_path / f"run{run}")
        assert _counts(providers) == dict.fromkeys(files, 0), run


@_STAGES
def test_a_delta_of_only_documents_the_model_rejects_is_excluded(
    tmp_path: Path, stage: str
) -> None:
    # Two documents whose every request the model rejects alike, alone in a
    # delta with a deletion: taken for one cause, they were counted for
    # none, so every run failed on them and the deletion never applied. The
    # model answers a known-good request: they are counted, excluded after
    # max_document_failures runs, and the deletion applies.
    rejects = {"stage": "", "text": "Zorbix"}
    config, providers, source = _rejecting(tmp_path, rejects)
    source.mkdir()
    for name, text in _FILES.items():
        (source / name).write_text(text)
    e2e._ingest(config, providers, source, tmp_path / "seed")
    (source / "d1.txt").unlink()
    (source / "z1.txt").write_text(_FAILING)
    (source / "z2.txt").write_text("Zorbix and Depot trade parts.")
    rejects["stage"] = stage
    seen = []
    for run in range(4):
        e2e._ingest(config, providers, source, tmp_path / f"run{run}")
        counts = _counts(providers)
        seen.append((counts["z1.txt"], counts["z2.txt"], "d1.txt" in counts))
    assert [run[:2] for run in seen] == [(1, 1), (2, 2), (2, 2), (2, 2)], seen
    # Graph extraction failing on every unit fails the run, so the deletion
    # waits for the exclusion; in the other stages it applies at once.
    assert seen[-1][2] is False, seen


def test_a_cached_known_good_answer_does_not_hide_a_rejected_setting(
    tmp_path: Path,
) -> None:
    # Run P1 again after the model began rejecting every request: the
    # known-good request's answer of the first run, served from P1's LLM
    # response cache, would have counted the setting's error against fz.txt.
    rejects = {"stage": "knowledge graph extraction expert", "text": "Zorbix"}
    config, providers, source = _rejecting(tmp_path, rejects)
    source.mkdir()
    (source / "fz.txt").write_text(_FAILING)
    e2e._ingest(config, providers, source, tmp_path / "P1")
    assert _counts(providers) == {"fz.txt": 1}
    rejects["text"] = ""
    e2e._ingest(config, providers, source, tmp_path / "P1")
    assert _counts(providers) == {"fz.txt": 1}


def test_failed_documents_holds_a_commit_stopped_after_its_registry_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The commit recorded z1.txt FAILED, then its cleanup raised: the run's
    # failed_documents read 0, the documents being added to the result only
    # once the commit returned.
    rejects = {"stage": "claim extraction specialist", "text": "Zorbix"}
    config, providers, source = _rejecting(tmp_path, rejects)
    source.mkdir()
    (source / "d0.txt").write_text(_FILES["d0.txt"])
    (source / "z1.txt").write_text(_FAILING)

    def stopped(self: Any) -> None:
        raise RuntimeError("cleanup stopped")

    monkeypatch.setattr(IndexingManager, "collect_extraction_records", stopped)
    context = e2e._ingest(config, providers, source, tmp_path / "run")
    assert context.status == PipelineStageStatus.FAILED
    assert _counts(providers)["z1.txt"] == 1
    assert len(context.recorded_failed_doc_ids) == 1


@pytest.mark.parametrize(
    "stage",
    ["knowledge graph extraction expert", "claim extraction specialist"],
    ids=["stage-stops-the-run", "commit"],
)
def test_a_document_failing_runs_for_an_error_not_its_own_is_reported(
    tmp_path: Path, stage: str, caplog: pytest.LogCaptureFixture
) -> None:
    # The model is down for fz.txt's requests (HTTP 503) run after run: it
    # is never counted toward exclusion, so nothing said it kept failing.
    # Its consecutive failed runs are kept (retry_count) and, from the second
    # on, warned of with the file and counted in documents_retrying.
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.processing.claim_extraction.enabled = True
    server = FakeOllama(e2e._respond, "x", e2e.DIM)

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            messages = json.loads(request.content)["messages"]
            system = "".join(m["content"] for m in messages if m["role"] == "system")
            human = "".join(m["content"] for m in messages if m["role"] == "user")
            if stage in system and "Zorbix" in human:
                return httpx.Response(503, json={"error": "unavailable"})
        return server(request)

    providers = e2e._providers(
        config, transport, e2e.FakeCrossEncoder("r")  # type: ignore[arg-type]
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "d0.txt").write_text(_FILES["d0.txt"])
    e2e._ingest(config, providers, source, tmp_path / "seed")
    (source / "fz.txt").write_text(_FAILING)
    runner = IngestionPipelineRunner.__new__(IngestionPipelineRunner)
    runner.pipeline = SimpleNamespace(executed_stages=frozenset())  # type: ignore[assignment]
    seen = []
    for run in range(3):
        caplog.clear()
        context = e2e._ingest(config, providers, source, tmp_path / f"run{run}")
        (fz,) = [
            r
            for r in providers.build_doc_status().list_all()
            if r.file_path == "fz.txt"
        ]
        result = runner._json_result(context)
        seen.append((fz.failure_count, fz.retry_count, result["documents_retrying"]))
    assert seen == [(0, 1, 0), (0, 2, 1), (0, 3, 1)], seen
    assert "fz.txt (3)" in caplog.text
    # Indexed once the model is back: the count starts over.
    server_ok = FakeOllama(e2e._respond, "x", e2e.DIM)
    providers = e2e._providers(config, server_ok, e2e.FakeCrossEncoder("r"))
    e2e._ingest(config, providers, source, tmp_path / "fixed")
    (fz,) = [
        r for r in providers.build_doc_status().list_all() if r.file_path == "fz.txt"
    ]
    assert (fz.status, fz.retry_count) == (DocStatus.PROCESSED, 0)


_STOPS_OR_COMMITS = pytest.mark.parametrize(
    "stage",
    ["knowledge graph extraction expert", "claim extraction specialist"],
    # Graph extraction failing on fz alone stops the run before indexing
    # (record_item_failures); claim extraction lets the commit record it.
    ids=["stage-stops-the-run", "commit"],
)


def _outage(tmp: Path, stage: str, mode: dict[str, int]) -> tuple[Any, Any, Path]:
    """Embedded config (claims on, max_document_failures 5) and providers
    whose model answers HTTP ``mode["status"]`` to the requests of ``stage``
    (a system-prompt phrase) for Zorbix's text; d0 is indexed."""
    config = e2e._config(tmp / "stores", tmp / "viz")
    config.processing.claim_extraction.enabled = True
    config.indexing.max_document_failures = 5
    server = FakeOllama(e2e._respond, "x", e2e.DIM)

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            messages = json.loads(request.content)["messages"]
            system = "".join(m["content"] for m in messages if m["role"] == "system")
            human = "".join(m["content"] for m in messages if m["role"] == "user")
            if stage in system and "Zorbix" in human:
                return httpx.Response(mode["status"], json={"error": "boom"})
        return server(request)

    providers = e2e._providers(
        config, transport, e2e.FakeCrossEncoder("r")  # type: ignore[arg-type]
    )
    source = tmp / "corpus"
    source.mkdir()
    (source / "d0.txt").write_text(_FILES["d0.txt"])
    e2e._ingest(config, providers, source, tmp / "seed")
    return config, providers, source


def _fz(providers: Any) -> Any:
    (fz,) = [
        r for r in providers.build_doc_status().list_all() if r.file_path == "fz.txt"
    ]
    return fz


@_STOPS_OR_COMMITS
def test_an_own_error_after_an_outage_starts_retry_count_over(
    tmp_path: Path, stage: str
) -> None:
    # Two runs of an outage, then an error of the document's own: it counts
    # toward exclusion, and the failed runs for errors not its own end.
    mode = {"status": 503}
    config, providers, source = _outage(tmp_path, stage, mode)
    (source / "fz.txt").write_text(_FAILING)
    seen = []
    for run in range(2):
        e2e._ingest(config, providers, source, tmp_path / f"o{run}")
        seen.append((_fz(providers).failure_count, _fz(providers).retry_count))
    mode["status"] = 400
    e2e._ingest(config, providers, source, tmp_path / "own")
    seen.append((_fz(providers).failure_count, _fz(providers).retry_count))
    assert seen == [(0, 1), (0, 2), (1, 0)], seen


@_STOPS_OR_COMMITS
def test_new_content_starts_retry_count_over(tmp_path: Path, stage: str) -> None:
    # The document changed between two outage runs: the second run's
    # failure is the new content's first (opening the attempt at the new
    # content kept the old count, which earlier_retries then credited to it).
    mode = {"status": 503}
    config, providers, source = _outage(tmp_path, stage, mode)
    seen = []
    for run, text in enumerate([_FAILING, "Zorbix and Buyer trade bolts."]):
        (source / "fz.txt").write_text(text)
        e2e._ingest(config, providers, source, tmp_path / f"c{run}")
        seen.append(_fz(providers).retry_count)
    assert seen == [1, 1], seen


def test_a_failed_registry_write_records_no_failed_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The commit's FAILED records are not written: the run's
    # failed_documents must not count them as recorded.
    from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer

    config, providers, source = _rejecting(
        tmp_path, {"stage": "claim extraction specialist", "text": "Zorbix"}
    )
    source.mkdir()
    (source / "d0.txt").write_text(_FILES["d0.txt"])
    (source / "z1.txt").write_text(_FAILING)
    put = IncrementalIndexer._put

    def failing_put(self: IncrementalIndexer, records: list[Any]) -> Any:
        if any(r.status is DocStatus.FAILED for r in records):
            raise RuntimeError("registry down")
        return put(self, records)

    monkeypatch.setattr(IncrementalIndexer, "_put", failing_put)
    context = e2e._ingest(config, providers, source, tmp_path / "run")
    assert context.recorded_failed_doc_ids == []


def test_a_same_pid_rerun_after_the_fix_reports_no_retrying(tmp_path: Path) -> None:
    # documents_retrying is the run's own: the rerun of the pipeline id that
    # reported fz indexes it and reports none.
    mode = {"status": 503}
    config, providers, source = _outage(tmp_path, "claim extraction specialist", mode)
    (source / "fz.txt").write_text(_FAILING)
    for _ in range(2):
        context = e2e._ingest(config, providers, source, tmp_path / "P1")
    assert context.retrying_doc_ids
    fixed = e2e._providers(
        config, FakeOllama(e2e._respond, "x", e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    context = e2e._ingest(config, fixed, source, tmp_path / "P1")
    assert context.retrying_doc_ids == []


def test_a_reset_rebuild_reports_a_document_retrying(tmp_path: Path) -> None:
    mode = {"status": 503}
    config, providers, source = _outage(tmp_path, "claim extraction specialist", mode)
    config.indexing.reset = True
    (source / "fz.txt").write_text(_FAILING)
    seen = []
    for run in range(2):
        context = e2e._ingest(config, providers, source, tmp_path / f"r{run}")
        seen.append((_fz(providers).retry_count, len(context.retrying_doc_ids)))
    assert seen == [(1, 0), (2, 1)], seen
