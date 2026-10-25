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

import httpx
import pytest
from botocore.exceptions import NoCredentialsError, TokenRetrievalError

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_ingestion_stages import _source_text
from unified_kg_rag.application.cli.run_ingestion_pipeline import (
    IngestionPipelineRunner,
)
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
) -> list[tuple]:
    """Index d0, d1; add fz.txt, whose extraction the model answers with
    ``status`` (and ``message``), or fails with an exception ``status``, and
    delete d1; run four times. Each run's (status, fz's
    record (status, failure count), d1 still recorded); the last run's
    graph_extraction error."""
    config = e2e._config(tmp / "stores", tmp / "viz")
    config.indexing.max_document_failures = 2
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
    *seen, errors = _runs(tmp_path, 400, pipeline_id)
    failed = PipelineStageStatus.FAILED.value
    done = PipelineStageStatus.COMPLETED.value
    assert seen[:2] == [
        (failed, (DocStatus.FAILED.value, 1), True),
        (failed, (DocStatus.FAILED.value, 2), True),
    ], seen
    # Skipped from the third run on: the run completes and deletes d1.
    assert seen[2:] == [(done, (DocStatus.FAILED.value, 2), False)] * 2, seen
    assert errors == []


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


@pytest.mark.parametrize(("status", "recorded"), [(503, 0), (400, 1)])
def test_failed_documents_counts_the_documents_recorded_failed(
    tmp_path: Path, status: int, recorded: int
) -> None:
    # The JSON result counted the documents a stage failed on: a model
    # outage that recorded nothing (fz.txt stays PENDING) read as one
    # document recorded FAILED.
    contexts: list[PipelineContext] = []
    _runs(tmp_path, status, contexts=contexts)
    runner = IngestionPipelineRunner.__new__(IngestionPipelineRunner)
    runner.pipeline = SimpleNamespace(executed_stages=frozenset())  # type: ignore[assignment]
    result = runner._json_result(contexts[0])
    assert result["failed_documents"] == recorded, result


@pytest.mark.parametrize(
    "stage",
    [
        "knowledge graph extraction expert",
        "claim extraction specialist",
        "graph refinement specialist",
    ],
    ids=["graph", "claims", "gleaning"],
)
def test_one_error_on_every_unit_of_a_delta_counts_for_no_document(
    tmp_path: Path, stage: str
) -> None:
    # The model answers every request of one stage, for both documents, with
    # one HTTP 400 no configuration pattern names: a setting it rejects, not
    # each document's content. Counted in claim extraction or gleaning (as it
    # was), three runs excluded both documents for good.
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.indexing.max_document_failures = 2
    config.processing.claim_extraction.enabled = True
    config.processing.gleaning.enabled = True
    server = FakeOllama(e2e._respond, "x", e2e.DIM)

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            messages = json.loads(request.content)["messages"]
            system = "".join(m["content"] for m in messages if m["role"] == "system")
            if stage in system:
                return httpx.Response(400, json={"error": "request rejected"})
        return server(request)

    providers = e2e._providers(
        config, transport, e2e.FakeCrossEncoder("r")  # type: ignore[arg-type]
    )
    source = tmp_path / "corpus"
    source.mkdir()
    for name, text in _FILES.items():
        (source / name).write_text(text)
    for run in range(3):
        e2e._ingest(config, providers, source, tmp_path / f"run{run}")
        counts = {
            r.file_path: r.failure_count
            for r in providers.build_doc_status().list_all()
        }
        assert counts == dict.fromkeys(_FILES, 0), (run, counts)
