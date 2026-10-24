# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A document that always fails extraction, alone in an incremental delta.

The extraction stage fails on all of its units, which fails the stage and
stops the run before the indexing stage records the document FAILED. Its
consecutive-failure count must still grow when the error is the document's
own (a 4xx), so that after ``indexing.max_document_failures`` runs it is
skipped and the run applies the deletions of its delta; a model outage
(5xx) must not count, so it never excludes a document.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import httpx
import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_ingestion_stages import _source_text
from unified_kg_rag.domain.models import DocStatus, PipelineStageStatus

pytestmark = pytest.mark.integration

_FAILING = "Zorbix and Vendor trade widgets."
_FILES = {
    "d0.txt": "Vendor and Buyer trade goods.",
    "d1.txt": "Depot and Carrier trade parts.",
}


def _runs(tmp: Path, status: int, pipeline_id: str | None = None) -> list[tuple]:
    """Index d0, d1; add fz.txt, whose extraction the model answers with
    ``status``, and delete d1; run four times. Each run's (status, fz's
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
                return httpx.Response(status, json={"error": "rejected"})
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
