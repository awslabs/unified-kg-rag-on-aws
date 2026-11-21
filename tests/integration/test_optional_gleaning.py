# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gleaning failing for an error of a unit's own (embedded stores).

Gleaning refines an extraction that stands without it: a document whose
graph was extracted is indexed with that extraction when the gleaning of
some of its units fails for an error of their own (output cut off at the
cap, unparseable, a request rejected for its content), and is not counted
toward ``indexing.max_document_failures``. As with claims
(``test_optional_claims``), it is recorded FAILED once, under an
``error_info`` that says it is indexed, so the next run extracts and gleans
those units again (the record stored for them is their ungleaned
extraction, which a later deletion derives their items from); failing so
again, it is recorded PROCESSED. It was recorded FAILED and counted: the
ARCH baseline's registry held 35% of the corpus FAILED while 13% lacked an
extraction.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.fixtures.doc_status_records import all_records
from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.domain.models import DocStatus, PipelineStageStatus

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

no_aws = e2e.no_aws

_KINDS = {
    "knowledge graph extraction expert": "graph",
    "graph refinement specialist": "gleaning",
}
_FAILING = ["z1.txt", "z2.txt"]


def _setup(tmp: Path, mode: dict[str, str]) -> tuple[Any, Any, Path, Counter]:
    """Index d0, then add z1 and z2 (Zorbix's). While ``mode["m"]`` is "cut",
    the gleaning output of Zorbix's units is cut at the output token limit
    (alike across both documents, as a server-side output cap). Returns the
    model calls by kind, cleared after the setup."""
    config = e2e._config(tmp / "stores", tmp / "viz")
    config.processing.gleaning.enabled = True
    server = FakeOllama(e2e._respond, "x", e2e.DIM)
    calls: Counter = Counter()

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            text = "".join(m["content"] for m in body["messages"])
            kind = next((k for p, k in _KINDS.items() if p in text), "other")
            calls[kind] += 1
            if kind == "gleaning" and "Zorbix" in text and mode["m"] == "cut":
                chunk = {
                    "model": body["model"],
                    "created_at": "2026-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": "<issues><issue>"},
                    "done": True,
                    "done_reason": "length",
                }
                return httpx.Response(200, content=(json.dumps(chunk) + "\n").encode())
        return server(request)

    providers = e2e._providers(
        config, transport, e2e.FakeCrossEncoder("r")  # type: ignore[arg-type]
    )
    source = tmp / "corpus"
    source.mkdir()
    (source / "d0.txt").write_text("Vendor and Buyer trade goods.")
    e2e._ingest(config, providers, source, tmp / "seed")
    (source / "z1.txt").write_text("Zorbix and Depot trade parts.")
    (source / "z2.txt").write_text("Zorbix and Carrier trade parts.")
    calls.clear()
    return config, providers, source, calls


def _records(providers: Any) -> dict[str, tuple]:
    return {
        r.file_path: (r.status, r.failure_count, r.retry_count, r.error_info)
        for r in all_records(providers.build_doc_status())
    }


def _entity_names(config: Any, providers: Any) -> set[str]:
    return {
        hit["_source"]["name"]
        for hit in e2e._stored(
            config, providers, config.indexing.opensearch.entities_index_prefix
        )
    }


def test_documents_are_indexed_without_the_failed_gleaning_and_retried_once(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    mode = {"m": "cut"}
    config, providers, source, calls = _setup(tmp_path, mode)
    caplog.set_level("WARNING")
    context = e2e._ingest(config, providers, source, tmp_path / "run0")
    assert context.status is PipelineStageStatus.COMPLETED
    records = _records(providers)
    for name in _FAILING:
        status, failures, retries, info = records[name]
        # FAILED once, uncounted, and saying what was indexed.
        assert (status, failures, retries) == (DocStatus.FAILED, 0, 0), records
        assert info is not None and info.startswith("indexed without the gleaning")
    assert records["d0.txt"][0] is DocStatus.PROCESSED
    # Not among the run's failed documents (failed_documents).
    assert context.recorded_failed_doc_ids == []
    assert (
        "Gleaning failed on part of 2 documents for an error of their own"
        in caplog.text
    )
    # Their extraction is indexed: the entities only they name are stored.
    assert {"Depot", "Carrier"} <= _entity_names(config, providers)

    # Failing again: each failing unit extracted and gleaned once more (its
    # stored record is not gleaned), then recorded PROCESSED with its
    # extraction.
    caplog.clear()
    calls.clear()
    context = e2e._ingest(config, providers, source, tmp_path / "run1")
    assert (calls["graph"], calls["gleaning"]) == (2, 2), calls
    records = _records(providers)
    assert {name: r[:3] for name, r in records.items()} == dict.fromkeys(
        records, (DocStatus.PROCESSED, 0, 0)
    ), records
    assert "which later runs do not extract again" in caplog.text
    assert context.recorded_failed_doc_ids == []
    assert {"Depot", "Carrier"} <= _entity_names(config, providers)

    # And not again: the documents read unchanged.
    calls.clear()
    e2e._ingest(config, providers, source, tmp_path / "run2")
    assert sum(calls.values()) == 0, calls


def test_the_next_run_gleans_the_units_again(tmp_path: Path) -> None:
    mode = {"m": "cut"}
    config, providers, source, calls = _setup(tmp_path, mode)
    e2e._ingest(config, providers, source, tmp_path / "run0")
    mode["m"] = "ok"
    calls.clear()
    e2e._ingest(config, providers, source, tmp_path / "run1")
    assert (calls["graph"], calls["gleaning"]) == (2, 2), calls
    assert {r[0] for r in _records(providers).values()} == {DocStatus.PROCESSED}


def test_a_reset_rebuild_indexes_without_the_failed_gleaning(tmp_path: Path) -> None:
    mode = {"m": "cut"}
    config, providers, source, _ = _setup(tmp_path, mode)
    config.indexing.reset = True
    seen = []
    for run in range(2):
        context = e2e._ingest(config, providers, source, tmp_path / f"r{run}")
        assert context.status is PipelineStageStatus.COMPLETED
        assert context.recorded_failed_doc_ids == []
        seen.append(_records(providers)["z1.txt"][:3])
    assert seen == [(DocStatus.FAILED, 0, 0), (DocStatus.PROCESSED, 0, 0)], seen


def test_a_deletion_derives_shared_entities_from_an_ungleaned_record(
    tmp_path: Path,
) -> None:
    # R2A-3: a unit whose gleaning failed stored no record, so a later
    # deletion of a document sharing its entities could not derive them
    # again: Depot kept the deleted document's text and file filter.
    mode = {"m": "cut"}
    config, providers, source, calls = _setup(tmp_path, mode)
    (source / "y.txt").write_text("Depot and Buyer store goods.")
    for run in ("run0", "run1"):  # FAILED, then given up: PROCESSED
        e2e._ingest(config, providers, source, tmp_path / run)
    assert _records(providers)["z1.txt"][0] is DocStatus.PROCESSED

    (source / "y.txt").unlink()
    context = e2e._ingest(config, providers, source, tmp_path / "run2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    (depot,) = [
        hit["_source"]
        for hit in e2e._stored(
            config, providers, config.indexing.opensearch.entities_index_prefix
        )
        if hit["_source"]["name"] == "Depot"
    ]
    assert depot["attr_file"] in ("z1.txt", ["z1.txt"])
