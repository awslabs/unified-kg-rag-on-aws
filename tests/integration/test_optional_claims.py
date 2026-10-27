# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Claim extraction failing for an error of a unit's own (embedded stores).

Claims are optional: a document whose graph was extracted is indexed without
the claims of the units whose claim extraction failed (a request rejected
for its content, output cut off, alike across documents too) and is not
counted toward ``indexing.max_document_failures``. It is recorded FAILED
once, so the next run extracts the missing claims again, reusing every
stored record; failing so again, it is recorded PROCESSED without them, so a
lasting failure (a server-side output cap) re-indexes it once, not on every
run.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_claim_convergence as convergence
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.domain.models import DocStatus, PipelineStageStatus

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

no_aws = e2e.no_aws

_KINDS = {
    "knowledge graph extraction expert": "graph",
    "claim extraction specialist": "claims",
}


def _setup(tmp: Path, mode: dict[str, str]) -> tuple[Any, Any, Path, Counter]:
    """Index d0, then add z1 and z2 (Globex's). While ``mode["m"]`` is "cut",
    the claim extraction output of Globex's units is cut at the output token
    limit (alike across both documents); while it is "rejected", z1's claim
    request is rejected for its content (HTTP 400; a known-good request is
    answered). Returns the model calls by kind, cleared after the setup."""
    config = e2e._config(tmp / "stores", tmp / "viz")
    config.processing.claim_extraction.enabled = True
    server = FakeOllama(convergence._respond, "x", e2e.DIM)
    calls: Counter = Counter()

    def transport(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            text = "".join(m["content"] for m in body["messages"])
            kind = next((k for p, k in _KINDS.items() if p in text), "other")
            calls[kind] += 1
            if kind == "claims" and "Globex" in text and mode["m"] == "cut":
                chunk = {
                    "model": body["model"],
                    "created_at": "2026-01-01T00:00:00Z",
                    "message": {"role": "assistant", "content": "<claims><claim>"},
                    "done": True,
                    "done_reason": "length",
                }
                return httpx.Response(200, content=(json.dumps(chunk) + "\n").encode())
            if kind == "claims" and "Depot" in text and mode["m"] == "rejected":
                return httpx.Response(400, json={"error": "request rejected"})
        return server(request)

    providers = e2e._providers(
        config, transport, e2e.FakeCrossEncoder("r")  # type: ignore[arg-type]
    )
    source = tmp / "corpus"
    source.mkdir()
    (source / "d0.txt").write_text("Acme supplies Buyer.")
    e2e._ingest(config, providers, source, tmp / "seed")
    (source / "z1.txt").write_text("Depot supplies Globex.")
    (source / "z2.txt").write_text("Globex supplies Buyer.")
    calls.clear()
    return config, providers, source, calls


def _state(providers: Any) -> dict[str, tuple]:
    return {
        r.file_path: (r.status, r.failure_count, r.retry_count)
        for r in providers.build_doc_status().list_all()
    }


def _claims(config: Any, providers: Any) -> list[str]:
    hits = e2e._stored(
        config, providers, config.indexing.opensearch.claims_index_prefix
    )
    return sorted(hit["_source"]["description"] for hit in hits)


@pytest.mark.parametrize("cause", ["cut", "rejected"])
def test_documents_are_indexed_without_the_failed_claims_and_retried_once(
    tmp_path: Path, cause: str, caplog: pytest.LogCaptureFixture
) -> None:
    # Counted like the graph's errors, three runs of a server-side output
    # cap on claims excluded every document of the corpus (FAILED at the
    # limit) though its graph was extracted.
    mode = {"m": cause}
    config, providers, source, calls = _setup(tmp_path, mode)
    failing = ["z1.txt", "z2.txt"] if cause == "cut" else ["z1.txt"]
    caplog.set_level("WARNING")
    context = e2e._ingest(config, providers, source, tmp_path / "run0")
    assert context.status is PipelineStageStatus.COMPLETED
    state = _state(providers)
    for name in failing:
        assert state[name] == (DocStatus.FAILED, 0, 0), state
    assert context.recorded_failed_doc_ids == []
    assert context.retrying_doc_ids == []
    assert "the next run extracts them again: " + ", ".join(failing) in caplog.text
    indexed = _claims(config, providers)
    assert "Acme must deliver to Buyer" in indexed
    # Their graph is indexed: the entities are stored.
    names = {
        hit["_source"]["name"]
        for hit in e2e._stored(
            config, providers, config.indexing.opensearch.entities_index_prefix
        )
    }
    assert {"Depot", "Globex"} <= names

    # Failing again: one claims call per failing unit, nothing else
    # extracted again, then recorded PROCESSED without those claims.
    caplog.clear()
    calls.clear()
    context = e2e._ingest(config, providers, source, tmp_path / "run1")
    assert calls["graph"] == 0, calls
    # (A rejection of one document's every unit also asks a known-good
    # request.)
    assert calls["claims"] == len(failing) + (cause == "rejected"), calls
    state = _state(providers)
    assert all(state[name][0] is DocStatus.PROCESSED for name in state), state
    assert "which are not extracted again" in caplog.text
    assert context.recorded_failed_doc_ids == []

    # And not again: the documents read unchanged.
    calls.clear()
    e2e._ingest(config, providers, source, tmp_path / "run2")
    assert sum(calls.values()) == 0, calls
    assert _claims(config, providers) == indexed


def test_the_next_run_extracts_the_missing_claims(tmp_path: Path) -> None:
    mode = {"m": "cut"}
    config, providers, source, calls = _setup(tmp_path, mode)
    e2e._ingest(config, providers, source, tmp_path / "run0")
    assert "Globex must deliver to Buyer" not in _claims(config, providers)
    mode["m"] = "ok"
    calls.clear()
    e2e._ingest(config, providers, source, tmp_path / "run1")
    # Only the missing claims are extracted; the graph records are reused.
    assert (calls["graph"], calls["claims"]) == (0, 2), calls
    assert {"Depot must deliver to Globex", "Globex must deliver to Buyer"} <= set(
        _claims(config, providers)
    )
    state = _state(providers)
    assert {s for s, _, _ in state.values()} == {DocStatus.PROCESSED}, state


def test_a_reset_rebuild_indexes_without_the_failed_claims(tmp_path: Path) -> None:
    mode = {"m": "cut"}
    config, providers, source, _ = _setup(tmp_path, mode)
    config.indexing.reset = True
    seen = []
    for run in range(2):
        context = e2e._ingest(config, providers, source, tmp_path / f"r{run}")
        assert context.status is PipelineStageStatus.COMPLETED
        seen.append(_state(providers)["z1.txt"])
        assert context.recorded_failed_doc_ids == []
    assert seen == [(DocStatus.FAILED, 0, 0), (DocStatus.PROCESSED, 0, 0)], seen


def test_a_claims_outage_still_fails_the_document_uncounted(tmp_path: Path) -> None:
    # An error not the unit's own (HTTP 503) is retried as in every stage.
    mode = {"m": "ok"}
    config, providers, source, _ = _setup(tmp_path, mode)
    server = FakeOllama(convergence._respond, "x", e2e.DIM)

    def transport(request: httpx.Request) -> httpx.Response:
        text = request.content.decode()
        if "claim extraction specialist" in text and "Globex" in text:
            return httpx.Response(503, json={"error": "unavailable"})
        return server(request)

    providers = e2e._providers(
        config, transport, e2e.FakeCrossEncoder("r")  # type: ignore[arg-type]
    )
    for run in range(2):
        context = e2e._ingest(config, providers, source, tmp_path / f"o{run}")
    assert _state(providers)["z1.txt"] == (DocStatus.FAILED, 0, 2)
    assert set(context.recorded_failed_doc_ids) == set(context.retrying_doc_ids)
