# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry records of a run an artifact stage stopped before indexing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.application.ingestion.incremental import (
    record_stopped_run_failed,
)
from unified_kg_rag.domain.models import (
    PENDING_CONTENT_HASH,
    DocStatus,
    DocStatusRecord,
    DocumentDelta,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
)

pytestmark = pytest.mark.unit


def _context(pipeline_id: str, delta: DocumentDelta) -> PipelineContext:
    return PipelineContext(
        pipeline_id=pipeline_id,
        config=PipelineConfig(),
        status=PipelineStageStatus.RUNNING,
        start_time=datetime.now(),
        source_directory=Path("/corpus"),
        incremental_delta=delta,
        incremental_fingerprints=dict.fromkeys(delta.to_process, "new-hash"),
        incremental_scope="default|/corpus",
        incremental_extraction_fingerprint="settings-1",
    )


def _stored(doc_id: str, status: DocStatus, content_hash: str = "old-hash"):
    return DocStatusRecord(
        doc_id=doc_id,
        content_hash=content_hash,
        status=status,
        scope="default|/corpus",
        file_path=f"{doc_id}.txt",
        entity_ids=[f"{doc_id}-entity"],
        text_unit_ids=[f"{doc_id}-unit"],
    )


def test_changed_documents_keep_their_indexed_lineage() -> None:
    # Nothing was written for the delta, so the stored lineage is still what
    # the stores hold for the document: the next run removes it on re-extract.
    store = FakeDocStatusStore()
    store.put(_stored("changed", DocStatus.PROCESSED))
    context = _context("run-1", DocumentDelta(new=["added"], changed=["changed"]))

    assert record_stopped_run_failed(store, context, [], "graph_extraction") == 2

    changed = store.get("changed")
    assert changed is not None
    assert changed.status is DocStatus.FAILED
    assert changed.content_hash == "new-hash"
    assert changed.failure_count == 1
    assert changed.entity_ids == ["changed-entity"]
    assert changed.text_unit_ids == ["changed-unit"]
    assert changed.extraction_fingerprint == "settings-1"
    assert "graph_extraction" in (changed.error_info or "")
    added = store.get("added")
    assert added is not None
    assert added.status is DocStatus.FAILED
    assert added.failure_count == 1
    assert added.scope == "default|/corpus"
    assert added.entity_ids == []


def test_failures_count_per_run_not_per_resume() -> None:
    store = FakeDocStatusStore()
    delta = DocumentDelta(new=["doc"])
    record_stopped_run_failed(store, _context("run-1", delta), [], "gleaning")
    # Resuming run-1 and failing the same stage again is the same run.
    record_stopped_run_failed(store, _context("run-1", delta), [], "gleaning")
    assert store.get("doc").failure_count == 1  # type: ignore[union-attr]

    changed = DocumentDelta(changed=["doc"])
    record_stopped_run_failed(store, _context("run-2", changed), [], "gleaning")
    assert store.get("doc").failure_count == 2  # type: ignore[union-attr]


def test_new_content_restarts_the_count() -> None:
    store = FakeDocStatusStore()
    failed = _stored("doc", DocStatus.FAILED, content_hash="older-hash")
    store.put(failed.model_copy(update={"failure_count": 2}))
    context = _context("run-3", DocumentDelta(changed=["doc"]))

    record_stopped_run_failed(store, context, [], "graph_extraction")

    assert store.get("doc").failure_count == 1  # type: ignore[union-attr]


def test_a_pending_write_ahead_record_is_kept() -> None:
    # Its lineage (and any overflow) lists what an interrupted run may have
    # written; replacing it would orphan those artifacts.
    store = FakeDocStatusStore()
    pending = _stored("doc", DocStatus.PENDING, content_hash=PENDING_CONTENT_HASH)
    store.put(pending)
    context = _context("run-4", DocumentDelta(changed=["doc"]))

    assert record_stopped_run_failed(store, context, [], "graph_extraction") == 0
    assert store.get("doc") == pending
