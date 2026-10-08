# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""IndexingStage._index_incremental: the write path of an incremental run.

Pins the contract between the stage and the incremental orchestrator, which is
what keeps the live stores and the doc-status registry consistent:

* stale artifacts of changed docs are pruned, then deleted docs' artifacts and
  registry rows are removed, and only then is the delta upserted + recorded;
* the run's index suffix comes from the first text unit (default otherwise);
* a missing delta falls back to a full ``index_all_data`` without touching the
  registry.

AWS-free: an in-memory registry plus a recording indexing manager.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.domain.ingestion.delta_detector import compute_doc_id
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    Document,
    DocumentDelta,
    Entity,
    PipelineContext,
    PipelineStageStatus,
    TextUnit,
)
from unified_kg_rag.ports.indexer import IndexingStats
from unified_kg_rag.shared import PipelineStageError

pytestmark = pytest.mark.unit

_CHANGED = "/corpus/changed.txt"
_DELETED = "/corpus/deleted.txt"


class _RecordingManager:
    """IndexingManager stand-in that records the order of store operations."""

    def __init__(self, delete_failures: int = 0) -> None:
        self.config = Config()
        self.calls: list[tuple[str, Any]] = []
        self._delete_failures = delete_failures

    def index_all_data(self, **kwargs: Any) -> dict[str, IndexingStats]:
        self.calls.append(("index_all_data", kwargs))
        return {}

    def delete_documents(
        self, ids_by_suffix: dict[str, list[str]]
    ) -> dict[str, IndexingStats]:
        self.calls.append(("delete_documents", ids_by_suffix))
        removed = sum(len(ids) for ids in ids_by_suffix.values())
        return {
            "store": IndexingStats(
                total_items=removed,
                successful_items=removed - self._delete_failures,
                failed_items=self._delete_failures,
            )
        }

    def index_delta(self, **kwargs: Any) -> dict[str, IndexingStats]:
        self.calls.append(("index_delta", kwargs))
        written = len(kwargs.get("entities") or [])
        return {
            "entities": IndexingStats(total_items=written, successful_items=written)
        }


def _stage(mocker, store: FakeDocStatusStore, manager: _RecordingManager):
    mocker.patch.object(ps, "IndexingManager", return_value=manager)
    return ps.IndexingStage(
        config=Config(), boto_session=mocker.MagicMock(), doc_status=store
    )


def _document(run_id: str, path: str) -> Document:
    return Document(
        page_content="synthetic text",
        document_id=run_id,
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
        file_type="txt",
        total_pages=1,
    )


def _context(delta: DocumentDelta | None, documents: list[Document]) -> PipelineContext:
    ctx = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory="/corpus",
        documents=documents,
    )
    ctx.incremental_delta = delta
    ctx.incremental_fingerprints = {compute_doc_id(_CHANGED): "hash-v2"}
    return ctx


def _seed_registry(store: FakeDocStatusStore, suffix: str = "default") -> None:
    store.put(
        DocStatusRecord(
            doc_id=compute_doc_id(_CHANGED),
            content_hash="hash-v1",
            status=DocStatus.PROCESSED,
            suffix=suffix,
            entity_ids=["e-stale"],
            text_unit_ids=["t-old"],
        )
    )
    store.put(
        DocStatusRecord(
            doc_id=compute_doc_id(_DELETED),
            content_hash="hash-gone",
            status=DocStatus.PROCESSED,
            suffix=suffix,
            entity_ids=["e-deleted"],
        )
    )


def _delta_inputs(attributes: dict[str, Any] | None = None):
    text_units = [
        TextUnit(
            id="t-new",
            text="Vendor supplies Buyer.",
            document_ids=["run-changed"],
            attributes=attributes,
        )
    ]
    entities = [Entity(id="e-new", name="Vendor", type="ORG", text_unit_ids=["t-new"])]
    return text_units, entities


def _run(stage, ctx, text_units, entities) -> dict[str, IndexingStats]:
    return stage._index_incremental(
        ctx,
        text_units=text_units,
        entities=entities,
        relationships=[],
        communities=[],
        community_reports=[],
        claims=[],
    )


def _full_delta() -> DocumentDelta:
    return DocumentDelta(
        changed=[compute_doc_id(_CHANGED)], deleted=[compute_doc_id(_DELETED)]
    )


def test_prunes_changed_then_removes_deleted_then_commits(mocker) -> None:
    store = FakeDocStatusStore()
    _seed_registry(store)
    manager = _RecordingManager()
    stage = _stage(mocker, store, manager)
    text_units, entities = _delta_inputs()
    ctx = _context(_full_delta(), [_document("run-changed", _CHANGED)])

    results = _run(stage, ctx, text_units, entities)

    assert [name for name, _ in manager.calls] == [
        "delete_documents",  # prune the changed doc's stale artifacts
        "delete_documents",  # remove the deleted doc's artifacts
        "index_delta",  # upsert the freshly extracted delta
    ]
    assert manager.calls[0][1] == {"default": ["e-stale", "t-old"]}
    assert manager.calls[1][1] == {"default": ["e-deleted"]}
    assert results["entities"].successful_items == 1

    # Registry: changed doc re-recorded with the new hash and lineage; deleted
    # doc's row gone only after its artifacts were removed.
    changed = store.get(compute_doc_id(_CHANGED))
    assert changed is not None
    assert changed.content_hash == "hash-v2"
    assert changed.entity_ids == ["e-new"]
    assert changed.text_unit_ids == ["t-new"]
    assert store.get(compute_doc_id(_DELETED)) is None


def test_suffix_is_taken_from_first_text_unit(mocker) -> None:
    store = FakeDocStatusStore()
    _seed_registry(store, suffix="tenant1")
    manager = _RecordingManager()
    stage = _stage(mocker, store, manager)
    text_units, entities = _delta_inputs(attributes={"index": "tenant1"})
    ctx = _context(_full_delta(), [_document("run-changed", _CHANGED)])

    _run(stage, ctx, text_units, entities)

    changed = store.get(compute_doc_id(_CHANGED))
    assert changed is not None and changed.suffix == "tenant1"
    assert all(
        set(ids_by_suffix) == {"tenant1"}
        for name, ids_by_suffix in manager.calls
        if name == "delete_documents"
    )


def test_suffix_defaults_when_there_are_no_text_units(mocker) -> None:
    store = FakeDocStatusStore()
    manager = _RecordingManager()
    stage = _stage(mocker, store, manager)
    # A delta of only deletions extracts nothing; lineage still needs a suffix.
    ctx = _context(DocumentDelta(), [_document("run-changed", _CHANGED)])

    _run(stage, ctx, text_units=[], entities=[])

    changed = store.get(compute_doc_id(_CHANGED))
    assert changed is not None and changed.suffix == "default"


def test_missing_delta_falls_back_to_full_indexing(mocker) -> None:
    manager = _RecordingManager()
    stage = _stage(mocker, FakeDocStatusStore(), manager)
    build_store = mocker.patch.object(stage, "_build_doc_status_store")
    text_units, entities = _delta_inputs()

    _run(stage, _context(None, []), text_units, entities)

    assert [name for name, _ in manager.calls] == ["index_all_data"]
    assert manager.calls[0][1]["entities"] == entities
    build_store.assert_not_called()


def test_failed_deleted_doc_removal_keeps_its_row_and_fails_the_stage(
    mocker,
) -> None:
    store = FakeDocStatusStore()
    _seed_registry(store)
    manager = _RecordingManager(delete_failures=1)
    stage = _stage(mocker, store, manager)
    text_units, entities = _delta_inputs()
    ctx = _context(
        DocumentDelta(deleted=[compute_doc_id(_DELETED)]),
        [_document("run-changed", _CHANGED)],
    )

    with pytest.raises(PipelineStageError, match="deleted documents"):
        _run(stage, ctx, text_units, entities)

    # Removal reported a failure, so the row stays for a retry next run instead
    # of orphaning still-live artifacts, and the stage fails so the run (and
    # the IndexingFailures alarm) reports it. The delta itself was still
    # committed: a failed deletion does not hold back new documents.
    assert store.get(compute_doc_id(_DELETED)) is not None
    assert [name for name, _ in manager.calls][-1] == "index_delta"
    changed = store.get(compute_doc_id(_CHANGED))
    assert changed is not None and changed.content_hash == "hash-v2"


def test_failed_prune_does_not_record_changed_doc_as_processed(mocker) -> None:
    store = FakeDocStatusStore()
    _seed_registry(store)
    stage = _stage(mocker, store, _RecordingManager(delete_failures=1))
    text_units, entities = _delta_inputs()
    ctx = _context(
        DocumentDelta(changed=[compute_doc_id(_CHANGED)]),
        [_document("run-changed", _CHANGED)],
    )

    try:
        _run(stage, ctx, text_units, entities)
    except Exception:  # noqa: BLE001 - failing the run is an acceptable fix
        pass

    # Stale artifacts may still be live, so the old lineage must be kept (or
    # the doc retried) rather than overwritten with the new hash.
    record = store.get(compute_doc_id(_CHANGED))
    assert record is not None and record.content_hash == "hash-v1"


def test_doc_with_failed_extraction_is_recorded_failed_and_retried(mocker) -> None:
    store = FakeDocStatusStore()
    _seed_registry(store)
    stage = _stage(mocker, store, _RecordingManager())
    text_units, entities = _delta_inputs()
    ctx = _context(
        DocumentDelta(changed=[compute_doc_id(_CHANGED)]),
        [_document("run-changed", _CHANGED)],
    )
    ctx.failed_text_unit_ids = {"graph_extraction": ["t-new"]}

    _run(stage, ctx, text_units, entities)

    # The lineage of what WAS written is kept (so the next prune removes it),
    # but the doc is FAILED, so the next run re-extracts it although its
    # content hash is unchanged.
    record = store.get(compute_doc_id(_CHANGED))
    assert record is not None
    assert record.status is DocStatus.FAILED
    assert record.content_hash == "hash-v2"
    assert record.entity_ids == ["e-new"]
    assert store.diff({compute_doc_id(_CHANGED): "hash-v2"}).changed == [
        compute_doc_id(_CHANGED)
    ]
