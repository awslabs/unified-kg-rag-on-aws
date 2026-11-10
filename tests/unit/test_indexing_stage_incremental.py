# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""IndexingStage._index_incremental: the write path of an incremental run.

Pins the contract between the stage and the incremental orchestrator, which is
what keeps the live stores and the doc-status registry consistent:

* stale artifacts of changed and deleted docs are removed in one pass, the
  deleted docs' registry rows dropped, and only then is the delta upserted +
  recorded;
* the run's index suffix comes from the first text unit (the configured
  ``index_value``, else default, otherwise);
* a missing delta raises without touching the stores or the registry.

AWS-free: an in-memory registry plus a recording indexing manager.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any

import pytest

from tests.fixtures.doc_status_records import all_records
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.storage.indexing_manager import CrossRunMergeResult
from unified_kg_rag.application.storage.pending_writes import NotedWrites
from unified_kg_rag.domain.ingestion.cache_keys import extraction_fingerprint
from unified_kg_rag.domain.ingestion.delta_detector import compute_doc_id
from unified_kg_rag.domain.models import (
    PENDING_CONTENT_HASH,
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
from unified_kg_rag.shared import ConfigurationError, PipelineStageError

pytestmark = pytest.mark.unit

_CHANGED = "/corpus/changed.txt"
_DELETED = "/corpus/deleted.txt"


class _RecordingManager:
    """IndexingManager stand-in that records the order of store operations."""

    def __init__(self, delete_failures: int = 0) -> None:
        self.config = Config()
        self.calls: list[tuple[str, Any]] = []
        self._delete_failures = delete_failures
        self.outdated_reports: set[str] = set()
        self.deleted_communities: dict[str, set[str]] = {}
        # The graph still holds what the registry lists.
        self.graph_indexer = SimpleNamespace(
            find_ids_citing=lambda ids, suffix=None: {"e": list(ids)}
        )
        # The live indices were built for this run's settings.
        self.vector_indexer = SimpleNamespace(
            check_index_settings=lambda suffix=None: None,
            set_doc_status_tracking=lambda tracked: None,
            set_replaced_vector_keeping=lambda keep: None,
            record_build=lambda record, suffix: None,
            read_build_record=lambda suffix: None,
            read_text_unit_ids=lambda ids, suffix=None: ids,
        )

    def index_all_data(self, **kwargs: Any) -> dict[str, IndexingStats]:
        self.calls.append(("index_all_data", kwargs))
        return {}

    def remove_text_units(
        self, suffix: str, text_unit_ids: list[str]
    ) -> dict[str, IndexingStats]:
        self.calls.append(("remove_text_units", {suffix: text_unit_ids}))
        removed = len(text_unit_ids)
        return {
            "store": IndexingStats(
                total_items=removed,
                successful_items=removed - self._delete_failures,
                failed_items=self._delete_failures,
            )
        }

    def delete_retired_ids(self) -> dict[str, IndexingStats]:
        return {}

    def discard_retired_ids(self) -> None:
        return None

    def reset_run_state(self) -> None:
        return None

    def offer_summaries(self, entities: Any) -> None:
        return None

    def expect_delta(self, entities: Any, relationships: Any) -> None:
        return None

    def collect_extraction_records(self) -> dict[str, IndexingStats]:
        return {}

    def merge_with_existing_graph(self, entities, relationships, text_units=()):
        return CrossRunMergeResult(entities=entities, relationships=relationships)

    def read_pending_writes(self, suffixes, text_units=()):  # noqa: ANN001, ANN201
        return NotedWrites()

    def note_pending_writes(  # noqa: ANN001
        self, results, written, *, recorded=True
    ) -> set[str]:
        return set()

    def merge_with_stored_claims(self, claims, text_units=()):
        return claims

    def note_pending_commit(self) -> None:
        return None

    def clear_pending_commit(self) -> None:
        return None

    def clear_stripped_summaries(self) -> None:
        return None

    def index_delta(self, **kwargs: Any) -> dict[str, IndexingStats]:
        self.calls.append(("index_delta", kwargs))
        written = len(kwargs.get("entities") or [])
        return {
            "entities": IndexingStats(total_items=written, successful_items=written)
        }


def _stage(mocker, store: FakeDocStatusStore, manager: _RecordingManager):
    mocker.patch.object(ps, "IndexingManager", return_value=manager)
    return ps.IndexingStage(config=Config(), doc_status=store)


def _document(run_id: str, path: str) -> Document:
    return Document(
        page_content="synthetic text",
        document_id=run_id,
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
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
            text_unit_ids=["t-old"],
        )
    )
    store.put(
        DocStatusRecord(
            doc_id=compute_doc_id(_DELETED),
            content_hash="hash-gone",
            status=DocStatus.PROCESSED,
            suffix=suffix,
            text_unit_ids=["t-gone"],
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
    results, _drift = stage._index_incremental(
        ctx,
        text_units=text_units,
        entities=entities,
        relationships=[],
        communities=[],
        community_reports=[],
        claims=[],
    )
    return results


def _full_delta() -> DocumentDelta:
    return DocumentDelta(
        changed=[compute_doc_id(_CHANGED)], deleted=[compute_doc_id(_DELETED)]
    )


def test_removes_changed_and_deleted_in_one_pass_then_commits(mocker) -> None:
    store = FakeDocStatusStore()
    _seed_registry(store)
    manager = _RecordingManager()
    stage = _stage(mocker, store, manager)
    text_units, entities = _delta_inputs()
    ctx = _context(_full_delta(), [_document("run-changed", _CHANGED)])

    results = _run(stage, ctx, text_units, entities)

    assert [name for name, _ in manager.calls] == [
        "remove_text_units",  # stale text units of the changed + deleted docs
        "index_delta",  # upsert the freshly extracted delta
    ]
    assert manager.calls[0][1] == {"default": ["t-gone", "t-old"]}
    assert results["entities"].successful_items == 1

    # Registry: changed doc re-recorded with the new hash and lineage; deleted
    # doc's row gone only after its artifacts were removed.
    changed = store.get(compute_doc_id(_CHANGED))
    assert changed is not None
    assert changed.content_hash == "hash-v2"
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
        if name == "remove_text_units"
    )


def test_embedding_mismatch_stops_the_run_before_any_write(mocker) -> None:
    store = FakeDocStatusStore()
    _seed_registry(store, suffix="tenant1")
    before = all_records(store)
    manager = _RecordingManager()
    checked: list[str | None] = []

    def mismatch(suffix: str | None = None) -> None:
        checked.append(suffix)
        raise ConfigurationError("embedded with another model")

    manager.vector_indexer = SimpleNamespace(check_index_settings=mismatch)
    stage = _stage(mocker, store, manager)
    text_units, entities = _delta_inputs(attributes={"index": "tenant1"})
    ctx = _context(_full_delta(), [_document("run-changed", _CHANGED)])

    with pytest.raises(ConfigurationError, match="another model"):
        _run(stage, ctx, text_units, entities)

    # Checked on the run's suffix; nothing written to the stores or registry.
    assert checked == ["tenant1"]
    assert manager.calls == []
    assert all_records(store) == before


def test_suffix_defaults_when_there_are_no_text_units(mocker) -> None:
    store = FakeDocStatusStore()
    manager = _RecordingManager()
    stage = _stage(mocker, store, manager)
    # A delta of only deletions extracts nothing; lineage still needs a suffix.
    ctx = _context(DocumentDelta(), [_document("run-changed", _CHANGED)])

    _run(stage, ctx, text_units=[], entities=[])

    changed = store.get(compute_doc_id(_CHANGED))
    assert changed is not None and changed.suffix == "default"


def test_missing_delta_raises_instead_of_replacing_the_stores(mocker) -> None:
    # A full replace with only this run's documents would wipe the rest.
    manager = _RecordingManager()
    stage = _stage(mocker, FakeDocStatusStore(), manager)
    build_store = mocker.patch.object(stage, "_build_doc_status_store")
    text_units, entities = _delta_inputs()

    with pytest.raises(PipelineStageError, match="delta"):
        _run(stage, _context(None, []), text_units, entities)

    assert manager.calls == []
    build_store.assert_not_called()


def test_deletion_only_run_checks_the_configured_suffix(mocker) -> None:
    # Nothing is extracted, so the suffix cannot come from a text unit.
    store = FakeDocStatusStore()
    _seed_registry(store, suffix="tenant1")
    manager = _RecordingManager()
    manager.config.processing.document_parsing.index_value = "tenant1"
    checked: list[str | None] = []
    manager.vector_indexer = SimpleNamespace(
        check_index_settings=checked.append,
        set_doc_status_tracking=lambda tracked: None,
        set_replaced_vector_keeping=lambda keep: None,
        record_build=lambda record, suffix: None,
        read_text_unit_ids=lambda ids, suffix=None: ids,
    )
    stage = _stage(mocker, store, manager)
    ctx = _context(DocumentDelta(deleted=[compute_doc_id(_DELETED)]), [])

    _run(stage, ctx, text_units=[], entities=[])

    assert checked == ["tenant1"]
    assert store.get(compute_doc_id(_DELETED)) is None


def test_failed_deleted_doc_removal_keeps_its_row_and_commits_nothing(
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

    with pytest.raises(PipelineStageError, match="not committing"):
        _run(stage, ctx, text_units, entities)

    # Removal reported a failure, so the row stays for a retry next run instead
    # of orphaning still-live artifacts, and the stage fails so the run (and
    # the IndexingFailures alarm) reports it. The delta is not committed: a
    # failed strip leaves kept items citing the removed text units, and a
    # commit would merge the new documents into them (an edge it drops
    # against their stale counts is written by no later run). So a deleted
    # document whose removal keeps failing holds back new documents, as a
    # changed one does.
    assert store.get(compute_doc_id(_DELETED)) is not None
    assert "index_delta" not in [name for name, _ in manager.calls]
    # The run's document keeps its write-ahead record, so the next run
    # indexes it again.
    changed = store.get(compute_doc_id(_CHANGED))
    assert changed is not None and changed.status is DocStatus.PENDING


def test_failed_joint_removal_commits_nothing_and_keeps_both_rows(mocker) -> None:
    store = FakeDocStatusStore()
    _seed_registry(store)
    manager = _RecordingManager(delete_failures=1)
    stage = _stage(mocker, store, manager)
    text_units, entities = _delta_inputs()
    ctx = _context(_full_delta(), [_document("run-changed", _CHANGED)])

    with pytest.raises(PipelineStageError, match="not committing") as failed:
        _run(stage, ctx, text_units, entities)

    # The error names the documents whose removal failed.
    assert str(failed.value).endswith(
        ": " + ", ".join(sorted([_CHANGED, compute_doc_id(_DELETED)]))
    )
    # The changed doc keeps its old lineage (in its write-ahead PENDING
    # record) and the deleted doc its row, so the next run retries the whole
    # removal.
    assert "index_delta" not in [name for name, _ in manager.calls]
    _assert_pending_with_old_lineage(store)
    assert store.get(compute_doc_id(_DELETED)) is not None


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
    _assert_pending_with_old_lineage(store)


def _assert_pending_with_old_lineage(store: FakeDocStatusStore) -> None:
    """The changed doc's record is the write-ahead one: PENDING, keeping its
    old lineage (plus the planned one), and read as changed even if its
    content reverts to the indexed version."""
    record = store.get(compute_doc_id(_CHANGED))
    assert record is not None
    assert record.status is DocStatus.PENDING
    assert record.content_hash == PENDING_CONTENT_HASH
    assert {"t-old", "t-new"} <= set(record.text_unit_ids)
    assert store.diff({compute_doc_id(_CHANGED): "hash-v1"}).changed == [
        compute_doc_id(_CHANGED)
    ]


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
    assert record.text_unit_ids == ["t-new"]
    assert store.diff({compute_doc_id(_CHANGED): "hash-v2"}).changed == [
        compute_doc_id(_CHANGED)
    ]


@pytest.mark.parametrize(
    ("applied", "warned"),
    [
        ([], ["'/old/a/'", "'/old/b'"]),
        (["/old/a"], ["'/old/b'"]),
        (["/old/a/", "/old/b"], []),
    ],
)
def test_resumed_delta_without_retired_scopes_warns(
    mocker, caplog, applied: list[str], warned: list[str]
) -> None:
    """A run resumed after document_loading reuses the stored delta: retired
    scopes configured since are not applied, which must not pass silently."""
    manager = _RecordingManager()
    manager.initialize = lambda: True  # type: ignore[attr-defined]
    stage = _stage(mocker, FakeDocStatusStore(), manager)
    stage.config.indexing.retire_source_scopes = ["/old/a/", "/old/b"]
    index = mocker.patch.object(stage, "_index_incremental", return_value=({}, None))
    ctx = _context(DocumentDelta(), [])
    ctx.incremental_retired_scopes = applied

    with caplog.at_level("WARNING"):
        stage._execute_core(ctx)

    index.assert_called_once()
    messages = [
        r.getMessage()
        for r in caplog.records
        if "retire_source_scopes" in r.getMessage()
    ]
    if not warned:
        assert messages == []
        return
    assert len(messages) == 1
    for scope in warned:
        assert scope in messages[0]
    assert "--resume-from-stage document_loading" in messages[0]
    if applied:
        assert "'/old/a" not in messages[0]


@pytest.mark.parametrize("retire", [["/old/a"], []])
def test_resumed_context_saved_before_retired_scopes_warns_softly(
    mocker, caplog, retire: list[str]
) -> None:
    """A context saved before the delta recorded its retired scopes does not
    say which were applied: no "NOT applied" claim, only a hedged warning
    when retire scopes are configured."""
    manager = _RecordingManager()
    manager.initialize = lambda: True  # type: ignore[attr-defined]
    stage = _stage(mocker, FakeDocStatusStore(), manager)
    stage.config.indexing.retire_source_scopes = retire
    mocker.patch.object(stage, "_index_incremental", return_value=({}, None))
    saved = _context(DocumentDelta(), []).model_dump(mode="json")
    saved.pop("incremental_retired_scopes")
    ctx = PipelineContext.model_validate(saved)
    assert ctx.incremental_retired_scopes is None

    with caplog.at_level("WARNING"):
        stage._execute_core(ctx)

    messages = [
        r.getMessage()
        for r in caplog.records
        if "retire_source_scopes" in r.getMessage()
    ]
    if not retire:
        assert messages == []
        return
    assert len(messages) == 1
    assert "may not have been applied" in messages[0]
    assert "NOT applied" not in messages[0]
    assert "'/old/a'" in messages[0]


def test_delta_run_reports_its_community_drift(mocker) -> None:
    from unified_kg_rag.domain.ingestion.community_drift import CommunityDrift

    manager = _RecordingManager()
    manager.initialize = lambda: True  # type: ignore[attr-defined]
    stage = _stage(mocker, FakeDocStatusStore(), manager)
    drift = CommunityDrift(
        delta_communities=3,
        clustered_entities=4,
        overlapping_entities=2,
        stale_communities=1,
        added_roots=1,
        overlap_share=0.5,
        rebuild_recommended=True,
    )
    mocker.patch.object(stage, "_index_incremental", return_value=({}, drift))

    _in, _total, metrics = stage._execute_core(_context(DocumentDelta(), []))

    assert metrics["community_drift"] == drift.model_dump()
    assert metrics["community_drift_overlap_share"] == 0.5
    assert metrics["community_rebuild_recommended"] == 1


@pytest.mark.parametrize(
    ("delta", "mode"),
    [
        (DocumentDelta(changed=["x"]), "incremental"),
        (None, "full"),
        # The registry knew no document: the run built the suffix from nothing.
        (DocumentDelta(new=["x"]), "full"),
    ],
)
def test_records_the_build_on_each_written_suffix(
    mocker, delta: DocumentDelta | None, mode: str
) -> None:
    manager = _RecordingManager()
    recorded: list[tuple[Any, str | None]] = []
    manager.vector_indexer.record_build = lambda record, suffix: recorded.append(
        (record, suffix)
    )
    stage = _stage(mocker, FakeDocStatusStore(), manager)
    text_units = [
        TextUnit(id="t1", text="a", document_ids=["d1"], attributes={"index": "t1x"}),
        TextUnit(id="t2", text="b", document_ids=["d1", "d2"]),
    ]

    stage._record_build(_context(delta, []), text_units)

    assert [suffix for _, suffix in recorded] == ["default", "t1x"]
    record = recorded[0][0]
    assert (record.pipeline_id, record.mode, record.documents) == ("pid", mode, 2)
    assert record.extraction_fingerprint == extraction_fingerprint(stage.config)


def test_delete_only_run_records_its_configured_suffix(mocker) -> None:
    manager = _RecordingManager()
    manager.config.processing.document_parsing.index_value = "tenant1"
    recorded: list[str | None] = []
    manager.vector_indexer.record_build = lambda record, suffix: recorded.append(suffix)
    stage = _stage(mocker, FakeDocStatusStore(), manager)
    stage.config.processing.document_parsing.index_value = "tenant1"
    stage._record_build(_context(DocumentDelta(deleted=["x"]), []), [])
    assert recorded == ["tenant1"]


def test_a_run_starts_with_no_state_left_from_an_earlier_one(mocker) -> None:
    # A reused pipeline runs the same stage again: what an earlier run's
    # derivation decided (a failed commit's re-keys) must not carry over.
    store = FakeDocStatusStore()
    _seed_registry(store)
    manager = _RecordingManager()
    stage = _stage(mocker, store, manager)
    order: list[str] = []
    manager.initialize = lambda: None  # type: ignore[attr-defined]
    mocker.patch.object(
        manager, "reset_run_state", side_effect=lambda: order.append("reset")
    )
    mocker.patch.object(
        stage,
        "_index_incremental",
        side_effect=lambda *a, **k: order.append("index") or ({}, None),
    )

    stage._execute_core(_context(_full_delta(), [_document("run-changed", _CHANGED)]))

    assert order[:2] == ["reset", "index"]


def test_a_run_with_no_change_narrows_every_suffix_its_documents_hold(
    mocker,
) -> None:
    # A no-change run under fewer methodologies narrowed only the configured
    # suffix's build record: a document indexed under its own index value
    # kept its suffix serving strategies whose artifacts no run maintains.
    from tests.fixtures.fakes.stores import FakeVectorStore
    from unified_kg_rag.domain.models import IndexBuildRecord, Methodology

    store = FakeDocStatusStore()
    vectors = FakeVectorStore()
    manager = _RecordingManager()
    manager.vector_indexer = vectors
    config = Config()
    held = IndexBuildRecord(
        pipeline_id="run-1",
        mode="full",
        built_at=datetime(2026, 1, 1),
        documents=1,
        extraction_fingerprint=extraction_fingerprint(config),
    ).with_artifacts(ps.artifacts_to_build(config))
    for suffix in ("default", "tenant-b"):
        vectors.record_build(held, suffix)
    own = compute_doc_id("own.json", "tenant-b", "/corpus")
    store.put(
        DocStatusRecord(
            doc_id=own,
            file_path="own.json",
            content_hash="h",
            status=DocStatus.PROCESSED,
            suffix="tenant-b",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )
    mocker.patch.object(ps, "IndexingManager", return_value=manager)
    stage = ps.IndexingStage(config=config, doc_status=store)
    config.indexing.methodologies = [Methodology.LIGHTRAG]

    stage._record_removed_artifacts([], DocumentDelta(unchanged=[own]))

    for suffix in ("default", "tenant-b"):
        record = vectors.read_build_record(suffix)
        assert record is not None and "graphrag" not in record["methodologies"]


def test_a_run_with_no_change_reads_the_suffixes_from_the_diff(mocker) -> None:
    # The diff reads every unchanged document's suffix: the narrowing finds
    # the suffixes without reading the registry records again.
    from tests.fixtures.fakes.stores import FakeVectorStore
    from unified_kg_rag.domain.models import IndexBuildRecord, Methodology

    store = FakeDocStatusStore()
    vectors = FakeVectorStore()
    manager = _RecordingManager()
    manager.vector_indexer = vectors
    config = Config()
    held = IndexBuildRecord(
        pipeline_id="run-1",
        mode="full",
        built_at=datetime(2026, 1, 1),
        documents=1,
        extraction_fingerprint=extraction_fingerprint(config),
    ).with_artifacts(ps.artifacts_to_build(config))
    for suffix in ("default", "tenant-b"):
        vectors.record_build(held, suffix)
    own = compute_doc_id("own.json", "tenant-b", "/corpus")
    store.put(
        DocStatusRecord(
            doc_id=own,
            file_path="own.json",
            content_hash="h",
            status=DocStatus.PROCESSED,
            suffix="tenant-b",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )
    delta = store.diff({own: "h"})
    assert delta.unchanged_suffixes == ["tenant-b"]
    mocker.patch.object(ps, "IndexingManager", return_value=manager)
    stage = ps.IndexingStage(config=config, doc_status=store)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    reads = mocker.spy(store, "get_many")

    stage._record_removed_artifacts([], delta)

    assert reads.call_count == 0
    for suffix in ("default", "tenant-b"):
        record = vectors.read_build_record(suffix)
        assert record is not None and "graphrag" not in record["methodologies"]
