# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for PipelineStateManager / PipelineResumeManager.

Complements ``test_pipeline_resume`` (which focuses on the phased-execution
resume-strategy + cross-phase save accumulation) by covering the load/save
round-trip, serialization-field exclusion, context reconstruction, the
verify/repair surface, auto-resume strategy, and the cache-backed
context-restoration / integrity-validation paths. A stub cache manager + tmp
filesystem stand in for the real S3/local cache; no AWS or network is touched.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from unified_kg_rag.domain.models import (
    Entity,
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
)
from unified_kg_rag.shared.exceptions import (
    PipelineResumeError,
    PipelineStateError,
)
from unified_kg_rag.shared.pipeline_manager import (
    PipelineResumeManager,
    PipelineStateManager,
)

pytestmark = pytest.mark.unit


# --- stubs / helpers -----------------------------------------------------


class _StubCacheManager:
    """Cache manager exposing the small surface the managers consume.

    ``get_pipeline_cache_dir`` is enough for the state manager; the resume
    manager additionally calls ``load_stage_result`` and ``cache_exists`` /
    ``cache_exists``, which we back with an in-memory dict keyed by cache_key.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        # cache_key -> data (list of model instances or raw)
        self.store: dict[str, Any] = {}

    def get_pipeline_cache_dir(self, pipeline_id: str) -> Path:
        path = self._root / pipeline_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_stage_result(
        self, cache_key: str, pipeline_id: str, data_type: Any = None
    ) -> Any:
        return self.store.get(cache_key)

    def cache_exists(self, cache_key: str, pipeline_id: str) -> bool:
        return cache_key in self.store


def _stage_result(name: str, status: str = "completed") -> PipelineStageResult:
    return PipelineStageResult(
        stage_name=name,
        status=PipelineStageStatus(status),
        start_time=datetime(2026, 1, 1),
    )


def _context(
    pipeline_id: str, results: list[PipelineStageResult] | None = None
) -> PipelineContext:
    ctx = PipelineContext(
        pipeline_id=pipeline_id,
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=Path("/tmp/src"),
    )
    ctx.stage_results = results or []
    return ctx


# --- pipeline_exists / load round-trip -----------------------------------


def test_pipeline_exists_false_when_no_metadata(tmp_path) -> None:
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    assert state.pipeline_exists("never-saved") is False


def test_pipeline_exists_true_after_save(tmp_path) -> None:
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    state.save_pipeline_metadata(_context("pid", [_stage_result("indexing")]))
    assert state.pipeline_exists("pid") is True


def test_save_excludes_heavy_data_fields(tmp_path) -> None:
    # Heavy graph payloads must never be serialized into the metadata file.
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    ctx = _context("pid", [_stage_result("graph_extraction")])
    ctx.entities = [Entity(id="e1", name="A"), Entity(id="e2", name="B")]
    ctx.relationships = []
    state.save_pipeline_metadata(ctx)

    raw = state.load_pipeline_metadata("pid")
    for excluded in PipelineStateManager.EXCLUDED_DATA_FIELDS:
        assert excluded not in raw


def test_save_sets_end_time_on_completion(tmp_path) -> None:
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    ctx = _context("pid", [_stage_result("indexing", "completed")])
    state.save_pipeline_metadata(ctx)
    # status derives to COMPLETED -> end_time stamped.
    assert ctx.status == PipelineStageStatus.COMPLETED
    assert ctx.end_time is not None


# --- load_pipeline_metadata error branches -------------------------------


def test_load_missing_file_raises(tmp_path) -> None:
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    with pytest.raises(PipelineStateError, match="not found"):
        state.load_pipeline_metadata("absent")


def test_load_empty_file_raises(tmp_path) -> None:
    cache = _StubCacheManager(tmp_path)
    state = PipelineStateManager(cache)
    path = (
        cache.get_pipeline_cache_dir("empty")
        / PipelineStateManager.PIPELINE_METADATA_FILE
    )
    path.write_text("")
    with pytest.raises(PipelineStateError, match="empty"):
        state.load_pipeline_metadata("empty")


def test_load_invalid_json_raises(tmp_path) -> None:
    cache = _StubCacheManager(tmp_path)
    state = PipelineStateManager(cache)
    path = (
        cache.get_pipeline_cache_dir("bad")
        / PipelineStateManager.PIPELINE_METADATA_FILE
    )
    path.write_text("{not json")
    with pytest.raises(PipelineStateError, match="invalid JSON"):
        state.load_pipeline_metadata("bad")


def test_load_non_object_json_raises(tmp_path) -> None:
    cache = _StubCacheManager(tmp_path)
    state = PipelineStateManager(cache)
    path = (
        cache.get_pipeline_cache_dir("arr")
        / PipelineStateManager.PIPELINE_METADATA_FILE
    )
    path.write_text("[1, 2, 3]")
    with pytest.raises(PipelineStateError, match="JSON object"):
        state.load_pipeline_metadata("arr")


def test_load_missing_required_field_raises(tmp_path) -> None:
    cache = _StubCacheManager(tmp_path)
    state = PipelineStateManager(cache)
    path = (
        cache.get_pipeline_cache_dir("partial")
        / PipelineStateManager.PIPELINE_METADATA_FILE
    )
    path.write_text(json.dumps({"pipeline_id": "x"}))  # missing start_time/status
    with pytest.raises(PipelineStateError, match="required fields"):
        state.load_pipeline_metadata("partial")


# --- create_context_from_metadata ----------------------------------------


def test_create_context_from_metadata_drops_excluded_and_forces_running(
    tmp_path,
) -> None:
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    metadata = {
        "pipeline_id": "pid",
        "start_time": "2026-01-01T00:00:00",
        "status": "completed",
        "config": {},
        "source_directory": "/tmp/src",
        # Heavy field that must be filtered out before model_validate.
        "entities": [{"id": "e1", "name": "A"}],
        "stage_results": [
            {
                "stage_name": "indexing",
                "status": "completed",
                "start_time": "2026-01-01T00:00:00",
            }
        ],
    }
    ctx = state.create_context_from_metadata(metadata)
    assert ctx.pipeline_id == "pid"
    # Forced to RUNNING regardless of persisted status.
    assert ctx.status == PipelineStageStatus.RUNNING
    # Excluded data not hydrated onto the context.
    assert not ctx.entities


# --- verify / repair ------------------------------------------------------


def test_verify_pipeline_metadata_roundtrip(tmp_path) -> None:
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    assert state.verify_pipeline_metadata("missing") is False
    state.save_pipeline_metadata(_context("pid", [_stage_result("indexing")]))
    assert state.verify_pipeline_metadata("pid") is True


def test_repair_pipeline_metadata_saves_and_verifies(tmp_path) -> None:
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    ctx = _context("pid", [_stage_result("indexing")])
    assert state.repair_pipeline_metadata(ctx) is True
    assert state.verify_pipeline_metadata("pid") is True


# --- update_stage_result --------------------------------------------------


def test_update_stage_result_appends_new_stage(tmp_path) -> None:
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    ctx = _context("pid", [_stage_result("document_loading")])
    state.update_stage_result(ctx, _stage_result("text_chunking"))
    names = [r.stage_name for r in ctx.stage_results]
    assert "text_chunking" in names
    # Persisted too.
    reloaded = state.load_pipeline_metadata("pid")
    assert any(r["stage_name"] == "text_chunking" for r in reloaded["stage_results"])


def test_update_stage_result_replaces_existing_stage(tmp_path) -> None:
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    ctx = _context("pid", [_stage_result("gleaning", "failed")])
    state.update_stage_result(ctx, _stage_result("gleaning", "completed"))
    gleaning = [r for r in ctx.stage_results if r.stage_name == "gleaning"]
    assert len(gleaning) == 1
    assert gleaning[0].status == PipelineStageStatus.COMPLETED


# --- _update_context_status branches -------------------------------------


def test_update_context_status_pending_when_no_results() -> None:
    ctx = _context("pid", [])
    PipelineStateManager._update_context_status(ctx)
    assert ctx.status == PipelineStageStatus.PENDING


def test_update_context_status_failed_takes_priority() -> None:
    ctx = _context(
        "pid", [_stage_result("a", "completed"), _stage_result("b", "failed")]
    )
    PipelineStateManager._update_context_status(ctx)
    assert ctx.status == PipelineStageStatus.FAILED


def test_update_context_status_running_when_mixed_incomplete() -> None:
    ctx = _context(
        "pid", [_stage_result("a", "completed"), _stage_result("b", "pending")]
    )
    PipelineStateManager._update_context_status(ctx)
    assert ctx.status == PipelineStageStatus.RUNNING


# --- PipelineResumeManager.determine_resume_strategy (auto) --------------


def test_determine_resume_strategy_auto_resumes_at_first_incomplete(
    tmp_path,
) -> None:
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    ctx = _context(
        "pid",
        [
            _stage_result("document_loading", "completed"),
            _stage_result("text_chunking", "failed"),
        ],
    )
    state.save_pipeline_metadata(ctx)
    resume = PipelineResumeManager(state)

    start, completed = resume.determine_resume_strategy("pid")
    assert start == "text_chunking"
    assert completed == ["document_loading"]


def test_determine_resume_strategy_explicit_stage(tmp_path) -> None:
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    ctx = _context(
        "pid",
        [
            _stage_result("document_parsing", "completed"),
            _stage_result("document_loading", "completed"),
        ],
    )
    state.save_pipeline_metadata(ctx)
    resume = PipelineResumeManager(state)

    start, completed = resume.determine_resume_strategy(
        "pid", explicit_stage="graph_extraction"
    )
    assert start == "graph_extraction"
    assert completed == ["document_parsing", "document_loading"]


def test_determine_resume_strategy_wraps_load_error(tmp_path) -> None:
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    resume = PipelineResumeManager(state)
    with pytest.raises(PipelineResumeError):
        resume.determine_resume_strategy("does-not-exist")


def test_handle_auto_resume_all_completed_resumes_at_last() -> None:
    results = [
        {"stage_name": "document_loading", "status": "completed"},
        {"stage_name": "indexing", "status": "completed"},
    ]
    start, completed = PipelineResumeManager._handle_auto_resume(results)
    assert start == "indexing"
    assert completed == ["document_loading", "indexing"]


def test_handle_auto_resume_empty_defaults_to_document_loading() -> None:
    start, completed = PipelineResumeManager._handle_auto_resume([])
    assert start == "document_loading"
    assert completed == []


# --- restore_pipeline_context + cache loading ----------------------------


def test_restore_pipeline_context_loads_cached_stage_data(tmp_path) -> None:
    cache = _StubCacheManager(tmp_path)
    state = PipelineStateManager(cache)
    ctx = _context(
        "pid",
        [
            _stage_result("graph_extraction", "completed"),
        ],
    )
    state.save_pipeline_metadata(ctx)

    # Seed the cache with what graph_extraction would have produced.
    cache.store["entities"] = [Entity(id="e1", name="A")]
    cache.store["relationships"] = []

    resume = PipelineResumeManager(state)
    restored = resume.restore_pipeline_context("pid", ["graph_extraction"])
    assert [e.id for e in restored.entities] == ["e1"]


def test_restore_pipeline_context_tolerates_unknown_stage(tmp_path) -> None:
    # A stage name with no STAGE_CACHE_MAPPING entry / not a PipelineStageType
    # is treated as "nothing to load" and does not break restoration.
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    state.save_pipeline_metadata(_context("pid", [_stage_result("document_parsing")]))
    resume = PipelineResumeManager(state)
    restored = resume.restore_pipeline_context(
        "pid", ["document_parsing", "totally_made_up_stage"]
    )
    assert restored.pipeline_id == "pid"


def test_restore_pipeline_context_wraps_failure() -> None:
    class _Boom:
        def get_pipeline_cache_dir(self, pipeline_id: str):  # noqa: ANN001
            raise RuntimeError("disk gone")

    state = PipelineStateManager(_Boom())
    resume = PipelineResumeManager(state)
    with pytest.raises(PipelineResumeError):
        resume.restore_pipeline_context("pid", ["graph_extraction"])


def test_load_stage_cache_data_returns_false_when_cache_empty(tmp_path) -> None:
    # Mapped stage but no cached payload -> reports nothing loaded.
    cache = _StubCacheManager(tmp_path)
    state = PipelineStateManager(cache)
    resume = PipelineResumeManager(state)
    ctx = _context("pid")
    loaded = resume._load_stage_cache_data("pid", "graph_extraction", ctx)
    assert loaded is False


# --- validate_pipeline_integrity -----------------------------------------


def test_validate_integrity_passes_when_all_cache_present(tmp_path) -> None:
    cache = _StubCacheManager(tmp_path)
    state = PipelineStateManager(cache)
    state.save_pipeline_metadata(
        _context("pid", [_stage_result("graph_extraction", "completed")])
    )
    # graph_extraction maps to context attrs "entities" + "relationships".
    cache.store["entities"] = [Entity(id="e1", name="A")]
    cache.store["relationships"] = []
    resume = PipelineResumeManager(state)

    ok, errors = resume.validate_pipeline_integrity("pid")
    assert ok is True
    assert errors == []


def test_validate_integrity_reports_missing_cache(tmp_path) -> None:
    cache = _StubCacheManager(tmp_path)
    state = PipelineStateManager(cache)
    state.save_pipeline_metadata(
        _context("pid", [_stage_result("graph_extraction", "completed")])
    )
    # No cache seeded -> both keys missing.
    resume = PipelineResumeManager(state)

    ok, errors = resume.validate_pipeline_integrity("pid")
    assert ok is False
    assert any("entities" in e for e in errors)
    assert any("relationships" in e for e in errors)


def test_validate_integrity_returns_error_when_metadata_unloadable(
    tmp_path,
) -> None:
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    resume = PipelineResumeManager(state)
    ok, errors = resume.validate_pipeline_integrity("missing")
    assert ok is False
    assert len(errors) == 1


def test_validate_integrity_skips_non_completed_and_unmapped(tmp_path) -> None:
    cache = _StubCacheManager(tmp_path)
    state = PipelineStateManager(cache)
    state.save_pipeline_metadata(
        _context(
            "pid",
            [
                # Failed stage -> not checked.
                _stage_result("graph_extraction", "failed"),
                # Completed but no cache mapping -> nothing to check.
                _stage_result("document_parsing", "completed"),
            ],
        )
    )
    resume = PipelineResumeManager(state)
    ok, errors = resume.validate_pipeline_integrity("pid")
    assert ok is True
    assert errors == []
