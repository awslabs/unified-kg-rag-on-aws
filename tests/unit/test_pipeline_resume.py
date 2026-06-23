# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Resume-strategy unit tests for PipelineResumeManager.

These cover the phased-execution handoff used by the Step Functions ingestion
pipeline: each phase runs in its own Fargate task and resumes at a stage that
the *previous* phase did not record, so the upstream completed stages must still
be restored from the (S3-synced) cache.
"""

from datetime import datetime
from pathlib import Path

import pytest

from aws_graphrag.domain.models import (
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
)
from aws_graphrag.shared.pipeline_manager import (
    PipelineResumeManager,
    PipelineStateManager,
)

pytestmark = pytest.mark.unit


def _results(*names_statuses: tuple[str, str]) -> list[dict]:
    return [{"stage_name": n, "status": s} for n, s in names_statuses]


class _StubCacheManager:
    """Minimal cache manager exposing only get_pipeline_cache_dir for state tests."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def get_pipeline_cache_dir(self, pipeline_id: str) -> Path:
        path = self._root / pipeline_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def _stage_result(name: str, status: str = "completed") -> PipelineStageResult:
    return PipelineStageResult(
        stage_name=name,
        status=PipelineStageStatus(status),
        start_time=datetime(2026, 1, 1),
    )


def _context(pipeline_id: str, results: list[PipelineStageResult]) -> PipelineContext:
    ctx = PipelineContext(
        pipeline_id=pipeline_id,
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=Path("/tmp/src"),
    )
    ctx.stage_results = results
    return ctx


def test_explicit_resume_loads_all_upstream_when_target_not_recorded() -> None:
    # Prep phase recorded only its own stages; GraphBuild phase resumes at
    # graph_extraction, which is NOT in the recorded history. All recorded
    # completed stages are upstream and must be restored (the bug: returned []).
    stage_results = _results(
        ("document_parsing", "completed"),
        ("document_loading", "completed"),
        ("text_chunking", "completed"),
        ("translation", "completed"),
    )

    resume, completed = PipelineResumeManager._handle_explicit_stage_resume(
        stage_results, "graph_extraction"
    )

    assert resume == "graph_extraction"
    assert completed == [
        "document_parsing",
        "document_loading",
        "text_chunking",
        "translation",
    ]


def test_explicit_resume_skips_failed_upstream_stages() -> None:
    stage_results = _results(
        ("document_parsing", "completed"),
        ("document_loading", "failed"),
    )

    resume, completed = PipelineResumeManager._handle_explicit_stage_resume(
        stage_results, "graph_extraction"
    )

    assert resume == "graph_extraction"
    assert completed == ["document_parsing"]


def test_explicit_resume_within_recorded_history_loads_only_prior() -> None:
    # When the target IS recorded, only stages before it count as completed.
    stage_results = _results(
        ("document_parsing", "completed"),
        ("text_chunking", "completed"),
        ("translation", "failed"),
    )

    resume, completed = PipelineResumeManager._handle_explicit_stage_resume(
        stage_results, "text_chunking"
    )

    assert resume == "text_chunking"
    assert completed == ["document_parsing"]


def test_explicit_resume_empty_history() -> None:
    resume, completed = PipelineResumeManager._handle_explicit_stage_resume(
        [], "graph_extraction"
    )
    assert resume == "graph_extraction"
    assert completed == []


def test_save_metadata_accumulates_stage_results_across_phases(tmp_path) -> None:
    # The core fix: a later phase saving only its own stage_results must NOT
    # clobber the earlier phases' history. This is what caused docparser-007 to
    # index entities/communities but silently drop relationships and claims.
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    pid = "phased-run"

    # GraphBuild + Analysis phases recorded the upstream stages.
    upstream = [
        _stage_result(n)
        for n in (
            "graph_resolution",  # produces resolved_relationships
            "claim_resolution",  # produces resolved_claims
            "graph_analysis",  # produces resolved_entities
            "community_detection",
        )
    ]
    state.save_pipeline_metadata(_context(pid, upstream))

    # Index phase runs in its own task and only knows about "indexing".
    state.save_pipeline_metadata(_context(pid, [_stage_result("indexing")]))

    reloaded = state.load_pipeline_metadata(pid)
    names = [r["stage_name"] for r in reloaded["stage_results"]]

    # All upstream stages survive the Index phase's save, plus indexing itself.
    assert names == [
        "graph_resolution",
        "claim_resolution",
        "graph_analysis",
        "community_detection",
        "indexing",
    ]


def test_save_metadata_in_memory_result_wins_for_same_stage(tmp_path) -> None:
    # A re-run of a stage in a later phase should overwrite the persisted entry
    # for that stage rather than duplicate it.
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    pid = "rerun"

    state.save_pipeline_metadata(_context(pid, [_stage_result("gleaning", "failed")]))
    state.save_pipeline_metadata(
        _context(pid, [_stage_result("gleaning", "completed")])
    )

    reloaded = state.load_pipeline_metadata(pid)
    gleaning = [r for r in reloaded["stage_results"] if r["stage_name"] == "gleaning"]
    assert len(gleaning) == 1
    assert gleaning[0]["status"] == "completed"


def test_prepare_context_for_resume_keeps_upstream_in_narrow_phase() -> None:
    # Defense-in-depth for the same bug: when the Index phase narrows self.stages
    # to just ["indexing"], _prepare_context_for_resume must judge upstream-ness
    # against the CANONICAL order, not the narrow window, or it wipes everything.
    from aws_graphrag.application.ingestion.pipeline import DataIngestionPipeline

    ctx = _context(
        "phased",
        [
            _stage_result("graph_resolution"),
            _stage_result("claim_resolution"),
            _stage_result("graph_analysis"),
            _stage_result("community_detection"),
        ],
    )

    # Stub self: only STAGE_CLASSES is consulted now (no self.stages).
    class _Stub:
        STAGE_CLASSES = DataIngestionPipeline.STAGE_CLASSES

    DataIngestionPipeline._prepare_context_for_resume(_Stub(), ctx, "indexing")

    names = [r.stage_name for r in ctx.stage_results]
    assert names == [
        "graph_resolution",
        "claim_resolution",
        "graph_analysis",
        "community_detection",
    ]


def test_prepare_context_for_resume_drops_results_at_or_after_start() -> None:
    from aws_graphrag.application.ingestion.pipeline import DataIngestionPipeline

    ctx = _context(
        "resume",
        [
            _stage_result("document_parsing"),
            _stage_result("graph_extraction"),
            _stage_result("graph_resolution"),
        ],
    )

    class _Stub:
        STAGE_CLASSES = DataIngestionPipeline.STAGE_CLASSES

    # Resuming at graph_extraction keeps only strictly-upstream results.
    DataIngestionPipeline._prepare_context_for_resume(_Stub(), ctx, "graph_extraction")
    assert [r.stage_name for r in ctx.stage_results] == ["document_parsing"]


def test_save_metadata_sorts_into_canonical_order(tmp_path) -> None:
    # Even if phases write out of order, the merged history reads in pipeline order.
    state = PipelineStateManager(_StubCacheManager(tmp_path))
    pid = "ordering"

    state.save_pipeline_metadata(_context(pid, [_stage_result("indexing")]))
    state.save_pipeline_metadata(
        _context(pid, [_stage_result("translation"), _stage_result("graph_extraction")])
    )

    reloaded = state.load_pipeline_metadata(pid)
    names = [r["stage_name"] for r in reloaded["stage_results"]]
    assert names == ["translation", "graph_extraction", "indexing"]
