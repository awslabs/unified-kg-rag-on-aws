# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free smoke + branch tests for the rich console display helpers.

These functions are mostly formatters: their risk is in the branches (empty
vs. populated inputs, truncation, lookups, optional metrics), not in heavy
logic. We drive each renderer with representative inputs and capture the
console output, asserting key strings render and that early-return guards fire
without printing. The module-level ``console`` is captured via
``console.capture()`` so nothing actually hits a TTY.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Entity,
    PipelineContext,
    PipelineMetrics,
    PipelineStageResult,
    PipelineStageStatus,
    Relationship,
)
from unified_kg_rag.shared.utils import display

pytestmark = pytest.mark.unit


@pytest.fixture
def cap():
    """Capture everything printed to the module-level console."""

    def _capture(fn, *args, **kwargs):
        with display.console.capture() as capture:
            fn(*args, **kwargs)
        return capture.get()

    return _capture


# --- builders ------------------------------------------------------------


def _entity(i: str = "e1", name: str = "Alice") -> Entity:
    return Entity(id=i, name=name, type="PERSON", description="desc " * 40)


def _relationship() -> Relationship:
    return Relationship(
        id="r1",
        source_id="e1",
        target_id="e2",
        source_name="Alice",
        target_name="Bob",
        type="KNOWS",
        description="rel description " * 20,
    )


def _claim() -> Claim:
    return Claim(
        id="c1",
        subject_id="e1",
        subject_name="Alice",
        object_name="Bob",
        type="FACT",
        status="TRUE",
        description="claim description " * 20,
    )


def _community() -> Community:
    return Community(
        id="comm1",
        name="Community A",
        level="0",
        parent="",
        children=[],
        entity_ids=[f"e{i}" for i in range(8)],
        size=8,
    )


def _report() -> CommunityReport:
    return CommunityReport(
        id="cr1",
        name="Report A",
        community_id="comm1",
        summary="A long community summary. " * 20,
        rank=3.0,
    )


def _stage_result(
    name: str = "indexing",
    status: PipelineStageStatus = PipelineStageStatus.COMPLETED,
    duration: float | None = 1.5,
) -> PipelineStageResult:
    return PipelineStageResult(
        stage_name=name,
        status=status,
        start_time=datetime(2026, 1, 1),
        duration_seconds=duration,
        input_count=10,
        output_count=8,
    )


def _context() -> PipelineContext:
    ctx = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.COMPLETED,
        start_time=datetime(2026, 1, 1),
        source_directory="/tmp/src",
    )
    ctx.resolved_entities = [_entity()]
    ctx.resolved_relationships = [_relationship()]
    ctx.resolved_claims = [_claim()]
    ctx.communities = [_community()]
    ctx.community_reports = [_report()]
    ctx.stage_results = [_stage_result()]
    return ctx


# --- display_ascii_art ----------------------------------------------------


def test_display_ascii_art_includes_version(cap) -> None:
    out = cap(display.display_ascii_art, "9.9.9")
    assert "9.9.9" in out


# --- display_pipeline_summary --------------------------------------------


def test_pipeline_summary_renders_metrics(cap) -> None:
    out = cap(display.display_pipeline_summary, _context())
    assert "Pipeline Results Summary" in out
    assert "Total Entities" in out


def test_pipeline_summary_shows_zero_baseline_rows_for_empty(cap) -> None:
    ctx = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.PENDING,
        start_time=datetime(2026, 1, 1),
        source_directory="/tmp/src",
    )
    out = cap(display.display_pipeline_summary, ctx)
    # Documents / Text Units rows are always shown even at zero.
    assert "Documents Processed" in out
    assert "Text Units Created" in out


# --- display_stage_results -----------------------------------------------


def test_stage_results_empty_prints_nothing(cap) -> None:
    assert cap(display.display_stage_results, []) == ""


def test_stage_results_renders_each_status(cap) -> None:
    results = [
        _stage_result("indexing", PipelineStageStatus.COMPLETED),
        _stage_result("gleaning", PipelineStageStatus.CACHED),
        _stage_result("translation", PipelineStageStatus.FAILED),
        _stage_result("graph_analysis", PipelineStageStatus.SKIPPED),
        # Unmapped status -> falls into the dim default branch + N/A duration.
        _stage_result("text_chunking", PipelineStageStatus.RUNNING, duration=None),
    ]
    out = cap(display.display_stage_results, results)
    assert "Stage Execution" in out
    assert "Indexing" in out
    assert "N/A" in out


# --- display_sample_entities / _truncate_text ----------------------------


def test_sample_entities_empty_prints_nothing(cap) -> None:
    assert cap(display.display_sample_entities, None) == ""


def test_sample_entities_renders_and_truncates(cap) -> None:
    out = cap(display.display_sample_entities, [_entity()], 5)
    assert "Alice" in out
    assert "..." in out  # long description truncated


def test_truncate_text_behaviour() -> None:
    assert display._truncate_text(None, 10) == ""
    assert display._truncate_text("short", 100) == "short"
    truncated = display._truncate_text("x" * 50, 10)
    assert truncated.endswith("...")
    assert len(truncated) == 10


# --- display_sample_relationships ----------------------------------------


def test_sample_relationships_empty(cap) -> None:
    assert cap(display.display_sample_relationships, None) == ""


def test_sample_relationships_renders(cap) -> None:
    out = cap(display.display_sample_relationships, [_relationship()])
    assert "Alice" in out
    assert "KNOWS" in out


# --- display_sample_claims -----------------------------------------------


def test_sample_claims_empty(cap) -> None:
    assert cap(display.display_sample_claims, None) == ""


def test_sample_claims_renders(cap) -> None:
    out = cap(display.display_sample_claims, [_claim()])
    assert "Alice" in out
    assert "FACT" in out


# --- display_communities + lookups ---------------------------------------


def test_communities_empty(cap) -> None:
    assert cap(display.display_communities, None, None, None, None) == ""


def test_communities_renders_with_reports_and_lookup(cap) -> None:
    out = cap(
        display.display_communities,
        [_community()],
        [_report()],
        [_entity("e0", "Alice"), _entity("e1", "Bob")],
        [_claim()],
        5,
    )
    assert "Community A" in out
    # Entity-name lookup resolved at least one id to a name.
    assert "Alice" in out
    # "... and N more" branch (8 entity_ids, only 5 shown).
    assert "more" in out
    # Report summary rendered.
    assert "Summary" in out


def test_create_entity_and_claim_lookup_fallbacks() -> None:
    # Entity with id but no name -> id[:8]; claim with full names -> arrow form.
    entity_no_name = Entity(id="abcdef123456", name="")
    lookup = display._create_entity_and_claim_lookup(
        [_entity("e1", "Alice"), entity_no_name],
        [_claim()],
    )
    assert lookup["e1"] == "Alice"
    assert lookup["abcdef123456"] == "abcdef12"
    assert lookup["c1"] == "Alice -> Bob"


def test_create_reports_lookup_keys_by_community_id() -> None:
    lookup = display._create_reports_lookup([_report()])
    assert "comm1" in lookup


# --- display_performance_summary -----------------------------------------


def test_performance_summary_empty(cap) -> None:
    assert cap(display.display_performance_summary, [], None) == ""


def test_performance_summary_without_metrics(cap) -> None:
    out = cap(display.display_performance_summary, [_stage_result()], None)
    assert "Performance Summary" in out
    assert "Total Pipeline Duration" in out
    assert "Stages Executed" in out


def test_performance_summary_with_metrics(cap) -> None:
    metrics = PipelineMetrics(
        pipeline_id="pid",
        total_duration_seconds=10.0,
        total_documents_processed=1,
        total_text_units_created=2,
        total_translated_units=0,
        total_entities_extracted=2,
        total_relationships_extracted=1,
        total_claims_extracted=1,
        total_communities_detected=1,
        total_community_reports_generated=1,
        cache_hit_rate=0.5,
        entity_resolution_merge_rate=0.25,
        community_modularity_score=0.42,
        stage_throughput={"indexing": 4.0, "gleaning": 2.0},
    )
    out = cap(display.display_performance_summary, [_stage_result()], metrics)
    assert "Cache Hit Rate" in out
    assert "Community Modularity" in out
    assert "Avg. Throughput" in out


# --- display_pipeline_results (top-level orchestrator) -------------------


def test_display_pipeline_results_runs_end_to_end(cap) -> None:
    out = cap(display.display_pipeline_results, _context(), 5)
    # Touches summary + stages + entities + relationships + claims +
    # communities + performance without raising.
    assert "Pipeline Results Summary" in out
    assert "Sample Extracted Entities" in out
    assert "Detected Communities" in out
