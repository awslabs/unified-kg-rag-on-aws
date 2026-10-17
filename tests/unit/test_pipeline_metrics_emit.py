# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pipeline metrics emission (observability).

Regression context: a run that silently dropped all relationships at the
indexing backend looked identical to a healthy run because only EXTRACTED counts
were emitted. These tests assert the indexed-count metrics exist and that
per-stage durations are flattened into emitted scalars (a dict member was
previously dropped by the scalar filter).
"""

from __future__ import annotations

import pytest

from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import PipelineMetrics

pytestmark = pytest.mark.unit


class _RecordingSink:
    def __init__(self) -> None:
        self.metrics: dict = {}

    def emit(self, namespace: str, metrics: dict, dimensions: dict) -> None:
        self.metrics = metrics


def _emit(metrics: PipelineMetrics) -> dict:
    """Everything a full run emits: each stage's metrics, then the run's."""
    from unified_kg_rag.domain.models import PipelineStageType

    stages = [stage.value for stage in PipelineStageType]
    pipe, sink = _stage_pipeline(*stages)
    emitted: dict = {}
    for stage in [*stages, None]:
        pipe._emit_metrics(metrics, "pid", stage)
        emitted.update(sink.metrics)
    return emitted


def test_indexed_counts_are_emitted() -> None:
    m = PipelineMetrics(
        pipeline_id="p",
        total_duration_seconds=1.0,
        total_documents_processed=1,
        total_text_units_created=1,
        total_translated_units=0,
        total_entities_extracted=10,
        total_relationships_extracted=20,
        total_claims_extracted=0,
        total_communities_detected=0,
        total_community_reports_generated=0,
        total_items_indexed=30,
        total_items_index_failed=2,
        relationships_indexed=20,
    )
    emitted = _emit(m)
    assert emitted["total_items_indexed"] == 30
    assert emitted["total_items_index_failed"] == 2
    # The metric that catches the silent-drop failure mode.
    assert emitted["relationships_indexed"] == 20


def test_stage_durations_flattened_into_scalars() -> None:
    m = PipelineMetrics(
        pipeline_id="p",
        total_duration_seconds=5.0,
        total_documents_processed=1,
        total_text_units_created=1,
        total_translated_units=0,
        total_entities_extracted=0,
        total_relationships_extracted=0,
        total_claims_extracted=0,
        total_communities_detected=0,
        total_community_reports_generated=0,
        stage_durations={"indexing": 2.5, "graph_extraction": 1.0},
    )
    emitted = _emit(m)
    # Per-stage durations must reach the sink as individual scalars, not be
    # dropped as a dict member.
    assert emitted["duration_indexing_seconds"] == 2.5
    assert emitted["duration_graph_extraction_seconds"] == 1.0
    # The dict itself must not leak through as a non-scalar.
    assert "stage_durations" not in emitted


def test_relationships_indexed_zero_is_visible_when_extracted() -> None:
    # The exact silent-drop signature: extracted > 0 but indexed == 0.
    m = PipelineMetrics(
        pipeline_id="p",
        total_duration_seconds=1.0,
        total_documents_processed=1,
        total_text_units_created=1,
        total_translated_units=0,
        total_entities_extracted=10,
        total_relationships_extracted=20,
        total_claims_extracted=0,
        total_communities_detected=0,
        total_community_reports_generated=0,
        relationships_indexed=0,
    )
    emitted = _emit(m)
    assert emitted["total_relationships_extracted"] == 20
    assert emitted["relationships_indexed"] == 0


def _stage_pipeline(*stage_names: str) -> tuple[DataIngestionPipeline, _RecordingSink]:
    from types import SimpleNamespace

    pipe = DataIngestionPipeline.__new__(DataIngestionPipeline)
    pipe.metrics_sink = sink = _RecordingSink()
    pipe.stages = [SimpleNamespace(name=name) for name in stage_names]
    return pipe, sink


def _metrics() -> PipelineMetrics:
    return PipelineMetrics(
        pipeline_id="p",
        total_duration_seconds=1.0,
        total_documents_processed=2,
        total_text_units_created=3,
        total_translated_units=0,
        total_entities_extracted=10,
        total_relationships_extracted=20,
        total_claims_extracted=0,
        total_communities_detected=1,
        total_community_reports_generated=1,
        total_items_indexed=30,
        stage_durations={"indexing": 2.5, "graph_extraction": 1.0},
    )


def test_a_stage_emits_the_metrics_it_completes_as_it_finishes() -> None:
    # Emitted right after the stage: a run stopped or timed out later still
    # reports it. Counts several stages produce wait for the last enabled one.
    pipe, sink = _stage_pipeline("graph_extraction", "gleaning", "indexing")
    pipe._emit_metrics(_metrics(), "pid", "graph_extraction")
    assert "total_entities_extracted" not in sink.metrics  # gleaning finishes it
    assert sink.metrics["duration_graph_extraction_seconds"] == 1.0
    assert "total_extraction_failures" in sink.metrics

    pipe._emit_metrics(_metrics(), "pid", "gleaning")
    assert sink.metrics["total_entities_extracted"] == 10

    pipe._emit_metrics(_metrics(), "pid", "indexing")
    assert sink.metrics["total_items_indexed"] == 30
    assert sink.metrics["duration_indexing_seconds"] == 2.5
    for name in ("total_duration_seconds", "total_documents_processed"):
        assert name not in sink.metrics


def test_with_gleaning_off_extraction_completes_the_entity_count() -> None:
    pipe, sink = _stage_pipeline("graph_extraction", "indexing")
    pipe._emit_metrics(_metrics(), "pid", "graph_extraction")
    assert sink.metrics["total_entities_extracted"] == 10


def test_the_run_end_emits_only_the_run_metrics() -> None:
    # The stage metrics went out as their stages finished; a resumed run
    # restores the others and must not report their counts again.
    pipe, sink = _stage_pipeline("graph_extraction", "indexing")
    pipe._emit_metrics(_metrics(), "pid")
    assert sink.metrics["total_duration_seconds"] == 1.0
    for name in (
        "total_documents_processed",
        "total_entities_extracted",
        "total_items_indexed",
        "duration_indexing_seconds",
    ):
        assert name not in sink.metrics


def test_community_drift_is_emitted() -> None:
    m = PipelineMetrics(
        pipeline_id="p",
        total_duration_seconds=1.0,
        total_documents_processed=1,
        total_text_units_created=1,
        total_translated_units=0,
        total_entities_extracted=1,
        total_relationships_extracted=0,
        total_claims_extracted=0,
        total_communities_detected=1,
        total_community_reports_generated=1,
        community_drift_overlap_share=0.5,
        community_rebuild_recommended=1,
    )
    emitted = _emit(m)
    assert emitted["community_drift_overlap_share"] == 0.5
    assert emitted["community_rebuild_recommended"] == 1
