# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Pipeline metrics emission (observability).

Regression context: a run that silently dropped all relationships at the
indexing backend looked identical to a healthy run because only EXTRACTED counts
were emitted. These tests assert the indexed-count metrics exist and that
per-stage durations are flattened into emitted scalars (a dict member was
previously dropped by the scalar filter).
"""

from __future__ import annotations

import pytest

from aws_graphrag.application.ingestion.pipeline import DataIngestionPipeline
from aws_graphrag.domain.models import PipelineMetrics

pytestmark = pytest.mark.unit


class _RecordingSink:
    def __init__(self) -> None:
        self.metrics: dict = {}

    def emit(self, namespace: str, metrics: dict, dimensions: dict) -> None:
        self.metrics = metrics


def _emit(metrics: PipelineMetrics) -> dict:
    pipe = DataIngestionPipeline.__new__(DataIngestionPipeline)
    sink = _RecordingSink()
    pipe.metrics_sink = sink
    pipe._emit_metrics(metrics, "pid")
    return sink.metrics


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
