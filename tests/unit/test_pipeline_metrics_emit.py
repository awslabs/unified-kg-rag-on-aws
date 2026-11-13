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
from unified_kg_rag.domain.models import Config, PipelineMetrics

pytestmark = pytest.mark.unit


class _RecordingSink:
    def __init__(self) -> None:
        self.metrics: dict = {}
        self.dimensions: dict = {}

    def emit(self, namespace: str, metrics: dict, dimensions: dict) -> None:
        self.metrics = metrics
        self.dimensions = dimensions


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
    pipe.config = Config()
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


def test_the_metrics_name_the_namespace_the_run_wrote() -> None:
    # Regression (rv24): only pipeline_id; the alarms could not be split
    # per namespace (tenant) in Logs Insights.
    pipe, sink = _stage_pipeline("indexing")
    pipe.config.processing.document_parsing.index_value = "tenant-a"
    pipe._emit_metrics(_metrics(), "pid")
    assert sink.dimensions == {"pipeline_id": "pid", "namespace": "tenant-a"}


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


# --- LLM usage per stage -------------------------------------------------------


def _usage_pipeline(mocker) -> DataIngestionPipeline:  # type: ignore[no-untyped-def]
    from contextlib import nullcontext

    pipe, _ = _stage_pipeline("document_parsing", "graph_extraction")
    pipe.state_manager = mocker.MagicMock()
    pipe._checkpoint_write = nullcontext  # type: ignore[method-assign]
    pipe._save_stage_outputs_to_cache = mocker.MagicMock()  # type: ignore[method-assign]
    pipe.corpus_fingerprint = None
    return pipe


def _stage(name: str, calls: int, *, fail: bool = False):  # type: ignore[no-untyped-def]
    from datetime import datetime
    from types import SimpleNamespace

    from unified_kg_rag.domain.models import (
        PipelineStageResult,
        PipelineStageStatus,
        PipelineStageType,
    )
    from unified_kg_rag.shared import usage

    def execute(context):  # type: ignore[no-untyped-def]
        for _ in range(calls):
            usage.record_llm_call(
                "ingestion", "model-a", input_tokens=100, output_tokens=7
            )
        usage.record_cache_hit("ingestion", "model-a")
        if fail:
            raise RuntimeError("model outage")
        return PipelineStageResult(
            stage_name=name,
            status=PipelineStageStatus.COMPLETED,
            start_time=datetime.now(),
        )

    return SimpleNamespace(
        name=name, stage_type=PipelineStageType(name), execute=execute
    )


def _context():  # type: ignore[no-untyped-def]
    from datetime import datetime
    from pathlib import Path

    from unified_kg_rag.domain.models import (
        PipelineConfig,
        PipelineContext,
        PipelineStageStatus,
    )

    return PipelineContext(
        pipeline_id="p",
        config=PipelineConfig(),
        status=PipelineStageStatus.RUNNING,
        start_time=datetime.now(),
        source_directory=Path("src"),
    )


def test_a_stage_result_carries_its_llm_usage_even_when_it_fails(mocker) -> None:
    pipe = _usage_pipeline(mocker)
    context = _context()
    done = pipe._run_stage(_stage("graph_extraction", 3), context)
    usage = done.metrics["llm_usage"]
    assert (usage["llm_calls"], usage["input_tokens"]) == (3, 300)
    assert (usage["output_tokens"], usage["cache_hits"]) == (21, 1)
    assert usage["by_model"]["model-a"]["llm_calls"] == 3
    failed = pipe._run_stage(_stage("graph_extraction", 2, fail=True), context)
    assert failed.metrics["llm_usage"]["llm_calls"] == 2


def test_stage_usage_is_emitted_flat_and_only_for_model_stages(mocker) -> None:
    pipe = _usage_pipeline(mocker)
    sink = pipe.metrics_sink
    context = _context()
    for stage in (_stage("document_parsing", 0), _stage("graph_extraction", 4)):
        result = pipe._run_stage(stage, context)
        context.stage_results.append(result)
    # A model-free stage (no call, no hit) creates no usage series.
    context.stage_results[0].metrics["llm_usage"]["cache_hits"] = 0
    mocker.patch.object(pipe, "_create_pipeline_metrics", return_value=_metrics())
    pipe._emit_stage_metrics(context, "document_parsing")
    assert not [name for name in sink.metrics if name.startswith("llm_")]
    pipe._emit_stage_metrics(context, "graph_extraction")
    assert sink.metrics["llm_calls_graph_extraction"] == 4
    assert sink.metrics["llm_input_tokens_graph_extraction"] == 400
    assert sink.metrics["llm_output_tokens_graph_extraction"] == 28
    assert sink.metrics["llm_cache_hits"] == 1


def test_run_token_usage_sums_the_stages_this_run_executed(mocker) -> None:
    from unified_kg_rag.application.ingestion.pipeline import run_token_usage

    pipe = _usage_pipeline(mocker)
    context = _context()
    for stage in (_stage("document_parsing", 1), _stage("graph_extraction", 2)):
        context.stage_results.append(pipe._run_stage(stage, context))
    # Restored from an earlier run's checkpoint: spent nothing in this one.
    usage = run_token_usage(context, frozenset({"graph_extraction"}))
    assert (usage["llm_calls"], usage["input_tokens"], usage["cache_hits"]) == (
        2,
        200,
        1,
    )
    assert list(usage["stages"]) == ["graph_extraction"]
    assert usage["by_role"]["ingestion"]["llm_calls"] == 2
    both = run_token_usage(context, frozenset({"document_parsing", "graph_extraction"}))
    assert both["llm_calls"] == 3


def test_throttled_requests_are_emitted_even_when_every_call_failed() -> None:
    # Old behaviour: no throttle metric at all, and a stage whose calls all
    # failed (throttled past their retries) emitted no usage.
    from types import SimpleNamespace

    from unified_kg_rag.application.ingestion.pipeline_metrics import (
        LLM_USAGE_METRIC,
        stage_usage_metrics,
    )

    def context(usage: dict[str, int]) -> SimpleNamespace:
        result = SimpleNamespace(
            stage_name="graph_extraction", metrics={LLM_USAGE_METRIC: usage}
        )
        return SimpleNamespace(stage_results=[result])

    failed = stage_usage_metrics(
        context(
            {
                "failed_calls": 2,
                "request_retries": 10,
                "sent_requests": 12,
                "throttled_requests": 12,
            }
        ),
        "graph_extraction",
    )
    assert failed["llm_throttles_graph_extraction"] == 12
    # Every request sent: 2 failed calls of 6 requests each, all throttled.
    assert (failed["llm_throttles"], failed["llm_requests"]) == (12, 12)
    mixed = stage_usage_metrics(
        context(
            {
                "llm_calls": 8,
                "request_retries": 2,
                "sent_requests": 10,
                "throttled_requests": 1,
            }
        ),
        "graph_extraction",
    )
    assert (mixed["llm_throttles"], mixed["llm_requests"]) == (1, 10)
    # Regression (rv25): a stage that only embeds (indexing) made no model
    # call, so its requests and throttles were never emitted.
    embedding = stage_usage_metrics(
        context({"sent_requests": 40, "throttled_requests": 3}), "graph_extraction"
    )
    assert (embedding["llm_throttles"], embedding["llm_requests"]) == (3, 40)
    # Regression (rv26): with always-zero calls and token series (billed).
    assert not any(
        name.startswith(("llm_calls_", "llm_input_tokens_", "llm_output_tokens_"))
        for name in embedding
    )
    assert failed["llm_calls_graph_extraction"] == 0
    assert stage_usage_metrics(context({}), "graph_extraction") == {}
