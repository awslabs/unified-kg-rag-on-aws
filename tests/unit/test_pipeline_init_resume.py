# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for DataIngestionPipeline orchestration helpers (AWS-free).

Complements ``test_pipeline_resume`` (which covers
``_prepare_context_for_resume`` canonical ordering) by exercising the OTHER
helpers: stage enable/disable from config, pipeline-id resolution/generation,
completed-stage skipping in ``_execute_pipeline_stages``, ``_get_stage_metric``,
and ``_create_pipeline_metrics`` field population. Heavy stage classes are
replaced with lightweight stubs so no adapters/boto sessions are built.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from aws_graphrag.application.ingestion.pipeline import DataIngestionPipeline
from aws_graphrag.domain.models import (
    Config,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageType,
)

pytestmark = pytest.mark.unit


class _StubStage:
    """A no-adapter stand-in for a pipeline stage class.

    Records the kwargs it was constructed with so enable/disable + injection
    behaviour can be asserted. ``name`` mirrors the real stages (the stage-type
    value).
    """

    instances: list[_StubStage] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.stage_type = kwargs.get("_stage_type")
        _StubStage.instances.append(self)

    @property
    def name(self) -> str:
        return self.stage_type.value


def _stub_stage_classes() -> dict:
    """STAGE_CLASSES mapping where every value is a _StubStage factory.

    Each factory closes over its stage type so the constructed stub knows its
    own name without the orchestrator passing it.
    """

    def make(st: PipelineStageType):
        def factory(**kwargs):
            return _StubStage(_stage_type=st, **kwargs)

        return factory

    return {st: make(st) for st in PipelineStageType}


def _pipeline_for_init(config: Config) -> DataIngestionPipeline:
    """A DataIngestionPipeline with __init__ bypassed and the bits
    ``_initialize_stages`` reads set up against stub stage classes."""
    pipe = object.__new__(DataIngestionPipeline)
    pipe.config = config
    pipe.pipeline_config = SimpleNamespace(
        stages_enabled=dict.fromkeys(PipelineStageType, True)
    )
    pipe.source_directory = Path("/tmp/src")
    pipe.target_directory = Path("/tmp/out")
    pipe.boto_session = None
    pipe.STAGE_CLASSES = _stub_stage_classes()
    return pipe


# --- _initialize_stages: enable/disable ----------------------------------


def test_initialize_stages_all_enabled_by_default() -> None:
    _StubStage.instances = []
    cfg = Config()
    # gleaning enabled by default, claim_extraction disabled by default.
    pipe = _pipeline_for_init(cfg)
    stages, name_to_type = pipe._initialize_stages()

    names = {s.name for s in stages}
    # Claim extraction defaults to OFF -> excluded.
    assert "claim_extraction" not in names
    assert "claim_resolution" not in names
    # Gleaning defaults ON -> included.
    assert "gleaning" in names
    assert name_to_type["gleaning"] == PipelineStageType.GLEANING


def test_initialize_stages_claim_extraction_enabled_includes_claim_stages() -> None:
    _StubStage.instances = []
    cfg = Config()
    cfg.processing.claim_extraction.enabled = True
    pipe = _pipeline_for_init(cfg)
    stages, _ = pipe._initialize_stages()
    names = {s.name for s in stages}
    assert "claim_extraction" in names
    assert "claim_resolution" in names


def test_initialize_stages_gleaning_disabled_excludes_gleaning() -> None:
    _StubStage.instances = []
    cfg = Config()
    cfg.processing.gleaning.enabled = False
    pipe = _pipeline_for_init(cfg)
    stages, _ = pipe._initialize_stages()
    names = {s.name for s in stages}
    assert "gleaning" not in names


def test_initialize_stages_respects_pipeline_config_stages_enabled() -> None:
    _StubStage.instances = []
    cfg = Config()
    pipe = _pipeline_for_init(cfg)
    # Disable text_chunking via the pipeline-config toggle.
    pipe.pipeline_config.stages_enabled[PipelineStageType.TEXT_CHUNKING] = False
    stages, _ = pipe._initialize_stages()
    names = {s.name for s in stages}
    assert "text_chunking" not in names


def test_initialize_stages_injects_directories_and_config() -> None:
    _StubStage.instances = []
    cfg = Config()
    pipe = _pipeline_for_init(cfg)
    stages, _ = pipe._initialize_stages()
    by_name = {s.name: s for s in stages}

    # DOCUMENT_PARSING gets source + target directory; DOCUMENT_LOADING gets source.
    parsing = by_name["document_parsing"]
    assert parsing.kwargs["source_directory"] == pipe.source_directory
    assert parsing.kwargs["target_directory"] == pipe.target_directory
    loading = by_name["document_loading"]
    assert loading.kwargs["source_directory"] == pipe.source_directory
    # Every stage receives the config.
    assert all(s.kwargs["config"] is cfg for s in stages)


def test_initialize_stages_doc_status_not_injected_when_dynamodb_disabled() -> None:
    _StubStage.instances = []
    cfg = Config()
    # DynamoDB disabled by default -> _build_doc_status_store returns None ->
    # no doc_status kwarg on DOC_STATUS_STAGES.
    pipe = _pipeline_for_init(cfg)
    stages, _ = pipe._initialize_stages()
    by_name = {s.name: s for s in stages}
    assert "doc_status" not in by_name["indexing"].kwargs
    assert "doc_status" not in by_name["document_loading"].kwargs


# --- _resolve_pipeline_id / _generate_pipeline_id ------------------------


def test_resolve_pipeline_id_prefers_explicit_argument() -> None:
    pipe = object.__new__(DataIngestionPipeline)
    pipe.pipeline_config = SimpleNamespace(pipeline_id="config-id")
    resolved = DataIngestionPipeline._resolve_pipeline_id(
        pipe, Path("/tmp/src"), "explicit-id"
    )
    assert resolved == "explicit-id"


def test_resolve_pipeline_id_falls_back_to_config() -> None:
    pipe = object.__new__(DataIngestionPipeline)
    pipe.pipeline_config = SimpleNamespace(pipeline_id="config-id")
    resolved = DataIngestionPipeline._resolve_pipeline_id(pipe, Path("/tmp/src"), None)
    assert resolved == "config-id"


def test_resolve_pipeline_id_generates_when_none(tmp_path) -> None:
    pipe = object.__new__(DataIngestionPipeline)
    pipe.pipeline_config = SimpleNamespace(pipeline_id=None)
    resolved = DataIngestionPipeline._resolve_pipeline_id(pipe, tmp_path, None)
    assert resolved.startswith("pipeline-")


def test_generate_pipeline_id_deterministic_hash_for_same_subdirs(tmp_path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    id1 = DataIngestionPipeline._generate_pipeline_id(tmp_path)
    id2 = DataIngestionPipeline._generate_pipeline_id(tmp_path)
    # Hash component (between "pipeline-" and the trailing timestamp) is stable.
    hash1 = id1.split("-")[1]
    hash2 = id2.split("-")[1]
    assert hash1 == hash2
    assert id1.startswith("pipeline-")


def test_generate_pipeline_id_none_source() -> None:
    pid = DataIngestionPipeline._generate_pipeline_id(None)
    assert pid.startswith("pipeline-")
    # No directory hash component (only "pipeline-<timestamp>").
    assert len(pid.split("-")) == 2


def test_generate_pipeline_id_distinct_for_different_dirs(tmp_path) -> None:
    d1 = tmp_path / "one"
    d2 = tmp_path / "two"
    (d1 / "sub1").mkdir(parents=True)
    (d2 / "sub2").mkdir(parents=True)
    h1 = DataIngestionPipeline._generate_pipeline_id(d1).split("-")[1]
    h2 = DataIngestionPipeline._generate_pipeline_id(d2).split("-")[1]
    assert h1 != h2


# --- _execute_pipeline_stages: completed-stage skipping ------------------


def test_execute_pipeline_stages_skips_completed_and_runs_rest(mocker) -> None:
    pipe = object.__new__(DataIngestionPipeline)
    stage_a = SimpleNamespace(name="a")
    stage_b = SimpleNamespace(name="b")
    stage_c = SimpleNamespace(name="c")
    pipe.stages = [stage_a, stage_b, stage_c]

    executed: list[str] = []

    def fake_execute_stage(stage, context):
        executed.append(stage.name)
        return PipelineStageResult(
            stage_name=stage.name,
            status=PipelineStageStatus.COMPLETED,
            start_time=datetime(2026, 1, 1),
        )

    mocker.patch.object(pipe, "_execute_stage", side_effect=fake_execute_stage)
    mocker.patch.object(pipe, "_should_stop_pipeline", return_value=False)

    ctx = _ctx_with_results(
        [
            PipelineStageResult(
                stage_name="b",
                status=PipelineStageStatus.COMPLETED,
                start_time=datetime(2026, 1, 1),
            )
        ]
    )

    DataIngestionPipeline._execute_pipeline_stages(pipe, ctx, start_stage_name=None)

    # "b" was already completed -> skipped; "a" and "c" run.
    assert executed == ["a", "c"]


def test_execute_pipeline_stages_starts_from_named_stage(mocker) -> None:
    pipe = object.__new__(DataIngestionPipeline)
    pipe.stages = [SimpleNamespace(name=n) for n in ("a", "b", "c")]
    executed: list[str] = []
    mocker.patch.object(
        pipe,
        "_execute_stage",
        side_effect=lambda stage, ctx: (
            executed.append(stage.name)
            or PipelineStageResult(
                stage_name=stage.name,
                status=PipelineStageStatus.COMPLETED,
                start_time=datetime(2026, 1, 1),
            )
        ),
    )
    mocker.patch.object(pipe, "_should_stop_pipeline", return_value=False)
    ctx = _ctx_with_results([])

    DataIngestionPipeline._execute_pipeline_stages(pipe, ctx, start_stage_name="b")
    assert executed == ["b", "c"]


def test_execute_pipeline_stages_unknown_start_runs_all(mocker) -> None:
    pipe = object.__new__(DataIngestionPipeline)
    pipe.stages = [SimpleNamespace(name=n) for n in ("a", "b")]
    executed: list[str] = []
    mocker.patch.object(
        pipe,
        "_execute_stage",
        side_effect=lambda stage, ctx: (
            executed.append(stage.name)
            or PipelineStageResult(
                stage_name=stage.name,
                status=PipelineStageStatus.COMPLETED,
                start_time=datetime(2026, 1, 1),
            )
        ),
    )
    mocker.patch.object(pipe, "_should_stop_pipeline", return_value=False)
    ctx = _ctx_with_results([])

    DataIngestionPipeline._execute_pipeline_stages(
        pipe, ctx, start_stage_name="nonexistent"
    )
    assert executed == ["a", "b"]


def test_execute_pipeline_stages_stops_on_failure(mocker) -> None:
    pipe = object.__new__(DataIngestionPipeline)
    pipe.stages = [SimpleNamespace(name=n) for n in ("a", "b", "c")]
    executed: list[str] = []
    mocker.patch.object(
        pipe,
        "_execute_stage",
        side_effect=lambda stage, ctx: (
            executed.append(stage.name)
            or PipelineStageResult(
                stage_name=stage.name,
                status=PipelineStageStatus.FAILED,
                start_time=datetime(2026, 1, 1),
            )
        ),
    )
    # Stop after the first failure.
    mocker.patch.object(pipe, "_should_stop_pipeline", return_value=True)
    ctx = _ctx_with_results([])

    DataIngestionPipeline._execute_pipeline_stages(pipe, ctx, start_stage_name=None)
    assert executed == ["a"]


# --- _get_stage_metric ---------------------------------------------------


def test_get_stage_metric_returns_matching_prefix_value() -> None:
    ctx = _ctx_with_results(
        [
            _result("graph_resolution", metrics={"entity_merge_rate": 0.25}),
            _result("indexing", metrics={"total_indexed": 10}),
        ]
    )
    val = DataIngestionPipeline._get_stage_metric(
        ctx, "graph_resolution", "entity_merge_rate"
    )
    assert val == 0.25


def test_get_stage_metric_defaults_zero_when_absent() -> None:
    ctx = _ctx_with_results([_result("indexing", metrics={})])
    # Missing metric key -> 0.0.
    assert DataIngestionPipeline._get_stage_metric(ctx, "indexing", "nope") == 0.0
    # No matching stage prefix -> 0.0.
    assert DataIngestionPipeline._get_stage_metric(ctx, "missing", "x") == 0.0


def test_get_stage_metric_matches_by_prefix_case_insensitive() -> None:
    ctx = _ctx_with_results([_result("Indexing", metrics={"total_failed": 3.0})])
    assert (
        DataIngestionPipeline._get_stage_metric(ctx, "indexing", "total_failed") == 3.0
    )


# --- _calculate_stage_performance ----------------------------------------


def test_calculate_stage_performance_durations_and_throughput() -> None:
    ctx = _ctx_with_results(
        [
            _result("a", duration=2.0, output=10),
            _result("b", duration=0.0, output=5),  # zero duration -> skipped
            _result("c", duration=4.0, output=0),  # zero output -> no throughput
        ]
    )
    durations, throughput = DataIngestionPipeline._calculate_stage_performance(ctx)
    assert durations == {"a": 2.0, "c": 4.0}
    assert throughput == {"a": 5.0}


# --- _create_pipeline_metrics --------------------------------------------


def test_create_pipeline_metrics_populates_counts_and_stage_metrics() -> None:
    from aws_graphrag.domain.models import Entity, Relationship, TextUnit
    from aws_graphrag.domain.models.cache import CacheStats

    pipe = object.__new__(DataIngestionPipeline)
    pipe.cache_manager = SimpleNamespace(
        get_cache_stats=lambda pid: CacheStats(
            hit_count=3, miss_count=1, total_size_bytes=1024 * 1024
        )
    )

    ctx = _ctx_with_results(
        [
            _result("graph_resolution", metrics={"entity_merge_rate": 0.5}),
            _result(
                "indexing",
                duration=2.0,
                output=4,
                metrics={
                    "total_indexed": 4,
                    "total_failed": 1,
                    "relationships_indexed": 2,
                },
            ),
        ]
    )
    ctx.duration_seconds = 12.5
    ctx.text_units = [TextUnit(id="t1", text="a")]
    ctx.entities = [Entity(id="e1", name="A"), Entity(id="e2", name="B")]
    ctx.relationships = [Relationship(id="r1", source_id="e1", target_id="e2")]

    metrics = DataIngestionPipeline._create_pipeline_metrics(pipe, ctx)

    assert metrics.pipeline_id == ctx.pipeline_id
    assert metrics.total_duration_seconds == 12.5
    assert metrics.total_text_units_created == 1
    assert metrics.total_entities_extracted == 2
    assert metrics.total_relationships_extracted == 1
    assert metrics.entity_resolution_merge_rate == 0.5
    assert metrics.total_items_indexed == 4
    assert metrics.total_items_index_failed == 1
    assert metrics.relationships_indexed == 2
    assert metrics.cache_hit_rate == 0.75
    assert metrics.cache_size_mb == 1.0
    # Stage durations flow through from _calculate_stage_performance.
    assert metrics.stage_durations.get("indexing") == 2.0


# --- helpers -------------------------------------------------------------


def _result(
    name: str,
    status: str = "completed",
    duration: float | None = None,
    output: int = 0,
    metrics: dict | None = None,
) -> PipelineStageResult:
    return PipelineStageResult(
        stage_name=name,
        status=PipelineStageStatus(status),
        start_time=datetime(2026, 1, 1),
        duration_seconds=duration,
        output_count=output,
        metrics=metrics or {},
    )


def _ctx_with_results(results: list[PipelineStageResult]):
    from aws_graphrag.domain.models import PipelineContext

    ctx = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory="/tmp/src",
    )
    ctx.stage_results = results
    return ctx
