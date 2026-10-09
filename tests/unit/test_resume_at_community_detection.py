# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run resumed at community_detection rebuilds the knowledge graph (AWS-free).

graph_analysis builds the graph on the context, but only its entities and
relationships are cached and the graph is excluded from the run metadata. The
documented recovery for a failed community_detection, re-running with the same
pipeline id, therefore always failed with "Knowledge graph is required for
community detection". Drives the real cache, state and resume managers and the
real graph_analysis stage; the community detector (an LLM caller) is stubbed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.ingestion.pipeline_stages import (
    CommunityDetectionStage,
    GraphAnalysisStage,
)
from unified_kg_rag.domain.models import (
    Community,
    Config,
    Entity,
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageType,
    Relationship,
)
from unified_kg_rag.shared.cache_manager import CacheManager
from unified_kg_rag.shared.pipeline_manager import (
    PipelineResumeManager,
    PipelineStateManager,
)

pytestmark = pytest.mark.unit

PIPELINE_ID = "pid"
COMMUNITY_DETECTION = PipelineStageType.COMMUNITY_DETECTION.value


class _Detector:
    """Stands in for CommunityDetector; records the graph it was given."""

    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.graphs: list[nx.Graph] = []

    def __call__(self, graph: nx.Graph) -> None:
        self.graphs.append(graph)
        if self.fail:
            raise RuntimeError("community report model unavailable")

    def generate_community_objects(self) -> list[Community]:
        return [
            Community(
                id="c1",
                name="Community 1",
                level="0",
                parent="-1",
                children=[],
                entity_ids=["e-vendor", "e-buyer"],
            )
        ]

    def generate_reports(self, communities, text_units=None):  # noqa: ANN001
        return [], []

    def get_community_metrics(self) -> None:
        return None


def _community_stage(config: Config, detector: _Detector) -> CommunityDetectionStage:
    stage = object.__new__(CommunityDetectionStage)
    stage.config = config
    stage.stage_type = PipelineStageType.COMMUNITY_DETECTION
    stage.detector = detector  # type: ignore[assignment]
    stage.cache_directory = None
    return stage


def _pipeline(config: Config, root: Path, stages: list) -> DataIngestionPipeline:
    """A pipeline with ``__init__`` bypassed, wired to real managers on ``root``."""
    cache_manager = CacheManager(config=config, cache_directory=root)
    pipeline = object.__new__(DataIngestionPipeline)
    pipeline.config = config
    pipeline.pipeline_config = SimpleNamespace(
        cache_enabled=True, continue_on_error=False
    )
    pipeline.cache_manager = cache_manager
    pipeline.state_manager = PipelineStateManager(cache_manager)
    pipeline.resume_manager = PipelineResumeManager(pipeline.state_manager, config)
    pipeline.name_to_type_map = {stage.value: stage for stage in PipelineStageType}
    pipeline.stages = stages
    return pipeline


def _resolved_context() -> PipelineContext:
    """graph_resolution completed with a small synthetic graph."""
    context = PipelineContext(
        pipeline_id=PIPELINE_ID,
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=Path("/tmp/src"),
    )
    context.resolved_entities = [
        Entity(id="e-vendor", name="Vendor", type="ORG", text_unit_ids=["t1"]),
        Entity(id="e-buyer", name="Buyer", type="ORG", text_unit_ids=["t1"]),
    ]
    context.resolved_relationships = [
        Relationship(
            id="r1",
            source_id="e-vendor",
            target_id="e-buyer",
            source_name="Vendor",
            target_name="Buyer",
            type="SUPPLIES",
            text_unit_ids=["t1"],
        )
    ]
    context.stage_results = [
        PipelineStageResult(
            stage_name=PipelineStageType.GRAPH_RESOLUTION.value,
            status=PipelineStageStatus.COMPLETED,
            start_time=datetime(2026, 1, 1),
        )
    ]
    return context


def test_rerun_with_the_same_pipeline_id_resumes_at_community_detection(
    tmp_path, mocker
) -> None:
    mocker.patch("boto3.Session")
    config = Config()

    # Run 1: graph_analysis completes, community_detection fails.
    failing = _Detector(fail=True)
    first = _pipeline(
        config,
        tmp_path,
        [GraphAnalysisStage(config), _community_stage(config, failing)],
    )
    context = _resolved_context()
    first.state_manager.save_pipeline_metadata(context)
    first._save_stage_outputs_to_cache(
        context, PipelineStageType.GRAPH_RESOLUTION.value
    )
    first._execute_pipeline_stages(context, PipelineStageType.GRAPH_ANALYSIS.value)
    statuses = {r.stage_name: r.status for r in context.stage_results}
    assert statuses[COMMUNITY_DETECTION] is PipelineStageStatus.FAILED

    # Run 2: a fresh process, same pipeline id.
    detector = _Detector(fail=False)
    analysis = GraphAnalysisStage(config)
    analysis_run = mocker.spy(analysis, "_execute_core")
    second = _pipeline(config, tmp_path, [analysis, _community_stage(config, detector)])
    resumed, start = second._prepare_resume(PIPELINE_ID, None)
    assert start == COMMUNITY_DETECTION
    assert resumed.knowledge_graph is None

    second._execute_pipeline_stages(resumed, start)

    statuses = {r.stage_name: r.status for r in resumed.stage_results}
    assert statuses[COMMUNITY_DETECTION] is PipelineStageStatus.COMPLETED
    analysis_run.assert_not_called()
    (graph,) = detector.graphs
    assert {"e-vendor", "e-buyer"} <= set(graph.nodes)
    assert graph.number_of_edges() == 1
    assert [c.id for c in resumed.communities] == ["c1"]
