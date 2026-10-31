# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for pipeline_stages pure helpers + IndexingStage aggregation.

Covers the base ``PipelineStage`` validation/result helpers and the
``IndexingStage`` metric computation (``relationships_indexed``,
``total_indexed``/``total_failed``) and ``_validate_backend_success`` guard,
with the heavy ``IndexingManager`` stubbed so no AWS clients are built.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pytest

from unified_kg_rag.domain.models import (
    Config,
    LanguageCode,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.ports.indexer import IndexingStats
from unified_kg_rag.shared import PipelineStageError

pytestmark = pytest.mark.unit


# A minimal concrete stage to exercise the abstract base's shared helpers.
def _make_stage(stage_type: PipelineStageType):
    from unified_kg_rag.application.ingestion.pipeline_stages import PipelineStage

    class _Concrete(PipelineStage):
        def __init__(self, st: PipelineStageType) -> None:
            # Bypass PipelineStage.__init__ (which builds a boto session); set the
            # two attributes the helpers under test actually read.
            self.config = Config()
            self.stage_type = st

        def _execute_core(self, context):  # pragma: no cover - not exercised
            return 0, 0, None

    return _Concrete(stage_type)


# --- _validate_critical_stage_output -------------------------------------


def test_validate_critical_stage_raises_on_missing_required_input() -> None:
    stage = _make_stage(PipelineStageType.GRAPH_EXTRACTION)  # MUST_HAVE_INPUT
    with pytest.raises(PipelineStageError, match="received 0 inputs"):
        stage._validate_critical_stage_output(input_count=0, output_count=0)


def test_validate_critical_stage_raises_on_zero_output_for_critical() -> None:
    stage = _make_stage(PipelineStageType.GRAPH_EXTRACTION)  # CRITICAL
    with pytest.raises(PipelineStageError, match="produced 0 outputs"):
        stage._validate_critical_stage_output(input_count=5, output_count=0)


def test_validate_critical_stage_passes_with_outputs() -> None:
    stage = _make_stage(PipelineStageType.GRAPH_EXTRACTION)
    # No exception when output is produced.
    stage._validate_critical_stage_output(input_count=5, output_count=3)


def test_validate_optional_output_stage_allows_zero_output() -> None:
    # Claim extraction is in OPTIONAL_OUTPUT_STAGES: 0 outputs is allowed (warns).
    stage = _make_stage(PipelineStageType.CLAIM_EXTRACTION)
    stage._validate_critical_stage_output(input_count=5, output_count=0)


def test_validate_no_required_input_returns_on_zero_input() -> None:
    # Claim extraction is NOT a MUST_HAVE_INPUT stage: zero input is a no-op.
    stage = _make_stage(PipelineStageType.CLAIM_EXTRACTION)
    stage._validate_critical_stage_output(input_count=0, output_count=0)


@pytest.mark.parametrize("stage_type", list(PipelineStageType))
def test_zero_output_fails_every_stage_but_the_optional_ones(stage_type) -> None:
    stage = _make_stage(stage_type)
    # Claims and gleaning may legitimately find nothing new.
    if stage_type in {
        PipelineStageType.CLAIM_EXTRACTION,
        PipelineStageType.CLAIM_RESOLUTION,
        PipelineStageType.GLEANING,
    }:
        stage._validate_critical_stage_output(input_count=5, output_count=0)
    else:
        with pytest.raises(PipelineStageError, match="produced 0 outputs"):
            stage._validate_critical_stage_output(input_count=5, output_count=0)


# --- _allows_empty_output ------------------------------------------------


def test_allows_empty_output_when_incremental_delta_empty_docs() -> None:
    from unified_kg_rag.domain.models.document import DocumentDelta

    stage = _make_stage(PipelineStageType.GRAPH_EXTRACTION)
    ctx = _context()
    ctx.incremental_delta = DocumentDelta()
    ctx.documents = []
    assert stage._allows_empty_output(ctx) is True


def test_disallows_empty_output_without_delta() -> None:
    stage = _make_stage(PipelineStageType.GRAPH_EXTRACTION)
    ctx = _context()
    ctx.documents = []
    assert stage._allows_empty_output(ctx) is False


def _delta_context_with_a_document() -> PipelineContext:
    from unified_kg_rag.domain.models import Document
    from unified_kg_rag.domain.models.document import DocumentDelta

    ctx = _context()
    ctx.incremental_delta = DocumentDelta()
    ctx.documents = [
        Document(
            page_content="x",
            document_id="d1",
            file_name="f.txt",
            file_path="/tmp/f.txt",
        )
    ]
    return ctx


def test_disallows_empty_output_of_a_corpus_stage_when_delta_has_docs() -> None:
    stage = _make_stage(PipelineStageType.TEXT_CHUNKING)
    assert stage._allows_empty_output(_delta_context_with_a_document()) is False


@pytest.mark.parametrize(
    "stage_type",
    [
        PipelineStageType.GRAPH_EXTRACTION,
        PipelineStageType.GRAPH_RESOLUTION,
        PipelineStageType.GRAPH_ANALYSIS,
        PipelineStageType.COMMUNITY_DETECTION,
    ],
)
def test_an_incremental_delta_may_hold_no_entity(stage_type) -> None:
    # The delta's documents may name nothing, unless a unit failed (an
    # unreachable model extracts nothing too) or the run is a full build.
    stage = _make_stage(stage_type)
    ctx = _delta_context_with_a_document()
    assert stage._allows_empty_output(ctx) is True
    ctx.failed_text_unit_ids = {"graph_extraction": ["u1"]}
    assert stage._allows_empty_output(ctx) is False
    ctx.failed_text_unit_ids = {"graph_extraction": []}
    assert stage._allows_empty_output(ctx) is True
    stage.config.indexing.reset = True
    assert stage._allows_empty_output(ctx) is False


# --- _create_result ------------------------------------------------------


def test_create_result_populates_fields() -> None:
    stage = _make_stage(PipelineStageType.TEXT_CHUNKING)
    start = datetime(2026, 1, 1, 0, 0, 0)
    end = datetime(2026, 1, 1, 0, 0, 5)
    result = stage._create_result(
        PipelineStageStatus.COMPLETED,
        start,
        end,
        input_count=3,
        output_count=9,
        metrics={"k": "v"},
    )
    assert result.stage_name == "text_chunking"
    assert result.status == PipelineStageStatus.COMPLETED
    assert result.duration_seconds == 5.0
    assert result.input_count == 3
    assert result.output_count == 9
    assert result.metrics == {"k": "v"}


def test_create_result_defaults_metrics_to_empty_dict() -> None:
    stage = _make_stage(PipelineStageType.TEXT_CHUNKING)
    start = end = datetime(2026, 1, 1)
    result = stage._create_result(
        PipelineStageStatus.FAILED, start, end, error_message="boom"
    )
    assert result.metrics == {}
    assert result.error_message == "boom"


# --- IndexingStage metric aggregation + backend validation ---------------


def _indexing_stage(
    mocker,
    indexing_results: dict[str, IndexingStats],
    max_failure_rate: float | None = None,
):
    """Build an IndexingStage with its IndexingManager fully stubbed."""
    from unified_kg_rag.application.ingestion import pipeline_stages as ps

    fake_mgr = mocker.MagicMock()
    fake_mgr.index_all_data.return_value = indexing_results
    mocker.patch.object(ps, "IndexingManager", return_value=fake_mgr)

    cfg = Config()
    cfg.indexing.reset = False
    if max_failure_rate is not None:
        cfg.indexing.max_failure_rate = max_failure_rate
    stage = ps.IndexingStage(config=cfg)
    return stage, fake_mgr


def test_indexing_stage_computes_relationships_indexed_metric(mocker) -> None:
    results = {
        "neptune_entities": IndexingStats(total_items=3, successful_items=3),
        "neptune_relationships": IndexingStats(total_items=4, successful_items=4),
        "opensearch_relationships": IndexingStats(
            total_items=4, successful_items=2, failed_items=2
        ),
        "opensearch_text_units": IndexingStats(total_items=2, successful_items=2),
    }
    # This test exercises metric aggregation, not the failure gate; the 50%
    # opensearch_relationships failure would otherwise trip the (default 20%)
    # partial-failure gate. Disable the gate so metrics are tested in isolation.
    stage, _mgr = _indexing_stage(mocker, results, max_failure_rate=1.0)

    input_count, total_indexed, metrics = stage._execute_core(_context())

    # relationships_indexed sums successes across every key containing
    # "relationship": neptune (4) + opensearch (2) = 6.
    assert metrics["relationships_indexed"] == 6
    assert metrics["total_indexed"] == 3 + 4 + 2 + 2
    assert metrics["total_failed"] == 2  # opensearch_relationships failed_items
    assert total_indexed == metrics["total_indexed"]


def test_indexing_stage_records_each_entitys_level0_community(mocker) -> None:
    from unified_kg_rag.domain.models import Community, Entity

    stage, mgr = _indexing_stage(mocker, {})
    ctx = _context()
    ctx.resolved_entities = [Entity(id="e1", name="A"), Entity(id="e2", name="B")]
    ctx.communities = [
        Community(
            id=cid, name=cid, level=level, parent="", children=[], entity_ids=members
        )
        for cid, level, members in (("c0", "0", ["e1"]), ("c1", "1", ["e1", "e2"]))
    ]

    stage._execute_core(ctx)

    written = mgr.index_all_data.call_args.kwargs["entities"]
    assert [e.community_ids for e in written] == [["c0"], None]


def test_indexing_stage_success_rate_zero_when_nothing(mocker) -> None:
    stage, _mgr = _indexing_stage(mocker, {})
    _in, total, metrics = stage._execute_core(_context())
    assert total == 0
    assert metrics["success_rate"] == 0
    assert metrics["relationships_indexed"] == 0


def test_indexing_stage_input_count_sums_all_artifact_types(mocker) -> None:
    from unified_kg_rag.domain.models import Entity, Relationship, TextUnit

    stage, _mgr = _indexing_stage(
        mocker, {"neptune_entities": IndexingStats(total_items=1, successful_items=1)}
    )
    ctx = _context()
    ctx.text_units = [TextUnit(id="t1", text="a")]
    ctx.resolved_entities = [Entity(id="e1", name="A"), Entity(id="e2", name="B")]
    ctx.resolved_relationships = [Relationship(id="r1", source_id="e1", target_id="e2")]

    input_count, _total, _metrics = stage._execute_core(ctx)
    # 1 text unit + 2 entities + 1 relationship = 4 (no claims/communities).
    assert input_count == 4


def test_validate_backend_success_raises_on_index_type_total_failure(mocker) -> None:
    stage, _mgr = _indexing_stage(mocker, {})
    results = {
        "opensearch_entities": IndexingStats(total_items=5, successful_items=0),
    }
    with pytest.raises(PipelineStageError, match="opensearch_entities"):
        stage._validate_backend_success(results)


def test_validate_backend_success_names_the_first_error(mocker) -> None:
    # The stage error said only "check the logs above"; the engine's reason
    # (an index setting it refuses, say) is the one thing the user needs.
    stage, _mgr = _indexing_stage(mocker, {})
    failed = IndexingStats(total_items=5)
    failed.add_error("unknown setting [index.refresh_intervl]", 5)
    with pytest.raises(PipelineStageError, match=r"refresh_intervl"):
        stage._validate_backend_success({"opensearch_entities": failed})


def test_validate_backend_success_passes_with_partial_success(mocker) -> None:
    # Default gate is 20%; 3/5 (40% failed) trips it, so raise the tolerance to
    # confirm a below-threshold partial failure still passes.
    stage, _mgr = _indexing_stage(mocker, {}, max_failure_rate=0.5)
    results = {
        "opensearch_entities": IndexingStats(
            total_items=5, successful_items=3, failed_items=2
        ),
        "neptune_entities": IndexingStats(total_items=5, successful_items=5),
    }
    # No exception: 40% failure <= 50% tolerated, and every key has a success.
    stage._validate_backend_success(results)


def test_validate_backend_success_raises_on_partial_failure_over_threshold(
    mocker,
) -> None:
    # The silent-partial-failure gate: an index type with SOME successes but a
    # failure rate above the tolerance must fail the stage (previously it would
    # pass and the run was reported successful despite most writes failing).
    stage, _mgr = _indexing_stage(mocker, {})  # default 20% tolerance
    results = {
        "neptune_relationships": IndexingStats(
            total_items=100, successful_items=22, failed_items=78
        ),
    }
    with pytest.raises(PipelineStageError, match="neptune_relationships"):
        stage._validate_backend_success(results)


def test_validate_backend_success_ignores_zero_total_items(mocker) -> None:
    stage, _mgr = _indexing_stage(mocker, {})
    # total_items==0 AND failed_items==0 keys had no work to do -> skipped (not a
    # failure). This is distinct from the exception-before-seeding case below.
    results = {
        "opensearch_entities": IndexingStats(total_items=0, successful_items=0),
    }
    stage._validate_backend_success(results)


def test_validate_backend_success_raises_on_exception_before_seeding(mocker) -> None:
    # Regression: an index type that raises before _perform_indexing seeds
    # total_items surfaces as total_items=0, failed_items=N (via add_error).
    # The per-index-type gate previously skipped total_items<=0 keys, so this
    # complete write failure passed as success (and, in incremental mode, was
    # durably recorded PROCESSED -> permanent silent data loss). It must raise.
    stage, _mgr = _indexing_stage(mocker, {})
    stats = IndexingStats()
    stats.add_error("Bedrock embedding endpoint unreachable", 500)
    results = {"opensearch_entities": stats}
    assert stats.total_items == 0 and stats.failed_items == 500
    with pytest.raises(PipelineStageError, match="opensearch_entities"):
        stage._validate_backend_success(results)


def test_validate_backend_success_raises_on_whole_backend_exception(mocker) -> None:
    # Regression: every index type in a backend raises before seeding, so the
    # backend aggregate is total_items=0 across the board. The backend-level
    # gate previously only summed total_items (=0) and never fired; it must now
    # detect the failure via failed_items.
    stage, _mgr = _indexing_stage(mocker, {})
    s1, s2 = IndexingStats(), IndexingStats()
    s1.add_error("conn refused", 3)
    s2.add_error("conn refused", 4)
    results = {"neptune_entities": s1, "neptune_relationships": s2}
    with pytest.raises(PipelineStageError):
        stage._validate_backend_success(results)


# --- helpers -------------------------------------------------------------


def _context() -> PipelineContext:
    return PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory="/tmp/src",
    )


# --- CommunityDetectionStage: visualization output directory ----------------


def _viz_dir(outputs_directory: str | None, cache_directory: Any) -> Any:
    from types import SimpleNamespace

    from unified_kg_rag.application.ingestion.pipeline_stages import (
        CommunityDetectionStage,
    )
    from unified_kg_rag.domain.models import Config

    config = Config()
    config.graph.visualization.outputs_directory = outputs_directory
    stage = SimpleNamespace(config=config, cache_directory=cache_directory)
    context = SimpleNamespace(pipeline_id="pid")
    return CommunityDetectionStage._visualization_outputs_dir(stage, context)


def test_visualization_defaults_under_pipeline_cache_dir(tmp_path) -> None:
    # The S3 cache sync uploads <cache dir>/<pipeline_id>, so a deployed run's
    # visualization_data.json survives the task.
    assert _viz_dir(None, tmp_path) == tmp_path / "pid" / "visualization"


def test_visualization_explicit_directory_wins(tmp_path) -> None:
    assert _viz_dir("custom/viz", tmp_path) is None  # manager uses the config path


def test_visualization_without_cache_dir_falls_back(tmp_path) -> None:
    assert _viz_dir(None, None) is None


def _data_export_stage(mocker, tmp_path, *, export_data: bool):
    import networkx as nx

    from unified_kg_rag.application.ingestion import pipeline_stages as ps

    detector = mocker.MagicMock()
    detector.generate_community_objects.return_value = []
    detector.generate_reports.return_value = ([], [])
    detector.get_community_metrics.return_value = None
    detector.export_community_data.return_value = {"hierarchy": [{"nodes": ["e1"]}]}
    mocker.patch.object(ps, "CommunityDetector", return_value=detector)
    config = Config()
    config.graph.visualization.enabled = False
    config.graph.visualization.export_data = export_data
    config.graph.visualization.outputs_directory = str(tmp_path / "viz")
    stage = ps.CommunityDetectionStage(config=config)
    ctx = _context()
    graph = nx.Graph()
    graph.add_node("e1", name="Vendor", embedding=[0.1, 0.2])
    graph.add_node("e2", name="Buyer")
    graph.add_edge("e1", "e2", weight=2.0)
    ctx.knowledge_graph = graph
    return stage, ctx, detector


def test_community_stage_exports_visualization_data_without_rendering(
    mocker, tmp_path
) -> None:
    # Rendering is off by default, but run-visualization needs the data file;
    # writing it costs one pass over the graph and no embedding call.
    import json

    stage, ctx, _ = _data_export_stage(mocker, tmp_path, export_data=True)
    manager = mocker.patch("unified_kg_rag.visualization.GraphVisualizationManager")

    stage._execute_core(ctx)

    manager.assert_not_called()
    data = json.loads((tmp_path / "viz" / "visualization_data.json").read_text())
    assert [n["id"] for n in data["nodes"]] == ["e1", "e2"]
    assert "embedding" not in data["nodes"][0]["attributes"]
    assert data["edges"] == [
        {"source": "e1", "target": "e2", "attributes": {"weight": 2.0}}
    ]
    assert data["layout"] == {}
    assert data["communities"] == {"hierarchy": [{"nodes": ["e1"]}]}


def test_community_stage_skips_the_export_when_disabled(mocker, tmp_path) -> None:
    stage, ctx, _ = _data_export_stage(mocker, tmp_path, export_data=False)

    stage._execute_core(ctx)

    assert not (tmp_path / "viz").exists()


def test_failed_visualization_data_export_does_not_fail_the_stage(
    mocker, tmp_path
) -> None:
    stage, ctx, detector = _data_export_stage(mocker, tmp_path, export_data=True)
    detector.export_community_data.side_effect = RuntimeError("no communities")

    _, _, metrics = stage._execute_core(ctx)

    assert metrics is not None
    assert not (tmp_path / "viz" / "visualization_data.json").exists()


# --- extraction stages: failed text units reach the context ---------------


def test_extraction_stage_records_failed_units_and_rerun_replaces_them(
    mocker,
) -> None:
    from unified_kg_rag.adapters.ingestion.graph_extractor import ExtractionStats
    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.domain.models import TextUnit

    extractor = mocker.MagicMock()
    mocker.patch.object(ps, "GraphExtractor", return_value=extractor)
    stage = ps.GraphExtractionStage(config=Config())
    ctx = _context()
    ctx.text_units = [TextUnit(id="t1", text="x")]

    extractor.extract_units.return_value = (
        [],
        ExtractionStats(num_failed_extractions=1, failed_text_unit_ids=["t1"]),
    )
    stage._execute_core(ctx)
    assert ctx.failed_text_unit_ids == {"graph_extraction": ["t1"]}

    # A re-run of the stage replaces its entry rather than accumulating.
    extractor.extract_units.return_value = ([], ExtractionStats())
    stage._execute_core(ctx)
    assert ctx.failed_text_unit_ids == {"graph_extraction": []}


def test_translation_stage_records_failed_units(mocker) -> None:
    # A unit that kept its original text would be extracted in the source
    # language; its document is recorded FAILED and retried like an
    # extraction failure.
    from unified_kg_rag.adapters.ingestion.translator import TranslationStats
    from unified_kg_rag.application.ingestion import pipeline_stages as ps

    translator = mocker.MagicMock()
    translator.translate_text_units.side_effect = lambda units: units
    # t2's error was its own (counted), t1's an outage (retried only).
    translator.stats = TranslationStats(
        failed_text_unit_ids=["t1", "t2"], item_failed_text_unit_ids=["t2"]
    )
    mocker.patch.object(ps, "TextUnitTranslator", return_value=translator)
    config = Config()
    config.processing.translation.enabled = True
    config.processing.translation.additional_target_languages = [LanguageCode.KO]
    stage = ps.TranslationStage(config=config)
    ctx = _context()

    stage._execute_core(ctx)

    assert ctx.failed_text_unit_ids == {"translation": ["t1", "t2"]}
    assert ctx.item_failed_text_unit_ids == {"translation": ["t2"]}


def test_gleaning_stage_improvement_rate_is_graph_growth(mocker) -> None:
    from unified_kg_rag.adapters.ingestion.gleaner import GleaningStats
    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.domain.models import (
        Entity,
        Relationship,
        TextUnitExtraction,
    )

    gleaner = mocker.MagicMock()
    mocker.patch.object(ps, "GraphGleaner", return_value=gleaner)
    stage = ps.GleaningStage(config=Config())
    ctx = _context()
    ctx.unit_extractions = [
        TextUnitExtraction(
            text_unit_id="t1",
            entities=[Entity(id=f"e{i}", name=f"n{i}") for i in range(3)],
            relationships=[Relationship(id="r1", source_id="e0", target_id="e1")],
        )
    ]
    gleaner.glean.return_value = (
        ctx.unit_extractions,
        GleaningStats(total_entities_added=1, total_relationships_added=1),
    )

    _in, _out, metrics = stage._execute_core(ctx)

    # 2 items added on top of the 4 extracted.
    assert metrics["improvement_rate"] == pytest.approx(0.5)


# --- CommunityDetectionStage: failed community reports ----------------------


def _community_stage(mocker, failed: list[str], max_failure_rate: float):
    import networkx as nx

    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.domain.models import Community

    communities = [
        Community(
            id=f"c{i}",
            name=f"Community {i}",
            level="0",
            parent="",
            children=[],
            entity_ids=[f"e{i}"],
            text_unit_ids=[f"t{i}", "t-shared"],
        )
        for i in range(4)
    ]
    detector = mocker.MagicMock()
    detector.generate_community_objects.return_value = communities
    detector.generate_reports.return_value = ([], failed)
    detector.get_community_metrics.return_value = None
    mocker.patch.object(ps, "CommunityDetector", return_value=detector)
    config = Config()
    config.graph.visualization.enabled = False
    config.indexing.max_failure_rate = max_failure_rate
    stage = ps.CommunityDetectionStage(config=config)
    ctx = _context()
    ctx.knowledge_graph = nx.Graph()
    return stage, ctx


def test_community_stage_records_text_units_of_failed_reports(mocker) -> None:
    # 1 of 4 is within the 25% tolerance: the stage completes and the failed
    # community's text units are recorded so their documents are retried.
    stage, ctx = _community_stage(mocker, failed=["c1"], max_failure_rate=0.25)

    _, _, metrics = stage._execute_core(ctx)

    assert ctx.failed_text_unit_ids == {"community_detection": ["t-shared", "t1"]}
    assert metrics is not None and metrics["reports_failed"] == 1


def test_community_stage_fails_above_the_tolerated_failure_rate(mocker) -> None:
    stage, ctx = _community_stage(mocker, failed=["c1", "c2"], max_failure_rate=0.25)

    with pytest.raises(PipelineStageError, match="2 of 4 communities"):
        stage._execute_core(ctx)


def test_community_stage_without_report_failures_records_none(mocker) -> None:
    stage, ctx = _community_stage(mocker, failed=[], max_failure_rate=0.0)

    stage._execute_core(ctx)

    assert ctx.failed_text_unit_ids == {"community_detection": []}


# --- CommunityDetectionStage._check_report_failures --------------------


def test_only_a_failed_base_level_report_marks_its_documents_failed() -> None:
    # A parent report spans every child's text units: recording them would
    # re-extract most of the corpus next run for one missing report.
    from unified_kg_rag.application.ingestion.pipeline_stages import (
        CommunityDetectionStage,
    )
    from unified_kg_rag.domain.models import Community

    def community(cid: str, level: str, units: list[str]) -> Community:
        return Community(
            id=cid,
            name=cid,
            level=level,
            parent="",
            children=[],
            text_unit_ids=units,
        )

    stage = CommunityDetectionStage.__new__(CommunityDetectionStage)
    stage.config = Config()
    stage.config.indexing.max_failure_rate = 1.0
    communities = [
        community("leaf-a", "0", ["u1"]),
        community("leaf-b", "0", ["u2"]),
        community("root", "1", ["u1", "u2"]),
    ]
    assert stage._check_report_failures(communities, ["root"]) == []
    assert stage._check_report_failures(communities, ["root", "leaf-b"]) == ["u2"]


# --- GraphAnalysisStage --------------------------------------------------


def test_graph_analysis_builds_the_graph_without_centrality(mocker) -> None:
    # Nothing downstream reads centrality, and betweenness takes minutes on
    # a large graph; the stage also caches nothing graph_resolution did not.
    from unified_kg_rag.application.ingestion.pipeline_manager import STAGE_OUTPUTS
    from unified_kg_rag.application.ingestion.pipeline_stages import (
        GraphAnalysisStage,
    )
    from unified_kg_rag.domain.ingestion.graph_analyzer import GraphAnalyzer
    from unified_kg_rag.domain.models import Entity, Relationship

    centrality = mocker.patch.object(GraphAnalyzer, "calculate_centrality")
    stage = GraphAnalysisStage(Config())
    ctx = _context()
    ctx.resolved_entities = [Entity(id="a", name="A"), Entity(id="b", name="B")]
    ctx.resolved_relationships = [
        Relationship(id="r", source_id="a", target_id="b", type="WORKS_WITH")
    ]
    result = stage.execute(ctx)
    assert result.status is PipelineStageStatus.COMPLETED, result.error_message
    assert ctx.knowledge_graph.number_of_edges() == 1
    assert result.metrics["graph_metrics"]["num_nodes"] == 2
    centrality.assert_not_called()
    assert PipelineStageType.GRAPH_ANALYSIS not in STAGE_OUTPUTS


def test_claims_are_handed_on_per_statement_in_text_unit_id_order(mocker) -> None:
    # Regression: the stage merged a claim's statements in document order,
    # derivation in text-unit id order, so the description's line order (and
    # a status tie) of a first build and an incremental run differed. Then
    # both merged the statements by the names they state before resolving
    # them, so statements another unit's entities would resolve elsewhere
    # merged (option S): each unit's statement is resolved on its own, and
    # claim resolution merges those that resolve to one claim.
    from unified_kg_rag.adapters.ingestion.claim_extractor import (
        ClaimExtractionStats,
    )
    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.domain.models import Claim, TextUnit, TextUnitExtraction

    def statement(unit: str) -> Claim:
        return Claim(
            id="c",
            subject_id="",
            subject_name="Vendor",
            object_name="Buyer",
            type="OWES",
            status="TRUE" if unit == "u2" else "FALSE",
            description=f"stated in {unit}",
            text_unit_ids=[unit],
        )

    extractor = mocker.patch.object(ps, "ClaimExtractor", autospec=True)
    extractor.return_value.extract_units.return_value = (
        {"u2": [statement("u2")], "u1": [statement("u1")]},
        ClaimExtractionStats(),
    )
    store = mocker.patch.object(ps, "ExtractionStore", autospec=True)
    store.return_value.load.return_value = {}
    store.return_value.save.return_value = set()
    stage = ps.ClaimExtractionStage(Config(), providers=mocker.MagicMock())
    context = PipelineContext(
        pipeline_id="p",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory="/corpus",
        # Document order puts u2 first.
        text_units=[TextUnit(id="u2", text="b"), TextUnit(id="u1", text="a")],
        unit_extractions=[
            TextUnitExtraction(text_unit_id="u2"),
            TextUnitExtraction(text_unit_id="u1"),
        ],
    )

    stage._execute_core(context)

    assert [(c.text_unit_ids, c.description) for c in context.claims] == [
        (["u1"], "stated in u1"),
        (["u2"], "stated in u2"),
    ]


@pytest.mark.parametrize("stage_name", ["claim_extraction", "gleaning"])
def test_a_stage_hands_on_the_units_failed_for_their_own_error(
    mocker, stage_name: str
) -> None:
    # Only the units failed for an error of their own count against their
    # document (indexing.max_document_failures); a stage that left them out
    # of the context counted none of its failures.
    from unified_kg_rag.adapters.ingestion.claim_extractor import (
        ClaimExtractionStats,
    )
    from unified_kg_rag.adapters.ingestion.gleaner import GleaningStats
    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.domain.models import Config, TextUnit, TextUnitExtraction

    failed = {"failed_text_unit_ids": ["u1", "u2"], "item_failed_text_unit_ids": ["u1"]}
    context = _context()
    context.text_units = [TextUnit(id=u, text=u) for u in ("u1", "u2")]
    context.unit_extractions = [TextUnitExtraction(text_unit_id="u1")]
    if stage_name == "gleaning":
        gleaner = mocker.MagicMock()
        mocker.patch.object(ps, "GraphGleaner", return_value=gleaner)
        gleaner.glean.return_value = (context.unit_extractions, GleaningStats(**failed))
        stage: Any = ps.GleaningStage(config=Config())
    else:
        extractor = mocker.patch.object(ps, "ClaimExtractor", autospec=True)
        extractor.return_value.extract_units.return_value = (
            {},
            ClaimExtractionStats(**failed),
        )
        store = mocker.patch.object(ps, "ExtractionStore", autospec=True)
        store.return_value.load.return_value = {}
        store.return_value.save.return_value = set()
        stage = ps.ClaimExtractionStage(Config(), providers=mocker.MagicMock())

    stage._execute_core(context)

    assert context.failed_text_unit_ids[stage_name] == ["u1", "u2"]
    assert context.item_failed_text_unit_ids[stage_name] == ["u1"]
