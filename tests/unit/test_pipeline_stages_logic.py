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

from aws_graphrag.domain.models import (
    Config,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)
from aws_graphrag.ports.indexer import IndexingStats
from aws_graphrag.shared import PipelineStageError

pytestmark = pytest.mark.unit


# A minimal concrete stage to exercise the abstract base's shared helpers.
def _make_stage(stage_type: PipelineStageType):
    from aws_graphrag.application.ingestion.pipeline_stages import PipelineStage

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


def test_should_validate_output_matrix() -> None:
    critical = _make_stage(PipelineStageType.INDEXING)
    optional = _make_stage(PipelineStageType.GLEANING)
    other = _make_stage(PipelineStageType.CLAIM_RESOLUTION)  # not critical/optional
    assert critical._should_validate_output() is True
    assert optional._should_validate_output() is False
    # Unlisted stage types default to requiring output validation.
    assert other._should_validate_output() is False  # CLAIM_RESOLUTION is optional


def test_should_validate_output_defaults_true_for_unlisted() -> None:
    # GRAPH_ANALYSIS is critical; pick a type that's neither critical nor optional.
    stage = _make_stage(PipelineStageType.DOCUMENT_LOADING)  # critical -> True
    assert stage._should_validate_output() is True


# --- _allows_empty_output ------------------------------------------------


def test_allows_empty_output_when_incremental_delta_empty_docs() -> None:
    from aws_graphrag.domain.models.document import DocumentDelta

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


def test_disallows_empty_output_when_delta_has_docs() -> None:
    from aws_graphrag.domain.models import Document
    from aws_graphrag.domain.models.document import DocumentDelta

    stage = _make_stage(PipelineStageType.GRAPH_EXTRACTION)
    ctx = _context()
    ctx.incremental_delta = DocumentDelta()
    ctx.documents = [
        Document(
            page_content="x",
            document_id="d1",
            file_name="f.txt",
            file_path="/tmp/f.txt",
            file_type="txt",
            total_pages=1,
        )
    ]
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
    assert result.cache_path is None


def test_create_result_defaults_metrics_to_empty_dict() -> None:
    stage = _make_stage(PipelineStageType.TEXT_CHUNKING)
    start = end = datetime(2026, 1, 1)
    result = stage._create_result(
        PipelineStageStatus.FAILED, start, end, error_message="boom"
    )
    assert result.metrics == {}
    assert result.error_message == "boom"


# --- _stats_to_dict ------------------------------------------------------


def test_stats_to_dict_none_returns_empty() -> None:
    from aws_graphrag.application.ingestion.pipeline_stages import PipelineStage

    assert PipelineStage._stats_to_dict(None) == {}


def test_stats_to_dict_uses_to_dict_when_available() -> None:
    from aws_graphrag.application.ingestion.pipeline_stages import PipelineStage

    class _Stats:
        def to_dict(self) -> dict[str, Any]:
            return {"a": 1}

    assert PipelineStage._stats_to_dict(_Stats()) == {"a": 1}


def test_stats_to_dict_falls_back_to_dunder_dict() -> None:
    from aws_graphrag.application.ingestion.pipeline_stages import PipelineStage

    class _Stats:
        def __init__(self) -> None:
            self.x = 7

    assert PipelineStage._stats_to_dict(_Stats()) == {"x": 7}


# --- IndexingStage metric aggregation + backend validation ---------------


def _indexing_stage(mocker, indexing_results: dict[str, IndexingStats]):
    """Build an IndexingStage with its IndexingManager fully stubbed."""
    from aws_graphrag.application.ingestion import pipeline_stages as ps

    fake_mgr = mocker.MagicMock()
    fake_mgr.initialize.return_value = True
    fake_mgr.index_all_data.return_value = indexing_results
    mocker.patch.object(ps, "IndexingManager", return_value=fake_mgr)

    cfg = Config()
    cfg.indexing.reset = False
    stage = ps.IndexingStage(config=cfg, boto_session=mocker.MagicMock())
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
    stage, _mgr = _indexing_stage(mocker, results)

    input_count, total_indexed, metrics = stage._execute_core(_context())

    # relationships_indexed sums successes across every key containing
    # "relationship": neptune (4) + opensearch (2) = 6.
    assert metrics["relationships_indexed"] == 6
    assert metrics["total_indexed"] == 3 + 4 + 2 + 2
    assert metrics["total_failed"] == 2  # opensearch_relationships failed_items
    assert total_indexed == metrics["total_indexed"]


def test_indexing_stage_success_rate_zero_when_nothing(mocker) -> None:
    stage, _mgr = _indexing_stage(mocker, {})
    _in, total, metrics = stage._execute_core(_context())
    assert total == 0
    assert metrics["success_rate"] == 0
    assert metrics["relationships_indexed"] == 0


def test_indexing_stage_input_count_sums_all_artifact_types(mocker) -> None:
    from aws_graphrag.domain.models import Entity, Relationship, TextUnit

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


def test_validate_backend_success_raises_on_backend_total_failure(mocker) -> None:
    stage, _mgr = _indexing_stage(mocker, {})
    # Each individual key has SOME success, but the whole Neptune backend has 0.
    # Construct so no single key is all-zero (avoids the index-type guard) yet
    # the backend aggregate is zero -> backend-level guard fires.
    results = {
        "neptune_entities": IndexingStats(total_items=2, successful_items=0),
        "neptune_relationships": IndexingStats(total_items=0, successful_items=0),
    }
    # Here neptune_entities IS all-zero, so the per-index guard fires first.
    with pytest.raises(PipelineStageError):
        stage._validate_backend_success(results)


def test_validate_backend_success_passes_with_partial_success(mocker) -> None:
    stage, _mgr = _indexing_stage(mocker, {})
    results = {
        "opensearch_entities": IndexingStats(total_items=5, successful_items=3),
        "neptune_entities": IndexingStats(total_items=5, successful_items=5),
    }
    # No exception: every index type and backend has at least one success.
    stage._validate_backend_success(results)


def test_validate_backend_success_ignores_zero_total_items(mocker) -> None:
    stage, _mgr = _indexing_stage(mocker, {})
    # total_items==0 keys are skipped (not a failure).
    results = {
        "opensearch_entities": IndexingStats(total_items=0, successful_items=0),
    }
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
