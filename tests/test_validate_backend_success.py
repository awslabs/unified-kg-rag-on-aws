"""Tests for IndexingStage._validate_backend_success() per-index-type validation."""

from unittest.mock import MagicMock, patch

import pytest

from aws_graphrag.core.exceptions import PipelineStageError
from aws_graphrag.storage.base import IndexingStats


@pytest.fixture
def indexing_stage():
    """Create an IndexingStage instance with mocked dependencies."""
    with patch("aws_graphrag.ingestion.pipeline_stages.boto3.Session"):
        with patch("aws_graphrag.ingestion.pipeline_stages.IndexingManager"):
            from aws_graphrag.ingestion.pipeline_stages import IndexingStage

            mock_config = MagicMock()
            mock_config.aws.profile_name = "test"
            stage = IndexingStage(config=mock_config)
            return stage


def _make_stats(total: int, successful: int, failed: int) -> IndexingStats:
    stats = IndexingStats()
    stats.total_items = total
    stats.successful_items = successful
    stats.failed_items = failed
    return stats


class TestValidateBackendSuccess:
    """Tests for _validate_backend_success method."""

    def test_all_indexes_success(self, indexing_stage):
        """All 3 OpenSearch indexes succeed - no error raised."""
        results = {
            "opensearch_text_units": _make_stats(1000, 1000, 0),
            "opensearch_entities": _make_stats(500, 500, 0),
            "opensearch_community_reports": _make_stats(200, 200, 0),
            "neptune_entities": _make_stats(500, 500, 0),
            "neptune_relationships": _make_stats(300, 300, 0),
        }
        # Should not raise
        indexing_stage._validate_backend_success(results)

    def test_one_opensearch_index_fails(self, indexing_stage):
        """community_reports fails completely while others succeed - should raise."""
        results = {
            "opensearch_text_units": _make_stats(1000, 1000, 0),
            "opensearch_entities": _make_stats(500, 500, 0),
            "opensearch_community_reports": _make_stats(200, 0, 200),
        }
        with pytest.raises(PipelineStageError, match="opensearch_community_reports"):
            indexing_stage._validate_backend_success(results)

    def test_one_neptune_index_fails(self, indexing_stage):
        """neptune_entities fails completely while others succeed - should raise."""
        results = {
            "opensearch_text_units": _make_stats(1000, 1000, 0),
            "opensearch_entities": _make_stats(500, 500, 0),
            "opensearch_community_reports": _make_stats(200, 200, 0),
            "neptune_entities": _make_stats(500, 0, 500),
            "neptune_relationships": _make_stats(300, 300, 0),
        }
        with pytest.raises(PipelineStageError, match="neptune_entities"):
            indexing_stage._validate_backend_success(results)

    def test_all_indexes_fail(self, indexing_stage):
        """All indexes fail - should raise with all names listed."""
        results = {
            "opensearch_text_units": _make_stats(1000, 0, 1000),
            "opensearch_entities": _make_stats(500, 0, 500),
            "opensearch_community_reports": _make_stats(200, 0, 200),
        }
        with pytest.raises(PipelineStageError) as exc_info:
            indexing_stage._validate_backend_success(results)
        error_msg = str(exc_info.value)
        assert "opensearch_text_units" in error_msg
        assert "opensearch_entities" in error_msg
        assert "opensearch_community_reports" in error_msg

    def test_empty_results(self, indexing_stage):
        """Empty results dict - no error raised."""
        indexing_stage._validate_backend_success({})

    def test_zero_total_items_not_flagged(self, indexing_stage):
        """Index with 0 total items should not be flagged as failed."""
        results = {
            "opensearch_text_units": _make_stats(1000, 1000, 0),
            "opensearch_entities": _make_stats(0, 0, 0),
            "opensearch_community_reports": _make_stats(200, 200, 0),
        }
        # Should not raise - zero total means nothing was attempted
        indexing_stage._validate_backend_success(results)

    def test_partial_failure_not_flagged(self, indexing_stage):
        """Index with partial success (some items succeed) should not raise."""
        results = {
            "opensearch_text_units": _make_stats(1000, 900, 100),
            "opensearch_entities": _make_stats(500, 250, 250),
            "opensearch_community_reports": _make_stats(200, 1, 199),
        }
        # Should not raise - at least 1 item succeeded per index
        indexing_stage._validate_backend_success(results)

    def test_none_stats_ignored(self, indexing_stage):
        """None stats values should be safely ignored."""
        results = {
            "opensearch_text_units": _make_stats(1000, 1000, 0),
            "opensearch_entities": None,
            "opensearch_community_reports": _make_stats(200, 200, 0),
        }
        # Should not raise
        indexing_stage._validate_backend_success(results)

    def test_multiple_indexes_fail_lists_all(self, indexing_stage):
        """Two out of three indexes fail - error message lists both."""
        results = {
            "opensearch_text_units": _make_stats(1000, 1000, 0),
            "opensearch_entities": _make_stats(500, 0, 500),
            "opensearch_community_reports": _make_stats(200, 0, 200),
        }
        with pytest.raises(PipelineStageError) as exc_info:
            indexing_stage._validate_backend_success(results)
        error_msg = str(exc_info.value)
        assert "opensearch_entities" in error_msg
        assert "opensearch_community_reports" in error_msg
        # text_units succeeded, should NOT be in error
        assert "opensearch_text_units" not in error_msg
