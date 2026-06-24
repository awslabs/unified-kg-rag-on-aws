# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for storage/base.py: IndexingStats and BaseIndexer suffix logic."""

from __future__ import annotations

import pytest

from aws_graphrag.domain.models import Constants, Entity
from aws_graphrag.ports.indexer import BaseIndexer, IndexingStats

pytestmark = pytest.mark.unit


class TestIndexingStats:
    def test_rates_with_zero_total(self) -> None:
        stats = IndexingStats()
        assert stats.success_rate == 0.0
        assert stats.error_rate == 0.0

    def test_success_and_error_rates(self) -> None:
        stats = IndexingStats(total_items=10)
        stats.add_success(7)
        stats.add_error("boom", count=3)
        assert stats.success_rate == 0.7
        assert stats.error_rate == 0.3

    def test_add_error_deduplicates_messages(self) -> None:
        stats = IndexingStats()
        stats.add_error("same")
        stats.add_error("same")
        assert stats.errors == ["same"]
        assert stats.failed_items == 2

    def test_merge_combines_counts_and_unions_errors(self) -> None:
        a = IndexingStats(total_items=2, successful_items=2)
        a.add_error("e1")
        b = IndexingStats(total_items=3, successful_items=1)
        b.add_error("e2")
        a.merge(b)
        assert a.total_items == 5
        assert a.successful_items == 3
        assert set(a.errors) == {"e1", "e2"}

    def test_to_dict_caps_sample_errors(self) -> None:
        stats = IndexingStats(total_items=10)
        for i in range(7):
            stats.add_error(f"err{i}")
        result = stats.to_dict()
        assert result["error_count"] == 7
        assert len(result["sample_errors"]) == 5


class TestSuffix:
    def test_default_suffix_when_no_attributes(self) -> None:
        entity = Entity(id="e1", name="X")
        assert BaseIndexer.get_suffix(entity) == Constants.DEFAULT_SUFFIX.value

    def test_suffix_from_string_index_attribute(self) -> None:
        entity = Entity(
            id="e1", name="X", attributes={Constants.INDEX.value: "tenant1"}
        )
        assert BaseIndexer.get_suffix(entity) == "tenant1"

    def test_suffix_from_list_index_attribute(self) -> None:
        entity = Entity(
            id="e1", name="X", attributes={Constants.INDEX.value: ["tenant2", "other"]}
        )
        assert BaseIndexer.get_suffix(entity) == "tenant2"

    def test_invalid_suffix_format_raises(self) -> None:
        entity = Entity(
            id="e1", name="X", attributes={Constants.INDEX.value: "Bad Suffix!"}
        )
        with pytest.raises(ValueError, match="Invalid suffix format"):
            BaseIndexer.get_suffix(entity)
