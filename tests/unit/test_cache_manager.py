# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for CacheManager — the local-disk resume/checkpoint cache on the
incremental-indexing critical path (AWS-free, tmp_path only).

These cover key/dir construction, single-file and chunked save/load round-trips,
the chunking threshold, TTL/expiry semantics, FORCE_REFRESH, and cache-index
update/read. A regression in any of these would silently re-run (or skip)
expensive pipeline stages, so each assertion guards real resume behavior.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import BaseModel

from aws_graphrag.domain.models import CacheStrategy, Config
from aws_graphrag.shared.cache_manager import CacheManager

pytestmark = pytest.mark.unit

PIPELINE = "pipe-1"
STAGE = "extract"


class _Item(BaseModel):
    name: str
    value: int


def _manager(tmp_path: Path, **kwargs) -> CacheManager:  # noqa: ANN003
    return CacheManager(Config(), cache_directory=tmp_path, **kwargs)


class TestKeyAndDirConstruction:
    def test_pipeline_cache_dir_is_nested_under_root(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        assert mgr.get_pipeline_cache_dir(PIPELINE) == tmp_path / PIPELINE

    def test_root_directory_created_on_init(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "cache"
        CacheManager(Config(), cache_directory=target)
        assert target.exists()

    def test_max_file_size_converted_to_bytes(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path, max_file_size_mb=2)
        assert mgr.max_file_size_bytes == 2 * 1024 * 1024


class TestSingleFileRoundTrip:
    def test_save_then_load_dict(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        data = {"alpha": 1, "beta": [1, 2, 3]}
        mgr.save_stage_result(data, "k1", STAGE, PIPELINE)
        loaded = mgr.load_stage_result("k1", PIPELINE)
        assert loaded == data

    def test_save_writes_single_json_file(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        entry = mgr.save_stage_result({"x": 1}, "k1", STAGE, PIPELINE)
        assert entry is not None
        assert entry.metadata["is_chunked"] is False
        assert entry.local_path == tmp_path / PIPELINE / STAGE / "k1.json"
        assert entry.local_path.exists()

    def test_round_trip_with_pydantic_data_type(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        items = [_Item(name="a", value=1), _Item(name="b", value=2)]
        mgr.save_stage_result(items, "items", STAGE, PIPELINE)
        loaded = mgr.load_stage_result("items", PIPELINE, data_type=_Item)
        assert loaded == items
        assert all(isinstance(i, _Item) for i in loaded)

    def test_load_missing_key_records_miss(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        assert mgr.load_stage_result("nope", PIPELINE) is None
        assert mgr.stats.miss_count == 1

    def test_load_existing_records_hit(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        mgr.save_stage_result({"x": 1}, "k1", STAGE, PIPELINE)
        mgr.load_stage_result("k1", PIPELINE)
        assert mgr.stats.hit_count == 1

    def test_record_count_for_list_vs_scalar(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        list_entry = mgr.save_stage_result([1, 2, 3], "lst", STAGE, PIPELINE)
        dict_entry = mgr.save_stage_result({"x": 1}, "dct", STAGE, PIPELINE)
        assert list_entry.record_count == 3
        assert dict_entry.record_count == 1


class TestChunking:
    def test_list_over_chunk_size_is_chunked(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path, chunk_size=3)
        data = list(range(10))  # > chunk_size 3 -> ceil(10/3) = 4 chunks
        entry = mgr.save_stage_result(data, "big", STAGE, PIPELINE)
        assert entry.metadata["is_chunked"] is True
        assert entry.metadata["chunk_count"] == 4

    def test_chunked_round_trip_preserves_order_and_items(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path, chunk_size=3)
        data = list(range(10))
        mgr.save_stage_result(data, "big", STAGE, PIPELINE)
        loaded = mgr.load_stage_result("big", PIPELINE)
        assert loaded == data

    def test_chunked_files_written_with_index_naming(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path, chunk_size=2)
        mgr.save_stage_result([1, 2, 3, 4, 5], "c", STAGE, PIPELINE)
        stage_dir = tmp_path / PIPELINE / STAGE
        assert (stage_dir / "c_chunk_0000.json").exists()
        assert (stage_dir / "c_chunk_0001.json").exists()
        assert (stage_dir / "c_chunk_0002.json").exists()

    def test_small_list_below_threshold_not_chunked(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path, chunk_size=1000)
        entry = mgr.save_stage_result([1, 2, 3], "small", STAGE, PIPELINE)
        assert entry.metadata["is_chunked"] is False

    def test_size_estimate_uses_indented_serialization(self, tmp_path: Path) -> None:
        # Regression: _should_chunk_data estimated with non-indented json.dumps
        # while the write uses serialize_data(indent=2), undershooting the on-disk
        # size. A list UNDER chunk_size but whose indented bytes exceed the file
        # limit must still chunk. Each item is a dict so indent=2 inflates size.
        mgr = _manager(tmp_path, chunk_size=100_000, max_file_size_mb=1)
        # ~8000 dicts; indented JSON comfortably exceeds 1 MB (well above the
        # ~946 KB a smaller set produced) so the size-based trigger fires.
        data = [{"id": i, "v": "x" * 200} for i in range(8000)]
        entry = mgr.save_stage_result(data, "bysize", STAGE, PIPELINE)
        assert entry.metadata["is_chunked"] is True
        # Round-trips intact despite size-triggered chunking.
        assert mgr.load_stage_result("bysize", PIPELINE) == data

    def test_chunking_disabled_keeps_single_file(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path, chunk_size=2, enable_chunking=False)
        entry = mgr.save_stage_result([1, 2, 3, 4, 5], "k", STAGE, PIPELINE)
        assert entry.metadata["is_chunked"] is False

    def test_chunked_max_items_truncates_load(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path, chunk_size=2)
        mgr.save_stage_result(list(range(10)), "c", STAGE, PIPELINE)
        loaded = mgr.load_stage_result("c", PIPELINE, max_items=3)
        assert loaded == [0, 1, 2]

    def test_chunked_filter_applied_on_load(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path, chunk_size=2)
        mgr.save_stage_result(list(range(10)), "c", STAGE, PIPELINE)
        loaded = mgr.load_stage_result("c", PIPELINE, chunk_filter=lambda x: x % 2 == 0)
        assert loaded == [0, 2, 4, 6, 8]

    def test_estimated_size_threshold_triggers_chunking(self, tmp_path: Path) -> None:
        # Below chunk_size count, but each item is large enough that the
        # estimated total exceeds the tiny max_file_size limit -> chunk.
        mgr = _manager(tmp_path, chunk_size=10000, max_file_size_mb=0)
        data = ["x" * 500 for _ in range(20)]
        entry = mgr.save_stage_result(data, "fat", STAGE, PIPELINE)
        assert entry.metadata["is_chunked"] is True


class TestExpiryAndStrategy:
    def test_expired_entry_is_a_miss(self, tmp_path: Path) -> None:
        # Write with a positive TTL, then rewind its expiry into the past so the
        # is_expired check fires (TTL is only set when ttl_seconds > 0).
        mgr = _manager(tmp_path, ttl_seconds=3600)
        mgr.save_stage_result({"x": 1}, "k", STAGE, PIPELINE)
        index = mgr.load_cache_index(PIPELINE)
        index.entries["k"].expires_at = datetime.now() - timedelta(seconds=1)
        mgr._save_cache_index(index)
        assert mgr.load_stage_result("k", PIPELINE) is None
        assert mgr.cache_exists("k", PIPELINE) is False

    def test_non_expiring_entry_when_no_ttl(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)  # ttl_seconds None
        entry = mgr.save_stage_result({"x": 1}, "k", STAGE, PIPELINE)
        assert entry.expires_at is None
        assert mgr.cache_exists("k", PIPELINE) is True

    def test_positive_ttl_sets_future_expiry(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path, ttl_seconds=3600)
        entry = mgr.save_stage_result({"x": 1}, "k", STAGE, PIPELINE)
        assert entry.expires_at is not None
        assert mgr.load_stage_result("k", PIPELINE) == {"x": 1}

    def test_force_refresh_treats_everything_as_miss(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path, strategy=CacheStrategy.FORCE_REFRESH)
        mgr.save_stage_result({"x": 1}, "k", STAGE, PIPELINE)
        assert mgr.cache_exists("k", PIPELINE) is False
        assert mgr.load_stage_result("k", PIPELINE) is None


class TestCacheIndex:
    def test_index_created_on_save_and_persisted(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        mgr.save_stage_result({"x": 1}, "k", STAGE, PIPELINE)
        index_path = tmp_path / PIPELINE / "cache_index.json"
        assert index_path.exists()

    def test_missing_index_returns_fresh_empty(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        index = mgr.load_cache_index("never-written")
        assert index.entries == {}
        assert index.pipeline_id == "never-written"

    def test_index_reloaded_by_new_manager(self, tmp_path: Path) -> None:
        # Persisted index must survive a fresh manager (the resume scenario).
        _manager(tmp_path).save_stage_result({"x": 1}, "k", STAGE, PIPELINE)
        reloaded = _manager(tmp_path)
        assert reloaded.cache_exists("k", PIPELINE) is True
        assert reloaded.load_stage_result("k", PIPELINE) == {"x": 1}

    def test_corrupt_index_falls_back_to_fresh(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        index_path = tmp_path / PIPELINE / "cache_index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text("{ not json", encoding="utf-8")
        index = mgr.load_cache_index(PIPELINE)
        assert index.entries == {}

    def test_multiple_entries_accumulate_in_index(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        mgr.save_stage_result({"x": 1}, "k1", STAGE, PIPELINE)
        mgr.save_stage_result({"y": 2}, "k2", STAGE, PIPELINE)
        index = mgr.load_cache_index(PIPELINE)
        assert set(index.entries) == {"k1", "k2"}


class TestStats:
    def test_per_pipeline_stats_aggregate_from_index(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        mgr.save_stage_result([1, 2], "k1", STAGE, PIPELINE)
        mgr.save_stage_result([3, 4], "k2", STAGE, PIPELINE)
        stats = mgr.get_cache_stats(PIPELINE)
        assert stats.total_entries == 2
        assert stats.local_entries == 2
        assert stats.total_size_bytes > 0

    def test_default_stats_without_pipeline(self, tmp_path: Path) -> None:
        mgr = _manager(tmp_path)
        mgr.save_stage_result({"x": 1}, "k", STAGE, PIPELINE)
        # No pipeline_id -> returns the live, mutated stats object.
        assert mgr.get_cache_stats() is mgr.stats
        assert mgr.stats.total_entries == 1
