# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import json
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from aws_graphrag.domain.models import (
    CacheEntry,
    CacheIndex,
    CacheStats,
    CacheStrategy,
    Config,
)
from aws_graphrag.shared.utils import compute_hash

from .logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class CacheManager:
    def __init__(
        self,
        config: Config,
        cache_directory: str | Path,
        strategy: CacheStrategy = CacheStrategy.CONTENT_HASH,
        ttl_seconds: int | None = None,
        chunk_size: int = 1000,
        max_file_size_mb: int = 50,
        enable_chunking: bool = True,
    ):
        self.config = config
        self.cache_directory = Path(cache_directory)
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        self.chunk_size = chunk_size
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.enable_chunking = enable_chunking
        self.stats = CacheStats()
        self.cache_directory.mkdir(parents=True, exist_ok=True)

        ttl_display = f"{ttl_seconds}s" if ttl_seconds else "none"
        logger.info(
            f"Initialized Cache Manager at '{self.cache_directory}' with "
            f"strategy='{strategy.value}', TTL={ttl_display}, "
            f"chunk_size={chunk_size}, max_file_size={max_file_size_mb}MB, "
            f"chunking_enabled={enable_chunking}"
        )

    def cache_exists(self, cache_key: str, pipeline_id: str) -> bool:
        if self.strategy == CacheStrategy.FORCE_REFRESH:
            return False

        index = self.load_cache_index(pipeline_id)
        entry = index.get_entry(cache_key)
        if entry is None:
            return False

        if entry.metadata.get("is_chunked", False):
            return self._chunked_cache_exists(entry, pipeline_id)

        return entry.exists_locally and not entry.is_expired

    def _chunked_cache_exists(self, entry: CacheEntry, pipeline_id: str) -> bool:
        chunk_count = entry.metadata.get("chunk_count", 0)
        cache_dir = self.get_pipeline_cache_dir(pipeline_id) / entry.stage_name

        for i in range(chunk_count):
            chunk_file = cache_dir / f"{entry.key}_chunk_{i:04d}.json"
            if not chunk_file.exists():
                logger.debug(f"Missing chunk file: {chunk_file}")
                return False

        return not entry.is_expired

    def load_cache_index(self, pipeline_id: str) -> CacheIndex:
        index_path = self.get_pipeline_cache_dir(pipeline_id) / "cache_index.json"
        if not index_path.exists():
            return CacheIndex(
                pipeline_id=pipeline_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

        try:
            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)
            return CacheIndex.model_validate(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                f"Failed to load cache index '{index_path}', creating a new one. "
                f"Error: {e}"
            )
            return CacheIndex(
                pipeline_id=pipeline_id,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

    def get_cache_stats(self, pipeline_id: str | None = None) -> CacheStats:
        if not pipeline_id:
            return self.stats

        stats = CacheStats()
        index = self.load_cache_index(pipeline_id)
        stats.total_entries = len(index.entries)
        stats.total_size_bytes = sum(
            e.file_size for e in index.entries.values() if e.file_size
        )
        stats.local_entries = sum(1 for e in index.entries.values() if e.exists_locally)
        return stats

    def get_pipeline_cache_dir(self, pipeline_id: str) -> Path:
        return self.cache_directory / pipeline_id

    def load_stage_result(
        self,
        cache_key: str,
        pipeline_id: str,
        data_type: type[T] | None = None,
        chunk_filter: Callable | None = None,
        max_items: int | None = None,
    ) -> T | list[T] | Any | None:
        try:
            index = self.load_cache_index(pipeline_id)
            entry = index.get_entry(cache_key)

            if entry is None or not entry.exists_locally:
                self.stats.record_miss(cache_key)
                return None

            if self.strategy == CacheStrategy.FORCE_REFRESH or entry.is_expired:
                log_msg = (
                    "forcing refresh"
                    if self.strategy == CacheStrategy.FORCE_REFRESH
                    else "entry expired"
                )
                logger.debug(f"Cache miss for key '{cache_key}' due to {log_msg}")
                self.stats.record_miss(cache_key)
                return None

            if entry.metadata.get("is_chunked", False):
                data = self._load_chunked_data(
                    entry, data_type, chunk_filter, max_items
                )
            else:
                data = self._load_single_file_data(entry, data_type)

            if data is not None:
                self.stats.record_hit(cache_key)
                logger.debug(f"Cache hit for key '{cache_key}'")

            return data
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load cache entry '{cache_key}': {e}")
            self.stats.record_miss(cache_key)
            return None

    def _load_single_file_data(
        self, entry: CacheEntry, data_type: type[T] | None = None
    ) -> Any:
        if entry.local_path is None or not entry.local_path.exists():
            return None

        with open(entry.local_path, encoding="utf-8") as f:
            content = f.read()

        return (
            self._deserialize_data(content, data_type)
            if data_type
            else json.loads(content)
        )

    @staticmethod
    def _load_chunked_data(
        entry: CacheEntry,
        data_type: type[T] | None = None,
        chunk_filter: Callable | None = None,
        max_items: int | None = None,
    ) -> list[Any] | None:
        if entry.local_path is None or not entry.local_path.exists():
            return None

        chunk_count = entry.metadata.get("chunk_count", 0)
        chunks_dir = Path(entry.local_path)
        chunk_file_pattern = f"{entry.key}_chunk_{{:04d}}.json"
        all_data = []
        items_loaded = 0

        for i in range(chunk_count):
            if max_items and items_loaded >= max_items:
                break

            chunk_file = chunks_dir / chunk_file_pattern.format(i)
            if not chunk_file.exists():
                logger.warning(f"Missing chunk file: {chunk_file}")
                continue

            try:
                with open(chunk_file, encoding="utf-8") as f:
                    chunk_data = json.loads(f.read())

                if chunk_filter:
                    chunk_data = [item for item in chunk_data if chunk_filter(item)]

                if max_items:
                    remaining_items = max_items - items_loaded
                    chunk_data = chunk_data[:remaining_items]

                if data_type and chunk_data:
                    chunk_data = [data_type.model_validate(item) for item in chunk_data]

                all_data.extend(chunk_data)
                items_loaded += len(chunk_data)
            except Exception as e:
                logger.error(f"Failed to load chunk {i}: {e}")
                continue

        if chunk_count > 1:
            logger.debug(f"Loaded {items_loaded} items from {chunk_count} chunks")

        return all_data

    @staticmethod
    def _deserialize_data(content: str, data_type: type[T]) -> T | list[T] | Any:
        data = json.loads(content)

        try:
            if isinstance(data, list):
                return [data_type.model_validate(item) for item in data]
            if isinstance(data, dict):
                return data_type.model_validate(data)
        except Exception as e:
            logger.warning(
                f"Could not deserialize data into '{data_type.__name__}', returning "
                f"raw data. Error: {e}"
            )

        return data

    def save_stage_result(
        self,
        data: Any,
        cache_key: str,
        stage_name: str,
        pipeline_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry | None:
        try:
            cache_directory = self.get_pipeline_cache_dir(pipeline_id)
            cache_directory.mkdir(parents=True, exist_ok=True)

            if self.enable_chunking and self._should_chunk_data(data):
                return self._save_chunked_data(
                    data, cache_key, stage_name, pipeline_id, metadata
                )
            else:
                return self._save_single_file_data(
                    data, cache_key, stage_name, pipeline_id, metadata
                )
        except (OSError, TypeError) as e:
            logger.error(f"Failed to save cache entry '{cache_key}': {e}")
            return None

    def _should_chunk_data(self, data: Any) -> bool:
        if not isinstance(data, list):
            return False

        data_length = len(data)
        if data_length == 0:
            return False

        if data_length > self.chunk_size:
            logger.debug(
                f"Data size ({data_length}) exceeds chunk size ({self.chunk_size}), "
                f"will chunk"
            )
            return True

        try:
            sample_size = min(10, data_length)
            sample_data = data[:sample_size]
            sample_json = json.dumps(sample_data, default=self._json_default)
            estimated_size = (len(sample_json) / sample_size) * data_length
            estimated_size_mb = estimated_size / 1024 / 1024

            if estimated_size > self.max_file_size_bytes:
                logger.debug(
                    f"Estimated file size ({estimated_size_mb:.2f} MB) exceeds limit "
                    f"({self.max_file_size_bytes / 1024 / 1024:.2f} MB), will chunk"
                )
                return True
        except Exception as e:
            logger.warning(f"Failed to estimate data size: {e}")

        return False

    def _save_single_file_data(
        self,
        data: Any,
        cache_key: str,
        stage_name: str,
        pipeline_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        cache_directory = self.get_pipeline_cache_dir(pipeline_id)
        stage_cache_dir = cache_directory / stage_name
        stage_cache_dir.mkdir(parents=True, exist_ok=True)

        serialized_data = self.serialize_data(data)
        content_hash = compute_hash(serialized_data, length=16)
        cache_file = stage_cache_dir / f"{cache_key}.json"

        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(serialized_data)

        entry = CacheEntry(
            key=cache_key,
            stage_name=stage_name,
            pipeline_id=pipeline_id,
            local_path=cache_file,
            file_size=cache_file.stat().st_size,
            content_hash=content_hash,
            record_count=len(data) if isinstance(data, list) else 1,
            data_type=type(data).__name__,
            metadata={**(metadata or {}), "is_chunked": False},
            created_at=datetime.now(),
            expires_at=(
                datetime.now() + timedelta(seconds=self.ttl_seconds)
                if self.ttl_seconds is not None and self.ttl_seconds > 0
                else None
            ),
        )

        self._update_cache_index(entry)
        logger.info(
            f"Cached single file for key '{cache_key}' (size: {entry.file_size} bytes)"
        )
        return entry

    def _save_chunked_data(
        self,
        data: list[Any],
        cache_key: str,
        stage_name: str,
        pipeline_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        cache_directory = self.get_pipeline_cache_dir(pipeline_id)
        stage_cache_dir = cache_directory / stage_name
        stage_cache_dir.mkdir(parents=True, exist_ok=True)

        if len(data) > self.chunk_size:
            chunks = list(self._chunk_data(data, self.chunk_size))
        else:
            sample_size = min(5, len(data))
            sample_data = data[:sample_size]
            sample_json = self.serialize_data(sample_data)
            avg_size_per_item = len(sample_json) / sample_size
            target_chunk_size_bytes = self.max_file_size_bytes
            items_per_chunk = max(1, int(target_chunk_size_bytes / avg_size_per_item))
            chunks = list(self._chunk_data(data, items_per_chunk))

        chunk_count = len(chunks)
        total_size = 0
        chunk_hashes = []

        for i, chunk in enumerate(chunks):
            chunk_file = stage_cache_dir / f"{cache_key}_chunk_{i:04d}.json"
            chunk_json = self.serialize_data(chunk)

            with open(chunk_file, "w", encoding="utf-8") as f:
                f.write(chunk_json)

            chunk_size = chunk_file.stat().st_size
            total_size += chunk_size
            chunk_hashes.append(compute_hash(chunk_json, length=16))

        master_hash = compute_hash("".join(chunk_hashes), length=16)

        entry = CacheEntry(
            key=cache_key,
            stage_name=stage_name,
            pipeline_id=pipeline_id,
            local_path=stage_cache_dir,
            file_size=total_size,
            content_hash=master_hash,
            record_count=len(data),
            data_type=type(data).__name__,
            metadata={
                **(metadata or {}),
                "is_chunked": True,
                "chunk_count": chunk_count,
                "chunk_size": self.chunk_size,
                "chunk_hashes": chunk_hashes,
            },
            created_at=datetime.now(),
            expires_at=(
                datetime.now() + timedelta(seconds=self.ttl_seconds)
                if self.ttl_seconds is not None and self.ttl_seconds > 0
                else None
            ),
        )

        self._update_cache_index(entry)
        logger.info(
            f"Cached chunked data for key '{cache_key}' "
            f"({chunk_count} chunks, {total_size} bytes total, {len(data)} records)"
        )
        return entry

    @staticmethod
    def _chunk_data(data: list[Any], chunk_size: int) -> Iterator[list[Any]]:
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]

    def _update_cache_index(self, entry: CacheEntry) -> None:
        index = self.load_cache_index(entry.pipeline_id)
        index.add_entry(entry)
        self._save_cache_index(index)
        self.stats.total_entries += 1
        self.stats.total_size_bytes += entry.file_size or 0

    def get_chunk_info(self, cache_key: str, pipeline_id: str) -> dict[str, Any] | None:
        index = self.load_cache_index(pipeline_id)
        entry = index.get_entry(cache_key)

        if entry is None or not entry.metadata.get("is_chunked", False):
            return None

        cache_dir = self.get_pipeline_cache_dir(pipeline_id) / entry.stage_name
        chunk_count = entry.metadata.get("chunk_count", 0)
        chunk_info = []

        for i in range(chunk_count):
            chunk_file = cache_dir / f"{entry.key}_chunk_{i:04d}.json"
            if chunk_file.exists():
                stat = chunk_file.stat()
                chunk_info.append(
                    {
                        "chunk_id": i,
                        "file_path": str(chunk_file),
                        "file_size": stat.st_size,
                        "exists": True,
                    }
                )
            else:
                chunk_info.append(
                    {
                        "chunk_id": i,
                        "file_path": str(chunk_file),
                        "file_size": 0,
                        "exists": False,
                    }
                )

        return {
            "cache_key": cache_key,
            "pipeline_id": pipeline_id,
            "is_chunked": True,
            "chunk_count": chunk_count,
            "total_records": entry.record_count,
            "total_size": entry.file_size,
            "chunks": chunk_info,
        }

    def load_chunk_preview(
        self,
        cache_key: str,
        pipeline_id: str,
        chunk_id: int = 0,
        max_items: int = 10,
        data_type: type[T] | None = None,
    ) -> list[Any] | None:
        index = self.load_cache_index(pipeline_id)
        entry = index.get_entry(cache_key)

        if entry is None or not entry.metadata.get("is_chunked", False):
            logger.warning(f"Cache key '{cache_key}' is not chunked")
            return None

        cache_dir = self.get_pipeline_cache_dir(pipeline_id) / entry.stage_name
        chunk_file = cache_dir / f"{entry.key}_chunk_{chunk_id:04d}.json"

        if not chunk_file.exists():
            logger.warning(f"Chunk file does not exist: {chunk_file}")
            return None

        try:
            with open(chunk_file, encoding="utf-8") as f:
                chunk_data = json.loads(f.read())

            preview_data: list[Any] = chunk_data[:max_items]

            if data_type and preview_data:
                preview_data = [data_type.model_validate(item) for item in preview_data]

            return preview_data
        except Exception as e:
            logger.error(f"Failed to load chunk preview: {e}")
            return None

    def serialize_data(self, data: Any, indent: int | None = 2) -> str:
        return json.dumps(data, indent=indent, default=self._json_default)

    @staticmethod
    def _json_default(o: Any) -> Any:
        if isinstance(o, BaseModel):
            return o.model_dump()
        if isinstance(o, (Path | datetime)):
            return str(o)
        raise TypeError(
            f"Object of type '{o.__class__.__name__}' is not JSON serializable"
        )

    def _save_cache_index(self, index: CacheIndex) -> None:
        index_path = self.get_pipeline_cache_dir(index.pipeline_id) / "cache_index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = index_path.with_suffix(".json.tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                f.write(index.model_dump_json(indent=2))
            temp_path.replace(index_path)
        except OSError as e:
            logger.error(f"Failed to save cache index to '{index_path}': {e}")
