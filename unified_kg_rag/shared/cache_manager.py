# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import uuid
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from unified_kg_rag.domain.models import (
    CacheEntry,
    CacheIndex,
    CacheStats,
    CacheStrategy,
    Config,
)
from unified_kg_rag.shared.utils import compute_hash

from .logging import get_logger

logger = get_logger(__name__)
T = TypeVar("T", bound=BaseModel)


class _CorruptCacheEntry(Exception):
    """A cached entry's files are missing, truncated or fail their hash."""


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so a reader sees the old file or the new one.

    The data goes to a uniquely named sibling first and is then renamed over
    ``path`` (atomic on POSIX and Windows), so a crash mid-write leaves the
    previous file intact instead of a truncated one. The temp name does not end
    in ``.json``, so the S3 cache sync never uploads one.
    """
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(text, encoding="utf-8")
        temp_path.replace(path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


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
            "Initialized Cache Manager at '%s' with strategy='%s', TTL=%s, chunk_size=%s, max_file_size=%sMB, chunking_enabled=%s",
            self.cache_directory,
            strategy.value,
            ttl_display,
            chunk_size,
            max_file_size_mb,
            enable_chunking,
        )

    def cache_exists(self, cache_key: str, pipeline_id: str) -> bool:
        if self.strategy == CacheStrategy.FORCE_REFRESH:
            return False

        index = self.load_cache_index(pipeline_id)
        entry = index.get_entry(cache_key)
        if entry is None:
            return False

        if entry.is_expired:
            return False

        # A resume trusts this answer to skip the stage, so the entry must be
        # complete and intact, not merely present: a missing, truncated or
        # rewritten file is a miss and the stage is recomputed.
        try:
            if entry.metadata.get("is_chunked", False):
                for _ in self._iter_chunks(entry, pipeline_id):
                    pass
            else:
                self._read_single_file(entry, pipeline_id)
        except _CorruptCacheEntry as e:
            logger.warning(
                "Cache entry '%s' is incomplete, treating it as a miss: %s",
                cache_key,
                e,
            )
            return False
        return True

    def _read_single_file(self, entry: CacheEntry, pipeline_id: str) -> str:
        """Read a single-file entry, verifying its content hash.

        Raises:
            _CorruptCacheEntry: The file is missing or its hash does not match.
        """
        path = entry.local_path
        if path is None or not path.is_file():
            raise _CorruptCacheEntry(f"missing file '{path}'")
        content = path.read_text(encoding="utf-8")
        if entry.content_hash and compute_hash(content) != entry.content_hash:
            raise _CorruptCacheEntry(f"content hash mismatch in '{path}'")
        return content

    def _iter_chunks(self, entry: CacheEntry, pipeline_id: str) -> Iterator[str]:
        """Yield a chunked entry's chunk files in order, each hash-verified.

        Every chunk file is checked to exist before the first is yielded, so a
        caller that stops early (``max_items``) still never treats an entry
        with a missing chunk as a hit.

        Raises:
            _CorruptCacheEntry: The chunk count is invalid, or a chunk file is
                missing or fails its recorded hash.
        """
        chunks_dir = entry.local_path
        chunk_count = entry.metadata.get("chunk_count")
        if (
            chunks_dir is None
            or not isinstance(chunk_count, int)
            or isinstance(chunk_count, bool)
            or chunk_count < 1
        ):
            raise _CorruptCacheEntry(f"invalid chunk count {chunk_count!r}")
        # Entries written by earlier versions may lack hashes; they are still
        # checked for missing chunks and unparseable JSON.
        chunk_hashes = entry.metadata.get("chunk_hashes")
        if chunk_hashes is not None and len(chunk_hashes) != chunk_count:
            raise _CorruptCacheEntry(
                f"{len(chunk_hashes)} chunk hashes recorded for {chunk_count} chunks"
            )
        chunk_files = [
            chunks_dir / f"{entry.key}_chunk_{i:04d}.json" for i in range(chunk_count)
        ]
        missing = [f.name for f in chunk_files if not f.is_file()]
        if missing:
            raise _CorruptCacheEntry(f"missing chunk file(s) {', '.join(missing)}")
        for i, chunk_file in enumerate(chunk_files):
            content = chunk_file.read_text(encoding="utf-8")
            if chunk_hashes is not None:
                if compute_hash(content) != chunk_hashes[i]:
                    raise _CorruptCacheEntry(
                        f"content hash mismatch in '{chunk_file.name}'"
                    )
            else:
                try:
                    json.loads(content)
                except json.JSONDecodeError as e:
                    raise _CorruptCacheEntry(
                        f"unparseable chunk '{chunk_file.name}': {e}"
                    ) from e
            yield content

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
                "Failed to load cache index '%s', creating a new one. Error: %s",
                index_path,
                e,
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

            if entry is None:
                self.stats.record_miss(cache_key)
                return None

            if self.strategy == CacheStrategy.FORCE_REFRESH or entry.is_expired:
                log_msg = (
                    "forcing refresh"
                    if self.strategy == CacheStrategy.FORCE_REFRESH
                    else "entry expired"
                )
                logger.debug("Cache miss for key '%s' due to %s", cache_key, log_msg)
                self.stats.record_miss(cache_key)
                return None

            try:
                if entry.metadata.get("is_chunked", False):
                    data = self._load_chunked_data(
                        entry, pipeline_id, data_type, chunk_filter, max_items
                    )
                else:
                    data = self._load_single_file_data(entry, pipeline_id, data_type)
            except _CorruptCacheEntry as e:
                logger.warning(
                    "Cache entry '%s' is incomplete, treating it as a miss: %s",
                    cache_key,
                    e,
                )
                self.stats.record_miss(cache_key)
                return None

            self.stats.record_hit(cache_key)
            logger.debug("Cache hit for key '%s'", cache_key)
            return data
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load cache entry '%s': %s", cache_key, e)
            self.stats.record_miss(cache_key)
            return None

    def _load_single_file_data(
        self, entry: CacheEntry, pipeline_id: str, data_type: type[T] | None = None
    ) -> Any:
        content = self._read_single_file(entry, pipeline_id)
        return (
            self._deserialize_data(content, data_type)
            if data_type
            else json.loads(content)
        )

    def _load_chunked_data(
        self,
        entry: CacheEntry,
        pipeline_id: str,
        data_type: type[T] | None = None,
        chunk_filter: Callable | None = None,
        max_items: int | None = None,
    ) -> list[Any]:
        """Load a chunked entry; any bad chunk makes the whole entry a miss.

        Raises:
            _CorruptCacheEntry: A chunk is missing, fails its hash, or does not
                parse or validate as ``data_type``. Skipping it would return
                part of the stage output as if it were all of it.
        """
        chunk_count = entry.metadata.get("chunk_count", 0)
        all_data: list[Any] = []
        items_loaded = 0

        for i, content in enumerate(self._iter_chunks(entry, pipeline_id)):
            if max_items and items_loaded >= max_items:
                break
            try:
                chunk_data = json.loads(content)
                if chunk_filter:
                    chunk_data = [item for item in chunk_data if chunk_filter(item)]
                if max_items:
                    chunk_data = chunk_data[: max_items - items_loaded]
                if data_type and chunk_data:
                    chunk_data = [data_type.model_validate(item) for item in chunk_data]
            except Exception as e:
                raise _CorruptCacheEntry(f"chunk {i} could not be loaded: {e}") from e
            all_data.extend(chunk_data)
            items_loaded += len(chunk_data)

        if chunk_count > 1:
            logger.debug("Loaded %s items from %s chunks", items_loaded, chunk_count)

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
                "Could not deserialize data into '%s', returning raw data. Error: %s",
                data_type.__name__,
                e,
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
            logger.error("Failed to save cache entry '%s': %s", cache_key, e)
            return None

    def _should_chunk_data(self, data: Any) -> bool:
        if not isinstance(data, list):
            return False

        data_length = len(data)
        if data_length == 0:
            return False

        if data_length > self.chunk_size:
            logger.debug(
                "Data size (%s) exceeds chunk size (%s), will chunk",
                data_length,
                self.chunk_size,
            )
            return True

        try:
            sample_size = min(10, data_length)
            sample_data = data[:sample_size]
            # Estimate with the SAME serialization used to write (serialize_data,
            # indent=2). json.dumps without indent under-counts the on-disk size,
            # so a file that will exceed the limit could be written un-chunked.
            sample_json = self.serialize_data(sample_data)
            estimated_size = (len(sample_json) / sample_size) * data_length
            estimated_size_mb = estimated_size / 1024 / 1024

            if estimated_size > self.max_file_size_bytes:
                logger.debug(
                    "Estimated file size (%.2f MB) exceeds limit (%.2f MB), will chunk",
                    estimated_size_mb,
                    self.max_file_size_bytes / 1024 / 1024,
                )
                return True
        except Exception as e:
            logger.warning("Failed to estimate data size: %s", e)

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

        _atomic_write_text(cache_file, serialized_data)

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
            "Cached single file for key '%s' (size: %s bytes)",
            cache_key,
            entry.file_size,
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

        # Chunks first, the index last: until the index names the new chunk
        # count and hashes, a reader still checks against the old entry, and a
        # chunk already replaced fails that check (a miss) instead of mixing
        # old and new data.
        for i, chunk in enumerate(chunks):
            chunk_file = stage_cache_dir / f"{cache_key}_chunk_{i:04d}.json"
            chunk_json = self.serialize_data(chunk)

            _atomic_write_text(chunk_file, chunk_json)

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
            "Cached chunked data for key '%s' (%s chunks, %s bytes total, %s records)",
            cache_key,
            chunk_count,
            total_size,
            len(data),
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

        try:
            _atomic_write_text(index_path, index.model_dump_json(indent=2))
        except OSError as e:
            logger.error("Failed to save cache index to '%s': %s", index_path, e)
