# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class CacheStrategy(str, Enum):
    CONTENT_HASH = "content_hash"
    FORCE_REFRESH = "force_refresh"


class CacheEntry(BaseModel):
    key: str = Field(description="Unique identifier for the cache entry")
    stage_name: str = Field(description="Pipeline stage that created this cache entry")
    pipeline_id: str = Field(
        description="Unique pipeline ID this cache entry belongs to"
    )
    created_at: datetime = Field(description="When the cache entry was created")
    expires_at: datetime | None = Field(
        None,
        description="When the cache entry expires (None if never expires)",
    )
    local_path: Path | None = Field(
        None, description="Local filesystem path where cached data is stored"
    )
    file_size: int = Field(0, description="Size of cached file in bytes")
    content_hash: str = Field(
        "", description="Hash of cached content for integrity verification"
    )
    record_count: int = Field(
        0, description="Number of records contained in cached data"
    )
    data_type: str = Field(
        "",
        description="Type of data stored in cache (e.g., 'entities', 'relationships')",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the cache entry",
    )

    model_config = {"arbitrary_types_allowed": True}

    @property
    def exists_locally(self) -> bool:
        if "is_chunked" in self.metadata and self.metadata["is_chunked"]:
            return True
        if self.local_path is None:
            return False
        return Path(self.local_path).exists()

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at


class CacheIndex(BaseModel):
    pipeline_id: str = Field(
        description="Unique identifier of the pipeline this cache index belongs to"
    )
    created_at: datetime = Field(
        description="Timestamp when the cache index was created"
    )
    updated_at: datetime = Field(
        description="Timestamp when the cache index was last updated"
    )
    entries: dict[str, CacheEntry] = Field(
        default_factory=dict,
        description="Dictionary mapping cache keys to their corresponding CacheEntry objects",
    )

    def add_entry(self, entry: CacheEntry) -> None:
        self.entries[entry.key] = entry
        self.updated_at = datetime.now()

    def get_entry(self, key: str) -> CacheEntry | None:
        return self.entries[key] if key in self.entries else None


class CacheStats(BaseModel):
    total_entries: int = Field(
        default=0, description="Total number of entries in the cache"
    )
    total_size_bytes: int = Field(
        default=0, description="Total size of all cache entries in bytes"
    )
    hit_count: int = Field(default=0, description="Number of cache hits")
    miss_count: int = Field(default=0, description="Number of cache misses")
    stage_stats: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Per-stage statistics tracking hits and misses for each pipeline stage",
    )
    local_entries: int = Field(
        default=0, description="Number of cache entries stored locally"
    )
    s3_entries: int = Field(
        default=0, description="Number of cache entries stored in S3"
    )

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)

    def record_hit(self, stage_name: str) -> None:
        self.hit_count += 1
        if stage_name not in self.stage_stats:
            self.stage_stats[stage_name] = {"hits": 0, "misses": 0}
        self.stage_stats[stage_name]["hits"] += 1

    def record_miss(self, stage_name: str) -> None:
        self.miss_count += 1
        if stage_name not in self.stage_stats:
            self.stage_stats[stage_name] = {"hits": 0, "misses": 0}
        self.stage_stats[stage_name]["misses"] += 1
