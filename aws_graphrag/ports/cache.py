# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache port — the stage-result persistence boundary for the pipeline.

The ingestion pipeline persists each stage's output (entities, relationships,
text units, ...) so a run can resume from a checkpoint and so phased Step
Functions tasks can hand results to each other. That persistence is reached
through a *cache manager* that resolves a ``(cache_key, pipeline_id)`` to stored
data, with optional content-hash invalidation and large-payload chunking.

This ``Protocol`` captures the surface the pipeline depends on so an alternate
cache backend (the local-filesystem ``CacheManager`` is the default today) can
be introduced without changing call sites. The concrete ``CacheManager`` in
``shared.cache_manager`` conforms structurally — no base-class change is needed.
Defined ``runtime_checkable`` (structural typing).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from aws_graphrag.shared.cache_manager import CacheEntry, CacheIndex, CacheStats


@runtime_checkable
class CachePort(Protocol):
    """Persistent store for pipeline stage results, keyed by (cache_key, pipeline_id)."""

    def cache_exists(self, cache_key: str, pipeline_id: str) -> bool:
        """Return whether a (valid, non-expired) entry exists for the key."""
        ...

    def save_stage_result(
        self,
        data: Any,
        cache_key: str,
        stage_name: str,
        pipeline_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry | None:
        """Persist ``data`` for the key and return the resulting entry."""
        ...

    def load_stage_result(
        self,
        cache_key: str,
        pipeline_id: str,
        data_type: type | None = None,
        chunk_filter: Callable[..., Any] | None = None,
        max_items: int | None = None,
    ) -> Any:
        """Load previously cached data for the key (``None`` on miss)."""
        ...

    def get_pipeline_cache_dir(self, pipeline_id: str) -> Path:
        """Return the (created) cache directory for a pipeline run."""
        ...

    def load_cache_index(self, pipeline_id: str) -> CacheIndex:
        """Return the index of cached entries for a pipeline run."""
        ...

    def get_cache_stats(self, pipeline_id: str | None = None) -> CacheStats:
        """Return aggregate cache statistics (optionally scoped to one run)."""
        ...
