# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""S3-persisted content-hash embedding cache.

Each Step Functions phase is a fresh Fargate process, so the in-process
embedding cache provides no cross-phase/run benefit and the corpus is re-embedded
every run. This optional cache loads a ``{content_hash: vector}`` map from a
single S3 object once, serves lookups in memory, and flushes newly-computed
vectors back. Keyed by the same content hash the indexer already computes, and
namespaced by embedding model + dimension so a model/dim change can't return
stale vectors.

Best-effort: any S3 error degrades to an in-memory-only cache (load returns
empty, flush is skipped) rather than failing the run.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import boto3

from unified_kg_rag.shared import get_logger

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

logger = get_logger(__name__)


class S3EmbeddingCache:
    """A hash->vector embedding cache backed by a single S3 JSON object."""

    def __init__(
        self,
        bucket_name: str,
        key: str,
        model_id: str,
        dimension: int,
        boto_session: boto3.Session | None = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.key = key
        # Namespace entries so a model/dimension change never returns a stale
        # vector of the wrong shape/semantics.
        self._namespace = f"{model_id}:{dimension}"
        self._session = boto_session or boto3.Session()
        self._client: S3Client | None = None
        self._cache: dict[str, list[float]] = {}
        # Keys this process computed since the last load — flushed by merging
        # onto the freshest remote state so a concurrent writer's entries are
        # preserved rather than clobbered by a whole-object overwrite.
        self._pending: dict[str, list[float]] = {}
        self._loaded = False
        self._dirty = False

    @property
    def client(self) -> S3Client:
        if self._client is None:
            self._client = self._session.client("s3")
        return self._client

    def _namespaced(self, content_hash: str) -> str:
        return f"{self._namespace}|{content_hash}"

    def _read_remote(self) -> dict[str, list[float]]:
        """Read the current persisted map from S3 (all namespaces), or {}."""
        try:
            obj = self.client.get_object(Bucket=self.bucket_name, Key=self.key)
            data = json.loads(obj["Body"].read())
            return data if isinstance(data, dict) else {}
        except Exception as e:  # noqa: BLE001 - missing/unreadable object -> empty
            logger.info("Embedding cache not read from S3 (starting empty): %s", e)
            return {}

    def load(self) -> None:
        """Load the persisted cache from S3 (best-effort, once)."""
        if self._loaded:
            return
        self._loaded = True
        remote = self._read_remote()
        # Only keep entries for the current model/dimension namespace.
        prefix = f"{self._namespace}|"
        self._cache = {k: v for k, v in remote.items() if k.startswith(prefix)}
        if self._cache:
            logger.info(
                "Loaded %s embedding-cache entries from 's3://%s/%s'",
                len(self._cache),
                self.bucket_name,
                self.key,
            )

    def get(self, content_hash: str) -> list[float] | None:
        return self._cache.get(self._namespaced(content_hash))

    def put(self, content_hash: str, vector: list[float]) -> None:
        key = self._namespaced(content_hash)
        self._cache[key] = vector
        self._pending[key] = vector
        self._dirty = True

    def flush(self) -> None:
        """Persist newly-computed vectors back to S3 (best-effort).

        Re-reads the current remote object and merges this process's pending
        entries on top before writing, so a concurrent writer's entries (in a
        different namespace, or hashes this process never saw) are preserved
        rather than clobbered by a blind whole-object overwrite. Worst case under
        a true write-write race is re-embedding a few vectors, never a wrong one.
        """
        if not self._dirty:
            return
        try:
            merged = self._read_remote()
            merged.update(self._pending)
            body = json.dumps(merged).encode("utf-8")
            self.client.put_object(Bucket=self.bucket_name, Key=self.key, Body=body)
            flushed_count = len(self._pending)
            self._pending.clear()
            self._dirty = False
            logger.info(
                "Flushed %s embedding-cache entries to 's3://%s/%s' (%s total)",
                flushed_count,
                self.bucket_name,
                self.key,
                len(merged),
            )
        except Exception as e:  # noqa: BLE001 - persistence is best-effort
            logger.warning("Failed to flush embedding cache to S3: %s", e)
