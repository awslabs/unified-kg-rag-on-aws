# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
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

from aws_graphrag.shared import get_logger

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
        self._loaded = False
        self._dirty = False

    @property
    def client(self) -> S3Client:
        if self._client is None:
            self._client = self._session.client("s3")
        return self._client

    def _namespaced(self, content_hash: str) -> str:
        return f"{self._namespace}|{content_hash}"

    def load(self) -> None:
        """Load the persisted cache from S3 (best-effort, once)."""
        if self._loaded:
            return
        self._loaded = True
        try:
            obj = self.client.get_object(Bucket=self.bucket_name, Key=self.key)
            data = json.loads(obj["Body"].read())
            if isinstance(data, dict):
                # Only keep entries for the current model/dimension namespace.
                prefix = f"{self._namespace}|"
                self._cache = {
                    k: v for k, v in data.items() if k.startswith(prefix)
                }
                logger.info(
                    "Loaded %s embedding-cache entries from 's3://%s/%s'",
                    len(self._cache),
                    self.bucket_name,
                    self.key,
                )
        except Exception as e:  # noqa: BLE001 - degrade to empty in-memory cache
            logger.info(
                "Embedding cache not loaded from S3 (starting empty): %s", e
            )
            self._cache = {}

    def get(self, content_hash: str) -> list[float] | None:
        return self._cache.get(self._namespaced(content_hash))

    def put(self, content_hash: str, vector: list[float]) -> None:
        self._cache[self._namespaced(content_hash)] = vector
        self._dirty = True

    def flush(self) -> None:
        """Persist the cache back to S3 if it changed (best-effort)."""
        if not self._dirty:
            return
        try:
            body = json.dumps(self._cache).encode("utf-8")
            self.client.put_object(Bucket=self.bucket_name, Key=self.key, Body=body)
            self._dirty = False
            logger.info(
                "Flushed %s embedding-cache entries to 's3://%s/%s'",
                len(self._cache),
                self.bucket_name,
                self.key,
            )
        except Exception as e:  # noqa: BLE001 - persistence is best-effort
            logger.warning("Failed to flush embedding cache to S3: %s", e)
