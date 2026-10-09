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

Vectors are held as float32 arrays (an OpenSearch knn field stores float32
anyway), about an eighth of a list of Python floats. The S3 object stays a
JSON ``{key: [float, ...]}`` map, so caches written by earlier versions load
unchanged and vice versa.

Thread-safe: indexing tasks running on threads share one cache, so ``put``
and ``flush`` may overlap. A flush uploads a snapshot and clears only the
keys it uploaded.

Best-effort: any S3 error degrades to an in-memory-only cache (load returns
empty, flush is skipped) rather than failing the run.
"""

from __future__ import annotations

import json
import threading
from array import array
from typing import TYPE_CHECKING, Any

import boto3

from unified_kg_rag.adapters.aws.s3_cache import sse_extra_args
from unified_kg_rag.shared import get_logger

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

    from unified_kg_rag.domain.models.config import S3EncryptionConfig

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
        encryption: S3EncryptionConfig | None = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.key = key
        # Same per-object SSE as the stage-cache sync (aws.s3.encryption).
        self._sse_args: dict[str, Any] = (
            sse_extra_args(encryption) if encryption else {}
        )
        # Namespace entries so a model/dimension change never returns a stale
        # vector of the wrong shape/semantics.
        self._namespace = f"{model_id}:{dimension}"
        self._session = boto_session or boto3.Session()
        self._client: S3Client | None = None
        # This namespace's vectors (float32), loaded and computed.
        self._cache: dict[str, array] = {}
        # Other namespaces' entries, kept as read so a flush writes them back.
        self._foreign: dict[str, Any] = {}
        # Keys this process computed since the last flush.
        self._pending: set[str] = set()
        # ETag of the remote object as last read or written. While it is
        # unchanged, this process already holds every entry, so a flush skips
        # re-reading (and re-parsing) the whole object.
        self._etag: str | None = None
        self._loaded = False
        # Guards the maps above against concurrent put/flush; held only for
        # in-memory updates and snapshots, never across an S3 call.
        self._lock = threading.Lock()
        # Serialises flushes, so two never upload interleaved snapshots.
        self._flush_lock = threading.Lock()

    @property
    def client(self) -> S3Client:
        if self._client is None:
            self._client = self._session.client("s3")
        return self._client

    def _namespaced(self, content_hash: str) -> str:
        return f"{self._namespace}|{content_hash}"

    def _merge_remote(self) -> None:
        """Read the persisted map from S3 (all namespaces) and merge it in.

        Other namespaces take the remote entries; in this namespace the
        vectors this process holds win, so a pending one is never replaced.
        A missing or unreadable object merges nothing.
        """
        try:
            obj = self.client.get_object(Bucket=self.bucket_name, Key=self.key)
            data = json.loads(obj["Body"].read())
        except Exception as e:  # noqa: BLE001 - missing/unreadable object -> empty
            logger.info("Embedding cache not read from S3 (starting empty): %s", e)
            return
        if not isinstance(data, dict):
            with self._lock:
                self._etag = obj.get("ETag")
            return
        prefix = f"{self._namespace}|"
        foreign: dict[str, Any] = {}
        vectors: dict[str, array] = {}
        for key, vector in data.items():
            if not key.startswith(prefix):
                foreign[key] = vector
                continue
            try:
                vectors[key] = array("f", vector)
            except (TypeError, ValueError):
                logger.debug("Skipping malformed embedding-cache entry '%s'", key)
        with self._lock:
            self._etag = obj.get("ETag")
            self._foreign.update(foreign)
            for key, vector in vectors.items():
                self._cache.setdefault(key, vector)

    def load(self) -> None:
        """Load the persisted cache from S3 (best-effort, once)."""
        if self._loaded:
            return
        self._loaded = True
        self._merge_remote()
        if self._cache:
            logger.info(
                "Loaded %s embedding-cache entries from 's3://%s/%s'",
                len(self._cache),
                self.bucket_name,
                self.key,
            )

    def get(self, content_hash: str) -> list[float] | None:
        vector = self._cache.get(self._namespaced(content_hash))
        return vector.tolist() if vector is not None else None

    def put(self, content_hash: str, vector: list[float]) -> None:
        key = self._namespaced(content_hash)
        value = array("f", vector)
        with self._lock:
            self._cache[key] = value
            self._pending.add(key)

    def _remote_changed(self) -> bool:
        if self._etag is None:
            return True
        try:
            head = self.client.head_object(Bucket=self.bucket_name, Key=self.key)
        except Exception:  # noqa: BLE001 - unknown state -> re-read
            return True
        return head.get("ETag") != self._etag

    @staticmethod
    def _encode(
        foreign: list[tuple[str, Any]], vectors: list[tuple[str, array]]
    ) -> bytes:
        # One entry at a time, so the vectors never all exist as float lists.
        entries = [f"{json.dumps(k)}:{json.dumps(v)}" for k, v in foreign]
        entries.extend(f"{json.dumps(k)}:{json.dumps(v.tolist())}" for k, v in vectors)
        return ("{" + ",".join(entries) + "}").encode("utf-8")

    def flush(self) -> None:
        """Persist newly-computed vectors back to S3 (best-effort).

        When another writer changed the object since this process last read or
        wrote it, its entries are merged in first, so they are preserved rather
        than clobbered by the whole-object overwrite. Worst case under a true
        write-write race is re-embedding a few vectors, never a wrong one.

        The upload is a snapshot taken under the lock; vectors ``put`` while it
        runs stay pending for the next flush.
        """
        with self._flush_lock:
            with self._lock:
                if not self._pending:
                    return
            try:
                if self._remote_changed():
                    self._merge_remote()
                with self._lock:
                    flushed = set(self._pending)
                    foreign = list(self._foreign.items())
                    vectors = list(self._cache.items())
                response = self.client.put_object(
                    Bucket=self.bucket_name,
                    Key=self.key,
                    Body=self._encode(foreign, vectors),
                    **self._sse_args,
                )
                with self._lock:
                    self._etag = (
                        response.get("ETag") if isinstance(response, dict) else None
                    )
                    self._pending -= flushed
                logger.info(
                    "Flushed %s embedding-cache entries to 's3://%s/%s' (%s total)",
                    len(flushed),
                    self.bucket_name,
                    self.key,
                    len(foreign) + len(vectors),
                )
            except Exception as e:  # noqa: BLE001 - persistence is best-effort
                logger.warning("Failed to flush embedding cache to S3: %s", e)
