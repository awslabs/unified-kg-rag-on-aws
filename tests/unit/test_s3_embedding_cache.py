# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""S3-persisted embedding cache (perf: avoid re-embedding across runs/phases).

The in-process cache dies with each Fargate phase, so without persistence the
corpus is re-embedded every run. These tests cover the S3 tier's load/get/put/
flush, model+dimension namespacing, and best-effort degradation on S3 errors —
all with a fake S3 client (no AWS).
"""

from __future__ import annotations

import json

import pytest

from aws_graphrag.adapters.aws.embedding_cache import S3EmbeddingCache

pytestmark = pytest.mark.unit


class _FakeS3:
    """Minimal in-memory stand-in for the boto3 S3 client."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = objects or {}
        self.put_calls = 0

    def get_object(self, Bucket: str, Key: str):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(f"no such key: {Key}")  # stands in for ClientError
        return {"Body": _Body(self.objects[Key])}

    def put_object(self, Bucket: str, Key: str, Body: bytes):  # noqa: N803
        self.objects[Key] = Body
        self.put_calls += 1


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


def _cache(fake: _FakeS3, model="titan", dim=1024) -> S3EmbeddingCache:
    c = S3EmbeddingCache("bucket", "embedding-cache/cache.json", model, dim)
    c._client = fake  # inject fake, bypass boto session
    return c


def test_load_empty_when_absent() -> None:
    c = _cache(_FakeS3())
    c.load()
    assert c.get("abc") is None


def test_put_then_flush_persists_namespaced() -> None:
    fake = _FakeS3()
    c = _cache(fake, model="titan", dim=1024)
    c.load()
    c.put("hash1", [0.1, 0.2])
    c.flush()
    assert fake.put_calls == 1
    stored = json.loads(fake.objects["embedding-cache/cache.json"])
    # Key is namespaced by model:dim so a model/dim change can't return stale.
    assert "titan:1024|hash1" in stored
    assert stored["titan:1024|hash1"] == [0.1, 0.2]


def test_load_reads_back_persisted_entry() -> None:
    fake = _FakeS3(
        {"embedding-cache/cache.json": json.dumps({"titan:1024|h": [1.0]}).encode()}
    )
    c = _cache(fake, model="titan", dim=1024)
    c.load()
    assert c.get("h") == [1.0]


def test_namespace_isolates_model_and_dim() -> None:
    # An entry written under titan:1024 must NOT be visible to titan:512.
    fake = _FakeS3(
        {"embedding-cache/cache.json": json.dumps({"titan:1024|h": [1.0]}).encode()}
    )
    c = _cache(fake, model="titan", dim=512)
    c.load()
    assert c.get("h") is None


def test_flush_noop_when_not_dirty() -> None:
    fake = _FakeS3()
    c = _cache(fake)
    c.load()
    c.flush()  # nothing put -> no write
    assert fake.put_calls == 0


def test_flush_degrades_on_s3_error() -> None:
    class _Boom(_FakeS3):
        def put_object(self, **_):  # noqa: ANN003
            raise RuntimeError("s3 down")

    c = _cache(_Boom())
    c.load()
    c.put("h", [0.1])
    c.flush()  # must not raise — persistence is best-effort


def test_load_is_idempotent() -> None:
    fake = _FakeS3(
        {"embedding-cache/cache.json": json.dumps({"titan:1024|h": [1.0]}).encode()}
    )
    c = _cache(fake)
    c.load()
    c.load()  # second load is a no-op, doesn't reset
    assert c.get("h") == [1.0]
