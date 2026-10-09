# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""S3-persisted embedding cache (perf: avoid re-embedding across runs/phases).

The in-process cache dies with each Fargate phase, so without persistence the
corpus is re-embedded every run. These tests cover the S3 tier's load/get/put/
flush, model+dimension namespacing, and best-effort degradation on S3 errors —
all with a fake S3 client (no AWS).
"""

from __future__ import annotations

import json

import boto3
import pytest
from moto import mock_aws

from unified_kg_rag.adapters.aws.embedding_cache import S3EmbeddingCache
from unified_kg_rag.adapters.aws.s3_cache import sse_extra_args
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import Config, S3EncryptionType
from unified_kg_rag.domain.models.config import S3EncryptionConfig

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
    # Held and persisted as float32, the precision an OpenSearch knn field keeps.
    assert stored["titan:1024|hash1"] == pytest.approx([0.1, 0.2], rel=1e-6)


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


@pytest.mark.parametrize(
    "encryption_type", [S3EncryptionType.AES256, S3EncryptionType.KMS]
)
def test_flush_applies_configured_sse(encryption_type: S3EncryptionType) -> None:
    # The cache object holds corpus-derived vectors: its upload must honour
    # aws.s3.encryption exactly like the stage-cache sync does.
    with mock_aws():
        session = boto3.Session(region_name="us-east-1")
        s3 = session.client("s3")
        s3.create_bucket(Bucket="bucket")
        key = None
        if encryption_type == S3EncryptionType.KMS:
            key = session.client("kms").create_key()["KeyMetadata"]["KeyId"]
        encryption = S3EncryptionConfig(encryption_type=encryption_type, kms_key_id=key)
        c = S3EmbeddingCache(
            "bucket",
            "embedding-cache/cache.json",
            "titan",
            1024,
            boto_session=session,
            encryption=encryption,
        )
        c.load()
        c.put("h", [0.1])
        c.flush()

        head = s3.head_object(Bucket="bucket", Key="embedding-cache/cache.json")
        expected = sse_extra_args(encryption)
        assert head["ServerSideEncryption"] == expected["ServerSideEncryption"]
        if encryption.kms_key_id:
            assert encryption.kms_key_id in head["SSEKMSKeyId"]


def test_indexer_passes_s3_encryption_to_embedding_cache(config: Config) -> None:
    config.aws.s3.bucket_name = "bucket"
    config.aws.s3.encryption.encryption_type = S3EncryptionType.AES256
    config.indexing.opensearch.persist_embedding_cache = True
    indexer = OpenSearchIndexer.__new__(OpenSearchIndexer)
    indexer.config = config
    indexer.opensearch_config = config.indexing.opensearch
    indexer._embedding_dimension = 1024
    indexer.boto_session = None
    cache = indexer._build_s3_embedding_cache()
    assert cache is not None
    assert cache._sse_args == {"ServerSideEncryption": "AES256"}


class _VersionedS3(_FakeS3):
    """Fake S3 with ETags and head_object; counts full-object reads."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        super().__init__(objects)
        self.get_calls = 0

    def _etag(self, key: str) -> str:
        return f'"{hash(self.objects[key])}"'

    def get_object(self, Bucket: str, Key: str):  # noqa: N803
        self.get_calls += 1
        response = super().get_object(Bucket, Key)
        response["ETag"] = self._etag(Key)
        return response

    def head_object(self, Bucket: str, Key: str):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ETag": self._etag(Key)}

    def put_object(self, Bucket: str, Key: str, Body: bytes):  # noqa: N803
        super().put_object(Bucket, Key, Body)
        return {"ETag": self._etag(Key)}


_KEY = "embedding-cache/cache.json"


def test_vectors_are_held_as_float32() -> None:
    c = _cache(_FakeS3())
    c.load()
    c.put("h", [0.5, 0.25])
    assert c._cache["titan:1024|h"].typecode == "f"
    assert c.get("h") == [0.5, 0.25]


def test_flush_skips_rereading_an_unchanged_object() -> None:
    fake = _VersionedS3({_KEY: json.dumps({"titan:1024|old": [1.0]}).encode()})
    c = _cache(fake)
    c.load()
    for i in range(3):
        c.put(f"h{i}", [float(i)])
        c.flush()

    assert fake.get_calls == 1  # the load only
    stored = json.loads(fake.objects[_KEY])
    assert set(stored) == {f"titan:1024|{k}" for k in ("old", "h0", "h1", "h2")}


def test_flush_merges_a_concurrent_writers_entries() -> None:
    fake = _VersionedS3({_KEY: json.dumps({"other:512|x": [9.0]}).encode()})
    c = _cache(fake)
    c.load()
    # Another process writes after this one loaded.
    fake.objects[_KEY] = json.dumps(
        {"other:512|x": [9.5], "other:512|y": [8.0], "titan:1024|theirs": [7.0]}
    ).encode()
    c.put("mine", [1.0])
    c.flush()

    assert json.loads(fake.objects[_KEY]) == {
        "other:512|x": [9.5],
        "other:512|y": [8.0],
        "titan:1024|theirs": [7.0],
        "titan:1024|mine": [1.0],
    }
    assert c.get("theirs") == [7.0]
