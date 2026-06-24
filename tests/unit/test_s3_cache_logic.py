# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for S3CacheManager sync logic (AWS-free, via moto).

``moto.mock_aws`` provides an in-memory S3 surface so the real boto3 calls
(``upload_file``/``download_file``/``list_objects_v2`` pagination) run without
touching AWS. Covers prefix/key construction, the per-file rglob iteration,
stage-name grouping, round-trip up/down, encryption ExtraArgs wiring, and the
missing-directory / empty-pipeline guards.
"""

from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from aws_graphrag.adapters.aws.s3_cache import S3CacheManager
from aws_graphrag.domain.models import Config, S3EncryptionType

pytestmark = pytest.mark.unit

_BUCKET = "test-cache-bucket"
_REGION = "us-east-1"


@pytest.fixture
def s3_setup():
    """Create a mocked S3 bucket and yield a (Config, boto_session) pair."""
    with mock_aws():
        session = boto3.Session(region_name=_REGION)
        session.client("s3").create_bucket(Bucket=_BUCKET)
        config = Config()
        config.aws.region_name = _REGION
        yield config, session


def _manager(config, session, prefix="cache") -> S3CacheManager:
    return S3CacheManager(
        config, boto_session=session, bucket_name=_BUCKET, prefix=prefix
    )


# --- construction / validation -------------------------------------------


def test_requires_bucket_name() -> None:
    config = Config()
    # config default has no s3 bucket_name -> ValueError.
    with pytest.raises(ValueError, match="bucket_name"):
        S3CacheManager(config, bucket_name=None)


def test_prefix_is_stripped_of_slashes(s3_setup) -> None:
    config, session = s3_setup
    mgr = _manager(config, session, prefix="/foo/bar/")
    assert mgr.prefix == "foo/bar"


def test_get_base_prefix_with_and_without_prefix(s3_setup) -> None:
    config, session = s3_setup
    mgr = _manager(config, session, prefix="cache")
    assert mgr._get_base_prefix("pid-1") == "cache/pid-1"
    # An empty prefix falls back to DEFAULT_S3_PREFIX in the constructor; force
    # the falsy-prefix branch of _get_base_prefix directly.
    mgr.prefix = ""
    assert mgr._get_base_prefix("pid-1") == "pid-1"


# --- sync_pipeline_to_s3 --------------------------------------------------


def test_sync_to_s3_missing_local_dir_returns_empty(s3_setup, tmp_path) -> None:
    config, session = s3_setup
    mgr = _manager(config, session)
    assert mgr.sync_pipeline_to_s3("pid", tmp_path / "nope") == {}


def test_sync_to_s3_no_json_files(s3_setup, tmp_path) -> None:
    config, session = s3_setup
    mgr = _manager(config, session)
    (tmp_path / "notjson.txt").write_text("x", encoding="utf-8")
    # rglob('*.json') matches nothing -> empty results.
    assert mgr.sync_pipeline_to_s3("pid", tmp_path) == {}


def test_sync_to_s3_uploads_and_groups_by_stage(s3_setup, tmp_path) -> None:
    config, session = s3_setup
    mgr = _manager(config, session, prefix="cache")

    # Two stage subdirs + one top-level file.
    (tmp_path / "stageA").mkdir()
    (tmp_path / "stageB").mkdir()
    (tmp_path / "stageA" / "a1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stageA" / "a2.json").write_text("{}", encoding="utf-8")
    (tmp_path / "stageB" / "b1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "top.json").write_text("{}", encoding="utf-8")

    results = mgr.sync_pipeline_to_s3("pid-1", tmp_path)
    # Nested files group under their first path part; top-level under its stem.
    assert results == {"stageA": True, "stageB": True, "top": True}

    # Verify objects actually landed at the expected keys.
    s3 = session.client("s3")
    keys = {
        o["Key"]
        for o in s3.list_objects_v2(Bucket=_BUCKET, Prefix="cache/pid-1/").get(
            "Contents", []
        )
    }
    assert "cache/pid-1/stageA/a1.json" in keys
    assert "cache/pid-1/stageB/b1.json" in keys
    assert "cache/pid-1/top.json" in keys


# --- sync_pipeline_from_s3 ------------------------------------------------


def test_sync_from_s3_no_files(s3_setup, tmp_path) -> None:
    config, session = s3_setup
    mgr = _manager(config, session)
    local = tmp_path / "dl"
    results = mgr.sync_pipeline_from_s3("absent-pid", local)
    assert results == {}
    assert local.is_dir()  # created even when nothing to download


def test_sync_round_trip(s3_setup, tmp_path) -> None:
    config, session = s3_setup
    mgr = _manager(config, session, prefix="cache")

    src = tmp_path / "src"
    (src / "stageA").mkdir(parents=True)
    (src / "stageA" / "a1.json").write_text('{"v": 1}', encoding="utf-8")
    (src / "top.json").write_text('{"v": 2}', encoding="utf-8")
    mgr.sync_pipeline_to_s3("pid-rt", src)

    dest = tmp_path / "dest"
    results = mgr.sync_pipeline_from_s3("pid-rt", dest)
    assert results.get("stageA") is True
    assert results.get("top") is True
    # Files materialized at mirror-relative paths.
    assert (dest / "stageA" / "a1.json").read_text(encoding="utf-8") == '{"v": 1}'
    assert (dest / "top.json").read_text(encoding="utf-8") == '{"v": 2}'


# --- _upload_file_to_s3 encryption wiring ---------------------------------


def test_upload_applies_aes256_extra_args(s3_setup, tmp_path, mocker) -> None:
    config, session = s3_setup
    config.aws.s3.encryption.encryption_type = S3EncryptionType.AES256
    mgr = _manager(config, session)

    captured: dict = {}

    def _fake_upload(local, bucket, key, ExtraArgs):  # noqa: N803
        captured["ExtraArgs"] = ExtraArgs

    mocker.patch.object(mgr, "_s3_client", mocker.MagicMock(upload_file=_fake_upload))
    f = tmp_path / "x.json"
    f.write_text("{}", encoding="utf-8")
    assert mgr._upload_file_to_s3(f, "k/x.json") is True
    assert captured["ExtraArgs"]["ServerSideEncryption"] == "AES256"


def test_upload_applies_kms_extra_args(s3_setup, tmp_path, mocker) -> None:
    config, session = s3_setup
    config.aws.s3.encryption.encryption_type = S3EncryptionType.KMS
    config.aws.s3.encryption.kms_key_id = "key-123"
    mgr = _manager(config, session)

    captured: dict = {}

    def _fake_upload(local, bucket, key, ExtraArgs):  # noqa: N803
        captured["ExtraArgs"] = ExtraArgs

    mocker.patch.object(mgr, "_s3_client", mocker.MagicMock(upload_file=_fake_upload))
    f = tmp_path / "x.json"
    f.write_text("{}", encoding="utf-8")
    assert mgr._upload_file_to_s3(f, "k/x.json") is True
    assert captured["ExtraArgs"]["ServerSideEncryption"] == "aws:kms"
    assert captured["ExtraArgs"]["SSEKMSKeyId"] == "key-123"


def test_upload_missing_file_returns_false(s3_setup, tmp_path) -> None:
    config, session = s3_setup
    mgr = _manager(config, session)
    missing = tmp_path / "ghost.json"
    # FileNotFoundError is caught -> False, not raised.
    assert mgr._upload_file_to_s3(missing, "k/ghost.json") is False
