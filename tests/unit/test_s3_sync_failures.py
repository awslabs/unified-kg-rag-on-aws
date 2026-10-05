# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""S3 cache sync failures must fail the run, not degrade silently (AWS-free).

In a phased Step Functions run each phase hands its stage checkpoints to the
next one through S3. A swallowed upload/download error used to exit 0 and make
the NEXT phase fail with a misleading "missing stage" error.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import boto3
import pytest
from boto3.exceptions import S3UploadFailedError
from moto import mock_aws

from unified_kg_rag.adapters.aws.s3_cache import S3CacheManager
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.pipeline import PipelineContext, PipelineStageStatus
from unified_kg_rag.shared import AWSServiceError, CacheSyncError
from unified_kg_rag.shared.exceptions import PipelineExecutionError

pytestmark = pytest.mark.unit

_BUCKET = "test-cache-bucket"
_REGION = "us-east-1"


@pytest.fixture
def s3_setup():
    with mock_aws():
        session = boto3.Session(region_name=_REGION)
        session.client("s3").create_bucket(Bucket=_BUCKET)
        config = Config()
        config.aws.region_name = _REGION
        yield config, session


def _manager(config, session, bucket: str = _BUCKET) -> S3CacheManager:
    return S3CacheManager(config, boto_session=session, bucket_name=bucket)


def _write_cache(root: Path, names: list[str]) -> None:
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")


def test_cache_sync_error_is_an_aws_service_error() -> None:
    assert issubclass(CacheSyncError, AWSServiceError)


# --- upload ------------------------------------------------------------------


def test_partial_upload_raises_after_attempting_every_file(
    s3_setup, tmp_path, mocker
) -> None:
    config, session = s3_setup
    mgr = _manager(config, session)
    _write_cache(tmp_path, ["stageA/a.json", "stageA/b.json", "stageB/c.json"])
    real_upload = mgr._upload_file_to_s3
    attempted: list[str] = []

    def flaky(local_path: Path, s3_key: str) -> bool:
        attempted.append(s3_key)
        return False if s3_key.endswith("b.json") else real_upload(local_path, s3_key)

    mocker.patch.object(mgr, "_upload_file_to_s3", side_effect=flaky)
    with pytest.raises(CacheSyncError, match=r"upload .* 1/3 files.*stageA/b\.json"):
        mgr.sync_pipeline_to_s3("pid", tmp_path)
    assert len(attempted) == 3  # the failure did not short-circuit the rest


def test_upload_to_missing_bucket_raises(s3_setup, tmp_path) -> None:
    config, session = s3_setup
    _write_cache(tmp_path, ["stageA/a.json"])
    mgr = _manager(config, session, bucket="absent-bucket")
    # head_bucket fails inside the per-file upload, so it surfaces as a
    # per-file failure; either way the sync must raise.
    with pytest.raises(CacheSyncError, match=r"1/1 files in bucket 'absent-bucket'"):
        mgr.sync_pipeline_to_s3("pid", tmp_path)


def test_upload_failed_error_counts_as_file_failure(s3_setup, tmp_path, mocker) -> None:
    # boto3's managed upload_file raises S3UploadFailedError, not ClientError.
    config, session = s3_setup
    mgr = _manager(config, session)
    client = mocker.MagicMock()
    client.upload_file.side_effect = S3UploadFailedError("AccessDenied")
    mocker.patch.object(mgr, "_s3_client", client)
    f = tmp_path / "x.json"
    f.write_text("{}", encoding="utf-8")
    assert mgr._upload_file_to_s3(f, "k/x.json") is False


def test_failure_message_truncates_long_key_lists(s3_setup, tmp_path, mocker) -> None:
    config, session = s3_setup
    mgr = _manager(config, session)
    _write_cache(tmp_path, [f"stage/{i}.json" for i in range(8)])
    mocker.patch.object(mgr, "_upload_file_to_s3", return_value=False)
    with pytest.raises(CacheSyncError, match=r"8/8 files.*\(\+3 more\)"):
        mgr.sync_pipeline_to_s3("pid", tmp_path)


# --- download ----------------------------------------------------------------


def test_partial_download_raises(s3_setup, tmp_path, mocker) -> None:
    config, session = s3_setup
    mgr = _manager(config, session)
    src = tmp_path / "src"
    _write_cache(src, ["stageA/a.json", "stageB/b.json"])
    mgr.sync_pipeline_to_s3("pid", src)

    real_download = mgr._download_cache_file
    mocker.patch.object(
        mgr,
        "_download_cache_file",
        side_effect=lambda key, path: (
            False if key.endswith("b.json") else real_download(key, path)
        ),
    )
    with pytest.raises(CacheSyncError, match=r"download .* 1/2 files.*stageB/b\.json"):
        mgr.sync_pipeline_from_s3("pid", tmp_path / "dest")


def test_download_listing_failure_raises(s3_setup, tmp_path) -> None:
    config, session = s3_setup
    mgr = _manager(config, session, bucket="absent-bucket")
    with pytest.raises(CacheSyncError, match="from 's3://absent-bucket/"):
        mgr.sync_pipeline_from_s3("pid", tmp_path / "dest")


def test_download_with_no_remote_cache_is_not_an_error(s3_setup, tmp_path) -> None:
    # First phase of a fresh run: nothing in S3 yet is the expected state.
    config, session = s3_setup
    assert _manager(config, session).sync_pipeline_from_s3("fresh", tmp_path) == {}


@pytest.mark.parametrize("marker", ["cache/pid/", "cache/pid/stageA/"])
def test_download_skips_folder_marker_keys(s3_setup, tmp_path, marker) -> None:
    # Console-created "folders" are zero-byte keys ending in '/'; they are not
    # cache files and must not fail (or be counted by) the sync.
    config, session = s3_setup
    mgr = _manager(config, session)
    src = tmp_path / "src"
    _write_cache(src, ["stageA/a.json"])
    mgr.sync_pipeline_to_s3("pid", src)
    session.client("s3").put_object(Bucket=_BUCKET, Key=marker, Body=b"")

    dest = tmp_path / "dest"
    results = mgr.sync_pipeline_from_s3("pid", dest)
    assert results == {"stageA": True}
    assert (dest / "stageA" / "a.json").read_text(encoding="utf-8") == "{}"


def test_download_with_only_folder_marker_is_empty(s3_setup, tmp_path) -> None:
    config, session = s3_setup
    session.client("s3").put_object(Bucket=_BUCKET, Key="cache/pid/", Body=b"")
    assert _manager(config, session).sync_pipeline_from_s3("pid", tmp_path) == {}


# --- pipeline wiring ---------------------------------------------------------


def _pipeline_with_failing_sync(tmp_path: Path, mocker) -> DataIngestionPipeline:
    pipe = object.__new__(DataIngestionPipeline)
    pipe.pipeline_config = SimpleNamespace(
        s3_sync_enabled=True, pipeline_id=None, resume_from_stage=None
    )
    pipe.cache_manager = mocker.MagicMock()
    pipe.cache_manager.get_pipeline_cache_dir.return_value = tmp_path
    pipe.s3_cache_manager = mocker.MagicMock()
    error = CacheSyncError("S3 cache upload failed for 1/1 files")
    pipe.s3_cache_manager.sync_pipeline_to_s3.side_effect = error
    pipe.s3_cache_manager.sync_pipeline_from_s3.side_effect = error
    return pipe


@pytest.mark.parametrize("direction", ["upload", "download"])
def test_pipeline_sync_propagates_without_success_log(
    tmp_path, mocker, caplog, direction
) -> None:
    pipe = _pipeline_with_failing_sync(tmp_path, mocker)
    with caplog.at_level(logging.INFO), pytest.raises(CacheSyncError):
        pipe._sync_cache_with_s3("pid", direction)
    assert "completed successfully" not in caplog.text


def test_pipeline_run_fails_when_download_sync_fails(tmp_path, mocker) -> None:
    pipe = _pipeline_with_failing_sync(tmp_path, mocker)
    with pytest.raises(PipelineExecutionError) as exc:
        pipe.run(tmp_path, pipeline_id="pid")
    assert isinstance(exc.value.__cause__, CacheSyncError)


def test_cli_exits_non_zero_on_sync_failure(mocker) -> None:
    from unified_kg_rag.application.cli import run_ingestion_pipeline as cli

    runner = mocker.MagicMock()
    runner.run.side_effect = PipelineExecutionError("Failed to run pipeline: sync")
    mocker.patch.object(cli, "CommandLineInterface")
    mocker.patch.object(cli, "IngestionPipelineRunner", return_value=runner)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_upload_failure_after_stages_logs_summary_then_raises(
    tmp_path, mocker, caplog
) -> None:
    pipe = _pipeline_with_failing_sync(tmp_path, mocker)
    pipe.state_manager = mocker.MagicMock()
    mocker.patch.object(pipe, "_create_pipeline_metrics")
    mocker.patch.object(pipe, "_emit_metrics")
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=tmp_path,
    )

    with caplog.at_level(logging.INFO), pytest.raises(CacheSyncError):
        pipe._finalize_pipeline_execution(context, time.time())

    text = caplog.text
    assert "S3 cache upload failed after the pipeline stages finished" in text
    assert "PIPELINE SUMMARY" in text
    assert "Pipeline ID: pid" in text
    # Pipeline metadata was persisted locally before the upload was attempted.
    pipe.state_manager.save_pipeline_metadata.assert_called_once_with(context)
