# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``run-eval`` with ``s3://`` dataset and outputs paths (moto-mocked S3).

Covers the S3 transfer helpers (URI validation, download, directory upload
with SSE) and the ``EvaluationRunner`` flow: an ``s3://`` dataset is
downloaded, an ``s3://`` outputs directory is uploaded, and a failed upload
exits non-zero while keeping the local results.
"""

from __future__ import annotations

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from unified_kg_rag.adapters.aws import s3_transfer
from unified_kg_rag.adapters.aws.s3_transfer import (
    download_s3_object,
    is_s3_uri,
    parse_s3_uri,
    upload_directory_to_s3,
)
from unified_kg_rag.application.cli import run_evaluation
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import S3EncryptionType
from unified_kg_rag.shared import AWSServiceError, ConfigurationError

pytestmark = pytest.mark.unit

REGION = "us-east-1"
BUCKET = "eval-bucket"
DATASET = [{"question": "Who ships the goods?", "answer": "Vendor"}]


@pytest.fixture
def s3(monkeypatch):
    monkeypatch.setenv("AWS_DEFAULT_REGION", REGION)
    with mock_aws():
        client = boto3.client("s3", region_name=REGION)
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture
def eval_config(mocker) -> Config:
    config = Config()
    config.aws.region_name = REGION
    config.evaluation.enabled_evaluators = []
    mocker.patch.object(run_evaluation, "get_config", return_value=config)
    mocker.patch.object(run_evaluation, "display_ascii_art")
    return config


class _Chain:
    async def ainvoke(self, inputs, config=None):
        meta = {"search_strategy": "local", "processing_time": 0.1}
        return {"answer": "Vendor", "sources": [], "metadata": meta}

    async def abatch(self, inputs, config=None):
        return [await self.ainvoke(i) for i in inputs]


def _args(*argv: str):
    parser = run_evaluation.CommandLineInterface._setup_arguments()
    return parser.parse_args(list(argv))


def _keys(s3, prefix: str = "") -> list[str]:
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return sorted(obj["Key"] for obj in resp.get("Contents", []))


# --- URI parsing ----------------------------------------------------------


def test_is_s3_uri() -> None:
    assert is_s3_uri("s3://b/k")
    assert not is_s3_uri("data/eval.json")
    assert not is_s3_uri(Path("s3://b/k"))
    assert not is_s3_uri(None)


def test_parse_s3_uri_object_and_prefix() -> None:
    assert parse_s3_uri("s3://my-bucket/eval/data.json", require_key=True) == (
        "my-bucket",
        "eval/data.json",
    )
    assert parse_s3_uri("s3://my-bucket/eval/out/", require_key=False) == (
        "my-bucket",
        "eval/out",
    )
    assert parse_s3_uri("s3://my-bucket", require_key=False) == ("my-bucket", "")


@pytest.mark.parametrize(
    ("uri", "require_key"),
    [
        ("s3://", True),
        ("s3:///key.json", True),
        ("s3://Bad_Bucket/key.json", True),
        ("s3://my-bucket", True),
        ("s3://my-bucket/", True),
        ("s3://my-bucket/eval/", True),
        ("s3://ab/out/", False),
        ("https://my-bucket/key.json", True),
    ],
)
def test_parse_s3_uri_rejects_malformed(uri: str, require_key: bool) -> None:
    with pytest.raises(ConfigurationError):
        parse_s3_uri(uri, require_key=require_key)


# --- transfer helpers -----------------------------------------------------


def test_download_s3_object(s3, tmp_path: Path) -> None:
    s3.put_object(Bucket=BUCKET, Key="eval/data.json", Body=b"[]")
    local = download_s3_object(
        s3, f"s3://{BUCKET}/eval/data.json", tmp_path / "x" / "d.json"
    )
    assert local.read_bytes() == b"[]"


def test_download_missing_object_raises(s3, tmp_path: Path) -> None:
    with pytest.raises(AWSServiceError, match="eval/missing.json"):
        download_s3_object(s3, f"s3://{BUCKET}/eval/missing.json", tmp_path / "d")


def test_upload_directory_applies_sse(s3, tmp_path: Path) -> None:
    config = Config()
    config.aws.s3.encryption.encryption_type = S3EncryptionType.AES256
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "sub" / "b.json").write_text("{}")
    uploaded = upload_directory_to_s3(s3, tmp_path, f"s3://{BUCKET}/out/", config)
    assert uploaded == [f"s3://{BUCKET}/out/a.json", f"s3://{BUCKET}/out/sub/b.json"]
    head = s3.head_object(Bucket=BUCKET, Key="out/a.json")
    assert head["ServerSideEncryption"] == "AES256"


def test_upload_to_missing_bucket_raises(s3, tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{}")
    with pytest.raises(AWSServiceError, match="1/1"):
        upload_directory_to_s3(s3, tmp_path, "s3://no-such-bucket/out", Config())


# --- EvaluationRunner -----------------------------------------------------


async def test_runner_downloads_dataset_and_uploads_outputs(s3, eval_config) -> None:
    uri = f"s3://{BUCKET}/eval/data.json"
    s3.put_object(Bucket=BUCKET, Key="eval/data.json", Body=json.dumps(DATASET))
    args = _args("--eval-data-path", uri, "--outputs-directory", f"s3://{BUCKET}/out/")
    assert args.eval_data_path == uri  # not collapsed to a Path

    assert await run_evaluation.EvaluationRunner(args, _Chain()).run() == 0

    keys = _keys(s3, "out/")
    summary_key = next(k for k in keys if k.startswith("out/evaluation_summary_"))
    body = s3.get_object(Bucket=BUCKET, Key=summary_key)["Body"].read()
    manifest = json.loads(body)["run_manifest"]
    assert manifest["dataset"]["path"] == uri
    assert manifest["cli_args"]["eval_data_path"] == uri
    assert manifest["dataset"]["num_queries"] == 1


async def test_runner_s3_dataset_with_local_outputs(
    s3, eval_config, tmp_path: Path
) -> None:
    s3.put_object(Bucket=BUCKET, Key="data.json", Body=json.dumps(DATASET))
    out = tmp_path / "out"
    args = _args(
        "--eval-data-path",
        f"s3://{BUCKET}/data.json",
        "--outputs-directory",
        str(out),
    )
    assert await run_evaluation.EvaluationRunner(args, _Chain()).run() == 0
    assert list(out.glob("evaluation_summary_*.json"))
    assert _keys(s3) == ["data.json"]  # nothing uploaded


async def test_runner_upload_failure_exits_non_zero_and_keeps_results(
    s3, eval_config, mocker, tmp_path: Path
) -> None:
    data = tmp_path / "eval.json"
    data.write_text(json.dumps(DATASET))
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    mocker.patch.object(run_evaluation.tempfile, "mkdtemp", return_value=str(work_dir))
    args = _args(
        "--eval-data-path", str(data), "--outputs-directory", "s3://no-such-bucket/out"
    )
    assert await run_evaluation.EvaluationRunner(args, _Chain()).run() == 1
    assert list((work_dir / "outputs").glob("evaluation_summary_*.json"))


async def test_runner_cleans_up_work_dir_on_success(
    s3, eval_config, mocker, tmp_path: Path
) -> None:
    s3.put_object(Bucket=BUCKET, Key="data.json", Body=json.dumps(DATASET))
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    mocker.patch.object(run_evaluation.tempfile, "mkdtemp", return_value=str(work_dir))
    args = _args(
        "--eval-data-path",
        f"s3://{BUCKET}/data.json",
        "--outputs-directory",
        f"s3://{BUCKET}/out",
    )
    assert await run_evaluation.EvaluationRunner(args, _Chain()).run() == 0
    assert not work_dir.exists()


async def test_runner_missing_s3_dataset_raises(s3, eval_config) -> None:
    args = _args("--eval-data-path", f"s3://{BUCKET}/missing.json")
    with pytest.raises(AWSServiceError, match="missing.json"):
        await run_evaluation.EvaluationRunner(args, _Chain()).run()


@pytest.mark.parametrize(
    "argv",
    [
        ["--eval-data-path", "s3://my-bucket/"],
        ["--eval-data-path", "s3://Bad_Bucket/d.json"],
        ["--eval-data-path", "s3://my-bucket/d.json", "--outputs-directory", "s3://"],
    ],
)
def test_runner_rejects_malformed_uris_before_running(eval_config, argv) -> None:
    with pytest.raises(SystemExit) as exc:
        run_evaluation.EvaluationRunner(_args(*argv), _Chain())
    assert exc.value.code == 1


async def test_runner_local_paths_never_create_an_s3_client(
    eval_config, mocker, tmp_path: Path
) -> None:
    data = tmp_path / "eval.json"
    data.write_text(json.dumps(DATASET))
    client = mocker.patch.object(run_evaluation, "s3_client_for")
    args = _args("--eval-data-path", str(data), "--outputs-directory", str(tmp_path))
    assert args.eval_data_path == data  # local paths still parse to a Path
    assert await run_evaluation.EvaluationRunner(args, _Chain()).run() == 0
    client.assert_not_called()
    assert list(tmp_path.glob("evaluation_summary_*.json"))


def test_main_reports_s3_errors_and_exits_1(eval_config, mocker, monkeypatch) -> None:
    import sys

    monkeypatch.setattr(
        sys, "argv", ["run-eval", "--eval-data-path", "s3://my-bucket/d.json"]
    )
    mocker.patch.object(run_evaluation, "missing_endpoints_error", return_value=None)
    mocker.patch.object(run_evaluation, "GraphRAGChain")

    def _fail(coro, **_kwargs):
        coro.close()  # never started; close it so it is not left un-awaited
        raise AWSServiceError("Failed to download 's3://my-bucket/d.json'")

    mocker.patch.object(run_evaluation.event_loop, "run", side_effect=_fail)
    with pytest.raises(SystemExit) as exc:
        run_evaluation.main()
    assert exc.value.code == 1


def test_s3_client_uses_configured_region() -> None:
    config = Config()
    config.aws.region_name = "eu-west-1"
    assert s3_transfer.s3_client_for(config).meta.region_name == "eu-west-1"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "is not valid JSON"),
        ("[]", "contains no queries"),
        ('[{"id": "a", "question": "x", "answer_aliases": 5}]', "answer_aliases"),
    ],
)
def test_main_reports_an_invalid_dataset_and_exits_2(
    eval_config, mocker, monkeypatch, capsys, tmp_path, content, message
) -> None:
    # An input error, like a usage error: a clear message, no traceback.
    import sys

    dataset = tmp_path / "eval.json"
    dataset.write_text(content, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["run-eval", "--eval-data-path", str(dataset)])
    mocker.patch.object(run_evaluation, "missing_endpoints_error", return_value=None)
    chain = mocker.MagicMock()
    chain.aclose = mocker.AsyncMock()
    mocker.patch.object(run_evaluation, "GraphRAGChain", return_value=chain)
    with pytest.raises(SystemExit) as exc:
        run_evaluation.main()
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "invalid evaluation dataset" in out
    assert message in out
    assert "unexpected error" not in out
    assert "Traceback" not in out
