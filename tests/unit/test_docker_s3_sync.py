# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""docker/s3_sync.py, the image's stand-in for ``aws s3 sync``, against moto.

The entrypoint stages the source corpus with it, so it must keep the CLI's
folder semantics, skip what is current, report each failed key and exit
non-zero for it (the task then fails before ingestion starts).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import boto3
import pytest
from moto import mock_aws

pytestmark = pytest.mark.unit

_SCRIPT = Path(__file__).resolve().parents[2] / "docker" / "s3_sync.py"
_BUCKET = "corpus-bucket"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("docker_s3_sync", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


s3_sync = _load()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    for key in ("AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=_BUCKET)
        yield s3


def _put(client: Any, key: str, body: bytes = b"text") -> None:
    client.put_object(Bucket=_BUCKET, Key=key, Body=body)


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("suffix", ["", "/"])
def test_syncs_the_folder_with_or_without_a_trailing_slash(
    client: Any, tmp_path: Path, suffix: str
) -> None:
    _put(client, "corpus/a.txt", b"a")
    _put(client, "corpus/sub/b.pdf", b"b")
    _put(client, "corpus/sub/", b"")  # a console "folder" marker
    _put(client, "corpus-other/c.txt", b"c")  # shares the prefix, not the folder
    _put(client, "elsewhere/d.txt", b"d")

    status = s3_sync.sync(f"s3://{_BUCKET}/corpus{suffix}", tmp_path, client=client)

    assert status == 0
    assert _files(tmp_path) == {"a.txt": b"a", "sub/b.pdf": b"b"}


def test_the_bucket_root_syncs_every_key(client: Any, tmp_path: Path) -> None:
    _put(client, "a.txt", b"a")
    _put(client, "x/한국어.txt", "본문".encode())
    assert s3_sync.sync(f"s3://{_BUCKET}", tmp_path, client=client) == 0
    assert _files(tmp_path) == {"a.txt": b"a", "x/한국어.txt": "본문".encode()}


def test_a_second_run_downloads_only_what_changed(
    client: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _put(client, "corpus/a.txt", b"a")
    _put(client, "corpus/b.txt", b"b")
    assert s3_sync.sync(f"s3://{_BUCKET}/corpus", tmp_path, client=client) == 0
    # The local copy takes the object's modification time.
    head = client.head_object(Bucket=_BUCKET, Key="corpus/a.txt")
    assert (tmp_path / "a.txt").stat().st_mtime == pytest.approx(
        head["LastModified"].timestamp()
    )

    downloaded: list[str] = []
    real = s3_sync._download
    monkeypatch.setattr(
        s3_sync,
        "_download",
        lambda c, b, obj, path: (downloaded.append(obj.key), real(c, b, obj, path)),
    )
    assert s3_sync.sync(f"s3://{_BUCKET}/corpus", tmp_path, client=client) == 0
    assert downloaded == []

    _put(client, "corpus/b.txt", b"bigger")  # size differs
    # Same size, but modified before the object was.
    (tmp_path / "a.txt").write_bytes(b"x")
    os.utime(tmp_path / "a.txt", (0, 0))
    assert s3_sync.sync(f"s3://{_BUCKET}/corpus", tmp_path, client=client) == 0
    assert sorted(downloaded) == ["corpus/a.txt", "corpus/b.txt"]
    assert _files(tmp_path) == {"a.txt": b"a", "b.txt": b"bigger"}


def test_delete_removes_only_files_without_an_object(
    client: Any, tmp_path: Path
) -> None:
    _put(client, "corpus/keep.txt", b"k")
    (tmp_path / "stale").mkdir()
    (tmp_path / "stale" / "gone.txt").write_bytes(b"g")

    assert s3_sync.sync(f"s3://{_BUCKET}/corpus", tmp_path, client=client) == 0
    assert "stale/gone.txt" in _files(tmp_path)  # kept without --delete

    assert s3_sync.main([f"s3://{_BUCKET}/corpus", str(tmp_path), "--delete"]) == 0
    assert _files(tmp_path) == {"keep.txt": b"k"}


def test_a_failed_download_is_reported_and_the_rest_still_sync(
    client: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _put(client, "corpus/ok.txt", b"o")
    _put(client, "corpus/bad.txt", b"b")
    real = s3_sync._download

    def flaky(c: Any, bucket: str, obj: Any, path: Path) -> None:
        if obj.key.endswith("bad.txt"):
            raise OSError("disk full")
        real(c, bucket, obj, path)

    monkeypatch.setattr(s3_sync, "_download", flaky)
    assert s3_sync.sync(f"s3://{_BUCKET}/corpus", tmp_path, client=client) == 1
    assert _files(tmp_path) == {"ok.txt": b"o"}
    out = capsys.readouterr()
    # Only errors are printed (--only-show-errors), on stderr.
    assert out.out == ""
    assert "download failed: s3://corpus-bucket/corpus/bad.txt" in out.err
    assert "disk full" in out.err


def test_a_key_outside_the_directory_is_refused(
    client: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _put(client, "corpus/../escape.txt", b"e")
    _put(client, "corpus/fine.txt", b"f")
    target = tmp_path / "dest"
    assert s3_sync.sync(f"s3://{_BUCKET}/corpus", target, client=client) == 1
    assert _files(tmp_path) == {"dest/fine.txt": b"f"}
    assert "refused" in capsys.readouterr().err


@pytest.mark.parametrize("uri", ["/local/path", "s3://", "https://bucket/x"])
def test_a_bad_uri_exits_2(uri: str, tmp_path: Path) -> None:
    assert s3_sync.sync(uri, tmp_path, client=object()) == 2


def test_a_listing_error_exits_2(
    client: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert s3_sync.sync("s3://no-such-bucket/corpus", tmp_path, client=client) == 2
    assert "cannot list s3://no-such-bucket/corpus/" in capsys.readouterr().err


def test_the_entrypoint_runs_the_script_the_image_installs() -> None:
    root = _SCRIPT.parent
    entrypoint = (root / "entrypoint.sh").read_text(encoding="utf-8")
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    assert "/usr/local/lib/unified-kg-rag/s3_sync.py" in entrypoint
    assert "aws s3" not in entrypoint.replace("# aws s3 sync", "")
    assert (
        "COPY docker/s3_sync.py /usr/local/lib/unified-kg-rag/s3_sync.py" in dockerfile
    )
