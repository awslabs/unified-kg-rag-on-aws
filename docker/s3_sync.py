# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download an S3 folder to a local directory, like ``aws s3 sync``.

The data-plane entrypoint stages the source corpus with it, so the image
needs no AWS CLI (a second Python and its OS packages) for one command.
Usage::

    python s3_sync.py s3://bucket/prefix /local/dir [--delete]

What it keeps from ``aws s3 sync <s3 uri> <dir> --only-show-errors``:

- The URI names a folder: ``s3://bucket/corpus`` and ``s3://bucket/corpus/``
  both sync the keys under ``corpus/`` (``s3://bucket`` the whole bucket);
  folder marker keys (ending in ``/``) are skipped.
- A file is downloaded when it is missing locally, its size differs or the
  object was modified after the local file; the local file then takes the
  object's modification time, so a second run downloads nothing new.
- ``--delete`` removes local files that have no object under the prefix.
- Only errors are printed (to stderr), one line per failed key.
- Exit status: 0 when every object was synced; 1 when an object could not be
  downloaded (the others still are); 2 for a bad argument (not an
  ``s3://`` URI) or when the listing itself fails.

Unlike the CLI, a key that would land outside the directory (``..``
segments, an absolute path) is refused as an error rather than written.
Credentials and the region come from the usual boto3 chain (the task role
and ``AWS_REGION`` in ECS).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import boto3
from boto3.s3.transfer import TransferConfig

# The AWS CLI's default s3 max_concurrent_requests.
_WORKERS = 10


@dataclass(frozen=True)
class _Object:
    key: str
    relative: str
    size: int
    modified: datetime


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """``(bucket, prefix)`` of an ``s3://bucket[/prefix]`` URI; the prefix
    is empty or ends in ``/``."""
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    bucket, _, prefix = uri[len("s3://") :].partition("/")
    if not bucket:
        raise ValueError(f"no bucket in {uri!r}")
    prefix = prefix.strip("/")
    return bucket, f"{prefix}/" if prefix else ""


def _list(client: Any, bucket: str, prefix: str) -> Iterator[_Object]:
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith("/"):
                continue
            yield _Object(
                key=key,
                relative=key[len(prefix) :],
                size=int(item["Size"]),
                modified=item["LastModified"],
            )


def _target(destination: Path, relative: str) -> Path | None:
    """Where ``relative`` lands under ``destination``, or None when it would
    land outside it."""
    parts = PurePosixPath(relative).parts
    if not parts or relative.startswith("/") or ".." in parts:
        return None
    return destination.joinpath(*parts)


def _up_to_date(path: Path, obj: _Object) -> bool:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    return stat.st_size == obj.size and stat.st_mtime >= obj.modified.timestamp()


def _download(client: Any, bucket: str, obj: _Object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(
        bucket, obj.key, str(path), Config=TransferConfig(use_threads=False)
    )
    stamp = obj.modified.timestamp()
    os.utime(path, (stamp, stamp))


def _error(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def sync(
    uri: str, destination: Path, *, delete: bool = False, client: Any = None
) -> int:
    """Sync ``uri`` into ``destination``; the process exit status."""
    try:
        bucket, prefix = parse_s3_uri(uri)
    except ValueError as e:
        _error(f"s3_sync: {e}")
        return 2
    client = client or boto3.client("s3")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        objects = list(_list(client, bucket, prefix))
    except Exception as e:  # noqa: BLE001 - reported as the exit status
        _error(f"s3_sync: cannot list s3://{bucket}/{prefix}: {e}")
        return 2
    failed = 0
    wanted: set[Path] = set()
    jobs: list[tuple[_Object, Path]] = []
    for obj in objects:
        path = _target(destination, obj.relative)
        if path is None:
            _error(f"s3_sync: refused s3://{bucket}/{obj.key}: outside {destination}")
            failed += 1
            continue
        wanted.add(path)
        if not _up_to_date(path, obj):
            jobs.append((obj, path))

    def run(job: tuple[_Object, Path]) -> bool:
        obj, path = job
        try:
            _download(client, bucket, obj, path)
        except Exception as e:  # noqa: BLE001 - one failed key, the rest go on
            _error(f"download failed: s3://{bucket}/{obj.key} to {path}: {e}")
            return False
        return True

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        failed += sum(not ok for ok in pool.map(run, jobs))
    if delete:
        for path in sorted(destination.rglob("*"), reverse=True):
            if path.is_file() and path not in wanted:
                try:
                    path.unlink()
                except OSError as e:
                    _error(f"delete failed: {path}: {e}")
                    failed += 1
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download an S3 folder to a local directory (aws s3 sync)."
    )
    parser.add_argument("source", help="s3://bucket[/prefix]")
    parser.add_argument("destination", type=Path, help="local directory")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="remove local files that have no object under the prefix",
    )
    args = parser.parse_args(argv)
    return sync(args.source, args.destination, delete=args.delete)


if __name__ == "__main__":
    sys.exit(main())
