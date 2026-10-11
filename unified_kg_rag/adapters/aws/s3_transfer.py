# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-object download and directory upload for ``s3://`` CLI paths.

Used by CLIs whose inputs/outputs may live in S3 (``run-eval`` on the bundled
ECS stack, where the image holds no dataset and the task disk is discarded on
exit). Uploads apply the same per-object SSE args as every other pipeline
upload (``sse_extra_args``). The AWS SDK is imported by the functions that
call S3, so parsing a path (``is_s3_uri``) loads none of it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from unified_kg_rag.shared import AWSServiceError, ConfigurationError, get_logger

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

    from unified_kg_rag.domain.models import Config

logger = get_logger(__name__)

S3_SCHEME = "s3://"
# S3 general-purpose bucket naming rules (3-63 chars; lowercase letters,
# digits, dots and hyphens; starts and ends with a letter or digit).
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$")


def is_s3_uri(value: str | Path | None) -> bool:
    """True when ``value`` is an ``s3://`` URI (a ``Path`` never is)."""
    return isinstance(value, str) and value.startswith(S3_SCHEME)


def parse_s3_uri(uri: str, *, require_key: bool) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``.

    ``require_key`` demands an object key (not empty, not ending in ``/``);
    otherwise the key is a prefix, returned without a trailing ``/`` (empty =
    bucket root). Raises :class:`ConfigurationError` on a malformed URI.
    """
    if not is_s3_uri(uri):
        raise ConfigurationError(f"Not an S3 URI (expected s3://bucket/key): '{uri}'")
    bucket, _, key = uri[len(S3_SCHEME) :].partition("/")
    if not _BUCKET_RE.match(bucket):
        raise ConfigurationError(f"Invalid S3 bucket name in '{uri}': '{bucket}'")
    if require_key:
        if not key or key.endswith("/"):
            raise ConfigurationError(
                f"S3 URI '{uri}' must name an object (s3://bucket/key), not a prefix"
            )
        return bucket, key
    return bucket, key.strip("/")


def s3_client_for(config: Config) -> S3Client:
    """An S3 client on the configured profile and region."""
    import boto3

    session = boto3.Session(
        profile_name=config.aws.profile_name, region_name=config.aws.region_name
    )
    client: S3Client = session.client("s3")
    return client


def download_s3_object(s3_client: S3Client, uri: str, local_path: Path) -> Path:
    """Download the object at ``uri`` to ``local_path``.

    Raises :class:`AWSServiceError` naming the URI on any S3 failure.
    """
    from botocore.exceptions import BotoCoreError, ClientError

    bucket, key = parse_s3_uri(uri, require_key=True)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3_client.download_file(bucket, key, str(local_path))
    except (ClientError, BotoCoreError) as e:
        raise AWSServiceError(f"Failed to download '{uri}': {e}") from e
    logger.info("Downloaded '%s' to '%s'", uri, local_path)
    return local_path


def upload_directory_to_s3(
    s3_client: S3Client, local_dir: Path, uri: str, config: Config
) -> list[str]:
    """Upload every file under ``local_dir`` to the ``uri`` prefix.

    Keys keep the files' paths relative to ``local_dir``. Every file is
    attempted; raises :class:`AWSServiceError` listing the failed keys if any
    upload failed, else returns the uploaded ``s3://`` URIs.
    """
    from boto3.exceptions import S3UploadFailedError
    from botocore.exceptions import BotoCoreError, ClientError

    from unified_kg_rag.adapters.aws.s3_cache import sse_extra_args

    bucket, prefix = parse_s3_uri(uri, require_key=False)
    extra_args = sse_extra_args(config.aws.s3.encryption)
    uploaded: list[str] = []
    failed: list[str] = []
    for path in sorted(p for p in local_dir.rglob("*") if p.is_file()):
        relative = path.relative_to(local_dir).as_posix()
        key = f"{prefix}/{relative}" if prefix else relative
        try:
            s3_client.upload_file(str(path), bucket, key, ExtraArgs=extra_args)
        except (ClientError, S3UploadFailedError, BotoCoreError) as e:
            # boto3's managed upload_file wraps ClientError in
            # S3UploadFailedError, so both count as a per-file failure.
            logger.warning(
                "Failed to upload '%s' to 's3://%s/%s': %s", path, bucket, key, e
            )
            failed.append(key)
            continue
        uploaded.append(f"{S3_SCHEME}{bucket}/{key}")
    if failed:
        raise AWSServiceError(
            f"Failed to upload {len(failed)}/{len(failed) + len(uploaded)} file(s) "
            f"to 's3://{bucket}/{prefix}': {', '.join(failed)}"
        )
    logger.info("Uploaded %s file(s) to '%s'", len(uploaded), uri)
    return uploaded
