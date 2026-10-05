# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import boto3
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError, NoCredentialsError

from unified_kg_rag.domain.models import Config, S3EncryptionType
from unified_kg_rag.shared import CacheSyncError, get_logger

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

logger = get_logger(__name__)


class S3CacheManager:
    """Mirror a pipeline's local stage cache to and from S3.

    Both sync directions raise :class:`CacheSyncError` on any failure (listing,
    connection, or any single object) instead of returning a partial result: a
    phased run hands checkpoints between tasks through S3, so a silent partial
    sync would surface one phase later as a misleading "missing stage" error.
    """

    DEFAULT_S3_PREFIX: ClassVar[str] = "cache"
    # Failed keys quoted in a CacheSyncError message (the full count is always
    # reported); keeps the message readable for large partial failures.
    _MAX_REPORTED_FAILURES: ClassVar[int] = 5

    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        bucket_name: str | None = None,
        prefix: str | None = None,
    ) -> None:
        self.config = config
        self.s3_config = config.aws.s3
        self.bucket_name = bucket_name or self.s3_config.bucket_name
        if not self.bucket_name:
            raise ValueError("S3 'bucket_name' must be configured.")

        self.prefix = (prefix or self.DEFAULT_S3_PREFIX).strip("/")
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name,
            region_name=self.config.aws.region_name,
        )
        self._s3_client: S3Client | None = None

        logger.info(
            "Initialized S3CacheManager for 's3://%s/%s'", self.bucket_name, self.prefix
        )

    @property
    def s3_client(self) -> S3Client:
        if self._s3_client is None:
            try:
                self._s3_client = self.boto_session.client("s3")
                self._s3_client.head_bucket(Bucket=str(self.bucket_name))
            except NoCredentialsError:
                logger.error(
                    "AWS credentials not found. Please configure your credentials."
                )
                raise
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                if error_code in ["404", "NoSuchBucket"]:
                    logger.error("S3 bucket not found: %s", self.bucket_name)
                else:
                    logger.error(
                        "Failed to connect to S3 bucket '%s': %s", self.bucket_name, e
                    )
                raise
        return self._s3_client

    def sync_pipeline_from_s3(
        self, pipeline_id: str, local_cache_dir: Path
    ) -> dict[str, bool]:
        logger.info("Syncing pipeline '%s' from S3", pipeline_id)
        results: dict[str, bool] = {}
        failed_keys: list[str] = []
        base_prefix = self._get_base_prefix(pipeline_id)
        s3_prefix = f"{base_prefix}/"

        try:
            local_cache_dir.mkdir(parents=True, exist_ok=True)
            paginator = self.s3_client.get_paginator("list_objects_v2")
            total_files = 0
            downloaded_stages = set()

            for page in paginator.paginate(
                Bucket=str(self.bucket_name), Prefix=s3_prefix
            ):
                for obj in page.get("Contents", []):
                    s3_key = obj.get("Key")
                    if not s3_key:
                        continue

                    total_files += 1
                    relative_path = Path(s3_key).relative_to(base_prefix)
                    local_path = local_cache_dir / relative_path
                    # Guard against path traversal: an S3 key containing '..'
                    # segments (tampered/shared bucket) could otherwise resolve
                    # outside local_cache_dir and overwrite arbitrary files.
                    # relative_to() only strips the prefix; it does NOT normalize.
                    resolved = local_path.resolve()
                    cache_root = local_cache_dir.resolve()
                    if not resolved.is_relative_to(cache_root):
                        logger.warning(
                            "Skipping S3 key '%s': resolves outside the cache dir",
                            s3_key,
                        )
                        continue
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    success = self._download_cache_file(s3_key, local_path)
                    if not success:
                        failed_keys.append(s3_key)
                    stage_name = (
                        relative_path.parts[0]
                        if len(relative_path.parts) > 1
                        else relative_path.stem
                    )

                    if stage_name not in downloaded_stages:
                        results[stage_name] = success
                    else:
                        results[stage_name] = results[stage_name] and success
                    downloaded_stages.add(stage_name)

        except Exception as e:
            logger.error("Failed to sync pipeline '%s' from S3: %s", pipeline_id, e)
            raise CacheSyncError(
                f"Failed to sync pipeline '{pipeline_id}' from "
                f"'s3://{self.bucket_name}/{s3_prefix}': {e}"
            ) from e

        self._raise_on_failures("download", pipeline_id, failed_keys, total_files)
        if total_files > 0:
            logger.info(
                "Download completed: %s stages, %s files from 's3://%s/%s'",
                len(results),
                total_files,
                self.bucket_name,
                s3_prefix,
            )
        else:
            logger.info("No cache files found for pipeline '%s' in S3", pipeline_id)
        return results

    def sync_pipeline_to_s3(
        self, pipeline_id: str, local_cache_dir: Path
    ) -> dict[str, bool]:
        logger.info("Syncing pipeline '%s' to S3", pipeline_id)
        if not local_cache_dir.is_dir():
            logger.warning("Local cache directory not found: '%s'", local_cache_dir)
            return {}

        base_prefix = self._get_base_prefix(pipeline_id)
        stage_results: defaultdict[str, list[bool]] = defaultdict(list)
        failed_keys: list[str] = []
        total_files = 0

        try:
            for local_path in local_cache_dir.rglob("*.json"):
                total_files += 1
                relative_path = local_path.relative_to(local_cache_dir)
                s3_key = f"{base_prefix}/{relative_path.as_posix()}"
                success = self._upload_file_to_s3(local_path, s3_key)
                if not success:
                    failed_keys.append(s3_key)

                stage_name = (
                    relative_path.parts[0]
                    if len(relative_path.parts) > 1
                    else relative_path.stem
                )
                stage_results[stage_name].append(success)

        except Exception as e:
            logger.error("Failed to sync pipeline '%s' to S3: %s", pipeline_id, e)
            raise CacheSyncError(
                f"Failed to sync pipeline '{pipeline_id}' to "
                f"'s3://{self.bucket_name}/{base_prefix}': {e}"
            ) from e

        self._raise_on_failures("upload", pipeline_id, failed_keys, total_files)
        results = {stage: all(outcomes) for stage, outcomes in stage_results.items()}
        if total_files > 0:
            logger.info(
                "Upload completed: %s stages, %s files to 's3://%s/%s'",
                len(results),
                total_files,
                self.bucket_name,
                base_prefix,
            )
        else:
            logger.info("No cache files found for pipeline '%s' locally", pipeline_id)
        return results

    def _raise_on_failures(
        self, direction: str, pipeline_id: str, failed_keys: list[str], total: int
    ) -> None:
        if not failed_keys:
            return
        shown = ", ".join(failed_keys[: self._MAX_REPORTED_FAILURES])
        more = len(failed_keys) - self._MAX_REPORTED_FAILURES
        suffix = f" (+{more} more)" if more > 0 else ""
        raise CacheSyncError(
            f"S3 cache {direction} for pipeline '{pipeline_id}' failed for "
            f"{len(failed_keys)}/{total} files in bucket '{self.bucket_name}': "
            f"{shown}{suffix}"
        )

    def _get_base_prefix(self, pipeline_id: str) -> str:
        return f"{self.prefix}/{pipeline_id}" if self.prefix else pipeline_id

    def _download_cache_file(self, s3_key: str, local_path: Path) -> bool:
        try:
            self.s3_client.download_file(str(self.bucket_name), s3_key, str(local_path))
            return True
        except ClientError as e:
            logger.warning(
                "Failed to download 's3://%s/%s': %s", self.bucket_name, s3_key, e
            )
            return False

    def _upload_file_to_s3(self, local_path: Path, s3_key: str) -> bool:
        try:
            extra_args = {}
            encryption_conf = self.s3_config.encryption

            if encryption_conf.encryption_type == S3EncryptionType.AES256:
                extra_args["ServerSideEncryption"] = "AES256"
            elif encryption_conf.encryption_type == S3EncryptionType.KMS:
                extra_args["ServerSideEncryption"] = "aws:kms"
                if encryption_conf.kms_key_id:
                    extra_args["SSEKMSKeyId"] = encryption_conf.kms_key_id

            self.s3_client.upload_file(
                str(local_path), str(self.bucket_name), s3_key, ExtraArgs=extra_args
            )
            return True
        except (ClientError, S3UploadFailedError, FileNotFoundError) as e:
            # boto3's managed upload_file wraps ClientError in
            # S3UploadFailedError, so both must count as a per-file failure.
            logger.warning(
                "Failed to upload '%s' to 's3://%s/%s': %s",
                local_path,
                self.bucket_name,
                s3_key,
                e,
            )
            return False
