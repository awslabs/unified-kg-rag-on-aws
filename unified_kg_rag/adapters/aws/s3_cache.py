# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import boto3
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import ClientError, NoCredentialsError

from unified_kg_rag.domain.models import Config, S3EncryptionType
from unified_kg_rag.shared import CacheSyncError, ConfigurationError, get_logger
from unified_kg_rag.shared.pipeline_manager import PipelineStateManager

if TYPE_CHECKING:
    from types_boto3_s3 import S3Client

    from unified_kg_rag.domain.models.config import S3EncryptionConfig

logger = get_logger(__name__)


def sse_extra_args(encryption_conf: S3EncryptionConfig) -> dict[str, str]:
    """Per-object SSE upload args for the configured encryption type.

    BUCKET_DEFAULT sends no header so the bucket's
    default encryption applies. Sending ``ServerSideEncryption=AES256``
    would silently override a bucket whose default is a KMS CMK. Every S3
    upload of pipeline data uses these args.
    """
    if encryption_conf.encryption_type == S3EncryptionType.AES256:
        return {"ServerSideEncryption": "AES256"}
    if encryption_conf.encryption_type == S3EncryptionType.KMS:
        extra_args = {"ServerSideEncryption": "aws:kms"}
        if encryption_conf.kms_key_id:
            extra_args["SSEKMSKeyId"] = encryption_conf.kms_key_id
        return extra_args
    return {}


class S3CacheManager:
    """Mirror a pipeline's local stage cache to and from S3.

    Both sync directions raise :class:`CacheSyncError` on any failure (listing,
    connection, or any single object) instead of returning a partial result: a
    phased run hands checkpoints between tasks through S3, so a silent partial
    sync would surface one phase later as a misleading "missing stage" error.

    The pipeline uploads after every stage, so an upload sends only the files
    that changed since this manager last uploaded or downloaded them, stage
    data before the index and the run metadata that refer to it.
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
            raise ConfigurationError("S3 'bucket_name' must be configured.")

        self.prefix = (prefix or self.DEFAULT_S3_PREFIX).strip("/")
        self.boto_session = boto_session or boto3.Session(
            profile_name=self.config.aws.profile_name,
            region_name=self.config.aws.region_name,
        )
        self._s3_client: S3Client | None = None
        # Local cache file -> (size, mtime_ns) when its S3 copy was last
        # written or read; a file that still matches is not uploaded again.
        self._synced: dict[Path, tuple[int, int]] = {}

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

    def sync_pipeline_from_s3(self, pipeline_id: str, local_cache_dir: Path) -> None:
        """Download the pipeline's cache files; raises CacheSyncError on failure."""
        logger.info("Syncing pipeline '%s' from S3", pipeline_id)
        stages: set[str] = set()
        failed_keys: list[str] = []
        base_prefix = self._get_base_prefix(pipeline_id)
        s3_prefix = f"{base_prefix}/"

        try:
            local_cache_dir.mkdir(parents=True, exist_ok=True)
            paginator = self.s3_client.get_paginator("list_objects_v2")
            total_files = 0

            for page in paginator.paginate(
                Bucket=str(self.bucket_name), Prefix=s3_prefix
            ):
                for obj in page.get("Contents", []):
                    s3_key = obj.get("Key")
                    if not s3_key:
                        continue
                    # S3 "folder markers" (zero-byte keys ending in '/', created
                    # by the console or some sync tools) are not cache files;
                    # downloading one onto a directory path would fail the sync.
                    if s3_key.endswith("/"):
                        logger.debug("Skipping S3 folder marker '%s'", s3_key)
                        continue
                    relative_path = Path(s3_key).relative_to(base_prefix)
                    if relative_path == Path("."):
                        logger.debug("Skipping S3 folder marker '%s'", s3_key)
                        continue

                    total_files += 1
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

                    if self._download_cache_file(s3_key, local_path):
                        self._synced[resolved] = self._file_state(resolved)
                    else:
                        failed_keys.append(s3_key)
                    stages.add(self._stage_name(relative_path))

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
                len(stages),
                total_files,
                self.bucket_name,
                s3_prefix,
            )
        else:
            logger.info("No cache files found for pipeline '%s' in S3", pipeline_id)

    def sync_pipeline_to_s3(self, pipeline_id: str, local_cache_dir: Path) -> None:
        """Upload the pipeline's changed cache files; raises CacheSyncError on
        failure (a file that failed is uploaded again by the next call)."""
        logger.info("Syncing pipeline '%s' to S3", pipeline_id)
        if not local_cache_dir.is_dir():
            logger.warning("Local cache directory not found: '%s'", local_cache_dir)
            return

        base_prefix = self._get_base_prefix(pipeline_id)
        stages: set[str] = set()
        failed_keys: list[str] = []
        total_files = 0
        unchanged = 0

        try:
            for local_path in self._upload_order(local_cache_dir):
                state = self._file_state(local_path)
                if self._synced.get(local_path.resolve()) == state:
                    unchanged += 1
                    continue
                total_files += 1
                relative_path = local_path.relative_to(local_cache_dir)
                s3_key = f"{base_prefix}/{relative_path.as_posix()}"
                if self._upload_file_to_s3(local_path, s3_key):
                    self._synced[local_path.resolve()] = state
                else:
                    failed_keys.append(s3_key)
                stages.add(self._stage_name(relative_path))

        except Exception as e:
            logger.error("Failed to sync pipeline '%s' to S3: %s", pipeline_id, e)
            raise CacheSyncError(
                f"Failed to sync pipeline '{pipeline_id}' to "
                f"'s3://{self.bucket_name}/{base_prefix}': {e}"
            ) from e

        self._raise_on_failures("upload", pipeline_id, failed_keys, total_files)
        if total_files > 0:
            logger.info(
                "Upload completed: %s stages, %s files to 's3://%s/%s' "
                "(%s unchanged files skipped)",
                len(stages),
                total_files,
                self.bucket_name,
                base_prefix,
                unchanged,
            )
        elif unchanged:
            logger.info(
                "S3 copy of pipeline '%s' is current (%s files)", pipeline_id, unchanged
            )
        else:
            logger.info("No cache files found for pipeline '%s' locally", pipeline_id)

    def upload_file(
        self, pipeline_id: str, local_cache_dir: Path, local_path: Path
    ) -> None:
        """Upload one file of the pipeline's cache now (it is then current
        for ``sync_pipeline_to_s3``); raises CacheSyncError on failure."""
        state = self._file_state(local_path)
        relative_path = local_path.relative_to(local_cache_dir)
        s3_key = f"{self._get_base_prefix(pipeline_id)}/{relative_path.as_posix()}"
        if not self._upload_file_to_s3(local_path, s3_key):
            raise CacheSyncError(
                f"S3 cache upload for pipeline '{pipeline_id}' failed for "
                f"'{s3_key}' in bucket '{self.bucket_name}'"
            )
        self._synced[local_path.resolve()] = state

    @staticmethod
    def _upload_order(local_cache_dir: Path) -> list[Path]:
        """The cache files, stage data first, then the cache index, then the
        run metadata: an upload cut short never leaves S3 holding an index
        entry or a completed stage whose data was not uploaded."""
        metadata = PipelineStateManager.PIPELINE_METADATA_FILE
        return sorted(
            local_cache_dir.rglob("*.json"),
            key=lambda path: (
                path.parent == local_cache_dir,
                path.name == metadata,
                path.as_posix(),
            ),
        )

    @staticmethod
    def _file_state(path: Path) -> tuple[int, int]:
        stat = path.stat()
        return stat.st_size, stat.st_mtime_ns

    @staticmethod
    def _stage_name(relative_path: Path) -> str:
        # Stage files sit in a per-stage folder; top-level files are named by stem.
        return (
            relative_path.parts[0]
            if len(relative_path.parts) > 1
            else relative_path.stem
        )

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
            extra_args = sse_extra_args(self.s3_config.encryption)
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
