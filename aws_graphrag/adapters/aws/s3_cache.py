# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from types_boto3_s3 import S3Client

from aws_graphrag.domain.models import Config, S3EncryptionType
from aws_graphrag.shared import get_logger

logger = get_logger(__name__)


class S3CacheManager:
    DEFAULT_S3_PREFIX: ClassVar[str] = "cache"

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
            f"Initialized S3CacheManager for 's3://{self.bucket_name}/{self.prefix}'"
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
                    logger.error(f"S3 bucket not found: {self.bucket_name}")
                else:
                    logger.error(
                        f"Failed to connect to S3 bucket '{self.bucket_name}': {e}"
                    )
                raise
        return self._s3_client

    def sync_pipeline_from_s3(
        self, pipeline_id: str, local_cache_dir: Path
    ) -> dict[str, bool]:
        logger.info(f"Syncing pipeline '{pipeline_id}' from S3")
        results = {}
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
                    local_path.parent.mkdir(parents=True, exist_ok=True)

                    success = self._download_cache_file(s3_key, local_path)
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

            success_count = sum(results.values())
            if total_files > 0:
                logger.info(
                    f"Download completed: {success_count}/{len(results)} stages, "
                    f"{total_files} files from 's3://{self.bucket_name}/{s3_prefix}'"
                )
            else:
                logger.info(f"No cache files found for pipeline '{pipeline_id}' in S3")
        except Exception as e:
            logger.error(f"Failed to sync pipeline '{pipeline_id}' from S3: {e}")

        return results

    def sync_pipeline_to_s3(
        self, pipeline_id: str, local_cache_dir: Path
    ) -> dict[str, bool]:
        logger.info(f"Syncing pipeline '{pipeline_id}' to S3")
        if not local_cache_dir.is_dir():
            logger.warning(f"Local cache directory not found: '{local_cache_dir}'")
            return {}

        base_prefix = self._get_base_prefix(pipeline_id)
        stage_results = defaultdict(list)
        total_files = 0

        try:
            for local_path in local_cache_dir.rglob("*.json"):
                total_files += 1
                relative_path = local_path.relative_to(local_cache_dir)
                s3_key = f"{base_prefix}/{relative_path.as_posix()}"
                success = self._upload_file_to_s3(local_path, s3_key)

                stage_name = (
                    relative_path.parts[0]
                    if len(relative_path.parts) > 1
                    else relative_path.stem
                )
                stage_results[stage_name].append(success)

            results = {
                stage: all(outcomes) for stage, outcomes in stage_results.items()
            }
            success_count = sum(results.values())

            if total_files > 0:
                logger.info(
                    f"Upload completed: {success_count}/{len(results)} stages, "
                    f"{total_files} files to 's3://{self.bucket_name}/{base_prefix}'"
                )
            else:
                logger.info(
                    f"No cache files found for pipeline '{pipeline_id}' locally"
                )

            return results
        except Exception as e:
            logger.error(f"Failed to sync pipeline '{pipeline_id}' to S3: {e}")
            return dict.fromkeys(stage_results, False)

    def _get_base_prefix(self, pipeline_id: str) -> str:
        return f"{self.prefix}/{pipeline_id}" if self.prefix else pipeline_id

    def _download_cache_file(self, s3_key: str, local_path: Path) -> bool:
        try:
            self.s3_client.download_file(str(self.bucket_name), s3_key, str(local_path))
            return True
        except ClientError as e:
            logger.warning(
                f"Failed to download 's3://{self.bucket_name}/{s3_key}': {e}"
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
        except (ClientError, FileNotFoundError) as e:
            logger.warning(
                f"Failed to upload '{local_path}' to "
                f"'s3://{self.bucket_name}/{s3_key}': {e}"
            )
            return False
