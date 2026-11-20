# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every S3 client of the package is built by one factory, with explicit
timeouts and botocore's standard retries with 5 requests (not the SDK
defaults: 60 s to connect, standard retries with 3 requests)."""

from __future__ import annotations

from typing import Any

import boto3
import pytest
from moto import mock_aws

from unified_kg_rag.adapters.aws.s3_cache import S3CacheManager
from unified_kg_rag.adapters.aws.s3_transfer import (
    S3_CONNECT_TIMEOUT_SECONDS,
    S3_READ_TIMEOUT_SECONDS,
    S3_TOTAL_MAX_ATTEMPTS,
    s3_client_for,
)
from unified_kg_rag.adapters.source_files.locations import SourceReader
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit

_REGION = "us-east-1"
_BUCKET = "client-config-bucket"


def _assert_configured(client: Any) -> None:
    config = client.meta.config
    assert config.connect_timeout == S3_CONNECT_TIMEOUT_SECONDS
    assert config.read_timeout == S3_READ_TIMEOUT_SECONDS
    assert config.retries == {
        "mode": "standard",
        "total_max_attempts": S3_TOTAL_MAX_ATTEMPTS,
    }


def test_the_values_have_their_documented_basis() -> None:
    # Connect like the package's other AWS clients (Bedrock, the registry);
    # read as botocore does; as many requests as the legacy mode made.
    assert (S3_CONNECT_TIMEOUT_SECONDS, S3_READ_TIMEOUT_SECONDS) == (10.0, 60.0)
    assert S3_TOTAL_MAX_ATTEMPTS == 5


@pytest.fixture
def session() -> Any:
    with mock_aws():
        session = boto3.Session(region_name=_REGION)
        session.client("s3").create_bucket(Bucket=_BUCKET)
        yield session


def test_the_stage_cache_client(session: Any) -> None:
    manager = S3CacheManager(
        Config(), boto_session=session, bucket_name=_BUCKET, prefix="p"
    )
    _assert_configured(manager.s3_client)


def test_the_cli_path_client(session: Any) -> None:
    config = Config()
    config.aws.region_name = _REGION
    _assert_configured(s3_client_for(config))


def test_the_source_reader_client(session: Any) -> None:
    class _Provider:
        location = f"s3://{_BUCKET}/x.ttl"

        def client(self, service_name: str, **kwargs: Any) -> Any:
            kwargs.setdefault("region_name", _REGION)
            return session.client(service_name, **kwargs)

    _assert_configured(SourceReader(_Provider())._s3())
