# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Auth construction for OpenSearchClient (AWS-free).

Regression: the async client (AIOHttpConnection) was given a
requests_aws4auth.AWS4Auth, which it treats as a basic-auth credential and
calls .encode() on -> "'AWS4Auth' object has no attribute 'encode'", silently
returning zero search hits for every LightRAG/GraphRAG query. The async path
must use opensearch-py's AWSV4SignerAsyncAuth; the sync path AWSV4SignerAuth.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from opensearchpy import AWSV4SignerAsyncAuth, AWSV4SignerAuth

from aws_graphrag.adapters.aws.opensearch import OpenSearchClient

pytestmark = pytest.mark.unit


def _client(*, use_iam: bool, username: str | None = None) -> OpenSearchClient:
    client = OpenSearchClient.__new__(OpenSearchClient)
    oc = MagicMock()
    oc.use_iam = use_iam
    oc.endpoint = "search.example.com"
    oc.port = 443
    oc.username = username
    oc.password = None
    client.opensearch_config = oc
    cfg = MagicMock()
    cfg.aws.region_name = "ap-northeast-2"
    client.config = cfg
    session = MagicMock()
    session.get_credentials.return_value = MagicMock(
        access_key="AKIA", secret_key="secret", token="token"
    )
    client.boto_session = session
    return client


def test_iam_sync_auth_uses_sync_signer() -> None:
    auth = _client(use_iam=True)._get_auth(async_mode=False)
    assert isinstance(auth, AWSV4SignerAuth)


def test_iam_async_auth_uses_async_signer() -> None:
    # The crux of the bug: async MUST NOT get a sync/AWS4Auth credential.
    auth = _client(use_iam=True)._get_auth(async_mode=True)
    assert isinstance(auth, AWSV4SignerAsyncAuth)
    # An AWSV4Signer has no .encode (it is not a basic-auth string) — the very
    # attribute AIOHttpConnection tried to call on the old AWS4Auth.
    assert not hasattr(auth, "encode")


def test_async_and_sync_signers_are_distinct_types() -> None:
    client = _client(use_iam=True)
    assert type(client._get_auth(async_mode=True)) is not type(
        client._get_auth(async_mode=False)
    )


def test_basic_auth_tuple_when_iam_disabled() -> None:
    client = _client(use_iam=False, username="admin")
    client.opensearch_config.password = MagicMock(
        get_secret_value=lambda: "pw"  # noqa: PLW0108
    )
    for async_mode in (True, False):
        auth = client._get_auth(async_mode=async_mode)
        assert auth == ("admin", "pw")
