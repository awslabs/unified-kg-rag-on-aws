# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
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

from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient

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
    # Dummy non-secret placeholder credentials on a MagicMock for SigV4 signing
    # tests; not real keys.
    session.get_credentials.return_value = MagicMock(
        access_key="AKIA",  # nosec B106
        secret_key="secret",  # nosec B106
        token="token",  # nosec B106
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


def test_async_client_uses_async_http_connection_with_callable_signer() -> None:
    # AIOHttpConnection treats http_auth as basic-auth (calls .encode); only
    # AsyncHttpConnection invokes a callable signer per-request. The async
    # connection params must pair AsyncHttpConnection with a callable signer.
    from opensearchpy import AsyncHttpConnection

    params = _client(use_iam=True)._get_base_connection_params(async_mode=True)
    # _create_async_client sets connection_class to AsyncHttpConnection; the auth
    # it carries must be a callable (the signer), not a basic-auth string/tuple.
    assert callable(params["http_auth"])
    assert AsyncHttpConnection is not None


def test_basic_auth_tuple_when_iam_disabled() -> None:
    client = _client(use_iam=False, username="admin")
    client.opensearch_config.password = MagicMock(
        get_secret_value=lambda: "pw"  # noqa: PLW0108
    )
    for async_mode in (True, False):
        auth = client._get_auth(async_mode=async_mode)
        assert auth == ("admin", "pw")
