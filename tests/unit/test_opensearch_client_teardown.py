# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource-lifecycle teardown for OpenSearchClient (AWS-free).

Regression: the client never closed its sync/async HTTP connections, leaking
sockets ("Unclosed client session"), and on event-loop rotation it replaced
``_async_client`` without closing the previous one (one leaked pool per loop).
These tests assert close()/aclose()/context-manager support and that a loop
change closes the prior async client. The client is built via ``__new__`` so no
real connection is opened; the underlying clients are mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aws_graphrag.adapters.aws.opensearch import OpenSearchClient

pytestmark = pytest.mark.unit


def _client() -> OpenSearchClient:
    client = OpenSearchClient.__new__(OpenSearchClient)
    client._client = None
    client._async_client = None
    client._bound_loop_id = None
    return client


def test_close_closes_sync_client() -> None:
    client = _client()
    sync = MagicMock()
    client._client = sync
    client.close()
    sync.close.assert_called_once()
    assert client._client is None


def test_close_discards_async_client() -> None:
    client = _client()
    client._async_client = MagicMock()  # no real transport -> discard is a no-op
    client._bound_loop_id = 123
    client.close()
    assert client._async_client is None
    assert client._bound_loop_id is None


def test_close_never_raises_on_broken_sync_client() -> None:
    client = _client()
    sync = MagicMock()
    sync.close.side_effect = RuntimeError("boom")
    client._client = sync
    # Best-effort teardown must swallow errors.
    client.close()
    assert client._client is None


async def test_aclose_awaits_async_client_close() -> None:
    client = _client()
    async_client = MagicMock()
    async_client.close = AsyncMock()
    client._async_client = async_client
    client._bound_loop_id = 99
    sync = MagicMock()
    client._client = sync

    await client.aclose()

    async_client.close.assert_awaited_once()
    sync.close.assert_called_once()
    assert client._async_client is None
    assert client._client is None
    assert client._bound_loop_id is None


async def test_aclose_never_raises_on_broken_async_client() -> None:
    client = _client()
    async_client = MagicMock()
    async_client.close = AsyncMock(side_effect=RuntimeError("boom"))
    client._async_client = async_client
    await client.aclose()
    assert client._async_client is None


def test_context_manager_calls_close() -> None:
    client = _client()
    sync = MagicMock()
    client._client = sync
    with client as entered:
        assert entered is client
    sync.close.assert_called_once()


async def test_async_context_manager_calls_aclose() -> None:
    client = _client()
    async_client = MagicMock()
    async_client.close = AsyncMock()
    client._async_client = async_client
    async with client as entered:
        assert entered is client
    async_client.close.assert_awaited_once()


def test_loop_change_discards_previous_async_client(monkeypatch) -> None:
    """When the bound loop id changes, the old async client is discarded
    (its connector closed) before a new one is created — no leaked pool."""
    client = _client()

    # Build an old async client whose transport exposes a connector we can
    # assert is closed by the eager-discard path.
    connector = MagicMock()
    session = MagicMock(connector=connector)
    connection = MagicMock(session=session)
    pool = MagicMock(connections=[connection])
    transport = MagicMock(connection_pool=pool)
    old_async = MagicMock(transport=transport)
    client._async_client = old_async
    client._bound_loop_id = 1

    # Force the property to see a *different* current loop id, and stub the
    # async-client factory so no real AsyncOpenSearch is constructed.
    new_async = MagicMock()
    monkeypatch.setattr(client, "_get_current_loop_id", lambda: 2)
    monkeypatch.setattr(client, "_create_async_client", lambda: new_async)

    result = client.async_client

    # The previous client's connector was closed (no leaked aiohttp pool) and
    # the replacement is now bound to the new loop.
    connector.close.assert_called_once()
    assert result is new_async
    assert client._bound_loop_id == 2


def test_discard_async_client_never_raises() -> None:
    # A client whose transport attributes are missing must not raise.
    OpenSearchClient._discard_async_client(MagicMock(transport=None))
