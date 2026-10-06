# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discarding a rotated AsyncOpenSearch client is await-safe (AWS-free).

Regression: the discard path called aiohttp's ``connector.close()`` (a
coroutine-like awaitable in aiohttp 3.x) without awaiting it, and never
closed a client whose loop was still alive, which surfaced as "Unclosed client
session" / never-awaited warnings in E2E logs. A client whose loop still runs
is now closed there (awaited); otherwise the connector is closed and its
awaitable settled.
"""

from __future__ import annotations

import asyncio
import gc
import warnings
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient
from unified_kg_rag.application.retrieval.rag_chain import _LoopRunner

pytestmark = pytest.mark.unit


def _client_over(session: aiohttp.ClientSession) -> MagicMock:
    connection = MagicMock(session=session)
    pool = MagicMock(connections=[connection])
    return MagicMock(transport=MagicMock(connection_pool=pool))


def test_discard_after_the_loop_closed_emits_no_warnings() -> None:
    loop = asyncio.new_event_loop()

    async def make_session() -> aiohttp.ClientSession:
        return aiohttp.ClientSession()

    session = loop.run_until_complete(make_session())
    loop.close()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        OpenSearchClient._discard_async_client(_client_over(session), loop)
        assert session.closed
        del session
        gc.collect()

    messages = [str(w.message) for w in caught]
    assert not [m for m in messages if "await" in m or "Unclosed" in m], messages


def test_discard_while_the_loop_runs_awaits_close_there() -> None:
    runner = _LoopRunner()
    seen: list[asyncio.AbstractEventLoop] = []

    async def close() -> None:
        seen.append(asyncio.get_running_loop())

    async def bound_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    try:
        loop = runner.run(bound_loop())
        async_client = MagicMock()
        async_client.close = close
        OpenSearchClient._discard_async_client(async_client, loop)
        runner.run(asyncio.sleep(0.01))
        assert seen == [loop]
    finally:
        runner.stop()


async def test_close_from_inside_the_bound_loop_schedules_awaited_close() -> None:
    client = OpenSearchClient.__new__(OpenSearchClient)
    client._client = None
    async_client = MagicMock()
    async_client.close = AsyncMock()
    client._async_client = async_client
    client._bound_loop_id = id(asyncio.get_running_loop())
    client._bound_loop = asyncio.get_running_loop()

    client.close()
    await asyncio.sleep(0)

    async_client.close.assert_awaited_once()
    assert client._async_client is None and client._bound_loop is None


async def test_async_client_binds_the_loop_object(monkeypatch) -> None:
    client = OpenSearchClient.__new__(OpenSearchClient)
    client._client = None
    client._async_client = None
    client._bound_loop_id = None
    client._bound_loop = None
    monkeypatch.setattr(client, "_create_async_client", MagicMock)

    client.async_client  # noqa: B018 - property access binds the loop

    assert client._bound_loop is asyncio.get_running_loop()
