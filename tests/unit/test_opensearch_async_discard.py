# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Discarding a rotated AsyncOpenSearch client is await-safe (AWS-free).

Regression: the discard path called aiohttp's ``connector.close()`` (a
coroutine-like awaitable in aiohttp 3.x) without awaiting it, and never
closed a client whose loop was still alive, which surfaced as "Unclosed client
session" / never-awaited warnings in E2E logs. The connectors are closed
synchronously on the thread of the loop that owns them (posted there as a
callback when that loop runs on another thread), or at once when the loop is
gone, the connector's awaitable settled.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import warnings
from unittest.mock import MagicMock

import aiohttp
import pytest

from unified_kg_rag.adapters.aws import opensearch
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


def _bound(
    async_client: MagicMock, loop: asyncio.AbstractEventLoop
) -> OpenSearchClient:
    client = OpenSearchClient.__new__(OpenSearchClient)
    client._client = None
    client._async_client = async_client
    client._bound_loop_id = id(loop)
    client._bound_loop = loop
    return client


def test_discard_while_the_loop_runs_closes_the_connections_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # On the loop that owns them, as a callback (never a task its loop's
    # end can cancel before it runs).
    runner = _LoopRunner()
    seen: list[asyncio.AbstractEventLoop] = []
    close = opensearch._close_connections

    def recorded(async_client: MagicMock) -> None:
        seen.append(asyncio.get_running_loop())
        close(async_client)

    monkeypatch.setattr(opensearch, "_close_connections", recorded)

    async def make_session() -> aiohttp.ClientSession:
        return aiohttp.ClientSession()

    try:
        session = runner.run(make_session())
        loop = runner.run(_running())
        OpenSearchClient._discard_async_client(_client_over(session), loop)
        runner.run(asyncio.sleep(0.01))
        assert seen == [loop]
        assert session.closed
    finally:
        runner.stop()


async def _running() -> asyncio.AbstractEventLoop:
    return asyncio.get_running_loop()


async def test_close_from_inside_the_bound_loop_closes_the_connections_at_once() -> (
    None
):
    session = aiohttp.ClientSession()
    client = _bound(_client_over(session), asyncio.get_running_loop())

    client.close()

    # Before any await: nothing is left for the loop to run.
    assert session.closed
    assert client._async_client is None and client._bound_loop is None


def test_a_sync_close_that_blocks_the_bound_loop_still_closes_the_session() -> None:
    # R4B-1: GraphRAGChain.close() called from a coroutine runs the
    # retrievers' sync closes on a worker thread while the loop's thread
    # waits for it; the awaited close the worker scheduled on that loop was
    # cancelled by asyncio.run before its first step, and the session and
    # its sockets stayed open (7 of 7 in probes/chain_close_leak.py).
    held: list[aiohttp.ClientSession] = []

    async def main() -> None:
        session = aiohttp.ClientSession()
        held.append(session)
        async_client = _client_over(session)

        async def close() -> None:
            await session.close()

        async_client.close = close
        client = _bound(async_client, asyncio.get_running_loop())
        worker = threading.Thread(target=client.close)
        worker.start()
        worker.join()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        asyncio.run(main())
        assert held[0].closed
        held.clear()
        gc.collect()
    assert not [w for w in caught if "Unclosed" in str(w.message)]


async def test_async_client_binds_the_loop_object(monkeypatch) -> None:
    client = OpenSearchClient.__new__(OpenSearchClient)
    client._client = None
    client._async_client = None
    client._bound_loop_id = None
    client._bound_loop = None
    monkeypatch.setattr(client, "_create_async_client", MagicMock)

    client.async_client  # noqa: B018 - property access binds the loop

    assert client._bound_loop is asyncio.get_running_loop()
