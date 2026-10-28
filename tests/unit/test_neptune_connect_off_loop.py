# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neptune connect/close run off the event-loop thread (AWS-free).

Regression: ``aretrieve`` read ``NeptuneClient.g`` on the loop thread. Opening
the connection runs gremlinpython's websocket handshake with
``run_until_complete`` on a private loop, which raises "Cannot run the event
loop while another loop is running" under plain asyncio; it only worked because
the CLIs apply ``nest_asyncio``. The fake transport below fails the same way,
and also asserts directly that no loop is running on the connecting thread (so
the test holds even when another test module has applied ``nest_asyncio``).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest

from unified_kg_rag.adapters.aws import neptune as neptune_module
from unified_kg_rag.adapters.aws.neptune import NeptuneClient
from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.domain.models import Config, SearchQuery
from unified_kg_rag.shared import StoreConnectionError

pytestmark = pytest.mark.unit


class _FakeGremlinConnection:
    """Mimics DriverRemoteConnection's blocking, loop-driven connect."""

    threads: list[threading.Thread] = []
    close_threads: list[threading.Thread] = []

    def __init__(self, **_: Any) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "Cannot run the event loop while another loop is running"
            )
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(asyncio.sleep(0))
        finally:
            loop.close()
        type(self).threads.append(threading.current_thread())
        self._closed = False

    def is_closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        type(self).close_threads.append(threading.current_thread())
        self._closed = True


@pytest.fixture
def retriever(mocker) -> NeptuneRetriever:
    _FakeGremlinConnection.threads = []
    _FakeGremlinConnection.close_threads = []
    mocker.patch.object(
        neptune_module, "DriverRemoteConnection", _FakeGremlinConnection
    )
    mocker.patch.object(neptune_module, "traversal", return_value=MagicMock())
    config = Config()
    config.aws.neptune.endpoint = "neptune.example.invalid"
    config.aws.neptune.use_iam = False
    client = NeptuneClient(config=config, boto_session=MagicMock())
    # `traversal` is a MagicMock: every traversal built from it yields no rows.
    return NeptuneRetriever(config=config, neptune_client=client)


_SEEDED = SearchQuery(query="Vendor", filters={"id": ["e1"]})


def test_connect_inside_asyncio_run_without_nest_asyncio(retriever) -> None:
    main = threading.current_thread()

    async def query() -> Any:
        return await retriever.aretrieve(_SEEDED)

    loop = asyncio.new_event_loop()  # plain loop, nothing patched in
    try:
        assert loop.run_until_complete(query()) == []
    finally:
        loop.close()

    assert len(_FakeGremlinConnection.threads) == 1
    assert _FakeGremlinConnection.threads[0] is not main


async def test_aclose_closes_the_connection_off_the_loop(retriever) -> None:
    await retriever.aretrieve(_SEEDED)
    await retriever.aclose()
    assert len(_FakeGremlinConnection.close_threads) == 1
    assert _FakeGremlinConnection.close_threads[0] is not threading.current_thread()


async def test_a_traversal_racing_close_leaves_no_connection_open(
    retriever, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: a traversal whose worker waited on the client lock while
    # close() held it opened a new connection once close() released it; the
    # closed retriever never closed that one (its websocket and threads
    # stayed for the life of the process).
    closing, release = threading.Event(), threading.Event()
    close = _FakeGremlinConnection.close

    def slow_close(self: _FakeGremlinConnection) -> None:
        closing.set()
        release.wait(5)
        close(self)

    monkeypatch.setattr(_FakeGremlinConnection, "close", slow_close)
    await retriever.aretrieve(_SEEDED)  # opens the connection
    closer = asyncio.create_task(retriever.aclose())
    assert await asyncio.to_thread(closing.wait, 5)
    late = asyncio.create_task(retriever.aretrieve(_SEEDED))
    await asyncio.sleep(0.2)  # its worker now waits on the lock close() holds
    release.set()
    await closer
    with pytest.raises(StoreConnectionError):
        await late
    opened = _FakeGremlinConnection.threads
    assert len(_FakeGremlinConnection.close_threads) == len(opened)


async def test_a_closed_retriever_opens_no_connection(retriever) -> None:
    await retriever.aclose()
    with pytest.raises(StoreConnectionError, match="closed"):
        await retriever.aretrieve(_SEEDED)
    assert _FakeGremlinConnection.threads == []
    assert retriever._executor is None
