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
    instance = NeptuneRetriever(
        config=config, neptune_client=client, boto_session=MagicMock()
    )

    async def no_seeds(g: Any, query: SearchQuery) -> tuple[list, list]:
        return [], []

    mocker.patch.object(instance, "_get_seed_nodes", side_effect=no_seeds)
    return instance


def test_connect_inside_asyncio_run_without_nest_asyncio(retriever) -> None:
    main = threading.current_thread()

    async def query() -> Any:
        return await retriever.aretrieve(SearchQuery(query="Vendor"))

    loop = asyncio.new_event_loop()  # plain loop, nothing patched in
    try:
        assert loop.run_until_complete(query()) == []
    finally:
        loop.close()

    assert len(_FakeGremlinConnection.threads) == 1
    assert _FakeGremlinConnection.threads[0] is not main


async def test_aclose_closes_the_connection_off_the_loop(retriever) -> None:
    await retriever.aretrieve(SearchQuery(query="Vendor"))
    await retriever.aclose()
    assert len(_FakeGremlinConnection.close_threads) == 1
    assert _FakeGremlinConnection.close_threads[0] is not threading.current_thread()
