# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A reader that stops mid-answer releases the answer model's connection.

The answer streams from a model server over real loopback HTTP (the console
demo's fake Ollama, through the Ollama adapter and its httpx pool), so a
model response left open holds a connection of the pool. Each way a caller
stops reading ``aquery_events`` (its task cancelled, as the console's is
when a client disconnects; an ``asyncio.timeout`` around ``astream``; the
console's own ``/api/ask`` events) must leave the pool empty, log no event
loop error and leave the next query answering.
"""

from __future__ import annotations

import asyncio
import gc
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.console_demo import load, requires_console
from tests.unit.test_rag_chain_streaming import _input, _make_chain
from unified_kg_rag import AnswerChunk, GraphRAGChain
from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
from unified_kg_rag.domain.models import Config, SearchStrategy

pytestmark = [pytest.mark.integration]

# Queries stopped at once: enough that cancellations land at each point of
# the answer's stream (between its chunks and inside the model's read).
STOPPED = 24
# Seconds between the fake model's streamed chunks.
CHUNK_DELAY = 0.05


@pytest.fixture(scope="module")
def models() -> Iterator[int]:
    server, _ = load("fake_models").serve(chunk_delay=CHUNK_DELAY)
    yield server.server_address[1]
    server.shutdown()


@pytest.fixture
def chain(models: int) -> GraphRAGChain:
    config = Config()
    config.providers.llm.kind = "ollama"
    config.providers.llm.base_url = f"http://127.0.0.1:{models}"
    config.providers.rerank.kind = "none"
    chain, _, _ = _make_chain(
        config,
        factory=OllamaLanguageModelFactory(config),  # type: ignore[arg-type]
    )
    return chain


def _held_connections(port: int) -> int:
    """Connections the model clients of this process hold to ``port``: the
    clients keep no idle one (no keep-alive), so each is a response left
    open."""
    import httpx

    held = 0
    for client in [o for o in gc.get_objects() if isinstance(o, httpx.AsyncClient)]:
        pool = getattr(client._transport, "_pool", None)
        if pool is not None and client.base_url.port == port:
            held += len(pool.connections)
    return held


async def _settled(port: int) -> int:
    """The connections still held to ``port`` after up to 2 s (a closed
    response leaves the pool a moment later)."""
    for _ in range(40):
        if not (held := _held_connections(port)):
            return 0
        await asyncio.sleep(0.05)
    return held


def _loop_errors() -> list[str]:
    errors: list[str] = []
    asyncio.get_running_loop().set_exception_handler(
        lambda _loop, context: errors.append(
            str(context.get("exception") or context.get("message"))
        )
    )
    return errors


async def _stopped_mid_answer(stop: Any) -> None:
    """Start ``STOPPED`` readers, each stopped by ``stop(i, task, chunked)``
    once its first answer chunk arrived."""

    async def one(i: int) -> None:
        chunked = asyncio.Event()
        task = asyncio.create_task(stop.read(chunked))
        await chunked.wait()
        # Spread over one chunk's interval: some land while the answer
        # model waits for its next chunk.
        await asyncio.sleep(CHUNK_DELAY * (i % 6) / 6)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    await asyncio.gather(*(one(i) for i in range(STOPPED)))


class _CancelledReader:
    """A task reading ``aquery_events``, cancelled (a client that
    disconnected)."""

    def __init__(self, chain: GraphRAGChain) -> None:
        self.chain = chain

    async def read(self, chunked: asyncio.Event) -> None:
        async for event in self.chain.aquery_events(_input()):
            if isinstance(event, AnswerChunk):
                chunked.set()


class _AskReader:
    """The console's ``/api/ask`` events, their response task cancelled as
    Starlette cancels it when the client disconnects."""

    def __init__(self, service: Any, request: Any) -> None:
        self.service, self.request = service, request

    async def read(self, chunked: asyncio.Event) -> None:
        async for event in self.service.ask(self.request):
            if event["kind"] == "chunk":
                chunked.set()


async def _answers(chain: GraphRAGChain) -> None:
    output = await asyncio.wait_for(chain.ainvoke(_input()), 15)
    assert output.answer and not output.metadata.get("error")


async def test_a_cancelled_reader_releases_the_model_connection(
    chain: GraphRAGChain, models: int
) -> None:
    errors = _loop_errors()
    await _stopped_mid_answer(_CancelledReader(chain))
    assert await _settled(models) == 0
    await _answers(chain)
    gc.collect()
    await asyncio.sleep(0.1)
    assert errors == []
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "kg-rag-query-events"]


def _stop_at(i: int) -> float:
    """Seconds after the first chunk to stop the ``i``-th stream at:
    points spread over a chunk's interval."""
    return CHUNK_DELAY * (1 + i % 12 / 12)


async def _asyncio_timeout(chain: GraphRAGChain, i: int) -> None:
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(None) as timeout:
            async for _ in chain.astream(_input()):
                if timeout.when() is None:
                    timeout.reschedule(asyncio.get_running_loop().time() + _stop_at(i))


async def _anyio_scope(chain: GraphRAGChain, i: int) -> None:
    # anyio's scopes (Starlette's, a FastAPI handler's) cancel the task at
    # every await until it leaves the scope.
    import anyio

    with anyio.CancelScope() as scope:
        async for _ in chain.astream(_input()):
            if scope.deadline == float("inf"):
                scope.deadline = anyio.current_time() + _stop_at(i)
    assert scope.cancelled_caught


@pytest.mark.parametrize("stop", [_asyncio_timeout, _anyio_scope])
async def test_a_timeout_around_astream_releases_the_model_connection(
    chain: GraphRAGChain, models: int, stop: Any
) -> None:
    errors = _loop_errors()
    await asyncio.gather(*(stop(chain, i) for i in range(STOPPED)))
    assert await _settled(models) == 0
    await _answers(chain)
    gc.collect()
    await asyncio.sleep(0.1)
    assert errors == []


def test_a_sync_stream_closed_mid_answer_releases_the_model_connection(
    chain: GraphRAGChain, models: int
) -> None:
    for _ in range(4):
        stream = iter(chain.stream(_input()))
        next(stream)
        stream.close()  # type: ignore[attr-defined]
    assert asyncio.run(_settled(models)) == 0
    assert chain.invoke(_input()).answer


@requires_console
async def test_console_disconnects_leave_the_console_answering(
    chain: GraphRAGChain, models: int, tmp_path: Path
) -> None:
    from unified_kg_rag.application.cli.console.ask import AskRequest, AskService
    from unified_kg_rag.application.cli.console.store import ConsoleStore

    errors = _loop_errors()
    service = AskService(
        lambda: chain.config,
        ConsoleStore(tmp_path / "console.sqlite3"),
        chain_factory=lambda _config: chain,
    )
    request = AskRequest(query=_input().query, search_strategy=SearchStrategy.SIMPLE)
    try:
        # More disconnects than the pool's 100 connections: a leak fills it.
        for _ in range(-(-256 // STOPPED)):
            await _stopped_mid_answer(_AskReader(service, request))
        assert await _settled(models) == 0
        async with asyncio.timeout(15):
            events = [e async for e in service.ask(request)]
        assert events[-1]["kind"] == "result"
        # Each disconnect is recorded as cancelled, not failed.
        outcomes = {row["outcome"] for row in service.store.queries(limit=500)}
        assert outcomes <= {"cancelled", "answered"}
    finally:
        await service.close()
    gc.collect()
    await asyncio.sleep(0.1)
    assert errors == []
