# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A query cut off at its deadline, or at its answer stream's first-chunk
window, closes the answer model's response, wherever the cut lands.

The answer streams from the loopback model server (each provider's wire:
Ollama, an OpenAI-compatible server, Bedrock) over real HTTP. The cut is a
cancellation: landing in the model client's read, the client's own
generators close the response; landing while the chunk just read is on its
way up (a loaded event loop delays it), it leaves the stream's generators
suspended between chunks, which close nothing until garbage collection
finalizes them. The test keeps every async generator alive, as a reference
cycle does, so only the query itself can close the response: the server
must see every connection closed, the query fail with its timeout at the
limit, the loop log no error, and the next query answer.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

from tests.fixtures import stream_models
from tests.integration.test_query_events_disconnect import (
    _answers,
    _held_connections,
    _loop_errors,
    _wire_chain,
)
from tests.unit.test_rag_chain_streaming import _input
from unified_kg_rag.adapters.llm import chain_factory
from unified_kg_rag.application.retrieval import rag_chain
from unified_kg_rag.shared.exceptions import QueryTimeoutError

pytestmark = [pytest.mark.integration]

PROVIDERS = ["ollama", "openai", "bedrock"]
# The limit that cuts each query off (the deadline, or the window).
LIMIT = 1.0
# How far past the limit the timeout may be raised: the cut's unwinding and
# the closing of the responses, on a loaded runner.
LATE = 1.0
# Queries cut off at once.
QUERIES = 6
# Seconds a stalled answer waits: far past every bound here, so a response
# still open is one the server is still writing.
STALL = 60.0
# Seconds the stream takes to carry each chunk up past the model's read,
# under each limit: the cut lands in one.
SLOW = {"deadline": 0.4, "first-chunk window": 2 * LIMIT}


@pytest.fixture(scope="module")
def wire() -> Iterator[tuple[int, stream_models.StreamModels]]:
    server, models = stream_models.serve()
    yield server.server_address[1], models
    server.shutdown()


@contextlib.asynccontextmanager
async def _generators_kept() -> AsyncIterator[None]:
    """Keep every async generator started in the block alive (as a
    reference cycle keeps them, until a collection that may not come), so
    none closes a response by being finalized; close them afterwards."""
    firstiter, finalizer = sys.get_asyncgen_hooks()
    kept: list[Any] = []

    def keep(agen: Any) -> None:
        kept.append(agen)
        if firstiter is not None:
            firstiter(agen)

    sys.set_asyncgen_hooks(firstiter=keep, finalizer=finalizer)
    try:
        yield
    finally:
        sys.set_asyncgen_hooks(firstiter=firstiter, finalizer=finalizer)
        for agen in reversed(kept):
            # One abandoned while it was being read cannot be closed: it is
            # dropped, as a collection drops it.
            with contextlib.suppress(BaseException):
                await agen.aclose()


def _carried_slowly(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    """Each chunk takes ``seconds`` to reach the stream's reader after the
    model's read returned it: a cut lands there, the read done."""
    carry = chain_factory.ReasoningStripper.atransform

    async def slow(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
        async for chunk in carry(self, input, config, **kwargs):
            await asyncio.sleep(seconds)
            yield chunk

    monkeypatch.setattr(chain_factory.ReasoningStripper, "atransform", slow)


def _timed_limits(monkeypatch: pytest.MonkeyPatch, window: bool) -> list[float]:
    """The seconds each query's limit took to raise its ``QueryTimeoutError``
    (its calls unwound and their responses closed), counted from when the
    limit started: the deadline at the query's start, the window at the
    answer's call, after the query's other steps (which a loaded runner can
    delay)."""
    took: list[float] = []
    windows: dict[Any, float] = {}
    within = rag_chain._DeadlineScope.first_chunk_within
    expired = rag_chain._DeadlineScope._expired

    @contextlib.contextmanager
    def timed_within(scope: Any, seconds: float) -> Iterator[None]:
        windows.setdefault(scope, time.monotonic())
        with within(scope, seconds):
            yield

    def timed_expired(scope: Any) -> QueryTimeoutError:
        started = windows[scope] if window else scope._deadline.started
        took.append(time.monotonic() - started)
        return expired(scope)

    monkeypatch.setattr(rag_chain._DeadlineScope, "first_chunk_within", timed_within)
    monkeypatch.setattr(rag_chain._DeadlineScope, "_expired", timed_expired)
    return took


async def _cut_off(chain: Any) -> None:
    """Read one query's events: it fails with the timeout of its limit."""
    with pytest.raises(QueryTimeoutError) as raised:
        async for _ in chain.aquery_events(_input()):
            pass
    assert raised.value.timeout_seconds == LIMIT


async def _released(port: int, models: stream_models.StreamModels) -> None:
    # The server sees each connection closed (every provider) ...
    assert await asyncio.to_thread(models.settled, 3.0) == 0
    # ... and the local providers' pools hold none.
    for _ in range(40):
        if not _held_connections(port):
            break
        await asyncio.sleep(0.05)
    assert _held_connections(port) == 0


@pytest.mark.parametrize("where", ["in the read", "between chunks"])
@pytest.mark.parametrize("limit", ["deadline", "first-chunk window"])
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_a_query_cut_off_closes_its_answer_stream(
    wire: tuple[int, stream_models.StreamModels],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    provider: str,
    limit: str,
    where: str,
) -> None:
    port, models = wire
    window = limit == "first-chunk window"
    # The window is search.model_timeout_seconds, the model clients' read
    # timeout too; past the deadline it does not apply.
    chain = _wire_chain(
        provider, port, monkeypatch, model_timeout=LIMIT if window else 30
    )
    # Warm: a first call's imports and client setup are not the cut-off.
    chain.config.search.query_timeout_seconds = 30
    await _answers(chain)
    chain.config.search.query_timeout_seconds = 30 if window else LIMIT
    if where == "between chunks":
        # The first chunks come at once, then the answer stalls: the
        # server writes until the client closes the connection.
        models.knobs = stream_models.StreamKnobs(
            chunk_delay=0.01, chunks=40, stall_after=20, stall_seconds=STALL
        )
    elif window:
        models.knobs = stream_models.StreamKnobs(first_seconds=STALL)
    else:
        models.knobs = stream_models.StreamKnobs(stall_after=2, stall_seconds=STALL)
    errors = _loop_errors()
    try:
        with pytest.MonkeyPatch.context() as carried:
            if where == "between chunks":
                _carried_slowly(carried, SLOW[limit])
            took = _timed_limits(carried, window)
            async with _generators_kept():
                await asyncio.gather(*(_cut_off(chain) for _ in range(QUERIES)))
                assert len(took) == QUERIES
                assert LIMIT <= min(took) and max(took) < LIMIT + LATE
                await _released(port, models)
        models.knobs = stream_models.StreamKnobs()
        chain.config.search.query_timeout_seconds = 30
        await _answers(chain)
    finally:
        models.knobs.unstall.set()
        models.knobs = stream_models.StreamKnobs()
        await chain.aclose()
    assert errors == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_an_unstreamed_answer_cut_off_between_chunks_closes_its_response(
    wire: tuple[int, stream_models.StreamModels],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``ainvoke`` (no events read): the Ollama chat client streams under
    it too, so a deadline that lands between two chunks leaves its
    generators suspended; the answer call closes its response all the
    same."""
    from langchain_ollama import chat_models

    port, models = wire
    chain = _wire_chain("ollama", port, monkeypatch, model_timeout=30)
    chain.config.search.query_timeout_seconds = 30
    await chain.ainvoke(_input())
    chain.config.search.query_timeout_seconds = LIMIT
    models.knobs = stream_models.StreamKnobs(
        chunk_delay=0.01, chunks=40, stall_after=20, stall_seconds=STALL
    )
    errors = _loop_errors()

    async def cut_off() -> None:
        with pytest.raises(QueryTimeoutError):
            await chain.ainvoke(_input())

    try:
        with pytest.MonkeyPatch.context() as carried:
            # Each chunk takes a while past the client's read to reach
            # LangChain's stream: the cut lands there.
            read = chat_models.ChatOllama._acreate_chat_stream

            async def slow(self: Any, *args: Any, **kwargs: Any) -> Any:
                async for chunk in read(self, *args, **kwargs):
                    await asyncio.sleep(SLOW["deadline"])
                    yield chunk

            carried.setattr(chat_models.ChatOllama, "_acreate_chat_stream", slow)
            async with _generators_kept():
                await asyncio.gather(*(cut_off() for _ in range(QUERIES)))
                await _released(port, models)
        models.knobs = stream_models.StreamKnobs()
        chain.config.search.query_timeout_seconds = 30
        assert (await chain.ainvoke(_input())).answer
    finally:
        models.knobs.unstall.set()
        models.knobs = stream_models.StreamKnobs()
        await chain.aclose()
    assert errors == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
