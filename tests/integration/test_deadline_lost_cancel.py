# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A query whose deadline's cancellation a layer under it loses still ends
at its deadline, at its next stop point.

The deadline (or the answer stream's first-chunk window) cancels the
query's task once. Code under the query can lose that cancellation: anyio's
``connect_tcp`` (httpcore's connect, both local providers, any Python)
returns a connected stream to a task cancelled while it connected, and on
3.11 ``asyncio.wait_for`` (langchain-openai's chunk read) returns the
result its awaitable got in the same turn as the cancellation. The query
then went on until the model client's read timeout. Each test forces the
loss at one place, over real loopback HTTP, and cuts the query there on cue
(its limit's timer expired through its scope, once the connect or the first
chunk is held), never by the wall clock: a slow runner's query does not
reach that place before its limit. The query must fail with the
timeout its deadline scope made, at the next stop point (no request written
after a lost connect, no chunk read after a lost chunk read), leave no
connection held and no loop error, and the next query must answer.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterator
from typing import Any

import pytest

from tests.fixtures import stream_models
from tests.integration.test_deadline_closes_stream import (
    WARM,
    _generators_kept,
    _released,
)
from tests.integration.test_query_events_disconnect import (
    _answers,
    _loop_errors,
    _wire_chain,
)
from tests.unit.test_rag_chain_streaming import _input
from unified_kg_rag.adapters.llm import chain_factory
from unified_kg_rag.application.retrieval import rag_chain
from unified_kg_rag.shared.exceptions import QueryTimeoutError

pytestmark = [pytest.mark.integration]

# The limit each query is cut off at (the deadline, or the window): expired
# on cue, so only a bound past the hang backstop, which it never reaches on
# its own; under the deadline, below the model clients' read timeout (30 s).
LIMIT = 25.0
# A hang backstop far below what else ends a query whose cut was lost: the
# model clients' read timeout (30 s under the deadline) and the stall.
HANG = 15.0
# Seconds a stalled answer waits: far past every bound here.
STALL = 60.0


@pytest.fixture(scope="module")
def wire() -> Iterator[tuple[int, stream_models.StreamModels]]:
    server, models = stream_models.serve()
    yield server.server_address[1], models
    server.shutdown()


def _cue(window: bool) -> asyncio.Timeout | None:
    """Expire the timer of the query running here through its deadline
    scope (which sets the query's stop with it, as when it fires), and
    return it: the cut, cancelling the task next. ``None`` outside a scope
    still running, and under the window (``window``) outside the first
    chunk's wait (``first_chunk_within``)."""
    scope = rag_chain._DEADLINE_SCOPE.get()
    timer = None if scope is None else scope._timeout
    if scope is None or timer is None or timer.expired():
        return None
    if window and scope._first_chunk_seconds is None:
        return None
    scope._reschedule(asyncio.get_running_loop().time())
    return timer


async def _losing(step: Any) -> Any:
    """Run ``step()`` once the task is cancelled (the cut, :func:`_cue`), losing
    the cancellation: the step's result is returned as if nothing was
    cancelled (anyio's connect, 3.11's ``wait_for``)."""
    cut = asyncio.Event()

    async def stepped() -> Any:
        await cut.wait()
        return await step()

    inner = asyncio.ensure_future(stepped())
    try:
        return await asyncio.shield(inner)
    except asyncio.CancelledError:
        cut.set()
        return await inner


def _connect_loses_the_cut(monkeypatch: pytest.MonkeyPatch, window: bool) -> list[bool]:
    """httpcore's first connect under the query's limit cuts the query
    (:func:`_cue`), waits for the cancellation, then connects, losing it.
    Returns one entry per connect that lost it."""
    from httpcore._backends.anyio import AnyIOBackend

    connect = AnyIOBackend.connect_tcp
    lost: list[bool] = []

    async def losing(backend: Any, *args: Any, **kwargs: Any) -> Any:
        timer = None if lost else _cue(window)
        if timer is None:
            return await connect(backend, *args, **kwargs)
        stream = await _losing(lambda: connect(backend, *args, **kwargs))
        lost.append(timer.expired())
        return stream

    monkeypatch.setattr(AnyIOBackend, "connect_tcp", losing)
    return lost


def _read_loses_the_cut(
    monkeypatch: pytest.MonkeyPatch, window: bool
) -> tuple[list[bool], list[bool]]:
    """The answer stream's first chunk, held, cuts the query (:func:`_cue`,
    within the window under it), then reaches the stream's reader, the
    cancellation lost on its way. Returns the reads that lost it, and the
    chunks asked for after that one."""
    carry = chain_factory.ReasoningStripper.atransform
    lost: list[bool] = []
    asked: list[bool] = []

    async def losing(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
        async for chunk in carry(self, input, config, **kwargs):
            if lost:
                asked.append(True)
            elif (timer := _cue(window)) is not None:
                await _losing(lambda: asyncio.sleep(0))
                lost.append(timer.expired())
            yield chunk

    monkeypatch.setattr(chain_factory.ReasoningStripper, "atransform", losing)
    return lost, asked


def _made(monkeypatch: pytest.MonkeyPatch) -> list[QueryTimeoutError]:
    """The timeouts the deadline scopes made (``_DeadlineScope._expired``)."""
    made: list[QueryTimeoutError] = []
    expired = rag_chain._DeadlineScope._expired

    def recorded(scope: Any) -> QueryTimeoutError:
        made.append(expired(scope))
        return made[-1]

    monkeypatch.setattr(rag_chain._DeadlineScope, "_expired", recorded)
    return made


async def _cut_off(chain: Any, read: str, made: list[QueryTimeoutError]) -> None:
    async def run() -> None:
        if read == "events":
            async for _ in chain.aquery_events(_input()):
                pass
        else:
            await chain.ainvoke(_input())

    with pytest.raises(QueryTimeoutError) as raised:
        await asyncio.wait_for(run(), HANG)
    assert any(raised.value is error for error in made)
    assert raised.value.timeout_seconds == LIMIT


def _limited(
    provider: str, port: int, monkeypatch: pytest.MonkeyPatch, limit: str
) -> Any:
    window = limit == "first-chunk window"
    # The window is search.model_timeout_seconds, the model clients' read
    # timeout too; past the deadline it does not apply.
    return _wire_chain(
        provider, port, monkeypatch, model_timeout=LIMIT if window else 30
    )


async def _warm(chain: Any) -> None:
    chain.config.search.query_timeout_seconds = WARM
    await _answers(chain)


def _cut_at(chain: Any, limit: str) -> None:
    # Under the window the deadline is a backstop past it, so the window
    # applies however long the steps before the answer take.
    chain.config.search.query_timeout_seconds = (
        WARM if limit == "first-chunk window" else LIMIT
    )


async def _answers_again(chain: Any, models: stream_models.StreamModels) -> None:
    models.knobs = stream_models.StreamKnobs()
    chain.config.search.query_timeout_seconds = WARM
    await _answers(chain)


# An answer read as events streams under its first-chunk window; ainvoke's
# does not stream (no window), and is bounded by the deadline alone.
@pytest.mark.parametrize(
    ("limit", "read"),
    [("deadline", "events"), ("deadline", "ainvoke"), ("first-chunk window", "events")],
)
@pytest.mark.parametrize("provider", ["ollama", "openai"])
async def test_a_cut_the_connect_loses_ends_the_query_before_its_request(
    wire: tuple[int, stream_models.StreamModels],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    provider: str,
    limit: str,
    read: str,
) -> None:
    port, models = wire
    chain = _limited(provider, port, monkeypatch, limit)
    await _warm(chain)
    _cut_at(chain, limit)
    models.knobs = stream_models.StreamKnobs(first_seconds=STALL)
    errors = _loop_errors()
    try:
        with pytest.MonkeyPatch.context() as cut:
            lost = _connect_loses_the_cut(cut, limit == "first-chunk window")
            made = _made(cut)
            received = models.received
            async with _generators_kept():
                await _cut_off(chain, read, made)
                # The connect lost the cut, and the request it connected
                # for was not written: the next stop point ended the query.
                assert lost == [True]
                assert models.received == received
                await _released(port, models)
        await _answers_again(chain, models)
    finally:
        models.knobs.unstall.set()
        models.knobs = stream_models.StreamKnobs()
        await chain.aclose()
    assert errors == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


@pytest.mark.parametrize("limit", ["deadline", "first-chunk window"])
@pytest.mark.parametrize("provider", ["ollama", "openai", "bedrock"])
async def test_a_cut_a_chunk_read_loses_ends_the_query_at_that_chunk(
    wire: tuple[int, stream_models.StreamModels],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    provider: str,
    limit: str,
) -> None:
    port, models = wire
    chain = _limited(provider, port, monkeypatch, limit)
    await _warm(chain)
    _cut_at(chain, limit)
    # The first chunks come at once, then the answer stalls: a query that
    # reads on waits for the stall (or the read timeout).
    models.knobs = stream_models.StreamKnobs(
        chunk_delay=0.01, chunks=40, stall_after=2, stall_seconds=STALL
    )
    errors = _loop_errors()
    try:
        with pytest.MonkeyPatch.context() as cut:
            lost, asked = _read_loses_the_cut(cut, limit == "first-chunk window")
            made = _made(cut)
            async with _generators_kept():
                await _cut_off(chain, "events", made)
                # The read lost the cut, and no chunk after it was asked
                # for: the chunk's stop point ended the query.
                assert lost == [True]
                assert asked == []
                await _released(port, models)
        await _answers_again(chain, models)
    finally:
        models.knobs.unstall.set()
        models.knobs = stream_models.StreamKnobs()
        await chain.aclose()
    assert errors == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
