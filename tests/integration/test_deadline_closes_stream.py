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
must see every connection closed, the query fail with the timeout its
limit's timer raised, the loop log no error, and the next query answer.

What makes the timeout late is pinned by events, not by how long the query
took (a loaded runner delays the timer and every step after it): no chunk
in flight when the timer fired is waited out, and a wait on the stalled
read, the read timeout or the stall trips a backstop far below them.
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
# Seconds after its limit started within which every query's timeout is
# raised: only a backstop for a cut that waits for the stalled read, far
# below what else would end that wait (the model clients' read timeout,
# 30 s under the deadline; the stall). A loaded runner delays the timer and
# the unwinding by a second or two; the cut's promptness is pinned by
# events instead (_carried_slowly).
HANG = 10.0
# Queries cut off at once.
QUERIES = 6
# The deadline of the queries before and after the cut-off ones: only a
# backstop (as long as their own, _answers), as a loaded runner slows a
# first call's imports and client setup.
WARM = 60.0
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


def _carried_slowly(monkeypatch: pytest.MonkeyPatch, seconds: float) -> list[float]:
    """Each chunk takes ``seconds`` to reach the stream's reader after the
    model's read returned it: a cut lands there, the read done.

    Returns the carries that ended after their query's limit fired (its
    timer expired) during them: a cut that waited out the chunk in flight
    instead of cancelling it. The timer cancels the query's task, so a carry
    it lands in raises, however late a loaded loop runs either.

    A chunk that reaches the carry after the timer fired is one whose cut a
    layer under it lost (on 3.11 langchain-openai's chunk read through
    ``asyncio.wait_for``): its carry runs to its end, and the chunk's stop
    point ends the query. Such a carry is listed too if the stream asks for
    another chunk after it."""
    carry = chain_factory.ReasoningStripper.atransform
    waited_out: list[float] = []

    async def slow(self: Any, input: Any, config: Any = None, **kwargs: Any) -> Any:
        scope = rag_chain._DEADLINE_SCOPE.get()
        assert scope is not None and scope._timeout is not None
        timer = scope._timeout
        async for chunk in carry(self, input, config, **kwargs):
            lost = timer.expired()
            await asyncio.sleep(seconds)
            if timer.expired() and not lost:
                waited_out.append(seconds)
            yield chunk
            if lost:
                waited_out.append(seconds)

    monkeypatch.setattr(chain_factory.ReasoningStripper, "atransform", slow)
    return waited_out


def _timed_limits(
    monkeypatch: pytest.MonkeyPatch, window: bool
) -> tuple[list[float], list[QueryTimeoutError]]:
    """The seconds each query's limit took to raise its ``QueryTimeoutError``
    (its calls unwound and their responses closed), counted from when the
    limit started: the deadline at the query's start, the window at the
    answer's call, after the query's other steps (which a loaded runner can
    delay); and the errors the deadline scope made, once its timer expired
    (``_DeadlineScope._expired``)."""
    took: list[float] = []
    made: list[QueryTimeoutError] = []
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
        made.append(expired(scope))
        return made[-1]

    monkeypatch.setattr(rag_chain._DeadlineScope, "first_chunk_within", timed_within)
    monkeypatch.setattr(rag_chain._DeadlineScope, "_expired", timed_expired)
    return took, made


async def _cut_off(chain: Any, made: list[QueryTimeoutError]) -> None:
    """Read one query's events: it fails with the timeout its limit's timer
    raised (one the deadline scope made, of that limit's seconds)."""
    with pytest.raises(QueryTimeoutError) as raised:
        async for _ in chain.aquery_events(_input()):
            pass
    assert any(raised.value is error for error in made)
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
    chain.config.search.query_timeout_seconds = WARM
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
            waited_out = (
                _carried_slowly(carried, SLOW[limit])
                if where == "between chunks"
                else []
            )
            took, made = _timed_limits(carried, window)
            async with _generators_kept():
                await asyncio.gather(*(_cut_off(chain, made) for _ in range(QUERIES)))
                assert len(took) == QUERIES
                assert LIMIT <= min(took) and max(took) < HANG
                assert waited_out == []
                await _released(port, models)
        models.knobs = stream_models.StreamKnobs()
        chain.config.search.query_timeout_seconds = WARM
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
    chain.config.search.query_timeout_seconds = WARM
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
        chain.config.search.query_timeout_seconds = WARM
        assert (await chain.ainvoke(_input())).answer
    finally:
        models.knobs.unstall.set()
        models.knobs = stream_models.StreamKnobs()
        await chain.aclose()
    assert errors == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]


async def test_a_query_cut_off_before_its_answer_closes_its_model_response(
    wire: tuple[int, stream_models.StreamModels],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A call before the answer (the translation of a question not in
    English, streamed by the Ollama client under ``ainvoke``) cut off
    between two chunks by the deadline: its response is closed too, by the
    query, though its generators are left suspended."""
    from langchain_ollama import chat_models

    port, models = wire
    chain = _wire_chain("ollama", port, monkeypatch, model_timeout=30)
    question = _input(
        query="Vendor는 Buyer에게 무엇을 공급하나요?", enable_query_processing=True
    )
    chain.config.search.query_timeout_seconds = WARM
    await chain.ainvoke(question)
    chain.config.search.query_timeout_seconds = LIMIT
    models.knobs = stream_models.StreamKnobs(
        chunk_delay=0.01, chunks=40, stall_after=20, stall_seconds=STALL
    )
    errors = _loop_errors()
    answers = 0

    async def cut_off() -> None:
        with pytest.raises(QueryTimeoutError):
            await chain.ainvoke(question)

    original = chain._answer_generation_step

    def counted(state: Any) -> Any:
        nonlocal answers
        answers += 1
        return original(state)

    try:
        with pytest.MonkeyPatch.context() as carried:
            carried.setattr(chain, "_answer_generation_step", counted)
            read = chat_models.ChatOllama._acreate_chat_stream

            async def slow(self: Any, *args: Any, **kwargs: Any) -> Any:
                async for chunk in read(self, *args, **kwargs):
                    await asyncio.sleep(SLOW["deadline"])
                    yield chunk

            carried.setattr(chat_models.ChatOllama, "_acreate_chat_stream", slow)
            async with _generators_kept():
                await asyncio.gather(*(cut_off() for _ in range(QUERIES)))
                # Cut off before the answer step.
                assert answers == 0
                await _released(port, models)
        models.knobs = stream_models.StreamKnobs()
        chain.config.search.query_timeout_seconds = WARM
        assert (await chain.ainvoke(question)).answer
    finally:
        models.knobs.unstall.set()
        models.knobs = stream_models.StreamKnobs()
        await chain.aclose()
    assert errors == []
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
