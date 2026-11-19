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
import threading
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
from botocore.httpsession import URLLib3Session

from tests.fixtures import stream_models
from tests.fixtures.console_demo import load, requires_console
from tests.unit.test_rag_chain_streaming import _input, _make_chain
from unified_kg_rag import AnswerChunk, GraphRAGChain, QuerySources
from unified_kg_rag.adapters.local.models import (
    OllamaLanguageModelFactory,
    OpenAICompatibleLanguageModelFactory,
)
from unified_kg_rag.application.retrieval import rag_chain
from unified_kg_rag.domain.models import Config, SearchStrategy

pytestmark = [pytest.mark.integration]

# Queries stopped at once: enough that cancellations land at each point of
# the answer's stream (between its chunks and inside the model's read).
STOPPED = 24
# Seconds between the fake model's streamed chunks.
CHUNK_DELAY = 0.05
# botocore's own send, before the suite blocks it (the Bedrock loopback
# below lets only requests to 127.0.0.1 through).
_REAL_SEND = URLLib3Session.send


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


# A hang backstop (seconds), far past anything a passing run waits: these
# tests assert on what happened (connections, writes, the stop's order),
# not on how long a step took.
BACKSTOP = 60.0


async def _settled(port: int) -> int:
    """The connections still held to ``port`` once none is (a closed
    response leaves the pool a moment later), or after the backstop."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + BACKSTOP
    # Polled: a pool has no event for a connection leaving it.
    while True:
        held = _held_connections(port)
        if not held or loop.time() >= deadline:
            return held
        await asyncio.sleep(0.05)


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
    output = await asyncio.wait_for(chain.ainvoke(_input()), BACKSTOP)
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
        async with asyncio.timeout(BACKSTOP):
            events = [e async for e in service.ask(request)]
        assert events[-1]["kind"] == "result"
        # Each disconnect is recorded as cancelled, not failed.
        rows = service.store.queries(limit=500)
        assert {row["outcome"] for row in rows} <= {"cancelled", "answered"}
        # A disconnected question records the answer call it spent (rv53).
        cancelled = [row for row in rows if row["outcome"] == "cancelled"]
        assert cancelled
        assert all((row["usage"] or {}).get("llm_calls") for row in cancelled)
    finally:
        await service.close()
    gc.collect()
    await asyncio.sleep(0.1)
    assert errors == []


# --- each provider's wire: reasoning and stalls ---------------------------


@pytest.fixture(scope="module")
def wire() -> Iterator[tuple[int, stream_models.StreamModels]]:
    server, models = stream_models.serve()
    yield server.server_address[1], models
    server.shutdown()


def _wire_chain(
    provider: str,
    port: int,
    monkeypatch: pytest.MonkeyPatch,
    model_timeout: float,
    backoff: float | None = None,
) -> GraphRAGChain:
    """A chain whose answer model is ``provider``'s client, talking to the
    loopback server on ``port``; its model retries back off ``backoff`` s
    (with no budget) when given."""
    from unified_kg_rag.adapters.aws.bedrock import BedrockLanguageModelFactory

    config = Config()
    config.providers.rerank.kind = "none"
    config.search.model_timeout_seconds = model_timeout
    if backoff is not None:
        for policy in (config.retry.model, config.retry.model_server):
            policy.base_delay_seconds = policy.max_delay_seconds = backoff
            policy.jitter = "none"
            policy.max_total_seconds = None
    factory: Any
    if provider == "ollama":
        config.providers.llm.kind = "ollama"
        config.providers.llm.base_url = f"http://127.0.0.1:{port}"
        factory = OllamaLanguageModelFactory(config)
    elif provider == "openai":
        config.providers.llm.kind = "openai_compatible"
        config.providers.llm.base_url = f"http://127.0.0.1:{port}/v1"
        factory = OpenAICompatibleLanguageModelFactory(config)
    else:

        def local_only_send(self: URLLib3Session, request: Any) -> Any:
            assert request.url.startswith(f"http://127.0.0.1:{port}/")
            return _REAL_SEND(self, request)

        monkeypatch.setattr(URLLib3Session, "send", local_only_send)
        monkeypatch.setenv(
            "AWS_ENDPOINT_URL_BEDROCK_RUNTIME", f"http://127.0.0.1:{port}"
        )
        factory = BedrockLanguageModelFactory(config)
    chain, _, _ = _make_chain(config, factory=factory)
    return chain


async def _cancelled_at(
    chain: GraphRAGChain, event_type: type, delay: float
) -> float | None:
    """Read the query's events in a task, cancelled ``delay`` s after its
    first ``event_type`` event; the seconds the cancelled reader took to
    end (``None``: the query ended before that event, as one whose first
    chunk missed its window on a loaded runner does)."""
    seen = asyncio.Event()

    async def read() -> None:
        async for event in chain.aquery_events(_input()):
            if isinstance(event, event_type):
                seen.set()

    task = asyncio.create_task(read())
    waiting = asyncio.create_task(seen.wait())
    await asyncio.wait({task, waiting}, return_when=asyncio.FIRST_COMPLETED)
    if not seen.is_set():
        waiting.cancel()
        await asyncio.gather(task, waiting, return_exceptions=True)
        return None
    await asyncio.sleep(delay)
    loop = asyncio.get_running_loop()
    stopped = loop.time()
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return loop.time() - stopped


@pytest.mark.parametrize("provider", ["ollama", "openai", "bedrock"])
async def test_a_reader_cancelled_while_the_model_reasons_stops_the_model(
    wire: tuple[int, stream_models.StreamModels],
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    # Reasoning streams chunks without answer text: a reader cancelled
    # after its sources cancels its query at once, so it does not hold the
    # model while it reasons (40 s, past the backstop).
    port, models = wire
    models.knobs = stream_models.StreamKnobs(reasoning_seconds=40)
    chain = _wire_chain(provider, port, monkeypatch, model_timeout=60)
    errors = _loop_errors()
    try:
        async with asyncio.timeout(BACKSTOP / 2):
            released = await asyncio.gather(
                *(_cancelled_at(chain, QuerySources, 0.3) for _ in range(8))
            )
        assert None not in released
        # The model stopped writing: its client closed each response.
        assert await asyncio.to_thread(models.settled) == 0
        if provider != "bedrock":
            assert await _settled(port) == 0
        models.knobs = stream_models.StreamKnobs()
        await _answers(chain)
    finally:
        models.knobs = stream_models.StreamKnobs()
        await chain.aclose()
    assert errors == []


@pytest.mark.parametrize("provider", ["ollama", "openai"])
async def test_a_model_that_stalls_after_a_disconnect_is_cut_off_at_once(
    wire: tuple[int, stream_models.StreamModels],
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    # The answer stalls right after the chunk the reader was cancelled
    # past: the cancel lands in the stalled read (or while the query waits
    # for its reader) and closes the response at once, long before the
    # model client's read timeout (past the backstop), and leaves no
    # connection held.
    port, models = wire
    stalled = stream_models.StreamKnobs(stall_after=1, stall_seconds=3600)
    models.knobs = stalled
    chain = _wire_chain(provider, port, monkeypatch, model_timeout=600)
    # No first-chunk window, which a loaded runner can miss before the
    # stall: it cancels the query wherever it is, not this test's subject.
    monkeypatch.setattr(rag_chain, "_first_chunk_window", lambda _s: nullcontext())
    errors = _loop_errors()
    try:
        # Cancelled at points spread over a chunk's interval: some while the
        # query waits for the reader, most in the stalled read.
        async with asyncio.timeout(BACKSTOP):
            released = await asyncio.gather(
                *(
                    _cancelled_at(chain, AnswerChunk, CHUNK_DELAY * (i % 7) / 7)
                    for i in range(96)
                )
            )
        assert [r for r in released if r is not None]
        assert await _settled(port) == 0
        models.knobs = stream_models.StreamKnobs()
        await _answers(chain)
    finally:
        stalled.unstall.set()
        models.knobs = stream_models.StreamKnobs()
        await chain.aclose()
    gc.collect()
    await asyncio.sleep(0.1)
    assert errors == []


# --- a throttled model: the retry backoff is a stop point -------------------

# Readers stopped while their answer call backs off from a throttled model.
THROTTLED_READERS = 8


async def _cancelled_once_throttled(
    chain: GraphRAGChain, models: stream_models.StreamModels
) -> list[float]:
    """Start ``THROTTLED_READERS`` readers, cancel each once the model has
    refused as many calls (every answer call then in its retries), at
    points spread over 0.1 s; the seconds each cancelled reader took to
    end."""

    async def read() -> None:
        async for _ in chain.aquery_events(_input()):
            pass

    # The server's count is the module's: this test's calls are those past it.
    refused = models.throttled + THROTTLED_READERS
    tasks = [asyncio.create_task(read()) for _ in range(THROTTLED_READERS)]
    async with asyncio.timeout(BACKSTOP):
        while models.throttled < refused:
            assert not any(t.done() for t in tasks)
            await asyncio.sleep(0.01)
    loop = asyncio.get_running_loop()

    async def cancel(i: int, task: asyncio.Task[None]) -> float:
        await asyncio.sleep(0.1 * i / THROTTLED_READERS)
        stopped = loop.time()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        return loop.time() - stopped

    return list(await asyncio.gather(*(cancel(i, t) for i, t in enumerate(tasks))))


class _StopOrder:
    """The order, in the client, of each stop (``ReaderStop.set``) and each
    stop check a request passed (``raise_if_reader_stopped``, in every
    module that gates a model request on it), under one lock, so no wall
    clock decides which came first: a request that passed its check before
    the stop and is sent after it is one in flight, not one sent after the
    stop."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unified_kg_rag.adapters.aws import bedrock
        from unified_kg_rag.adapters.llm import chain_factory, retry
        from unified_kg_rag.adapters.local import models
        from unified_kg_rag.shared import reader_stop

        self.lock = threading.Lock()
        self.recording = True
        self.events: list[tuple[str, object]] = []
        check = reader_stop.raise_if_reader_stopped
        set_stop = reader_stop.ReaderStop.set

        def checked() -> None:
            with self.lock:
                check()
                if self.recording:
                    self.events.append(("passed", reader_stop._READER_STOP.get()))

        def stopped(stop: Any) -> None:
            with self.lock:
                if self.recording:
                    self.events.append(("stop", stop))
                set_stop(stop)

        monkeypatch.setattr(reader_stop.ReaderStop, "set", stopped)
        for module in (reader_stop, retry, chain_factory, bedrock, models):
            monkeypatch.setattr(module, "raise_if_reader_stopped", checked)

    def passed_after_a_stop(self) -> list[object]:
        """The checks passed with no stop bound, or after their query's
        stop: each would let a request out after it."""
        stops: set[int] = set()
        late: list[object] = []
        for kind, stop in self.events:
            if kind == "stop":
                stops.add(id(stop))
            elif stop is None or id(stop) in stops:
                late.append(stop)
        return late


@pytest.mark.parametrize(
    "provider", ["ollama", "openai", "bedrock", "bedrock-adaptive"]
)
async def test_a_reader_cancelled_in_retry_backoff_stops_the_query(
    wire: tuple[int, stream_models.StreamModels],
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
) -> None:
    # The model refuses every call (HTTP 429), so the answer call is in its
    # transient retries, before any chunk, when the reader is cancelled:
    # the query is cancelled at once, and the backoffs left running in a
    # worker thread (botocore's own) are stop points, so the reader is
    # released at once, no model call follows the stop and the query is
    # recorded as a stopped reader's, not as a failure once its retry
    # budget is spent.
    from unified_kg_rag.adapters.aws.bedrock import BaseBedrockModelFactory

    port, models = wire
    if provider == "bedrock":
        # botocore's adaptive mode (the default) also waits in its
        # client-side rate limiter, which no stop interrupts.
        monkeypatch.setattr(BaseBedrockModelFactory, "BOTO_RETRY_MODE", "standard")
    # Backoffs past the backstop that still fit the query's deadline (180 s)
    # and its first chunk's window (the model timeout): neither ends a
    # stopped query here, so one the reader's cancel did not end runs into
    # the backstop.
    chain = _wire_chain(
        provider.split("-")[0], port, monkeypatch, model_timeout=170, backoff=100
    )
    errors = _loop_errors()
    try:
        # Warm: a first call's imports and connections are not the backoff.
        await _answers(chain)
        recorded: list[BaseException | None] = []
        record = chain._record_query_metrics

        def recording(*args: Any, **kwargs: Any) -> Any:
            recorded.append(kwargs.get("error"))
            return record(*args, **kwargs)

        monkeypatch.setattr(chain, "_record_query_metrics", recording)
        models.knobs = stream_models.StreamKnobs(throttled=True)
        order = _StopOrder(monkeypatch)
        # The waits the clients run themselves (the OpenAI client's backoff,
        # botocore's and its adaptive rate limiter) still end, unsent.
        async with asyncio.timeout(BACKSTOP):
            await _cancelled_once_throttled(chain, models)
        # Every reader's stop came, and no request passed its stop check
        # after its query's stop (the botocore and OpenAI clients' own
        # resends included): one that passed it before is in flight. The
        # Ollama client sends from the event loop only, where the reader's
        # cancel ends the query, so it has no check of its own ...
        assert sum(kind == "stop" for kind, _ in order.events) == THROTTLED_READERS
        assert any(kind == "passed" for kind, _ in order.events) == (
            provider != "ollama"
        )
        assert order.passed_after_a_stop() == []
        # ... and none reached the model after those in flight: a reader is
        # released at once, before a worker thread's request that passed
        # its check before the stop reaches the model.
        async with asyncio.timeout(BACKSTOP):
            calls = -1
            while models.throttled != calls:
                calls = models.throttled
                await asyncio.sleep(0.5)
        await asyncio.sleep(2.5)
        assert models.throttled == calls
        assert recorded == [None] * THROTTLED_READERS
        order.recording = False
        models.knobs = stream_models.StreamKnobs()
        await _answers(chain)
    finally:
        models.knobs = stream_models.StreamKnobs()
        await chain.aclose()
    assert errors == []
