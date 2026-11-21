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
from unified_kg_rag.shared import query_stop

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
        # Each disconnect is recorded as stopped, not failed.
        rows = service.store.queries(limit=500)
        assert {row["outcome"] for row in rows} <= {"stopped", "answered"}
        # A disconnected question records the answer call it spent (rv53).
        cancelled = [row for row in rows if row["outcome"] == "stopped"]
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


def _sender() -> tuple[int, int]:
    """The thread and the task (0: none) a check or a send runs in: a
    request's last stop check runs where the request is then sent (a
    botocore worker thread's ``before-send``, an httpx client's ``request``
    hook in the task that sends)."""
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), id(task) if task is not None else 0


def _query_of(stop: object) -> object:
    """The query stop ``stop`` follows (its reader's), or ``stop``."""
    while isinstance(stop, query_stop.FollowingStop):
        stop = stop._leader
    return stop


class _StopOrder:
    """The order, in the client, of each stop (``QueryStop.set``), each
    stop check a request passed (``raise_if_stopped``, in every
    module that gates a model request on it) and each request the clients
    put on the wire (botocore's ``URLLib3Session.send``, httpx's
    transports), under one lock, so no wall clock decides which came
    first: a request that passed its check before the stop and is sent
    after it is one in flight, not one sent after the stop.

    It also counts the requests on the wire and the botocore calls (their
    attempts and botocore's backoffs between them) not yet returned:
    :meth:`settled` waits until none is, as the requests in flight end
    (a send returns once the server read the request and answered it)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import botocore.endpoint
        import httpx

        from unified_kg_rag.adapters.aws import bedrock
        from unified_kg_rag.adapters.llm import chain_factory, retry
        from unified_kg_rag.adapters.local import models

        # Reentrant: a stop sets its followers (a deadline window's) under
        # its own set.
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.recording = True
        self.events: list[tuple[str, object]] = []
        # The senders whose last check passed and who sent nothing since.
        self.checked: set[tuple[int, int]] = set()
        # Each request sent with no check passed in its sender since its
        # last one (its query's stop, or None).
        self.unchecked: list[object] = []
        self.sending = 0
        self.calling = 0
        check = query_stop.raise_if_stopped
        set_stop = query_stop.QueryStop.set

        def checked() -> None:
            with self.lock:
                check()
                if self.recording:
                    self.events.append(("passed", query_stop._QUERY_STOP.get()))
                    self.checked.add(_sender())

        def stopped(stop: Any, *cause: Any) -> None:
            with self.lock:
                if self.recording:
                    self.events.append(("stop", stop))
                set_stop(stop, *cause)

        monkeypatch.setattr(query_stop.QueryStop, "set", stopped)
        for module in (query_stop, retry, chain_factory, bedrock, models):
            monkeypatch.setattr(module, "raise_if_stopped", checked)

        def sent() -> None:
            with self.lock:
                self.sending += 1
                if self.recording:
                    stop = query_stop._QUERY_STOP.get()
                    self.events.append(("sent", stop))
                    if _sender() not in self.checked:
                        self.unchecked.append(stop)
                    self.checked.discard(_sender())

        def answered(ok: bool) -> None:
            with self.lock:
                self.sending -= 1
                if self.recording:
                    self.events.append(
                        ("answered" if ok else "lost", query_stop._QUERY_STOP.get())
                    )
                # A check the transport ran for this request (httpx's,
                # once connected) is not the next request's.
                self.checked.discard(_sender())
                self.changed.notify_all()

        def on_the_wire(send: Any) -> Any:
            def sending(owner: Any, request: Any) -> Any:
                sent()
                ok = False
                try:
                    response = send(owner, request)
                    ok = True
                    return response
                finally:
                    answered(ok)

            return sending

        async_send = httpx.AsyncHTTPTransport.handle_async_request

        async def async_sending(transport: Any, request: Any) -> Any:
            sent()
            ok = False
            try:
                response = await async_send(transport, request)
                ok = True
                return response
            finally:
                answered(ok)

        botocore_call = botocore.endpoint.Endpoint._send_request  # type: ignore[attr-defined]

        def calling(endpoint: Any, *args: Any, **kwargs: Any) -> Any:
            with self.lock:
                self.calling += 1
            try:
                return botocore_call(endpoint, *args, **kwargs)
            finally:
                with self.lock:
                    self.calling -= 1
                    self.changed.notify_all()

        monkeypatch.setattr(URLLib3Session, "send", on_the_wire(URLLib3Session.send))
        monkeypatch.setattr(
            httpx.HTTPTransport,
            "handle_request",
            on_the_wire(httpx.HTTPTransport.handle_request),
        )
        monkeypatch.setattr(
            httpx.AsyncHTTPTransport, "handle_async_request", async_sending
        )
        monkeypatch.setattr(botocore.endpoint.Endpoint, "_send_request", calling)

    def passed_after_a_stop(self) -> list[object]:
        """The checks passed with no stop bound, or after their query's
        stop: each would let a request out after it."""
        stops: set[int] = set()
        late: list[object] = []
        for kind, stop in self.events:
            if kind == "stop":
                stops.add(id(stop))
            elif kind == "passed" and (stop is None or id(stop) in stops):
                late.append(stop)
        return late

    def in_flight_at_the_stop(self) -> dict[int, int]:
        """For each stopped query, its requests on the wire when its stop
        came (its reader's, or a deadline window's of it, whichever was
        first) or put on it after: each passed its check before the stop
        (:meth:`passed_after_a_stop`)."""
        stops: set[int] = set()
        on_wire: dict[int, int] = {}
        in_flight: dict[int, int] = {}
        for kind, stop in self.events:
            query = id(_query_of(stop))
            if kind == "stop":
                if query not in stops:
                    stops.add(query)
                    in_flight[query] = on_wire.get(query, 0)
            elif kind == "sent":
                on_wire[query] = on_wire.get(query, 0) + 1
                if query in stops:
                    in_flight[query] += 1
            elif kind in ("answered", "lost"):
                on_wire[query] -= 1
        return in_flight

    def count(self, *kinds: str) -> int:
        return sum(kind in kinds for kind, _ in self.events)

    def settled(self, timeout: float) -> bool:
        """Whether, within ``timeout`` s, no request is on the wire and no
        botocore call is left to make one."""
        with self.changed:
            return self.changed.wait_for(
                lambda: not self.sending and not self.calling, timeout
            )


def _hold_a_send_past_its_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hold the first request botocore puts on the wire for a query until
    that query stops, as a connect does that outlasts the stop (one the
    server's full accept queue refused, retried after 1 s): the request is
    written after the stop, its check passed before it."""
    send = URLLib3Session.send
    lock = threading.Lock()
    held = False

    def holding(session: URLLib3Session, request: Any) -> Any:
        nonlocal held
        stop = query_stop._QUERY_STOP.get()
        with lock:
            hold = stop is not None and not held
            held = held or hold
        if hold:
            assert stop is not None
            stop._wait(BACKSTOP)
        return send(session, request)

    monkeypatch.setattr(URLLib3Session, "send", holding)


@pytest.mark.parametrize(
    "provider",
    ["ollama", "openai", "bedrock", "bedrock-adaptive", "bedrock-in-flight"],
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
    if provider != "bedrock-adaptive":
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
        if provider == "bedrock-in-flight":
            _hold_a_send_past_its_stop(monkeypatch)
        order = _StopOrder(monkeypatch)
        received = models.received
        models.knobs = stream_models.StreamKnobs(throttled=True)
        # The waits the clients run themselves (the OpenAI client's backoff,
        # botocore's and its adaptive rate limiter) still end, unsent.
        async with asyncio.timeout(BACKSTOP):
            await _cancelled_once_throttled(chain, models)
        # Every reader's stop came. Once no task, no request on the wire
        # and no botocore call (its backoffs included) is left, nothing
        # can send a request of these queries: their requests in flight
        # have been answered, so the server's count is final.
        current = asyncio.current_task()
        async with asyncio.timeout(BACKSTOP):
            others = [t for t in asyncio.all_tasks() if t is not current]
            if others:
                await asyncio.wait(others)
            assert await asyncio.to_thread(order.settled, BACKSTOP)
        assert (
            sum(
                kind == "stop" and not isinstance(stop, query_stop.FollowingStop)
                for kind, stop in order.events
            )
            == THROTTLED_READERS
        )
        # No request passed its stop check after its query's stop (the
        # botocore, OpenAI and Ollama clients' requests, their own resends
        # included), and none was sent without passing one ...
        assert any(kind == "passed" for kind, _ in order.events)
        assert order.passed_after_a_stop() == []
        assert order.unchecked == []
        # ... so a request on the wire at its query's stop, or put on it
        # after, passed its check before the stop and is in flight: at most
        # one of each query. botocore's last hook (``before-send``) runs
        # before its worker thread connects and writes the request in one
        # blocking send that no stop interrupts (a connect the server's
        # full accept queue refused retries after the kernel's 1 s SYN
        # timeout), and the attempt after it is refused (``needs-retry``,
        # ``before-send``).
        in_flight = order.in_flight_at_the_stop()
        assert len(in_flight) == THROTTLED_READERS
        assert max(in_flight.values()) <= 1
        if provider == "bedrock-in-flight":
            assert 1 in in_flight.values()
        # The model read no request but those the clients sent: each one
        # answered, and perhaps one whose connection failed.
        read = models.received - received
        assert order.count("answered") <= read <= order.count("sent")
        assert recorded == [None] * THROTTLED_READERS
        order.recording = False
        models.knobs = stream_models.StreamKnobs()
        await _answers(chain)
    finally:
        models.knobs = stream_models.StreamKnobs()
        await chain.aclose()
    assert errors == []
