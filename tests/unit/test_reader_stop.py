# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The retry backoffs are stop points of a query whose reader stopped.

``shared.query_stop``: once a query's reader stops, no retry attempt
starts and a backoff under way ends at once (``asyncio.CancelledError``),
in the call-level helpers, the stream retries and a botocore client's own
resends; without a stop bound the waits are plain sleeps.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from botocore.httpsession import URLLib3Session
from langchain_core.runnables import Runnable, RunnableConfig

from tests.fixtures.botocore_backoff import stub_botocore_backoff
from unified_kg_rag.adapters.llm.chain_factory import TransientRetryRunnable
from unified_kg_rag.adapters.llm.retry import (
    acall_with_transient_retry,
    call_with_transient_retry,
)
from unified_kg_rag.domain.models.config import RetryPolicy
from unified_kg_rag.shared.query_stop import (
    QueryStop,
    abackoff_wait,
    await_unless_stopped,
    backoff_wait,
    detached_context,
    query_stop_scope,
    reader_stopped,
)

pytestmark = pytest.mark.unit

# Backoffs no test outlives: a stop that does not end one hangs the test
# into the backstop. The stops are set as the first attempt fails (a cue),
# so they land before or during the first backoff, never on a timer.
_SLOW = RetryPolicy(
    max_attempts=5,
    base_delay_seconds=3600.0,
    max_delay_seconds=3600.0,
    jitter="none",
    max_total_seconds=None,
)
# A hang backstop (seconds), far past anything a passing run waits.
_BACKSTOP = 60.0
# botocore's own send, before the suite blocks it (the Bedrock loopback
# below lets only requests to 127.0.0.1 through).
_REAL_SEND = URLLib3Session.send


class _Throttled(Exception):
    pass


def _transient(exc: BaseException) -> bool:
    return isinstance(exc, _Throttled)


class _Refusing(Runnable[Any, Any]):
    """A model that refuses every call (a throttled one), counting them;
    ``on_call`` runs at each (a test's cue to stop its query)."""

    def __init__(self, on_call: Any = None) -> None:
        self.calls = 0
        self.on_call = on_call

    def _refuse(self) -> None:
        self.calls += 1
        if self.on_call is not None:
            self.on_call()
        raise _Throttled

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        self._refuse()

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        self._refuse()

    def stream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Iterator[Any]:
        self._refuse()
        yield  # pragma: no cover

    async def astream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        self._refuse()
        yield  # pragma: no cover


def _retrying(model: _Refusing) -> TransientRetryRunnable:
    return TransientRetryRunnable(
        model, operation="answer", retry=_SLOW, is_transient=_transient
    )


def _stop_soon(stop: QueryStop) -> Any:
    """An ``on_call`` that sets ``stop`` on the loop's next turn: the call
    fails, its retry starts its backoff, then the reader stops."""
    loop = asyncio.get_running_loop()
    return lambda: loop.call_soon(stop.set)


def _stop_from_a_thread(stop: QueryStop) -> Any:
    """An ``on_call`` that sets ``stop`` from another thread (a sync call's
    reader stops on the chain's loop)."""
    return lambda: threading.Thread(target=stop.set).start()


async def _stopped(stop: QueryStop, call: Any) -> None:
    """Run ``call()`` under ``stop``: it must end with ``CancelledError``
    (its backoff of an hour cut short), within the backstop."""
    with query_stop_scope(stop), pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(_BACKSTOP):
            await call()


async def test_ainvoke_backoff_ends_at_the_stop() -> None:
    stop = QueryStop()
    model = _Refusing(_stop_soon(stop))
    await _stopped(stop, lambda: _retrying(model).ainvoke("q"))
    assert model.calls == 1


async def test_astream_backoff_ends_at_the_stop() -> None:
    stop = QueryStop()
    model = _Refusing(_stop_soon(stop))

    async def read() -> None:
        async for _ in _retrying(model).astream("q"):
            pass

    await _stopped(stop, read)
    assert model.calls == 1


async def test_acall_with_transient_retry_backoff_ends_at_the_stop() -> None:
    stop = QueryStop()
    calls = 0
    stop_soon = _stop_soon(stop)

    async def call() -> None:
        nonlocal calls
        calls += 1
        stop_soon()
        raise _Throttled

    await _stopped(
        stop,
        lambda: acall_with_transient_retry(
            call, is_transient=_transient, operation="store", policy=_SLOW
        ),
    )
    assert calls == 1


async def test_no_attempt_starts_after_the_stop() -> None:
    model = _Refusing()
    stop = QueryStop()
    stop.set()
    with query_stop_scope(stop), pytest.raises(asyncio.CancelledError):
        await _retrying(model).ainvoke("q")
    assert model.calls == 0


def _in_a_worker(run: Any) -> BaseException | None:
    """``run()`` in a thread, joined within the backstop; what it raised."""
    raised: list[BaseException | None] = [None]

    def target() -> None:
        try:
            run()
        except BaseException as e:  # noqa: BLE001 - handed to the test
            raised[0] = e

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(_BACKSTOP)
    assert not worker.is_alive(), "the backoff was not cut short"
    return raised[0]


@pytest.mark.parametrize("api", ["stream", "invoke"])
def test_sync_backoff_ends_at_a_stop_from_another_thread(api: str) -> None:
    stop = QueryStop()
    model = _Refusing(_stop_from_a_thread(stop))
    retrying = _retrying(model)

    def run() -> None:
        with query_stop_scope(stop):
            if api == "stream":
                list(retrying.stream("q"))
            else:
                retrying.invoke("q")

    assert isinstance(_in_a_worker(run), asyncio.CancelledError)
    assert model.calls == 1


def test_call_with_transient_retry_honours_the_stop() -> None:
    stop = QueryStop()
    calls = 0
    stop_from_a_thread = _stop_from_a_thread(stop)

    def call() -> None:
        nonlocal calls
        calls += 1
        stop_from_a_thread()
        raise _Throttled

    def run() -> None:
        with query_stop_scope(stop):
            call_with_transient_retry(
                call, is_transient=_transient, operation="embed", policy=_SLOW
            )

    assert isinstance(_in_a_worker(run), asyncio.CancelledError)
    assert calls == 1


async def test_without_a_stop_the_waits_are_plain_sleeps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []

    async def fake_sleep(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await abackoff_wait(3.0)
    backoff_wait(2.0, slept.append)
    assert slept == [3.0, 2.0]
    assert not reader_stopped()


async def test_a_wait_not_stopped_lasts_its_delay() -> None:
    stop = QueryStop()
    loop = asyncio.get_running_loop()
    started = loop.time()
    with query_stop_scope(stop):
        await abackoff_wait(0.1)
    assert loop.time() - started >= 0.09
    assert not stop._waiters


def _bedrock_client() -> Any:
    import boto3

    from unified_kg_rag.adapters.aws.bedrock import stop_at_query_stop

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id="AKIDEXAMPLE",
        aws_secret_access_key="example",
    )
    stop_at_query_stop(client, "bedrock-runtime")
    return client


def _attempt(client: Any, status: int) -> dict[str, Any]:
    """``needs-retry``'s arguments for an attempt answered with ``status``."""
    from botocore.awsrequest import AWSResponse

    error = {"Error": {"Code": "ThrottlingException"}} if status >= 300 else {}
    return {
        "attempts": 1,
        "response": (
            AWSResponse("https://bedrock.invalid", status, {}, None),
            {"ResponseMetadata": {"HTTPStatusCode": status}, **error},
        ),
        "caught_exception": None,
        "request_dict": {"context": {}},
        "operation": client.meta.service_model.operation_model("ConverseStream"),
    }


@pytest.mark.parametrize("event", ["before-send", "needs-retry"])
def test_a_botocore_client_sends_nothing_after_the_stop(event: str) -> None:
    # botocore's own resends (and its rate limiter's sends) pass these
    # events in the calling thread: each raises once the reader stopped.
    client = _bedrock_client()
    name = f"{event}.bedrock-runtime.ConverseStream"
    kwargs = _attempt(client, 429) if event == "needs-retry" else {"request": None}
    stop = QueryStop()
    with query_stop_scope(stop):
        client.meta.events.emit(name, **kwargs)
        stop.set()
        with pytest.raises(asyncio.CancelledError):
            client.meta.events.emit(name, **kwargs)
    # Outside the stopped query's context nothing is stopped.
    client.meta.events.emit(name, **kwargs)


def test_a_botocore_attempt_that_answered_is_left_to_its_caller() -> None:
    # needs-retry fires after a successful attempt too: its response is an
    # open event stream its caller reads and closes on the stop, so raising
    # there would leave the model's response open.
    client = _bedrock_client()
    stop = QueryStop()
    stop.set()
    with query_stop_scope(stop):
        client.meta.events.emit(
            "needs-retry.bedrock-runtime.ConverseStream", **_attempt(client, 200)
        )


class _ThrottlingServer:
    """A loopback Bedrock endpoint that answers every request 429
    (ThrottlingException), counting what it received; ``on_request(n)``
    runs as it answers the n-th."""

    def __init__(self, on_request: Any = None) -> None:
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.received = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                server.received += 1
                if on_request is not None:
                    on_request(server.received)
                body = json.dumps({"message": "Too many requests"}).encode()
                self.send_response(429)
                self.send_header("Content-Type", "application/json")
                self.send_header("x-amzn-ErrorType", "ThrottlingException")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def client(self, monkeypatch: pytest.MonkeyPatch, mode: str = "adaptive") -> Any:
        """A client of this endpoint (``adaptive`` retry mode by default),
        its stops and attempt counter registered as the model factories
        register them."""
        import boto3
        from botocore.config import Config as BotoConfig
        from botocore.httpsession import URLLib3Session

        from unified_kg_rag.adapters.aws.bedrock import stop_at_query_stop
        from unified_kg_rag.adapters.aws.bedrock_throttles import count_requests

        url = f"http://127.0.0.1:{self.http.server_address[1]}"

        def local_only_send(session: URLLib3Session, request: Any) -> Any:
            assert request.url.startswith(f"{url}/")
            return _REAL_SEND(session, request)

        monkeypatch.setattr(URLLib3Session, "send", local_only_send)
        client = boto3.client(
            "bedrock-runtime",
            region_name="us-west-2",
            endpoint_url=url,
            aws_access_key_id="AKIDEXAMPLE",
            aws_secret_access_key="example",
            config=BotoConfig(retries={"mode": mode, "total_max_attempts": 5}),
        )
        count_requests(client, "bedrock-runtime", "answer_generation")
        stop_at_query_stop(client, "bedrock-runtime")
        # No real backoff: the stops land on the requests, not on a timer.
        stub_botocore_backoff(monkeypatch, calling_thread_only=True)
        return client

    def close(self) -> None:
        self.http.shutdown()
        self.http.server_close()


@pytest.mark.parametrize("calling_thread_only", [False, True])
def test_the_backoff_stub_runs_for_botocore_s_waits_only(
    monkeypatch: pytest.MonkeyPatch, calling_thread_only: bool
) -> None:
    # Regression (flaky in CI): the stop tests set
    # botocore.endpoint.time.sleep, which is every thread's time.sleep, so
    # another thread's wait set the stop before the 1st request was sent.
    import time

    import botocore.endpoint

    ran: list[float] = []
    stub_botocore_backoff(
        monkeypatch, ran.append, calling_thread_only=calling_thread_only
    )

    def elsewhere() -> None:
        time.sleep(0)
        botocore.endpoint.time.sleep(0.25)

    worker = threading.Thread(target=elsewhere)
    worker.start()
    worker.join(_BACKSTOP)
    assert not worker.is_alive()
    time.sleep(0)
    assert ran == ([] if calling_thread_only else [0.25])
    botocore.endpoint.time.sleep(0.5)
    assert ran[-1] == 0.5
    assert botocore.endpoint.time.monotonic is time.monotonic


def _converse(client: Any) -> None:
    client.converse(
        modelId="m", messages=[{"role": "user", "content": [{"text": "hi"}]}]
    )


def _spy(monkeypatch: pytest.MonkeyPatch, cls: Any, name: str, before: Any) -> None:
    original = getattr(cls, name)

    def spied(self: Any, *args: Any, **kwargs: Any) -> Any:
        before()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cls, name, spied)


def test_a_throttled_attempt_ended_by_the_stop_is_counted_and_seen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The stop comes while the 2nd attempt is answered (429): that attempt
    # was sent, so it counts (sent and throttled) and the adaptive rate
    # limiter sees its throttle before the stop ends the call unresent.
    from botocore.retries import adaptive

    from unified_kg_rag.shared import usage

    stop = QueryStop()

    def stop_at_the_second(received: int) -> None:
        if received == 2:
            stop.set()

    server = _ThrottlingServer(stop_at_the_second)
    seen: list[None] = []
    _spy(
        monkeypatch,
        adaptive.ClientRateLimiter,
        "on_receiving_response",
        lambda: seen.append(None),
    )
    try:
        client = server.client(monkeypatch)
        with usage.usage_scope() as meter, query_stop_scope(stop):
            with pytest.raises(asyncio.CancelledError):
                _converse(client)
    finally:
        server.close()
    assert server.received == 2
    snapshot = meter.snapshot()
    assert snapshot["sent_requests"] == server.received
    assert snapshot["throttled_requests"] == server.received
    assert len(seen) == server.received


def test_a_resend_the_stop_prevents_spends_no_retry_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: the stop, last on the bare needs-retry, ran after
    # botocore's retry handler had taken the resend's cost (5 units of the
    # client's 500 for a throttle): 20 calls stopped at their 2nd throttle
    # spent 200 units for 20 resends never sent. Only the resends sent
    # spend now. Standard mode keeps the same quota as adaptive, without
    # the rate limiter's waits 40 throttles would add.
    from botocore.retries.quota import RetryQuota

    from unified_kg_rag.adapters.aws.bedrock import _retry_quota

    stops: list[QueryStop] = []

    def stop_at_every_second(received: int) -> None:
        if received % 2 == 0:
            stops[-1].set()

    server = _ThrottlingServer(stop_at_every_second)
    try:
        client = server.client(monkeypatch, mode="standard")
        quota = _retry_quota(client, "bedrock-runtime")
        assert isinstance(quota, RetryQuota)
        full = quota.available_capacity
        for _ in range(20):
            stops.append(QueryStop())
            with query_stop_scope(stops[-1]):
                with pytest.raises(asyncio.CancelledError):
                    _converse(client)
    finally:
        server.close()
    assert server.received == 40
    # One resend per call was sent (its 1st throttle's), at 5 units each.
    assert quota.available_capacity == full - 20 * 5


def test_a_stop_on_the_last_attempt_gives_back_no_earlier_resend_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The stop lands on the 5th throttle of 5 allowed attempts: botocore's
    # retry handler retries none and takes nothing for it, so the request
    # context still holds the cost of the 4th attempt's resend, which was
    # sent. That cost stays spent: the refund gives back only what the
    # handler took for the attempt the stop ends.
    from botocore.retries.quota import RetryQuota

    from unified_kg_rag.adapters.aws.bedrock import _retry_quota

    stops: list[QueryStop] = []

    def stop_at_every_fifth(received: int) -> None:
        if received % 5 == 0:
            stops[-1].set()

    server = _ThrottlingServer(stop_at_every_fifth)
    try:
        client = server.client(monkeypatch, mode="standard")
        quota = _retry_quota(client, "bedrock-runtime")
        assert isinstance(quota, RetryQuota)
        full = quota.available_capacity
        for calls in range(1, 6):
            stops.append(QueryStop())
            with query_stop_scope(stops[-1]):
                with pytest.raises(asyncio.CancelledError):
                    _converse(client)
            # Four resends per call were sent, at 5 units each.
            assert quota.available_capacity == full - calls * 4 * 5
    finally:
        server.close()
    assert server.received == 25


def test_an_attempt_the_retry_handler_takes_no_quota_for_keeps_the_earlier_cost() -> (
    None
):
    # Between the two needs-retry handlers the cost of an earlier resend is
    # set aside: an attempt botocore does not retry (an answer) gets it back
    # in the request context, which botocore releases after the call.
    client = _bedrock_client()
    kwargs = _attempt(client, 200)
    kwargs["request_dict"]["context"]["retry_quota_capacity"] = 5
    client.meta.events.emit("needs-retry.bedrock-runtime.ConverseStream", **kwargs)
    assert kwargs["request_dict"]["context"] == {"retry_quota_capacity": 5}


def test_a_stop_during_the_rate_limiter_wait_sends_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The 1st throttle turns the adaptive limiter on; the stop comes while
    # the 2nd attempt waits for its token, after the check before the
    # limiter: the check after it refuses the request.
    from botocore.retries import bucket

    stop = QueryStop()
    server = _ThrottlingServer()
    _spy(monkeypatch, bucket.TokenBucket, "acquire", stop.set)
    try:
        client = server.client(monkeypatch)
        with query_stop_scope(stop):
            with pytest.raises(asyncio.CancelledError):
                _converse(client)
    finally:
        server.close()
    assert stop.is_set()
    assert server.received == 1


def test_a_stop_before_a_resend_takes_no_rate_limiter_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The stop comes during botocore's backoff after the 1st throttle (the
    # limiter now on): the check before the limiter refuses the resend
    # before it takes a token. Only this thread's backoff sets the stop:
    # another thread's time.sleep set it before the 1st request was sent.
    from botocore.retries import bucket

    stop = QueryStop()
    server = _ThrottlingServer()
    tokens: list[None] = []
    backoffs: list[float] = []

    def stop_in_the_backoff(seconds: float) -> None:
        backoffs.append(seconds)
        stop.set()

    _spy(monkeypatch, bucket.TokenBucket, "acquire", lambda: tokens.append(None))
    try:
        client = server.client(monkeypatch)
        stub_botocore_backoff(
            monkeypatch, stop_in_the_backoff, calling_thread_only=True
        )
        with query_stop_scope(stop):
            with pytest.raises(asyncio.CancelledError):
                _converse(client)
    finally:
        server.close()
    assert server.received == 1
    assert len(backoffs) == 1
    assert tokens == []


async def test_the_openai_client_sends_no_request_after_the_stop() -> None:
    # The OpenAI client retries a call itself: its every request, a resend
    # included, passes the stop.
    import httpx

    from unified_kg_rag.adapters.local.models import openai_http_async_client

    sent: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(429)

    client = openai_http_async_client(5.0, httpx.MockTransport(answer))
    stop = QueryStop()
    async with client:
        with query_stop_scope(stop):
            await client.post("http://127.0.0.1:1/v1/chat/completions")
            stop.set()
            with pytest.raises(asyncio.CancelledError):
                await client.post("http://127.0.0.1:1/v1/chat/completions")
        assert len(sent) == 1
        # Outside the stopped query's context nothing is stopped.
        await client.post("http://127.0.0.1:1/v1/chat/completions")
    assert len(sent) == 2


class _CountingServer:
    """A loopback model server that answers every request at once,
    counting the requests it received."""

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.received = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                server.received += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"

    def close(self) -> None:
        self.http.shutdown()
        self.http.server_close()


def _connect_loses_its_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[asyncio.Event, asyncio.Event]:
    """Make httpcore's connect lose a cancellation, as anyio's
    ``connect_tcp`` can (it returns a connected stream to a task cancelled
    while it connected): ``connecting`` is set once a connect is under way;
    each connect waits for ``go``, then connects, whatever was cancelled in
    between. Returns ``(connecting, go)``."""
    from httpcore._backends.anyio import AnyIOBackend

    connect = AnyIOBackend.connect_tcp
    connecting, go = asyncio.Event(), asyncio.Event()

    async def connected(backend: Any, *args: Any, **kwargs: Any) -> Any:
        await go.wait()
        return await connect(backend, *args, **kwargs)

    async def losing(backend: Any, *args: Any, **kwargs: Any) -> Any:
        inner = asyncio.ensure_future(connected(backend, *args, **kwargs))
        connecting.set()
        try:
            return await asyncio.shield(inner)
        except asyncio.CancelledError:
            return await inner

    monkeypatch.setattr(AnyIOBackend, "connect_tcp", losing)
    return connecting, go


def _model_client(kind: str) -> Any:
    """An httpx client as each local provider's model builds it."""
    import httpx

    from unified_kg_rag.adapters.local.models import (
        ollama_async_client_kwargs,
        openai_http_async_client,
    )

    if kind == "openai":
        return openai_http_async_client(5.0)
    return httpx.AsyncClient(**ollama_async_client_kwargs(5.0))


@pytest.mark.parametrize("ended", ["reader stop", "query end"])
@pytest.mark.parametrize("kind", ["ollama", "openai"])
async def test_a_request_whose_connect_lost_the_cancel_is_not_sent(
    monkeypatch: pytest.MonkeyPatch, kind: str, ended: str
) -> None:
    # Regression: a cancellation that lands while a model request connects
    # can be lost (anyio's connect_tcp, on 3.11 and 3.12 alike), and with no
    # keep-alive every request connects: the request went out after its
    # reader's stop, or after the query it is for ended. The connection
    # made, the request is stopped before it is written.
    from unified_kg_rag.shared.model_responses import closing_responses

    connecting, go = _connect_loses_its_cancel(monkeypatch)
    server = _CountingServer()
    stop = QueryStop()
    try:
        async with _model_client(kind) as client:
            with query_stop_scope(stop):
                async with closing_responses():
                    request = asyncio.ensure_future(client.post(server.url))
                    # The cancel lands while the request connects.
                    async with asyncio.timeout(_BACKSTOP):
                        await connecting.wait()
                    if ended == "reader stop":
                        stop.set()
                    request.cancel()
                # The query that sent it ended (its scope with it).
            go.set()
            (outcome,) = await asyncio.gather(request, return_exceptions=True)
            assert isinstance(outcome, asyncio.CancelledError)
            # The next request, outside any stopped query, is sent.
            assert (await client.post(server.url)).status_code == 200
    finally:
        server.close()
    assert server.received == 1


class _HoldingServer:
    """A loopback model server that holds its answer's headers until
    released: ``received`` is set once a request is read."""

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.received = threading.Event()
        self.release = threading.Event()
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_: Any) -> None:
                pass

            def do_POST(self) -> None:  # noqa: N802
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                server.received.set()
                server.release.wait(_BACKSTOP)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.http.daemon_threads = True
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"

    def close(self) -> None:
        self.release.set()
        self.http.shutdown()
        self.http.server_close()


@pytest.mark.parametrize("kind", ["ollama", "openai"])
async def test_a_response_whose_headers_arrive_after_the_stop_is_closed_unread(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    # The stop comes after the request was written, while the server holds
    # its headers: the response hook is the stop point as they arrive, and
    # httpx closes the response it raises for, its body unread.
    import httpx

    from unified_kg_rag.adapters.local import models

    opened: list[httpx.Response] = []

    async def capture(response: httpx.Response) -> None:
        opened.append(response)

    monkeypatch.setitem(
        models._ASYNC_EVENT_HOOKS, "response", [capture, models._track_response]
    )
    server = _HoldingServer()
    stop = QueryStop()
    try:
        async with _model_client(kind) as client:
            with query_stop_scope(stop):
                request = asyncio.ensure_future(client.post(server.url))
                assert await asyncio.to_thread(server.received.wait, _BACKSTOP)
                stop.set()
                server.release.set()
                with pytest.raises(asyncio.CancelledError):
                    await request
    finally:
        server.close()
    (response,) = opened
    assert response.status_code == 200
    assert response.is_closed
    assert not response.is_stream_consumed


async def test_a_detached_context_binds_its_own_stop_only() -> None:
    import contextvars

    other: contextvars.ContextVar[str] = contextvars.ContextVar("other")
    mine, shared = QueryStop(), QueryStop()
    mine.set()
    other.set("kept")
    with query_stop_scope(mine):
        assert reader_stopped()
        assert not detached_context().run(reader_stopped)
        assert detached_context().run(other.get) == "kept"
        shared.set()
        assert detached_context(shared).run(reader_stopped)
        # The reader's own context still has its stop.
        assert reader_stopped()


async def test_a_stopped_reader_leaves_the_shared_work_it_awaits() -> None:
    work = asyncio.get_running_loop().create_future()
    stop = QueryStop()
    with query_stop_scope(stop):
        waiting = asyncio.ensure_future(await_unless_stopped(work))
        await asyncio.sleep(0)
        stop.set()
        with pytest.raises(asyncio.CancelledError):
            async with asyncio.timeout(_BACKSTOP):
                await waiting
        # Ended at its own stop, not cancelled: the work is not its alone.
        assert not work.done()
        assert not stop._waiters
    work.set_result("value")
    # Done, or without a stop bound, it is a plain wait.
    with query_stop_scope(QueryStop()):
        await await_unless_stopped(work)
    other = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_later(0.01, other.set_result, "value")
    await await_unless_stopped(other)
    assert other.result() == "value"


# The package's context variables a call several queries share keeps from
# its first caller, and why: none is a query's own state.
_KEPT_IN_SHARED_CALLS = {
    # Rebound to the call's own stop (SharedContext).
    "shared/query_stop.py:_QUERY_STOP": "the call's SharedStop",
    # Taken from all its waiters (share_with_waiters).
    "adapters/llm/retry.py:_RETRY_DEADLINE": "the waiters' latest deadline",
    "shared/usage.py:_CURRENT": "the earliest waiter's meter",
    # Where the call was sent from: logged and traced there.
    "shared/observability.py:_BOUND_ATTRIBUTES": "span attributes",
    "shared/observability.py:_TIMINGS": "span timings",
    "adapters/aws/bedrock.py:_IN_EMBEDDING_SPAN": "span nesting",
    # A run's, not a query's.
    "shared/write_fence.py:_FENCES": "an ingestion run's write fences",
    "shared/store_lookups.py:_CACHE": "a run's store-lookup cache",
    "adapters/llm/retry.py:_LEDGER": "a stage's failure ledger",
    "adapters/llm/response_cache.py:_UNCACHED": "a run's cache bypass",
    "application/retrieval/query_processing.py:_EXTRACTION_SCOPE": (
        "a run's extraction cache scope"
    ),
}


def _creates_variable(value: Any) -> bool:
    """Whether ``value`` (an assignment's AST) calls ``ContextVar``."""
    import ast

    if not isinstance(value, ast.Call):
        return False
    func = value.func
    name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
    return name == "ContextVar"


def _package_context_variables() -> dict[str, Any]:
    """Every module-level ``ContextVar`` of the package, by
    ``path:name``."""
    import ast
    import importlib
    from pathlib import Path

    import unified_kg_rag

    root = Path(unified_kg_rag.__file__).parent
    found: dict[str, Any] = {}
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            target = (
                node.target
                if isinstance(node, ast.AnnAssign)
                else (
                    node.targets[0]
                    if isinstance(node, ast.Assign) and len(node.targets) == 1
                    else None
                )
            )
            value = getattr(node, "value", None)
            if not isinstance(target, ast.Name) or not _creates_variable(value):
                continue
            relative = path.relative_to(root)
            module = importlib.import_module(
                ".".join(["unified_kg_rag", *relative.with_suffix("").parts])
            )
            found[f"{relative.as_posix()}:{target.id}"] = getattr(module, target.id)
    return found


def test_a_shared_call_starts_without_any_query_s_own_state() -> None:
    # Every context variable is either blanked in a call several queries
    # share (blank_in_shared_calls) or kept for a stated reason: a new one
    # is classified here, so none leaks a first caller's state by accident.
    from unified_kg_rag.shared.query_stop import _BLANKED

    variables = _package_context_variables()
    assert len(variables) > len(_KEPT_IN_SHARED_CALLS)
    unclassified = [
        key
        for key, variable in variables.items()
        if variable not in _BLANKED and key not in _KEPT_IN_SHARED_CALLS
    ]
    assert unclassified == []
    assert [k for k in _KEPT_IN_SHARED_CALLS if k not in variables] == []
    assert [k for k in _KEPT_IN_SHARED_CALLS if variables[k] in _BLANKED] == []
