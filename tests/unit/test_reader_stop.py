# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The retry backoffs are stop points of a query whose reader stopped.

``shared.reader_stop``: once a query's reader stops, no retry attempt
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
from langchain_core.runnables import Runnable, RunnableConfig

from unified_kg_rag.adapters.llm.chain_factory import TransientRetryRunnable
from unified_kg_rag.adapters.llm.retry import (
    acall_with_transient_retry,
    call_with_transient_retry,
)
from unified_kg_rag.domain.models.config import RetryPolicy
from unified_kg_rag.shared.reader_stop import (
    ReaderStop,
    abackoff_wait,
    await_unless_stopped,
    backoff_wait,
    detached_context,
    reader_stop_scope,
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


def _stop_soon(stop: ReaderStop) -> Any:
    """An ``on_call`` that sets ``stop`` on the loop's next turn: the call
    fails, its retry starts its backoff, then the reader stops."""
    loop = asyncio.get_running_loop()
    return lambda: loop.call_soon(stop.set)


def _stop_from_a_thread(stop: ReaderStop) -> Any:
    """An ``on_call`` that sets ``stop`` from another thread (a sync call's
    reader stops on the chain's loop)."""
    return lambda: threading.Thread(target=stop.set).start()


async def _stopped(stop: ReaderStop, call: Any) -> None:
    """Run ``call()`` under ``stop``: it must end with ``CancelledError``
    (its backoff of an hour cut short), within the backstop."""
    with reader_stop_scope(stop), pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(_BACKSTOP):
            await call()


async def test_ainvoke_backoff_ends_at_the_stop() -> None:
    stop = ReaderStop()
    model = _Refusing(_stop_soon(stop))
    await _stopped(stop, lambda: _retrying(model).ainvoke("q"))
    assert model.calls == 1


async def test_astream_backoff_ends_at_the_stop() -> None:
    stop = ReaderStop()
    model = _Refusing(_stop_soon(stop))

    async def read() -> None:
        async for _ in _retrying(model).astream("q"):
            pass

    await _stopped(stop, read)
    assert model.calls == 1


async def test_acall_with_transient_retry_backoff_ends_at_the_stop() -> None:
    stop = ReaderStop()
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
    stop = ReaderStop()
    stop.set()
    with reader_stop_scope(stop), pytest.raises(asyncio.CancelledError):
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
    stop = ReaderStop()
    model = _Refusing(_stop_from_a_thread(stop))
    retrying = _retrying(model)

    def run() -> None:
        with reader_stop_scope(stop):
            if api == "stream":
                list(retrying.stream("q"))
            else:
                retrying.invoke("q")

    assert isinstance(_in_a_worker(run), asyncio.CancelledError)
    assert model.calls == 1


def test_call_with_transient_retry_honours_the_stop() -> None:
    stop = ReaderStop()
    calls = 0
    stop_from_a_thread = _stop_from_a_thread(stop)

    def call() -> None:
        nonlocal calls
        calls += 1
        stop_from_a_thread()
        raise _Throttled

    def run() -> None:
        with reader_stop_scope(stop):
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
    stop = ReaderStop()
    loop = asyncio.get_running_loop()
    started = loop.time()
    with reader_stop_scope(stop):
        await abackoff_wait(0.1)
    assert loop.time() - started >= 0.09
    assert not stop._waiters


def _bedrock_client() -> Any:
    import boto3

    from unified_kg_rag.adapters.aws.bedrock import stop_at_reader_stop

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id="AKIDEXAMPLE",
        aws_secret_access_key="example",
    )
    stop_at_reader_stop(client, "bedrock-runtime")
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
    stop = ReaderStop()
    with reader_stop_scope(stop):
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
    stop = ReaderStop()
    stop.set()
    with reader_stop_scope(stop):
        client.meta.events.emit(
            "needs-retry.bedrock-runtime.ConverseStream", **_attempt(client, 200)
        )


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
    stop = ReaderStop()
    async with client:
        with reader_stop_scope(stop):
            await client.post("http://127.0.0.1:1/v1/chat/completions")
            stop.set()
            with pytest.raises(asyncio.CancelledError):
                await client.post("http://127.0.0.1:1/v1/chat/completions")
        assert len(sent) == 1
        # Outside the stopped query's context nothing is stopped.
        await client.post("http://127.0.0.1:1/v1/chat/completions")
    assert len(sent) == 2


async def test_a_detached_context_binds_its_own_stop_only() -> None:
    import contextvars

    other: contextvars.ContextVar[str] = contextvars.ContextVar("other")
    mine, shared = ReaderStop(), ReaderStop()
    mine.set()
    other.set("kept")
    with reader_stop_scope(mine):
        assert reader_stopped()
        assert not detached_context().run(reader_stopped)
        assert detached_context().run(other.get) == "kept"
        shared.set()
        assert detached_context(shared).run(reader_stopped)
        # The reader's own context still has its stop.
        assert reader_stopped()


async def test_a_stopped_reader_leaves_the_shared_work_it_awaits() -> None:
    work = asyncio.get_running_loop().create_future()
    stop = ReaderStop()
    with reader_stop_scope(stop):
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
    with reader_stop_scope(ReaderStop()):
        await await_unless_stopped(work)
    other = asyncio.get_running_loop().create_future()
    asyncio.get_running_loop().call_later(0.01, other.set_result, "value")
    await await_unless_stopped(other)
    assert other.result() == "value"
