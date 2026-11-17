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
import time
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
    backoff_wait,
    reader_stop_scope,
    reader_stopped,
)

pytestmark = pytest.mark.unit

# Long waits: a stop that does not end them fails the test's timing.
_SLOW = RetryPolicy(
    max_attempts=5,
    base_delay_seconds=30.0,
    max_delay_seconds=30.0,
    jitter="none",
    max_total_seconds=None,
)


class _Throttled(Exception):
    pass


def _transient(exc: BaseException) -> bool:
    return isinstance(exc, _Throttled)


class _Refusing(Runnable[Any, Any]):
    """A model that refuses every call (a throttled one), counting them."""

    def __init__(self) -> None:
        self.calls = 0

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        self.calls += 1
        raise _Throttled

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Any:
        self.calls += 1
        raise _Throttled

    def stream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> Iterator[Any]:
        self.calls += 1
        raise _Throttled
        yield  # pragma: no cover

    async def astream(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[Any]:
        self.calls += 1
        raise _Throttled
        yield  # pragma: no cover


def _retrying(model: _Refusing) -> TransientRetryRunnable:
    return TransientRetryRunnable(
        model, operation="answer", retry=_SLOW, is_transient=_transient
    )


async def _stopped_after(stop: ReaderStop, seconds: float, call: Any) -> float:
    """Run ``call()`` under ``stop``, set ``seconds`` in; the seconds from
    the stop to its end, which must be a ``CancelledError``."""
    loop = asyncio.get_running_loop()
    stopped_at: list[float] = []

    def set_stop() -> None:
        stopped_at.append(loop.time())
        stop.set()

    loop.call_later(seconds, set_stop)
    with reader_stop_scope(stop), pytest.raises(asyncio.CancelledError):
        await call()
    return loop.time() - stopped_at[0]


async def test_ainvoke_backoff_ends_at_the_stop() -> None:
    model = _Refusing()
    released = await _stopped_after(
        ReaderStop(), 0.1, lambda: _retrying(model).ainvoke("q")
    )
    assert released < 0.5
    assert model.calls == 1


async def test_astream_backoff_ends_at_the_stop() -> None:
    model = _Refusing()

    async def read() -> None:
        async for _ in _retrying(model).astream("q"):
            pass

    assert await _stopped_after(ReaderStop(), 0.1, read) < 0.5
    assert model.calls == 1


async def test_acall_with_transient_retry_backoff_ends_at_the_stop() -> None:
    calls = 0

    async def call() -> None:
        nonlocal calls
        calls += 1
        raise _Throttled

    released = await _stopped_after(
        ReaderStop(),
        0.1,
        lambda: acall_with_transient_retry(
            call, is_transient=_transient, operation="store", policy=_SLOW
        ),
    )
    assert released < 0.5
    assert calls == 1


async def test_no_attempt_starts_after_the_stop() -> None:
    model = _Refusing()
    stop = ReaderStop()
    stop.set()
    with reader_stop_scope(stop), pytest.raises(asyncio.CancelledError):
        await _retrying(model).ainvoke("q")
    assert model.calls == 0


@pytest.mark.parametrize("api", ["stream", "invoke"])
def test_sync_backoff_ends_at_a_stop_from_another_thread(api: str) -> None:
    model = _Refusing()
    stop = ReaderStop()
    retrying = _retrying(model)

    def run() -> None:
        if api == "stream":
            list(retrying.stream("q"))
        else:
            retrying.invoke("q")

    threading.Timer(0.1, stop.set).start()
    started = time.monotonic()
    with reader_stop_scope(stop), pytest.raises(asyncio.CancelledError):
        run()
    assert time.monotonic() - started < 0.6
    assert model.calls == 1


def test_call_with_transient_retry_honours_the_stop() -> None:
    calls = 0

    def call() -> None:
        nonlocal calls
        calls += 1
        raise _Throttled

    stop = ReaderStop()
    threading.Timer(0.1, stop.set).start()
    started = time.monotonic()
    with reader_stop_scope(stop), pytest.raises(asyncio.CancelledError):
        call_with_transient_retry(
            call, is_transient=_transient, operation="embed", policy=_SLOW
        )
    assert time.monotonic() - started < 0.6
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


@pytest.mark.parametrize("event", ["before-send", "needs-retry"])
def test_a_botocore_client_sends_nothing_after_the_stop(event: str) -> None:
    # botocore's own resends (and its rate limiter's sends) pass these
    # events in the calling thread: each raises once the reader stopped.
    import boto3
    from botocore.awsrequest import AWSResponse

    from unified_kg_rag.adapters.aws.bedrock import stop_at_reader_stop

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id="AKIDEXAMPLE",
        aws_secret_access_key="example",
    )
    stop_at_reader_stop(client, "bedrock-runtime")
    name = f"{event}.bedrock-runtime.Converse"
    # An attempt that answered: no resend is due.
    kwargs: dict[str, Any] = (
        {
            "attempts": 1,
            "response": (
                AWSResponse("https://bedrock.invalid", 200, {}, None),
                {"ResponseMetadata": {"HTTPStatusCode": 200}},
            ),
            "caught_exception": None,
            "request_dict": {"context": {}},
            "operation": client.meta.service_model.operation_model("Converse"),
        }
        if event == "needs-retry"
        else {"request": None}
    )
    stop = ReaderStop()
    with reader_stop_scope(stop):
        client.meta.events.emit(name, **kwargs)
        stop.set()
        with pytest.raises(asyncio.CancelledError):
            client.meta.events.emit(name, **kwargs)
    # Outside the stopped query's context nothing is stopped.
    client.meta.events.emit(name, **kwargs)


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
