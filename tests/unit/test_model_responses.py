# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The model responses a scope has open are closed when it ends
(``shared.model_responses``), and Bedrock's event streams are closed from
any thread without touching a connection back in the pool."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import socket
import threading
from typing import Any

import pytest
from botocore.eventstream import EventStream

from unified_kg_rag.adapters.aws import bedrock
from unified_kg_rag.shared import model_responses
from unified_kg_rag.shared.model_responses import closing_responses, track_response

pytestmark = [pytest.mark.unit]

# A hang backstop (seconds), far past anything a passing run waits.
_BACKSTOP = 60.0


class _AsyncResponse:
    """An ``httpx.Response``'s surface: ``is_closed`` and ``aclose``."""

    def __init__(self, *, closed: bool = False, fails: bool = False) -> None:
        self.is_closed = closed
        self.closes = 0
        self.fails = fails

    async def aclose(self) -> None:
        self.closes += 1
        if self.fails:
            raise OSError("socket gone")
        self.is_closed = True


class _SyncResponse:
    def __init__(self) -> None:
        self.closes = 0

    def close(self) -> None:
        self.closes += 1


def test_nothing_is_tracked_outside_a_scope() -> None:
    response = _SyncResponse()
    track_response(response)
    assert response.closes == 0


async def test_a_scope_closes_what_is_still_open_when_it_ends() -> None:
    still_open, sync, read = _AsyncResponse(), _SyncResponse(), _AsyncResponse()
    async with closing_responses():
        for response in (still_open, sync, read):
            track_response(response)
        read.is_closed = True  # read to its end
    assert (still_open.closes, sync.closes, read.closes) == (1, 1, 0)


async def test_a_cancelled_scope_closes_them_and_stays_cancelled() -> None:
    response = _AsyncResponse()
    entered = asyncio.Event()

    async def run() -> None:
        async with closing_responses():
            track_response(response)
            entered.set()
            await asyncio.sleep(30)

    task = asyncio.create_task(run())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert response.closes == 1


async def test_a_scope_cancelled_again_while_it_closes_still_closes() -> None:
    # An anyio scope cancels every await: the closing goes on by itself.
    release = asyncio.Event()
    closed = asyncio.Event()

    class Slow(_AsyncResponse):
        async def aclose(self) -> None:
            await release.wait()
            closed.set()

    entered = asyncio.Event()

    async def run() -> None:
        async with closing_responses():
            track_response(Slow())
            entered.set()
            await asyncio.sleep(30)

    task = asyncio.create_task(run())
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    release.set()
    await asyncio.wait_for(closed.wait(), 5)


async def test_a_response_a_worker_thread_opens_is_tracked() -> None:
    response = _SyncResponse()
    async with closing_responses():
        await asyncio.to_thread(track_response, response)
        assert response.closes == 0
    assert response.closes == 1


async def test_one_opened_after_its_scope_ended_is_closed_at_once() -> None:
    # A worker thread the query left behind gets its response after the
    # scope ended: no one reads it.
    sync, later = _SyncResponse(), _AsyncResponse()
    async with closing_responses():
        context = contextvars.copy_context()
    context.run(track_response, sync)
    assert sync.closes == 1
    context.run(track_response, later)
    for _ in range(10):
        await asyncio.sleep(0)
    assert later.closes == 1


async def test_one_a_worker_thread_opens_after_its_scope_is_closed_there() -> None:
    # Off the event loop no task can close it: its own close() does, though
    # it closes on the loop by aclose().
    class Both(_SyncResponse):
        async def aclose(self) -> None:
            raise AssertionError("closed on the loop")

    response = Both()
    async with closing_responses():
        context = contextvars.copy_context()
    await asyncio.to_thread(context.run, track_response, response)
    assert response.closes == 1


async def test_a_failed_close_is_logged_and_the_others_still_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    failing, other = _AsyncResponse(fails=True), _SyncResponse()

    class Broken(_SyncResponse):
        def close(self) -> None:
            raise OSError("socket gone")

    caplog.set_level(logging.DEBUG, logger=model_responses.__name__)
    async with closing_responses():
        context = contextvars.copy_context()
        for response in (failing, other):
            track_response(response)
    context.run(track_response, Broken())
    assert (failing.closes, other.closes) == (1, 1)
    assert [r.levelno for r in caplog.records] == [logging.DEBUG] * 2


async def test_a_long_scope_keeps_only_the_responses_still_open() -> None:
    async with closing_responses():
        scope = model_responses._OPEN_RESPONSES.get()
        assert scope is not None
        for _ in range(50):
            track_response(_AsyncResponse(closed=True))
        track_response(_AsyncResponse())
        assert len(scope._open) == 1


# --- Bedrock's event streams ------------------------------------------------


class _Raw:
    """A urllib3 response's surface: its connection (None once released to
    the pool), ``closed`` and ``close``."""

    def __init__(self, sock: Any) -> None:
        self.connection: Any = None if sock is None else type("C", (), {})()
        if self.connection is not None:
            self.connection.sock = sock
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _Sock:
    def __init__(self) -> None:
        self.shut: list[int] = []

    def shutdown(self, how: int) -> None:
        self.shut.append(how)


def test_an_event_stream_s_socket_is_shut_for_reading_then_closed() -> None:
    sock = _Sock()
    raw = _Raw(sock)
    stream = bedrock._OpenEventStream(raw)
    assert not stream.is_closed
    stream.close()
    assert sock.shut == [socket.SHUT_RD] and raw.closed and stream.is_closed


def test_an_event_stream_released_to_the_pool_is_left_alone() -> None:
    # Its connection may already serve another request.
    raw = _Raw(None)
    stream = bedrock._OpenEventStream(raw)
    assert stream.is_closed
    stream.close()
    assert raw.closed


async def test_an_event_stream_closes_off_the_event_loop() -> None:
    loop_thread = threading.get_ident()
    closed_on: list[int] = []

    class Raw(_Raw):
        def close(self) -> None:
            closed_on.append(threading.get_ident())

    await bedrock._OpenEventStream(Raw(_Sock())).aclose()
    assert closed_on and closed_on[0] != loop_thread


async def test_only_a_call_that_opens_an_event_stream_is_tracked() -> None:
    raw = _Raw(_Sock())
    response = type("R", (), {"raw": raw})()
    async with closing_responses():
        bedrock._track_event_streams(http_response=response, parsed={"output": {}})
        scope = model_responses._OPEN_RESPONSES.get()
        assert scope is not None and scope._open == []
        stream: Any = EventStream.__new__(EventStream)
        bedrock._track_event_streams(http_response=response, parsed={"stream": stream})
        assert len(scope._open) == 1
    assert raw.closed


async def test_a_bedrock_request_after_its_scope_ended_is_not_sent() -> None:
    """A worker thread the query left (its context copied inside the
    answer stream's scope) sends no request once the scope ended: its
    before-send refuses it, so no billed call opens a stream nobody reads."""
    import boto3

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-west-2",
        aws_access_key_id="AKIDEXAMPLE",
        aws_secret_access_key="example",
    )
    bedrock.stop_at_query_stop(client, "bedrock-runtime")

    def send() -> None:
        client.meta.events.emit(
            "before-send.bedrock-runtime.ConverseStream", request=None
        )

    async with closing_responses():
        worker = contextvars.copy_context()
        await asyncio.to_thread(worker.run, send)  # in the scope: sent
    with pytest.raises(asyncio.CancelledError):
        await asyncio.to_thread(worker.run, send)
    # Outside any scope nothing is refused.
    send()


def test_an_event_stream_closed_as_its_connection_is_released_spares_the_pool() -> None:
    """The reading thread releases the connection as the stream ends: back
    in the pool first, then no longer the response's. A close that comes
    in between waits for the release, and then leaves the connection (by
    now another request's) alone."""
    pool: list[Any] = []
    in_pool, go_on = threading.Event(), threading.Event()

    class Raw(_Raw):
        def release_conn(self) -> None:
            pool.append(self.connection)
            in_pool.set()
            assert go_on.wait(_BACKSTOP)
            self.connection = None

    sock = _Sock()
    raw = Raw(sock)
    stream = bedrock._OpenEventStream(raw)
    reader = threading.Thread(target=raw.release_conn)
    reader.start()
    assert in_pool.wait(_BACKSTOP)
    closer = threading.Thread(target=stream.close)
    closer.start()
    closer.join(0.2)
    assert closer.is_alive()  # waits for the release
    go_on.set()
    reader.join(_BACKSTOP)
    closer.join(_BACKSTOP)
    assert sock.shut == [] and raw.closed and pool[0].sock is sock
