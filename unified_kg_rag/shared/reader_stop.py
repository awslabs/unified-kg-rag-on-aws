# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The stop of a query whose reader stopped reading its events.

A query streamed to a reader (``GraphRAGChain.aquery_events``) runs in a
task of its own and ends at its next stop point once the reader stops: its
next event, its answer model's next chunk, and the retry loops' backoff
waits (:func:`backoff_wait`, :func:`abackoff_wait`), which a stopped reader
would otherwise keep calling a throttled model for until the retry budget
runs out. The query binds its :class:`ReaderStop` (:func:`reader_stop_scope`)
and its reader sets it; a retry loop checks it before each attempt and
races its wait against it. A stop raises ``asyncio.CancelledError`` where
nothing is held (between attempts), which the query ends as a stopped
reader's, not as a failure.

Outside such a query no stop is bound and the waits are plain sleeps.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager


class ReaderStop:
    """Set once the reader of a query stopped reading its events; wakes the
    backoff waits of that query, in any thread or event loop."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._waiters: set[asyncio.Future[None]] = set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self) -> None:
        self._event.set()
        with self._lock:
            waiters, self._waiters = self._waiters, set()
        for waiter in waiters:
            waiter.get_loop().call_soon_threadsafe(_wake, waiter)

    def _wait(self, timeout: float) -> None:
        self._event.wait(timeout)

    async def _await(self, delay: float) -> None:
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        with self._lock:
            self._waiters.add(waiter)
        if self.is_set():
            _wake(waiter)
        slept = asyncio.ensure_future(asyncio.sleep(delay))
        try:
            await asyncio.wait({slept, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            with self._lock:
                self._waiters.discard(waiter)
            if not slept.done():
                slept.cancel()
                await asyncio.gather(slept, return_exceptions=True)


def _wake(waiter: asyncio.Future[None]) -> None:
    if not waiter.done():
        waiter.set_result(None)


_READER_STOP: contextvars.ContextVar[ReaderStop | None] = contextvars.ContextVar(
    "kg_rag_reader_stop", default=None
)


@contextmanager
def reader_stop_scope(stop: ReaderStop) -> Iterator[ReaderStop]:
    """Bind ``stop`` as the stop of the query running in this context (and
    of the tasks and threads that copy it)."""
    token = _READER_STOP.set(stop)
    try:
        yield stop
    finally:
        _READER_STOP.reset(token)


def reader_stopped() -> bool:
    """Whether the reader of the query running in this context stopped."""
    stop = _READER_STOP.get()
    return stop is not None and stop.is_set()


def raise_if_reader_stopped() -> None:
    """Raise ``asyncio.CancelledError`` once the query's reader stopped: a
    stop point before a call."""
    if reader_stopped():
        raise asyncio.CancelledError


def backoff_wait(delay: float, sleep: Callable[[float], object] | None = None) -> None:
    """Wait ``delay`` s before a retry, ending early with
    ``asyncio.CancelledError`` when the query's reader stops (before or
    during the wait); ``sleep`` (``time.sleep``) without a stop bound."""
    stop = _READER_STOP.get()
    if stop is None:
        (sleep or time.sleep)(delay)
        return
    raise_if_reader_stopped()
    stop._wait(delay)
    raise_if_reader_stopped()


async def abackoff_wait(delay: float) -> None:
    """Async twin of :func:`backoff_wait` (``asyncio.sleep`` without a stop
    bound)."""
    stop = _READER_STOP.get()
    if stop is None:
        await asyncio.sleep(delay)
        return
    raise_if_reader_stopped()
    await stop._await(delay)
    raise_if_reader_stopped()
