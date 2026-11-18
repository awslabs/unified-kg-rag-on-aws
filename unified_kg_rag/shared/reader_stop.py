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

A stop ends only the work its reader owns. A call several queries share (an
in-flight cache, ``shared.utils.concurrency.single_flight``) is each
waiter's: it runs in a :class:`SharedContext`, under a :class:`SharedStop`
of its own set once every waiter's reader stopped, and a waiter whose reader
stops leaves it (:func:`await_unless_stopped`), so one reader's stop never
ends another's query. The other per-query state such a call reads is taken
from all its waiters the same way (:func:`share_with_waiters`), never from
its first caller alone.
"""

from __future__ import annotations

import asyncio
import contextvars
import threading
import time
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from typing import Any, Protocol, TypeVar

_T = TypeVar("_T")


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

    async def _until(self, future: asyncio.Future[Any]) -> None:
        """Wait until ``future`` is done or this stop is set, leaving
        ``future`` running."""
        waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        with self._lock:
            self._waiters.add(waiter)
        if self.is_set():
            _wake(waiter)
        try:
            await asyncio.wait({future, waiter}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            with self._lock:
                self._waiters.discard(waiter)

    async def _await(self, delay: float) -> None:
        slept = asyncio.ensure_future(asyncio.sleep(delay))
        try:
            await self._until(slept)
        finally:
            if not slept.done():
                slept.cancel()
                await asyncio.gather(slept, return_exceptions=True)


class SharedStop(ReaderStop):
    """The stop of a call several queries share, set once the reader of
    each query waiting for it stopped (:meth:`leave`). A waiter with no
    stop bound (no streamed query's), or one cancelled otherwise, keeps it
    running: its call is still wanted, or was before. Counted on the event
    loop the call runs on."""

    def __init__(self) -> None:
        super().__init__()
        self._holders = 0

    def join(self) -> None:
        """A waiter came for the call."""
        self._holders += 1

    def leave(self) -> bool:
        """A waiter's reader stopped: the call is no longer its. Whether
        that was the last one's (the call stops)."""
        self._holders -= 1
        if self._holders > 0:
            return False
        self.set()
        return True


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


def detached_context(stop: ReaderStop | None = None) -> contextvars.Context:
    """A copy of the current context with ``stop`` bound in place of the
    reader's (``None``: no stop), for work that is not this reader's alone
    (``loop.create_task(..., context=...)``). The rest of the context (log
    and trace ids, the usage meter) is kept."""
    context = contextvars.copy_context()
    context.run(_READER_STOP.set, stop)
    return context


class WaiterState(Protocol):
    """One piece of per-query context state (a context variable a query
    binds: its retry deadline, its usage meter) as a call several queries
    share takes it from its waiters (:class:`SharedContext`)."""

    def bind(self) -> None:
        """Bind the call's value of the state; run in the call's context,
        before any waiter joined."""

    def join(self) -> Hashable:
        """A waiter came for the call; run in its context, so it reads the
        waiter's value. Returns the waiter's seat for :meth:`leave`."""

    def leave(self, seat: Hashable, *, stopped: bool) -> None:
        """The waiter of ``seat`` stopped waiting for the call: its reader
        stopped (``stopped``), or it was answered, failed or cancelled
        otherwise."""


# The factories of the per-query state a shared call takes from its
# waiters (share_with_waiters), one new state per call.
_WAITER_STATES: list[Callable[[], WaiterState]] = []


def share_with_waiters(state: Callable[[], WaiterState]) -> None:
    """Have each call several queries share take a piece of per-query
    context state from all its waiters through a new ``state()``
    (:class:`WaiterState`) instead of from its first caller's context: the
    module that owns the context variable registers it when it is
    imported."""
    if state not in _WAITER_STATES:
        _WAITER_STATES.append(state)


class SharedContext:
    """The context a call several queries share runs in
    (``single_flight(share_failure=True)``): a copy of its first caller's
    (log and trace ids: the call is logged and traced where it was sent
    from) with the per-query state of all its waiters in place of that
    caller's own: a :class:`SharedStop` set once every waiter's reader
    stopped, and each state registered with :func:`share_with_waiters`.
    Created, joined and left on the event loop the call runs on."""

    def __init__(self) -> None:
        self.stop = SharedStop()
        self._states = [state() for state in _WAITER_STATES]
        self.context = detached_context(self.stop)
        for state in self._states:
            self.context.run(state.bind)

    def join(self) -> list[Hashable]:
        """A waiter came for the call (run in its context); its seats."""
        self.stop.join()
        return [state.join() for state in self._states]

    def leave(self, seats: list[Hashable], *, stopped: bool) -> bool:
        """The waiter of ``seats`` stopped waiting: its reader stopped
        (``stopped``), or it ended otherwise. Whether that stopped the call
        (the last reader's stop: :meth:`SharedStop.leave`)."""
        for state, seat in zip(self._states, seats, strict=True):
            state.leave(seat, stopped=stopped)
        return stopped and self.stop.leave()


async def await_unless_stopped(future: asyncio.Future[_T]) -> None:
    """Wait for ``future`` (work this reader shares with others) to finish,
    ending with ``asyncio.CancelledError`` once this query's reader stops;
    ``future`` runs on, never cancelled here (as under ``asyncio.shield``).
    Without a stop bound, a plain wait."""
    stop = _READER_STOP.get()
    if stop is None:
        await asyncio.wait({future})
        return
    raise_if_reader_stopped()
    await stop._until(future)
    if not future.done():
        raise asyncio.CancelledError


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
