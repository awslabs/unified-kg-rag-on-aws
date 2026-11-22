# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The stop of a query: its end, requested by its reader or its deadline.

A query's end is requested two ways, through one :class:`QueryStop`: the
reader of a query streamed to it (``GraphRAGChain.aquery_events``, a task
of its own) stops reading its events, or the query's deadline (or its
answer stream's first-chunk window) passes. Either sets the stop, then
cancels the query's task; the stop keeps the :class:`StopCause` that set it
first, so the query ends as a stopped reader's (no failure) or as a timed
out one (``QueryTimeoutError``). The stop reaches what the cancellation
does not: the work the query left on worker threads (a retry loop there
checks it before each attempt and races its backoff wait against it,
:func:`backoff_wait`, and the model clients that resend a call themselves
check it before each request), and a cancellation a layer under the query
loses (the query's next stop point raises), so a stopped query sends no
model request. A stop raises ``asyncio.CancelledError`` where nothing is
held (between attempts).

A host that reads a query's result for a client (an MCP tool call, whose
client may leave) runs it as its reader (:func:`await_as_reader`): the
host's cancellation is then its reader's stop, as for a streamed query,
not a cancelled query's (a process stopping, ``Ctrl-C``), and the spans it
ends say so (no error status).

The reader's stop is bound for the whole query (:func:`query_stop_scope`);
each deadline window binds a :class:`FollowingStop` of it (or a stop of its
own without a reader), set by the reader's stop or by the deadline. Outside
both no stop is bound and the waits are plain sleeps.

A stop ends only the work its query owns. A call several queries share (an
in-flight cache, ``shared.utils.concurrency.single_flight``) is each
waiter's: it runs in a :class:`SharedContext`, under a :class:`SharedStop`
of its own set once every waiter's reader stopped, and a waiter whose query
stops leaves it (:func:`await_unless_stopped`), so one query's stop never
ends another's query (a waiter whose deadline passed leaves it running, as
a cancelled one does). The other per-query state such a call reads is taken
from all its waiters the same way (:func:`share_with_waiters`), or it starts
without it (:func:`blank_in_shared_calls`), never from its first caller
alone.

Work a query starts that may outlive it (``shared.query_work``) is stopped
by its query through a :class:`FollowingStop` (:func:`detachable_context`)
only until the query keeps it past its end: then the query's stop is no
longer its.
"""

from __future__ import annotations

import asyncio
import contextvars
import enum
import threading
import time
from collections.abc import Callable, Coroutine, Hashable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, Protocol, TypeVar

_T = TypeVar("_T")


class StopCause(enum.Enum):
    """What requested a query's end (:attr:`QueryStop.cause`)."""

    # Its reader stopped reading its events: the query ends with no failure.
    READER = "reader"
    # Its deadline, or its answer stream's first-chunk window, passed: the
    # query ends with its QueryTimeoutError.
    DEADLINE = "deadline"


class QueryStop:
    """Set once a query's end was requested (its reader stopped, or its
    deadline passed: :attr:`cause`); wakes the backoff waits of that query,
    in any thread or event loop."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._waiters: set[asyncio.Future[None]] = set()
        self._followers: set[FollowingStop] = set()
        # What set the stop first (None: not set).
        self.cause: StopCause | None = None

    def is_set(self) -> bool:
        return self._event.is_set()

    def set(self, cause: StopCause = StopCause.READER) -> None:
        """Request the query's end; a stop set already keeps its cause."""
        with self._lock:
            if self.cause is None:
                self.cause = cause
            first = self.cause
            self._event.set()
            waiters, self._waiters = self._waiters, set()
            followers, self._followers = self._followers, set()
        for waiter in waiters:
            waiter.get_loop().call_soon_threadsafe(_wake, waiter)
        for follower in followers:
            follower.set(first)

    def _follow(self, follower: FollowingStop) -> None:
        """Set ``follower`` when this stop is set (at once if it is), with
        this stop's cause."""
        with self._lock:
            if not self._event.is_set():
                self._followers.add(follower)
                return
        follower.set(self.cause or StopCause.READER)

    def _unfollow(self, follower: FollowingStop) -> None:
        with self._lock:
            self._followers.discard(follower)

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


class SharedStop(QueryStop):
    """The stop of a call several queries share, set once the reader of
    each query waiting for it stopped (:meth:`leave`). A waiter with no
    stop bound (no streamed query's), or one cancelled otherwise or past its
    deadline, keeps it running: its call is still wanted, or was before. Counted on the event
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


class FollowingStop(QueryStop):
    """A stop set when another (``leader``) is, with its cause, until
    :meth:`detach`, and also on its own: a deadline window's (set by its
    deadline too), or that of work a query started that may outlive it
    (:func:`detachable_context`)."""

    def __init__(self, leader: QueryStop) -> None:
        super().__init__()
        self._leader = leader
        leader._follow(self)

    def detach(self) -> None:
        """The work is no longer its query's: the query's stop no longer
        stops it (a stop already set stays set)."""
        self._leader._unfollow(self)


def _wake(waiter: asyncio.Future[None]) -> None:
    if not waiter.done():
        waiter.set_result(None)


_QUERY_STOP: contextvars.ContextVar[QueryStop | None] = contextvars.ContextVar(
    "kg_rag_query_stop", default=None
)


@contextmanager
def query_stop_scope(stop: QueryStop) -> Iterator[QueryStop]:
    """Bind ``stop`` as the stop of the query running in this context (and
    of the tasks and threads that copy it)."""
    token = _QUERY_STOP.set(stop)
    try:
        yield stop
    finally:
        _QUERY_STOP.reset(token)


def window_stop_scope() -> AbstractContextManager[QueryStop]:
    """Bind the stop of a deadline window of the query running in this
    context: a :class:`FollowingStop` of the query's stop (its reader's),
    else a stop of its own. The window sets it (``StopCause.DEADLINE``) when
    its deadline passes, so every stop point under it ends the query then,
    as at its reader's stop, a cancellation lost under it included."""
    leader = _QUERY_STOP.get()
    return query_stop_scope(QueryStop() if leader is None else FollowingStop(leader))


# The context variables of per-query state a call several queries share
# starts without (blank_in_shared_calls), each with its default.
_BLANKED: dict[contextvars.ContextVar[Any], Any] = {}


def blank_in_shared_calls(*variables: contextvars.ContextVar[Any]) -> None:
    """Have each call several queries share start with ``variables`` at
    their defaults instead of its first caller's values: per-query state
    (its events' reader, its open model responses, its memo and ledgers)
    that is no other waiter's. The module that owns a variable registers it
    when it is imported; state the call must take from all its waiters is
    registered with :func:`share_with_waiters` instead."""
    for variable in variables:
        _BLANKED[variable] = contextvars.Context().run(variable.get)


def detached_context(stop: QueryStop | None = None) -> contextvars.Context:
    """A copy of the current context with ``stop`` bound in place of the
    reader's (``None``: no stop) and the per-query state registered with
    :func:`blank_in_shared_calls` at its defaults, for work that is not this
    reader's alone (``loop.create_task(..., context=...)``). The rest of the
    context (log and trace ids, the usage meter) is kept."""
    context = contextvars.copy_context()

    def detach() -> None:
        _QUERY_STOP.set(stop)
        for variable, default in _BLANKED.items():
            variable.set(default)

    context.run(detach)
    return context


def detachable_context() -> tuple[contextvars.Context, Callable[[], None]]:
    """A copy of the current context for work this query starts that may
    outlive it, and the function that detaches the work from the query.

    Until it is detached the work stops when the query stops (a
    :class:`FollowingStop` of the query's stop is bound in its place), as
    the query's own work does; detached (the query kept it running past its
    end), the query's stop no longer reaches it. The rest of the context
    is the query's. Without a stop bound there is nothing to detach."""
    context = contextvars.copy_context()
    leader = _QUERY_STOP.get()
    if leader is None:
        return context, _nothing
    stop = FollowingStop(leader)
    context.run(_QUERY_STOP.set, stop)
    return context, stop.detach


def _nothing() -> None:
    return None


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
    """Wait for ``future`` (work this query shares with others) to finish,
    ending with ``asyncio.CancelledError`` once this query stops (its
    reader's stop or its deadline); ``future`` runs on, never cancelled here
    (as under ``asyncio.shield``). Without a stop bound, a plain wait."""
    stop = _QUERY_STOP.get()
    if stop is None:
        await asyncio.wait({future})
        return
    await stop._until(future)
    if not future.done():
        raise asyncio.CancelledError


def query_stopped() -> bool:
    """Whether the end of the query running in this context was requested
    (its reader stopped, or its deadline passed)."""
    stop = _QUERY_STOP.get()
    return stop is not None and stop.is_set()


def reader_stopped() -> bool:
    """Whether the query running in this context stopped because its reader
    did (no failure), not at its deadline."""
    stop = _QUERY_STOP.get()
    return stop is not None and stop.is_set() and stop.cause is StopCause.READER


def raise_if_stopped() -> None:
    """Raise ``asyncio.CancelledError`` once the query's end was requested
    (:func:`query_stopped`): a stop point before a call."""
    if query_stopped():
        raise asyncio.CancelledError


def backoff_wait(delay: float, sleep: Callable[[float], object] | None = None) -> None:
    """Wait ``delay`` s before a retry, ending early with
    ``asyncio.CancelledError`` when the query stops (before or during the
    wait); ``sleep`` (``time.sleep``) without a stop bound."""
    stop = _QUERY_STOP.get()
    if stop is None:
        (sleep or time.sleep)(delay)
        return
    # Set already, it returns at once.
    stop._wait(delay)
    raise_if_stopped()


async def abackoff_wait(delay: float) -> None:
    """Async twin of :func:`backoff_wait` (``asyncio.sleep`` without a stop
    bound)."""
    stop = _QUERY_STOP.get()
    if stop is None:
        await asyncio.sleep(delay)
        return
    # Set already, it returns at once.
    await stop._await(delay)
    raise_if_stopped()


async def await_as_reader(
    work: Coroutine[Any, Any, _T], *, until: float | None = None
) -> _T:
    """Run ``work`` (a query) in a task of its own, under a stop of its
    own, and return its result, this caller its reader.

    This caller cancelled (its client left) before ``until``
    (``time.monotonic()``: the work's deadline; ``None``: none), the stop is
    set (``StopCause.READER``) before ``work`` is cancelled, so ``work``
    ends as a stopped reader's (no failure, in its records and its spans),
    as a streamed query whose reader stops (``aquery_events``); cancelled
    past ``until`` (a host's backstop for its deadline), ``work`` is
    cancelled with no stop set, a timed-out query's. The cancellation is
    re-raised once ``work`` ended (:func:`wait_ended`)."""
    stop = QueryStop()
    context = contextvars.copy_context()
    context.run(_QUERY_STOP.set, stop)
    task = asyncio.get_running_loop().create_task(work, context=context)
    try:
        await asyncio.wait({task})
    except asyncio.CancelledError:
        if until is None or time.monotonic() < until:
            stop.set(StopCause.READER)
        task.cancel()
        await wait_ended(task)
        raise
    return task.result()


async def wait_ended(task: asyncio.Task[Any]) -> None:
    """Wait for ``task`` (a query its reader cancelled) to end, never
    cancelling it again: a waiter cancelled again while it waits (an anyio
    cancel scope cancels at every await) would pass that on to it through
    ``asyncio.gather``, in the middle of its unwinding, where it cuts the
    close of a model response short and leaves its connection held. Its
    outcome is retrieved, so none is logged as never retrieved.

    The query still has stop points, for a cancellation a layer under it
    loses, on any Python: anyio's ``connect_tcp`` (httpcore's connect) can
    return a connected stream to a task cancelled while it connected, and
    on 3.11 ``asyncio.wait_for`` returns the result its awaitable got in
    the same turn as the cancellation (langchain-openai reads each chunk of
    a stream through it, for ``stream_chunk_timeout``; 3.12 rebuilt it on
    ``asyncio.timeout``, which re-raises it), so the call goes on as if
    nothing was cancelled. The query's next event, answer chunk, model
    request (before its connection and again before it is written, its
    response as its headers arrive) or retry wait then raises
    ``CancelledError`` at the query's stop, set by its reader before the
    cancellation, as by its deadline (a deadline window's stop)."""
    try:
        await asyncio.wait({task})
    finally:
        if task.done():
            _retrieved(task)
        else:
            task.add_done_callback(_retrieved)


def _retrieved(task: asyncio.Future[Any]) -> None:
    if not task.cancelled():
        task.exception()
