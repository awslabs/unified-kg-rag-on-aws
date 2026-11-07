# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thread pools that keep the submitter's ``contextvars``, process pools that
never fork a threaded process, and asyncio tasks cancelled with the block
that awaits them."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import inspect
import multiprocessing
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Iterator,
    Sequence,
)
from concurrent.futures import (
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
)
from contextlib import asynccontextmanager, contextmanager
from types import TracebackType
from typing import Any, Literal, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")
_E = TypeVar("_E", bound=Executor)


def is_stop(exc_type: type[BaseException] | None) -> bool:
    """Whether a block that raised ``exc_type`` ended on a stop of the
    process (SIGTERM's ``PipelineInterrupted``, Ctrl-C, ``SystemExit``), the
    one ending that must not wait for work in flight. Any other
    ``BaseException`` (a closed generator, a test's simulated death) is
    treated as an error: in production a run dies by being killed, without
    an exception."""
    return exc_type is not None and issubclass(
        exc_type, (KeyboardInterrupt, SystemExit)
    )


@asynccontextmanager
async def cancel_on_exit(tasks: Sequence[asyncio.Future[Any]]) -> AsyncIterator[None]:
    """Cancel ``tasks`` and wait for them when the block raises.

    For blocks that await tasks through ``asyncio.as_completed`` (or tqdm's
    gather, built on it) or that stop at the first error: unlike a plain
    ``asyncio.gather``, those leave the tasks running when the awaiting task
    is cancelled (Ctrl-C, a timeout, a host cancelling the call), so they
    kept spending and their spans never ended.
    """
    try:
        yield
    except BaseException:
        await discard(tasks)
        raise


async def gather_or_cancel(aws: Iterable[Awaitable[_T]]) -> list[_T]:
    """``asyncio.gather`` of ``aws`` (results in order) that cancels the
    others, and waits for them, once one raises (or the caller is cancelled).

    A plain gather raises the first error and leaves its siblings running:
    their store requests and LLM calls kept spending after the query had
    already failed.
    """
    tasks = [asyncio.ensure_future(aw) for aw in aws]
    async with cancel_on_exit(tasks):
        return list(await asyncio.gather(*tasks))


async def gather_bounded(limit: int, aws: Iterable[Awaitable[_T]]) -> list[_T]:
    """``asyncio.gather`` of ``aws`` (results in order), at most ``limit``
    of them awaited at once: the batches of one search, so a long list does
    not put hundreds of requests on a store at the same time.

    When one raises (or the caller is cancelled) the others are cancelled
    and waited for before the error propagates: a plain ``gather`` left the
    queued ones going to the store after the search had already failed. An
    awaitable cancelled before it started is closed, not left unawaited."""
    semaphore = asyncio.Semaphore(max(1, limit))
    pending = list(aws)

    async def one(aw: Awaitable[_T]) -> _T:
        async with semaphore:
            return await aw

    tasks = [asyncio.ensure_future(one(aw)) for aw in pending]
    try:
        async with cancel_on_exit(tasks):
            return list(await asyncio.gather(*tasks))
    finally:
        for aw in pending:
            if inspect.iscoroutine(aw) and (
                inspect.getcoroutinestate(aw) == inspect.CORO_CREATED
            ):
                aw.close()


async def discard(tasks: Sequence[asyncio.Future[Any]]) -> None:
    """Cancel ``tasks`` and wait for them, ignoring how they ended."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


async def settle(tasks: Sequence[asyncio.Future[Any]], within: float | None) -> None:
    """Wait for ``tasks`` to finish, at most ``within`` seconds (``None``: no
    limit), then :func:`discard` the ones still running, ignoring how any
    ended."""
    if tasks:
        await asyncio.wait(tasks, timeout=within)
    await discard(tasks)


# The modules the forkserver imports when it starts (process_pool's
# ``preload`` and preload_in_forkserver's), so each worker it forks has them
# already.
_FORKSERVER_PRELOAD: list[str] = []


def preload_in_forkserver(*modules: str) -> None:
    """Have the forkserver import ``modules`` when it starts, whichever
    pool starts it: a module whose tasks run in a :func:`process_pool`
    registers itself when it is imported, before any pool starts. The
    server starts with the first pool of the process and reads the list
    once, so a pool whose modules only its own ``preload`` named imports
    them per worker when an earlier pool started the server."""
    _FORKSERVER_PRELOAD.extend(m for m in modules if m not in _FORKSERVER_PRELOAD)


def process_pool(
    max_workers: int | None,
    *,
    initializer: Callable[..., object] | None = None,
    initargs: tuple[Any, ...] = (),
    preload: Sequence[str] = (),
) -> ProcessPoolExecutor:
    """A ``ProcessPoolExecutor`` whose workers start from the ``forkserver``
    (``spawn`` where the platform has none, as on Windows), never by
    forking this process.

    A run is multi-threaded by the time it starts a pool (the event loop's
    I/O threads, the store and model clients' connection pools, logging),
    and ``fork`` copies only the calling thread: a lock another thread held
    stays held in the child, which then hangs (Python 3.12 warns of it, and
    3.14 no longer forks by default on Linux). A forkserver worker starts
    from a clean single-threaded process instead. So what crosses into it
    is pickled: the task callable and its arguments, and ``initializer``
    and ``initargs``, must be picklable (module-level functions, not
    lambdas or closures). A worker imports the main module as
    ``__mp_main__``: a script that runs ingestion must keep its entry
    behind ``if __name__ == "__main__":`` (the console scripts do).

    ``preload`` names the modules the tasks live in: the forkserver imports
    them once, when it starts (the first pool of the process), with those
    :func:`preload_in_forkserver` registered, and every worker forked from
    it has them, so a pool does not pay their import per worker (seconds
    for the ingestion modules).
    """
    method = (
        "forkserver"
        if "forkserver" in multiprocessing.get_all_start_methods()
        else "spawn"
    )
    context = multiprocessing.get_context(method)
    if method == "forkserver":
        # Read when the server starts; a pool after that finds its modules
        # there if they were registered by then, else imports them per worker.
        preload_in_forkserver(*preload)
        if _FORKSERVER_PRELOAD:
            context.set_forkserver_preload(list(_FORKSERVER_PRELOAD))
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=initializer,
        initargs=initargs,
    )


@contextmanager
def shut_down_after(executor: _E) -> Iterator[_E]:
    """Use ``executor`` for the enclosed block, then shut it down: waiting
    for its tasks when the block ends normally, and cancelling the queued
    ones without waiting for the running ones when it raises (a stop,
    Ctrl-C, an error).

    For a ``ProcessPoolExecutor``, used instead of its own ``with`` block:
    that block's exit shuts it down with ``wait=True``, which runs every
    queued task first, and resets an earlier ``cancel_futures=True``, so
    the queued tasks also hold up the interpreter's exit. The futures the
    block submitted are cancelled here, one by one: ``cancel_futures`` only
    sets a flag the pool's manager thread reads through a weak reference to
    the executor, so an executor collected first (the caller held it in a
    local) cancelled nothing. Unlike ``ContextThreadPoolExecutor``'s block,
    it does not wait for running tasks on an error either: process-pool
    tasks write nothing.
    """
    submitted: list[Future[Any]] = []
    submit = executor.submit

    def tracked(fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        future = submit(fn, *args, **kwargs)
        submitted.append(future)
        return future

    executor.submit = tracked  # type: ignore[method-assign]
    try:
        yield executor
    except BaseException:
        for future in submitted:
            future.cancel()  # a no-op once it runs
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        del executor.submit
    executor.shutdown()


class ContextThreadPoolExecutor(ThreadPoolExecutor):
    """A ``ThreadPoolExecutor`` whose tasks run in a copy of the caller's context.

    A plain pool thread starts from an empty context, so the log context bound
    with ``structlog.contextvars`` (``pipeline_id``, ``stage``, ``query_id``)
    and any trace context are missing from what the task logs or calls. Each
    ``submit`` (and so ``map``) copies the submitting thread's context, as
    ``asyncio.to_thread`` does.

    Process pools stay plain ``ProcessPoolExecutor``s: a ``Context`` cannot be
    pickled to another process, and a worker process has its own logging and
    contextvars anyway.

    A ``with`` block that raises cancels the queued tasks. On a stop (a
    ``BaseException`` that is not an ``Exception``: ``PipelineInterrupted``,
    Ctrl-C, ``SystemExit``) it does not wait for the running ones: they end
    at their next write fence check (``shared.write_fence``), so a stop is
    not held up until every task of the block finished, and the process
    ends. On an error (a task's error re-raised, a lost write lease) it
    waits for them, so a write a running task already sent has landed when
    the error reaches the caller, who may run again in this process (the
    next run's writes would otherwise race it).
    """

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.shutdown(wait=not is_stop(exc_type), cancel_futures=exc_type is not None)
        return False

    def submit(
        self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> Future[_T]:
        context = contextvars.copy_context()
        return super().submit(context.run, functools.partial(fn, *args, **kwargs))
