# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thread pools that keep the submitter's ``contextvars``, and asyncio tasks
cancelled with the block that awaits them."""

from __future__ import annotations

import asyncio
import contextvars
import functools
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from concurrent.futures import Executor, Future, ThreadPoolExecutor
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
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


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
