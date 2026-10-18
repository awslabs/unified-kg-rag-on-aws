# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thread pools that keep the submitter's ``contextvars``."""

from __future__ import annotations

import contextvars
import functools
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextlib import contextmanager
from types import TracebackType
from typing import Literal, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")


@contextmanager
def cancelling_on_error(executor: Executor) -> Iterator[None]:
    """Cancel ``executor``'s queued tasks, without waiting for the running
    ones, when the enclosed block raises (a stop, Ctrl-C, an error).

    For the ``with`` block of a ``ProcessPoolExecutor``, whose exit otherwise
    runs every queued task first; ``ContextThreadPoolExecutor`` does this
    itself.
    """
    try:
        yield
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise


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

    A ``with`` block that raises (a task's error re-raised, a lost write lease,
    or ``PipelineInterrupted`` from a stop request) cancels the queued tasks
    and does not wait for the running ones: they end at their next write
    fence check (``shared.write_fence``), so a stop is not held up until
    every task of the block finished.
    """

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.shutdown(wait=exc_type is None, cancel_futures=exc_type is not None)
        return False

    def submit(
        self, fn: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
    ) -> Future[_T]:
        context = contextvars.copy_context()
        return super().submit(context.run, functools.partial(fn, *args, **kwargs))
