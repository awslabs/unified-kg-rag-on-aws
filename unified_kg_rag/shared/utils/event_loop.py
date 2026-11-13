# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event-loop setup for the blocking I/O on the query path."""

from __future__ import annotations

import asyncio
import contextvars
import functools
import os
import threading
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ParamSpec, TypeVar

_T = TypeVar("_T")
_P = ParamSpec("_P")

_cpu_executor: ThreadPoolExecutor | None = None
_cpu_executor_lock = threading.Lock()


def configure_event_loop(
    loop: asyncio.AbstractEventLoop, io_workers: int
) -> ThreadPoolExecutor:
    """Size ``loop``'s default executor for blocking I/O.

    LangChain runs a chat model's ``ainvoke`` in the loop's default executor
    when the model has no native async path, as ``ChatBedrockConverse`` does
    not; Neptune traversals and reranking are offloaded there too. Python sizes
    that executor ``min(32, CPUs + 4)``, six threads on a 2-vCPU task, so no
    more than six Bedrock calls are in flight however many queries run
    concurrently. Call this once per loop before the first query, e.g. in an
    async web server's startup hook. Returns the executor.
    """
    executor = ThreadPoolExecutor(
        max_workers=io_workers, thread_name_prefix="graphrag-io"
    )
    loop.set_default_executor(executor)
    return executor


def _cpu_pool() -> ThreadPoolExecutor:
    global _cpu_executor
    with _cpu_executor_lock:
        if _cpu_executor is None:
            _cpu_executor = ThreadPoolExecutor(
                max_workers=os.cpu_count() or 1, thread_name_prefix="graphrag-cpu"
            )
        return _cpu_executor


async def to_cpu_thread(
    func: Callable[_P, _T], /, *args: _P.args, **kwargs: _P.kwargs
) -> _T:
    """``asyncio.to_thread`` for a step that only computes (parsing hits,
    fusing, citation lineages, PPR): it runs on a pool of its own, one thread
    per CPU, and with the caller's context.

    The default executor (:func:`configure_event_loop`) holds the blocking
    model, store and rerank calls, seconds each: under load every one of its
    threads can be inside such a call, and a step queued behind them waits
    for the slowest although it needs a millisecond of CPU. A step that
    calls a service stays on ``asyncio.to_thread``: a slow call would hold a
    CPU thread. Measured with 64 blocked default-executor threads, a query's
    p50 fell by 0.4 to 0.5 s of about 2 s (docs/design/aws-integration.md),
    and did not change below saturation."""
    context = contextvars.copy_context()
    call = functools.partial(context.run, func, *args, **kwargs)
    return await asyncio.get_running_loop().run_in_executor(_cpu_pool(), call)


def abandon_default_executor(
    loop: asyncio.AbstractEventLoop, executor: ThreadPoolExecutor
) -> None:
    """Leave ``executor``'s running threads to the process's exit.

    ``asyncio.run`` shuts down (and joins) the loop's default executor on the
    way out, so a model call the caller gave up on (a stop, a timed-out
    query) held the process until the call's own timeout. The loop gets an
    unused executor instead; ``executor``'s queued calls are dropped.
    """
    loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
    executor.shutdown(wait=False, cancel_futures=True)


def run(main: Coroutine[Any, Any, _T], *, io_workers: int) -> _T:
    """``asyncio.run`` with the default executor sized by :func:`configure_event_loop`.

    A cancelled ``main`` (a stopped CLI) does not wait for the executor's
    threads: ``asyncio.run`` joins them on the way out, so a stop during a
    model call waited for the call. Its queued calls are dropped and the
    running ones left to the process's exit.
    """

    async def _main() -> _T:
        loop = asyncio.get_running_loop()
        executor = configure_event_loop(loop, io_workers)
        try:
            return await main
        except asyncio.CancelledError:
            abandon_default_executor(loop, executor)
            raise

    return asyncio.run(_main())
