# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event-loop setup for the blocking I/O on the query path."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

_T = TypeVar("_T")


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
            # asyncio.run shuts down (and joins) the default executor then:
            # give it an unused one.
            loop.set_default_executor(ThreadPoolExecutor(max_workers=1))
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    return asyncio.run(_main())
