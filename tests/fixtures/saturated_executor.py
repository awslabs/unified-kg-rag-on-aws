# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A default executor whose every thread is inside a blocking call, as under
load, so a test can tell a compute-only step that runs on the CPU pool
(``to_cpu_thread``) from one queued behind the blocking calls."""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import AsyncIterator

from unified_kg_rag.shared.utils.event_loop import configure_event_loop

CPU_THREAD_PREFIX = "graphrag-cpu"


class ThreadSignal:
    """An ``asyncio.Event`` of the running loop that any thread may set."""

    def __init__(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._event = asyncio.Event()

    def set(self) -> None:
        self._loop.call_soon_threadsafe(self._event.set)

    async def wait(self, seconds: float = 5) -> None:
        await asyncio.wait_for(self._event.wait(), seconds)


@contextlib.asynccontextmanager
async def saturated_default_executor(workers: int = 2) -> AsyncIterator[None]:
    """Run the body with the running loop's default executor (``workers``
    threads) held by blocking calls; a call queued there starts only after
    the body ends. A query's fusion and rerank stay there (a service call),
    so a body that runs a whole query starts it, waits for its compute step
    (a ``ThreadSignal``) and awaits the query after the block."""
    loop = asyncio.get_running_loop()
    executor = configure_event_loop(loop, workers)
    release = threading.Event()
    holding: list[int] = []
    lock = threading.Lock()
    started = ThreadSignal()

    def hold() -> None:
        with lock:
            holding.append(threading.get_ident())
            if len(holding) == workers:
                started.set()
        release.wait(timeout=30)

    held = [loop.run_in_executor(None, hold) for _ in range(workers)]
    try:
        await started.wait(10)
        yield
    finally:
        release.set()
        await asyncio.gather(*held)
        executor.shutdown(wait=False)
