# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The query path's default-executor sizing (AWS-free)."""

from __future__ import annotations

import asyncio
import threading

import pytest

from unified_kg_rag.application.retrieval.rag_chain import _LoopRunner
from unified_kg_rag.shared.utils import event_loop

pytestmark = pytest.mark.unit

# Above Python's own ceiling for the default executor (min(32, CPUs + 4)), so
# the barrier is only reached when the configured size is in effect.
_WORKERS = 40


async def _fill_executor() -> set[str]:
    """Block ``_WORKERS`` default-executor calls until all of them run at once."""
    barrier = threading.Barrier(_WORKERS, timeout=10)

    def _wait() -> str:
        barrier.wait()
        return threading.current_thread().name

    loop = asyncio.get_running_loop()
    names = await asyncio.gather(
        *(loop.run_in_executor(None, _wait) for _ in range(_WORKERS))
    )
    return set(names)


def test_run_sizes_the_default_executor() -> None:
    names = event_loop.run(_fill_executor(), io_workers=_WORKERS)
    assert len(names) == _WORKERS
    assert all(name.startswith("graphrag-io") for name in names)


def test_chain_loop_runner_uses_the_configured_size() -> None:
    runner = _LoopRunner(io_workers=_WORKERS)
    try:
        assert len(runner.run(_fill_executor())) == _WORKERS
    finally:
        runner.stop()


def test_a_cancelled_run_does_not_wait_for_the_executors_calls() -> None:
    # asyncio.run joined the executor on the way out: a stopped
    # run-prompt-tuning waited for its model call (or the stop watchdog).
    import time

    release = threading.Event()
    ran: list[int] = []

    async def main() -> None:
        loop = asyncio.get_running_loop()
        task = asyncio.current_task()
        assert task is not None
        loop.call_later(0.2, task.cancel)
        calls = [loop.run_in_executor(None, release.wait, 30) for _ in range(2)]
        calls.append(loop.run_in_executor(None, ran.append, 1))  # queued
        await asyncio.gather(*calls)

    started = time.monotonic()
    try:
        with pytest.raises(asyncio.CancelledError):
            event_loop.run(main(), io_workers=2)
        assert time.monotonic() - started < 5
        assert ran == []  # the queued call was dropped
    finally:
        release.set()
