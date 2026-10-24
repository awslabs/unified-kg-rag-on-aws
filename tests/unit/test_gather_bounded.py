# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``gather_bounded``: the batches of one search, a bounded number at once,
none left going to the store once the search failed."""

from __future__ import annotations

import asyncio
import gc
import warnings

import pytest

from unified_kg_rag.shared.utils.concurrency import gather_bounded

pytestmark = pytest.mark.unit


class _Batches:
    """Batches that take a moment; the one numbered ``fail`` raises."""

    def __init__(self, fail: int | None = None) -> None:
        self.fail = fail
        self.started: list[int] = []
        self.in_flight = self.peak = 0

    async def one(self, i: int) -> int:
        self.started.append(i)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.005)
            if i == self.fail:
                raise RuntimeError("fatal store error")
            return i
        finally:
            self.in_flight -= 1


async def test_results_in_order_at_most_limit_at_once() -> None:
    batches = _Batches()
    assert await gather_bounded(3, [batches.one(i) for i in range(10)]) == list(
        range(10)
    )
    assert batches.peak == 3


async def test_a_limit_below_one_runs_one_at_a_time() -> None:
    batches = _Batches()
    result = await asyncio.wait_for(
        gather_bounded(0, [batches.one(i) for i in range(3)]), 1
    )
    assert result == [0, 1, 2] and batches.peak == 1


async def test_an_error_cancels_the_queued_batches() -> None:
    # Regression: the error reached the caller while the queued batches
    # kept going to the store (20 of 20 sent after a failure at the first).
    batches = _Batches(fail=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(RuntimeError, match="fatal"):
            await gather_bounded(2, [batches.one(i) for i in range(20)])
        sent = len(batches.started)
        await asyncio.sleep(0.05)
        gc.collect()  # a batch closed unstarted: no "never awaited" warning
    assert len(batches.started) == sent <= 4
    assert batches.in_flight == 0


async def test_a_cancelled_caller_cancels_every_batch() -> None:
    batches = _Batches()
    task = asyncio.ensure_future(gather_bounded(2, [batches.one(i) for i in range(20)]))
    await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    sent = len(batches.started)
    await asyncio.sleep(0.05)
    assert len(batches.started) == sent == 2
    assert batches.in_flight == 0
