# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``gather_bounded``: the batches of one search, a bounded number at once,
none left going to the store once the search failed."""

from __future__ import annotations

import asyncio
import gc
import warnings
from collections.abc import Generator
from typing import Any

import pytest

from unified_kg_rag.shared.utils.concurrency import gather_bounded

pytestmark = pytest.mark.unit


class _Batches:
    """Batches that take a moment (the ones numbered in ``slow``: long);
    the one numbered ``fail`` raises."""

    def __init__(
        self, fail: int | None = None, slow: frozenset[int] = frozenset()
    ) -> None:
        self.fail = fail
        self.slow = slow
        self.started: list[int] = []
        self.cancelled: list[int] = []
        self.in_flight = self.peak = 0

    async def one(self, i: int) -> int:
        self.started.append(i)
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(5 if i in self.slow else 0.005)
            if i == self.fail:
                raise RuntimeError("fatal store error")
            return i
        except asyncio.CancelledError:
            self.cancelled.append(i)
            raise
        finally:
            self.in_flight -= 1


class _Deferred:
    """An awaitable that is no coroutine: its batch starts when awaited, so
    nothing but cancelling its task keeps it from starting."""

    def __init__(self, batches: _Batches, i: int) -> None:
        self.batches, self.i = batches, i

    def __await__(self) -> Generator[Any, None, int]:
        return self.batches.one(self.i).__await__()


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


async def test_an_error_cancels_the_batches_in_flight_and_queued() -> None:
    # The tasks are cancelled, not only their unstarted coroutines closed:
    # the batch in flight stops at once, and an awaitable that is no
    # coroutine (which closing cannot stop) never starts.
    batches = _Batches(fail=0, slow=frozenset({1}))
    with pytest.raises(RuntimeError, match="fatal"):
        await asyncio.wait_for(
            gather_bounded(2, [_Deferred(batches, i) for i in range(20)]), 2
        )
    assert 1 in batches.cancelled
    sent = len(batches.started)
    await asyncio.sleep(0.05)
    assert len(batches.started) == sent <= 3
    assert batches.in_flight == 0


async def test_a_batch_cancelled_before_it_started_is_closed() -> None:
    # Cancelling a queued task leaves its coroutine unawaited; it is closed,
    # so collecting it warns of nothing.
    batches = _Batches(fail=0, slow=frozenset({1}))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="fatal"):
            await gather_bounded(2, [batches.one(i) for i in range(20)])
        await asyncio.sleep(0.05)
        gc.collect()
    assert len(batches.started) <= 3
    assert [str(w.message) for w in caught if "never awaited" in str(w.message)] == []
