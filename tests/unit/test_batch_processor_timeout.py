# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock timeout behavior for BatchProcessor.

A hung Bedrock Converse call (open socket, no completion) is not caught by
botocore's byte-gap read_timeout and would block the single Fargate worker for
a whole stage (observed in claim_extraction). BatchProcessor runs every item
as its own call under a wall-clock timeout, so a hung item aborts and is
retried alone while the other items keep their results.

A "hung" call here blocks on an Event that the ``release_hung_calls`` fixture
sets at teardown. A timed-out call is abandoned, not cancelled, and executor
threads are joined at interpreter exit, so a plain ``time.sleep`` would keep
the test process alive for the whole sleep after the suite finished.
"""

import asyncio
import random
import threading
import time
from collections import Counter
from collections.abc import Iterator

import pytest
from langchain_core.runnables import RunnableLambda

from unified_kg_rag.adapters.aws.bedrock import BaseBedrockModelFactory
from unified_kg_rag.adapters.llm.langchain import BATCH_ITEM_FAILED, BatchProcessor

pytestmark = pytest.mark.unit


@pytest.fixture
def release_hung_calls() -> Iterator[threading.Event]:
    release = threading.Event()
    yield release
    release.set()


def test_run_with_timeout_aborts_hung_call(release_hung_calls) -> None:
    with pytest.raises(TimeoutError, match="call timeout"):
        BatchProcessor._run_with_timeout(release_hung_calls.wait, 1, "hung")


def test_run_with_timeout_reraises_a_timeout_the_call_raised(caplog) -> None:
    # TimeoutError is concurrent.futures.TimeoutError, so a
    # call that raised one (a socket timeout, say) read as the call timeout:
    # a misleading warning and a new error naming the wrong limit.
    def times_out() -> None:
        raise TimeoutError("read timed out")

    with pytest.raises(TimeoutError, match="^read timed out$"):
        BatchProcessor._run_with_timeout(times_out, 5, "own")
    assert "call timeout" not in caplog.text


def test_run_with_timeout_returns_fast_result() -> None:
    assert BatchProcessor._run_with_timeout(lambda: 42, 5, "fast") == 42


def test_run_with_timeout_zero_disables() -> None:
    # 0 means "no timeout" — run directly.
    assert BatchProcessor._run_with_timeout(lambda: "ok", 0, "nolimit") == "ok"


class _SlowFirstCall:
    """Per-item call whose first call for item ``slow`` hangs until released."""

    def __init__(self, slow: int, release: threading.Event) -> None:
        self.slow = slow
        self.release = release
        self.calls: Counter[int] = Counter()
        self._lock = threading.Lock()

    def __call__(self, item: dict[str, int]) -> dict[str, int]:
        with self._lock:
            self.calls[item["v"]] += 1
            first = self.calls[item["v"]] == 1
        if item["v"] == self.slow and first:
            self.release.wait()
        return {"echo": item["v"]}


def test_one_slow_item_times_out_alone(release_hung_calls) -> None:
    # Each item has its own timeout: the finished items keep their results
    # and only the slow one is called again.
    fake = _SlowFirstCall(slow=3, release=release_hung_calls)
    bp = BatchProcessor(
        call_timeout_seconds=1,
        retry_multiplier=1.0,
        retry_max_wait=0,
    )
    results = bp.execute_with_fallback(
        items_to_process=list(range(10)),
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=fake,
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": i} for i in range(10)]
    assert fake.calls[3] == 2
    assert sum(fake.calls.values()) == 11


def test_item_slow_on_every_attempt_fails_in_place(release_hung_calls) -> None:
    # Each of the max_attempts calls times out; only that position is
    # marked failed.
    calls: Counter[int] = Counter()
    lock = threading.Lock()

    def call(item: dict[str, int]) -> dict[str, int]:
        with lock:
            calls[item["v"]] += 1
        if item["v"] == 1:
            release_hung_calls.wait()
        return {"echo": item["v"]}

    bp = BatchProcessor(
        call_timeout_seconds=1,
        max_attempts=2,
        retry_multiplier=1.0,
        retry_max_wait=0,
    )
    results = bp.execute_with_fallback(
        items_to_process=[0, 1, 2],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": 0}, BATCH_ITEM_FAILED, {"echo": 2}]
    assert calls == Counter({0: 1, 1: 2, 2: 1})


def test_items_run_concurrently() -> None:
    # The items are in flight together, up to max_concurrency.
    barrier = threading.Barrier(3, timeout=10)

    def call(item: dict[str, int]) -> dict[str, int]:
        barrier.wait()
        return {"echo": item["v"]}

    bp = BatchProcessor(max_concurrency=3, call_timeout_seconds=0)
    results = bp.execute_with_fallback(
        items_to_process=[1, 2, 3],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": 1}, {"echo": 2}, {"echo": 3}]


def test_results_preserve_order_when_items_finish_out_of_order() -> None:
    # Each item waits for the next one to finish, so they complete strictly in
    # reverse submission order (the last item has nothing to wait for).
    bp = BatchProcessor(max_concurrency=4, call_timeout_seconds=0)
    n = 4
    done = [threading.Event() for _ in range(n)]

    def call(item: dict[str, int]) -> dict[str, int]:
        v = item["v"]
        if v + 1 < n:
            assert done[v + 1].wait(timeout=10), "items did not run concurrently"
        done[v].set()
        return {"echo": v}

    results = bp.execute_with_fallback(
        items_to_process=list(range(n)),
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": i} for i in range(n)]


class _InFlight:
    """Counts the calls in flight and the peak."""

    def __init__(self) -> None:
        self.now = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __enter__(self) -> None:
        with self._lock:
            self.now += 1
            self.peak = max(self.peak, self.now)

    def __exit__(self, *exc: object) -> None:
        with self._lock:
            self.now -= 1


def test_peak_in_flight_is_max_concurrency_under_random_latency() -> None:
    # One pool of max_concurrency slots: never more calls in flight (chunks
    # run in parallel used to reach max_concurrency x chunk_concurrency), and
    # a slot is refilled as soon as it frees up, so the peak is reached.
    rng = random.Random(0)
    latencies = [rng.uniform(0, 0.02) for _ in range(60)]
    in_flight = _InFlight()

    def call(item: dict[str, int]) -> int:
        with in_flight:
            time.sleep(latencies[item["v"]])
        return item["v"]

    bp = BatchProcessor(max_concurrency=4, call_timeout_seconds=0)
    results = bp.execute_with_fallback(
        items_to_process=list(range(60)),
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
    )
    assert results == list(range(60))
    assert in_flight.peak == 4


def test_slow_item_does_not_hold_back_the_other_slots() -> None:
    # Item 0 finishes only after every other item did: with one slot held by
    # it, the other slot must run all of them, and no extra slot opens to make
    # up for the held one (chunks used to run in parallel, each waiting for
    # its slowest item).
    others_done = threading.Event()
    finished: list[int] = []
    lock = threading.Lock()
    in_flight = _InFlight()
    n = 30

    def call(item: dict[str, int]) -> int:
        with in_flight:
            if item["v"] == 0:
                assert others_done.wait(timeout=10), "slow item blocked the others"
            else:
                time.sleep(0.002)
                with lock:
                    finished.append(item["v"])
                    if len(finished) == n - 1:
                        others_done.set()
        return item["v"]

    bp = BatchProcessor(max_concurrency=2, call_timeout_seconds=0)
    results = bp.execute_with_fallback(
        items_to_process=list(range(n)),
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
    )
    assert results == list(range(n))
    assert in_flight.peak == 2


async def test_async_slow_item_does_not_hold_back_the_other_slots() -> None:
    others_done = asyncio.Event()
    finished: list[int] = []
    in_flight = _InFlight()
    n = 30

    async def call(item: dict[str, int]) -> int:
        with in_flight:
            if item["v"] == 0:
                await asyncio.wait_for(others_done.wait(), timeout=10)
            else:
                await asyncio.sleep(0.001)
                finished.append(item["v"])
                if len(finished) == n - 1:
                    others_done.set()
        return item["v"]

    runnable = RunnableLambda(call)
    bp = BatchProcessor(max_concurrency=2, max_attempts=1)
    results = await bp.aexecute_with_fallback(
        items_to_process=list(range(n)),
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=runnable.ainvoke,
        task_name="t",
        show_progress=False,
    )
    assert results == list(range(n))
    assert in_flight.peak == 2


def test_bedrock_read_timeout_outlasts_the_call_timeout() -> None:
    # The wall-clock call timeout is the limit meant to end a long call: with
    # an equal socket read timeout the two raced, and botocore's silent
    # re-send on a read timeout was then abandoned by the call timeout.
    call_timeout = BatchProcessor.model_fields["call_timeout_seconds"].default
    assert BaseBedrockModelFactory.BOTO_READ_TIMEOUT > call_timeout


def test_cancelling_an_async_batch_cancels_its_items() -> None:
    # Before, the items kept running (and a cancelled one was retried) after
    # the caller was cancelled: a timed-out evaluation went on answering the
    # whole dataset in the background.
    calls = Counter()

    async def work(item: dict) -> int:
        calls["started"] += 1
        await asyncio.sleep(0.5)
        calls["finished"] += 1
        return item["i"]

    async def main() -> None:
        processor = BatchProcessor(max_concurrency=2)
        task = asyncio.create_task(
            processor.aexecute_with_fallback(
                list(range(10)),
                lambda batch: [{"i": i} for i in batch],
                work,
                "T",
                show_progress=False,
            )
        )
        await asyncio.sleep(0.1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.8)
        assert len(asyncio.all_tasks()) == 1  # only main

    asyncio.run(main())
    assert calls == Counter(started=2)
