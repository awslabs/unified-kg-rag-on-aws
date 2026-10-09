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

import threading
from collections import Counter
from collections.abc import Iterator

import pytest

from unified_kg_rag.adapters.aws.bedrock import BaseBedrockModelFactory
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED, BatchProcessor

pytestmark = pytest.mark.unit


@pytest.fixture
def release_hung_calls() -> Iterator[threading.Event]:
    release = threading.Event()
    yield release
    release.set()


def test_run_with_timeout_aborts_hung_call(release_hung_calls) -> None:
    with pytest.raises(TimeoutError, match="call timeout"):
        BatchProcessor._run_with_timeout(release_hung_calls.wait, 1, "hung")


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
    # and only the slow one is called again (it used to time out the whole
    # chunk and re-run all ten items).
    fake = _SlowFirstCall(slow=3, release=release_hung_calls)
    bp = BatchProcessor(
        call_timeout_seconds=1,
        batch_size=10,
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
    # The first call and each of the max_attempts retries time out; only
    # that position is marked failed.
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
        batch_size=10,
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
    assert calls == Counter({0: 1, 1: 3, 2: 1})


def test_items_of_a_chunk_run_concurrently() -> None:
    # The items of one chunk are in flight together, up to max_concurrency.
    barrier = threading.Barrier(3, timeout=10)

    def call(item: dict[str, int]) -> dict[str, int]:
        barrier.wait()
        return {"echo": item["v"]}

    bp = BatchProcessor(batch_size=3, max_concurrency=3, call_timeout_seconds=0)
    results = bp.execute_with_fallback(
        items_to_process=[1, 2, 3],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": 1}, {"echo": 2}, {"echo": 3}]


def test_chunk_results_preserve_order_when_concurrent() -> None:
    # With chunk_concurrency > 1 chunks run on a thread pool and complete out of
    # order; results must still be reassembled in input order.
    bp = BatchProcessor(batch_size=1, chunk_concurrency=4, call_timeout_seconds=0)
    # Each chunk waits for the next one to finish, so they complete strictly in
    # reverse submission order (the last chunk has nothing to wait for).
    n = 4
    done = [threading.Event() for _ in range(n)]

    def call(item: dict[str, int]) -> dict[str, int]:
        v = item["v"]
        if v + 1 < n:
            assert done[v + 1].wait(timeout=10), "chunks did not run concurrently"
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


def test_chunks_run_concurrently() -> None:
    # With chunk_concurrency=4, four chunks must be in flight at once. Each
    # chunk waits at a barrier for the other three; a serial executor would
    # never reach 4 in flight and the barrier would time out.
    bp = BatchProcessor(batch_size=1, chunk_concurrency=4, call_timeout_seconds=0)
    barrier = threading.Barrier(4, timeout=10)
    lock = threading.Lock()
    in_flight = 0
    max_in_flight = 0

    def call(item: dict[str, int]) -> dict[str, int]:
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        try:
            barrier.wait()
        finally:
            with lock:
                in_flight -= 1
        return {"echo": item["v"]}

    results = bp.execute_with_fallback(
        items_to_process=[1, 2, 3, 4],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=call,
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": i} for i in [1, 2, 3, 4]]
    assert max_in_flight == 4


def test_chunk_concurrency_one_is_serial() -> None:
    # chunk_concurrency=1 keeps the strictly-serial chunk loop.
    bp = BatchProcessor(batch_size=1, chunk_concurrency=1, call_timeout_seconds=0)
    results = bp.execute_with_fallback(
        items_to_process=[1, 2, 3],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        sequential_func=lambda item: {"echo": item["v"]},
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": 1}, {"echo": 2}, {"echo": 3}]


def test_bedrock_read_timeout_outlasts_the_call_timeout() -> None:
    # The wall-clock call timeout is the limit meant to end a long call: with
    # an equal socket read timeout the two raced, and botocore's silent
    # re-send on a read timeout was then abandoned by the call timeout.
    call_timeout = BatchProcessor.model_fields["call_timeout_seconds"].default
    assert BaseBedrockModelFactory.BOTO_READ_TIMEOUT > call_timeout
