# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock timeout behavior for BatchProcessor.

A hung Bedrock Converse call (open socket, no completion) is not caught by
botocore's byte-gap read_timeout and would block the single Fargate worker for
a whole stage (observed in claim_extraction). BatchProcessor wraps each
batch/sequential call in a wall-clock timeout so it aborts and falls back to
per-item retries instead.

A "hung" call here blocks on an Event that the ``release_hung_calls`` fixture
sets at teardown. A timed-out call is abandoned, not cancelled, and executor
threads are joined at interpreter exit, so a plain ``time.sleep`` would keep
the test process alive for the whole sleep after the suite finished.
"""

import threading
from collections.abc import Iterator

import pytest

from unified_kg_rag.shared.utils.langchain import BatchProcessor

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


def test_batch_timeout_falls_back_to_sequential(release_hung_calls) -> None:
    # A batch that hangs should time out, then the sequential path handles items.
    bp = BatchProcessor(call_timeout_seconds=1, batch_size=10)

    def hung_batch(
        _inputs, config=None, return_exceptions=False
    ):  # noqa: ANN001, ARG001
        release_hung_calls.wait()
        return []

    def sequential(item):  # noqa: ANN001
        return {"echo": item["v"]}

    results = bp.execute_with_fallback(
        items_to_process=[1, 2],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        batch_func=hung_batch,
        sequential_func=sequential,
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": 1}, {"echo": 2}]


def test_chunk_results_preserve_order_when_concurrent() -> None:
    # With chunk_concurrency > 1 chunks run on a thread pool and complete out of
    # order; results must still be reassembled in input order.
    bp = BatchProcessor(batch_size=1, chunk_concurrency=4, call_timeout_seconds=0)
    # Each chunk waits for the next one to finish, so they complete strictly in
    # reverse submission order (the last chunk has nothing to wait for).
    n = 4
    done = [threading.Event() for _ in range(n)]

    def batch(inputs, config=None, return_exceptions=False):  # noqa: ANN001, ARG001
        v = inputs[0]["v"]
        if v + 1 < n:
            assert done[v + 1].wait(timeout=10), "chunks did not run concurrently"
        done[v].set()
        return [{"echo": v}]

    results = bp.execute_with_fallback(
        items_to_process=list(range(n)),
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        batch_func=batch,
        sequential_func=lambda item: {"echo": -1},
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

    def batch(inputs, config=None, return_exceptions=False):  # noqa: ANN001, ARG001
        nonlocal in_flight, max_in_flight
        with lock:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        try:
            barrier.wait()
        finally:
            with lock:
                in_flight -= 1
        return [{"echo": inputs[0]["v"]}]

    results = bp.execute_with_fallback(
        items_to_process=[1, 2, 3, 4],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        batch_func=batch,
        sequential_func=lambda item: {"echo": -1},
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": i} for i in [1, 2, 3, 4]]
    assert max_in_flight == 4


def test_chunk_concurrency_one_is_serial() -> None:
    # chunk_concurrency=1 keeps the legacy strictly-serial path.
    bp = BatchProcessor(batch_size=1, chunk_concurrency=1, call_timeout_seconds=0)
    results = bp.execute_with_fallback(
        items_to_process=[1, 2, 3],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        batch_func=lambda inputs, **_: [{"echo": inputs[0]["v"]}],
        sequential_func=lambda item: {"echo": item["v"]},
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": 1}, {"echo": 2}, {"echo": 3}]
