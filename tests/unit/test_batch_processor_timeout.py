# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wall-clock timeout behavior for BatchProcessor.

A hung Bedrock Converse call (open socket, no completion) is not caught by
botocore's byte-gap read_timeout and would block the single Fargate worker for
a whole stage (observed in claim_extraction). BatchProcessor wraps each
batch/sequential call in a wall-clock timeout so it aborts and falls back to
per-item retries instead.
"""

import time

import pytest

from unified_kg_rag.shared.utils.langchain import BatchProcessor

pytestmark = pytest.mark.unit


def test_run_with_timeout_aborts_hung_call() -> None:
    with pytest.raises(TimeoutError, match="call timeout"):
        BatchProcessor._run_with_timeout(lambda: time.sleep(5), 1, "hung")


def test_run_with_timeout_returns_fast_result() -> None:
    assert BatchProcessor._run_with_timeout(lambda: 42, 5, "fast") == 42


def test_run_with_timeout_zero_disables() -> None:
    # 0 means "no timeout" — run directly.
    assert BatchProcessor._run_with_timeout(lambda: "ok", 0, "nolimit") == "ok"


def test_batch_timeout_falls_back_to_sequential() -> None:
    # A batch that hangs should time out, then the sequential path handles items.
    bp = BatchProcessor(call_timeout_seconds=1, batch_size=10)

    def hung_batch(_inputs, config=None):  # noqa: ANN001, ARG001
        time.sleep(30)
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

    def batch(inputs, config=None):  # noqa: ANN001, ARG001
        # Later items return faster, so completion order != submission order.
        v = inputs[0]["v"]
        time.sleep((10 - v) * 0.02)
        return [{"echo": v}]

    results = bp.execute_with_fallback(
        items_to_process=list(range(6)),
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        batch_func=batch,
        sequential_func=lambda item: {"echo": item["v"]},
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": i} for i in range(6)]


def test_chunks_run_concurrently() -> None:
    # 4 chunks that each sleep 0.3s should finish in well under 4*0.3s when run
    # with chunk_concurrency=4 (overlapping), proving they are not serial.
    bp = BatchProcessor(batch_size=1, chunk_concurrency=4, call_timeout_seconds=0)

    def batch(inputs, config=None):  # noqa: ANN001, ARG001
        time.sleep(0.3)
        return [{"echo": inputs[0]["v"]}]

    start = time.monotonic()
    results = bp.execute_with_fallback(
        items_to_process=[1, 2, 3, 4],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        batch_func=batch,
        sequential_func=lambda item: {"echo": item["v"]},
        task_name="t",
        show_progress=False,
    )
    elapsed = time.monotonic() - start
    assert len(results) == 4
    # Serial would be ~1.2s; concurrent should be ~0.3-0.5s.
    assert elapsed < 0.8, f"chunks did not overlap (took {elapsed:.2f}s)"


def test_chunk_concurrency_one_is_serial() -> None:
    # chunk_concurrency=1 keeps the legacy strictly-serial path.
    bp = BatchProcessor(batch_size=1, chunk_concurrency=1, call_timeout_seconds=0)
    results = bp.execute_with_fallback(
        items_to_process=[1, 2, 3],
        prepare_inputs_func=lambda items: [{"v": i} for i in items],
        batch_func=lambda inputs, config=None: [{"echo": inputs[0]["v"]}],
        sequential_func=lambda item: {"echo": item["v"]},
        task_name="t",
        show_progress=False,
    )
    assert results == [{"echo": 1}, {"echo": 2}, {"echo": 3}]
