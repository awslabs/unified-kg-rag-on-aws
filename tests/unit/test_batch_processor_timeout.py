# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Wall-clock timeout behavior for BatchProcessor.

A hung Bedrock Converse call (open socket, no completion) is not caught by
botocore's byte-gap read_timeout and would block the single Fargate worker for
a whole stage (observed in claim_extraction). BatchProcessor wraps each
batch/sequential call in a wall-clock timeout so it aborts and falls back to
per-item retries instead.
"""

import time

import pytest

from aws_graphrag.shared.utils.langchain import BatchProcessor

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
