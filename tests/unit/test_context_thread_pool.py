# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""ContextThreadPoolExecutor carries the caller's contextvars into its workers.

The log context (``pipeline_id``/``stage``/``query_id``) is bound with
``structlog.contextvars``; a plain pool thread starts from an empty context and
would drop it from every line it logs.
"""

from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from structlog.contextvars import bound_contextvars, get_contextvars

from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor
from unified_kg_rag.shared.utils.langchain import BatchProcessor

pytestmark = pytest.mark.unit

_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


def test_plain_thread_pool_drops_the_context() -> None:
    # The premise: why the plain pool is not enough.
    token = _request_id.set("r-1")
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(_request_id.get).result() is None
    finally:
        _request_id.reset(token)


def test_submit_and_map_run_in_the_callers_context() -> None:
    token = _request_id.set("r-1")
    try:
        with ContextThreadPoolExecutor(max_workers=2) as executor:
            assert executor.submit(_request_id.get).result() == "r-1"
            assert (
                list(executor.map(lambda _: _request_id.get(), range(3))) == ["r-1"] * 3
            )
    finally:
        _request_id.reset(token)


def test_context_is_copied_at_submit_time() -> None:
    with ContextThreadPoolExecutor(max_workers=1) as executor:
        token = _request_id.set("first")
        first = executor.submit(_request_id.get)
        _request_id.reset(token)
        second = executor.submit(_request_id.get)
        assert first.result() == "first"
        assert second.result() is None


def test_worker_writes_do_not_leak_into_the_caller() -> None:
    with ContextThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(_request_id.set, "worker").result()
    assert _request_id.get() is None


def test_batch_processor_workers_see_the_bound_log_context() -> None:
    # Each item on the item pool and under the per-call timeout wrapper's own
    # thread: every hop must keep the caller's log context.
    seen: list[dict[str, Any]] = []
    bp = BatchProcessor(max_concurrency=2, call_timeout_seconds=5)

    def call(item):  # noqa: ANN001
        seen.append(get_contextvars())
        return item["v"]

    with bound_contextvars(query_id="q-1"):
        results = bp.execute_with_fallback(
            items_to_process=[1, 2],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            sequential_func=call,
            task_name="t",
            show_progress=False,
        )

    assert results == [1, 2]
    assert [ctx.get("query_id") for ctx in seen] == ["q-1", "q-1"]


def test_shut_down_after_shuts_the_pool_down_when_the_block_ends() -> None:
    from unified_kg_rag.shared.utils.concurrency import shut_down_after

    with shut_down_after(ThreadPoolExecutor(max_workers=1)) as pool:
        future = pool.submit(lambda: 1)
    assert future.result() == 1
    with pytest.raises(RuntimeError, match="shutdown"):
        pool.submit(lambda: 2)


def test_shut_down_after_does_not_wait_for_running_tasks_on_error() -> None:
    import threading
    import time

    from unified_kg_rag.shared.utils.concurrency import shut_down_after

    release = threading.Event()
    started = threading.Event()

    def running() -> None:
        started.set()
        release.wait(10)

    began = time.time()
    try:
        with pytest.raises(RuntimeError, match="stage failed"):
            with shut_down_after(ThreadPoolExecutor(max_workers=1)) as pool:
                future = pool.submit(running)
                queued = pool.submit(running)
                started.wait(5)
                raise RuntimeError("stage failed")
        assert time.time() - began < 2
        assert queued.cancelled() and not future.done()
    finally:
        release.set()


@pytest.mark.parametrize(
    ("raised", "waits"),
    [(KeyboardInterrupt, False), (SystemExit, False), (GeneratorExit, True)],
)
def test_the_pool_skips_its_wait_on_the_leases_stop_only(
    raised: type[BaseException], waits: bool
) -> None:
    # One stop rule (is_stop) with the write lease: the pool skips waiting
    # for its workers exactly when the lease is given up expired for their
    # writes in flight. Before, any non-Exception skipped the wait, and the
    # lease released (no wait for the next run) over writes still running.
    import threading

    from unified_kg_rag.shared.utils.concurrency import is_stop

    assert is_stop(raised) is not waits
    release, started = threading.Event(), threading.Event()

    def running() -> None:
        started.set()
        release.wait(10)

    timer = threading.Timer(1.0, release.set)
    timer.start()
    try:
        with pytest.raises(raised):
            with ContextThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(running)
                started.wait(5)
                raise raised()
        assert future.done() is waits
    finally:
        release.set()
        timer.cancel()
