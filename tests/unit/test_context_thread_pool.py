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
    # Two chunks on the chunk pool, each under the per-call timeout wrapper's
    # own thread: both hops must keep the caller's log context.
    seen: list[dict[str, Any]] = []
    bp = BatchProcessor(batch_size=1, chunk_concurrency=2, call_timeout_seconds=5)

    def batch(inputs, config=None, return_exceptions=False):  # noqa: ANN001, ARG001
        seen.append(get_contextvars())
        return [inputs[0]["v"]]

    with bound_contextvars(query_id="q-1"):
        results = bp.execute_with_fallback(
            items_to_process=[1, 2],
            prepare_inputs_func=lambda items: [{"v": i} for i in items],
            batch_func=batch,
            sequential_func=lambda item: item["v"],
            task_name="t",
            show_progress=False,
        )

    assert results == [1, 2]
    assert [ctx.get("query_id") for ctx in seen] == ["q-1", "q-1"]
