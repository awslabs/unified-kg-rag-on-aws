# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``single_flight``: the one in-flight cache the query path's shared calls
use (query embeddings, index limits, index probes, incident edges, keyword
extractions), under both failure policies."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from unified_kg_rag.shared.utils.concurrency import single_flight

pytestmark = pytest.mark.unit


class _Call:
    """A call that waits for ``release`` and then returns or raises."""

    def __init__(self, result: Any = "value", error: Exception | None = None):
        self.result = result
        self.error = error
        self.calls = 0
        self.release = asyncio.Event()

    async def __call__(self) -> Any:
        self.calls += 1
        await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.parametrize("share_failure", [True, False])
async def test_concurrent_callers_share_one_call(share_failure: bool) -> None:
    memo: dict[str, Any] = {}
    call = _Call()
    callers = [
        asyncio.ensure_future(
            single_flight(memo, "k", call, share_failure=share_failure)
        )
        for _ in range(3)
    ]
    await asyncio.sleep(0)
    call.release.set()
    assert await asyncio.gather(*callers) == ["value"] * 3
    assert call.calls == 1
    # Not kept: the next caller calls again.
    assert memo == {}
    assert await single_flight(memo, "k", call, share_failure=share_failure)
    assert call.calls == 2


@pytest.mark.parametrize("share_failure", [True, False])
async def test_a_result_is_kept_while_keep_says_so(share_failure: bool) -> None:
    memo: dict[str, Any] = {}
    call = _Call(result=["hit"])
    call.release.set()
    keep = bool
    for _ in range(2):
        found = await single_flight(
            memo, "k", call, share_failure=share_failure, keep=keep
        )
        assert found == ["hit"]
    assert call.calls == 1
    empty = _Call(result=[])
    empty.release.set()
    for _ in range(2):
        await single_flight(memo, "e", empty, share_failure=share_failure, keep=keep)
    assert empty.calls == 2 and "e" not in memo


async def test_a_shared_failure_reaches_each_waiter_as_its_own_copy() -> None:
    memo: dict[str, Any] = {}
    error = ValueError("store down")
    call = _Call(error=error)
    callers = [
        asyncio.ensure_future(single_flight(memo, "k", call, share_failure=True))
        for _ in range(2)
    ]
    await asyncio.sleep(0)
    call.release.set()
    raised = await asyncio.gather(*callers, return_exceptions=True)
    assert call.calls == 1 and memo == {}
    assert all(isinstance(e, ValueError) and e.__cause__ is error for e in raised)
    assert raised[0] is not raised[1]


async def test_a_shared_call_outlives_its_cancelled_first_caller() -> None:
    memo: dict[str, Any] = {}
    call = _Call()
    first = asyncio.ensure_future(single_flight(memo, "k", call, share_failure=True))
    waiter = asyncio.ensure_future(single_flight(memo, "k", call, share_failure=True))
    await asyncio.sleep(0)
    first.cancel()
    await asyncio.sleep(0)
    call.release.set()
    assert await waiter == "value"
    assert first.cancelled() and call.calls == 1


@pytest.mark.parametrize("cancel", [False, True])
async def test_without_sharing_failures_each_waiter_calls_itself(
    cancel: bool,
) -> None:
    memo: dict[str, Any] = {}
    failing = _Call(error=ValueError("transient"))
    owner = asyncio.ensure_future(
        single_flight(memo, "k", failing, share_failure=False)
    )
    await asyncio.sleep(0)
    retry = _Call(result="own")
    retry.release.set()

    async def waiter_call() -> Any:
        # The waiter's own call, made once the owner's failed.
        return await retry()

    waiter = asyncio.ensure_future(
        single_flight(memo, "k", waiter_call, share_failure=False)
    )
    await asyncio.sleep(0)
    if cancel:
        owner.cancel()
    else:
        failing.release.set()
    with pytest.raises(asyncio.CancelledError if cancel else ValueError):
        await owner
    assert await waiter == "own"
    assert retry.calls == 1 and memo == {}


async def test_a_waiters_cancellation_cancels_no_other_caller() -> None:
    memo: dict[str, Any] = {}
    call = _Call()
    owner = asyncio.ensure_future(single_flight(memo, "k", call, share_failure=False))
    waiter = asyncio.ensure_future(single_flight(memo, "k", call, share_failure=False))
    await asyncio.sleep(0)
    waiter.cancel()
    await asyncio.sleep(0)
    call.release.set()
    assert await owner == "value"
    assert waiter.cancelled()


def _broken_keep(_: Any) -> bool:
    raise RuntimeError("keep failed")


async def test_a_raising_keep_resolves_the_waiters_and_drops_the_key() -> None:
    memo: dict[str, Any] = {}
    call = _Call()
    owner = asyncio.ensure_future(
        single_flight(memo, "k", call, share_failure=False, keep=_broken_keep)
    )
    waiter = asyncio.ensure_future(
        single_flight(memo, "k", call, share_failure=False, keep=_broken_keep)
    )
    await asyncio.sleep(0)
    call.release.set()
    with pytest.raises(RuntimeError, match="keep failed"):
        await owner
    # The waiter is not left hanging on the owner's unresolved future.
    assert await asyncio.wait_for(waiter, 1) == "value"
    assert memo == {}


async def test_a_raising_keep_does_not_leave_a_shared_call_in_the_memo() -> None:
    memo: dict[str, Any] = {}
    call = _Call()
    call.release.set()
    await single_flight(memo, "k", call, share_failure=True, keep=_broken_keep)
    await asyncio.sleep(0)
    assert memo == {}


def test_an_extraction_in_flight_on_another_loop_is_none_of_this_loops() -> None:
    """The query extractions in flight are shared by thread; a future of
    another event loop cannot be awaited here, so it reads as absent."""
    import threading

    from unified_kg_rag.application.retrieval.rag_chain import _InFlight

    inflight = _InFlight(threading.Lock())
    first, second = asyncio.new_event_loop(), asyncio.new_event_loop()
    try:

        async def hold() -> None:
            inflight["k"] = asyncio.get_running_loop().create_future()

        async def look() -> bool:
            return "k" in inflight

        first.run_until_complete(hold())
        assert first.run_until_complete(look()) is True
        assert second.run_until_complete(look()) is False
        with pytest.raises(KeyError):
            second.run_until_complete(_get(inflight))
    finally:
        first.close()
        second.close()


async def _get(inflight: Any) -> Any:
    return inflight["k"]
