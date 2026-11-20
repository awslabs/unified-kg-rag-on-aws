# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A shared call (``single_flight(share_failure=True)``) under random
schedules of its waiters.

Each case is a schedule of actions on one key: a waiter joins (with a
reader's stop bound or none), a waiter's reader stops, a waiter is
cancelled from outside, or a running call takes one step to its next stop
point (a backoff wait, on the event loop or on a worker thread). Each call
passes a drawn number of stop points, then returns or fails. The steps are
driven by the schedule, not by the clock, so a case replays the same way
on a loaded runner. After it every call runs to its end and the oracle
checks each waiter and each call:

- a waiter whose reader never stopped and that was not cancelled never
  ends in ``CancelledError`` (it was handed no stopped call), and gets the
  call's value or its own copy of the call's failure, raised from it;
- a call is cancelled only once each of its waiters' readers stopped, and
  passes no stop point after that;
- nothing stays in flight and the memo is empty.
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from unified_kg_rag.shared.query_stop import (
    QueryStop,
    abackoff_wait,
    backoff_wait,
    query_stop_scope,
)
from unified_kg_rag.shared.utils.concurrency import single_flight

pytestmark = pytest.mark.property

# A hang backstop (seconds) for each wait of a case, far past any step.
_BACKSTOP = 30.0


class _Boom(ValueError):
    pass


@dataclass
class _CallSpec:
    points: int
    fails: bool
    on_thread: bool


@dataclass
class _Call:
    """One shared call: its waiters, its stop points passed and the step
    the schedule releases."""

    spec: _CallSpec
    task: asyncio.Future[Any] | None = None
    joiners: list[int] = field(default_factory=list)
    passed: int = 0
    # Stop points passed when its last waiter's reader stopped.
    passed_when_all_stopped: int | None = None
    step: asyncio.Event = field(default_factory=asyncio.Event)
    parked: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _Waiter:
    stop: QueryStop | None
    task: asyncio.Future[Any] | None = None
    call: _Call | None = None
    stopped: bool = False
    cancelled: bool = False


_ACTIONS = st.one_of(
    st.tuples(st.just("join"), st.booleans()),
    st.tuples(st.just("stop"), st.integers(0, 3)),
    st.tuples(st.just("cancel"), st.integers(0, 3)),
    st.tuples(st.just("step"), st.integers(0, 3)),
)
_SPECS = st.builds(
    _CallSpec,
    points=st.integers(1, 4),
    fails=st.booleans(),
    on_thread=st.booleans(),
)


async def _settle() -> None:
    """Let the loop run what the last action woke (stops are delivered with
    ``call_soon_threadsafe``)."""
    for _ in range(20):
        await asyncio.sleep(0)


async def _schedule(actions: list[tuple[str, Any]], specs: list[_CallSpec]) -> None:
    memo: dict[str, asyncio.Future[Any]] = {}
    calls: list[_Call] = []
    waiters: list[_Waiter] = []

    def compute_for(record: _Call) -> Any:
        async def compute() -> str:
            record.task = asyncio.current_task()
            for _ in range(record.spec.points):
                record.parked.set()
                await record.step.wait()
                record.step.clear()
                if record.spec.on_thread:
                    await asyncio.to_thread(backoff_wait, 0)
                else:
                    await abackoff_wait(0)
                record.passed += 1
            if record.spec.fails:
                raise _Boom("shared failure")
            return f"value {calls.index(record)}"

        return compute

    async def wait(waiter: _Waiter) -> str:
        scope = (
            query_stop_scope(waiter.stop)
            if waiter.stop is not None
            else contextlib.nullcontext()
        )
        with scope:
            # The call single_flight hands this waiter: the one in the memo,
            # else a new one (it starts on the loop's next turn).
            pending = memo.get("k")
            if pending is None:
                record = _Call(specs[len(calls) % len(specs)])
                calls.append(record)
            else:
                record = next(c for c in calls if c.task is pending)
            waiter.call = record
            record.joiners.append(waiters.index(waiter))
            return await single_flight(
                memo, "k", compute_for(record), share_failure=True
            )

    def note_all_stopped() -> None:
        for call in calls:
            if (
                call.passed_when_all_stopped is None
                and call.joiners
                and all(waiters[j].stopped for j in call.joiners)
            ):
                call.passed_when_all_stopped = call.passed

    async def step(call: _Call) -> None:
        assert call.task is not None
        call.parked.clear()
        call.step.set()
        async with asyncio.timeout(_BACKSTOP):
            parked = asyncio.ensure_future(call.parked.wait())
            await asyncio.wait({parked, call.task}, return_when="FIRST_COMPLETED")
            parked.cancel()

    for action, arg in actions:
        if action == "join" and len(waiters) < 8:
            waiter = _Waiter(QueryStop() if arg else None)
            waiters.append(waiter)
            waiter.task = asyncio.ensure_future(wait(waiter))
        elif action in ("stop", "cancel"):
            # The arg-th of the waiters still waiting (that a stop can stop).
            targets = [
                w
                for w in waiters
                if w.task is not None
                and not w.task.done()
                and (action == "cancel" or w.stop is not None)
            ]
            if not targets:
                continue
            waiter = targets[arg % len(targets)]
            assert waiter.task is not None
            if action == "stop" and waiter.stop is not None:
                waiter.stopped = True
                waiter.stop.set()
            elif action == "cancel":
                waiter.cancelled = True
                waiter.task.cancel()
        elif action == "step":
            running = [c for c in calls if c.task is not None and not c.task.done()]
            if running:
                await step(running[arg % len(running)])
        await _settle()
        note_all_stopped()

    # Every call runs to its end.
    for _ in range(10):
        running = [c for c in calls if c.task is not None and not c.task.done()]
        if not running:
            break
        for call in running:
            await step(call)
        await _settle()
    async with asyncio.timeout(_BACKSTOP):
        outcomes = await asyncio.gather(
            *(w.task for w in waiters if w.task is not None), return_exceptions=True
        )

    failures: list[BaseException] = []
    for waiter, outcome in zip(waiters, outcomes, strict=True):
        assert not (
            isinstance(outcome, BaseException)
            and not isinstance(outcome, asyncio.CancelledError | _Boom)
        ), outcome
        if waiter.stopped or waiter.cancelled:
            continue
        assert not isinstance(outcome, asyncio.CancelledError), "live waiter cancelled"
        if isinstance(outcome, _Boom):
            assert isinstance(outcome.__cause__, _Boom)
            failures.append(outcome)
    assert len({id(e) for e in failures}) == len(failures)
    for call in calls:
        assert call.task is not None and call.task.done()
        if call.task.cancelled():
            assert call.passed_when_all_stopped is not None, call.joiners
        if call.passed_when_all_stopped is not None:
            assert call.passed == call.passed_when_all_stopped
    assert memo == {}


@settings(max_examples=150, deadline=None)
@given(
    actions=st.lists(_ACTIONS, max_size=24),
    specs=st.lists(_SPECS, min_size=1, max_size=4),
)
def test_a_shared_call_is_every_waiters_and_stops_only_for_all(
    actions: list[tuple[str, Any]], specs: list[_CallSpec]
) -> None:
    asyncio.run(_schedule(actions, specs))
