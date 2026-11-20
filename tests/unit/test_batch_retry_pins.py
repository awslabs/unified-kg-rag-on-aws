# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a batch item's retries do: the attempts, the waits, the budget, the
stops and the log line (``BatchProcessor._create_retry_decorator``).

Pinned on the outcome alone, so they hold for any retry engine behind it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.exceptions import OutputParserException

from unified_kg_rag.adapters.llm import retry as llm_retry
from unified_kg_rag.adapters.llm.langchain import BatchProcessor
from unified_kg_rag.domain.models.config import RetryPolicy

pytestmark = pytest.mark.unit


def _processor(**policy: Any) -> BatchProcessor:
    return BatchProcessor(retry=RetryPolicy(**policy))


def _recording_event() -> tuple[Any, list[float]]:
    slept: list[float] = []
    return MagicMock(is_set=lambda: False, wait=slept.append), slept


def test_the_last_attempts_error_is_raised_unchanged() -> None:
    processor = _processor(max_attempts=3, base_delay_seconds=0.0)
    errors = [TimeoutError("first"), TimeoutError("second"), TimeoutError("third")]
    calls = {"n": 0}

    @processor._create_retry_decorator("op")
    def failing() -> None:
        calls["n"] += 1
        raise errors[calls["n"] - 1]

    with pytest.raises(TimeoutError) as raised:
        failing()
    assert raised.value is errors[2]
    assert calls["n"] == 3


def test_each_wait_is_drawn_for_its_retry_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # No jitter: the ceilings 0.25, 0.5 and 1.0 s of retries 1 to 3.
    processor = _processor(
        max_attempts=4, base_delay_seconds=0.25, max_delay_seconds=None, jitter="none"
    )
    abandoned, slept = _recording_event()
    calls = {"n": 0}

    @processor._create_retry_decorator("extract", abandoned)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] < 4:
            raise TimeoutError("hung")
        return "ok"

    with caplog.at_level(logging.WARNING):
        assert flaky() == "ok"
    assert slept == [0.25, 0.5, 1.0]
    retries = [r.getMessage() for r in caplog.records if "Retrying" in r.getMessage()]
    assert retries == [
        "Retrying 'extract' (attempt 1 failed). Waiting 0.2s",
        "Retrying 'extract' (attempt 2 failed). Waiting 0.5s",
        "Retrying 'extract' (attempt 3 failed). Waiting 1.0s",
    ]


def test_a_zero_wait_retry_still_stops_at_the_attempts() -> None:
    processor = _processor(
        max_attempts=3, base_delay_seconds=10.0, max_total_seconds=60.0
    )
    abandoned, slept = _recording_event()
    calls = {"n": 0}

    @processor._create_retry_decorator("op", abandoned)
    def unparseable() -> None:
        calls["n"] += 1
        raise OutputParserException("no XML")

    with pytest.raises(OutputParserException):
        unparseable()
    assert calls["n"] == 3
    assert slept == [0.0, 0.0]


def test_the_budget_counts_the_time_the_attempts_took(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The budget is the time since the first attempt started plus the next
    # wait: a retry that would end past it is not made.
    clock = {"now": 100.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["now"])
    processor = _processor(
        max_attempts=10,
        base_delay_seconds=1.0,
        max_delay_seconds=1.0,
        max_total_seconds=10,
        jitter="none",
    )
    abandoned = MagicMock(is_set=lambda: False)
    abandoned.wait.side_effect = lambda delay: clock.__setitem__(
        "now", clock["now"] + delay
    )
    calls = {"n": 0}

    @processor._create_retry_decorator("op", abandoned)
    def slow() -> None:
        calls["n"] += 1
        clock["now"] += 2.0  # each attempt takes 2 s
        raise TimeoutError("hung")

    with pytest.raises(TimeoutError):
        slow()
    # 2 s + 1 s per failed attempt: the 4th failure ends at 11 s (wait 3 at
    # 8 + 1 = 9 s is in; the 4th wait, from 11 s, is not).
    assert calls["n"] == 4


def test_an_attempts_error_does_not_carry_the_previous_one() -> None:
    # Classifiers walk an error's causes (shared.exceptions.causes): the
    # first attempt's error must not become the context of the last's.
    processor = _processor(max_attempts=2, base_delay_seconds=0.0)
    calls = {"n": 0}

    @processor._create_retry_decorator("op")
    def failing() -> None:
        calls["n"] += 1
        raise TimeoutError(f"attempt {calls['n']}")

    with pytest.raises(TimeoutError) as raised:
        failing()
    assert raised.value.__context__ is None


def test_an_abandoned_batch_retries_no_failed_item() -> None:
    processor = _processor(max_attempts=5, base_delay_seconds=0.0)
    abandoned = threading.Event()
    error = TimeoutError("hung")
    calls = {"n": 0}

    @processor._create_retry_decorator("op", abandoned)
    def failing() -> None:
        calls["n"] += 1
        abandoned.set()
        raise error

    with pytest.raises(TimeoutError) as raised:
        failing()
    assert raised.value is error
    assert calls["n"] == 1


def test_the_backoff_wait_ends_when_the_batch_is_abandoned() -> None:
    processor = _processor(
        max_attempts=3, base_delay_seconds=30.0, max_delay_seconds=None, jitter="none"
    )
    abandoned = MagicMock(is_set=lambda: False)
    calls = {"n": 0}

    @processor._create_retry_decorator("op", abandoned)
    def flaky() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("hung")
        return "ok"

    assert flaky() == "ok"
    abandoned.wait.assert_called_once_with(30.0)


@pytest.mark.parametrize("stop", [KeyboardInterrupt, SystemExit])
def test_a_stop_is_never_retried(stop: type[BaseException]) -> None:
    processor = _processor(max_attempts=5, base_delay_seconds=0.0)
    calls = {"n": 0}

    @processor._create_retry_decorator("op")
    def stopped() -> None:
        calls["n"] += 1
        raise stop

    with pytest.raises(stop):
        stopped()
    assert calls["n"] == 1


async def test_async_items_retry_under_the_same_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm_retry._jitter_rng, "uniform", lambda a, b: b)
    slept: list[float] = []
    real_sleep = asyncio.sleep

    async def recorded(delay: float, *args: Any, **kwargs: Any) -> None:
        slept.append(delay)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", recorded)
    processor = _processor(
        max_attempts=4, base_delay_seconds=1.0, max_delay_seconds=2.0, jitter="full"
    )
    calls = {"n": 0}

    @processor._create_retry_decorator("op")
    async def failing() -> None:
        calls["n"] += 1
        raise TimeoutError("hung")

    with pytest.raises(TimeoutError):
        await failing()
    assert calls["n"] == 4
    assert slept == [1.0, 2.0, 2.0]


async def test_an_async_cancellation_is_never_retried() -> None:
    processor = _processor(max_attempts=5, base_delay_seconds=0.0)
    calls = {"n": 0}

    @processor._create_retry_decorator("op")
    async def cancelled() -> None:
        calls["n"] += 1
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await cancelled()
    assert calls["n"] == 1


async def test_an_async_budget_stops_the_retries() -> None:
    processor = _processor(
        max_attempts=5, base_delay_seconds=10.0, max_total_seconds=1.0, jitter="none"
    )
    calls = {"n": 0}

    @processor._create_retry_decorator("op")
    async def failing() -> None:
        calls["n"] += 1
        raise TimeoutError("hung")

    with pytest.raises(TimeoutError):
        await failing()
    assert calls["n"] == 1
