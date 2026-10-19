# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An abandoned batch makes no further attempt (AWS-free)."""

from __future__ import annotations

import os
import signal
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException

from unified_kg_rag.shared.utils import langchain as langchain_utils
from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED, BatchProcessor

pytestmark = pytest.mark.unit


class _RecordingEvent(threading.Event):
    """A ``threading.Event`` that remembers the instances made from it."""

    made: list[threading.Event] = []

    def __init__(self) -> None:
        super().__init__()
        self.made.append(self)


@pytest.fixture
def sigint_raises() -> Any:
    """Make SIGINT raise ``KeyboardInterrupt`` for the test.

    A process started in the background (``nohup``, ``cmd &``) inherits SIGINT
    as ignored, so the signal the test sends would otherwise do nothing.
    """
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


@pytest.mark.usefixtures("sigint_raises")
def test_an_abandoned_batch_makes_no_further_attempt(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # A stop (SIGTERM) or Ctrl-C unwinds the calling thread; an item still
    # running must not go on retrying behind it (interpreter exit waits for
    # the pool's threads, so the task would be killed mid-shutdown).
    #
    # Ordered by events, not by time: the item's first attempt interrupts the
    # calling thread, then waits until the batch is marked abandoned before it
    # fails, so the retry policy always sees the abandonment.
    monkeypatch.setattr(_RecordingEvent, "made", [])
    monkeypatch.setattr(
        langchain_utils,
        "threading",
        SimpleNamespace(Event=_RecordingEvent),
        raising=False,
    )
    calls: list[threading.Thread] = []

    def flaky(_: dict[str, Any]) -> Any:
        calls.append(threading.current_thread())
        if len(calls) == 1:
            os.kill(os.getpid(), signal.SIGINT)
            # The batch's abandonment flag (the only event the batch makes).
            (abandoned,) = _RecordingEvent.made
            assert abandoned.wait(timeout=30), "the batch was not abandoned"
        raise OutputParserException("malformed")

    processor = BatchProcessor(
        max_concurrency=1,
        max_attempts=4,
        retry_multiplier=1.0,
        retry_max_wait=1,
        call_timeout_seconds=0,
    )
    with pytest.raises(KeyboardInterrupt):
        processor.execute_with_fallback(
            [{"x": 1}], lambda items: items, flaky, "task", show_progress=False
        )
    # Wait for the item's thread itself: it ends after its last attempt.
    calls[0].join(timeout=30)
    assert not calls[0].is_alive()
    assert len(calls) == 1


def test_a_batch_that_is_not_abandoned_still_retries() -> None:
    calls: list[int] = []

    def flaky(_: dict[str, Any]) -> Any:
        calls.append(1)
        raise OutputParserException("malformed")

    processor = BatchProcessor(
        max_concurrency=1, max_attempts=3, retry_multiplier=1.0, retry_max_wait=0
    )
    results = processor.execute_with_fallback(
        [{"x": 1}], lambda items: items, flaky, "task", show_progress=False
    )
    assert results == [BATCH_ITEM_FAILED]
    assert len(calls) == 3


@pytest.mark.usefixtures("sigint_raises")
def test_an_abandoned_batch_cuts_a_backoff_wait_short() -> None:
    # The backoff waits on the abandonment flag, not in time.sleep: an item
    # backing off when the batch is abandoned ends at once, not after the
    # wait (here up to a minute).
    import time

    calls: list[threading.Thread] = []
    failed = threading.Event()

    def fails(_: dict[str, Any]) -> Any:
        calls.append(threading.current_thread())
        failed.set()
        raise OutputParserException("malformed")

    def interrupt_during_the_backoff() -> None:
        failed.wait(10)
        time.sleep(0.3)  # the item is backing off now
        os.kill(os.getpid(), signal.SIGINT)

    processor = BatchProcessor(
        max_concurrency=1,
        max_attempts=3,
        retry_multiplier=60.0,
        retry_max_wait=60,
        call_timeout_seconds=0,
    )
    threading.Thread(target=interrupt_during_the_backoff).start()
    with pytest.raises(KeyboardInterrupt):
        processor.execute_with_fallback(
            [{"x": 1}], lambda items: items, fails, "task", show_progress=False
        )
    started = time.time()
    calls[0].join(timeout=10)
    assert not calls[0].is_alive()
    assert time.time() - started < 5
    assert len(calls) == 1
