# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An abandoned batch makes no further attempt (AWS-free)."""

from __future__ import annotations

import _thread
import threading
import time
from typing import Any

import pytest
from langchain_core.exceptions import OutputParserException

from unified_kg_rag.shared.utils.langchain import BATCH_ITEM_FAILED, BatchProcessor

pytestmark = pytest.mark.unit


def test_an_abandoned_batch_makes_no_further_attempt() -> None:
    # A stop (SIGTERM) or Ctrl-C unwinds the calling thread; an item still
    # running must not go on retrying behind it (interpreter exit waits for
    # the pool's threads, so the task would be killed mid-shutdown).
    calls: list[int] = []
    started = threading.Event()

    def flaky(_: dict[str, Any]) -> Any:
        calls.append(1)
        if not started.is_set():
            started.set()
            _thread.interrupt_main()
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
    # Three more attempts, each after a wait of at most 1 s, would be made by now.
    time.sleep(3.5)
    assert calls == [1]


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
