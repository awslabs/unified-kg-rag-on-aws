# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stub for botocore's retry backoff that no other thread runs."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

import pytest


class _EndpointTime:
    """The ``time`` module as ``botocore.endpoint`` sees it, its ``sleep``
    replaced."""

    def __init__(self, sleep: Callable[[float], Any]) -> None:
        self.sleep = sleep

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


def stub_botocore_backoff(
    monkeypatch: pytest.MonkeyPatch,
    stub: Callable[[float], Any] = lambda _seconds: None,
    *,
    calling_thread_only: bool = False,
) -> None:
    """Run ``stub(seconds)`` where botocore's endpoint waits before a resend.

    ``botocore.endpoint.time`` is the process's ``time`` module, so setting
    ``botocore.endpoint.time.sleep`` replaces ``time.sleep`` for every
    thread of the test process: another thread's wait (a library's poll, a
    worker an earlier test left running) then runs the stub, and a stub that
    stops a query stops it before its first request is sent. Only the
    endpoint's own module reference is replaced here. With
    ``calling_thread_only`` only the calling thread's resends run the stub;
    any other thread's botocore backoff sleeps as usual.
    """
    import botocore.endpoint

    sleep = stub
    if calling_thread_only:
        caller = threading.get_ident()

        def sleep(seconds: float) -> Any:
            if threading.get_ident() == caller:
                return stub(seconds)
            return time.sleep(seconds)

    monkeypatch.setattr(botocore.endpoint, "time", _EndpointTime(sleep))
