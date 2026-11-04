# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deadlines a test expires on demand instead of by the wall clock."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest


class HeldTimeouts:
    """Records the ``asyncio.timeout`` scopes the chain opens, so a test can
    expire a long deadline at a moment it chooses (once the model call is in
    flight) instead of racing the setup before it on a slow runner."""

    class _Recorded(asyncio.Timeout):
        exited = False

        async def __aexit__(self, *exc: Any) -> bool | None:
            self.exited = True
            return await super().__aexit__(*exc)

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.opened: list[HeldTimeouts._Recorded] = []

        def opening(delay: float | None) -> asyncio.Timeout:
            loop = asyncio.get_running_loop()
            when = None if delay is None else loop.time() + delay
            timeout = self._Recorded(when)
            self.opened.append(timeout)
            return timeout

        monkeypatch.setattr(asyncio, "timeout", opening)

    def _held(self, beyond: float) -> list[asyncio.Timeout]:
        now = asyncio.get_running_loop().time()
        return [
            t
            for t in self.opened
            if not t.exited and (when := t.when()) is not None and when - now > beyond
        ]

    async def wait_open(self, count: int = 1, beyond: float = 10) -> None:
        """Wait (bounded) until exactly ``count`` scopes due more than
        ``beyond`` s from now are open."""
        deadline = time.monotonic() + 30
        while len(self._held(beyond)) != count:
            assert time.monotonic() < deadline, "timed out waiting for a scope"
            await asyncio.sleep(0.01)

    def expire(self, beyond: float = 10) -> None:
        """Expire every open scope due more than ``beyond`` s from now (the
        held deadlines; nothing else in the query waits that long)."""
        now = asyncio.get_running_loop().time()
        for timeout in self._held(beyond):
            timeout.reschedule(now)
