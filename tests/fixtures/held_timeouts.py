# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deadlines a test expires on demand instead of by the wall clock."""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
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


class HeldCallTimeouts:
    """Batch call timeouts (``BatchProcessor._run_with_timeout``) that expire
    on cue: a timed call that calls :meth:`hang` times out at once, and any
    other runs to its end however loaded the runner (the call timeout the
    test configures, long, is only a backstop)."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from unified_kg_rag.adapters.llm import langchain as langchain_mod
        from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor

        self.release = threading.Event()
        # A timed call's future -> the future that expires it.
        self._expiries: dict[Future[Any], Future[None]] = {}
        self._current = threading.local()
        held = self

        class _Pool(ContextThreadPoolExecutor):
            def submit(
                self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any
            ) -> Future[Any]:
                expiry: Future[None] = Future()
                registered = threading.Event()

                def timed() -> Any:
                    registered.wait()
                    held._current.expiry = expiry
                    return fn(*args, **kwargs)

                future: Future[Any] = super().submit(timed)
                held._expiries[future] = expiry
                registered.set()
                return future

        result = Future.result

        def timed_result(future: Future[Any], timeout: float | None = None) -> Any:
            expiry = self._expiries.get(future)
            if timeout is None or expiry is None:
                return result(future, timeout)
            concurrent.futures.wait(
                [future, expiry],
                timeout=timeout,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not future.done():
                raise TimeoutError
            return result(future)

        monkeypatch.setattr(langchain_mod, "ContextThreadPoolExecutor", _Pool)
        monkeypatch.setattr(Future, "result", timed_result)

    def hang(self) -> None:
        """In a timed call: time it out now, then hang until :attr:`release`
        is set (bounded)."""
        self._current.expiry.set_result(None)
        self.release.wait(30)

    def settle(self) -> None:
        """Wait (bounded) until every call that timed out has returned."""
        abandoned = [f for f, expiry in self._expiries.items() if expiry.done()]
        _, running = concurrent.futures.wait(abandoned, timeout=30)
        assert not running, "a timed-out call never returned"
