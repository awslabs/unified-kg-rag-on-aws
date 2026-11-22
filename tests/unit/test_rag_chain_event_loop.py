# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Event-loop ownership of GraphRAGChain's sync API and retriever cache.

Regression: ``invoke`` ran ``asyncio.run`` per call (so ``batch(6)`` built six
retriever sets on six loops and closed none), raised inside a running loop, and
the cache was keyed by ``id(loop)``, which a new loop can reuse. The sync API
now runs on one chain-owned loop; each loop keeps its own retrievers until that
loop closes (or the chain does), so a query on one loop never closes clients an
in-flight query on another loop is using.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.output_parsers import StrOutputParser

from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.retrieval import rag_chain as rag_chain_module
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, _LoopRunner
from unified_kg_rag.domain.models import Config, RetrieverRole
from unified_kg_rag.domain.prompts import (
    AnswerGenerationPrompt,
    StrategySelectionPrompt,
)
from unified_kg_rag.shared import StoreConnectionError

pytestmark = pytest.mark.unit


# A hang guard, not a timing claim: every wait below ends on a cue the code
# under test sets (a close, a thread's end), so only a hang reaches it.
BACKSTOP = 60.0


class _Retriever:
    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.closed = 0
        self.aclosed = 0
        self.aclose_loop: asyncio.AbstractEventLoop | None = None
        # Set by either close: the cue that the release reached it.
        self.released = threading.Event()

    def close(self) -> None:
        self.closed += 1
        self.released.set()

    async def aclose(self) -> None:
        self.aclosed += 1
        self.aclose_loop = asyncio.get_running_loop()
        self.released.set()


def _run_on_fresh_loop(coro: Any) -> Any:
    # A loop of the test's own, so the chain sees a second, distinct loop.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _chain(
    retriever_type: type[_Retriever] = _Retriever,
) -> tuple[GraphRAGChain, list[_Retriever]]:
    built: list[_Retriever] = []

    def build() -> _Retriever:
        retriever = retriever_type()
        built.append(retriever)
        return retriever

    config = Config()
    chain = GraphRAGChain(
        config=config,
        providers=Providers(
            config,
            boto_session=MagicMock(),
            retriever_builders={RetrieverRole.DOCUMENT: build},  # type: ignore[dict-item]
        ),
    )

    async def fake_ainvoke(input: Any, config: Any = None, **kwargs: Any) -> Any:
        retriever = chain._get_retriever(RetrieverRole.DOCUMENT)
        await asyncio.sleep(0)
        return {"loop": asyncio.get_running_loop(), "retriever": retriever}

    chain.ainvoke = fake_ainvoke  # type: ignore[method-assign]
    return chain, built


def test_invoke_and_batch_share_one_loop_and_retriever() -> None:
    chain, built = _chain()
    try:
        first = chain.invoke({"query": "q1"})  # type: ignore[arg-type]
        results = chain.batch([{"query": f"q{i}"} for i in range(6)])  # type: ignore[misc]
    finally:
        chain.close()

    loops = {first["loop"], *(r["loop"] for r in results)}  # type: ignore[index]
    assert len(loops) == 1
    assert len(built) == 1  # one retriever for 7 calls
    assert built[0].closed == 0


async def test_invoke_works_inside_a_running_loop() -> None:
    chain, built = _chain()
    try:
        result = chain.invoke({"query": "q"})  # type: ignore[arg-type]
    finally:
        await chain.aclose()
    assert result["loop"] is not asyncio.get_running_loop()  # type: ignore[index]
    assert len(built) == 1


def test_loop_change_closes_evicted_retrievers() -> None:
    chain, built = _chain()
    for _ in range(5):
        _run_on_fresh_loop(chain.ainvoke({"query": "q"}))  # type: ignore[arg-type]

    # Every loop got its own retriever; each evicted one was closed (its loop
    # is gone, so synchronously, in the background), and identity is not
    # fooled by id reuse.
    chain._background_closes.join()
    assert len(built) == 5
    assert [r.closed for r in built] == [1, 1, 1, 1, 0]
    chain.close()
    assert built[-1].closed == 1


class _LoopDrivingRetriever(_Retriever):
    """Closes like gremlinpython's transport: by driving a private loop."""

    def __init__(self) -> None:
        super().__init__()
        self._own_loop = asyncio.new_event_loop()

    def close(self) -> None:
        # Raises "Cannot run the event loop while another loop is running"
        # when called on a thread that runs a loop.
        self._own_loop.run_until_complete(asyncio.sleep(0))
        self._own_loop.close()
        super().close()


def test_evicted_retriever_whose_close_drives_a_loop_is_closed() -> None:
    chain, built = _chain(_LoopDrivingRetriever)
    for _ in range(3):
        _run_on_fresh_loop(chain.ainvoke({"query": "q"}))  # type: ignore[arg-type]

    # Evicted inside the next query's running loop, their blocking close
    # still runs to completion (on a worker thread).
    chain._background_closes.join()
    assert [r.closed for r in built] == [1, 1, 0]
    chain.close()
    assert built[-1].closed == 1


class _SlowClosingRetriever(_Retriever):
    """Closes like a Neptune client whose websocket close hangs; ``returned``
    counts the closes that returned (released or timed out)."""

    release = threading.Event()
    returned = 0

    def close(self) -> None:
        released = self.release.wait(timeout=10)
        type(self).returned += 1
        assert released
        super().close()


def test_evicting_a_slowly_closing_retriever_does_not_stall_the_query() -> None:
    # Regression: the eviction on the query path joined the close thread for
    # up to 30 s, so a slow Neptune close stalled the next query.
    _SlowClosingRetriever.release.clear()
    _SlowClosingRetriever.returned = 0
    chain, built = _chain(_SlowClosingRetriever)
    try:
        _run_on_fresh_loop(chain.ainvoke({"query": "q"}))  # type: ignore[arg-type]
        _run_on_fresh_loop(chain.ainvoke({"query": "q"}))  # type: ignore[arg-type]
        # Still closing in the background: the query did not wait for it.
        assert _SlowClosingRetriever.returned == 0
    finally:
        _SlowClosingRetriever.release.set()
        chain.close()

    # Chain close waits for the background close.
    assert [r.closed for r in built] == [1, 1]


def test_chain_close_does_not_wait_for_another_chains_background_close() -> None:
    # Regression: background closes were tracked process-wide, so closing
    # (or GC-finalizing) one chain waited up to 30 s for another chain's slow
    # close, which could block the event loop the finalizer ran on.
    _SlowClosingRetriever.release.clear()
    _SlowClosingRetriever.returned = 0
    slow, slow_built = _chain(_SlowClosingRetriever)
    other, _ = _chain()
    try:
        _run_on_fresh_loop(slow.ainvoke({"query": "q"}))  # type: ignore[arg-type]
        _run_on_fresh_loop(slow.ainvoke({"query": "q"}))  # type: ignore[arg-type]
        assert slow_built[0].closed == 0  # still closing in the background
        _run_on_fresh_loop(other.ainvoke({"query": "q"}))  # type: ignore[arg-type]

        # Neither waits for the other chain's close, still in flight.
        other.close()
        assert _SlowClosingRetriever.returned == 0
        _run_on_fresh_loop(other.aclose())
        assert _SlowClosingRetriever.returned == 0
    finally:
        _SlowClosingRetriever.release.set()
        slow.close()
    assert [r.closed for r in slow_built] == [1, 1]


async def test_aclose_closes_retrievers_of_a_closed_loop_off_the_loop() -> None:
    chain = GraphRAGChain.__new__(GraphRAGChain)
    dead = asyncio.new_event_loop()
    dead.close()
    retriever = _LoopDrivingRetriever()
    chain._retriever_cache = {("document", dead): retriever}  # type: ignore[dict-item]
    await chain.aclose()
    assert retriever.closed == 1


def test_retriever_on_a_live_loop_is_kept_then_closed_there() -> None:
    chain, built = _chain()
    other = _LoopRunner()
    try:
        other.run(chain.ainvoke({"query": "q"}))  # type: ignore[arg-type]
        _run_on_fresh_loop(chain.ainvoke({"query": "q"}))  # type: ignore[arg-type]
        # A query on another loop leaves the live loop's retriever alone ...
        assert built[0].aclosed == 0 and built[0].closed == 0
        # ... and the next query on that loop reuses it.
        other.run(chain.ainvoke({"query": "q"}))  # type: ignore[arg-type]
        assert len(built) == 2
        chain.close()
        # Chain close releases it on its own (still running) loop.
        assert built[0].aclosed == 1
        assert built[0].aclose_loop is built[0].loop
        assert built[0].closed == 0
    finally:
        other.stop()
        chain.close()


async def test_aclose_closes_runner_bound_retrievers_on_their_loop() -> None:
    chain, built = _chain()
    chain.invoke({"query": "q"})  # type: ignore[arg-type]
    runner_thread = chain._loop_runner._thread
    assert runner_thread is not None and runner_thread.is_alive()

    await chain.aclose()

    assert built[0].aclosed == 1
    assert built[0].aclose_loop is built[0].loop
    assert not runner_thread.is_alive()
    assert chain._retriever_cache == {}


def test_close_stops_the_loop_thread_and_restarts_on_demand() -> None:
    chain, built = _chain()
    chain.invoke({"query": "q"})  # type: ignore[arg-type]
    thread = chain._loop_runner._thread
    chain.close()
    assert thread is not None and not thread.is_alive()
    assert built[0].aclosed == 1  # closed on its (still running) loop

    chain.invoke({"query": "q"})  # type: ignore[arg-type]
    assert chain._loop_runner._thread is not thread
    chain.close()


def test_dropped_chain_releases_its_retrievers_when_collected() -> None:
    # A chain dropped without close()/aclose() must not leak its retrievers'
    # sockets: the GC finalizer closes them (on their still-running loop) and
    # then stops the sync-API loop.
    chain, built = _chain()
    chain.invoke({"query": "q"})  # type: ignore[arg-type]
    _run_on_fresh_loop(chain.ainvoke({"query": "q"}))  # type: ignore[arg-type]
    thread = chain._loop_runner._thread
    assert thread is not None

    del chain
    gc.collect()

    # Released on a thread of its own (the collector may run anywhere),
    # after any other dropped chain the collection found.
    assert all(r.released.wait(BACKSTOP) for r in built)
    thread.join(BACKSTOP)
    assert not thread.is_alive()
    assert len(built) == 2
    # The runner-bound retriever is closed on its own loop ...
    assert built[0].aclosed == 1 and built[0].closed == 0
    assert built[0].aclose_loop is built[0].loop
    # ... the one whose loop is gone, synchronously.
    assert built[1].closed == 1 and built[1].aclosed == 0


class _ThreadClosingRetriever(_Retriever):
    """Closes through the loop's executor, as the embedded stores do."""

    async def aclose(self) -> None:
        await asyncio.to_thread(lambda: None)
        await super().aclose()


def test_collecting_a_dropped_chain_never_waits_where_it_is_collected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression (CI python-3-13, a 120 s hang): the collector ran a dropped
    # chain's finalizer on another chain's loop thread inside
    # ThreadPoolExecutor.submit, under the executors' global lock; the
    # finalizer waited up to 30 s for the dropped chain's loop, whose
    # retriever close needed that lock to submit its own work. The
    # finalizer must return at once wherever it runs.
    import concurrent.futures.thread as pool_threads

    # A finalizer that waits for the release would now wait past the
    # backstop instead of 30 s, so the test tells "returns" from "waits"
    # by cues alone, not by how long the collection took.
    monkeypatch.setattr(rag_chain_module, "_TEARDOWN_TIMEOUT_SECONDS", 10 * BACKSTOP)
    chain, built = _chain(_ThreadClosingRetriever)
    chain.invoke({"query": "q"})  # type: ignore[arg-type]
    dropped = [chain]
    del chain
    collected, unlock = threading.Event(), threading.Event()

    def collect_under_the_lock() -> None:
        with pool_threads._global_shutdown_lock:
            dropped.clear()
            gc.collect()
            collected.set()
            unlock.wait(BACKSTOP)

    collector = threading.Thread(target=collect_under_the_lock, daemon=True)
    collector.start()
    try:
        assert collected.wait(BACKSTOP), "the collection waited for the release"
        # The close needs the lock the collector holds: not released yet.
        assert built[0].aclosed == 0
    finally:
        unlock.set()
        if collected.is_set():  # else it is stuck in the collection
            collector.join(BACKSTOP)
    assert built[0].released.wait(BACKSTOP)
    assert built[0].aclosed == 1


_COLLECTED_UNDER_THE_SHUTDOWN_LOCK = """
import gc, threading
from tests.unit.test_rag_chain_event_loop import BACKSTOP, _chain, _ThreadClosingRetriever
chain, built = _chain(_ThreadClosingRetriever)
chain.invoke({"query": "q"})
with threading._shutdown_locks_lock:
    del chain
    gc.collect()
if built[0].released.wait(BACKSTOP) and built[0].aclosed == 1:
    print("released")
"""


@pytest.mark.skipif(
    not hasattr(threading, "_shutdown_locks_lock"),
    reason="Python 3.13+ has no lock of the threads to join at shutdown",
)
def test_collecting_a_dropped_chain_starts_no_thread_where_it_is_collected() -> None:
    # rv33 (3.11/3.12): the finalizer started a thread, whose start waits
    # for it to run, which takes threading's _shutdown_locks_lock: a chain
    # collected by a thread holding that lock (every non-daemon thread's
    # start and join takes it, and allocates under it) hung forever. In a
    # child process, so a hang fails the test instead of the run.
    done = subprocess.run(
        [sys.executable, "-c", _COLLECTED_UNDER_THE_SHUTDOWN_LOCK],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        # No AWS: the child's chain is built on fakes, its env holds no
        # credentials either.
        env={
            **{k: v for k, v in os.environ.items() if not k.startswith("AWS_")},
            "AWS_EC2_METADATA_DISABLED": "true",
            "AWS_CONFIG_FILE": os.devnull,
            "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
        },
    )
    assert done.stdout.strip() == "released", done.stderr[-2000:]


class _EndingLoop(asyncio.SelectorEventLoop):
    """A loop on a thread of its own that can be made to end with a close
    submitted to it still queued: :meth:`hold` blocks it, ``submitted`` is
    set by the next submission from another thread, and :meth:`end_queued`
    stops it before it runs that submission, then closes it."""

    def __init__(self) -> None:
        super().__init__()
        self.submitted = threading.Event()
        self._held = threading.Event()
        self._thread = threading.Thread(target=self.run_forever, daemon=True)
        self._thread.start()

    def call_soon_threadsafe(  # type: ignore[override]
        self, callback: Any, *args: Any, **kwargs: Any
    ) -> Any:
        handle = super().call_soon_threadsafe(callback, *args, **kwargs)
        self.submitted.set()
        return handle

    def hold(self) -> None:
        def held() -> None:
            self._held.wait(BACKSTOP)
            self.stop()  # this iteration is the last: the queue is left

        super().call_soon_threadsafe(held)
        self.submitted.clear()

    def end_queued(self) -> None:
        self._held.set()
        self._thread.join(BACKSTOP)
        self.close()


def _closing_on(close: Any) -> tuple[_EndingLoop, list[_Retriever], threading.Thread]:
    """A chain whose retriever is bound to an :class:`_EndingLoop`, closed by
    ``close(chain)`` on another thread, the loop ending with that close
    queued on it."""
    loop = _EndingLoop()
    chain, built = _chain()
    query = chain.ainvoke({"query": "q"})  # type: ignore[arg-type]
    asyncio.run_coroutine_threadsafe(query, loop).result(BACKSTOP)
    loop.hold()
    closing = threading.Thread(target=close, args=(chain,), daemon=True)
    closing.start()
    assert loop.submitted.wait(BACKSTOP)
    loop.end_queued()
    return loop, built, closing


@pytest.mark.parametrize(
    "close",
    [GraphRAGChain.close, lambda chain: asyncio.run(chain.aclose())],
    ids=["close", "aclose"],
)
def test_retrievers_whose_loop_ends_before_their_close_runs_are_closed(
    close: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression (CI python-3-13, "coroutine '_aclose_retrievers' was never
    # awaited"): a close submitted to a loop that then ended never ran nor
    # resolved, so the retrievers stayed open and the closer waited out the
    # whole teardown timeout (now past the backstop). They are closed here
    # instead, once.
    monkeypatch.setattr(rag_chain_module, "_TEARDOWN_TIMEOUT_SECONDS", 10 * BACKSTOP)
    _, built, closing = _closing_on(close)
    assert built[0].released.wait(BACKSTOP)
    closing.join(BACKSTOP)
    assert not closing.is_alive()
    assert built[0].closed == 1 and built[0].aclosed == 0


def test_a_collected_chain_whose_loop_ended_holds_up_no_later_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The same on the release worker, which releases every dropped chain one
    # after another: it waited out the teardown timeout before the next one.
    monkeypatch.setattr(rag_chain_module, "_TEARDOWN_TIMEOUT_SECONDS", 10 * BACKSTOP)

    def drop(chain: GraphRAGChain) -> None:
        rag_chain_module._release_collected(
            chain._retriever_cache,
            chain._strategy_cache,
            chain._cache_lock,
            chain._loop_runner,
            chain._background_closes,
        )

    _, first, _ = _closing_on(drop)
    later, built = _chain()
    later.invoke({"query": "q"})  # type: ignore[arg-type]
    drop(later)
    assert built[0].released.wait(BACKSTOP)
    assert first[0].released.is_set()
    assert first[0].closed == 1 and first[0].aclosed == 0
    later.close()


class _LoopBound:
    """A retriever that counts its closes; ``gate`` (when given) holds its
    ``aclose`` open after it starts, on the loop it runs on."""

    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.gate = gate
        self.closed = 0
        self.aclosed = 0
        self.started = threading.Event()

    def close(self) -> None:
        self.closed += 1

    async def aclose(self) -> None:
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        self.aclosed += 1


def _close_on_thread(
    retriever: _LoopBound, loop: asyncio.AbstractEventLoop
) -> tuple[threading.Thread, list[bool]]:
    """``_aclose_on_their_loop`` (waiting) on a thread of its own; its
    answer lands in the returned list."""
    answer: list[bool] = []
    closer = threading.Thread(
        target=lambda: answer.append(
            rag_chain_module._aclose_on_their_loop([retriever], loop, wait=True)  # type: ignore[list-item]
        ),
        daemon=True,
    )
    closer.start()
    return closer, answer


def test_a_close_queued_on_a_loop_that_stops_is_cancelled_there(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A loop that stops (not closed) with the close still queued: the caller
    # closes the retrievers itself (False), and the queued close is
    # cancelled, so running that loop again closes nothing twice.
    monkeypatch.setattr(rag_chain_module, "_TEARDOWN_TIMEOUT_SECONDS", 10 * BACKSTOP)
    submitted: list[concurrent.futures.Future[None]] = []
    submit = asyncio.run_coroutine_threadsafe

    def spy(coro: Any, loop: asyncio.AbstractEventLoop) -> Any:
        submitted.append(submit(coro, loop))
        return submitted[-1]

    monkeypatch.setattr(rag_chain_module.asyncio, "run_coroutine_threadsafe", spy)
    loop = _EndingLoop()
    retriever = _LoopBound()
    try:
        loop.hold()
        closer, answer = _close_on_thread(retriever, loop)
        assert loop.submitted.wait(BACKSTOP)
        loop._held.set()  # the held iteration stops the loop, queue left
        loop._thread.join(BACKSTOP)
        closer.join(BACKSTOP)
        assert answer == [False]
        assert submitted[0].cancelled()
        loop.run_until_complete(asyncio.sleep(0))  # the loop run again
        assert retriever.aclosed == 0 and not retriever.started.is_set()
    finally:
        loop.close()


def test_a_close_that_started_on_a_loop_that_then_stops_is_left_to_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The loop took the close (it holds the claim) and stopped before the
    # close ended: the caller must not close the retrievers as well (True).
    monkeypatch.setattr(rag_chain_module, "_TEARDOWN_TIMEOUT_SECONDS", 10 * BACKSTOP)
    loop = asyncio.new_event_loop()
    running = threading.Thread(target=loop.run_forever, daemon=True)
    running.start()
    gate = asyncio.Event()
    retriever = _LoopBound(gate)
    try:
        closer, answer = _close_on_thread(retriever, loop)
        assert retriever.started.wait(BACKSTOP)
        loop.call_soon_threadsafe(loop.stop)
        running.join(BACKSTOP)
        closer.join(BACKSTOP)
        assert answer == [True]
        assert retriever.closed == 0
        loop.run_until_complete(_open(gate))  # the started close ends there
        assert retriever.aclosed == 1 and retriever.closed == 0
    finally:
        loop.close()


async def _open(gate: asyncio.Event) -> None:
    gate.set()
    for _ in range(3):
        await asyncio.sleep(0)


def test_a_close_still_running_past_the_teardown_timeout_stops_the_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A close slower than the teardown timeout on a loop that keeps running:
    # the waiter returns then (True: it ends there), not when the close does.
    timeout = 0.2
    monkeypatch.setattr(rag_chain_module, "_TEARDOWN_TIMEOUT_SECONDS", timeout)
    loop = asyncio.new_event_loop()
    running = threading.Thread(target=loop.run_forever, daemon=True)
    running.start()
    gate = asyncio.Event()
    retriever = _LoopBound(gate)
    try:
        closer, answer = _close_on_thread(retriever, loop)
        closer.join(25 * timeout)  # a hang guard: it ends at the timeout
        assert not closer.is_alive(), "the wait outlived the teardown timeout"
        assert answer == [True]
        assert retriever.closed == 0 and retriever.aclosed == 0
    finally:
        asyncio.run_coroutine_threadsafe(_open(gate), loop).result(BACKSTOP)
        loop.call_soon_threadsafe(loop.stop)
        running.join(BACKSTOP)
        closer.join(BACKSTOP)
        loop.close()
    assert retriever.aclosed == 1


def test_runner_rejects_reentrant_sync_call() -> None:
    runner = _LoopRunner()

    async def reenter() -> None:
        pending = asyncio.sleep(0)
        try:
            runner.run(pending)
        finally:
            pending.close()  # refused, so never awaited

    try:
        with pytest.raises(RuntimeError, match="own event loop"):
            runner.run(reenter())
    finally:
        runner.stop()


def test_runner_stop_is_idempotent() -> None:
    runner = _LoopRunner()
    runner.stop()
    assert runner.run(asyncio.sleep(0, result=7)) == 7
    runner.stop()
    runner.stop()


def test_stop_from_inside_the_loop_ends_its_tasks_and_closes_it() -> None:
    # Regression: a stop from a task on the loop only scheduled loop.stop, so
    # the loop halted with the tasks it had cancelled still pending (each
    # later destroyed mid-flight, "Task was destroyed but it is pending") and
    # was never closed.
    runner = _LoopRunner()
    unwound = threading.Event()
    seen: list[Any] = []

    async def background() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0)  # cleanup that awaits
            unwound.set()
            raise

    async def stop_inside() -> None:
        loop = asyncio.get_running_loop()
        seen.extend([loop, threading.current_thread(), loop.create_task(background())])
        await asyncio.sleep(0)
        runner.stop()
        await asyncio.sleep(0.01)

    with pytest.raises(StoreConnectionError):
        runner.run(stop_inside())
    loop, thread, task = seen
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert unwound.is_set()
    assert task.cancelled()
    assert loop.is_closed()
    # The runner starts a fresh loop on the next call.
    assert runner.run(asyncio.sleep(0, result=7)) == 7
    runner.stop()


async def test_release_skips_retrievers_without_close() -> None:
    retriever = MagicMock(spec=[])  # no close/aclose attributes
    GraphRAGChain._release_retrievers([retriever], None, wait=True)
    aclosing = MagicMock()
    aclosing.aclose = AsyncMock(side_effect=RuntimeError("boom"))
    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain._retriever_cache = {("document", None): aclosing}  # type: ignore[dict-item]
    await chain.aclose()  # never raises
    aclosing.aclose.assert_awaited_once()


async def test_query_on_another_loop_does_not_close_in_flight_retrievers() -> None:
    # invoke() runs on the chain's loop thread while ainvoke() runs on the
    # caller's loop. The second loop must get its own retriever without closing
    # the one an in-flight query on the first loop is still using.
    chain, built = _chain()
    started = asyncio.Event()
    caller_loop = asyncio.get_running_loop()

    async def fake_ainvoke(input: Any, config: Any = None, **kwargs: Any) -> Any:
        retriever = chain._get_retriever(RetrieverRole.DOCUMENT)
        if input["query"] == "slow":
            caller_loop.call_soon_threadsafe(started.set)
            await asyncio.sleep(0.2)
        return retriever.closed + retriever.aclosed

    chain.ainvoke = fake_ainvoke  # type: ignore[method-assign]
    try:
        slow = asyncio.create_task(
            asyncio.to_thread(chain.invoke, {"query": "slow"})  # type: ignore[arg-type]
        )
        await started.wait()
        fast_closed = await chain.ainvoke({"query": "fast"})  # type: ignore[arg-type]
        slow_closed = await slow
    finally:
        await chain.aclose()

    assert len(built) == 2
    assert built[0].loop is not built[1].loop
    assert slow_closed == 0
    assert fast_closed == 0
    # Both are released on chain close, each on its own loop when it is live.
    assert built[0].aclosed == 1 and built[0].aclose_loop is built[0].loop
    assert built[1].aclosed == 1


def test_prompt_chains_are_built_once_per_prompt_model_and_flags(mocker) -> None:
    # Each setup_chain builds the model and its boto clients (new connection
    # pools); doing it per query paid that on every request.
    setup = mocker.patch.object(
        rag_chain_module, "setup_chain", side_effect=lambda **_: object()
    )
    config = Config()
    chain = GraphRAGChain(
        config=config, providers=Providers(config, boto_session=MagicMock())
    )
    try:
        get = chain._get_chain_for_prompt
        first = get(AnswerGenerationPrompt, StrOutputParser(), enable_thinking=False)
        again = get(AnswerGenerationPrompt, StrOutputParser(), enable_thinking=False)
        thinking = get(AnswerGenerationPrompt, StrOutputParser(), enable_thinking=True)
        router = get(StrategySelectionPrompt, StrOutputParser())
    finally:
        chain.close()

    assert first is again
    assert len({id(first), id(thinking), id(router)}) == 3
    assert setup.call_count == 3


def test_each_prompt_runs_the_model_of_its_declared_role(mocker) -> None:
    from unified_kg_rag.domain.prompts import (
        ComprehensiveAnswerPrompt,
        EntityExtractionPrompt,
        KeywordsExtractionPrompt,
        QueryRewritePrompt,
        TranslationPrompt,
    )
    from unified_kg_rag.domain.prompts.base import BasePrompt

    setup = mocker.patch.object(
        rag_chain_module, "setup_chain", side_effect=lambda **_: object()
    )
    roles = {
        AnswerGenerationPrompt: "answer_generation",
        ComprehensiveAnswerPrompt: "answer_generation",
        EntityExtractionPrompt: "entity_extraction",
        KeywordsExtractionPrompt: "entity_extraction",
        QueryRewritePrompt: "query_rewrite",
        StrategySelectionPrompt: "strategy_selection",
        TranslationPrompt: "query_translation",
    }
    config = Config(models={"roles": {role: f"m-{role}" for role in roles.values()}})
    chain = GraphRAGChain(
        config=config, providers=Providers(config, boto_session=MagicMock())
    )
    try:
        for prompt in roles:
            chain._get_chain_for_prompt(prompt, StrOutputParser())

        class _Undeclared(BasePrompt):
            input_variables = ["query"]

        with pytest.raises(ValueError, match="model_role"):
            chain._get_chain_for_prompt(_Undeclared, StrOutputParser())
    finally:
        chain.close()
    models = [call.kwargs["model_id"] for call in setup.call_args_list]
    assert models == [f"m-{role}" for role in roles.values()]


def test_close_fails_sync_calls_in_flight_instead_of_hanging() -> None:
    # Regression: close() stopped the loop with the sync calls of other threads
    # still pending on it, so each of them waited forever for its result.
    chain, _ = _chain()
    chain.invoke({"query": "q"})  # type: ignore[arg-type]
    started = threading.Semaphore(0)
    unwound: list[str] = []

    async def stuck(input: Any, config: Any = None, **kwargs: Any) -> Any:
        started.release()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)  # cleanup that awaits (a client's aclose)
            unwound.append("cancelled")
            raise

    chain.ainvoke = stuck  # type: ignore[method-assign]
    outcomes: list[BaseException | None] = []

    def call() -> None:
        try:
            chain.invoke({"query": "q"})  # type: ignore[arg-type]
            outcomes.append(None)
        except BaseException as e:  # noqa: BLE001 - recorded for the assertion
            outcomes.append(e)

    threads = [threading.Thread(target=call, daemon=True) for _ in range(3)]
    for thread in threads:
        thread.start()
    for _ in threads:
        assert started.acquire(timeout=5)
    loop_thread = chain._loop_runner._thread

    chain.close()
    for thread in threads:
        thread.join(timeout=5)

    assert not any(thread.is_alive() for thread in threads)
    assert len(outcomes) == 3
    assert all(isinstance(e, StoreConnectionError) for e in outcomes), outcomes
    assert all("closed" in str(e) for e in outcomes)
    # The calls' tasks were cancelled and unwound before the loop stopped.
    assert unwound == ["cancelled"] * 3
    assert loop_thread is not None and not loop_thread.is_alive()


def test_sync_call_after_close_is_served_on_a_new_loop() -> None:
    # Closing fails only the calls in flight: the chain reconnects if used again.
    chain, _ = _chain()
    chain.invoke({"query": "q"})  # type: ignore[arg-type]
    chain.close()
    assert chain.invoke({"query": "q"})["loop"].is_running()  # type: ignore[arg-type]
    chain.close()


def test_runner_cancellation_inside_the_call_is_not_reported_as_close() -> None:
    runner = _LoopRunner()

    async def cancelled() -> None:
        raise asyncio.CancelledError

    try:
        with pytest.raises(concurrent.futures.CancelledError):
            runner.run(cancelled())
    finally:
        runner.stop()


def test_stream_stopped_by_close_does_not_restart_the_loop() -> None:
    runner = _LoopRunner()
    started = threading.Event()

    async def agen() -> Any:
        started.set()
        await asyncio.Event().wait()
        yield 1  # pragma: no cover - never reached

    outcome: list[BaseException] = []

    def consume() -> None:
        try:
            list(rag_chain_module._iterate_async_generator(agen(), runner))
        except BaseException as e:  # noqa: BLE001 - recorded for the assertion
            outcome.append(e)

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    assert started.wait(timeout=5)
    runner.stop()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(outcome) == 1 and isinstance(outcome[0], StoreConnectionError)
    assert runner._loop is None  # the iterator's finally did not restart it


def test_join_waits_for_a_close_registered_but_not_yet_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: start() registered its worker, released the lock, then
    # started it; a join() in between (chain close racing an eviction on the
    # query path) raised "cannot join thread before it is started", so
    # close() raised and never stopped the loop.
    background = rag_chain_module._BackgroundCloses()
    entered, release = threading.Event(), threading.Event()
    start = threading.Thread.start

    def gated_start(self: threading.Thread) -> None:
        if self.name == "kg-rag-retriever-close":
            entered.set()
            assert release.wait(timeout=5)
        start(self)

    monkeypatch.setattr(threading.Thread, "start", gated_start)
    starter = threading.Thread(target=background.start, args=([],), daemon=True)
    starter.start()
    assert entered.wait(timeout=5)
    errors: list[BaseException] = []

    def join() -> None:
        try:
            background.join(timeout=5)
        except BaseException as e:  # noqa: BLE001 - recorded for the assertion
            errors.append(e)

    joiner = threading.Thread(target=join, daemon=True)
    joiner.start()
    joiner.join(timeout=0.5)  # the join is in flight (or failed) by now
    release.set()
    starter.join(timeout=5)
    joiner.join(timeout=5)
    assert not joiner.is_alive() and errors == []


def test_close_and_aclose_stop_the_loop_even_if_waiting_for_closes_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain, _ = _chain()

    def failing_join(self: Any, timeout: float = 0.0) -> None:
        raise RuntimeError("join failed")

    monkeypatch.setattr(rag_chain_module._BackgroundCloses, "join", failing_join)
    for close in (chain.close, lambda: _run_on_fresh_loop(chain.aclose())):
        chain.invoke({"query": "q"})  # type: ignore[arg-type]
        thread = chain._loop_runner._thread
        close()  # never raises
        assert thread is not None and not thread.is_alive()


def test_stop_cancels_a_task_no_call_awaits(monkeypatch: pytest.MonkeyPatch) -> None:
    # A task a call spawned and left running (a shared embedding future,
    # say) is cancelled and ended before the loop stops, not waited for.
    # Waiting for it would now outlast the backstop (not 30 s), so a stop
    # that waits is told from one that does not by its end alone.
    monkeypatch.setattr(rag_chain_module, "_TEARDOWN_TIMEOUT_SECONDS", 10 * BACKSTOP)
    runner = _LoopRunner()
    spawned: list[asyncio.Task[None]] = []

    async def spawn() -> None:
        spawned.append(asyncio.ensure_future(asyncio.Event().wait()))

    runner.run(spawn())
    stopping = threading.Thread(target=runner.stop, daemon=True)
    stopping.start()
    stopping.join(BACKSTOP)
    assert not stopping.is_alive(), "stop waited for the task"
    assert spawned[0].cancelled()


def test_calls_racing_stop_never_hang() -> None:
    # A call finds (or starts) the loop and registers itself under one lock
    # hold, so a stop() racing it either fails it as closed or leaves it to
    # a restarted loop: it never waits on a stopped loop.
    runner = _LoopRunner()
    outcomes: list[str] = []

    def calls() -> None:
        for _ in range(100):
            call = asyncio.sleep(0)
            try:
                runner.run(call)
                outcomes.append("ok")
            except StoreConnectionError:
                call.close()  # cancelled before it ran
                outcomes.append("closed")

    threads = [threading.Thread(target=calls, daemon=True) for _ in range(4)]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + BACKSTOP
    try:
        while time.monotonic() < deadline and any(t.is_alive() for t in threads):
            runner.stop()
            threads[0].join(timeout=0.001)
    finally:
        runner.stop()
    assert not any(thread.is_alive() for thread in threads)
    assert len(outcomes) == 400


async def test_a_cpu_step_runs_beside_a_saturated_default_executor() -> None:
    """``to_cpu_thread`` has threads of its own: every thread of the default
    executor inside a blocking call does not delay it, and it runs with the
    caller's context."""
    import contextvars
    import threading

    from unified_kg_rag.shared.utils import event_loop

    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker")
    marker.set("caller")
    release = threading.Event()
    loop = asyncio.get_running_loop()
    executor = event_loop.configure_event_loop(loop, 2)
    try:
        held = [loop.run_in_executor(None, release.wait, 10) for _ in range(2)]
        # A third call to the default executor would queue behind them.
        queued = asyncio.ensure_future(asyncio.to_thread(lambda: "queued"))
        found = await asyncio.wait_for(
            event_loop.to_cpu_thread(
                lambda scale: (
                    marker.get(),
                    threading.current_thread().name,
                    scale,
                ),
                scale=3,
            ),
            timeout=5,
        )
        assert found[0] == "caller" and found[2] == 3
        assert found[1].startswith("graphrag-cpu")
        assert not queued.done()
    finally:
        release.set()
        await asyncio.gather(*held, queued)
        executor.shutdown(wait=False)


def test_the_cpu_pool_has_one_thread_per_available_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pool is sized by the CPUs the process may use (a container's
    quota, its affinity mask), not by the host's ``os.cpu_count()``."""
    import os

    from unified_kg_rag.shared.utils import common, event_loop

    monkeypatch.setattr(os, "cpu_count", lambda: 64)
    monkeypatch.setattr(common, "_available_cpu_count", lambda: 3)
    monkeypatch.setattr(event_loop, "_cpu_executor", None)
    pool = event_loop._cpu_pool()
    try:
        assert pool._max_workers == 3
        assert event_loop._cpu_pool() is pool
    finally:
        pool.shutdown(wait=False)


async def test_the_default_executor_takes_calls_after_a_saturated_block() -> None:
    """A query the block started goes on to its fusion (a default-executor
    call) after the block: the executor the block held must not be the
    loop's default by then, shut down."""
    from tests.fixtures.saturated_executor import saturated_default_executor

    async with saturated_default_executor():
        pass
    assert await asyncio.wait_for(asyncio.to_thread(lambda: "ran"), 5) == "ran"


def test_a_shut_down_cpu_pool_is_replaced_on_the_next_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing in the package shuts the CPU pool, but a host (or a test)
    may; the next step must get a new pool, not "cannot schedule new
    futures after shutdown", in any event loop of the process."""
    from unified_kg_rag.shared.utils import event_loop

    monkeypatch.setattr(event_loop, "_cpu_executor", None)
    first = event_loop._cpu_pool()
    first.shutdown(wait=True)
    try:
        # Two loops, one after the other, as two asyncio.run calls make.
        for _ in range(2):
            name = asyncio.run(
                event_loop.to_cpu_thread(lambda: threading.current_thread().name)
            )
            assert name.startswith("graphrag-cpu")
        assert event_loop._cpu_pool() is not first
    finally:
        event_loop._cpu_pool().shutdown(wait=False)
