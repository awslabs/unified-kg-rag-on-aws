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
import threading
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.output_parsers import StrOutputParser

from unified_kg_rag.application.retrieval import rag_chain as rag_chain_module
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, _LoopRunner
from unified_kg_rag.domain.models import Config, RetrieverRole
from unified_kg_rag.domain.prompts import (
    AnswerGenerationPrompt,
    StrategySelectionPrompt,
)
from unified_kg_rag.shared import StoreConnectionError

pytestmark = pytest.mark.unit


class _Retriever:
    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.closed = 0
        self.aclosed = 0
        self.aclose_loop: asyncio.AbstractEventLoop | None = None

    def close(self) -> None:
        self.closed += 1

    async def aclose(self) -> None:
        self.aclosed += 1
        self.aclose_loop = asyncio.get_running_loop()


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

    chain = GraphRAGChain(
        config=Config(),
        boto_session=MagicMock(),
        retriever_builders={RetrieverRole.DOCUMENT: build},  # type: ignore[dict-item]
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
    """Closes like a Neptune client whose websocket close hangs."""

    release = threading.Event()

    def close(self) -> None:
        assert self.release.wait(timeout=10)
        super().close()


def test_evicting_a_slowly_closing_retriever_does_not_stall_the_query() -> None:
    # Regression: the eviction on the query path joined the close thread for
    # up to 30 s, so a slow Neptune close stalled the next query.
    _SlowClosingRetriever.release.clear()
    chain, built = _chain(_SlowClosingRetriever)
    try:
        _run_on_fresh_loop(chain.ainvoke({"query": "q"}))  # type: ignore[arg-type]
        started = time.monotonic()
        _run_on_fresh_loop(chain.ainvoke({"query": "q"}))  # type: ignore[arg-type]
        assert time.monotonic() - started < 2
        assert built[0].closed == 0  # still closing in the background
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
    slow, slow_built = _chain(_SlowClosingRetriever)
    other, _ = _chain()
    try:
        _run_on_fresh_loop(slow.ainvoke({"query": "q"}))  # type: ignore[arg-type]
        _run_on_fresh_loop(slow.ainvoke({"query": "q"}))  # type: ignore[arg-type]
        assert slow_built[0].closed == 0  # still closing in the background
        _run_on_fresh_loop(other.ainvoke({"query": "q"}))  # type: ignore[arg-type]

        started = time.monotonic()
        other.close()
        assert time.monotonic() - started < 2
        started = time.monotonic()
        _run_on_fresh_loop(other.aclose())
        assert time.monotonic() - started < 2
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

    assert len(built) == 2
    # The runner-bound retriever is closed on its own loop ...
    assert built[0].aclosed == 1 and built[0].aclose_loop is built[0].loop
    # ... the one whose loop is gone, synchronously.
    assert built[1].closed == 1
    assert not thread.is_alive()


def test_runner_rejects_reentrant_sync_call() -> None:
    runner = _LoopRunner()

    async def reenter() -> None:
        runner.run(asyncio.sleep(0))

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
    chain = GraphRAGChain(config=Config(), boto_session=MagicMock())
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


def test_stop_cancels_a_task_no_call_awaits() -> None:
    # A task a call spawned and left running (a shared embedding future,
    # say) is cancelled and ended before the loop stops, not waited for.
    runner = _LoopRunner()
    spawned: list[asyncio.Task[None]] = []

    async def spawn() -> None:
        spawned.append(asyncio.ensure_future(asyncio.Event().wait()))

    runner.run(spawn())
    started = time.monotonic()
    runner.stop()
    assert spawned[0].cancelled()
    assert time.monotonic() - started < 5


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
    deadline = time.monotonic() + 20
    try:
        while time.monotonic() < deadline and any(t.is_alive() for t in threads):
            runner.stop()
            threads[0].join(timeout=0.001)
    finally:
        runner.stop()
    assert not any(thread.is_alive() for thread in threads)
    assert len(outcomes) == 400
