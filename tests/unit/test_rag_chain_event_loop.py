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
import gc
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
    # Not asyncio.run: once nest_asyncio is applied (the CLI modules do so at
    # import), asyncio.run reuses one loop instead of creating a new one.
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _chain() -> tuple[GraphRAGChain, list[_Retriever]]:
    built: list[_Retriever] = []

    def build() -> _Retriever:
        retriever = _Retriever()
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
    # is gone, so synchronously), and identity is not fooled by id reuse.
    assert len(built) == 5
    assert [r.closed for r in built] == [1, 1, 1, 1, 0]
    chain.close()
    assert built[-1].closed == 1


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
