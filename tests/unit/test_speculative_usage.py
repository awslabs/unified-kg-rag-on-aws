# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A query's usage counts every model call it sent (AWS-free).

Regression: ``auto`` cancelled the speculative extraction its route did not
read, and the usage callback counted only calls that returned. A cancelled
task does not stop a request already sent (a Bedrock call runs on a worker
thread to the end), so the call was billed but missing from
``RAGOutput.metadata["usage"]``: offline, 12 of 20 calls of auto queries
routed to global. A call now counts from its start, at an estimate of its
input tokens until it answers (``estimated_calls``), so the query returns
without waiting for that extraction and leaves it running in the
background, to fill the extraction cache, until it answers, the query's
deadline passes or the chain closes. A call cancelled in flight counts the
same way.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from tests.fixtures.fakes.chat_models import fake_model_info
from tests.fixtures.held_timeouts import HeldTimeouts
from unified_kg_rag.adapters.local.token_counter import local_token_counter_factory
from unified_kg_rag.adapters.observability.usage_callback import UsageCallbackHandler
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.storage.filter_schema import IndexLimits
from unified_kg_rag.application.retrieval import rag_chain
from unified_kg_rag.application.retrieval.rag_chain import (
    DEFAULT_ERROR_MESSAGE,
    GraphRAGChain,
    RAGInput,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
    SectionType,
)
from unified_kg_rag.shared.model_responses import track_response
from unified_kg_rag.shared.query_stop import QueryStop, reader_stopped
from unified_kg_rag.shared.usage import usage_scope

pytestmark = pytest.mark.unit

_KEYWORDS = '{"high_level_keywords": ["supply"], "low_level_keywords": ["Vendor"]}'
# What a failing call's error says (model output, never logged above DEBUG).
_FAILURE_TEXT = "the model replied: Vendor owes Buyer 1200"


class _ThreadedChat(BaseChatModel):
    """Answers by prompt kind on a worker thread, as ChatBedrockConverse
    runs ``ainvoke``: a cancelled caller leaves the request running to the
    end. ``delays`` holds the seconds a kind takes, ``gates`` an event a
    kind waits for (on the worker thread) before it answers; ``started`` and
    ``served`` the calls sent and completed at the "server"."""

    route: str = "simple"
    delays: dict[str, float] = Field(default_factory=dict)
    gates: dict[str, threading.Event] = Field(default_factory=dict)
    # defaultdict[str, threading.Event]: set as a kind answers.
    answered: Any = Field(default_factory=lambda: defaultdict(threading.Event))
    started: list[str] = Field(default_factory=list)
    served: list[str] = Field(default_factory=list)
    cancelled: list[str] = Field(default_factory=list)
    # False: answer on the event loop instead (a cancel stops the request).
    threaded: bool = True
    # On the event loop: each call opens a response the model clients'
    # hooks track (shared.model_responses), read after its gate; one closed
    # before it is read fails the call, as a cut connection does (``cut``).
    tracked: bool = False
    cut: list[str] = Field(default_factory=list)
    # Whether the call's reader had stopped once its gate opened, per kind.
    stopped: dict[str, list[bool]] = Field(default_factory=lambda: defaultdict(list))
    # Kinds whose call fails after its gate (a model error quoting text).
    failing: set[str] = Field(default_factory=set)
    lock: Any = Field(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "threaded"

    @staticmethod
    def kind(messages: list[BaseMessage]) -> str:
        system = "".join(str(m.content) for m in messages if m.type == "system")
        if "search strategy selector" in system:
            return "router"
        if "keyword extractor" in system:
            return "keywords"
        return "answer"

    def _generate(self, messages: list[BaseMessage], *a: Any, **k: Any) -> ChatResult:
        kind = self.kind(messages)
        with self.lock:
            self.started.append(kind)
        if (gate := self.gates.get(kind)) is not None:
            # Bounded, so a failed test never leaves the thread behind.
            gate.wait(30)
        time.sleep(self.delays.get(kind, 0.0))
        if kind in self.failing:
            raise ValueError(_FAILURE_TEXT)
        return self._reply(kind)

    async def _agenerate(
        self, messages: list[BaseMessage], *a: Any, **k: Any
    ) -> ChatResult:
        if self.threaded:
            return await super()._agenerate(messages, *a, **k)
        kind = self.kind(messages)
        response = _Response()
        if self.tracked:
            with self.lock:
                self.started.append(kind)
            track_response(response)
        try:
            if (gate := self.gates.get(kind)) is not None:
                await asyncio.to_thread(gate.wait, 30)
            await asyncio.sleep(self.delays.get(kind, 0.0))
        except asyncio.CancelledError:
            self.cancelled.append(kind)
            raise
        self.stopped[kind].append(reader_stopped())
        if response.is_closed:
            self.cut.append(kind)
            raise ConnectionResetError("the response was closed before its body")
        response.is_closed = True
        return self._reply(kind)

    def _reply(self, kind: str) -> ChatResult:
        with self.lock:
            self.served.append(kind)
        self.answered[kind].set()
        content = {"router": self.route, "keywords": _KEYWORDS}.get(
            kind, "Vendor supplies Buyer."
        )
        message = AIMessage(
            content=content,
            usage_metadata={"input_tokens": 50, "output_tokens": 5, "total_tokens": 55},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


class _Response:
    """A model response as the model clients' HTTP hooks track it: open
    until its body is read."""

    def __init__(self) -> None:
        self.is_closed = False

    async def aclose(self) -> None:
        self.is_closed = True


class _Factory:
    def __init__(self, model: BaseChatModel) -> None:
        self.model = model

    def get_model(self, model_id: Any, **kwargs: Any) -> BaseChatModel:
        return self.model

    def get_model_info(self, model_id: Any) -> Any:
        return fake_model_info(model_id)


class _Retriever:
    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        return IndexLimits()

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                content="Vendor and Buyer signed the supply agreement.",
                score=0.9,
                source="supply.txt",
                retriever_type=SectionType.TEXT,
                metadata={"id": "doc-1", "text_unit_ids": []},
            )
        ]


def _chain(model: _ThreadedChat, **search: Any) -> GraphRAGChain:
    config = Config()
    config.search.auto_routable_strategies = [SearchStrategy.SIMPLE, SearchStrategy.MIX]
    config.search.query_extraction_cache_size = 0
    # No rerank model: nothing here may build an AWS client.
    config.providers.rerank.kind = "none"
    for key, value in search.items():
        setattr(config.search, key, value)
    translation = config.processing.translation
    translation.source_language = translation.target_language
    model.callbacks = [UsageCallbackHandler("m", role="query")]
    retriever = _Retriever()
    providers = Providers(
        config,
        llm_factory=_Factory(model),
        token_counter_factory=local_token_counter_factory(config),
        retriever_builders={RetrieverRole.DOCUMENT: lambda: retriever},
    )
    return GraphRAGChain(config=config, providers=providers)


def _auto() -> RAGInput:
    return RAGInput(query="What does Vendor supply?", search_strategy="auto")


async def _until(check: Any, within: float = 10.0) -> None:
    """Wait (bounded) until ``check()`` holds."""
    deadline = time.monotonic() + within
    while not check():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


@pytest.fixture
def release() -> Iterator[threading.Event]:
    """Holds a gated call at the "server" until set (at the end at the
    latest, so no worker thread outlives the test)."""
    event = threading.Event()
    yield event
    event.set()


class _HeldCancels:
    """Records the long ``loop.call_later`` timers that cancel a task on the
    running loop, so a test can fire a long deadline at a moment it chooses
    instead of racing the query to it on a slow runner."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, beyond: float = 100) -> None:
        loop = asyncio.get_running_loop()
        self.timers: list[tuple[float, asyncio.Future[Any], asyncio.TimerHandle]] = []
        real = loop.call_later

        def call_later(delay: float, callback: Any, *args: Any, **kw: Any) -> Any:
            handle = real(delay, callback, *args, **kw)
            task = getattr(callback, "__self__", None)
            if (
                delay > beyond
                and isinstance(task, asyncio.Future)
                and callback == task.cancel
            ):
                self.timers.append((delay, task, handle))
            return handle

        monkeypatch.setattr(loop, "call_later", call_later)

    def fire(self) -> None:
        for _, task, handle in self.timers:
            if not handle.cancelled():
                handle.cancel()
                task.cancel()


async def test_the_query_does_not_wait_for_the_unread_extraction(
    release: threading.Event,
) -> None:
    # Routed to simple, the keyword extraction mix would read runs on in the
    # background: the query returns at the answer, counting it from its start
    # at an estimate. The extraction is held at the "server" until the query
    # has returned (a query that waited for it times out here).
    model = _ThreadedChat(gates={"keywords": release})
    chain = _chain(model)
    out = await asyncio.wait_for(chain.ainvoke(_auto()), 30)
    assert "keywords" in model.started and "keywords" not in model.served
    (pending,) = chain._background_extractions._snapshot()
    assert not pending.done()
    assert out.metadata["search_strategy"] == "simple"
    usage = out.metadata["usage"]
    assert (usage["llm_calls"], usage["estimated_calls"]) == (3, 1)
    assert usage["output_tokens"] == 10 and usage["input_tokens"] > 100
    # Never cancelled: the request completes at the server.
    release.set()
    await _until(lambda: "keywords" in model.served)
    await _until(lambda: not chain._background_extractions._snapshot())
    assert not pending.cancelled()


async def test_one_that_answers_before_the_snapshot_has_its_real_tokens() -> None:
    model = _ThreadedChat()
    model.gates["answer"] = model.answered["keywords"]
    chain = _chain(model)
    out = await chain.ainvoke(_auto())
    assert sorted(model.served) == ["answer", "keywords", "router"]
    usage = out.metadata["usage"]
    assert (usage["llm_calls"], usage["estimated_calls"]) == (3, 0)
    assert (usage["input_tokens"], usage["output_tokens"]) == (150, 15)


async def test_one_that_answers_later_settles_in_the_open_scopes(
    release: threading.Event,
) -> None:
    model = _ThreadedChat(gates={"keywords": release})
    chain = _chain(model)
    with usage_scope() as run:
        out = await chain.ainvoke(_auto())
        release.set()
        await _until(lambda: not chain._background_extractions._snapshot())
    assert out.metadata["usage"]["estimated_calls"] == 1
    totals = run.snapshot()
    assert (totals["llm_calls"], totals["estimated_calls"]) == (3, 0)
    assert (totals["input_tokens"], totals["output_tokens"]) == (150, 15)


async def test_the_unread_extraction_fills_the_cache(
    release: threading.Event,
) -> None:
    model = _ThreadedChat(gates={"keywords": release})
    chain = _chain(model, query_extraction_cache_size=8)
    await chain.ainvoke(_auto())
    assert "keywords" not in model.served
    release.set()
    await _until(lambda: not chain._background_extractions._snapshot())
    model.route = "mix"
    out = await chain.ainvoke(_auto())
    assert model.served.count("keywords") == 1
    assert out.metadata["usage"]["cache_hits"] == 1


@pytest.mark.parametrize("events", [False, True])
async def test_the_unread_extraction_reads_its_response_past_the_query(
    release: threading.Event, events: bool
) -> None:
    # Regression: the query's model-response scope (closing_responses)
    # closed the response of the extraction it kept running, so the
    # extraction failed, was retried and failed again (a ReadError over
    # HTTP), and the next query that reads it extracted it once more.
    model = _ThreadedChat(gates={"keywords": release}, threaded=False, tracked=True)
    chain = _chain(model, query_extraction_cache_size=8)
    if events:
        _ = [event async for event in chain.aquery_events(_auto())]
    else:
        await chain.ainvoke(_auto())
    assert "keywords" not in model.served
    release.set()
    await _until(lambda: not chain._background_extractions._snapshot())
    model.route = "mix"
    out = await chain.ainvoke(_auto())
    assert model.cut == []
    assert model.started.count("keywords") == model.served.count("keywords") == 1
    assert out.metadata["usage"]["cache_hits"] == 1


async def test_a_kept_extraction_that_fails_logs_its_failure(
    release: threading.Event, caplog: pytest.LogCaptureFixture
) -> None:
    # Nobody reads a kept extraction's outcome: its final failure (the
    # retries given up) is logged as it ends, by type and digest, its text
    # (which may quote the model's reply) at DEBUG only.
    import logging

    model = _ThreadedChat(gates={"keywords": release}, failing={"keywords"})
    chain = _chain(model, query_extraction_cache_size=8)
    await chain.ainvoke(_auto())
    assert chain._background_extractions._snapshot()
    with caplog.at_level(logging.DEBUG, logger=rag_chain.__name__):
        release.set()
        await _until(lambda: not chain._background_extractions._snapshot())
    ended = [
        record
        for record in caplog.records
        if record.name == rag_chain.__name__ and "extraction" in record.getMessage()
    ]
    (warning,) = [record for record in ended if record.levelno == logging.WARNING]
    assert "ValueError" in warning.getMessage()
    assert _FAILURE_TEXT not in warning.getMessage()
    assert any(_FAILURE_TEXT in record.getMessage() for record in ended)


async def test_a_kept_extraction_is_no_longer_its_readers(
    monkeypatch: pytest.MonkeyPatch, release: threading.Event
) -> None:
    # Kept past its query, the extraction is detached from the query's
    # reader: a stop of that reader set afterwards does not end it.
    stops: list[QueryStop] = []
    scope = rag_chain.query_stop_scope

    def recording(stop: QueryStop) -> Any:
        stops.append(stop)
        return scope(stop)

    monkeypatch.setattr(rag_chain, "query_stop_scope", recording)
    model = _ThreadedChat(gates={"keywords": release}, threaded=False)
    chain = _chain(model, query_extraction_cache_size=8)
    _ = [event async for event in chain.aquery_events(_auto())]
    assert chain._background_extractions._snapshot()
    (stop,) = stops
    stop.set()
    release.set()
    await _until(lambda: not chain._background_extractions._snapshot())
    assert model.stopped["keywords"] == [False]
    assert model.served.count("keywords") == 1


async def test_a_stream_leaves_the_unread_extraction_running(
    release: threading.Event,
) -> None:
    model = _ThreadedChat(gates={"keywords": release})
    chain = _chain(model, query_extraction_cache_size=8)
    chunks = [chunk async for chunk in chain.astream(_auto())]
    assert "".join(chunks)
    assert "keywords" not in model.served
    release.set()
    await _until(lambda: not chain._background_extractions._snapshot())
    model.route = "mix"
    await chain.ainvoke(_auto())
    assert model.served.count("keywords") == 1


async def test_the_query_deadline_cancels_the_background_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A long deadline, fired once the query has returned: the background
    # extraction is cancelled at what was left of it.
    model = _ThreadedChat(delays={"keywords": 600.0}, threaded=False)
    chain = _chain(model, query_timeout_seconds=300)
    timers = _HeldCancels(monkeypatch, beyond=100)
    out = await chain.ainvoke(_auto())
    assert out.answer != DEFAULT_ERROR_MESSAGE
    assert out.metadata["usage"]["estimated_calls"] == 1
    (pending,) = chain._background_extractions._snapshot()
    # One timer, at what is left of the query's deadline.
    assert [(task, 100 < delay <= 300) for delay, task, _ in timers.timers] == [
        (pending, True)
    ]
    assert model.cancelled == []
    timers.fire()
    await _until(lambda: "keywords" in model.cancelled)
    await _until(lambda: not chain._background_extractions._snapshot())


async def test_aclose_cancels_the_background_extraction() -> None:
    model = _ThreadedChat(delays={"keywords": 30.0}, threaded=False)
    chain = _chain(model, query_timeout_seconds=None)
    await chain.ainvoke(_auto())
    assert chain._background_extractions._snapshot()
    await chain.aclose()
    assert model.cancelled == ["keywords"]
    assert not chain._background_extractions._snapshot()


async def test_close_cancels_a_background_extraction_on_the_callers_loop() -> None:
    # The async path ran on the caller's loop, which no runner of the chain
    # stops: only the sync close's own cancel ends the extraction.
    model = _ThreadedChat(delays={"keywords": 30.0}, threaded=False)
    chain = _chain(model, query_timeout_seconds=None)
    await chain.ainvoke(_auto())
    assert chain._background_extractions._snapshot()
    chain.close()
    assert not chain._background_extractions._snapshot()
    await _until(lambda: bool(model.cancelled))
    assert model.cancelled == ["keywords"]


def test_close_ends_the_sync_paths_background_extraction(
    caplog: pytest.LogCaptureFixture,
) -> None:
    model = _ThreadedChat(delays={"keywords": 30.0}, threaded=False)
    chain = _chain(model, query_timeout_seconds=None)
    with caplog.at_level("DEBUG", logger="asyncio"):
        out = chain.invoke(_auto())
        assert out.metadata["usage"]["estimated_calls"] == 1
        assert chain._background_extractions._snapshot()
        started = time.monotonic()
        chain.close()
        # Generous headroom: the extraction would take 30 s, the cancel ms.
        assert time.monotonic() - started < 5
        gc.collect()
    assert model.cancelled == ["keywords"]
    assert not chain._background_extractions._snapshot()
    assert "destroyed but it is pending" not in caplog.text
    assert "exception was never retrieved" not in caplog.text


async def test_a_call_the_deadline_cancels_is_counted(
    monkeypatch: pytest.MonkeyPatch, release: threading.Event
) -> None:
    # The answer call is cut by the deadline, expired once that call is in
    # flight: under ignore_errors the query returns the error answer, whose
    # usage still holds the cut call.
    model = _ThreadedChat(gates={"answer": release})
    chain = _chain(model, query_timeout_seconds=300)
    chain.ignore_errors = True
    timeouts = HeldTimeouts(monkeypatch)
    query = asyncio.ensure_future(chain.ainvoke(_auto()))
    await _until(lambda: "answer" in model.started)
    timeouts.expire()
    out = await asyncio.wait_for(query, 30)
    assert "answer" not in model.served
    release.set()
    assert out.answer == DEFAULT_ERROR_MESSAGE
    usage = out.metadata["usage"]
    assert (usage["llm_calls"], usage["estimated_calls"]) == (3, 1)


async def test_a_call_limit_starts_no_speculative_extraction() -> None:
    # Under search.query_max_model_calls an extraction the route does not
    # read would spend a call the query may need: auto routes first.
    from unified_kg_rag import QueryLimits

    model = _ThreadedChat()
    chain = _chain(model)
    query = _auto().model_copy(update={"limits": QueryLimits(max_model_calls=5)})
    out = await asyncio.wait_for(chain.ainvoke(query), 30)
    assert out.metadata["search_strategy"] == "simple"
    assert model.started == ["router", "answer"]
    assert out.metadata["usage"]["llm_calls"] == 2


async def test_an_input_token_limit_starts_no_speculative_extraction() -> None:
    # Under search.query_max_input_tokens alone the extraction the route
    # does not read would spend tokens the answer may need, as a call
    # would under a call limit.
    from unified_kg_rag import QueryLimits

    model = _ThreadedChat()
    chain = _chain(model)
    query = _auto().model_copy(
        update={"limits": QueryLimits(max_input_tokens=1_000_000)}
    )
    out = await asyncio.wait_for(chain.ainvoke(query), 30)
    assert out.metadata["search_strategy"] == "simple"
    assert model.started == ["router", "answer"]


@pytest.mark.parametrize("stream", [False, True])
async def test_unread_extractions_are_kept_only_until_the_lowered_deadline(
    stream: bool,
) -> None:
    # Regression (W1 review L10): the speculative extractions the route did
    # not read were kept until the config's deadline, not the query's own
    # limits.deadline_seconds.
    from unified_kg_rag import QueryLimits

    model = _ThreadedChat()
    chain = _chain(model, query_timeout_seconds=300)
    kept: list[float | None] = []
    keep = chain._background_extractions.keep

    def record(leftovers: Any, timeout: float | None) -> None:
        kept.append(timeout)
        keep(leftovers, timeout)

    chain._background_extractions.keep = record  # type: ignore[method-assign]
    query = _auto().model_copy(update={"limits": QueryLimits(deadline_seconds=7)})
    if stream:
        _ = [c async for c in chain.astream(query)]
    else:
        await asyncio.wait_for(chain.ainvoke(query), 30)
    assert kept and all(t is not None and t <= 7 for t in kept)
