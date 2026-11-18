# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GraphRAGChain.aquery_events`` / ``query_events`` and ``metadata["routing"]``.

The query streams as typed events: its start, its route, its sources, the
answer's chunks as the answer model writes them, then the ``RAGOutput``
``ainvoke`` returns. Nothing is streamed where the answer may not be shown
before it is checked (the output guardrail) or no model is asked (an empty
context); the ``[NO_ANSWER]`` marker is held back and stripped. The
deprecated ``ainvoke(on_answer_chunk=...)`` reads the same chunks. An
``auto`` query reports how it chose its strategy. Fakes as in
``test_rag_chain_streaming``.
"""

from __future__ import annotations

import asyncio
import itertools
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk

from tests.unit.test_rag_chain_streaming import (
    _ANSWER,
    _input,
    _make_chain,
    _StreamingModelFactory,
)
from unified_kg_rag import (
    AnswerChunk,
    QueryResult,
    QueryRouted,
    QuerySources,
    QueryStart,
)
from unified_kg_rag.application.retrieval.rag_chain import (
    DEFAULT_ERROR_MESSAGE,
    ROUTING_KEY,
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import Config, SearchStrategy
from unified_kg_rag.domain.retrieval.abstention import ABSTENTION_MARKER
from unified_kg_rag.shared.exceptions import LLMOutputTruncatedError

pytestmark = pytest.mark.unit


async def _events(chain: GraphRAGChain, rag_input: RAGInput) -> list[Any]:
    return [event async for event in chain.aquery_events(rag_input)]


def _chunks(events: list[Any]) -> list[str]:
    return [e.text for e in events if isinstance(e, AnswerChunk)]


def _output(events: list[Any]) -> RAGOutput:
    assert isinstance(events[-1], QueryResult)
    return events[-1].output


async def test_the_events_stream_the_query_ainvoke_answers(config: Config) -> None:
    chain, factory, _ = _make_chain(config)
    events = await _events(chain, _input(conversation_id="c-1"))
    kinds = [e.kind for e in events]
    assert kinds[:3] == ["start", "routed", "sources"]
    assert set(kinds[3:-1]) == {"chunk"} and len(kinds[3:-1]) > 1
    assert kinds[-1] == "result"
    start, routed, sources = events[:3]
    assert isinstance(start, QueryStart) and len(start.query_id) == 12
    assert start.conversation_id == "c-1" and start.namespace == "default"
    assert start.requested_strategy is SearchStrategy.SIMPLE
    assert isinstance(routed, QueryRouted)
    assert routed.strategy is SearchStrategy.SIMPLE and routed.routing is None
    output = _output(events)
    assert "".join(_chunks(events)) == output.answer == _ANSWER
    # The sources shown before the answer are the output's.
    assert isinstance(sources, QuerySources)
    assert sources.sources == output.sources and output.sources
    assert factory.get_model_calls == 1
    # The same output as an unstreamed query's.
    unstreamed = await chain.ainvoke(_input())
    assert unstreamed.answer == output.answer
    assert unstreamed.sources == output.sources


async def test_the_events_dump_as_json_by_kind(config: Config) -> None:
    from pydantic import TypeAdapter

    from unified_kg_rag import QueryEvent

    chain, _, _ = _make_chain(config)
    adapter: TypeAdapter[Any] = TypeAdapter(QueryEvent)
    for event in await _events(chain, _input()):
        assert adapter.validate_json(adapter.dump_json(event)) == event


async def test_auto_s_route_is_an_event(config: Config) -> None:
    config.search.auto_routable_strategies = [SearchStrategy.SIMPLE]
    chain, _, _ = _make_chain(config)
    events = await _events(chain, _input(search_strategy=SearchStrategy.AUTO))
    routed = next(e for e in events if isinstance(e, QueryRouted))
    assert routed.strategy is SearchStrategy.SIMPLE
    assert routed.routing == _output(events).metadata[ROUTING_KEY]


async def test_the_deprecated_callback_reads_the_chunks(config: Config) -> None:
    chain, _, _ = _make_chain(config)
    chunks: list[str] = []
    with pytest.warns(DeprecationWarning, match="aquery_events"):
        output = await chain.ainvoke(_input(), on_answer_chunk=chunks.append)
    assert len(chunks) > 1
    assert "".join(chunks) == output.answer == _ANSWER
    assert output.sources


async def test_without_a_reader_nothing_changes(config: Config) -> None:
    chain, _, _ = _make_chain(config)
    output = await chain.ainvoke(_input())
    assert output.answer == _ANSWER


class _MarkedFactory(_StreamingModelFactory):
    """Streams a decline: the marker, then the explanation."""

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        self.get_model_calls += 1
        return GenericFakeChatModel(
            messages=itertools.repeat(
                AIMessage(content=f"{ABSTENTION_MARKER} 공급업체 정보가 없습니다.")
            )
        )


async def test_an_answer_read_for_the_marker_streams_without_it(
    config: Config,
) -> None:
    config.search.abstention.mode = "mark"
    chain, _, _ = _make_chain(config)
    events = await _events(chain, _input())
    assert "".join(_chunks(events)) == _output(events).answer == _ANSWER
    chain, _, _ = _make_chain(config.model_copy(deep=True), factory=_MarkedFactory())
    events = await _events(chain, _input())
    chunks = _chunks(events)
    assert chunks and not any("NO_ANSWER" in c for c in chunks)
    assert "".join(chunks) == _output(events).answer == "공급업체 정보가 없습니다."
    assert _output(events).metadata["abstention_reason"] == "marker"


async def test_an_empty_context_streams_nothing(config: Config) -> None:
    chain, factory, _ = _make_chain(config, empty=True)
    events = await _events(chain, _input())
    assert _chunks(events) == []
    sources = [e for e in events if isinstance(e, QuerySources)]
    assert [s.sources for s in sources] == [[]]
    assert _output(events).metadata.get("abstained") is True
    assert factory.get_model_calls == 0


async def test_a_failure_under_ignore_errors_ends_with_its_output(
    config: Config,
) -> None:
    chain, _, _ = _make_chain(
        config,
        factory=_StreamingModelFactory(fail_mid_stream=True),
        ignore_errors=True,
    )
    events = await _events(chain, _input())
    kinds = [e.kind for e in events]
    assert kinds[:3] == ["start", "routed", "sources"] and kinds[-1] == "result"
    assert _chunks(events)  # shown before the failure
    assert _output(events).answer == DEFAULT_ERROR_MESSAGE
    assert _output(events).metadata["error"] is True and _output(events).sources == []
    # One that fails before it is admitted: its (empty) sources and result.
    chain, _, _ = _make_chain(
        config.model_copy(deep=True), retrieval_error=True, ignore_errors=True
    )
    assert [e.kind for e in await _events(chain, _input())] == ["sources", "result"]


async def test_a_failure_raises_after_the_events_before_it(config: Config) -> None:
    chain, _, memory = _make_chain(
        config, factory=_StreamingModelFactory(fail_mid_stream=True)
    )
    kinds: list[str] = []
    with pytest.raises(RuntimeError, match="synthetic model failure"):
        async for event in chain.aquery_events(
            _input(use_memory=True, conversation_id="c-4")
        ):
            kinds.append(event.kind)
    assert kinds[:3] == ["start", "routed", "sources"]
    assert set(kinds[3:]) == {"chunk"}
    assert memory.messages == []


# Set once a _HeldModel holds its next chunk, with the stop of the query
# it streams for (shared.reader_stop): the cues the stop tests wait on.
_HOLDING = threading.Event()
_HELD_FOR: list[Any] = []
# A hang backstop (seconds), far past anything a passing run waits: the
# stop tests assert on order, not on how long a step took.
_BACKSTOP = 60.0


class _HeldModel(GenericFakeChatModel):
    """Streams one chunk, then a second only after a long wait."""

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        from unified_kg_rag.shared import reader_stop

        yield ChatGenerationChunk(message=AIMessageChunk(content="Vendor "))
        _HELD_FOR.append(reader_stop._READER_STOP.get())
        _HOLDING.set()
        await asyncio.sleep(600)
        yield ChatGenerationChunk(message=AIMessageChunk(content="ships."))


async def _until(condition: Callable[[], bool]) -> None:
    """Wait for ``condition`` (a cue), turn by turn of the loop."""
    async with asyncio.timeout(_BACKSTOP):
        # Polled: the cues (a threading.Event, a reader's stop) have no
        # asyncio event of their own.
        while True:
            if condition():
                return
            await asyncio.sleep(0)


class _HeldFactory(_StreamingModelFactory):
    def __init__(self) -> None:
        super().__init__()
        self.model = _HeldModel(messages=iter([]))
        _HOLDING.clear()
        _HELD_FOR.clear()

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        self.get_model_calls += 1
        return self.model


async def test_a_reader_that_stops_cancels_the_query(config: Config) -> None:
    factory = _HeldFactory()
    chain, _, memory = _make_chain(config, factory=factory)
    events = chain.aquery_events(_input(use_memory=True, conversation_id="c-2"))
    async for event in events:
        if isinstance(event, AnswerChunk):
            break
    await events.aclose()
    # The query's task ended with the reader: none is left running.
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "kg-rag-query-events"]
    # Abandoned: nothing is saved as a finished turn.
    assert memory.messages == []


class _MetricsSink:
    def __init__(self) -> None:
        self.emitted: list[dict[str, float]] = []

    def emit(
        self, namespace: str, metrics: dict[str, float], dimensions: Any = None
    ) -> None:
        self.emitted.append(metrics)


async def _cancelled_after_first_chunk(chain: GraphRAGChain, held: bool) -> None:
    """Read the query's events in a task, cancelled (a disconnected client)
    after the first answer chunk, once the model holds its next one when
    ``held`` (a _HeldModel: the reader then waits for an event none sends)."""
    chunked = asyncio.Event()

    async def read() -> None:
        async for event in chain.aquery_events(
            _input(use_memory=True, conversation_id="c-9")
        ):
            if isinstance(event, AnswerChunk):
                chunked.set()

    reader = asyncio.create_task(read())
    await chunked.wait()
    if held:
        await _until(_HOLDING.is_set)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(_BACKSTOP):
            await reader


async def test_a_cancelled_reader_ends_the_query_as_a_reader_that_stopped(
    config: Config,
) -> None:
    from unified_kg_rag.shared import metrics

    chain, _, memory = _make_chain(config)
    sink = _MetricsSink()
    chain.providers.metrics_sink = sink
    await _cancelled_after_first_chunk(chain, held=False)
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "kg-rag-query-events"]
    assert memory.messages == []
    # Not a failure: its latency up to there, as a stream closed early.
    (values,) = sink.emitted
    assert values[metrics.QUERY_FAILED_METRIC] == 0.0
    assert values[metrics.QUERY_LATENCY_METRIC] > 0


class _FailingModel(GenericFakeChatModel):
    """Fails its stream before any chunk once its query's reader stopped:
    a client's own retries giving up, which no stop point interrupts."""

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        from unified_kg_rag.shared.reader_stop import reader_stopped

        _HOLDING.set()
        await _until(reader_stopped)
        raise RuntimeError("synthetic retry budget spent")
        yield  # pragma: no cover


class _FailingFactory(_HeldFactory):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FailingModel(messages=iter([]))


async def test_a_failure_after_the_reader_stopped_is_a_stopped_reader_s_end(
    config: Config,
) -> None:
    # Whatever the query raises once its reader stopped (here a call that
    # gives up after the stop) ends it as a stopped reader's: no failure.
    from unified_kg_rag.shared import metrics

    chain, _, memory = _make_chain(config, factory=_FailingFactory())
    sink = _MetricsSink()
    chain.providers.metrics_sink = sink
    sourced = asyncio.Event()

    async def read() -> None:
        async for event in chain.aquery_events(_input()):
            if isinstance(event, QuerySources):
                sourced.set()

    reader = asyncio.create_task(read())
    await sourced.wait()
    # The model is called: its failure comes only after the stop.
    await _until(_HOLDING.is_set)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(_BACKSTOP):
            await reader
    # Recorded before the cancelled reader returned.
    (values,) = sink.emitted
    assert values[metrics.QUERY_FAILED_METRIC] == 0.0
    assert memory.messages == []


async def test_a_reader_cancelled_while_no_event_comes_cancels_the_query_at_once(
    config: Config,
) -> None:
    # The answer model holds its next chunk: the cancelled reader cancels
    # the query at once, as its deadline would, never waiting for the model
    # client's read timeout (past the backstop); the query is recorded once,
    # as a stopped reader's (no failure).
    from unified_kg_rag.shared import metrics

    config.search.model_timeout_seconds = 10 * _BACKSTOP
    chain, _, memory = _make_chain(config, factory=_HeldFactory())
    sink = _MetricsSink()
    chain.providers.metrics_sink = sink
    await _cancelled_after_first_chunk(chain, held=True)
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "kg-rag-query-events"]
    assert memory.messages == []
    (values,) = sink.emitted
    assert values[metrics.QUERY_FAILED_METRIC] == 0.0


# Set once a _ClosingModel's stream, cancelled, is closing its response
# (an await, as an HTTP response's close is), then once it closed it.
_CLOSING = threading.Event()
_CLOSED: list[str] = []


class _ClosingModel(GenericFakeChatModel):
    """Streams one chunk, holds the next; cancelled, closes its response
    with an await, which a second cancellation would cut short."""

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        yield ChatGenerationChunk(message=AIMessageChunk(content="Vendor "))
        try:
            _HOLDING.set()
            await asyncio.sleep(600)
        finally:
            _CLOSING.set()
            await asyncio.sleep(0.05)
            _CLOSED.append("closed")
        yield ChatGenerationChunk(message=AIMessageChunk(content="ships."))


class _ClosingFactory(_HeldFactory):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ClosingModel(messages=iter([]))
        _CLOSING.clear()
        _CLOSED.clear()


async def test_a_reader_cancelled_again_while_it_waits_cancels_the_query_once(
    config: Config,
) -> None:
    # An anyio cancel scope cancels the reader at every await: cancelled
    # again while it waits for its cancelled query to end, it passes no
    # second cancellation on, which would cut the close of the model's
    # response short (its connection held).
    config.search.model_timeout_seconds = 10 * _BACKSTOP
    chain, _, memory = _make_chain(config, factory=_ClosingFactory())
    chunked = asyncio.Event()

    async def read() -> None:
        async for event in chain.aquery_events(_input()):
            if isinstance(event, AnswerChunk):
                chunked.set()

    reader = asyncio.create_task(read())
    await chunked.wait()
    await _until(_HOLDING.is_set)
    reader.cancel()
    # The query, cancelled, closes its response: cancel the reader again.
    await _until(_CLOSING.is_set)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    await _until(
        lambda: not [
            t for t in asyncio.all_tasks() if t.get_name() == "kg-rag-query-events"
        ]
    )
    # The query ended with its response closed.
    assert _CLOSED == ["closed"]
    assert memory.messages == []


async def test_a_reader_that_stops_after_the_answer_saves_nothing(
    config: Config,
) -> None:
    # The query waits for its reader at each event: one that stops at the
    # answer's last chunk leaves it waiting, never saving the turn.
    chain, _, memory = _make_chain(config)
    events = chain.aquery_events(_input(use_memory=True, conversation_id="c-3"))
    text = ""
    async for event in events:
        if isinstance(event, AnswerChunk):
            text += event.text
            if text == _ANSWER:
                break
    # A reader that awaits something else before it closes the stream.
    await asyncio.sleep(0.2)
    await events.aclose()
    assert memory.messages == []


async def test_a_named_strategy_reports_no_routing(config: Config) -> None:
    chain, _, _ = _make_chain(config)
    output = await chain.ainvoke(_input(search_strategy=SearchStrategy.SIMPLE))
    assert ROUTING_KEY not in output.metadata


async def test_auto_reports_a_router_reply_naming_no_strategy_as_fallback(
    config: Config,
) -> None:
    # The fake model answers the router with the answer text: no strategy.
    config.search.auto_routable_strategies = [
        SearchStrategy.SIMPLE,
        SearchStrategy.NAIVE,
    ]
    chain, _, _ = _make_chain(config)
    output = await chain.ainvoke(_input(search_strategy=SearchStrategy.AUTO))
    routing = output.metadata[ROUTING_KEY]
    assert routing["decided_by"] == "fallback"
    assert routing["candidates"] == ["simple", "naive"]
    assert output.metadata["search_strategy"] == "simple"


async def test_auto_with_one_candidate_makes_no_router_call(config: Config) -> None:
    config.search.auto_routable_strategies = [SearchStrategy.SIMPLE]
    chain, factory, _ = _make_chain(config)
    output = await chain.ainvoke(_input(search_strategy=SearchStrategy.AUTO))
    assert output.metadata[ROUTING_KEY] == {
        "decided_by": "only_candidate",
        "candidates": ["simple"],
    }
    assert factory.get_model_calls == 1  # the answer only


async def test_router_reply_naming_its_strategy_counts_as_the_router() -> None:
    from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

    chain = GraphRAGChain.__new__(GraphRAGChain)
    chain.config = Config()
    chain.factory = ScriptedLLMFactory(lambda _system, _human: "no JSON")
    routable = (SearchStrategy.LOCAL, SearchStrategy.MIX)
    assert await chain._routed_strategy('{"strategy": "mix"}', routable) == (
        SearchStrategy.MIX,
        True,
    )
    assert await chain._routed_strategy("I am not sure", routable) == (
        SearchStrategy.LOCAL,
        False,
    )


class _CutModel(GenericFakeChatModel):
    """Streams the answer, its last chunk marked cut at the output limit."""

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        async for chunk in super()._astream(*args, **kwargs):
            yield chunk
        yield ChatGenerationChunk(
            message=AIMessageChunk(
                content="", response_metadata={"stopReason": "max_tokens"}
            )
        )

    def _generate(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        result = super()._generate(*args, **kwargs)
        result.generations[0].message.response_metadata["stopReason"] = "max_tokens"
        return result


class _CutFactory(_StreamingModelFactory):
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        self.get_model_calls += 1
        return _CutModel(messages=itertools.repeat(AIMessage(content=_ANSWER)))


@pytest.mark.parametrize("ignore_errors", [False, True])
async def test_a_truncated_answer_fails_streamed_as_unstreamed(
    config: Config, ignore_errors: bool
) -> None:
    outcomes = []
    for stream in (False, True):
        chain, _, _ = _make_chain(
            config.model_copy(deep=True),
            factory=_CutFactory(),
            ignore_errors=ignore_errors,
        )
        chunks: list[str] = []
        if ignore_errors:
            if stream:
                events = await _events(chain, _input())
                chunks, output = _chunks(events), _output(events)
            else:
                output = await chain.ainvoke(_input())
            outcomes.append((output.answer, bool(output.metadata.get("error"))))
        else:
            with pytest.raises(LLMOutputTruncatedError):
                if stream:
                    async for event in chain.aquery_events(_input()):
                        if isinstance(event, AnswerChunk):
                            chunks.append(event.text)
                else:
                    await chain.ainvoke(_input())
            outcomes.append("raised")
        if stream:
            assert chunks  # the preview was shown before the cut was seen
    assert outcomes[0] == outcomes[1]
    if ignore_errors:
        assert outcomes[0] == (DEFAULT_ERROR_MESSAGE, True)


_CITED_ANSWER = "Vendor supplies Buyer [1] under the agreement [1, 7]."


class _CitingFactory(_StreamingModelFactory):
    """Streams an answer citing the context's [1] and an unknown [7], the
    list marker split across word chunks."""

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        self.get_model_calls += 1
        return GenericFakeChatModel(
            messages=itertools.repeat(AIMessage(content=_CITED_ANSWER))
        )


async def test_a_streamed_answer_has_the_citations_of_an_unstreamed_one(
    config: Config,
) -> None:
    from unified_kg_rag.domain.retrieval.citations import (
        CitationMarkerScanner,
        cited_markers,
    )

    assert config.search.citations.enabled
    chain, _, _ = _make_chain(config.model_copy(deep=True), factory=_CitingFactory())
    unstreamed = await chain.ainvoke(_input())
    chain, _, _ = _make_chain(config.model_copy(deep=True), factory=_CitingFactory())
    events = await _events(chain, _input())
    chunks, streamed = _chunks(events), _output(events)
    assert streamed.answer == unstreamed.answer == "".join(chunks) == _CITED_ANSWER
    assert streamed.citations == unstreamed.citations
    assert [c.marker for c in streamed.citations if c.cited] == [1]
    assert (
        streamed.metadata["unknown_citation_markers"]
        == unstreamed.metadata["unknown_citation_markers"]
        == [7]
    )
    # The chunks split "[1, 7]"; read with the scanner they still give the
    # answer's markers.
    assert not any("[1, 7]" in chunk for chunk in chunks)
    scanner = CitationMarkerScanner()
    for chunk in chunks:
        scanner.feed(chunk)
    scanner.finish()
    assert scanner.markers == cited_markers(_CITED_ANSWER) == [1, 7]


def test_the_sync_events_are_the_async_ones(config: Config) -> None:
    chain, _, memory = _make_chain(config)
    try:
        events = list(chain.query_events(_input()))
        assert [e.kind for e in events][:3] == ["start", "routed", "sources"]
        assert "".join(_chunks(events)) == _output(events).answer == _ANSWER
        # Closed early: the query is cancelled, no turn is saved.
        stream = chain.query_events(_input(use_memory=True, conversation_id="c-5"))
        next(e for e in stream if isinstance(e, AnswerChunk))
        stream.close()
        assert memory.messages == []
    finally:
        chain.close()


def test_langchain_s_astream_events_is_left_as_it_is() -> None:
    from langchain_core.runnables import Runnable

    # LangChain tooling (LangServe, LangGraph) reads its callback events.
    assert issubclass(GraphRAGChain, Runnable)
    assert GraphRAGChain.astream_events is Runnable.astream_events


class _PassingGuard:
    """An output guardrail that passes every text, recording the answers
    it screened."""

    def __init__(self) -> None:
        self.screened: list[str] = []

    async def acheck(self, text: str, source: Any, **kwargs: Any) -> Any:
        from unified_kg_rag.ports.guardrail import (
            GuardAction,
            GuardSource,
            GuardVerdict,
        )

        if source is GuardSource.OUTPUT:
            self.screened.append(text)
        return GuardVerdict(action=GuardAction.NONE, text=text)


def _screened_chain(config: Config) -> tuple[GraphRAGChain, Any, _PassingGuard]:
    chain, _, memory = _make_chain(config)
    guard = _PassingGuard()
    chain.guard = guard  # type: ignore[assignment]
    chain._screen_outputs = True  # what a configured check_output sets
    return chain, memory, guard


async def test_a_screened_answer_streams_once_screened(config: Config) -> None:
    # Regression (W2 review S1/S2): under check_output the answer model is
    # still streamed (its first chunk bounded), nothing is shown before the
    # guardrail screened the whole answer, then the screened answer is one
    # chunk, its sources before it.
    chain, _, guard = _screened_chain(config)
    events = await _events(chain, _input())
    kinds = [e.kind for e in events]
    assert kinds == ["start", "routed", "sources", "chunk", "result"]
    assert _chunks(events) == [_output(events).answer] == guard.screened
    assert events[2].sources == _output(events).sources


@pytest.mark.parametrize("stop_at", ["sources", "chunk"])
async def test_a_screened_query_its_reader_stopped_saves_nothing(
    config: Config, stop_at: str
) -> None:
    # Regression (W2 review S1): the screened sources were sent after the
    # query had written its memory, so a reader that stopped there (never
    # shown the answer) left the turn in memory.
    chain, memory, _ = _screened_chain(config)
    events = chain.aquery_events(_input(use_memory=True, conversation_id="c-9"))
    async for event in events:
        if event.kind == stop_at:
            break
    await events.aclose()
    assert memory.messages == []


async def test_a_screened_stream_broken_after_its_answer_saves_nothing(
    config: Config,
) -> None:
    chain, memory, _ = _screened_chain(config)
    stream = chain.astream(_input(use_memory=True, conversation_id="c-8"))
    assert await anext(stream) == _ANSWER
    await stream.aclose()
    assert memory.messages == []


# Each _ReasoningModel chunk, by whether its query's reader had stopped
# when it was written, and "ended" once its stream ended.
_REASONING: list[str] = []
_REASONING_STARTED = threading.Event()


class _ReasoningModel(GenericFakeChatModel):
    """Reasons at length: chunks without text (a reasoning model's
    reasoning, emptied), then its answer; records each chunk and its end."""

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        from unified_kg_rag.shared.reader_stop import reader_stopped

        try:
            for _ in range(30_000):
                _REASONING.append("after-stop" if reader_stopped() else "before")
                yield ChatGenerationChunk(message=AIMessageChunk(content=""))
                _REASONING_STARTED.set()
                await asyncio.sleep(0.02)
            yield ChatGenerationChunk(message=AIMessageChunk(content=_ANSWER))
        finally:
            _REASONING.append("ended")


class _ReasoningFactory(_StreamingModelFactory):
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        self.get_model_calls += 1
        return _ReasoningModel(messages=iter([]))


@pytest.mark.parametrize("screened", [False, True], ids=["shown", "screened"])
async def test_a_reader_cancelled_while_the_model_reasons_ends_it_at_once(
    config: Config, screened: bool
) -> None:
    # While the model streams chunks without text (reasoning, a held-back
    # marker, a screened answer's), nothing is sent to the reader: its
    # cancel still ends the query at once, its model writing nothing more
    # for a reader that left.
    config.search.model_timeout_seconds = 60
    chain, _, memory = _make_chain(config, factory=_ReasoningFactory())
    if screened:
        chain.guard = _PassingGuard()  # type: ignore[assignment]
        chain._screen_outputs = True
    _REASONING.clear()
    _REASONING_STARTED.clear()

    async def read() -> None:
        async for _ in chain.aquery_events(
            _input(use_memory=True, conversation_id="c-11")
        ):
            pass

    reader = asyncio.create_task(read())
    # The model is reasoning (its first chunk read).
    await _until(_REASONING_STARTED.is_set)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        async with asyncio.timeout(_BACKSTOP):
            await reader
    # Ended at once: no chunk was written after the stop, and the model's
    # stream had ended before the reader returned.
    assert _REASONING.count("after-stop") == 0
    assert _REASONING[-1] == "ended"
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "kg-rag-query-events"]
    assert memory.messages == []


def _interrupt_once_waiting() -> None:
    """Send SIGINT to the main thread once it blocks in the chain's loop
    runner, waiting for a step (``_LoopRunner.run``'s ``future.result``)."""
    main = threading.main_thread().ident
    assert main is not None
    deadline = time.monotonic() + _BACKSTOP
    while time.monotonic() < deadline:
        frame = sys._current_frames().get(main)
        names = []
        while frame is not None:
            names.append(frame.f_code.co_name)
            frame = frame.f_back
        # Innermost first: a wait inside result(), inside the runner's run().
        if "result" in names and "run" in names and names[0] == "wait":
            signal.pthread_kill(main, signal.SIGINT)
            return
        time.sleep(0.001)
    raise AssertionError("the main thread never waited for the step")


@pytest.mark.parametrize("api", ["stream", "query_events"])
def test_an_interrupt_while_a_sync_stream_waits_cancels_the_query(
    config: Config, api: str
) -> None:
    # Ctrl-C while the sync iterator waits for the answer's next chunk:
    # the interrupt is raised at once (not a RuntimeError from closing a
    # step still running) and the query is cancelled, not left running on
    # the loop.
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signals reach the main thread only")
    config.search.model_timeout_seconds = 10 * _BACKSTOP
    chain, _, memory = _make_chain(config, factory=_HeldFactory())
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        rag_input = _input(use_memory=True, conversation_id="c-12")
        stream: Iterator[Any] = getattr(chain, api)(rag_input)
        # Up to the answer's first chunk (the model then holds its next).
        while not isinstance(first := next(stream), AnswerChunk | str):
            pass
        assert first
        # The interrupt lands once the main thread waits for the next chunk
        # (blocked in the loop runner's wait), whenever that is.
        interrupt = threading.Thread(target=_interrupt_once_waiting)
        interrupt.start()
        with pytest.raises(KeyboardInterrupt):
            next(stream)
        interrupt.join()

        async def queries() -> list[asyncio.Task[Any]]:
            return [
                t for t in asyncio.all_tasks() if t.get_name() == "kg-rag-query-events"
            ]

        # The query ends (cancelled): a backstop, not a measure.
        deadline = time.monotonic() + _BACKSTOP
        while chain._loop_runner.run(queries()) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert chain._loop_runner.run(queries()) == []
        assert memory.messages == []
    finally:
        signal.signal(signal.SIGINT, previous)
        chain.close()
