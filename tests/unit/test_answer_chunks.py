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
import threading
import time
from collections.abc import Iterator
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
from unified_kg_rag.application.retrieval import rag_chain
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


class _HeldModel(GenericFakeChatModel):
    """Streams one chunk, then a second only after a long wait."""

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        yield ChatGenerationChunk(message=AIMessageChunk(content="Vendor "))
        await asyncio.sleep(600)
        yield ChatGenerationChunk(message=AIMessageChunk(content="ships."))


class _HeldFactory(_StreamingModelFactory):
    def __init__(self) -> None:
        super().__init__()
        self.model = _HeldModel(messages=iter([]))

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


async def _cancelled_after_first_chunk(chain: GraphRAGChain, wait: float) -> None:
    """Read the query's events in a task, cancelled (a disconnected client)
    ``wait`` seconds after the first answer chunk, while it waits for the
    next event."""
    chunked = asyncio.Event()

    async def read() -> None:
        async for event in chain.aquery_events(
            _input(use_memory=True, conversation_id="c-9")
        ):
            if isinstance(event, AnswerChunk):
                chunked.set()

    reader = asyncio.create_task(read())
    await chunked.wait()
    await asyncio.sleep(wait)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader


async def test_a_cancelled_reader_ends_the_query_as_a_reader_that_stopped(
    config: Config,
) -> None:
    from unified_kg_rag.shared import metrics

    chain, _, memory = _make_chain(config)
    sink = _MetricsSink()
    chain.providers.metrics_sink = sink
    await _cancelled_after_first_chunk(chain, 0)
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "kg-rag-query-events"]
    assert memory.messages == []
    # Not a failure: its latency up to there, as a stream closed early.
    (values,) = sink.emitted
    assert values[metrics.QUERY_FAILED_METRIC] == 0.0
    assert values[metrics.QUERY_LATENCY_METRIC] > 0


class _FailingModel(GenericFakeChatModel):
    """Fails its stream before any chunk, a moment after the call: a
    client's own retries giving up, which no stop point interrupts."""

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        await asyncio.sleep(0.3)
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
    await asyncio.sleep(0.05)
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    (values,) = sink.emitted
    assert values[metrics.QUERY_FAILED_METRIC] == 0.0
    assert memory.messages == []


async def test_a_reader_cancelled_while_no_event_comes_cancels_at_the_bound(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The answer model holds its next chunk: the cancelled reader waits for
    # the query's next event a grace past search.model_timeout_seconds (the
    # model clients' read timeout, which ends a stalled stream cleanly
    # first), then cancels it.
    monkeypatch.setattr(rag_chain, "_STOP_GRACE_SECONDS", 0.3)
    config.search.model_timeout_seconds = 0.3
    chain, _, memory = _make_chain(config, factory=_HeldFactory())
    started = asyncio.get_running_loop().time()
    await _cancelled_after_first_chunk(chain, 0.05)
    elapsed = asyncio.get_running_loop().time() - started
    assert 0.6 <= elapsed < 5
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "kg-rag-query-events"]
    assert memory.messages == []


async def test_a_reader_cancelled_again_while_it_waits_leaves_the_bound_to_end_it(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An anyio cancel scope cancels the reader at every await: its wait for
    # the query's next event is cut short, and the query is not cancelled
    # there (mid-chunk) but still ends at its next event or at the bound.
    monkeypatch.setattr(rag_chain, "_STOP_GRACE_SECONDS", 0.0)
    config.search.model_timeout_seconds = 0.3
    chain, _, _ = _make_chain(config, factory=_HeldFactory())
    chunked = asyncio.Event()

    async def read() -> None:
        async for event in chain.aquery_events(_input()):
            if isinstance(event, AnswerChunk):
                chunked.set()

    reader = asyncio.create_task(read())
    await chunked.wait()
    await asyncio.sleep(0.05)
    reader.cancel()
    await asyncio.sleep(0.05)
    reader.cancel()
    again = asyncio.get_running_loop().time()
    with pytest.raises(asyncio.CancelledError):
        await reader
    (query,) = [t for t in asyncio.all_tasks() if t.get_name() == "kg-rag-query-events"]
    await asyncio.wait({query}, timeout=5)
    # Cancelled at the bound after the second cancellation, not at it.
    assert asyncio.get_running_loop().time() - again >= 0.25
    assert query.done()


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


# When each _ReasoningModel stream ended (loop time).
_REASONING_ENDED: list[float] = []


class _ReasoningModel(GenericFakeChatModel):
    """Reasons at length: chunks without text (a reasoning model's
    reasoning, emptied), then its answer; records when its stream ends."""

    async def _astream(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        try:
            for _ in range(30_000):
                yield ChatGenerationChunk(message=AIMessageChunk(content=""))
                await asyncio.sleep(0.02)
            yield ChatGenerationChunk(message=AIMessageChunk(content=_ANSWER))
        finally:
            _REASONING_ENDED.append(asyncio.get_running_loop().time())


class _ReasoningFactory(_StreamingModelFactory):
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        self.get_model_calls += 1
        return _ReasoningModel(messages=iter([]))


@pytest.mark.parametrize("screened", [False, True], ids=["shown", "screened"])
async def test_a_reader_cancelled_while_the_model_reasons_ends_it_at_its_next_chunk(
    config: Config, screened: bool
) -> None:
    # A chunk without text (reasoning, a held-back marker, a screened
    # answer's) is a stop point too: the query does not run on for the
    # wait's bound, its model writing for a reader that left.
    config.search.model_timeout_seconds = 60
    chain, _, memory = _make_chain(config, factory=_ReasoningFactory())
    if screened:
        chain.guard = _PassingGuard()  # type: ignore[assignment]
        chain._screen_outputs = True
    _REASONING_ENDED.clear()
    routed = asyncio.Event()

    async def read() -> None:
        async for event in chain.aquery_events(
            _input(use_memory=True, conversation_id="c-11")
        ):
            if isinstance(event, QueryRouted):
                routed.set()

    reader = asyncio.create_task(read())
    await routed.wait()
    await asyncio.sleep(0.2)  # the model is reasoning
    loop = asyncio.get_running_loop()
    stopped = loop.time()
    reader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await reader
    assert loop.time() - stopped < 5
    (ended,) = _REASONING_ENDED
    assert ended - stopped < 5
    assert not [t for t in asyncio.all_tasks() if t.get_name() == "kg-rag-query-events"]
    assert memory.messages == []


@pytest.mark.parametrize("wait", [None, 0.0], ids=["ended", "still-ending"])
@pytest.mark.parametrize("api", ["stream", "query_events"])
def test_an_interrupt_while_a_sync_stream_waits_cancels_the_query(
    config: Config, monkeypatch: pytest.MonkeyPatch, api: str, wait: float | None
) -> None:
    # Ctrl-C while the sync iterator waits for the answer's next chunk:
    # the interrupt is raised (not a RuntimeError from closing a step still
    # running, when the query has not ended by the time it is raised) and
    # the query is cancelled, not left running on the loop.
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signals reach the main thread only")
    monkeypatch.setattr(rag_chain, "_STOP_GRACE_SECONDS", 0.0)
    if wait is not None:
        monkeypatch.setattr(rag_chain, "_INTERRUPT_WAIT_SECONDS", wait)
    config.search.model_timeout_seconds = 0.3
    chain, _, memory = _make_chain(config, factory=_HeldFactory())
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        rag_input = _input(use_memory=True, conversation_id="c-12")
        stream: Iterator[Any] = getattr(chain, api)(rag_input)
        # Up to the answer's first chunk (the model then holds its next).
        while not isinstance(first := next(stream), AnswerChunk | str):
            pass
        assert first
        interrupt = threading.Timer(
            0.2, signal.pthread_kill, (threading.main_thread().ident, signal.SIGINT)
        )
        interrupt.start()
        with pytest.raises(KeyboardInterrupt):
            next(stream)
        interrupt.join()

        async def queries() -> list[asyncio.Task[Any]]:
            return [
                t for t in asyncio.all_tasks() if t.get_name() == "kg-rag-query-events"
            ]

        for _ in range(50):
            if not chain._loop_runner.run(queries()):
                break
            time.sleep(0.1)
        assert chain._loop_runner.run(queries()) == []
        assert memory.messages == []
    finally:
        signal.signal(signal.SIGINT, previous)
        chain.close()
