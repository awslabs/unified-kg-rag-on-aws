# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``GraphRAGChain.ainvoke(on_answer_chunk=...)`` and ``metadata["routing"]``.

The answer step streams the answer model to the callback while ``ainvoke``
still returns the whole ``RAGOutput``; nothing is streamed where the answer
may not be shown before it is checked (the ``[NO_ANSWER]`` marker, the
output guardrail) or no model is asked (an empty context). An ``auto``
query reports how it chose its strategy. Fakes as in
``test_rag_chain_streaming``.
"""

from __future__ import annotations

import itertools
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
from unified_kg_rag.application.retrieval.rag_chain import (
    DEFAULT_ERROR_MESSAGE,
    ROUTING_KEY,
)
from unified_kg_rag.domain.models import Config, SearchStrategy
from unified_kg_rag.shared.exceptions import LLMOutputTruncatedError

pytestmark = pytest.mark.unit


async def test_answer_chunks_stream_the_answer_ainvoke_returns(config: Config) -> None:
    chain, factory, _ = _make_chain(config)
    chunks: list[str] = []
    output = await chain.ainvoke(_input(), on_answer_chunk=chunks.append)
    assert len(chunks) > 1
    assert "".join(chunks) == output.answer == _ANSWER
    # Sources and metadata are those of an unstreamed answer.
    assert output.sources
    assert factory.get_model_calls == 1


async def test_without_a_callback_nothing_changes(config: Config) -> None:
    chain, _, _ = _make_chain(config)
    output = await chain.ainvoke(_input())
    assert output.answer == _ANSWER


async def test_an_answer_read_for_the_marker_is_not_streamed(config: Config) -> None:
    config.search.abstention.mode = "mark"
    chain, _, _ = _make_chain(config)
    chunks: list[str] = []
    output = await chain.ainvoke(_input(), on_answer_chunk=chunks.append)
    assert chunks == []
    assert output.answer == _ANSWER


def test_a_screened_answer_is_not_streamed(config: Config) -> None:
    from langchain_core.runnables import RunnableLambda

    from unified_kg_rag.application.retrieval import rag_chain

    chain, _, _ = _make_chain(config)
    state = {"context": "Vendor and Buyer signed.", "query": "q"}
    token = rag_chain._ANSWER_CHUNKS.set(lambda _text: None)
    try:
        streamed = chain._answer_generation_step(state)
        chain._screen_outputs = True  # what a configured output guardrail sets
        screened = chain._answer_generation_step(state)
    finally:
        rag_chain._ANSWER_CHUNKS.reset(token)
    assert isinstance(streamed, RunnableLambda)
    assert not isinstance(screened, RunnableLambda)


async def test_an_empty_context_streams_nothing(config: Config) -> None:
    chain, factory, _ = _make_chain(config, empty=True)
    chunks: list[str] = []
    output = await chain.ainvoke(_input(), on_answer_chunk=chunks.append)
    assert chunks == []
    assert output.metadata.get("abstained") is True
    assert factory.get_model_calls == 0


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
        callback = chunks.append if stream else None
        if ignore_errors:
            output = await chain.ainvoke(_input(), on_answer_chunk=callback)
            outcomes.append((output.answer, bool(output.metadata.get("error"))))
        else:
            with pytest.raises(LLMOutputTruncatedError):
                await chain.ainvoke(_input(), on_answer_chunk=callback)
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
    outputs = []
    chunks: list[str] = []
    for callback in (None, chunks.append):
        chain, _, _ = _make_chain(
            config.model_copy(deep=True), factory=_CitingFactory()
        )
        outputs.append(await chain.ainvoke(_input(), on_answer_chunk=callback))
    unstreamed, streamed = outputs
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
