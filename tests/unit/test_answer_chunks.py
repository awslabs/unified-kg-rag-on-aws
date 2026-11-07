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

import pytest

from tests.unit.test_rag_chain_streaming import _ANSWER, _input, _make_chain
from unified_kg_rag.application.retrieval.rag_chain import ROUTING_KEY
from unified_kg_rag.domain.models import Config, SearchStrategy

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


def test_router_reply_naming_its_strategy_counts_as_the_router() -> None:
    from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain

    assert GraphRAGChain._names_strategy("Local search.", SearchStrategy.LOCAL)
    assert not GraphRAGChain._names_strategy("I am not sure", SearchStrategy.LOCAL)
