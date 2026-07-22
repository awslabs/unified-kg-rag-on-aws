# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GraphRAGChain parsing + state-graph node helpers (AWS-free).

Covers the LightRAG keyword-JSON parser (``_parse_keyword_json``), the
``RAGInput``/``RAGOutput``/``ProcessedQuery`` model boundaries,
``_prepare_invoke`` conversation-id handling, the LightRAG mode predicate, and
the synchronous output-formatting nodes. Does NOT duplicate the strategy/
retriever dispatch in ``test_rag_chain_dispatch`` or the router parsing already
asserted elsewhere.
"""

from __future__ import annotations

import json

import pytest

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    ProcessedQuery,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)

pytestmark = pytest.mark.unit


# --- _parse_keyword_json -------------------------------------------------


def test_parse_keyword_json_plain_object() -> None:
    raw = '{"high_level_keywords": ["ai"], "low_level_keywords": ["llm", "rag"]}'
    parsed = GraphRAGChain._parse_keyword_json(raw)
    assert parsed["high_level_keywords"] == ["ai"]
    assert parsed["low_level_keywords"] == ["llm", "rag"]


def test_parse_keyword_json_strips_json_code_fence() -> None:
    raw = '```json\n{"high_level_keywords": ["x"], "low_level_keywords": []}\n```'
    parsed = GraphRAGChain._parse_keyword_json(raw)
    assert parsed["high_level_keywords"] == ["x"]


def test_parse_keyword_json_strips_bare_code_fence() -> None:
    raw = '```\n{"high_level_keywords": ["y"]}\n```'
    parsed = GraphRAGChain._parse_keyword_json(raw)
    assert parsed["high_level_keywords"] == ["y"]


def test_parse_keyword_json_isolates_object_from_prose() -> None:
    raw = 'Here are the keywords: {"high_level_keywords": ["topic"]} hope that helps!'
    parsed = GraphRAGChain._parse_keyword_json(raw)
    assert parsed["high_level_keywords"] == ["topic"]


def test_parse_keyword_json_array_payload_returns_empty_dict() -> None:
    # A top-level JSON array is not a dict -> normalized to {} so callers get
    # empty keyword lists rather than a crash.
    raw = '["not", "a", "dict"]'
    assert GraphRAGChain._parse_keyword_json(raw) == {}


def test_parse_keyword_json_invalid_raises() -> None:
    with pytest.raises(json.JSONDecodeError):
        GraphRAGChain._parse_keyword_json("not json at all")


def test_parse_keyword_json_extracts_innermost_braces_span() -> None:
    # find('{')..rfind('}') spans the whole object even with nested braces.
    raw = '{"high_level_keywords": ["a"], "meta": {"nested": 1}}'
    parsed = GraphRAGChain._parse_keyword_json(raw)
    assert parsed["high_level_keywords"] == ["a"]
    assert parsed["meta"] == {"nested": 1}


# --- _is_lightrag_mode ---------------------------------------------------


@pytest.mark.parametrize(
    "strategy,expected",
    [
        (SearchStrategy.MIX, True),
        (SearchStrategy.HYBRID, True),
        (SearchStrategy.NAIVE, True),
        (SearchStrategy.LOCAL, False),
        (SearchStrategy.GLOBAL, False),
        (None, False),
    ],
)
def test_is_lightrag_mode(strategy, expected) -> None:
    assert GraphRAGChain._is_lightrag_mode({"resolved_strategy": strategy}) is expected


# --- query-translation refusal guard (Issue F side observation) ----------


@pytest.mark.parametrize(
    "candidate",
    [
        "I appreciate your request, but I notice the text you provided...",
        "I notice the text you provided is already in English.",
        "There is no text to translate.",
        "As an AI, I cannot translate this.",
    ],
)
def test_translation_refusal_detected(candidate) -> None:
    # LLM meta-output (not a translation) must be flagged so the caller falls
    # back to the original query instead of searching for the commentary.
    assert (
        GraphRAGChain._looks_like_translation_refusal(candidate, "오다 노부나가")
        is True
    )


@pytest.mark.parametrize(
    "candidate,original",
    [
        ("Tell me about Oda Nobunaga", "오다 노부나가"),  # a real translation
        ("오다 노부나가", "오다 노부나가"),  # identical passthrough
        ("The relationship between X and Y", "X와 Y의 관계"),
    ],
)
def test_genuine_translation_not_flagged(candidate, original) -> None:
    # A real translation (even one containing 'the') and an identical passthrough
    # must NOT be treated as refusals.
    assert GraphRAGChain._looks_like_translation_refusal(candidate, original) is False


# --- RAGInput / ProcessedQuery / RAGOutput boundaries --------------------


def test_rag_input_defaults() -> None:
    ri = RAGInput(query="hello")
    assert ri.search_strategy == SearchStrategy.AUTO
    assert ri.top_k == 10
    assert ri.use_memory is False
    assert ri.enable_query_processing is True
    assert ri.conversation_id is None


def test_processed_query_defaults_empty_keyword_lists() -> None:
    pq = ProcessedQuery(original_query="q", final_query="q")
    assert pq.entities == []
    assert pq.hl_keywords == []
    assert pq.ll_keywords == []
    assert pq.translated_query is None


# --- _prepare_invoke -----------------------------------------------------


def test_prepare_invoke_assigns_conversation_id_when_memory_on() -> None:
    ri = RAGInput(query="q", use_memory=True)
    rag_input, input_dict = GraphRAGChain._prepare_invoke(ri)
    assert rag_input.conversation_id is not None
    assert input_dict["conversation_id"] == rag_input.conversation_id
    assert "start_time" in input_dict


def test_prepare_invoke_keeps_existing_conversation_id() -> None:
    ri = RAGInput(query="q", use_memory=True, conversation_id="conv-1")
    rag_input, _ = GraphRAGChain._prepare_invoke(ri)
    assert rag_input.conversation_id == "conv-1"


def test_prepare_invoke_no_conversation_id_without_memory() -> None:
    ri = RAGInput(query="q", use_memory=False)
    rag_input, _ = GraphRAGChain._prepare_invoke(ri)
    assert rag_input.conversation_id is None


def test_prepare_invoke_accepts_dict_input() -> None:
    rag_input, input_dict = GraphRAGChain._prepare_invoke({"query": "from-dict"})
    assert isinstance(rag_input, RAGInput)
    assert rag_input.query == "from-dict"
    assert input_dict["query"] == "from-dict"


# --- _format_output_step / _format_search_output_step --------------------


def _search_result() -> SearchResult:
    return SearchResult(
        query=SearchQuery(query="q"),
        results=[
            RetrievalResult(
                content="c1",
                score=0.9,
                source="doc-1",
                retriever_type="document",
                metadata={"k": "v"},
            )
        ],
        total_results=1,
        search_strategy="pending",
        processing_time=0.0,
        metadata={"extra": 1},
    )


def test_format_output_step_builds_rag_output() -> None:
    state = {
        "search_results": _search_result(),
        "resolved_strategy": SearchStrategy.LOCAL,
        "start_time": 0.0,
        "answer": "the answer",
        "conversation_id": "conv-9",
        "processed_query": ProcessedQuery(original_query="q", final_query="q"),
    }
    out = GraphRAGChain._format_output_step(state)
    assert isinstance(out, RAGOutput)
    assert out.answer == "the answer"
    assert out.conversation_id == "conv-9"
    # search_strategy is stamped from the resolved strategy.
    assert out.search_results.search_strategy == SearchStrategy.LOCAL.value
    assert out.metadata["search_strategy"] == SearchStrategy.LOCAL.value
    assert out.metadata["total_results"] == 1
    # carried-over metadata from the SearchResult is merged in.
    assert out.metadata["extra"] == 1
    # sources project only source/score/metadata.
    assert out.sources[0]["source"] == "doc-1"
    assert set(out.sources[0].keys()) == {"source", "score", "metadata"}


def test_format_search_output_step_returns_serializable_dict() -> None:
    state = {
        "search_results": _search_result(),
        "resolved_strategy": SearchStrategy.GLOBAL,
        "start_time": 0.0,
        "processed_query": ProcessedQuery(original_query="q", final_query="q"),
    }
    out = GraphRAGChain._format_search_output_step(state)
    assert isinstance(out, dict)
    assert out["search_results"]["search_strategy"] == SearchStrategy.GLOBAL.value
    assert out["metadata"]["total_results"] == 1
    assert out["processed_query"]["original_query"] == "q"


# --- _query_processing_branch simple path --------------------------------


def test_query_processing_branch_simple_query_passthrough(
    config: Config, monkeypatch
) -> None:
    chain = GraphRAGChain(config=config)
    branch = chain._query_processing_branch()
    # enable_query_processing=False -> the no-LLM simple path runs.
    result = branch.invoke({"query": "raw query", "enable_query_processing": False})
    assert isinstance(result, ProcessedQuery)
    assert result.original_query == "raw query"
    assert result.final_query == "raw query"
    assert result.entities == []


# --- _context_building_step history branch -------------------------------


def _ctx_state(history: str) -> dict:
    return {
        "processed_query": ProcessedQuery(original_query="q", final_query="q"),
        "search_results": _search_result(),
        "history": history,
    }


def test_context_building_no_history_returns_raw_context(config: Config) -> None:
    chain = GraphRAGChain(config=config)
    out = chain._context_building_step(_ctx_state(""))
    # No history -> the raw (token-optimized) search context is returned as-is,
    # with no ContextBuildingPrompt LLM call.
    assert isinstance(out, str)
    assert out  # non-empty (the one result's content)


def test_context_building_with_history_invokes_llm(config: Config) -> None:
    chain = GraphRAGChain(config=config)

    class _FakeBuilder:
        def invoke(self, _inputs):
            return "FOLDED-WITH-HISTORY"

    chain._get_chain_for_prompt = lambda *a, **k: _FakeBuilder()  # type: ignore[assignment]
    out = chain._context_building_step(_ctx_state("earlier turn"))
    assert out == "FOLDED-WITH-HISTORY"


def test_context_building_ignore_errors_degrades_to_empty(config: Config) -> None:
    chain = GraphRAGChain(config=config)
    chain.ignore_errors = True

    def _boom(*a, **k):
        raise RuntimeError("context builder down")

    chain._get_chain_for_prompt = _boom  # type: ignore[assignment]
    out = chain._context_building_step(_ctx_state("earlier turn"))
    assert out == ""


# --- _load_memory_step ---------------------------------------------------


async def test_load_memory_step_skips_without_memory(config: Config) -> None:
    chain = GraphRAGChain(config=config)
    state = {"use_memory": False}
    out = await chain._load_memory_step(state)
    assert out["history"] == ""
    assert out["relevant_entities"] == []


async def test_load_memory_step_skips_without_conversation_id(config: Config) -> None:
    chain = GraphRAGChain(config=config)
    state = {"use_memory": True, "conversation_id": None}
    out = await chain._load_memory_step(state)
    assert out["history"] == ""
    assert out["relevant_entities"] == []


# --- _resolve_strategy ---------------------------------------------------


async def test_resolve_strategy_passthrough_for_explicit_strategy(
    config: Config,
) -> None:
    chain = GraphRAGChain(config=config)
    state = {"search_strategy": SearchStrategy.GLOBAL, "query": "q"}
    out = await chain._resolve_strategy(state)
    # Non-AUTO strategy resolves to itself without invoking the router LLM.
    assert out["resolved_strategy"] == SearchStrategy.GLOBAL


async def test_resolve_strategy_auto_falls_back_to_local_on_error(
    config: Config, mocker
) -> None:
    chain = GraphRAGChain(config=config)
    chain.ignore_errors = True
    # Make the router construction/invocation blow up -> graceful LOCAL fallback.
    mocker.patch.object(
        chain, "_get_chain_for_prompt", side_effect=RuntimeError("no bedrock")
    )
    state = {"search_strategy": SearchStrategy.AUTO, "query": "q"}
    out = await chain._resolve_strategy(state)
    assert out["resolved_strategy"] == SearchStrategy.LOCAL
