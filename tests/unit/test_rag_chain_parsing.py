# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GraphRAGChain parsing + state-graph node helpers (AWS-free).

Covers the ``RAGInput``/``RAGOutput``/``ProcessedQuery`` model boundaries,
``_prepare_invoke`` conversation-id handling, the LightRAG mode predicate, and
the synchronous output-formatting nodes. Does NOT duplicate the strategy/
retriever dispatch in ``test_rag_chain_dispatch`` or the router parsing already
asserted elsewhere.
"""

from __future__ import annotations

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from unified_kg_rag.adapters.retrieval.token_manager import EMPTY_CONTEXT_PLACEHOLDER
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


@pytest.mark.parametrize(
    "field", [{"top_k": 0}, {"retrieval_multiplier": 0}, {"max_tokens": -5}]
)
@pytest.mark.parametrize("model", [RAGInput, SearchQuery])
def test_query_sizes_must_be_positive(model, field: dict) -> None:
    # 0 results, a 0x depth or a negative budget are not a query.
    with pytest.raises(ValidationError):
        model(query="q", **field)


def test_rag_input_composes_decomposed_hangul() -> None:
    # Indexed text is NFC; an NFD query ("보증", conjoining jamo) must match it.
    nfd = "\u1107\u1169\u110c\u1173\u11bc"
    assert RAGInput(query=nfd).query == "보증"
    assert RAGInput(query="ＡＢＣ").query == "ＡＢＣ"  # NFC, not NFKC


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
    # sources project content/source/score/metadata. `content` is required: a vector
    # result carries its text there and nowhere else, so dropping it left the
    # highest-scoring sources with no text for RAGAS / recall scoring to match on.
    assert out.sources[0]["source"] == "doc-1"
    assert out.sources[0]["content"] == "c1"
    # The budget-truncation flag lives in metadata; the four keys stay.
    assert set(out.sources[0].keys()) == {"content", "source", "score", "metadata"}
    assert out.sources[0]["metadata"]["truncated"] is False
    # retriever_type / chunk_id stay out — the projection is still explicit, not a
    # full model_dump.
    assert "retriever_type" not in out.sources[0]


def test_format_output_step_excludes_synthesized_results_from_sources() -> None:
    # The global map-reduce summary stays in the search results (and thus the
    # answer context) but is LLM output, so it must not be reported as a source.
    sr = _search_result()
    sr.results.insert(
        0,
        RetrievalResult(
            content="summary of the reports",
            score=1.0,
            source="synthesized_summary",
            retriever_type="general",
            metadata={"synthesized": True},
        ),
    )
    state = {
        "search_results": sr,
        "resolved_strategy": SearchStrategy.GLOBAL,
        "start_time": 0.0,
        "answer": "the answer",
        "processed_query": ProcessedQuery(original_query="q", final_query="q"),
    }
    out = GraphRAGChain._format_output_step(state)
    assert [s["source"] for s in out.sources] == ["doc-1"]
    assert len(out.search_results.results) == 2


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


# --- context building and the follow-up rewrite ----------------------------


def _ctx_state(history: str) -> dict:
    return {
        "query": "what about its price?",
        "processed_query": ProcessedQuery(original_query="q", final_query="q"),
        "search_results": _search_result(),
        "history": history,
    }


def test_context_building_returns_the_budgeted_context(config: Config) -> None:
    chain = GraphRAGChain(config=config)

    def _no_llm(*a, **k):
        raise AssertionError("context building makes no LLM call")

    chain._get_chain_for_prompt = _no_llm  # type: ignore[assignment]
    # History no longer reaches context building: it shaped the question.
    out = chain._context_building_step(_ctx_state("earlier turn"))
    assert out and out != EMPTY_CONTEXT_PLACEHOLDER


def test_context_building_empty_retrieval_answers_no_data(config: Config) -> None:
    chain = GraphRAGChain(config=config)
    state = _ctx_state("earlier turn about the Vendor")
    state["search_results"] = SearchResult(
        query=SearchQuery(query="q"),
        results=[],
        total_results=0,
        search_strategy="local",
        processing_time=0.0,
    )
    context = chain._context_building_step(state)
    assert context == EMPTY_CONTEXT_PLACEHOLDER

    answer = chain._answer_generation_step({"context": context}).invoke({})
    assert answer.startswith("I could not find relevant information")


class _FakeRewriter:
    def __init__(self, output: str = "What is the price of the Widget?") -> None:
        self.output = output
        self.inputs: list[dict] = []

    async def ainvoke(self, inputs, config=None):
        self.inputs.append(inputs)
        return self.output


async def test_rewrite_skips_the_llm_without_history(config: Config) -> None:
    chain = GraphRAGChain(config=config)

    def _no_llm(*a, **k):
        raise AssertionError("no rewrite without history")

    chain._get_chain_for_prompt = _no_llm  # type: ignore[assignment]
    assert await chain._rewrite_query_step(_ctx_state("")) is None


async def test_rewrite_resolves_a_follow_up_from_history(config: Config) -> None:
    chain = GraphRAGChain(config=config)
    rewriter = _FakeRewriter(" What is the price of the Widget? \n")
    chain._get_chain_for_prompt = lambda *a, **k: rewriter  # type: ignore[assignment]

    out = await chain._rewrite_query_step(_ctx_state("Human: Tell me about the Widget"))

    assert out == "What is the price of the Widget?"
    assert rewriter.inputs == [
        {
            "query": "what about its price?",
            "conversation_history": "Human: Tell me about the Widget",
        }
    ]


async def test_rewrite_failure_degrades_under_ignore_errors(config: Config) -> None:
    chain = GraphRAGChain(config=config)

    def _boom(*a, **k):
        raise RuntimeError("rewriter down")

    chain._get_chain_for_prompt = _boom  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        await chain._rewrite_query_step(_ctx_state("earlier turn"))
    chain.ignore_errors = True
    assert await chain._rewrite_query_step(_ctx_state("earlier turn")) is None


async def test_follow_up_is_searched_and_answered_as_the_standalone_question(
    config: Config, mocker
) -> None:
    chain = GraphRAGChain(config=config)
    rewriter = _FakeRewriter()
    answers: list[dict] = []

    def _answer(inputs: dict) -> str:
        answers.append(inputs)
        return "It costs 10."

    def _chain_for(prompt_class, *a, **k):
        if prompt_class.__name__ == "QueryRewritePrompt":
            return rewriter
        return RunnableLambda(_answer)

    chain._get_chain_for_prompt = _chain_for  # type: ignore[assignment]
    mocker.patch.object(
        chain.memory_manager,
        "load_context",
        mocker.AsyncMock(return_value=("Human: Tell me about the Widget", [])),
    )
    mocker.patch.object(chain.memory_manager, "add_turn", mocker.AsyncMock())
    searched: list[str] = []

    async def _search(state, config=None):
        searched.append(state["processed_query"].final_query)
        return _search_result()

    mocker.patch.object(chain, "_search_step", _search)
    chain.chain = chain._build_chain()

    output = await chain.ainvoke(
        {
            "query": "what about its price?",
            "search_strategy": "simple",
            "use_memory": True,
            "conversation_id": "c-1",
            "enable_query_processing": False,
        }
    )

    assert searched == ["What is the price of the Widget?"]
    assert answers[0]["query"] == "What is the price of the Widget?"
    assert output.processed_query.original_query == "what about its price?"
    assert output.processed_query.standalone_query == (
        "What is the price of the Widget?"
    )


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
