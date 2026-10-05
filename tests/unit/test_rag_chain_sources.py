# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free tests: reported ``sources`` match what the answer model saw.

``RAGOutput.sources`` used to list every search result, including sections the
token budgeter cut, and carried raw stored documents (embedding vectors
included) as metadata. These tests pin the corrected contract: sources are the
budgeter's selection in retrieval-rank order, truncated sections are flagged,
provenance (document ids / chunk id / type / score) is kept, and no
``*_embedding`` field reaches source metadata.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from unified_kg_rag.adapters.retrieval.token_manager import (
    EMPTY_CONTEXT_PLACEHOLDER,
    OptimizedContext,
)
from unified_kg_rag.application.retrieval.rag_chain import (
    ChainMode,
    GraphRAGChain,
    ProcessedQuery,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)

pytestmark = pytest.mark.unit

_CANNED_ANSWER = "Vendor ships parts to Buyer monthly."


def _words(tag: str, n: int) -> str:
    return " ".join(f"{tag}{i}" for i in range(n))


def _text_result(i: int, *, words: int, score: float) -> RetrievalResult:
    return RetrievalResult(
        content=_words(f"w{i}_", words),
        score=score,
        source=f"chunk-{i}",
        retriever_type="text",
        metadata={
            "id": f"chunk-{i}",
            "document_ids": [f"doc-{i}"],
            "text_embedding": [0.1, 0.2, 0.3],
        },
    )


def _chain(config: Config, mode: ChainMode = ChainMode.RAG, **kw: Any) -> GraphRAGChain:
    config = config.model_copy(deep=True)  # don't leak the tweak below
    config.search.token_manager.min_truncated_section_tokens = 10
    chain = GraphRAGChain(config=config, mode=mode, **kw)
    # Local whitespace token count (no Bedrock count_tokens call).
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    return chain


def _state(results: list[RetrievalResult], max_tokens: int) -> dict[str, Any]:
    return {
        "search_results": SearchResult(
            query=SearchQuery(query="q"),
            results=results,
            total_results=len(results),
            search_strategy="pending",
            processing_time=0.0,
        ),
        "resolved_strategy": SearchStrategy.SIMPLE,
        "start_time": 0.0,
        "answer": _CANNED_ANSWER,
        "conversation_id": None,
        "processed_query": ProcessedQuery(original_query="q", final_query="q"),
        "max_tokens": max_tokens,
        "history": "",
    }


def _run_context_and_format(
    chain: GraphRAGChain, state: dict[str, Any]
) -> tuple[OptimizedContext, RAGOutput]:
    state["optimized_context"] = chain._context_optimization_step(state)
    context = chain._context_building_step(state)
    # The step may be sync or async depending on the context-building path;
    # accept both so this helper does not pin that implementation detail.
    if inspect.isawaitable(context):
        context = asyncio.run(context)
    state["context"] = context
    return state["optimized_context"], GraphRAGChain._format_output_step(state)


def test_sources_exclude_budget_cut_sections_and_flag_truncation(
    config: Config,
) -> None:
    # Available budget = max_tokens - query(1) - buffer(512) = 250 tokens, all
    # TEXT. chunk-0 and chunk-1 (100 each) fit, chunk-2 is truncated to the
    # remaining 50, and chunk-3 is cut entirely.
    chain = _chain(config)
    results = [
        _text_result(0, words=100, score=0.9),
        _text_result(1, words=100, score=0.8),
        _text_result(2, words=100, score=0.7),
        _text_result(3, words=100, score=0.6),
    ]
    optimized, out = _run_context_and_format(chain, _state(results, 763))

    assert optimized.sections_excluded == 1
    assert [s["source"] for s in out.sources] == ["chunk-0", "chunk-1", "chunk-2"]
    assert "chunk-3" not in {s["source"] for s in out.sources}

    by_source = {s["source"]: s for s in out.sources}
    assert by_source["chunk-0"]["truncated"] is False
    assert by_source["chunk-0"]["content"] == results[0].content
    truncated = by_source["chunk-2"]
    assert truncated["truncated"] is True
    assert truncated["metadata"]["truncated"] is True
    # The reported content is the truncated text the model saw, not the original.
    assert truncated["content"] != results[2].content
    assert truncated["content"].endswith("…")

    assert out.metadata["context_sections_included"] == 3
    assert out.metadata["context_sections_excluded"] == 1


def test_sources_keep_provenance_and_drop_embeddings(config: Config) -> None:
    chain = _chain(config)
    results = [_text_result(0, words=20, score=0.9)]
    _optimized, out = _run_context_and_format(chain, _state(results, 4000))

    (source,) = out.sources
    meta = source["metadata"]
    assert not [k for k in meta if k.endswith("_embedding")]
    assert meta["document_ids"] == ["doc-0"]
    assert meta["chunk_id"] == "chunk-0"
    assert meta["section_type"] == "text"
    assert meta["score"] == 0.9
    assert meta["source_id"] == "chunk-0"
    # Backward compatible: the original keys are all still present.
    assert {"content", "source", "score", "metadata"} <= set(source)


def test_sources_follow_retrieval_rank_not_section_type_order(
    config: Config,
) -> None:
    # The budgeter groups by section type; sources are reported in the order
    # retrieval ranked them.
    chain = _chain(config)
    results = [
        RetrievalResult(
            content="Vendor is a supplier",
            score=0.9,
            source="entity-1",
            retriever_type="entity",
            metadata={"description_embedding": [0.5]},
        ),
        _text_result(1, words=10, score=0.8),
    ]
    _optimized, out = _run_context_and_format(chain, _state(results, 4000))
    assert [s["source"] for s in out.sources] == ["entity-1", "chunk-1"]
    assert out.sources[0]["metadata"]["document_ids"] == []
    assert out.sources[0]["metadata"]["chunk_id"] is None


def test_sources_empty_when_model_saw_no_context(config: Config) -> None:
    chain = _chain(config)
    state = _state([], 4000)
    _optimized, out = _run_context_and_format(chain, state)
    assert state["context"] == EMPTY_CONTEXT_PLACEHOLDER
    assert out.sources == []


# --------------------------------------------------------------------------- #
# Full chain (fake LLM factory + fake retriever via the DI seams)
# --------------------------------------------------------------------------- #


class _FakeModelFactory:
    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return RunnableLambda(lambda _prompt_value: _CANNED_ANSWER)

    def get_model_info(self, model_id: Any) -> Any:
        return None


class _FakeRetriever:
    def __init__(self, results: list[RetrievalResult]) -> None:
        self._results = results

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        return [r.model_copy(deep=True) for r in self._results]


async def test_full_chain_sources_are_budgeted_selection_without_vectors(
    config: Config,
) -> None:
    hits = [_text_result(i, words=100, score=0.9 - i * 0.1) for i in range(4)]
    retriever = _FakeRetriever(hits)
    chain = _chain(
        config,
        model_factory=_FakeModelFactory(),
        retriever_builders={
            RetrieverRole.DOCUMENT: lambda: retriever,
            RetrieverRole.GRAPH: lambda: retriever,
        },
    )
    out = await chain.ainvoke(
        RAGInput(
            query="q",
            search_strategy=SearchStrategy.SIMPLE,
            enable_query_processing=False,
            max_tokens=763,
        )
    )
    assert isinstance(out, RAGOutput)
    assert out.answer == _CANNED_ANSWER
    retrieved = {r.source for r in out.search_results.results}
    reported = [s["source"] for s in out.sources]
    # Every reported source was retrieved, and something was cut for budget.
    assert set(reported) <= retrieved
    assert len(reported) < len(out.search_results.results)
    assert out.metadata["context_sections_excluded"] >= 1
    for source in out.sources:
        assert not [k for k in source["metadata"] if k.endswith("_embedding")]
