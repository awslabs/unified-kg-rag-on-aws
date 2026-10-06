# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free end-to-end GraphRAGChain.ainvoke round-trips.

Drives the headline retrieval object through its full assembled LCEL chain
(strategy resolution -> search -> context build -> answer generation ->
output format) using the hexagonal DI seams: a fake LLM factory (whose
``get_model`` returns a canned-answer Runnable) injected via ``model_factory``,
and fake retrievers injected via ``retriever_builders`` keyed by role. No
Bedrock / Neptune / OpenSearch. This closes the gap where the only full-chain
exercise lived in the real-AWS (CI-skipped) test.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda

import unified_kg_rag.adapters.search_strategies  # noqa: F401  (registers strategies)
from unified_kg_rag.application.retrieval.rag_chain import (
    NO_CONTEXT_ANSWER,
    ChainMode,
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluatorType,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.evaluation import EvaluationManager

pytestmark = pytest.mark.integration

_CANNED_ANSWER = "Vendor supplies Buyer with parts under the agreement."


class _FakeModelFactory:
    """ModelFactoryPort: get_model returns a canned-answer Runnable.

    Every LLM step in the chain (here only answer generation, since query
    processing is disabled and the strategy is explicit) resolves through this,
    so no Bedrock client is ever used.
    """

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        return RunnableLambda(lambda _prompt_value: _CANNED_ANSWER)

    def get_model_info(self, model_id: Any) -> Any:
        return None


class _FakeRetriever:
    """Returns one fixed hit; records the queries it received."""

    def __init__(self, tag: str, *, empty: bool = False) -> None:
        self.tag = tag
        self.empty = empty
        self.calls: list[SearchQuery] = []

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        self.calls.append(query)
        if self.empty:
            return []
        return [
            RetrievalResult(
                content=f"{self.tag}: Vendor and Buyer signed the supply agreement.",
                score=0.9,
                source=f"{self.tag}-doc-1",
                retriever_type=self.tag,
                metadata={"id": f"{self.tag}-1", "text_unit_ids": []},
            )
        ]


def _make_chain(
    config: Config, mode: ChainMode, *, empty_retrieval: bool = False
) -> GraphRAGChain:
    doc_retriever = _FakeRetriever("document", empty=empty_retrieval)
    graph_retriever = _FakeRetriever("graph", empty=empty_retrieval)
    chain = GraphRAGChain(
        config=config,
        mode=mode,
        model_factory=_FakeModelFactory(),
        retriever_builders={
            RetrieverRole.DOCUMENT: lambda: doc_retriever,
            RetrieverRole.GRAPH: lambda: graph_retriever,
        },
    )
    # Keep token counting local (no Bedrock count_tokens call) by using the
    # script-aware estimate the manager already falls back to.
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    return chain


async def test_rag_mode_round_trip_produces_answer_and_sources(
    config: Config,
) -> None:
    chain = _make_chain(config, ChainMode.RAG)
    out = await chain.ainvoke(
        RAGInput(
            query="What does Vendor supply to Buyer?",
            search_strategy=SearchStrategy.SIMPLE,  # DOCUMENT role only
            enable_query_processing=False,  # no translation/entity-extraction LLM
        )
    )
    assert isinstance(out, RAGOutput)
    assert out.answer == _CANNED_ANSWER
    assert "abstained" not in out.metadata
    # The fake retriever hit flows through to sources.
    assert out.sources and out.sources[0]["source"] == "document-doc-1"
    assert out.search_results.total_results >= 1
    assert out.metadata["search_strategy"] == SearchStrategy.SIMPLE.value


async def test_search_mode_round_trip_returns_results_without_llm_answer(
    config: Config,
) -> None:
    chain = _make_chain(config, ChainMode.SEARCH)
    out = await chain.ainvoke(
        RAGInput(
            query="Vendor Buyer supply agreement",
            search_strategy=SearchStrategy.SIMPLE,
            enable_query_processing=False,
        )
    )
    # SEARCH mode skips answer generation; results still come back.
    result = out if isinstance(out, dict) else out.model_dump()
    assert result["search_results"]["total_results"] >= 1


async def test_rag_mode_empty_retrieval_short_circuits_answer_generation(
    config: Config,
) -> None:
    # When retrieval yields no context, the chain must NOT invoke the answer LLM
    # (which would hallucinate); it returns an explicit "cannot answer" instead.
    chain = _make_chain(config, ChainMode.RAG, empty_retrieval=True)
    out = await chain.ainvoke(
        RAGInput(
            query="What does Vendor supply to Buyer?",
            search_strategy=SearchStrategy.SIMPLE,
            enable_query_processing=False,
        )
    )
    assert isinstance(out, RAGOutput)
    # The canned LLM answer must NOT appear; the refusal sentinel must.
    assert out.answer != _CANNED_ANSWER
    assert "could not find relevant information" in out.answer.lower()
    assert out.answer == NO_CONTEXT_ANSWER
    assert out.metadata["abstained"] is True
    assert out.sources == []


async def test_rag_mode_local_strategy_uses_graph_and_document_roles(
    config: Config,
) -> None:
    # LOCAL needs both roles; this exercises the multi-role injection seam end to
    # end (graph expansion + document retrieval) through the full chain.
    chain = _make_chain(config, ChainMode.RAG)
    out = await chain.ainvoke(
        RAGInput(
            query="Who are the parties?",
            search_strategy=SearchStrategy.LOCAL,
            enable_query_processing=False,
        )
    )
    assert isinstance(out, RAGOutput)
    assert out.answer == _CANNED_ANSWER


class _LineageStore:
    """Document store whose search hit is an entity with chunk lineage only."""

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        ids = (query.filters or {}).get("id")
        if ids:  # the evaluator's text-unit id fetch
            return [
                RetrievalResult(
                    content="",
                    score=0.0,
                    source=i,
                    retriever_type="text",
                    metadata={"id": i, "attributes": {"file_name": "supply.txt"}},
                )
                for i in ids
            ]
        return [
            RetrievalResult(
                content="Vendor: supplies parts to Buyer.",
                score=0.9,
                source="entity-1",
                retriever_type="entity",
                metadata={"id": "entity-1", "text_unit_ids": ["tu-1"]},
            )
        ]


async def test_evaluation_attributes_graph_sources_via_chain_store(
    config: Config,
) -> None:
    # The evaluation manager's default resolver reuses the chain's DOCUMENT
    # binding to map an entity's text-unit lineage to its file.
    store = _LineageStore()
    chain = GraphRAGChain(
        config=config,
        model_factory=_FakeModelFactory(),
        retriever_builders={RetrieverRole.DOCUMENT: lambda: store},
    )
    chain.token_manager.count_tokens = lambda text: len((text or "").split())
    config.evaluation.enabled_evaluators = [EvaluatorType.RETRIEVAL]
    manager = EvaluationManager(config, rag_chain=chain)
    queries = [
        EvaluationQuery(
            query_id="q1",
            question="What does Vendor supply?",
            metadata={"search_strategy": "simple", "enable_query_processing": False},
        )
    ]
    gts = [
        EvaluationGroundTruth(
            query_id="q1", ground_truth="", reference_sources=["supply.txt"]
        )
    ]
    results, _, summary = await manager.evaluate_dataset(
        queries, gts, show_progress=False
    )
    assert results[0].retrieved_source_ids == [["supply.txt"]]
    assert summary.metric_statistics["hit_at_k"]["mean"] == 1.0
