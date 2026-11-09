# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a query is checked for before its first model call.

- An empty or blank question was routed, extracted from and searched for:
  ``RAGInput`` strips the query and refuses an empty one.
- A query on a namespace nothing was ingested under paid for its keyword
  extraction (and auto's router) before ``IndexNotFoundError``: the chain
  checks the namespace's text-units index first, once per namespace.
- Global search on a namespace without community reports declined with the
  generic ``no_context``: its abstention reason is ``no_reports``.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError

import unified_kg_rag.adapters.search_strategies  # noqa: F401 - registers them
from tests.fixtures.fakes.chat_models import fake_model_info
from tests.fixtures.fakes.retrievers import FakeRetriever
from unified_kg_rag.adapters.search_strategies.global_search import (
    GlobalSearchStrategy,
)
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.shared import IndexNotFoundError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "query",
    # Regression (rv9 query L8): zero-width space, joiner, BOM and a
    # direction mark alone passed str.strip.
    ["", "   ", "\n\t", "　", "\u200b", " \u200d\ufeff ", "\u200f"],
)
def test_an_empty_query_is_refused(query: str) -> None:
    with pytest.raises(ValidationError, match="The query is empty"):
        RAGInput(query=query)


def test_a_query_is_stripped() -> None:
    assert RAGInput(query="  공급사는 누구인가?\n").query == "공급사는 누구인가?"
    assert RAGInput(query="\ufeff공급사는?\u200b").query == "공급사는?"
    # Inside the text a joiner stays (an emoji sequence).
    assert RAGInput(query="a\u200db").query == "a\u200db"


class _Factory:
    """A model factory that counts the models asked for."""

    def __init__(self) -> None:
        self.models = 0

    def get_model(self, model_id: Any, **kwargs: Any) -> Any:
        self.models += 1
        return GenericFakeChatModel(messages=itertools.repeat(AIMessage("local")))

    def get_model_info(self, model_id: Any) -> Any:
        return fake_model_info(model_id)


class _Retriever(FakeRetriever):
    """A document retriever over namespaces ``ingested``."""

    def __init__(self, ingested: set[str]) -> None:
        super().__init__("text")
        self.ingested = ingested
        self.checks: list[str | None] = []

    async def arequire_namespace(self, suffix: str | None) -> None:
        self.checks.append(suffix)
        if suffix not in self.ingested:
            raise IndexNotFoundError(f"No indices found for suffix '{suffix}'")


def _chain(retriever: _Retriever, factory: _Factory, config: Config) -> GraphRAGChain:
    return GraphRAGChain(
        config=config,
        model_factory=factory,
        retriever_builders={
            RetrieverRole.DOCUMENT: lambda: retriever,
            RetrieverRole.GRAPH: lambda: retriever,
        },
    )


@pytest.mark.parametrize("strategy", [SearchStrategy.AUTO, SearchStrategy.MIX])
async def test_a_namespace_without_an_index_fails_before_any_model_call(
    strategy: SearchStrategy,
) -> None:
    factory = _Factory()
    retriever = _Retriever({"tenant-a"})
    config = Config()
    config.processing.document_parsing.index_value = "tenant-a"
    chain = _chain(retriever, factory, config)
    try:
        with pytest.raises(IndexNotFoundError, match="tenant-b"):
            await chain.ainvoke(
                RAGInput(
                    query="Who supplies Buyer?",
                    suffix="tenant-b",
                    search_strategy=strategy,
                )
            )
        assert factory.models == 0
        # Found once, it is not checked again (the config's index_value).
        for _ in range(2):
            await chain.asearch(
                RAGInput(
                    query="Who supplies Buyer?", search_strategy=SearchStrategy.NAIVE
                )
            )
    finally:
        await chain.aclose()
    assert retriever.checks == ["tenant-b", "tenant-a"]


async def test_under_ignore_errors_a_missing_namespace_is_the_error_answer() -> None:
    factory = _Factory()
    config = Config()
    config.processing.ignore_errors = True
    chain = _chain(_Retriever(set()), factory, config)
    try:
        out = await chain.ainvoke(
            RAGInput(query="q", search_strategy=SearchStrategy.MIX)
        )
    finally:
        await chain.aclose()
    assert out.metadata["error"] is True
    assert out.metadata["error_class"] == "index_not_found"
    assert factory.models == 0


class _NoReports:
    """A document retriever whose namespace holds no community report."""

    async def aindex_limits(self, prefix: str, suffix: str | None) -> Any:
        from unified_kg_rag.adapters.storage.filter_schema import IndexLimits

        return IndexLimits()

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        return []


async def test_global_search_on_a_namespace_without_reports_says_so() -> None:
    # Regression (rv7 algo-ret L8): the decline read as an ordinary
    # no_context one; the namespace holds no community report to select.
    config = Config()
    config.providers.rerank.kind = "none"
    strategy = GlobalSearchStrategy(
        config=config, retrievers={RetrieverRole.DOCUMENT.value: _NoReports()}
    )
    result = await strategy.asearch(SearchQuery(query="What are the main themes?"))
    assert result.results == []
    assert result.metadata["no_context_reason"] == "no_reports"
    # The chain declines with it.
    state = {"search_results": result}
    assert GraphRAGChain._no_context_reason(state) == "no_reports"
    result.metadata.pop("no_context_reason")
    assert GraphRAGChain._no_context_reason(state) == "no_context"
