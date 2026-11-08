# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stand-in for the chain ``run-rag`` builds."""

from __future__ import annotations

from typing import Any

from unified_kg_rag.application.cli import run_rag_chain
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain


def patch_chain(mocker: Any, **kwargs: Any) -> Any:
    """Patch the ``GraphRAGChain`` ``run-rag`` constructs (``kwargs`` go to
    ``mocker.patch.object``: ``return_value=``, ``side_effect=``); the CLI's
    filter conversion (``GraphRAGChain.typed_filter_values``) stays real.
    Returns the mock class."""
    chain_cls = mocker.patch.object(run_rag_chain, "GraphRAGChain", **kwargs)
    chain_cls.typed_filter_values = GraphRAGChain.typed_filter_values
    return chain_cls


def answer_document(
    answer: str = "ok",
    *,
    sources: list[dict[str, Any]] | None = None,
    citations: list[Any] | None = None,
    metadata: dict[str, Any] | None = None,
    search_metadata: dict[str, Any] | None = None,
    results: list[dict[str, Any]] | None = None,
    processed_query: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A successful ``--mode rag`` JSON document of ``run-rag`` (what
    ``_run_query`` returns), from a real ``RAGOutput``."""
    from unified_kg_rag.application.retrieval.rag_chain import (
        ProcessedQuery,
        RAGOutput,
    )
    from unified_kg_rag.domain.models import SearchQuery, SearchResult

    output = RAGOutput(
        answer=answer,
        sources=sources or [],
        search_results=SearchResult(
            query=SearchQuery(query="q"),
            results=results or [],
            total_results=len(results or []),
            search_strategy="local",
            processing_time=0.0,
            metadata=search_metadata or {},
        ),
        citations=citations or [],
        conversation_id=None,
        processed_query=ProcessedQuery.model_validate(
            processed_query or {"original_query": "q", "final_query": "q"}
        ),
        metadata=metadata or {},
    )
    return {"success": True, **output.model_dump()}


def search_document(
    results: list[dict[str, Any]], *, metadata: dict[str, Any] | None = None
) -> dict[str, Any]:
    """A successful ``--mode search`` JSON document of ``run-rag``."""
    from unified_kg_rag.application.retrieval.rag_chain import (
        ProcessedQuery,
        SearchOutput,
    )
    from unified_kg_rag.domain.models import SearchQuery, SearchResult

    output = SearchOutput(
        search_results=SearchResult(
            query=SearchQuery(query="q"),
            results=results,
            total_results=len(results),
            search_strategy="local",
            processing_time=0.0,
        ),
        processed_query=ProcessedQuery(original_query="q", final_query="q"),
        metadata=metadata or {},
    )
    return {"success": True, **output.model_dump()}
