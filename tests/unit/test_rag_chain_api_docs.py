# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The library entry points document their contract (``help()`` shows it)."""

from __future__ import annotations

import pytest

from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    RAGInput,
    RAGOutput,
    SearchOutput,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("obj", "phrases"),
    [
        (GraphRAGChain, ["close", "aclose", "search", "SearchOutput", "event loop"]),
        (RAGInput, ["namespace", "suffix", "processing.document_parsing.namespace"]),
        (RAGOutput, ["sources"]),
        (SearchOutput, ["search", "metadata"]),
        (GraphRAGChain.invoke, ["RAGOutput", "ainvoke"]),
        (GraphRAGChain.ainvoke, ["RAGOutput", "ignore_errors"]),
        (GraphRAGChain.search, ["asearch"]),
        (GraphRAGChain.asearch, ["without generating an answer", "memory"]),
        (GraphRAGChain.stream, ["str", "ainvoke", "sources"]),
        (GraphRAGChain.astream, ["str", "ainvoke", "sources"]),
        (GraphRAGChain.aquery_events, ["QueryResult", "AnswerChunk", "ainvoke"]),
        (GraphRAGChain.query_events, ["aquery_events", "cancels"]),
    ],
)
def test_public_api_documents_its_contract(obj: object, phrases: list[str]) -> None:
    doc = obj.__doc__ or ""
    missing = [p for p in phrases if p not in doc]
    assert not missing, f"{getattr(obj, '__qualname__', obj)} docstring lacks {missing}"
