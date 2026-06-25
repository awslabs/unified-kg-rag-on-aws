# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Teardown delegation for retrievers / indexers / GraphRAGChain (AWS-free).

NeptuneClient owned by a retriever opens a websocket + thread pool, and
OpenSearchClient opens sync/async HTTP pools; these survive until GC unless the
owner closes them. These tests assert close()/aclose() delegate to the
underlying clients, and that GraphRAGChain.aclose() closes every cached
retriever. Owners are built via ``__new__`` so no real connection opens.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aws_graphrag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from aws_graphrag.adapters.retrievers.opensearch_retriever import OpenSearchRetriever
from aws_graphrag.adapters.storage.neptune_indexer import NeptuneIndexer
from aws_graphrag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from aws_graphrag.application.retrieval.rag_chain import GraphRAGChain

pytestmark = pytest.mark.unit


# --- retrievers ----------------------------------------------------------


def test_opensearch_retriever_close_delegates() -> None:
    inst = OpenSearchRetriever.__new__(OpenSearchRetriever)
    client = MagicMock()
    object.__setattr__(inst, "_opensearch_client", client)
    inst.close()
    client.close.assert_called_once()


async def test_opensearch_retriever_aclose_delegates() -> None:
    inst = OpenSearchRetriever.__new__(OpenSearchRetriever)
    client = MagicMock()
    client.aclose = AsyncMock()
    object.__setattr__(inst, "_opensearch_client", client)
    await inst.aclose()
    client.aclose.assert_awaited_once()


def test_neptune_retriever_close_delegates() -> None:
    inst = NeptuneRetriever.__new__(NeptuneRetriever)
    client = MagicMock()
    object.__setattr__(inst, "_neptune_client", client)
    inst.close()
    client.close.assert_called_once()


# --- indexers ------------------------------------------------------------


def test_opensearch_indexer_close_delegates() -> None:
    inst = OpenSearchIndexer.__new__(OpenSearchIndexer)
    client = MagicMock()
    inst.opensearch_client = client
    inst.close()
    client.close.assert_called_once()


def test_neptune_indexer_close_delegates() -> None:
    inst = NeptuneIndexer.__new__(NeptuneIndexer)
    client = MagicMock()
    inst.neptune_client = client
    inst.close()
    client.close.assert_called_once()


# --- GraphRAGChain -------------------------------------------------------


async def test_chain_aclose_closes_all_cached_retrievers() -> None:
    chain = GraphRAGChain.__new__(GraphRAGChain)
    r1 = MagicMock()
    r1.aclose = AsyncMock()
    r2 = MagicMock()
    r2.aclose = AsyncMock()
    chain._retriever_cache = {("graph", None): r1, ("document", None): r2}

    await chain.aclose()

    r1.aclose.assert_awaited_once()
    r2.aclose.assert_awaited_once()
    assert chain._retriever_cache == {}


async def test_chain_aclose_never_raises_on_broken_retriever() -> None:
    chain = GraphRAGChain.__new__(GraphRAGChain)
    bad = MagicMock()
    bad.aclose = AsyncMock(side_effect=RuntimeError("boom"))
    chain._retriever_cache = {("graph", None): bad}
    # Best-effort: a failing retriever close must not propagate.
    await chain.aclose()
    assert chain._retriever_cache == {}


def test_chain_close_closes_all_cached_retrievers() -> None:
    chain = GraphRAGChain.__new__(GraphRAGChain)
    r1 = MagicMock()
    r2 = MagicMock()
    chain._retriever_cache = {("graph", None): r1, ("document", None): r2}
    chain.close()
    r1.close.assert_called_once()
    r2.close.assert_called_once()
    assert chain._retriever_cache == {}
