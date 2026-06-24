# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public retrieval API (stable facade).

The implementation now lives across the hexagonal layers — abstract bases and
support in ``adapters.retrieval``, concrete backends in ``adapters.retrievers``/
``adapters.search_strategies``, and the chain orchestration in
``application.retrieval``. This module re-exports the public surface so callers
keep a single import path.
"""

from aws_graphrag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
)
from aws_graphrag.adapters.retrieval.hybrid_scorer import HybridScorer
from aws_graphrag.adapters.retrieval.memory_manager import (
    GraphRAGChatMessageHistory,
    GraphRAGConversationBufferMemory,
    MemoryManager,
)
from aws_graphrag.adapters.retrieval.token_manager import (
    OptimizedContext,
    TokenManager,
)
from aws_graphrag.adapters.retrievers import NeptuneRetriever, OpenSearchRetriever
from aws_graphrag.adapters.search_strategies import (
    DriftSearchStrategy,
    GlobalSearchStrategy,
    LocalSearchStrategy,
    SimpleSearchStrategy,
)
from aws_graphrag.application.retrieval.rag_chain import (
    ChainMode,
    GraphRAGChain,
    RAGInput,
    RAGOutput,
    create_rag_chain,
)

__all__ = [
    "BaseGraphRAGRetriever",
    "BaseSearchStrategy",
    "ChainMode",
    "DriftSearchStrategy",
    "GlobalSearchStrategy",
    "GraphRAGChatMessageHistory",
    "GraphRAGChain",
    "GraphRAGConversationBufferMemory",
    "HybridScorer",
    "LocalSearchStrategy",
    "MemoryManager",
    "NeptuneRetriever",
    "OpenSearchRetriever",
    "OptimizedContext",
    "RAGInput",
    "RAGOutput",
    "SimpleSearchStrategy",
    "TokenManager",
    "create_rag_chain",
]
