# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public retrieval API (stable facade).

The implementation now lives across the hexagonal layers — abstract bases and
support in ``adapters.retrieval``, concrete backends in ``adapters.retrievers``/
``adapters.search_strategies``, and the chain orchestration in
``application.retrieval``. This module re-exports the public surface so callers
keep a single import path.
"""

from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
)
from unified_kg_rag.adapters.retrieval.hybrid_scorer import HybridScorer
from unified_kg_rag.adapters.retrieval.memory_manager import (
    GraphRAGChatMessageHistory,
    GraphRAGConversationBufferMemory,
    MemoryManager,
)
from unified_kg_rag.adapters.retrieval.token_manager import (
    OptimizedContext,
    TokenManager,
)
from unified_kg_rag.adapters.retrievers import NeptuneRetriever, OpenSearchRetriever
from unified_kg_rag.adapters.search_strategies import (
    DriftSearchStrategy,
    GlobalSearchStrategy,
    LocalSearchStrategy,
    SimpleSearchStrategy,
)
from unified_kg_rag.application.retrieval.rag_chain import (
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
