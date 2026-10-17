# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified Knowledge Graph RAG on AWS.

The names exported here are the package's public API; everything else is
internal and may change in any release. Two groups:

- **Using the framework**: load a :class:`Config` (:func:`get_config`),
  ingest a corpus (:class:`DataIngestionPipeline`, :class:`PipelineConfig`),
  query it (:class:`GraphRAGChain` with :class:`RAGInput`, answering with a
  :class:`RAGOutput` or, from ``search``, a :class:`SearchOutput`) and
  evaluate it (:class:`EvaluationManager`).
- **Extending it**: the ports a backend implements, the :class:`Providers`
  bundle that injects them, and the registries for providers, search
  strategies, parsers and renderers (see the design doc's extension guide).

Names load on first access, so ``import unified_kg_rag`` stays cheap and
pulls in no model, AWS or store SDK.

``__version__`` is the single source of truth for the package version:
``[tool.hatch.version]`` reads it from this file, and the CLIs import it.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.2.3"

# Public name -> the module that defines it.
_EXPORTS: dict[str, str] = {
    # Configuration and logging
    "Config": "unified_kg_rag.domain.models",
    "ModelRole": "unified_kg_rag.domain.models",
    "get_config": "unified_kg_rag.shared.config",
    "setup_logging": "unified_kg_rag.shared.logging",
    "configure_event_loop": "unified_kg_rag.shared.utils.event_loop",
    # Ingestion and indexing
    "DataIngestionPipeline": "unified_kg_rag.application.ingestion.pipeline",
    "PipelineConfig": "unified_kg_rag.domain.models",
    "PipelineContext": "unified_kg_rag.domain.models",
    "PipelineStageStatus": "unified_kg_rag.domain.models",
    "PipelineStageType": "unified_kg_rag.domain.models",
    "IndexingManager": "unified_kg_rag.application.storage.indexing_manager",
    # Querying
    "GraphRAGChain": "unified_kg_rag.application.retrieval.rag_chain",
    "RAGInput": "unified_kg_rag.application.retrieval.rag_chain",
    "RAGOutput": "unified_kg_rag.application.retrieval.rag_chain",
    "SearchOutput": "unified_kg_rag.application.retrieval.rag_chain",
    "ProcessedQuery": "unified_kg_rag.application.retrieval.rag_chain",
    "SearchResult": "unified_kg_rag.domain.models",
    "SearchStrategy": "unified_kg_rag.domain.models",
    "SearchType": "unified_kg_rag.domain.models",
    "MemoryManager": "unified_kg_rag.adapters.retrieval.memory_manager",
    "ConversationKey": "unified_kg_rag.adapters.retrieval.memory_manager",
    # Evaluation
    "EvaluationManager": "unified_kg_rag.evaluation.evaluation_manager",
    # Extension: provider injection and the provider registry
    "Providers": "unified_kg_rag.adapters.providers",
    "build_providers": "unified_kg_rag.application.bootstrap",
    "ProviderRole": "unified_kg_rag.adapters.provider_registry",
    "register_provider": "unified_kg_rag.adapters.provider_registry",
    # Extension: ports
    "ModelFactoryPort": "unified_kg_rag.ports",
    "LLMFactoryPort": "unified_kg_rag.ports",
    "EmbeddingFactoryPort": "unified_kg_rag.ports",
    "RerankFactoryPort": "unified_kg_rag.ports",
    "TokenCounterPort": "unified_kg_rag.ports",
    "GraphIndexer": "unified_kg_rag.ports",
    "VectorIndexer": "unified_kg_rag.ports",
    "DocStatusPort": "unified_kg_rag.ports",
    "ContentGuardPort": "unified_kg_rag.ports",
    "GuardVerdict": "unified_kg_rag.ports",
    "GuardAction": "unified_kg_rag.ports",
    "GuardSource": "unified_kg_rag.ports",
    # Extension: search strategies and retrievers
    "BaseSearchStrategy": "unified_kg_rag.adapters.retrieval.base",
    "BaseGraphRAGRetriever": "unified_kg_rag.adapters.retrieval.base",
    "register_strategy": "unified_kg_rag.domain.retrieval.strategy_registry",
    "QueryInput": "unified_kg_rag.domain.retrieval.strategy_registry",
    "RetrieverRole": "unified_kg_rag.domain.models",
    # Extension: parsers and renderers
    "ParserFactory": "unified_kg_rag.adapters.ingestion.parser",
    "BaseRenderer": "unified_kg_rag.adapters.renderers.base",
    "register_renderer": "unified_kg_rag.adapters.renderers.base",
}

__all__ = ["__version__", *_EXPORTS]

if TYPE_CHECKING:
    from unified_kg_rag.adapters.ingestion.parser import ParserFactory
    from unified_kg_rag.adapters.provider_registry import (
        ProviderRole,
        register_provider,
    )
    from unified_kg_rag.adapters.providers import Providers
    from unified_kg_rag.adapters.renderers.base import BaseRenderer, register_renderer
    from unified_kg_rag.adapters.retrieval.base import (
        BaseGraphRAGRetriever,
        BaseSearchStrategy,
    )
    from unified_kg_rag.adapters.retrieval.memory_manager import (
        ConversationKey,
        MemoryManager,
    )
    from unified_kg_rag.application.bootstrap import build_providers
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.application.retrieval.rag_chain import (
        GraphRAGChain,
        ProcessedQuery,
        RAGInput,
        RAGOutput,
        SearchOutput,
    )
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager
    from unified_kg_rag.domain.models import (
        Config,
        ModelRole,
        PipelineConfig,
        PipelineContext,
        PipelineStageStatus,
        PipelineStageType,
        RetrieverRole,
        SearchResult,
        SearchStrategy,
        SearchType,
    )
    from unified_kg_rag.domain.retrieval.strategy_registry import (
        QueryInput,
        register_strategy,
    )
    from unified_kg_rag.evaluation.evaluation_manager import EvaluationManager
    from unified_kg_rag.ports import (
        ContentGuardPort,
        DocStatusPort,
        EmbeddingFactoryPort,
        GraphIndexer,
        GuardAction,
        GuardSource,
        GuardVerdict,
        LLMFactoryPort,
        ModelFactoryPort,
        RerankFactoryPort,
        TokenCounterPort,
        VectorIndexer,
    )
    from unified_kg_rag.shared.config import get_config
    from unified_kg_rag.shared.logging import setup_logging
    from unified_kg_rag.shared.utils.event_loop import configure_event_loop


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'unified_kg_rag' has no attribute '{name}'")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value  # later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
