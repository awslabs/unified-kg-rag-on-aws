# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ports layer — the abstract interfaces the domain depends on (hexagonal).

A *port* is an interface owned by the domain; *adapters* (under
``unified_kg_rag.adapters``) are the concrete technology bindings. Domain and
application code import ports from here and never import a concrete backend.

Catalog:
- ``DocStatusPort`` — document-status registry for incremental indexing
  (adapter: ``adapters.aws.dynamodb``; in-memory fake in tests).
- ``BaseIndexer`` / ``GraphIndexer`` / ``VectorIndexer`` — write-side store
  contracts (adapters: ``adapters.storage``). ``IndexingStats`` is the shared
  result record.
- ``ModelFactoryPort`` (+ ``LLMFactoryPort`` / ``EmbeddingFactoryPort`` /
  ``RerankFactoryPort`` aliases) — the LLM/Embedding/Rerank provider boundary
  (``Protocol``; the Bedrock factories in ``adapters.aws.bedrock`` conform
  structurally). Annotate against these for provider-agnostic call sites.
- ``CachePort`` — the stage-result persistence boundary (``Protocol``; the
  filesystem ``CacheManager`` in ``shared.cache_manager`` conforms structurally).

Two further contracts are abstract *adapter bases* rather than pure ports —
they construct infrastructure in ``__init__`` (HybridScorer/TokenManager) — so
they are NOT exported from this catalog; they live beside their adapters and are
imported from there directly. Listed here only as a pointer:
- ``BaseGraphRAGRetriever`` / ``BaseSearchStrategy``
  (``adapters.retrieval.base``; adapters: ``adapters.retrievers`` / ``adapters.search_strategies``).
- ``BaseGraphRAGEvaluator`` (``evaluation.base``; adapters: ``adapters.evaluators``).
"""

from unified_kg_rag.ports.cache import CachePort
from unified_kg_rag.ports.doc_status import DocStatusPort
from unified_kg_rag.ports.indexer import (
    BaseIndexer,
    GraphIndexer,
    IndexingStats,
    VectorIndexer,
)
from unified_kg_rag.ports.model_factory import (
    EmbeddingFactoryPort,
    LLMFactoryPort,
    ModelFactoryPort,
    RerankFactoryPort,
)

__all__ = [
    "BaseIndexer",
    "CachePort",
    "DocStatusPort",
    "EmbeddingFactoryPort",
    "GraphIndexer",
    "IndexingStats",
    "LLMFactoryPort",
    "ModelFactoryPort",
    "RerankFactoryPort",
    "VectorIndexer",
]
