# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ports layer — the interfaces the application wires and the adapters
implement (hexagonal).

A *port* is written in terms of the domain models; *adapters* (under
``unified_kg_rag.adapters``) are the concrete technology bindings. The domain
imports no port (code that reads a store declares the subset it needs as its
own ``Protocol``), and the application imports ports from here and never a
concrete backend (only the composition root and the CLIs do).

Catalog:
- ``DocStatusPort`` — document-status registry for incremental indexing
  (adapters: ``adapters.doc_status.dynamodb``, ``adapters.doc_status.sqlite``;
  in-memory fake in tests).
- ``BaseIndexer`` / ``GraphIndexer`` / ``VectorIndexer`` — write-side store
  contracts (adapters: ``adapters.storage``). ``IndexingStats`` is the shared
  result record. A store's other concerns are ``Protocol``s with "cannot"
  defaults, each its own port so a caller takes only the one it uses:
  ``SchemaAnnotationStore`` (both indexers), and, for the vector store,
  ``ExtractionRecordStore``, ``SummaryTreeStore``, ``BuildRecordStore`` and
  ``StoreInspection``. ``VectorIndexer`` inherits all five, so the composition
  root builds one vector store that implements every one of them.
- ``ModelFactoryPort`` (+ ``LLMFactoryPort`` / ``EmbeddingFactoryPort`` /
  ``RerankFactoryPort`` aliases) — the LLM/Embedding/Rerank provider boundary
  (``Protocol``; the Bedrock factories in ``adapters.aws.bedrock`` and the
  local ones in ``adapters.local`` conform structurally). Annotate against
  these for provider-agnostic call sites.
- ``ContentGuardPort`` (+ ``GuardVerdict`` / ``GuardAction`` /
  ``GuardSource``) — screening of the raw user query and the final answer
  (``Protocol``; adapter: ``adapters.guardrail.bedrock.BedrockGuardrail``).
- ``TokenCounterPort`` — token counting/truncation for one model (``Protocol``;
  ``adapters.aws.token_counter.BedrockTokenCounter`` and
  ``adapters.local.token_counter.LocalTokenCounter`` conform structurally).
- ``GraphQueryPort`` — structured queries over one suffix's graph, for the
  ``structured`` strategy (``Protocol``; adapters: the graph retrievers in
  ``adapters.retrievers``).

Two further contracts are abstract *adapter bases* rather than pure ports —
they construct infrastructure in ``__init__`` (HybridScorer/TokenManager) — so
they are NOT exported from this catalog; they live beside their adapters and are
imported from there directly. Listed here only as a pointer:
- ``BaseGraphRAGRetriever`` / ``BaseSearchStrategy``
  (``adapters.retrieval.base``; adapters: ``adapters.retrievers`` / ``adapters.search_strategies``).
- ``BaseGraphRAGEvaluator`` (``adapters.evaluators.base``).
"""

from unified_kg_rag.ports.doc_status import DocStatusPort
from unified_kg_rag.ports.graph_query import GraphQueryPort
from unified_kg_rag.ports.guardrail import (
    ContentGuardPort,
    GuardAction,
    GuardSource,
    GuardVerdict,
)
from unified_kg_rag.ports.indexer import (
    BaseIndexer,
    BuildRecordStore,
    ExtractionRecordStore,
    GraphIndexer,
    IndexingStats,
    SchemaAnnotationStore,
    StoreInspection,
    SummaryTreeStore,
    VectorIndexer,
)
from unified_kg_rag.ports.model_factory import (
    EmbeddingFactoryPort,
    LLMFactoryPort,
    ModelFactoryPort,
    RerankFactoryPort,
    TokenCounterPort,
)

__all__ = [
    "BaseIndexer",
    "BuildRecordStore",
    "ContentGuardPort",
    "DocStatusPort",
    "EmbeddingFactoryPort",
    "ExtractionRecordStore",
    "GraphIndexer",
    "GraphQueryPort",
    "GuardAction",
    "GuardSource",
    "GuardVerdict",
    "IndexingStats",
    "LLMFactoryPort",
    "ModelFactoryPort",
    "RerankFactoryPort",
    "SchemaAnnotationStore",
    "StoreInspection",
    "SummaryTreeStore",
    "TokenCounterPort",
    "VectorIndexer",
]
