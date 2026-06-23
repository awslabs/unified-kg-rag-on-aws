# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Ports layer — the abstract interfaces the domain depends on (hexagonal).

A *port* is an interface owned by the domain; *adapters* (under
``aws_graphrag.adapters``) are the concrete technology bindings. Domain and
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

Two further contracts are abstract *adapter bases* rather than pure ports —
they construct infrastructure in ``__init__`` (HybridScorer/TokenManager, tqdm)
— so they live beside their adapters but are re-exported here for discovery:
- ``BaseGraphRAGRetriever`` / ``BaseSearchStrategy``
  (``adapters.retrieval.base``; adapters: ``adapters.retrievers`` / ``adapters.search_strategies``).
- ``BaseGraphRAGEvaluator`` (``evaluation.base``; adapters: ``adapters.evaluators``).
"""

from aws_graphrag.ports.doc_status import DocStatusPort
from aws_graphrag.ports.indexer import (
    BaseIndexer,
    GraphIndexer,
    IndexingStats,
    VectorIndexer,
)
from aws_graphrag.ports.model_factory import (
    EmbeddingFactoryPort,
    LLMFactoryPort,
    ModelFactoryPort,
    RerankFactoryPort,
)

__all__ = [
    "BaseIndexer",
    "DocStatusPort",
    "EmbeddingFactoryPort",
    "GraphIndexer",
    "IndexingStats",
    "LLMFactoryPort",
    "ModelFactoryPort",
    "RerankFactoryPort",
    "VectorIndexer",
]
