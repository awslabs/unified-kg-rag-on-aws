# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``backend: local`` end to end: ingest a corpus, then answer a query.

The real local adapters run against a fake Ollama server (an
``httpx.MockTransport`` that answers each prompt like a model would), with the
SQLite doc-status registry from config, a fake cross-encoder and the
in-memory graph/vector stores. Every boto3 client, Bedrock factory and
DynamoDB store constructor is patched to fail, so the run proves local mode
needs no AWS at all. No network, no model download.
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

import boto3
import httpx
import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from tests.integration.test_ingestion_stages import _respond
from unified_kg_rag.adapters.aws import bedrock as bedrock_module
from unified_kg_rag.adapters.aws import token_counter as token_counter_module
from unified_kg_rag.adapters.doc_status import dynamodb as dynamodb_module
from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore
from unified_kg_rag.adapters.local.embeddings import OllamaEmbeddingFactory
from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
from unified_kg_rag.adapters.local.rerank import CrossEncoderRerankFactory
from unified_kg_rag.application.bootstrap import build_providers
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    PipelineConfig,
    PipelineStageStatus,
    PipelineStageType,
    RetrievalResult,
    RetrieverRole,
    SearchQuery,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import ChunkingStrategy

pytestmark = pytest.mark.integration

DIM = 32
ANSWER = "Vendor supplies widgets to Buyer, and Depot stores them."


class FakeCrossEncoder:
    def __init__(self, model_id: str, **_: Any) -> None:
        self.model_id = model_id
        self.calls = 0

    def predict(self, pairs: list[tuple[str, str]], **_: Any) -> list[float]:
        self.calls += 1
        return [float("Depot" in passage) for _, passage in pairs]


class StoreRetriever:
    """DOCUMENT retriever over the in-memory vector store's text units,
    ranked by cosine similarity of the local embedding model's vectors."""

    def __init__(self, store: FakeVectorStore, embeddings: Any) -> None:
        self.store = store
        self.embeddings = embeddings

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        units = list(self.store.data.get("text_units", {}).values())
        query_vector = await self.embeddings.aembed_query(query.query)
        unit_vectors = await self.embeddings.aembed_documents([u.text for u in units])
        scored = sorted(
            zip(units, (_cosine(query_vector, v) for v in unit_vectors), strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return [
            RetrievalResult(
                content=unit.text,
                score=score,
                source=unit.id,
                retriever_type="text",
                chunk_id=unit.id,
                metadata={"id": unit.id},
            )
            for unit, score in scored[: query.top_k]
        ]

    def close(self) -> None:
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


@pytest.fixture
def no_aws(mocker) -> None:
    def forbid(target: Any, name: str) -> None:
        mocker.patch.object(
            target, name, side_effect=AssertionError(f"{target}.{name} called")
        )

    forbid(bedrock_module.BaseBedrockModelFactory, "__init__")
    forbid(token_counter_module.BedrockTokenCounter, "__init__")
    forbid(dynamodb_module.DynamoDBDocStatusStore, "__init__")
    forbid(boto3.Session, "client")


def _config(tmp_path: Path) -> Config:
    config = Config(
        backend="local",
        models={
            "default": "synthetic-llm",
            "fast": "synthetic-fast",
            "embedding": {"model_id": "synthetic-embed", "dimension": DIM},
            "rerank": "synthetic-reranker",
        },
        providers={
            "doc_status": {
                "enabled": True,
                "sqlite_path": str(tmp_path / "registry" / "doc-status.sqlite3"),
            },
        },
    )
    chunking = config.processing.chunking
    chunking.chunker_type = ChunkingStrategy.SIMPLE
    chunking.fallback_chunk_size = 60
    chunking.chunk_overlap = 0
    chunking.min_chunk_size = 1
    config.processing.translation.enabled = False
    config.processing.gleaning.enabled = False
    config.processing.claim_extraction.enabled = False
    config.processing.max_attempts = 1
    config.fixing.enabled = False
    return config


async def test_local_backend_ingests_and_answers_without_aws(
    tmp_path: Path, no_aws
) -> None:
    config = _config(tmp_path)
    server = FakeOllama(_respond, ANSWER, DIM)
    transport = httpx.MockTransport(server)
    graph, vectors = FakeGraphStore(), FakeVectorStore(config.indexing.opensearch)
    encoder = FakeCrossEncoder("synthetic-reranker")
    providers = build_providers(
        config,
        # The real local adapters, pointed at the fake server.
        llm_factory=OllamaLanguageModelFactory(config, transport=transport),
        embedding_factory=OllamaEmbeddingFactory(config, transport=transport),
        rerank_factory=CrossEncoderRerankFactory(
            config, loader=lambda *_, **__: encoder
        ),
        graph_indexer_factory=lambda: graph,
        vector_indexer_factory=lambda: vectors,
        retriever_builders={
            RetrieverRole.DOCUMENT: lambda: StoreRetriever(
                vectors,
                providers.embedding_factory.get_model(config.embedding_model_id),
            )
        },
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")

    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=tmp_path / "cache",
        ),
        source_directory=source,
        providers=providers,
    )
    for stage in pipeline.stages:
        if isinstance(stage, ps.GraphResolutionStage):
            stage.resolver.entity_resolver.use_process_pool = False
            stage.resolver.relationship_resolver.use_process_pool = False
    context = pipeline.run(source, pipeline_id="local-e2e")
    pipeline.close()

    assert context.status is PipelineStageStatus.COMPLETED
    assert len(graph.ids("entities")) == 3
    assert vectors.ids("text_units") == {u.id for u in context.text_units}
    # Extraction ran on the configured tier models through the Ollama API.
    assert set(server.chat_models) <= {"synthetic-llm", "synthetic-fast"}
    assert "synthetic-llm" in server.chat_models
    # The incremental registry is the SQLite file from config.
    # A fresh store on the configured file sees what the run committed.
    registry = providers.build_doc_status()
    assert isinstance(registry, SQLiteDocStatusStore)
    records = registry.list_all()
    assert len(records) == 2 and {r.status for r in records} == {DocStatus.PROCESSED}
    assert await asyncio.to_thread(Path(config.providers.doc_status.sqlite_path).exists)

    chain = GraphRAGChain(config, providers=providers)
    try:
        out = await chain.ainvoke(
            RAGInput(
                query="Who stores the widgets?",
                search_strategy=SearchStrategy.SIMPLE,
                enable_query_processing=False,
            )
        )
    finally:
        await chain.aclose()

    assert isinstance(out, RAGOutput)
    assert out.answer == ANSWER
    assert server.answers == 1
    assert out.sources
    # The query embedding and the reranker went through the local adapters.
    assert server.embed_models and set(server.embed_models) == {"synthetic-embed"}
    assert encoder.calls >= 1
    assert "Depot" in out.search_results.results[0].content

    # A second run over the unchanged corpus has nothing to (re)index.
    rerun = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=tmp_path / "cache-2",
        ),
        source_directory=source,
        providers=providers,
    )
    calls_before = len(server.chat_models)
    rerun_context = rerun.run(source, pipeline_id="local-e2e-2")
    rerun.close()
    assert rerun_context.status is PipelineStageStatus.COMPLETED
    assert len(server.chat_models) == calls_before
