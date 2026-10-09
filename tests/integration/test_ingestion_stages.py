# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ingestion stages run end to end over a tiny synthetic corpus (AWS-free).

Chunking, graph extraction, gleaning, resolution, claim extraction and
resolution, graph analysis, community detection and an incremental indexing
commit run through ``PipelineStage.execute`` with a scripted chat model (via
the ``Providers`` seam) and the in-memory stores and registry. One chunk gets
an unparseable answer: the run still succeeds, the failure is reported, and
its document is recorded FAILED so the next run re-extracts it.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.delta_detector import (
    assign_document_identity,
    document_doc_id,
    fingerprint_documents,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    Document,
    DocumentContent,
    DocumentDelta,
    PipelineContext,
    PipelineStageStatus,
)
from unified_kg_rag.domain.models.config import ChunkingStrategy

pytestmark = pytest.mark.integration

_ROOT = Path("/corpus")
_NAMES = ("Vendor", "Buyer", "Depot")
_UNREADABLE = "UNREADABLE"


def _source_text(human: str) -> str:
    match = re.search(r"## SOURCE TEXT:\n(.*?)\n\n##", human, re.DOTALL)
    assert match, "prompt has no SOURCE TEXT section"
    return match.group(1)


def _extraction(text: str) -> str:
    if _UNREADABLE in text:
        return "Sorry, I cannot help with that."
    names = [name for name in _NAMES if name in text]
    entities = "".join(
        f"<entity><name>{n}</name><type>ORGANIZATION</type>"
        f"<description>{n} in the supply chain</description>"
        f"<confidence>9</confidence><source_text>{n}</source_text></entity>"
        for n in names
    )
    relationships = "".join(
        f"<relationship><source>{a}</source><target>{b}</target>"
        f"<type>WORKS_WITH</type><description>{a} works with {b}</description>"
        f"<strength>8</strength><source_text>{text.strip()}</source_text>"
        "</relationship>"
        for a, b in zip(names, names[1:], strict=False)
    )
    return (
        f"<entities>{entities}</entities>"
        f"<relationships>{relationships}</relationships>"
    )


_REFINEMENT = (
    "<refinement_plan><identified_issues></identified_issues></refinement_plan>"
)
_REPORT = (
    "<community_name>Widget supply chain partners</community_name>"
    "<summary>Vendor, Buyer and Depot trade widgets.</summary>"
    "<rating>5.0</rating><rating_explanation>A small trading group.</rating_explanation>"
    "<findings><finding><summary>Vendor is central</summary>"
    "<explanation>Vendor works with Buyer and Depot.</explanation></finding></findings>"
)


def _respond(system: str, human: str) -> str:
    if "knowledge graph extraction expert" in system:
        return _extraction(_source_text(human))
    if "graph refinement specialist" in system:
        return _REFINEMENT
    if "claim extraction specialist" in system:
        return "<claims></claims>"
    if "community analysis" in system:
        return _REPORT
    raise AssertionError(f"unexpected prompt: {system[:80]!r}")


def _config() -> Config:
    config = Config()
    chunking = config.processing.chunking
    chunking.chunker_type = ChunkingStrategy.SIMPLE
    chunking.fallback_chunk_size = 60
    chunking.chunk_overlap = 0
    chunking.min_chunk_size = 1
    config.processing.translation.enabled = False
    config.processing.gleaning.enabled = True
    config.processing.gleaning.max_rounds = 1
    config.processing.claim_extraction.enabled = True
    config.processing.max_attempts = 1
    config.fixing.enabled = False
    config.graph.visualization.enabled = False
    return config


def _document(name: str, text: str) -> Document:
    document = Document(
        page_content=text,
        content=DocumentContent(text=text),
        document_id=name,
        file_name=name,
        file_path=str(_ROOT / name),
        file_type="txt",
        total_pages=1,
    )
    assign_document_identity(document, _ROOT)
    return document


def _run(stage: ps.PipelineStage, context: PipelineContext) -> dict:
    result = stage.execute(context)
    assert result.status is PipelineStageStatus.COMPLETED, result.error_message
    return result.metrics


def test_stages_run_end_to_end_and_report_a_failed_extraction() -> None:
    config = _config()
    providers = Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=ScriptedLLMFactory(_respond),
        embedding_factory=HashingEmbeddingFactory(),
    )
    contract = _document("contract.txt", "Vendor supplies widgets to Buyer.")
    logistics = _document(
        "logistics.txt",
        "Depot stores widgets for Vendor.\n\n"
        f"{_UNREADABLE} paragraph the model cannot answer.",
    )
    documents = [contract, logistics]
    store = FakeDocStatusStore()
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=str(_ROOT),
        documents=documents,
    )
    context.incremental_fingerprints = fingerprint_documents(documents)
    context.incremental_delta = DocumentDelta(
        new=sorted(context.incremental_fingerprints)
    )

    _run(ps.TextChunkingStage(config, providers=providers), context)
    assert len(context.text_units) == 3

    metrics = _run(ps.GraphExtractionStage(config, providers=providers), context)
    assert {e.name for e in context.entities} == set(_NAMES)
    assert len(context.relationships) == 2
    assert metrics["failed_units"] == 1
    failed_unit = next(u for u in context.text_units if _UNREADABLE in u.text)
    assert context.failed_text_unit_ids["graph_extraction"] == [failed_unit.id]

    # Stages that prepare inputs in a process pool use threads here: forking
    # the multi-threaded test process can deadlock the child.
    gleaning = ps.GleaningStage(config, providers=providers)
    gleaning.gleaner.use_process_pool = False
    metrics = _run(gleaning, context)
    assert metrics["failed_units"] == 0
    # The empty answer adds nothing, so every unit is gleaned exactly once.
    assert metrics["improvement_rate"] == 0.0
    assert metrics["refinement_calls"] == metrics["text_units_processed"]
    assert len(context.entities) == 3 and len(context.relationships) == 2

    resolution = ps.GraphResolutionStage(config, providers=providers)
    resolution.resolver.entity_resolver.use_process_pool = False
    resolution.resolver.relationship_resolver.use_process_pool = False
    _run(resolution, context)
    assert len(context.resolved_entities) == 3
    assert len(context.resolved_relationships) == 2

    claim_extraction = ps.ClaimExtractionStage(config, providers=providers)
    claim_extraction.extractor.use_process_pool = False
    metrics = _run(claim_extraction, context)
    # An empty <claims/> answer is a valid result, not a failure.
    assert context.claims == [] and metrics["failed_units"] == 0
    _run(ps.ClaimResolutionStage(config), context)

    metrics = _run(ps.GraphAnalysisStage(config, providers=providers), context)
    assert metrics["graph_metrics"]["num_nodes"] == 3
    assert metrics["graph_metrics"]["num_edges"] == 2

    _run(ps.CommunityDetectionStage(config, providers=providers), context)
    assert len(context.communities) == 1
    assert len(context.community_reports) == 1

    indexing = ps.IndexingStage(config, doc_status=store, providers=providers)
    graph, vectors = FakeGraphStore(), FakeVectorStore(config.indexing.opensearch)
    indexing.indexing_manager = IndexingManager(
        config, vector_indexer=vectors, graph_indexer=graph, providers=providers
    )
    metrics = _run(indexing, context)
    assert metrics["total_failed"] == 0
    assert vectors.ids("text_units") == {u.id for u in context.text_units}
    assert len(graph.ids("entities")) == 3

    # The document with the failed chunk is FAILED, so the next diff re-extracts
    # it; the fully extracted one is PROCESSED and unchanged.
    contract_record = store.get(document_doc_id(contract))
    logistics_record = store.get(document_doc_id(logistics))
    assert contract_record is not None and logistics_record is not None
    assert contract_record.status is DocStatus.PROCESSED
    assert logistics_record.status is DocStatus.FAILED
    delta = store.diff(context.incremental_fingerprints)
    assert delta.unchanged == [document_doc_id(contract)]
    assert delta.changed == [document_doc_id(logistics)]


def test_whole_pipeline_runs_without_aws_through_the_public_constructor(
    tmp_path: Path,
) -> None:
    # The documented "run without AWS" path: inject the providers, the
    # doc-status registry and both indexers into DataIngestionPipeline; no
    # stage builds a DynamoDB, OpenSearch or Neptune client.
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.domain.models import PipelineConfig, PipelineStageType

    config = _config()
    config.processing.gleaning.enabled = False
    config.processing.claim_extraction.enabled = False
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    providers = Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=ScriptedLLMFactory(_respond),
        embedding_factory=HashingEmbeddingFactory(),
    )
    store = FakeDocStatusStore()
    graph, vectors = FakeGraphStore(), FakeVectorStore(config.indexing.opensearch)

    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=tmp_path / "cache",
        ),
        source_directory=source,
        providers=providers,
        doc_status=store,
        vector_indexer=vectors,
        graph_indexer=graph,
    )
    for stage in pipeline.stages:
        if isinstance(stage, ps.GraphResolutionStage):
            stage.resolver.entity_resolver.use_process_pool = False
            stage.resolver.relationship_resolver.use_process_pool = False
    context = pipeline.run(source, pipeline_id="no-aws")

    assert context.status is PipelineStageStatus.COMPLETED
    assert vectors.ids("text_units") == {u.id for u in context.text_units}
    assert graph.ids("entities") and len(graph.ids("entities")) == 3
    # Passing a registry turns on incremental indexing: both documents are
    # recorded, so an unchanged re-run would skip them.
    records = store.list_all()
    assert len(records) == 2
    assert {r.status for r in records} == {DocStatus.PROCESSED}
