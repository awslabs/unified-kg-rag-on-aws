# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What a run of an unchanged corpus costs, on the embedded stores.

An incremental run that finds nothing to index asks no model, embeds
nothing and writes nothing but its lease; a reset of the unchanged corpus
asks the boundary model nothing (the stored answers are read, so a model
that answers otherwise each time cuts the same chunks) and embeds no text
unit, entity or relationship again (their stored vectors are reused); an
edit embeds only what changed. The models are the Ollama adapters against a
fake server, the stores SQLite files (no AWS, no container).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import httpx
import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore
from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.ingestion.cache_keys import chunk_boundary_fingerprint
from unified_kg_rag.domain.ingestion.contributions import (
    CHUNKS_RECORD_KIND,
    attribute_record_keys,
)
from unified_kg_rag.domain.models import (
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.domain.models.config import ChunkingStrategy

pytestmark = pytest.mark.integration

no_aws = e2e.no_aws

_DOCUMENT = "\n".join(
    f"{name} line {i}: {name} works with Depot on gadget storage and delivery."
    for i, name in enumerate(("Vendor", "Buyer", "Carrier", "Vendor", "Buyer", "Depot"))
)


class _Server(FakeOllama):
    """The e2e fake model; boundary answers that change on every call, and
    a record of the chat prompts and embedded texts."""

    def __init__(self) -> None:
        super().__init__(self._answer, e2e.ANSWER, e2e.DIM)
        self.boundary_calls = 0
        self.embedded: list[str] = []
        self._jitter = itertools.cycle((3, 4))

    def _answer(self, system: str, human: str) -> str:
        if "intelligent semantic chunking" in system:
            self.boundary_calls += 1
            line = next(self._jitter)
            return (
                "<chunk_boundaries><line_number>"
                f"{line}</line_number></chunk_boundaries>"
            )
        return e2e._respond(system, human)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/embed":
            import json

            inputs = json.loads(request.content)["input"]
            self.embedded.extend(inputs if isinstance(inputs, list) else [inputs])
        return super().__call__(request)


_WRITES = {
    SQLiteSearchEngine: (
        "bulk_index",
        "bulk_update",
        "bulk_delete",
        "create_index",
        "delete_indices",
        "delete_alias",
        "update_alias",
        "put_index_meta",
        "put_index_mapping",
    ),
    SQLiteDocStatusStore: (
        "put",
        "put_many",
        "delete",
        "put_extraction_settings",
        "put_annotation_digest",
        "record_fits",
    ),
}


def _chunk_records(config, providers) -> tuple[set[str], set[str]]:
    """The boundary records stored, and those the stored text units name."""
    prefix = f"{chunk_boundary_fingerprint(config)}:"
    stored = {
        hit["_id"]
        for hit in e2e._stored(
            config,
            providers,
            config.indexing.opensearch.extraction_records_index_prefix,
        )
        if hit["_id"].startswith(prefix)
    }
    named = {
        key
        for hit in e2e._stored(
            config, providers, config.indexing.opensearch.text_units_index_prefix
        )
        for kind, key in attribute_record_keys(hit["_source"].get("attributes")).items()
        if kind == CHUNKS_RECORD_KIND
    }
    return stored, named


def test_unchanged_runs_ask_embed_and_write_nothing_again(
    tmp_path: Path, mocker, no_aws
) -> None:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    chunking = config.processing.chunking
    chunking.chunker_type = ChunkingStrategy.INTELLIGENT
    chunking.pre_chunk_size = 4000
    chunking.min_chunk_size = 20
    chunking.max_chunk_size = 2000
    server = _Server()
    providers = e2e._providers(config, server, e2e.FakeCrossEncoder("rerank"))
    source = tmp_path / "corpus"
    source.mkdir()
    for name in ("a", "b", "c"):
        (source / f"{name}.txt").write_text(_DOCUMENT.replace("Depot", name.upper()))

    def run(name: str) -> PipelineContext:
        context = e2e._ingest(config, providers, source, tmp_path / name)
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        return context

    run("run-1")
    assert server.boundary_calls == 3
    full = len(server.embedded)
    stored_units = e2e._stored_texts(config, providers)

    # Nothing changed: no model call, no embedding, no write.
    chats, server.embedded = len(server.chat_models), []
    writes = [
        mocker.spy(store, method)
        for store, methods in _WRITES.items()
        for method in methods
    ]
    lookups = mocker.spy(SQLiteSearchEngine, "get_index_name_by_alias")
    run("run-2")
    assert len(server.chat_models) == chats
    assert server.embedded == []
    assert {spy._spy_name: spy.call_count for spy in writes if spy.call_count} == {}
    # One lookup per alias for the run (the four vector indices and the
    # extraction records).
    assert lookups.call_count <= 6
    mocker.stopall()

    # One document edited: only what it changed is embedded again.
    (source / "a.txt").write_text(_DOCUMENT.replace("Depot", "A") + "\nA is new.")
    run("run-3")
    assert 0 < len(server.embedded) < full / 2
    assert server.boundary_calls == 4
    # The edited pre-chunk's old answer went with the text units it cut.
    stored, named = _chunk_records(config, providers)
    assert len(named) == 3 and stored == named
    (source / "a.txt").write_text(_DOCUMENT.replace("Depot", "A"))
    run("run-4")
    stored_units = e2e._stored_texts(config, providers)

    # A reset of the unchanged corpus: the boundary model is not asked (it
    # would answer otherwise), and the text units, entities and relationships
    # keep their stored vectors.
    calls, server.embedded = server.boundary_calls, []
    config.indexing.reset = True
    context = run("run-5")
    assert server.boundary_calls == calls
    # The stage's usage counts the stored answers as cache hits.
    (chunking,) = [
        r
        for r in context.stage_results
        if r.stage_name == PipelineStageType.TEXT_CHUNKING.value
    ]
    assert chunking.metrics["llm_usage"]["cache_hits"] == 3
    assert chunking.metrics["llm_usage"]["llm_calls"] == 0
    assert e2e._stored_texts(config, providers) == stored_units
    # The rebuild keeps the answers its text units name, and only those.
    stored, named = _chunk_records(config, providers)
    assert len(named) == 3 and stored == named
    report_texts = {
        hit["_source"][field]
        for hit in e2e._stored(
            config, providers, config.indexing.opensearch.community_reports_index_prefix
        )
        for field in ("name", "summary", "full_content")
    }
    assert set(server.embedded) <= report_texts


@pytest.mark.parametrize("force_rebuild", [False, True])
def test_force_rebuild_asks_and_embeds_everything_again(
    tmp_path: Path, no_aws, force_rebuild: bool
) -> None:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.processing.chunking.chunker_type = ChunkingStrategy.INTELLIGENT
    providers = e2e._providers(config, _Server(), e2e.FakeCrossEncoder("rerank"))
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=tmp_path / "cache",
            force_rebuild=force_rebuild,
        ),
        source_directory=tmp_path,
        providers=providers,
    )
    try:
        stages = {stage.name: stage for stage in pipeline.stages}
        chunking = stages[PipelineStageType.TEXT_CHUNKING.value]
        indexing = stages[PipelineStageType.INDEXING.value]
        assert isinstance(chunking, ps.TextChunkingStage)
        assert isinstance(indexing, ps.IndexingStage)
        assert chunking.store is not None
        assert chunking.store._reuse is not force_rebuild
        vector = indexing.indexing_manager.opensearch_indexer
        assert vector._reuse_embeddings is not force_rebuild
    finally:
        pipeline.close()
