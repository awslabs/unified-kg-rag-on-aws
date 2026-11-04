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


def test_an_incremental_commit_keeps_no_vector_it_replaces(
    tmp_path: Path, mocker, no_aws
) -> None:
    # The removal of an edited document rewrites what it shared, and the
    # commit writes it back, taking the vectors the removal replaced. The
    # commit is the run's last write: it reads only the vectors it takes.
    from unified_kg_rag.adapters.storage import opensearch_indexer as osi

    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    server = _Server()
    providers = e2e._providers(config, server, e2e.FakeCrossEncoder("rerank"))
    source = tmp_path / "corpus"
    source.mkdir()
    for name in ("a", "b", "c"):
        (source / f"{name}.txt").write_text(_DOCUMENT.replace("Depot", name.upper()))
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)

    keeping = mocker.spy(osi.OpenSearchIndexer, "set_replaced_vector_keeping")
    upserts = mocker.spy(osi.OpenSearchIndexer, "_reusable_vectors")
    (source / "a.txt").write_text(_DOCUMENT.replace("Depot", "A") + "\nA is new.")
    (source / "d.txt").write_text("Vendor ships gadgets to Buyer by sea.")
    context = e2e._ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)

    assert [call.args[1] for call in keeping.call_args_list] == [False, True]
    # The removal's writes (if any) kept the vectors they replaced; the
    # commit's, the last ones, kept none.
    kept = [call.args[5] for call in upserts.call_args_list]
    assert False in kept and kept == sorted(kept, reverse=True)


class _CountingServer(_Server):
    """``_Server``, counting the summary and community report prompts."""

    def __init__(self) -> None:
        super().__init__()
        self.summaries = 0
        self.reports = 0

    def _answer(self, system: str, human: str) -> str:
        if "knowledge-graph editor" in system:
            self.summaries += 1
        elif "community analysis" in system:
            self.reports += 1
        return super()._answer(system, human)


def _snapshot(config, providers, graph_path: Path) -> dict:
    """What the stores hold: every vector document by index and id, and
    every graph vertex and edge (without the rebuild's tag)."""
    import json
    import sqlite3

    os_config = config.indexing.opensearch
    stored: dict = {
        prefix: {
            hit["_id"]: hit["_source"] for hit in e2e._stored(config, providers, prefix)
        }
        for prefix in (
            os_config.text_units_index_prefix,
            os_config.entities_index_prefix,
            os_config.relationships_index_prefix,
            os_config.community_reports_index_prefix,
            os_config.extraction_records_index_prefix,
        )
    }
    connection = sqlite3.connect(graph_path)
    try:
        for table in ("vertices", "edges"):
            for row in connection.execute(f"SELECT * FROM {table}"):
                props = json.loads(row[-1])
                props.pop("build_tag", None)
                stored[(table, *row[:-1])] = props
    finally:
        connection.close()
    return stored


def test_a_reset_of_an_unchanged_corpus_asks_no_summary_or_report(
    tmp_path: Path, no_aws
) -> None:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    summarization = config.processing.graph_extraction.description_summarization
    # Every merged description is summarized.
    summarization.force_summary_threshold_tokens = 3
    server = _CountingServer()
    providers = e2e._providers(config, server, e2e.FakeCrossEncoder("rerank"))
    source = tmp_path / "corpus"
    source.mkdir()
    for name in ("a", "b", "c"):
        (source / f"{name}.txt").write_text(f"{_DOCUMENT}\n{name.upper()} closes.")
    graph_path = tmp_path / "stores" / "graph.sqlite3"

    def run(name: str) -> PipelineContext:
        context = e2e._ingest(config, providers, source, tmp_path / name)
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        return context

    def metrics(context: PipelineContext, stage: PipelineStageType) -> dict:
        (result,) = [r for r in context.stage_results if r.stage_name == stage.value]
        return result.metrics

    run("run-1")
    summaries, reports = server.summaries, server.reports
    assert summaries > 0 and reports > 0
    before = _snapshot(config, providers, graph_path)

    # The first build's summaries and report answers are reused: no model
    # call, the stores as they were (report times and vectors included).
    config.indexing.reset = True
    context = run("run-2")
    assert (server.summaries, server.reports) == (summaries, reports)
    assert _snapshot(config, providers, graph_path) == before
    resolution = metrics(context, PipelineStageType.GRAPH_RESOLUTION)
    assert resolution["summaries_reused"] > 0
    assert resolution["llm_usage"]["llm_calls"] == 0
    detection = metrics(context, PipelineStageType.COMMUNITY_DETECTION)
    assert detection["llm_usage"]["cache_hits"] == detection["reports_generated"]
    assert detection["llm_usage"]["llm_calls"] == 0

    # Each stored report names its answer, and the rebuild keeps the
    # answers its reports name.
    os_config = config.indexing.opensearch
    named = {
        hit["_source"]["attributes"]["report_record"]
        for hit in e2e._stored(
            config, providers, os_config.community_reports_index_prefix
        )
    }
    records = set(before[os_config.extraction_records_index_prefix])
    assert len(named) == detection["reports_generated"] and named <= records

    # A summarization setting changed: the reset summarizes again.
    summarization.max_summary_tokens = 64
    run("run-3")
    assert server.summaries == 2 * summaries


def test_a_reset_after_a_disjoint_document_asks_only_its_report(
    tmp_path: Path, no_aws
) -> None:
    """A document whose entities join no stored community adds one community;
    the others keep their names (content-derived, not their rank in the
    sorted partition), so their report keys match and only the new
    community's report is asked for."""
    from tests.integration import test_ingestion_stages as stages

    def group(tag: str) -> tuple[str, ...]:
        return tuple(f"{word}{tag}" for word in ("Alpha", "Bravo", "Charlie"))

    # The added group sorts first, ahead of every stored community.
    groups, added = [group(tag) for tag in "KLMN"], group("A")
    names = tuple(name for g in (*groups, added) for name in g)

    class Server(_CountingServer):
        def _answer(self, system: str, human: str) -> str:
            if "knowledge graph extraction expert" in system:
                return stages._extraction(stages._source_text(human), names)
            return super()._answer(system, human)

    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.graph.community_detection.min_community_size = 2
    server = Server()
    providers = e2e._providers(config, server, e2e.FakeCrossEncoder("rerank"))
    source = tmp_path / "corpus"
    source.mkdir()

    def write(name: str, g: tuple[str, ...]) -> None:
        (source / f"{name}.txt").write_text(f"{g[0]} works with {g[1]} and {g[2]}.")

    def run(name: str) -> dict[str, str]:
        context = e2e._ingest(config, providers, source, tmp_path / name)
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        return {c.id: c.name for c in context.communities}

    for i, g in enumerate(groups):
        write(f"g{i}", g)
    first = run("run-1")
    assert server.reports == len(first) == len(groups)

    write("added", added)
    config.indexing.reset = True
    second = run("run-2")
    assert len(second) == len(first) + 1
    assert {cid: second[cid] for cid in first} == first
    assert server.reports == len(first) + 1


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
        resolution = stages[PipelineStageType.GRAPH_RESOLUTION.value]
        detection = stages[PipelineStageType.COMMUNITY_DETECTION.value]
        assert isinstance(chunking, ps.TextChunkingStage)
        assert isinstance(indexing, ps.IndexingStage)
        assert isinstance(resolution, ps.GraphResolutionStage)
        assert isinstance(detection, ps.CommunityDetectionStage)
        assert chunking.store is not None
        assert chunking.store._reuse is not force_rebuild
        assert resolution.reuse_stored is not force_rebuild
        assert detection.store._reuse is not force_rebuild
        vector = indexing.indexing_manager.opensearch_indexer
        assert vector._reuse_embeddings is not force_rebuild
    finally:
        pipeline.close()
