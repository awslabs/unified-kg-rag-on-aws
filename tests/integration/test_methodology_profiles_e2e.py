# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Methodology profiles end to end on the embedded stores (no AWS, no
container; docs/design.md#46-methodology-profiles).

A LightRAG-only ingestion runs no community detection and makes no report
call; its strategies answer and GraphRAG's are refused before any model
call. Adding graphrag later rebuilds the namespace from the stored
extractions: the only new model calls are the community reports. Removing it
stops maintaining the reports without deleting them; a reset with the
remaining methodology replaces them.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - fixture
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.retrieval.rag_chain import (
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    Methodology,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
    SearchStrategy,
)
from unified_kg_rag.shared import ConfigurationError, MethodologyNotIndexedError

pytestmark = pytest.mark.integration

GRAPHRAG = (
    SearchStrategy.LOCAL,
    SearchStrategy.GLOBAL,
    SearchStrategy.DRIFT,
    SearchStrategy.SIMPLE,
)
# iterative over its default base, mix, reads lightrag.
LIGHTRAG = (
    SearchStrategy.MIX,
    SearchStrategy.HYBRID,
    SearchStrategy.NAIVE,
    SearchStrategy.ITERATIVE,
)


class _Calls:
    """The fake model's answers (the embedded E2E's), counted by kind."""

    def __init__(self) -> None:
        self.kinds: Counter[str] = Counter()

    def __call__(self, system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system:
            self.kinds["extraction"] += 1
        elif "community analysis" in system:
            self.kinds["report"] += 1
        else:
            self.kinds["other"] += 1
        return e2e._respond(system, human)

    def since(self, before: Counter[str]) -> Counter[str]:
        return self.kinds - before


def _corpus(root: Path) -> Path:
    source = root / "corpus"
    source.mkdir(parents=True)
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    (source / "freight.txt").write_text("Carrier moves gadgets for Depot.")
    return source


def _setup(
    tmp_path: Path, *, registry: bool = True
) -> tuple[Config, Providers, FakeOllama, _Calls]:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.providers.doc_status.enabled = registry
    calls = _Calls()
    server = FakeOllama(calls, e2e.ANSWER, e2e.DIM)
    encoder = e2e.FakeCrossEncoder("synthetic-reranker")
    return config, e2e._providers(config, server, encoder), server, calls


def _ran(context: PipelineContext) -> set[str]:
    return {
        r.stage_name
        for r in context.stage_results
        if r.status is PipelineStageStatus.COMPLETED
    }


def _record(providers: Providers) -> dict:
    indexer = providers.build_vector_indexer()
    try:
        record = indexer.read_build_record(None)
    finally:
        indexer.close()
    assert record is not None
    return record


def _reports(config: Config, providers: Providers) -> list[dict]:
    prefix = config.indexing.opensearch.community_reports_index_prefix
    try:
        return e2e._stored(config, providers, prefix)
    except Exception:  # noqa: BLE001 - no such index
        return []


def _community_vertices(config: Config) -> int:
    """The community vertices the SQLite graph holds."""
    prefix = config.indexing.neptune.community_label_prefix
    with sqlite3.connect(config.providers.graph.sqlite_path) as connection:
        return connection.execute(
            "SELECT COUNT(*) FROM vertices WHERE label LIKE ?", (f"{prefix}%",)
        ).fetchone()[0]


def _ask(chain: GraphRAGChain, strategy: SearchStrategy) -> RAGOutput:
    out = chain.invoke(
        RAGInput(query="Who stores the gadgets for Vendor?", search_strategy=strategy)
    )
    assert isinstance(out, RAGOutput)
    return out


def test_a_lightrag_only_namespace_builds_less_and_gains_graphrag_later(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    config, providers, server, calls = _setup(tmp_path)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    source = _corpus(tmp_path)

    # LightRAG only: no community detection, no report call.
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert "community_detection" not in _ran(context)
    light_calls = calls.since(Counter())
    assert light_calls["extraction"] > 0 and light_calls["report"] == 0
    record = _record(providers)
    assert record["methodologies"] == ["lightrag", "hipporag"]
    assert "community_reports" not in record["artifacts"]

    # The same corpus with the default methodologies in another namespace
    # pays the reports on top: the LightRAG-only build cost less.
    full_config, full_providers, _, full_calls = _setup(tmp_path / "full")
    full_source = _corpus(tmp_path / "full")
    context = e2e._ingest(full_config, full_providers, full_source, tmp_path / "f")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert full_calls.kinds["report"] > 0
    assert full_calls.kinds["extraction"] == light_calls["extraction"]
    assert sum(full_calls.kinds.values()) > sum(light_calls.values())

    # Its strategies answer; GraphRAG's are refused before any model call.
    chain = GraphRAGChain(config, providers=providers)
    try:
        for strategy in LIGHTRAG:
            out = _ask(chain, strategy)
            assert out.answer == e2e.ANSWER, strategy
            assert out.sources, strategy
        for strategy in GRAPHRAG:
            before = len(server.chat_models)
            with pytest.raises(MethodologyNotIndexedError) as refused:
                _ask(chain, strategy)
            assert len(server.chat_models) == before, strategy
            message = str(refused.value)
            assert "reads the graphrag methodology" in message
            assert "holds lightrag, hipporag" in message
            assert "mix, hybrid, naive" in message
            assert "indexing.methodologies" in message
        # auto has one strategy left of local, mix and global: no router call.
        out = _ask(chain, SearchStrategy.AUTO)
        assert out.metadata["search_strategy"] == "mix"
        # iterative over local reads graphrag: refused there.
        over_local = config.model_copy(deep=True)
        over_local.search.iterative_search.base = "local"
        local_chain = GraphRAGChain(over_local, providers=providers)
        try:
            before = len(server.chat_models)
            with pytest.raises(MethodologyNotIndexedError, match="'iterative'"):
                _ask(local_chain, SearchStrategy.ITERATIVE)
            assert len(server.chat_models) == before
        finally:
            local_chain.close()

        # Adding graphrag: the run rebuilds the namespace from the stored
        # extractions; the community reports are its only new model calls.
        config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
        before = Counter(calls.kinds)
        context = e2e._ingest(config, providers, source, tmp_path / "run-2")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        assert "community_detection" in _ran(context)
        added = calls.since(before)
        assert added["report"] > 0
        assert added["extraction"] == 0
        assert set(added) == {"report"}
        assert config.indexing.reset is False  # set for that run only
        assert _record(providers)["methodologies"] == [
            "graphrag",
            "lightrag",
            "hipporag",
        ]
        assert _reports(config, providers)
        assert _community_vertices(config) > 0
        # The chain reads the record again before refusing: global answers.
        for strategy in (SearchStrategy.GLOBAL, SearchStrategy.LOCAL):
            assert _ask(chain, strategy).answer == e2e.ANSWER, strategy

        # Removing graphrag: the next run that writes the namespace stops
        # maintaining the reports and records them gone, deleting nothing.
        config.indexing.methodologies = [Methodology.LIGHTRAG]
        (source / "freight.txt").write_text("Carrier moves gadgets for Vendor.")
        context = e2e._ingest(config, providers, source, tmp_path / "run-3")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        assert "community_detection" not in _ran(context)
        assert _record(providers)["methodologies"] == ["lightrag", "hipporag"]
        assert _reports(config, providers)
        assert _community_vertices(config) > 0
    finally:
        chain.close()

    # A new chain reads the record: GraphRAG's strategies are refused again.
    chain = GraphRAGChain(config, providers=providers)
    try:
        with pytest.raises(MethodologyNotIndexedError):
            _ask(chain, SearchStrategy.GLOBAL)
    finally:
        chain.close()

    # A reset with the remaining methodology replaces the reports index with
    # an empty one.
    config.indexing.reset = True
    context = e2e._ingest(config, providers, source, tmp_path / "run-4")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    prefix = config.indexing.opensearch.community_reports_index_prefix
    assert e2e._stored(config, providers, prefix) == []
    assert _community_vertices(config) == 0


def test_a_resumed_pipeline_id_runs_only_the_added_stages(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Without the registry, a methodology added to a pipeline id runs only
    # the stages it adds (and indexing, downstream of them) from the stage
    # cache.
    config, providers, _, calls = _setup(tmp_path, registry=False)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    source = _corpus(tmp_path)
    cache = tmp_path / "run"
    context = e2e._ingest(config, providers, source, cache)
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert "community_detection" not in _ran(context)

    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    before = Counter(calls.kinds)
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=cache,
        ),
        source_directory=source,
        providers=providers,
    )
    try:
        context = pipeline.run(source, pipeline_id=cache.name)
    finally:
        pipeline.close()
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert set(calls.since(before)) == {"report"}
    assert pipeline.executed_stages == {"community_detection", "indexing"}
    assert _record(providers)["methodologies"] == ["graphrag", "lightrag", "hipporag"]


def test_adding_a_methodology_never_rebuilds_over_another_scope(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # A rebuild from one corpus deletes the namespace's other source scopes:
    # the run is refused before any model call instead, and writes nothing.
    config, providers, server, _ = _setup(tmp_path)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    first, second = _corpus(tmp_path / "a"), tmp_path / "b" / "corpus"
    second.mkdir(parents=True)
    (second / "notes.txt").write_text("Bank lends money to Vendor.")
    for source in (first, second):
        cache = tmp_path / f"run-{source.parent.name}"
        context = e2e._ingest(config, providers, source, cache)
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    record = _record(providers)

    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    calls = len(server.chat_models)
    with pytest.raises(ConfigurationError, match="also holds documents of source"):
        e2e._ingest(config, providers, first, tmp_path / "add")
    assert len(server.chat_models) == calls
    assert config.indexing.reset is False
    assert _record(providers) == record
    assert e2e._registry(providers) == {
        "contract.txt": DocStatus.PROCESSED,
        "logistics.txt": DocStatus.PROCESSED,
        "freight.txt": DocStatus.PROCESSED,
        "notes.txt": DocStatus.PROCESSED,
    }


def test_without_the_registry_a_new_run_reuses_the_stored_extractions(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    # Every run is a full build: one under a new pipeline id that adds
    # graphrag takes the extraction store's records, so its only new model
    # calls are the community reports.
    config, providers, _, calls = _setup(tmp_path, registry=False)
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    source = _corpus(tmp_path)
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    config.indexing.methodologies = [Methodology.GRAPHRAG, Methodology.LIGHTRAG]
    before = Counter(calls.kinds)
    context = e2e._ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert set(calls.since(before)) == {"report"}
    assert _record(providers)["methodologies"] == ["graphrag", "lightrag", "hipporag"]
