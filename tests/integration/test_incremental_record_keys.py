# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Indexed text units name their stored extraction records, always.

An incremental run derives a stored item's content from the records of the
text units it cites (``domain.ingestion.derivation``), found through the keys
the stored units name. Whole-pipeline runs over one registry (AWS-free): a
run resumed after the extraction stages writes the same keys as one that was
not interrupted, and a unit whose record was not stored names none and its
document is extracted again by the next run.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from tests.integration.test_ingestion_stages import _config, _respond
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.ingestion.contributions import record_keys
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.ports.indexer import IndexingStats

pytestmark = pytest.mark.integration


class _Died(BaseException):
    """A crash the pipeline does not catch (the task was killed)."""


class _Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.source = tmp_path / "corpus"
        self.source.mkdir(parents=True)
        (self.source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
        (self.source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
        self.store = FakeDocStatusStore()
        self.graph = FakeGraphStore()
        self.config = _config()
        self.config.processing.gleaning.enabled = False
        self.config.processing.claim_extraction.enabled = True
        self.vectors = FakeVectorStore(self.config.indexing.opensearch)

    def run(self, pipeline_id: str, config: Config | None = None) -> PipelineContext:
        config = config or self.config
        providers = Providers(
            config,
            boto_session=MagicMock(),
            llm_factory=ScriptedLLMFactory(_respond),
            embedding_factory=HashingEmbeddingFactory(),
            doc_status=self.store,
            vector_indexer_factory=lambda: self.vectors,
            graph_indexer_factory=lambda: self.graph,
        )
        pipeline = DataIngestionPipeline(
            config,
            PipelineConfig(
                stages_enabled=dict.fromkeys(PipelineStageType, True),
                local_directory=self.tmp_path / "cache",
            ),
            source_directory=self.source,
            providers=providers,
        )
        for stage in pipeline.stages:
            if isinstance(stage, ps.GraphResolutionStage):
                stage.resolver.entity_resolver.use_process_pool = False
                stage.resolver.relationship_resolver.use_process_pool = False
        try:
            return pipeline.run(self.source, pipeline_id=pipeline_id)
        finally:
            pipeline.close()

    def unit_keys(self) -> dict[str, dict[str, str]]:
        return {
            unit.text[:12]: record_keys(unit)
            for unit in self.vectors.data["text_units"].values()
        }


def test_a_resume_after_the_extraction_stages_names_the_records(
    tmp_path, monkeypatch
) -> None:
    uninterrupted = _Harness(tmp_path / "a")
    uninterrupted.run("run-1")
    expected = uninterrupted.unit_keys()
    assert all(set(keys) == {"graph", "claims"} for keys in expected.values())

    harness = _Harness(tmp_path / "b")
    execute = ps.IndexingStage._execute_core
    crash = {"armed": True}

    def killed_once(self, context):
        if crash.pop("armed", False):
            raise _Died("killed while indexing")
        return execute(self, context)

    monkeypatch.setattr(ps.IndexingStage, "_execute_core", killed_once)
    with pytest.raises(_Died):
        harness.run("run-1")
    # The same pipeline id: every stage up to indexing is restored from the
    # stage cache, none re-runs.
    harness.run("run-1")

    assert harness.unit_keys() == expected


def test_a_record_that_was_not_stored_fails_its_document(tmp_path, monkeypatch) -> None:
    harness = _Harness(tmp_path)
    write = FakeVectorStore.write_extraction_records

    def rejecting(self, records, suffix=None):
        stats = IndexingStats(total_items=len(records))
        stats.add_error("store full", len(records), ids=list(records))
        return stats

    monkeypatch.setattr(FakeVectorStore, "write_extraction_records", rejecting)
    context = harness.run("run-1")

    # Indexed, but recorded FAILED: without its records no later run could
    # derive what the document's units state.
    assert {r.status for r in harness.store.list_all()} == {DocStatus.FAILED}
    assert all(not keys for keys in harness.unit_keys().values())
    resolution = next(
        r for r in context.stage_results if r.stage_name == "graph_resolution"
    )
    assert resolution.metrics["records_not_stored"] == 2

    monkeypatch.setattr(FakeVectorStore, "write_extraction_records", write)
    harness.run("run-2")
    assert {r.status for r in harness.store.list_all()} == {DocStatus.PROCESSED}
    assert all(
        set(keys) == {"graph", "claims"} for keys in harness.unit_keys().values()
    )


def test_a_record_the_store_rejected_is_written_again(tmp_path) -> None:
    # A record the store rejected is not marked stored, so the stage's next
    # attempt in the run writes it again (and only it).
    harness = _Harness(tmp_path)
    providers = Providers(
        harness.config,
        boto_session=MagicMock(),
        llm_factory=ScriptedLLMFactory(_respond),
        embedding_factory=HashingEmbeddingFactory(),
        doc_status=harness.store,
        vector_indexer_factory=lambda: harness.vectors,
        graph_indexer_factory=lambda: harness.graph,
    )
    stage = ps.GraphResolutionStage(harness.config, providers=providers)
    written: list[set[str]] = []

    def save(units, fingerprint):
        ids = {unit.id for unit, _ in units}
        written.append(ids)
        return ids & {"t1"} if len(written) == 1 else set()

    stage.store.save = save  # type: ignore[method-assign]
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=str(harness.source),
        text_units=[TextUnit(id=i, text=f"text {i}") for i in ("t1", "t2")],
        unit_extractions=[TextUnitExtraction(text_unit_id=i) for i in ("t1", "t2")],
    )

    assert stage._store_records(context) == {"t1"}
    assert stage._store_records(context) == set()
    assert [ids for ids in written if ids] == [{"t1", "t2"}, {"t1"}]
