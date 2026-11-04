# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The build record's community counters across whole-pipeline runs.

A full build starts them with the communities it clustered; each
incremental commit adds a run and the communities it appended, less those
its removals deleted; a reset starts them again. A commit is counted once
even when a resumed run writes the record again. AWS-free: in-memory stores
and a scripted model.
"""

from __future__ import annotations

import logging
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
from unified_kg_rag.domain.models import (
    CommunityCounters,
    Config,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)

pytestmark = pytest.mark.integration

_RECOMMENDS = "due for re-clustering"


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
        self.config.processing.claim_extraction.enabled = False
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
            context = pipeline.run(self.source, pipeline_id=pipeline_id)
        finally:
            pipeline.close()
        assert context.status is PipelineStageStatus.COMPLETED, [
            r.error_message for r in context.stage_results if r.error_message
        ]
        return context

    def add(self, name: str, text: str) -> None:
        (self.source / name).write_text(text)

    def counters(self) -> CommunityCounters | None:
        return self.vectors.builds["default"].communities

    def stored_communities(self) -> int:
        return len(self.graph.data.get("communities", {}))


def _indexing_metrics(context: PipelineContext) -> dict:
    return next(r.metrics for r in context.stage_results if r.stage_name == "indexing")


def test_counters_follow_full_build_incremental_rounds_and_reset(
    tmp_path, caplog
) -> None:
    harness = _Harness(tmp_path)
    harness.config.indexing.community_compaction.max_runs_since_clustering = 2
    harness.config.indexing.community_compaction.max_community_growth = None
    harness.run("full")
    full = harness.counters()
    assert full is not None
    clustered = harness.stored_communities()
    assert clustered > 0
    assert (full.communities_at_clustering, full.communities_now) == (
        clustered,
        clustered,
    )
    assert full.runs_since == 0

    caplog.set_level(logging.WARNING)
    harness.add("carrier.txt", "Carrier ships widgets for Depot.")
    first = harness.run("add-1")
    counted = harness.counters()
    assert counted is not None
    assert counted.runs_since == 1
    assert counted.communities_at_clustering == clustered
    assert counted.communities_now == harness.stored_communities()
    assert counted.clustered_at == full.clustered_at
    assert _indexing_metrics(first)["runs_since_clustering"] == 1
    assert _RECOMMENDS not in caplog.text

    # A deletion subtracts the communities its removal deleted.
    before = counted.communities_now
    (harness.source / "carrier.txt").unlink()
    second = harness.run("delete-1")
    counted = harness.counters()
    assert counted is not None
    assert counted.runs_since == 2
    assert counted.communities_now < before
    assert counted.communities_now == harness.stored_communities()
    metrics = _indexing_metrics(second)
    assert metrics["runs_since_clustering"] == 2
    assert metrics["community_growth_ratio"] == counted.growth_ratio
    # The second commit reaches max_runs_since_clustering: one WARNING.
    warnings = [r.getMessage() for r in caplog.records if _RECOMMENDS in r.getMessage()]
    assert len(warnings) == 1
    assert "2 incremental runs since the last full clustering" in warnings[0]
    assert "indexing.reset: true" in warnings[0]

    reset = harness.config.model_copy(deep=True)
    reset.indexing.reset = True
    harness.run("reset", reset)
    rebuilt = harness.counters()
    assert rebuilt is not None
    assert rebuilt.runs_since == 0
    assert rebuilt.communities_now == rebuilt.communities_at_clustering
    assert rebuilt.communities_at_clustering == harness.stored_communities()
    assert rebuilt.clustered_at > full.clustered_at


def test_growth_threshold_recommends_a_reset(tmp_path, caplog) -> None:
    harness = _Harness(tmp_path)
    harness.config.indexing.community_compaction.max_runs_since_clustering = None
    harness.config.indexing.community_compaction.max_community_growth = 1.01
    harness.run("full")
    caplog.set_level(logging.WARNING)
    harness.add("carrier.txt", "Carrier ships widgets for Depot.")
    harness.run("add-1")

    counted = harness.counters()
    assert counted is not None and counted.growth_ratio is not None
    assert counted.growth_ratio >= 1.01
    (warning,) = [
        r.getMessage() for r in caplog.records if _RECOMMENDS in r.getMessage()
    ]
    assert f"{counted.communities_now} communities against the" in warning
    assert "max_community_growth: 1.01" in warning


def test_a_record_without_counters_starts_them_at_the_next_full_build(
    tmp_path,
) -> None:
    harness = _Harness(tmp_path)
    harness.run("full")
    # As an earlier version wrote it: no community counters.
    old = harness.vectors.builds["default"]
    harness.vectors.builds["default"] = old.model_copy(update={"communities": None})

    harness.add("carrier.txt", "Carrier ships widgets for Depot.")
    context = harness.run("add-1")
    assert harness.counters() is None
    assert "runs_since_clustering" not in _indexing_metrics(context)

    reset = harness.config.model_copy(deep=True)
    reset.indexing.reset = True
    harness.run("reset", reset)
    counted = harness.counters()
    assert counted is not None
    assert counted.runs_since == 0
    assert counted.communities_at_clustering == harness.stored_communities()


def test_a_resumed_commit_is_counted_once(tmp_path, monkeypatch) -> None:
    harness = _Harness(tmp_path)
    # Killed once the record is written, before the stage completes.
    check = ps.IndexingStage._check_community_counters
    crash = {"armed": True}

    def killed_once(self, counters, drift):
        if crash.pop("armed", False):
            raise _Died("killed after recording the build")
        return check(self, counters, drift)

    monkeypatch.setattr(ps.IndexingStage, "_check_community_counters", killed_once)
    # The first run (a delta of new documents only) resumed: still none.
    with pytest.raises(_Died):
        harness.run("full")
    harness.run("full")
    first = harness.counters()
    assert first is not None and first.runs_since == 0
    assert first.communities_now == harness.stored_communities()

    harness.add("carrier.txt", "Carrier ships widgets for Depot.")
    crash["armed"] = True
    with pytest.raises(_Died):
        harness.run("add-1")
    killed = harness.counters()
    assert killed is not None and killed.runs_since == 1

    # The same pipeline id resumes at indexing and commits the delta again.
    context = harness.run("add-1")
    assert _indexing_metrics(context)["runs_since_clustering"] == 1
    assert harness.counters() == killed

    # A later run under the same pipeline id commits another delta.
    harness.add("depot.txt", "Depot ships gadgets for Carrier.")
    harness.run("add-1")
    later = harness.counters()
    assert later is not None and later.runs_since == 2
