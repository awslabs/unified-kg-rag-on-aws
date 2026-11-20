# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""docs/guides/extend.md's "A new vector backend", followed literally: a
``VectorIndexer`` with only its abstract writes and ``clear()``
(``tests/fixtures/toy_vector_backend.py``) builds and updates a namespace,
each concern it leaves at its "cannot" default skipping what needs it, and
the store conformance suites run on it through ``UKR_TEST_BACKEND``."""

from __future__ import annotations

import logging
import os
import subprocess  # nosec B404 - runs pytest on a fixed argv
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tests.fixtures import store_backends
from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeGraphStore
from tests.fixtures.toy_vector_backend import TOY_BACKEND, ToyVectorStore
from tests.integration import test_ingestion_stages as stages
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion import pipeline_stages
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import (
    Config,
    PipelineConfig,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.ports import VectorIndexer

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_toy_implements_only_the_recipe() -> None:
    own = {
        name
        for name, value in vars(ToyVectorStore).items()
        if callable(value) and not name.startswith("_")
    }
    assert own == set(VectorIndexer.__abstractmethods__)


def _pipeline(
    config: Config,
    source: Path,
    tmp_path: Path,
    vectors: ToyVectorStore,
    graph: FakeGraphStore,
    registry: FakeDocStatusStore,
) -> DataIngestionPipeline:
    providers = Providers(
        config,
        boto_session=MagicMock(),
        llm_factory=ScriptedLLMFactory(stages._respond),
        embedding_factory=HashingEmbeddingFactory(),
        doc_status=registry,
        vector_indexer_factory=lambda: vectors,
        graph_indexer_factory=lambda: graph,
    )
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
        if isinstance(stage, pipeline_stages.GraphResolutionStage):
            stage.resolver.entity_resolver.use_process_pool = False
            stage.resolver.relationship_resolver.use_process_pool = False
    return pipeline


def test_a_full_build_and_an_incremental_run_complete(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config = stages._config()
    config.processing.gleaning.enabled = False
    config.processing.claim_extraction.enabled = False
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    registry = FakeDocStatusStore()
    graph = FakeGraphStore(config)
    vectors = ToyVectorStore(config)

    with caplog.at_level(logging.INFO):
        full = _pipeline(config, source, tmp_path, vectors, graph, registry).run(
            source, pipeline_id="toy-1"
        )
    assert full.status is PipelineStageStatus.COMPLETED
    assert vectors.items["text_units"] and vectors.items["entities"]
    # The visualization export (on by default) needs StoreInspection: it is
    # skipped like the summary tree, saying why, not reported failed.
    exports = [r for r in caplog.records if "visualization" in r.getMessage()]
    assert [r.levelno for r in exports] == [logging.WARNING]
    assert "StoreInspection.scan_items" in exports[0].getMessage()
    assert "failed" not in exports[0].getMessage()
    assert full.visualization_data_path is None

    (source / "logistics.txt").unlink()
    (source / "new.txt").write_text("Carrier moves widgets for Depot.")
    incremental = _pipeline(config, source, tmp_path, vectors, graph, registry).run(
        source, pipeline_id="toy-2"
    )
    assert incremental.status is PipelineStageStatus.COMPLETED


def test_an_extra_backend_is_loaded_by_name() -> None:
    spec = "tests.fixtures.toy_vector_backend:TOY_BACKEND"
    assert store_backends.extra_backend(spec) is TOY_BACKEND
    assert store_backends.extra_backend(None) is None
    with pytest.raises(TypeError, match="not an ExtraBackend"):
        store_backends.extra_backend("tests.fixtures.toy_vector_backend:ToyVectorStore")


def test_the_harness_opens_an_extra_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store_backends, "EXTRA_BACKEND", TOY_BACKEND)
    monkeypatch.setitem(store_backends.CAPABILITIES, "toy", TOY_BACKEND.capabilities)
    harness_iterator = store_backends.open_harness("toy", tmp_path)
    harness = next(harness_iterator)
    assert isinstance(harness.vector_indexer, ToyVectorStore)
    # Without RETRIEVERS among its capabilities, no retriever is built.
    assert harness.vector_retriever is None and harness.graph_retriever is None
    with pytest.raises(pytest.skip.Exception, match="cannot list its entities"):
        harness.stored_ids("entities")
    with pytest.raises(StopIteration):
        next(harness_iterator)


def test_the_suites_collect_the_extra_backend(tmp_path: Path) -> None:
    env = {
        **os.environ,
        "UKR_TEST_BACKEND": "tests.fixtures.toy_vector_backend:TOY_BACKEND",
    }
    collected = subprocess.run(  # nosec B603 - fixed argv
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "--basetemp",
            str(tmp_path / "collect"),
            "tests/integration/test_vector_store_conformance.py",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert collected.returncode == 0, (
        collected.stdout[-2000:] + collected.stderr[-2000:]
    )
    assert "[toy]" in collected.stdout
