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
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.fixtures import store_backends
from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
from tests.fixtures.fakes.stores import FakeGraphStore
from tests.fixtures.toy_vector_backend import (
    BROKEN_TOY_BACKEND,
    TOY_BACKEND,
    BrokenToyVectorStore,
    ToyVectorStore,
)
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
from unified_kg_rag.ports.indexer import SummaryTreeStore

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

    unkept = "keeps no extraction records"
    assert not [r for r in caplog.records if unkept in r.getMessage()]

    caplog.clear()
    (source / "logistics.txt").unlink()
    (source / "new.txt").write_text("Carrier moves widgets for Depot.")
    incremental = _pipeline(config, source, tmp_path, vectors, graph, registry).run(
        source, pipeline_id="toy-2"
    )
    assert incremental.status is PipelineStageStatus.COMPLETED
    # The removal's and the commit's recovery notes live in the
    # extraction-record store, which the toy leaves at its default: the run
    # says once that they are not stored.
    notes = [r for r in caplog.records if unkept in r.getMessage()]
    assert [r.levelno for r in notes] == [logging.WARNING]
    assert "cannot resume them" in notes[0].getMessage()


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


class _FindingToy(ToyVectorStore):
    """The toy with one optional method of its own."""

    def find_documents(self, *args: Any, **kwargs: Any) -> dict[str, list[str]]:
        return {}


class _AnnotatingToy(ToyVectorStore):
    """The toy listing its schema annotations: ``holds_any_item``'s default
    derives its answer from them."""

    def read_schema_annotations(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []


class _RecordingToy(ToyVectorStore):
    """The toy reading extraction records: ``stored_extraction_record_keys``'
    default returns the keys that read holds."""

    def read_extraction_records(
        self, *args: Any, **kwargs: Any
    ) -> dict[str, dict[str, Any]]:
        return {}


def test_a_test_needing_a_default_method_skips_naming_its_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(store_backends, "EXTRA_BACKEND", TOY_BACKEND)
    monkeypatch.setitem(store_backends.CAPABILITIES, "toy", TOY_BACKEND.capabilities)
    harness = next(store_backends.open_harness("toy", tmp_path))
    with pytest.raises(
        pytest.skip.Exception,
        match=r"toy leaves VectorIndexer\.find_documents, "
        r"ExtractionRecordStore\.read_extraction_records at their defaults",
    ):
        harness.needs("find_documents", "read_extraction_records")
    with pytest.raises(
        pytest.skip.Exception, match="toy leaves SummaryTreeStore at its defaults"
    ):
        harness.needs_port(SummaryTreeStore)
    with pytest.raises(pytest.skip.Exception, match="holds_any_item at its default"):
        harness.needs("holds_any_item")
    # A method the store overrides runs its test.
    harness.vector_indexer = _FindingToy(harness.config)
    harness.needs("find_documents")
    # So does a default that derives its answer from one it overrides
    # (R3D-2), and a port with any method of its own runs its suite.
    harness.vector_indexer = _AnnotatingToy(harness.config)
    assert harness.implements("holds_any_item")
    assert not harness.implements("holds_any_text_unit")
    harness.needs("holds_any_item")
    harness.vector_indexer = _RecordingToy(harness.config)
    assert harness.implements("stored_extraction_record_keys")
    harness.vector_indexer = BrokenToyVectorStore(harness.config)
    harness.needs_port(SummaryTreeStore)
    # A packaged backend states every contract: nothing is skipped.
    fake = next(store_backends.open_harness("fake", tmp_path))
    fake.vector_indexer = ToyVectorStore(fake.config)
    fake.needs("find_documents")
    fake.needs_port(SummaryTreeStore)


def test_the_suites_pass_or_skip_on_the_smallest_backend(tmp_path: Path) -> None:
    # The recipe's command, on a backend that leaves every concern at its
    # default: no test fails; each skips naming what the backend lacks.
    ran = subprocess.run(  # nosec B603 - fixed argv
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-rs",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "-p",
            "no:xdist",
            "--basetemp",
            str(tmp_path / "run"),
            "tests/integration/test_vector_store_conformance.py",
            "tests/integration/test_summary_tree_conformance.py",
            "-k",
            TOY_BACKEND.name,
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "UKR_TEST_BACKEND": "tests.fixtures.toy_vector_backend:TOY_BACKEND",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    output = ran.stdout[-6000:] + ran.stderr[-2000:]
    assert ran.returncode == 0, output
    assert " failed" not in output and " error" not in output, output
    assert "toy leaves VectorIndexer.find_documents at its default" in output
    assert "toy leaves SummaryTreeStore at its defaults" in output


def test_the_suites_fail_a_backend_that_overrides_methods_wrongly(
    tmp_path: Path,
) -> None:
    # Regression (R3D-2): one method of a port left at its default (the
    # optional read_text_unit_fields, a holds probe's derived default)
    # skipped every test of the port, so a summary tree that keeps nothing
    # and a text-unit probe that always answers no passed: 0 failed. Each
    # test skips only on the methods it reads.
    ran = subprocess.run(  # nosec B603 - fixed argv
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-rfs",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:randomly",
            "-p",
            "no:xdist",
            "--basetemp",
            str(tmp_path / "run"),
            "tests/integration/test_vector_store_conformance.py",
            "tests/integration/test_summary_tree_conformance.py",
            "-k",
            BROKEN_TOY_BACKEND.name,
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "UKR_TEST_BACKEND": "tests.fixtures.toy_vector_backend:BROKEN_TOY_BACKEND",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    output = ran.stdout[-12000:] + ran.stderr[-2000:]
    assert ran.returncode == 1, output
    failed = {
        line.split("::")[-1].split("[")[0].split(" ")[0]
        for line in output.splitlines()
        if line.startswith("FAILED ")
    }
    # The tree it reports written and never keeps, the probe that says no.
    assert {
        "test_a_tree_is_written_whole_with_its_record_and_swapped_in",
        "test_holds_any_text_unit_tells_a_suffix_with_text_units",
    } <= failed, output
    # Only the default's own test skips for it.
    assert (
        "brokentoy leaves SummaryTreeStore.read_text_unit_fields at its default"
        in output
    )
    assert "leaves SummaryTreeStore at its defaults" not in output


def test_a_store_keeping_no_notes_is_named_once_per_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from tests.fixtures.fakes.stores import FakeVectorStore
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager
    from unified_kg_rag.application.storage.stored_items import (
        PENDING_COMMIT,
        STRIPPED_SUMMARIES,
    )

    config = Config()
    for vectors, logged in (
        (ToyVectorStore(config), 1),
        (FakeVectorStore(config.indexing.opensearch), 0),
    ):
        caplog.clear()
        manager = IndexingManager(
            config, vector_indexer=vectors, graph_indexer=FakeGraphStore(config)
        )
        manager._write_note(PENDING_COMMIT, "default", {"ids": []})
        manager._write_note(STRIPPED_SUMMARIES, "default", {"ids": []})
        notes = [
            r for r in caplog.records if "keeps no extraction records" in r.getMessage()
        ]
        assert len(notes) == logged, type(vectors).__name__
