# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental runs follow the extraction settings, not only the content.

Two whole-pipeline runs over one registry (AWS-free): a document whose content
did not change is re-extracted when the settings it was extracted with
(here a tuned extraction prompt) changed, and not when they did not. An
incremental run that skips unchanged documents does not export a
visualization of its partial graph.
"""

from __future__ import annotations

import json
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
from unified_kg_rag.domain.ingestion.cache_keys import extraction_fingerprint
from unified_kg_rag.domain.ingestion.visualization_data import (
    VISUALIZATION_DATA_FILENAME,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatusRecord,
    ExtractionSettings,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)

pytestmark = pytest.mark.integration

_TUNED = "TUNED-PERSONA"


class _Harness:
    def __init__(self, tmp_path: Path) -> None:
        self.tmp_path = tmp_path
        self.source = tmp_path / "corpus"
        self.source.mkdir()
        (self.source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
        (self.source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
        self.store = FakeDocStatusStore()
        self.graph = FakeGraphStore()
        # Both stores persist across runs, as the registry does.
        self.vectors: FakeVectorStore | None = None
        self.extractions: list[str] = []
        self.claim_calls = 0

    def _respond(self, system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system:
            self.extractions.append("tuned" if _TUNED in system else "builtin")
        if "claim extraction specialist" in system:
            self.claim_calls += 1
        return _respond(system, human)

    def run(self, config: Config, pipeline_id: str) -> PipelineContext:
        if self.vectors is None:
            self.vectors = FakeVectorStore(config.indexing.opensearch)
        vectors = self.vectors
        providers = Providers(
            config,
            boto_session=MagicMock(),
            llm_factory=ScriptedLLMFactory(self._respond),
            embedding_factory=HashingEmbeddingFactory(),
            doc_status=self.store,
            vector_indexer_factory=lambda: vectors,
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
        context = pipeline.run(self.source, pipeline_id=pipeline_id)
        assert context.status is PipelineStageStatus.COMPLETED
        return context


def _base_config() -> Config:
    config = _config()
    config.processing.gleaning.enabled = False
    config.processing.claim_extraction.enabled = False
    return config


def _tuned(config: Config) -> Config:
    tuned = config.model_copy(deep=True)
    # A persona line ahead of the built-in system prompt, as prompt tuning
    # writes one.
    from unified_kg_rag.domain.prompts.graph_extraction import (
        GraphExtractionPrompt,
    )

    tuned.custom_prompts.graph_extraction_system = "\n".join(
        (
            _TUNED,
            GraphExtractionPrompt.system_preamble,
            GraphExtractionPrompt.output_rules,
        )
    )
    return tuned


def _fingerprints(store: FakeDocStatusStore) -> set[str | None]:
    return {record.extraction_fingerprint for record in store.list_all()}


def test_changed_extraction_settings_reextract_unchanged_documents(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    harness = _Harness(tmp_path)
    builtin = _base_config()
    harness.run(builtin, "first")
    assert harness.extractions == ["builtin", "builtin"]
    assert _fingerprints(harness.store) == {extraction_fingerprint(builtin)}

    tuned = _tuned(builtin)
    harness.extractions.clear()
    with caplog.at_level("WARNING"):
        harness.run(tuned, "tuned")
    assert harness.extractions == ["tuned", "tuned"]
    assert "Re-extracting 2 documents" in caplog.text
    assert _fingerprints(harness.store) == {extraction_fingerprint(tuned)}

    # The same settings again: nothing to re-extract.
    harness.extractions.clear()
    harness.run(tuned, "tuned-again")
    assert harness.extractions == []


def test_switching_claim_extraction_on_reextracts_unchanged_documents(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    harness = _Harness(tmp_path)
    without_claims = _base_config()
    harness.run(without_claims, "first")
    assert harness.claim_calls == 0

    # Claims are per-document output: documents extracted without them are
    # stale once claim extraction is on, although their content is unchanged.
    # Their graph records do not depend on the claim settings, so the
    # extraction store serves them and only the claims are extracted.
    with_claims = without_claims.model_copy(deep=True)
    with_claims.processing.claim_extraction.enabled = True
    harness.extractions.clear()
    with caplog.at_level("WARNING"):
        harness.run(with_claims, "claims")
    assert harness.extractions == []
    assert harness.claim_calls > 0
    assert "Re-extracting 2 documents" in caplog.text
    assert _fingerprints(harness.store) == {extraction_fingerprint(with_claims)}

    # A tuned claim prompt shapes the claims too.
    tuned_claims = with_claims.model_copy(deep=True)
    tuned_claims.custom_prompts.claim_extraction_system = "tuned claims"
    assert extraction_fingerprint(tuned_claims) != extraction_fingerprint(with_claims)


def test_a_setting_added_at_its_default_does_not_reextract(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # A newer version adds an extraction setting whose default keeps the
    # earlier behaviour: its record has the field, the stored one does not,
    # so the fingerprints differ but the settings compare equal field by
    # field, and the run keeps the recorded fingerprint.
    harness = _Harness(tmp_path)
    config = _base_config()
    harness.run(config, "first")
    ((scope, (fingerprint, settings)),) = harness.store.extraction_settings.items()
    assert fingerprint == extraction_fingerprint(config) == settings.fingerprint

    added = "processing.graph_extraction.max_entities_per_chunk"
    older = ExtractionSettings(
        version=settings.version,
        values={k: v for k, v in settings.values.items() if k != added},
    )
    harness.store.put_extraction_settings(scope, older.fingerprint, older)
    harness.store.put_many(
        [
            record.model_copy(update={"extraction_fingerprint": older.fingerprint})
            for record in harness.store.list_all()
        ]
    )

    harness.extractions.clear()
    harness.run(config, "upgraded")
    assert harness.extractions == []
    assert _fingerprints(harness.store) == {older.fingerprint}

    # Set away from its default, the added setting re-extracts and is named.
    tuned = config.model_copy(deep=True)
    tuned.processing.graph_extraction.max_entities_per_chunk += 1
    harness.extractions.clear()
    with caplog.at_level("WARNING"):
        harness.run(tuned, "tuned")
    assert harness.extractions == ["builtin", "builtin"]
    assert added in caplog.text
    assert _fingerprints(harness.store) == {extraction_fingerprint(tuned)}


def test_records_without_a_fingerprint_are_not_reextracted(tmp_path: Path) -> None:
    harness = _Harness(tmp_path)
    builtin = _base_config()
    harness.run(builtin, "first")
    # As written before records carried a fingerprint.
    harness.store.put_many(
        [
            DocStatusRecord.model_validate(
                {**record.model_dump(), "extraction_fingerprint": None}
            )
            for record in harness.store.list_all()
        ]
    )

    harness.extractions.clear()
    harness.run(_tuned(builtin), "tuned")
    assert harness.extractions == []


def _visualization_file(tmp_path: Path, pipeline_id: str) -> Path:
    return (
        tmp_path
        / "cache"
        / pipeline_id
        / "visualization"
        / (VISUALIZATION_DATA_FILENAME)
    )


def test_incremental_run_with_unchanged_documents_exports_no_partial_graph(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    harness = _Harness(tmp_path)
    config = _base_config()
    harness.run(config, "full")
    full = json.loads(_visualization_file(tmp_path, "full").read_text())
    assert len(full["nodes"]) == 3
    # Every node carries its level-0 community, which the renderers color by.
    assert all(node["attributes"].get("community_id") for node in full["nodes"])

    (harness.source / "third.txt").write_text("Depot ships widgets to Buyer.")
    with caplog.at_level("INFO"):
        harness.run(config, "delta")
    assert not _visualization_file(tmp_path, "delta").exists()
    assert "Visualization skipped" in caplog.text
