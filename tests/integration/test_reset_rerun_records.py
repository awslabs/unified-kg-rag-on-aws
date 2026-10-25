# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A reset pipeline id run again after another pipeline id replaced its
extraction records (``backend: local``).

The rerun resumed at indexing from its cached stage outputs; its text units
named records the other run's incremental commit deleted, so the full build
kept none of them and the next incremental run could not derive the items
those units cite. It now resumes from graph_extraction, which extracts the
units whose records are gone again and stores them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_name_variant_convergence import _respond
from tests.integration.test_pipeline_id_rerun import _EXTRACTION, _write_corpus
from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.ingestion.contributions import attribute_record_keys
from unified_kg_rag.domain.models import (
    Config,
    PipelineConfig,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.shared import PipelineExecutionError

pytestmark = pytest.mark.integration

_CORPUS = {
    "d0.txt": "Buyer trade parts 2. Depot and Acme LLC and Acme Corp trade cash 1.",
    "d3.txt": "Buyer and Depot trade cash 3. Acme Inc and Buyer and Acme Corp "
    "trade goods 3.",
}


@pytest.mark.parametrize("cached_responses", [True, False])
def test_a_reset_rerun_keeps_the_records_its_text_units_name(
    tmp_path: Path, cached_responses: bool
) -> None:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.processing.claim_extraction.enabled = True
    config.processing.document_parsing.source_scope = "rerun-corpus"
    config.cache.llm_responses.enabled = cached_responses
    extractions: list[str] = []

    def respond(system: str, human: str) -> str:
        if _EXTRACTION in system:
            extractions.append(human)
        return _respond(system, human)

    providers = e2e._providers(
        config, FakeOllama(respond, "x", e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    _write_corpus(source, _CORPUS)
    calls = []
    for pipeline_id, reset, gleaning in (
        ("P0", True, True),
        ("P1", False, False),  # re-extracts every unit, deletes P0's records
        ("P0", True, True),  # rerun: resumes from P0's cached stages
    ):
        config.indexing.reset = reset
        config.processing.gleaning.enabled = gleaning
        before = len(extractions)
        e2e._ingest(config, providers, source, tmp_path / pipeline_id)
        calls.append(len(extractions) - before)
    opensearch = config.indexing.opensearch
    named = {
        key
        for unit in e2e._stored(config, providers, opensearch.text_units_index_prefix)
        for key in attribute_record_keys(unit["_source"].get("attributes")).values()
    }
    stored = {
        record["_id"]
        for record in e2e._stored(
            config, providers, opensearch.extraction_records_index_prefix
        )
    }
    assert named
    assert named <= stored
    # The rerun extracts again only without the response cache, and then
    # only the units whose records are gone (all of them here).
    assert calls[2] == (0 if cached_responses else calls[0]), calls


def _uncached_setup(
    tmp_path: Path,
) -> tuple[Config, Providers, Path, list[str]]:
    """The corpus, with the LLM response cache off, and the list of every
    model call's prompt."""
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.processing.claim_extraction.enabled = True
    config.processing.document_parsing.source_scope = "rerun-corpus"
    config.cache.llm_responses.enabled = False
    calls: list[str] = []

    def respond(system: str, human: str) -> str:
        calls.append(human)
        return _respond(system, human)

    providers = e2e._providers(
        config, FakeOllama(respond, "x", e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    _write_corpus(source, _CORPUS)
    return config, providers, source, calls


def test_a_reset_rerun_whose_records_are_stored_runs_no_stage_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Nothing replaced the records: the rerun resumes at indexing from the
    # cached stages and, with the response cache off, calls no model.
    config, providers, source, calls = _uncached_setup(tmp_path)
    config.indexing.reset = True
    e2e._ingest(config, providers, source, tmp_path / "P0")
    first = len(calls)
    assert first
    records = config.indexing.opensearch.extraction_records_index_prefix
    read: list[object] = []
    search = SQLiteSearchEngine.search

    def recording(engine: SQLiteSearchEngine, **kwargs: Any) -> dict[str, Any]:
        if records in kwargs["index"]:
            read.append(kwargs["body"].get("_source"))
        return search(engine, **kwargs)

    monkeypatch.setattr(SQLiteSearchEngine, "search", recording)
    context = e2e._ingest(config, providers, source, tmp_path / "P0")
    assert len(calls) == first
    # The check that the records are still there (the first read, before
    # any stage runs) reads their ids only.
    assert read[0] is False, read
    assert context.stage_results[-1].stage_name == "indexing"
    assert context.stage_results[-1].status is PipelineStageStatus.COMPLETED


def test_a_reset_rerun_refuses_when_the_stage_storing_gone_records_is_disabled(
    tmp_path: Path,
) -> None:
    # P1 replaced P0's records; with graph_extraction off the rerun of P0
    # cannot store them again, so it stops instead of indexing units that
    # name records that are gone.
    config, providers, source, _ = _uncached_setup(tmp_path)
    for pipeline_id, reset in (("P0", True), ("P1", False)):
        config.indexing.reset = reset
        config.processing.gleaning.enabled = reset
        e2e._ingest(config, providers, source, tmp_path / pipeline_id)
    config.indexing.reset = True
    config.processing.gleaning.enabled = True
    enabled = dict.fromkeys(PipelineStageType, True)
    enabled[PipelineStageType.GRAPH_EXTRACTION] = False
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(stages_enabled=enabled, local_directory=tmp_path / "P0"),
        source_directory=source,
        providers=providers,
    )
    for stage in pipeline.stages:
        if isinstance(stage, ps.GraphResolutionStage):
            stage.resolver.entity_resolver.use_process_pool = False
            stage.resolver.relationship_resolver.use_process_pool = False
    try:
        with pytest.raises(
            PipelineExecutionError, match="stored are gone .*'graph_extraction' must"
        ):
            pipeline.run(source, pipeline_id="P0")
    finally:
        pipeline.close()
