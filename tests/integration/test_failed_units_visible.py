# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run that completes with text units its extraction failed on says so.

A full build (no doc-status registry) records no document FAILED, so its
``failed_documents`` is 0 however many units failed; a real run lost 138 of
3,318 units that way with only a mid-run WARNING per unit. The run now
reports ``units_failed`` (JSON and the ``total_units_failed`` metric) and
ends with a WARNING naming the count, the stage, the cause and what retries
them; the next run extracts those units again.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_ingestion_stages import _source_text
from unified_kg_rag.application.cli.run_ingestion_pipeline import (
    IngestionPipelineRunner,
)
from unified_kg_rag.domain.models import DocStatus, PipelineStageStatus

pytestmark = pytest.mark.integration

_FAILING = "Zorbix and Vendor trade widgets."
# Output cut off before the entities' names arrived, without a stop reason:
# cut-off output (LLMOutputTruncatedError), not retried within the run.
_CUT_OFF = "<entities><entity><type>ORGANIZATION</type></entity><entity><na"


def _setup(tmp: Path, *, registry: bool):
    config = e2e._config(tmp / "stores", tmp / "viz")
    config.providers.doc_status.enabled = registry
    state = {"cut": True, "asked": 0}

    def respond(system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system and _FAILING in (
            _source_text(human)
        ):
            state["asked"] += 1
            if state["cut"]:
                return _CUT_OFF
            return e2e.stages._extraction(_FAILING, ("Zorbix", "Vendor"))
        return e2e._respond(system, human)

    server = FakeOllama(respond, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(
        config, server, e2e.FakeCrossEncoder("r")  # type: ignore[arg-type]
    )
    source = tmp / "corpus"
    source.mkdir()
    (source / "d0.txt").write_text("Vendor and Buyer trade goods.")
    (source / "fz.txt").write_text(_FAILING)
    return config, providers, source, state


def _json(context) -> dict:
    runner = IngestionPipelineRunner.__new__(IngestionPipelineRunner)
    runner.args = argparse.Namespace(max_failure_rate=None)
    runner.pipeline = SimpleNamespace(executed_stages=set())  # type: ignore[assignment]
    return runner._json_result(context)


def test_a_full_build_reports_its_failed_units_and_the_next_run_retries_them(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config, providers, source, state = _setup(tmp_path, registry=False)
    with caplog.at_level(logging.WARNING):
        context = e2e._ingest(config, providers, source, tmp_path / "run0")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    result = _json(context)
    assert (result["failed_documents"], result["units_failed"]) == (0, 1)
    assert context.global_metrics is not None
    assert context.global_metrics.total_units_failed == 1
    summary = [
        r.getMessage() for r in caplog.records if "(units_failed)" in r.getMessage()
    ]
    assert len(summary) == 1, summary
    assert "1 text units failed and are indexed without that stage's output" in (
        summary[0]
    )
    assert "graph_extraction 1 (1 for an error of their own" in summary[0]
    assert "providers.doc_status.enabled" in summary[0]
    assert "Zorbix" not in e2e._entity_names(providers)
    assert state["asked"] == 1

    # The next build asks for that unit only (the extraction store keeps
    # the other's record; the failed response was not cached).
    state["cut"] = False
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        context = e2e._ingest(config, providers, source, tmp_path / "run1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert _json(context)["units_failed"] == 0
    assert not any("(units_failed)" in r.getMessage() for r in caplog.records)
    assert state["asked"] == 2
    assert "Zorbix" in e2e._entity_names(providers)


def test_an_incremental_run_records_the_documents_and_the_next_run_retries_them(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    config, providers, source, state = _setup(tmp_path, registry=True)
    with caplog.at_level(logging.WARNING):
        context = e2e._ingest(config, providers, source, tmp_path / "run0")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    result = _json(context)
    assert (result["failed_documents"], result["units_failed"]) == (1, 1)
    summary = [
        r.getMessage() for r in caplog.records if "(units_failed)" in r.getMessage()
    ]
    assert len(summary) == 1 and "1 documents recorded FAILED" in summary[0]
    assert e2e._registry(providers) == {
        "d0.txt": DocStatus.PROCESSED,
        "fz.txt": DocStatus.FAILED,
    }

    state["cut"] = False
    context = e2e._ingest(config, providers, source, tmp_path / "run1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert _json(context)["units_failed"] == 0
    assert state["asked"] == 2
    assert e2e._registry(providers)["fz.txt"] is DocStatus.PROCESSED
    assert "Zorbix" in e2e._entity_names(providers)


def test_failed_units_of_a_run_that_stopped_before_indexing_are_not_indexed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Regression (rv24): the summary said "indexed without that stage's
    # output" of a run that failed before its indexing stage.
    from datetime import datetime

    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.domain.models import PipelineContext

    pipeline = object.__new__(DataIngestionPipeline)
    pipeline._doc_status = None
    context = PipelineContext(
        pipeline_id="p",
        config={},
        status=PipelineStageStatus.FAILED,
        start_time=datetime(2026, 1, 1),
        source_directory=Path("."),
        failed_text_unit_ids={"graph_extraction": ["u1"]},
    )
    with caplog.at_level(logging.WARNING):
        pipeline._warn_failed_units(context)
    (summary,) = [r.getMessage() for r in caplog.records]
    assert "are not indexed: the run stopped before indexing" in summary
