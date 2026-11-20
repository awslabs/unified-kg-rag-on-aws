# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A base-level community report that fails for an error of its own counts
toward ``indexing.max_document_failures`` (embedded stores, no AWS).

The ARCH baseline's local run: a report the model cuts off at temperature 0
is cut off again, and its documents, recorded FAILED for a failure that was
never counted, were re-processed by every incremental run. Counted like an
extraction's own error, the document stops being retried once it failed
that many consecutive runs. The gate over the report failures tests them
against the tolerated chance per report: one failed report of a small delta
does not fail the stage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.doc_status_records import all_records
from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.rerank import FakeCrossEncoder
from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401
from unified_kg_rag.domain.models import DocStatus, PipelineStageStatus

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

_NAMES = ("Vendor", "Buyer", "Depot", "Kestrel", "Umbra", "Harbor")


class _Model:
    """Reports of the community naming Kestrel come back as output that does
    not parse (every attempt, the output fixer's too); the others are
    answered."""

    def __init__(self) -> None:
        self.report_inputs: list[str] = []

    def __call__(self, system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system:
            return stages._extraction(stages._source_text(human), _NAMES)
        if "community analysis" in system:
            self.report_inputs.append(human)
            if "Kestrel" in human:
                return "Sorry, I cannot help with that."
            return stages._REPORT
        return e2e._respond(system, human)


@pytest.mark.timeout(600)
def test_a_report_failing_alike_every_run_stops_retrying_its_document(
    tmp_path: Path,
) -> None:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.indexing.max_document_failures = 2
    model = _Model()
    providers = e2e._providers(
        config, FakeOllama(model, e2e.ANSWER, e2e.DIM), FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    # Two components of three entities: a community each.
    (source / "a.txt").write_text("Vendor supplies Buyer and Depot.", "utf-8")
    (source / "c.txt").write_text("Kestrel ships Umbra parts via Harbor.", "utf-8")

    def records() -> dict[str, tuple[DocStatus, int]]:
        registry = providers.build_doc_status()
        try:
            return {
                Path(r.file_path or "").name: (r.status, r.failure_count)
                for r in all_records(registry)
            }
        finally:
            registry.close()

    def kestrel_reports() -> int:
        return sum("Kestrel" in text for text in model.report_inputs)

    first = e2e._ingest(config, providers, source, tmp_path / "run1")
    # One of two reports failed: within the chance the gate tolerates.
    assert first.status is PipelineStageStatus.COMPLETED, e2e._errors(first)
    assert records()["c.txt"] == (DocStatus.FAILED, 1)
    assert records()["a.txt"][0] is DocStatus.PROCESSED
    asked = kestrel_reports()

    # The next run retries it (one failed report of one: still within it).
    second = e2e._ingest(config, providers, source, tmp_path / "run2")
    assert second.status is PipelineStageStatus.COMPLETED, e2e._errors(second)
    assert records()["c.txt"] == (DocStatus.FAILED, 2)
    assert kestrel_reports() > asked
    asked = kestrel_reports()

    # Failed max_document_failures consecutive runs: no longer retried.
    third = e2e._ingest(config, providers, source, tmp_path / "run3")
    assert third.status is PipelineStageStatus.COMPLETED, e2e._errors(third)
    assert records()["c.txt"] == (DocStatus.FAILED, 2)
    assert kestrel_reports() == asked
