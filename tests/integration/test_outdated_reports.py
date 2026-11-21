# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A community report written partly from a deleted document, end to end on
the store backends with retrievers (``backend: local``: the embedded
stores, and with ``LOCAL_STORES=1`` Gremlin Server and OpenSearch; no AWS).

The report of a community over two documents cites both. Deleting one keeps
the community for the other's text units, and the incremental run writes
its report again from the members that remain, with one model call: the
deleted text stops reaching the report and the answers that read it. A
report the run cannot write again (the model fails) is listed in the build
record, a rebuild reason for every reader of the record whatever the drift
settings, and flagged ``outdated`` where a query reads it, until a later run
writes it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.rerank import FakeCrossEncoder
from tests.fixtures.store_backends import (
    RETRIEVERS,
    StoreHarness,
    backends,
    open_harness,
)
from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
from unified_kg_rag.domain.ingestion.rebuild_due import RebuildRemedy, rebuild_reasons
from unified_kg_rag.domain.ingestion.report_references import REPORT_OUTDATED_FIELD
from unified_kg_rag.domain.models import (
    Config,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
    SearchStrategy,
)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

_KEPT = "Vendor supplies widgets to Buyer. Depot stores widgets for Vendor."
# Carrier is named only here: what the deletion removes.
_DELETED = "Carrier moves widgets for Vendor and Buyer every week."
_NAMES = ("Vendor", "Buyer", "Depot", "Carrier")


@pytest.fixture(params=backends(RETRIEVERS))
def store_harness(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Iterator[StoreHarness]:
    yield from open_harness(request.param, tmp_path / "harness")


class _Model:
    """The fake model; a report names the members its input lists, so what
    a report states is what its input held. ``fail_reports`` answers the
    report prompts with what no parser reads."""

    def __init__(self) -> None:
        self.report_inputs: list[str] = []
        self.fail_reports = False

    def __call__(self, system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system:
            return stages._extraction(stages._source_text(human), _NAMES)
        if "community analysis" in system:
            self.report_inputs.append(human)
            if self.fail_reports:
                return "no report"
            members = ", ".join(n for n in _NAMES if n in human)
            return (
                "<community_name>Widget partners</community_name>"
                f"<summary>Members: {members}.</summary>"
                "<rating>5.0</rating><rating_explanation>Small.</rating_explanation>"
                f"<findings><finding><summary>{members}</summary>"
                "<explanation>They trade widgets.</explanation></finding></findings>"
            )
        return e2e._respond(system, human)


def _scan(providers: Any, kind: str) -> dict[str, dict]:
    vectors = providers.build_vector_indexer()
    try:
        pages = vectors.scan_items(kind, "default")
        assert pages is not None
        return {item["id"]: item for page in pages for item in page}
    finally:
        vectors.close()


def _reports(config: Config, providers: Any) -> dict[str, dict]:
    return _scan(providers, "community_reports")


def _record(providers: Any) -> dict:
    vectors = providers.build_vector_indexer()
    try:
        record = vectors.read_build_record("default")
    finally:
        vectors.close()
    assert record is not None
    return record


def _indexing_metrics(context: PipelineContext) -> dict:
    (result,) = [
        r
        for r in context.stage_results
        if r.stage_name == PipelineStageType.INDEXING.value
    ]
    return result.metrics or {}


def _community_sources(config: Config, providers: Any) -> list[dict]:
    chain = GraphRAGChain(config, providers=providers)
    try:
        out = chain.invoke(
            RAGInput(
                query="Who stores the widgets for Vendor?",
                search_strategy=SearchStrategy.LOCAL,
            )
        )
    finally:
        chain.close()
    community = [
        s for s in out.sources if s["metadata"].get("section_type") == "community"
    ]
    assert community, out.sources
    return community


def _two_documents(
    tmp_path: Path, harness: StoreHarness
) -> tuple[Config, Any, _Model, Path, list[str]]:
    """Both documents ingested into ``harness``'s stores; the reports
    written from both."""
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    e2e._on_harness_stores(config, harness)
    model = _Model()
    server = FakeOllama(model, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(config, server, FakeCrossEncoder("reranker"))
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "a.txt").write_text(_KEPT, encoding="utf-8")
    (source / "b.txt").write_text(_DELETED, encoding="utf-8")
    first = e2e._ingest(config, providers, source, tmp_path / "run1")
    assert first.status is PipelineStageStatus.COMPLETED, e2e._errors(first)
    # A report written from both documents states the deleted one's Carrier.
    both = sorted(
        i for i, r in _reports(config, providers).items() if "Carrier" in r["summary"]
    )
    assert both
    assert _record(providers)["outdated_reports"] == []
    return config, providers, model, source, both


@pytest.mark.timeout(600)
def test_a_report_written_from_a_deleted_document_is_written_again(
    tmp_path: Path, store_harness: StoreHarness, caplog: pytest.LogCaptureFixture
) -> None:
    config, providers, model, source, both = _two_documents(tmp_path, store_harness)
    calls = len(model.report_inputs)
    units_before = _unit_ids(config, providers)

    (source / "b.txt").unlink()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        second = e2e._ingest(config, providers, source, tmp_path / "run2")
    assert second.status is PipelineStageStatus.COMPLETED, e2e._errors(second)
    # RV7-4: the removal said the reports stay flagged until a reset, then
    # the run wrote them again. It says what happens, and nothing is left
    # to warn of.
    assert _logged(caplog, logging.INFO, "flagged outdated until the run writes")
    assert not _logged(caplog, logging.WARNING, "outdated")

    after = _reports(config, providers)
    kept = [i for i in both if i in after]
    assert kept, "the report citing a.txt's units too is kept"
    # One model call per report written again, from what remains: the
    # deleted document's Carrier is in neither its input nor the report.
    refreshed = model.report_inputs[calls:]
    assert len(refreshed) == len(kept)
    assert all("Carrier" not in text for text in refreshed)
    for report_id in kept:
        report = after[report_id]
        assert "Carrier" not in report["summary"]
        assert "Vendor" in report["summary"]
        assert REPORT_OUTDATED_FIELD not in report
        assert all(u in _unit_ids(config, providers) for u in report["text_unit_ids"])
    # RV3-I4: the graph store's community vertices no longer cite the
    # deleted document's text units either.
    dead = sorted(units_before - _unit_ids(config, providers))
    assert dead
    graph = providers.build_graph_indexer()
    try:
        assert not graph.find_ids_citing(dead, suffix="default", communities=True)
    finally:
        graph.close()
    record = _record(providers)
    assert record["outdated_reports"] == []
    assert rebuild_reasons(record, config) == []
    metrics = _indexing_metrics(second)
    assert metrics["rebuild_due_reasons"] == 0
    # No outdated report left, and nothing clustered: no drift to report.
    assert "community_drift" not in metrics
    # The calls are the indexing stage's, on the run's meter (the one
    # processing.max_run_* budgets read).
    assert metrics["llm_usage"]["llm_calls"] == len(kept)
    # What a query reads of the report no longer states the deleted text.
    for source_entry in _community_sources(config, providers):
        assert "Carrier" not in source_entry["content"]
        assert REPORT_OUTDATED_FIELD not in source_entry["metadata"]


def _logged(caplog: pytest.LogCaptureFixture, level: int, text: str) -> list[str]:
    """The messages at ``level`` naming ``text``."""
    return [
        r.getMessage()
        for r in caplog.records
        if r.levelno == level and text in r.getMessage()
    ]


def _unit_ids(config: Config, providers: Any) -> set[str]:
    return set(_scan(providers, "text_units"))


@pytest.mark.timeout(600)
def test_a_report_that_cannot_be_written_again_stays_recorded_and_flagged(
    tmp_path: Path, store_harness: StoreHarness, caplog: pytest.LogCaptureFixture
) -> None:
    config, providers, model, source, both = _two_documents(tmp_path, store_harness)
    # The rebuild reason does not wait for a drift recommendation.
    assert not config.indexing.community_compaction.on_drift_recommendation

    (source / "b.txt").unlink()
    model.fail_reports = True
    caplog.clear()
    second = e2e._ingest(config, providers, source, tmp_path / "run2")
    assert second.status is PipelineStageStatus.COMPLETED, e2e._errors(second)

    after = _reports(config, providers)
    kept = sorted(i for i in both if i in after)
    assert kept
    # The run's end warns of what really stays flagged.
    (left,) = _logged(caplog, logging.WARNING, "stay flagged outdated")
    assert left.startswith(f"{len(kept)} community reports in 'default'")
    units = _unit_ids(config, providers)
    for report_id in kept:
        assert after[report_id][REPORT_OUTDATED_FIELD] is True
        # RV3-I5: its lineage is narrowed to what remains, so a filter on the
        # deleted file no longer matches it.
        assert set(after[report_id]["text_unit_ids"]) <= units
        assert after[report_id]["attr_file"] == "a.txt"
    record = _record(providers)
    assert record["outdated_reports"] == kept
    # Every reader of the record names it, the run's metrics too.
    (reason,) = rebuild_reasons(record, config)
    assert reason.settings == ["community_reports"]
    assert reason.remedy is RebuildRemedy.NEXT_RUN
    assert reason.detail.startswith(
        "1 community report was kept"
        if len(kept) == 1
        else f"{len(kept)} community reports were kept"
    )
    metrics = _indexing_metrics(second)
    assert metrics["rebuild_due_reasons"] == 1
    assert metrics["community_rebuild_recommended"] == 1
    assert all(
        s["metadata"].get(REPORT_OUTDATED_FIELD) is True
        for s in _community_sources(config, providers)
    )

    # The next run, with nothing to index, writes them again from the record.
    model.fail_reports = False
    third = e2e._ingest(config, providers, source, tmp_path / "run3")
    assert third.status is PipelineStageStatus.COMPLETED, e2e._errors(third)
    after = _reports(config, providers)
    for report_id in kept:
        assert REPORT_OUTDATED_FIELD not in after[report_id]
        assert "Carrier" not in after[report_id]["summary"]
    assert _record(providers)["outdated_reports"] == []
