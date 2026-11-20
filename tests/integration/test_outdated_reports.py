# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A community report written partly from a deleted document, end to end on
the embedded stores (``backend: local``, no container, no AWS).

The report of a community over two documents cites both. Deleting one keeps
the report for the other's text units: until a run writes it again it may
state the deleted text, so the build record lists it, every reader of the
record reports a rebuild reason for it whatever the drift settings, and a
query that reads it gets it flagged ``outdated``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.rerank import FakeCrossEncoder
from tests.integration import test_embedded_e2e as e2e
from tests.integration import test_ingestion_stages as stages
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401
from unified_kg_rag.application.retrieval.rag_chain import GraphRAGChain, RAGInput
from unified_kg_rag.domain.ingestion.rebuild_due import rebuild_reasons
from unified_kg_rag.domain.ingestion.report_references import REPORT_OUTDATED_FIELD
from unified_kg_rag.domain.models import (
    PipelineStageStatus,
    PipelineStageType,
    SearchStrategy,
)

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

_KEPT = "Vendor supplies widgets to Buyer. Depot stores widgets for Vendor."
# Carrier is named only here: what the deletion removes.
_DELETED = "Carrier moves widgets for Vendor and Buyer every week."
_NAMES = ("Vendor", "Buyer", "Depot", "Carrier")


class _Model:
    """The fake model; a report names the members its input lists, so what
    a report states is what its input held."""

    def __init__(self) -> None:
        self.report_inputs: list[str] = []

    def __call__(self, system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system:
            return stages._extraction(stages._source_text(human), _NAMES)
        if "community analysis" in system:
            self.report_inputs.append(human)
            members = ", ".join(n for n in _NAMES if n in human)
            return (
                "<community_name>Widget partners</community_name>"
                f"<summary>Members: {members}.</summary>"
                "<rating>5.0</rating><rating_explanation>Small.</rating_explanation>"
                f"<findings><finding><summary>{members}</summary>"
                "<explanation>They trade widgets.</explanation></finding></findings>"
            )
        return e2e._respond(system, human)


def _reports(config, providers) -> dict[str, dict]:
    prefix = config.indexing.opensearch.community_reports_index_prefix
    return {h["_id"]: h["_source"] for h in e2e._stored(config, providers, prefix)}


def _record(providers) -> dict:
    vectors = providers.build_vector_indexer()
    try:
        record = vectors.read_build_record("default")
    finally:
        vectors.close()
    assert record is not None
    return record


def _indexing_metrics(context) -> dict:
    (result,) = [
        r
        for r in context.stage_results
        if r.stage_name == PipelineStageType.INDEXING.value
    ]
    return result.metrics or {}


@pytest.mark.timeout(600)
def test_a_report_kept_after_a_deletion_is_recorded_and_flagged_outdated(
    tmp_path: Path,
) -> None:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    # The rebuild reason does not wait for a drift recommendation.
    assert not config.indexing.community_compaction.on_drift_recommendation
    model = _Model()
    server = FakeOllama(model, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(config, server, FakeCrossEncoder("reranker"))
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "a.txt").write_text(_KEPT, encoding="utf-8")
    (source / "b.txt").write_text(_DELETED, encoding="utf-8")
    first = e2e._ingest(config, providers, source, tmp_path / "run1")
    assert first.status is PipelineStageStatus.COMPLETED, e2e._errors(first)
    before = _reports(config, providers)
    # A report written from both documents states the deleted one's Carrier.
    both = [i for i, r in before.items() if "Carrier" in r["summary"]]
    assert both, before
    assert _record(providers)["outdated_reports"] == []

    (source / "b.txt").unlink()
    second = e2e._ingest(config, providers, source, tmp_path / "run2")
    assert second.status is PipelineStageStatus.COMPLETED, e2e._errors(second)

    after = _reports(config, providers)
    kept = sorted(i for i in both if i in after)
    assert kept, "the report citing a.txt's units too is kept"
    for report_id in kept:
        assert after[report_id][REPORT_OUTDATED_FIELD] is True
    record = _record(providers)
    assert record["outdated_reports"] == kept
    # Every reader of the record names it, the run's metrics too.
    reasons = rebuild_reasons(record, config)
    assert [r.settings for r in reasons] == [["community_reports"]]
    assert f"{len(kept)} community reports were kept" in reasons[0].detail
    metrics = _indexing_metrics(second)
    assert metrics["rebuild_due_reasons"] == 1
    assert metrics["community_rebuild_recommended"] == 1

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
        s["metadata"]
        for s in out.sources
        if s["metadata"].get("section_type") == "community"
    ]
    assert community, out.sources
    assert all(m.get(REPORT_OUTDATED_FIELD) is True for m in community)
