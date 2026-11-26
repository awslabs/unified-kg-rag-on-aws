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

import json
import logging
import re
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
from unified_kg_rag.domain.ingestion.rebuild_due import (
    RebuildReasonKind,
    RebuildRemedy,
    rebuild_reasons,
)
from unified_kg_rag.domain.ingestion.report_references import REPORT_OUTDATED_FIELD
from unified_kg_rag.domain.models import (
    Config,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
    SearchStrategy,
)
from unified_kg_rag.domain.models.config import SeedGraphConfig
from unified_kg_rag.ports.indexer import GraphIndexer

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

_KEPT = "Vendor supplies widgets to Buyer. Depot stores widgets for Vendor."
# Carrier is named only here: what the deletion removes.
_DELETED = "Carrier moves widgets for Vendor and Buyer every week."
# The last four: a second group, for corpora of two communities
# (test_ingestion_crash).
_NAMES = ("Vendor", "Buyer", "Depot", "Carrier", "Alpha", "Beta", "Gamma", "Omega")


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


# --- A hierarchy whose children a removal deletes or stores without a report
#
# Four triangles of entities, each a level-0 community, grouped in pairs at
# level 1 by the edges between them (arch/final-B b7_stuck, B-1): the full
# build's hierarchy over P(A, B) and P(C, D).
_GROUPS = {
    "A": ("Alder", "Aspen", "Acacia"),
    "B": ("Birch", "Beech", "Baobab"),
    "C": ("Cedar", "Cypress", "Chestnut"),
    "D": ("Dogwood", "Damson", "Deodar"),
}
# The seed graph's triangle (test_a_parent_over_a_seed_only_child_...).
_SEEDED = ("Sequoia", "Spruce", "Sumac")
_TREES = (*(name for names in _GROUPS.values() for name in names), *_SEEDED)
_PAIR = re.compile(r"(\w+) works with (\w+)")


def _triangle(group: str) -> str:
    a, b, c = _GROUPS[group]
    return f"{a} works with {b}. {b} works with {c}. {c} works with {a}. "


class _PairModel(_Model):
    """The fake model, extracting each "X works with Y" of a text unit."""

    def __call__(self, system: str, human: str) -> str:
        if "knowledge graph extraction expert" in system:
            text = stages._source_text(human)
            pairs = _PAIR.findall(text)
            names = list(dict.fromkeys(n for pair in pairs for n in pair))
            return (
                "<entities>"
                + "".join(
                    f"<entity><name>{n}</name><type>ORGANIZATION</type>"
                    f"<description>{n} in the supply chain</description>"
                    f"<confidence>9</confidence><source_text>{n}</source_text>"
                    "</entity>"
                    for n in names
                )
                + "</entities><relationships>"
                + "".join(
                    f"<relationship><source>{a}</source><target>{b}</target>"
                    f"<type>WORKS_WITH</type><description>{a} works with {b}"
                    f"</description><strength>8</strength><source_text>{a} works "
                    f"with {b}</source_text></relationship>"
                    for a, b in pairs
                )
                + "</relationships>"
            )
        if "community analysis" in system:
            self.report_inputs.append(human)
            members = ", ".join(n for n in _TREES if re.search(rf"\b{n}\b", human))
            return (
                "<community_name>Trees</community_name>"
                f"<summary>Members: {members}.</summary>"
                "<rating>5.0</rating><rating_explanation>x</rating_explanation>"
                f"<findings><finding><summary>{members}</summary>"
                "<explanation>They work together.</explanation></finding>"
                "</findings>"
            )
        return super().__call__(system, human)


def _forest(
    tmp_path: Path,
    harness: StoreHarness,
    files: dict[str, str],
    seed: list[dict[str, str]] | None = None,
) -> tuple[Config, Any, Path]:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    e2e._on_harness_stores(config, harness)
    if seed:
        path = tmp_path / "seed" / "trees.jsonl"
        path.parent.mkdir()
        path.write_text("".join(json.dumps(r) + "\n" for r in seed), encoding="utf-8")
        config.processing.seed_graphs = [
            SeedGraphConfig(name="trees", kind="jsonl", path=str(path))
        ]
    # One text unit per document: what a document states is its unit's.
    config.processing.chunking.fallback_chunk_size = 2000
    config.processing.chunking.max_chunk_size = 4000
    server = FakeOllama(_PairModel(), e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(config, server, FakeCrossEncoder("reranker"))
    source = tmp_path / "corpus"
    source.mkdir()
    for name, text in files.items():
        (source / name).write_text(text, encoding="utf-8")
    first = e2e._ingest(config, providers, source, tmp_path / "run1")
    assert first.status is PipelineStageStatus.COMPLETED, e2e._errors(first)
    return config, providers, source


def _level1(reports: dict[str, dict]) -> dict[str, dict]:
    """The level-1 reports, by the trees they state."""
    return {r["summary"]: r for r in reports.values() if str(r.get("level")) == "1"}


def _assert_current(
    config: Config, providers: Any, caplog: pytest.LogCaptureFixture, dead: set[str]
) -> dict[str, dict]:
    """No report outdated, warned of or listed, none citing a text unit not
    stored, and no community vertex citing a ``dead`` one; the reports."""
    assert dead
    assert not _logged(caplog, logging.WARNING, "outdated")
    reports = _reports(config, providers)
    units = _unit_ids(config, providers)
    assert not dead & units
    for report in reports.values():
        assert REPORT_OUTDATED_FIELD not in report
        assert set(report["text_unit_ids"]) <= units
    assert _record(providers)["outdated_reports"] == []
    graph = providers.build_graph_indexer()
    try:
        assert not graph.find_ids_citing(
            sorted(dead), suffix="default", communities=True
        )
    finally:
        graph.close()
    return reports


@pytest.mark.timeout(600)
def test_a_parent_whose_child_the_removal_deleted_is_written_again(
    tmp_path: Path, store_harness: StoreHarness, caplog: pytest.LogCaptureFixture
) -> None:
    # B-1: deleting a.txt deletes community A (its every text unit), its
    # report and vertex. P(A, B) cites a.txt's unit too: outdated, it
    # listed A, and every run kept it ("a community below them without a
    # stored report"), stating the deleted Alder, until a reset.
    a, b = _GROUPS["A"], _GROUPS["B"]
    files = {
        "a.txt": _triangle("A")
        + f"{a[0]} works with {b[0]}. {a[1]} works with {b[1]}.",
        "b.txt": _triangle("B") + f"{b[2]} works with {_GROUPS['C'][2]}.",
        "c.txt": _triangle("C")
        + f"{_GROUPS['C'][0]} works with {_GROUPS['D'][0]}. "
        + f"{_GROUPS['C'][1]} works with {_GROUPS['D'][1]}.",
        "d.txt": _triangle("D"),
    }
    config, providers, source = _forest(tmp_path, store_harness, files)
    before = _level1(_reports(config, providers))
    (parent,) = [r for s, r in before.items() if "Alder" in s]
    assert len(parent["child_community_ids"]) == 2
    units = _unit_ids(config, providers)

    (source / "a.txt").unlink()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        second = e2e._ingest(config, providers, source, tmp_path / "run2")
    assert second.status is PipelineStageStatus.COMPLETED, e2e._errors(second)
    reports = _assert_current(
        config, providers, caplog, units - _unit_ids(config, providers)
    )
    written = reports[parent["id"]]
    assert written["summary"] == "Members: Birch, Beech, Baobab."
    # The deleted child is no longer listed; the one left is.
    (child,) = written["child_community_ids"]
    assert child in {r["community_id"] for r in reports.values()}
    assert not any("Alder" in r["summary"] for r in reports.values())


def _seeded_forest(
    tmp_path: Path, harness: StoreHarness
) -> tuple[Config, Any, Path, dict, str]:
    """The seed graph's triangle S, a community of its own stored without a
    report (no document states it), under P(A, S), which lists it: the
    config, providers, corpus, P's report and S's id."""
    a = _GROUPS["A"]
    seeded = _SEEDED
    nodes = [
        {"kind": "node", "id": f"n-{n.lower()}", "name": n, "type": "ORGANIZATION"}
        for n in (*seeded, a[0], a[1])
    ]
    edges = [
        {
            "kind": "edge",
            "source": f"n-{x.lower()}",
            "target": f"n-{y.lower()}",
            "type": "WORKS_WITH",
        }
        for x, y in [
            (seeded[0], seeded[1]),
            (seeded[1], seeded[2]),
            (seeded[2], seeded[0]),
            (seeded[0], a[0]),
            (seeded[1], a[1]),
        ]
    ]
    b, d = _GROUPS["B"], _GROUPS["D"]
    files = {
        "a1.txt": _triangle("A") + f"{a[2]} works with {b[2]}.",
        "a2.txt": f"{a[0]} works with {a[1]}.",
        "b.txt": _triangle("B")
        + f"{b[0]} works with {d[0]}. {b[1]} works with {d[1]}.",
        "d.txt": _triangle("D"),
    }
    config, providers, source = _forest(tmp_path, harness, files, nodes + edges)
    before = _reports(config, providers)
    (parent,) = [
        r for s, r in _level1(before).items() if "Alder" in s and "Sequoia" in s
    ]
    (reportless,) = set(parent["child_community_ids"]) - {
        r["community_id"] for r in before.values()
    }
    return config, providers, source, parent, reportless


@pytest.mark.timeout(600)
def test_a_parent_over_a_seed_only_child_is_written_again(
    tmp_path: Path, store_harness: StoreHarness, caplog: pytest.LogCaptureFixture
) -> None:
    # B-1, from a fresh full build: deleting a2.txt outdates P; it was kept
    # for good. A full build writes P from A's members and S's: so does the
    # refresh.
    config, providers, source, parent, reportless = _seeded_forest(
        tmp_path, store_harness
    )
    units = _unit_ids(config, providers)

    (source / "a2.txt").unlink()
    caplog.clear()
    with caplog.at_level(logging.INFO):
        second = e2e._ingest(config, providers, source, tmp_path / "run2")
    assert second.status is PipelineStageStatus.COMPLETED, e2e._errors(second)
    reports = _assert_current(
        config, providers, caplog, units - _unit_ids(config, providers)
    )
    written = reports[parent["id"]]
    assert all(n in written["summary"] for n in (*_GROUPS["A"], *_SEEDED)), written[
        "summary"
    ]
    assert reportless in written["child_community_ids"]


@pytest.mark.timeout(600)
def test_a_parent_a_backend_cannot_read_below_says_only_a_reset_writes_it(
    tmp_path: Path,
    store_harness: StoreHarness,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # r8diff-2: a custom graph backend that cannot read community vertices
    # (GraphIndexer.read_communities and read_community_children at the
    # port's defaults; the latter reads through the former) keeps P
    # outdated until a reset (the maintainer's B-1 decision). Every surface
    # names that remedy: one WARNING (not also "the next run writes them
    # again"), the build record's reset_only_reports, and the rebuild
    # reason run-describe shows (remedy reset, not next_run), run after run.
    config, providers, source, parent, _ = _seeded_forest(tmp_path, store_harness)
    graph = providers.build_graph_indexer()
    try:
        backend = type(graph)
    finally:
        graph.close()
    monkeypatch.setattr(backend, "read_communities", GraphIndexer.read_communities)
    monkeypatch.setattr(
        backend, "read_community_children", GraphIndexer.read_community_children
    )
    (source / "a2.txt").unlink()
    for run in ("run2", "run3"):
        caplog.clear()
        with caplog.at_level(logging.INFO):
            done = e2e._ingest(config, providers, source, tmp_path / run)
        assert done.status is PipelineStageStatus.COMPLETED, e2e._errors(done)
        assert REPORT_OUTDATED_FIELD in _reports(config, providers)[parent["id"]]
        warned = _logged(caplog, logging.WARNING, "outdated")
        (kept,) = [w for w in warned if "due for a rebuild" not in w]
        assert "indexing.reset: true writes them again" in kept
        assert not [w for w in warned if "the next run writes" in w], warned
        # The rebuild-due WARNING of the run that writes the record carries
        # its reason.
        due = [w for w in warned if "due for a rebuild" in w]
        assert due or run == "run3"
        assert all("no incremental run writes it" in w for w in due), due
        record = _record(providers)
        assert record["outdated_reports"] == [parent["id"]]
        assert record["reset_only_reports"] == [parent["id"]]
        (reason,) = rebuild_reasons(record, config)
        assert reason.kind is RebuildReasonKind.OUTDATED_REPORTS
        assert reason.remedy is RebuildRemedy.RESET
        assert "no incremental run writes it" in reason.detail


def _community_vertices(providers: Any, ids: list[str]) -> dict[str, Any]:
    graph = providers.build_graph_indexer()
    try:
        communities = graph.read_communities(ids, suffix="default")
    finally:
        graph.close()
    assert communities is not None
    return {c.id: c for c in communities}


@pytest.mark.timeout(600)
@pytest.mark.parametrize("linked", [True, False])
def test_a_seed_only_community_edited_in_one_shard_keeps_no_removed_unit(
    tmp_path: Path,
    store_harness: StoreHarness,
    caplog: pytest.LogCaptureFixture,
    linked: bool,
) -> None:
    # r8diff-6: the seed graph's triangle S is a community of its own,
    # stored without a report. Editing Sequoia's description re-keys the
    # units of its shard only (Spruce and Sumac are in another): S keeps
    # the units of the shard left, and its vertex kept the re-keyed ones
    # too, which nothing stores any more. Its vertex is written again
    # from its members, as a refreshed report's is. r9diff-3: with no link
    # to a document entity (``linked`` False) the removal flags no report
    # at all, only S: the commit refreshes S alone.
    a = _GROUPS["A"]
    seeded = _SEEDED

    def seed(sequoia: str) -> list[dict[str, str]]:
        nodes = [
            {"kind": "node", "id": f"n-{n.lower()}", "name": n, "type": "ORGANIZATION"}
            for n in (*seeded, *([a[0]] if linked else []))
        ]
        nodes[0]["description"] = sequoia
        pairs = [
            (seeded[0], seeded[1]),
            (seeded[1], seeded[2]),
            (seeded[2], seeded[0]),
            *([(seeded[0], a[0])] if linked else []),
        ]
        return nodes + [
            {
                "kind": "edge",
                "source": f"n-{x.lower()}",
                "target": f"n-{y.lower()}",
                "type": "WORKS_WITH",
            }
            for x, y in pairs
        ]

    files = {"a.txt": _triangle("A"), "b.txt": _triangle("B")}
    config, providers, source = _forest(
        tmp_path, store_harness, files, seed("A tall tree.")
    )
    reports = {r["community_id"] for r in _reports(config, providers).values()}
    graph = providers.build_graph_indexer()
    try:
        units = _unit_ids(config, providers)
        stored = graph.find_ids_citing(
            sorted(units), suffix="default", communities=True
        )
    finally:
        graph.close()
    (community,) = [
        c
        for c, cited in stored.items()
        if c not in reports and all(u.startswith("sg-") for u in cited)
    ]

    (tmp_path / "seed" / "trees.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in seed("A taller tree.")),
        encoding="utf-8",
    )
    caplog.clear()
    with caplog.at_level(logging.INFO):
        second = e2e._ingest(config, providers, source, tmp_path / "run2")
    assert second.status is PipelineStageStatus.COMPLETED, e2e._errors(second)
    flagged = _logged(caplog, logging.INFO, "flagged outdated until")
    assert bool(flagged) is linked, flagged
    left = _unit_ids(config, providers)
    dead = units - left
    _assert_current(config, providers, caplog, dead)
    (vertex,) = _community_vertices(providers, [community]).values()
    assert set(vertex.text_unit_ids or []) <= left
    assert vertex.size == len(seeded)
