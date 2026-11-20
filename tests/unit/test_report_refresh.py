# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``ReportRefresh``: an outdated community report written again from what
the stores hold of its community after a removal (B-2).

The stores hold the state a removal and its commit left: a level-1 community
``p`` over ``c1`` = {Acme, Buyer} and ``c2`` = {Carrier, Depot}, after a
document whose only text unit ``t1`` stated Buyer and the Acme-Buyer edge was
removed (Buyer deleted, Acme stripped to ``t2``). The reports of ``c1`` and
``p`` cite ``t1``; that of ``c2`` does not.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.ingestion.community_detector import CommunityDetector
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion.report_refresh import ReportRefresh
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.community_clustering import (
    community_label,
    community_name,
)
from unified_kg_rag.domain.ingestion.community_hierarchy import (
    ROOT_CUT_DEPTH,
    community_report_id,
)
from unified_kg_rag.domain.ingestion.graph_builder import GraphBuilder
from unified_kg_rag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Config,
    Entity,
    Relationship,
    TextUnit,
)

pytestmark = pytest.mark.unit

_NAMES = ("Acme", "Buyer", "Carrier", "Depot", "Xeno")


class _Reports:
    """The report model: each report names the members its input lists."""

    def __init__(self) -> None:
        self.inputs: list[str] = []
        self.failing: set[str] = set()

    def __call__(self, system: str, human: str) -> str:
        assert "community analysis" in system, system[:80]
        self.inputs.append(human)
        members = [n for n in _NAMES if re.search(rf"\b{n}\b", human)]
        if self.failing & set(members):
            return "no report"
        return (
            f"<community_name>{' '.join(members)}</community_name>"
            f"<summary>Members: {', '.join(members)}.</summary>"
            "<rating>5</rating><rating_explanation>x</rating_explanation>"
            "<findings><finding><summary>s</summary>"
            "<explanation>e</explanation></finding></findings>"
        )


def _entity(name: str, units: list[str], communities: list[str]) -> Entity:
    return Entity(
        id=name.lower(),
        name=name,
        type="ORGANIZATION",
        description=f"{name} trades widgets",
        text_unit_ids=units,
        community_ids=communities,
    )


def _edge(a: str, b: str, units: list[str]) -> Relationship:
    return Relationship(
        id=f"{a}-{b}",
        source_id=a,
        target_id=b,
        source_name=a.title(),
        target_name=b.title(),
        type="WORKS_WITH",
        description=f"{a} works with {b}",
        weight=1.0,
        text_unit_ids=units,
    )


def _report(
    community: str,
    units: list[str],
    level: int,
    *,
    parent: str | None = None,
    children: list[str] | None = None,
    summary: str = "stale",
) -> CommunityReport:
    return CommunityReport(
        id=community_report_id(community),
        community_id=community,
        name=f"report {community}",
        summary=summary,
        full_content=summary,
        rating=5.0,
        text_unit_ids=units,
        level=level,
        depth=1 - level,
        min_cut_depth=1 - level if parent else ROOT_CUT_DEPTH,
        parent_community_id=parent,
        child_community_ids=children or [],
    )


# The surviving graph: Buyer and Acme-Buyer are gone, Acme cites t2 only.
_ENTITIES = [
    _entity("Acme", ["t2"], ["c1"]),
    _entity("Carrier", ["t2", "t3"], ["c2"]),
    _entity("Depot", ["t3"], ["c2"]),
    # A neighbour of Depot outside the hierarchy: its edge counts in
    # Depot's degree (so Carrier-Depot is packed before Acme-Carrier), as
    # in a full build's graph.
    _entity("Xeno", ["t9"], ["c9"]),
]
_EDGES = [
    _edge("acme", "carrier", ["t2"]),
    _edge("carrier", "depot", ["t3"]),
    _edge("depot", "xeno", ["t9"]),
]


def _stores(config: Config, *, child_report: bool = True) -> tuple[Any, Any]:
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    graph.upsert_entities([e.model_copy(deep=True) for e in _ENTITIES])
    graph.upsert_relationships([r.model_copy(deep=True) for r in _EDGES])
    vector.upsert_text_units(
        [
            TextUnit(id=u, text=u, attributes={"document_id": f"doc-{u}"})
            for u in ("t2", "t3", "t9")
        ]
    )
    reports = [
        _report("c1", ["t1", "t2"], 0, parent="p"),
        _report("p", ["t1", "t2", "t3"], 1, children=["c1", "c2"]),
    ]
    if child_report:
        reports.append(
            _report("c2", ["t2", "t3"], 0, parent="p", summary="Carrier, Depot")
        )
    vector.upsert_community_reports(reports)
    return graph, vector


def _refresh(config: Config, graph: Any, vector: Any, model: _Reports) -> Any:
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    providers = Providers(
        config, boto_session=MagicMock(), llm_factory=ScriptedLLMFactory(model)
    )
    return ReportRefresh(manager, providers)


def _config() -> Config:
    return Config()


_OUTDATED = {community_report_id("c1"), community_report_id("p")}


def test_outdated_reports_are_written_again_from_what_remains() -> None:
    config = _config()
    graph, vector = _stores(config)
    model = _Reports()

    done = _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})

    assert done.written == _OUTDATED and not done.kept and not done.deleted
    # One model call per report, the child before its parent.
    assert len(model.inputs) == 2
    reports = {r.community_id: r for r in vector.read_community_reports(_OUTDATED)}
    assert reports["c1"].summary == "Members: Acme."
    # The parent's members are its children's, the unchanged c2's included.
    assert reports["p"].summary == "Members: Acme, Carrier, Depot."
    # Its lineage is its current members', its place in the hierarchy kept.
    assert reports["p"].text_unit_ids == ["t2", "t3"]
    assert reports["p"].document_ids == ["doc-t2", "doc-t3"]
    assert (reports["p"].level, reports["p"].depth) == (1, 0)
    assert reports["p"].min_cut_depth == ROOT_CUT_DEPTH
    assert reports["c1"].min_cut_depth == 1
    assert reports["p"].child_community_ids == ["c1", "c2"]
    assert reports["c1"].parent_community_id == "p"
    # c2 was not outdated: not asked again.
    (c2,) = vector.read_community_reports([community_report_id("c2")])
    assert c2.summary == "Carrier, Depot"


def test_a_written_report_reads_what_a_full_build_would() -> None:
    # The convergence law: the input a refreshed report is asked with is the
    # one a full build's detector packs for the same community over the
    # surviving graph (members, edges, degrees over every stored edge).
    config = _config()
    graph, vector = _stores(config)
    refreshed = _Reports()
    _refresh(config, graph, vector, refreshed)("default", _OUTDATED, {"acme"})

    full = _Reports()
    providers = Providers(
        config, boto_session=MagicMock(), llm_factory=ScriptedLLMFactory(full)
    )
    detector = CommunityDetector(config, show_progress=False, providers=providers)
    detector.report_on(GraphBuilder(list(_ENTITIES), list(_EDGES)).build())

    def community(cid: str, level: int, members: list[str], edges: list[str]):
        label = community_label(level, cid)
        return Community(
            id=cid,
            short_id=label,
            name=community_name(level, label),
            level=str(level),
            parent="p" if level == 0 else "",
            children=["c1", "c2"] if level else [],
            entity_ids=members,
            relationship_ids=edges,
            size=len(members),
        )

    (c2,) = vector.read_community_reports([community_report_id("c2")])
    detector.generate_reports(
        [
            community("c1", 0, ["acme"], []),
            community(
                "p", 1, ["acme", "carrier", "depot"], ["acme-carrier", "carrier-depot"]
            ),
        ],
        sub_reports={"c2": c2},
    )
    # The calls of one level run concurrently: compared as a multiset.
    assert sorted(refreshed.inputs) == sorted(full.inputs)


def test_a_parent_rolls_up_the_stored_report_of_a_child_not_written_again() -> None:
    config = _config()
    # Not every member fits: the parent's context is truncated and rolls its
    # children's reports up (both fit), the stored c2's beside the
    # refreshed c1's.
    config.graph.community_detection.report_generation.max_report_context_tokens = 40
    graph, vector = _stores(config)
    model = _Reports()
    _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})
    parent_input = model.inputs[-1]
    assert "Carrier, Depot" in parent_input
    assert "Members: Acme." in parent_input


def test_a_member_stripped_of_the_reports_units_is_found_through_the_removal() -> None:
    config = _config()
    graph, vector = _stores(config)
    # Acme's only unit the reports cite was removed; it lives on through t8.
    (acme,) = graph.read_entities(["acme"])
    graph.upsert_entities([acme.model_copy(update={"text_unit_ids": ["t8"]})])
    model = _Reports()
    report = community_report_id("c1")

    missed = _refresh(config, graph, vector, model)("default", {report})
    # Without the removal's stripped entities nothing cites c1's units: the
    # community reads as empty and its report would be deleted.
    assert missed.deleted == {report}

    graph, vector = _stores(config)
    (acme,) = graph.read_entities(["acme"])
    graph.upsert_entities([acme.model_copy(update={"text_unit_ids": ["t8"]})])
    found = _refresh(config, graph, vector, _Reports())("default", {report}, {"acme"})
    assert found.written == {report}


def test_a_report_whose_hierarchy_is_unreadable_stays_outdated() -> None:
    config = _config()
    graph, vector = _stores(config, child_report=False)
    model = _Reports()
    done = _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})
    # c2 has no stored report: p's members cannot be told, so p is kept.
    assert done.written == {community_report_id("c1")}
    assert done.kept == {community_report_id("p")}
    assert len(model.inputs) == 1


def test_a_failed_answer_leaves_the_report_outdated() -> None:
    config = _config()
    graph, vector = _stores(config)
    model = _Reports()
    model.failing = {"Carrier"}
    done = _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})
    assert done.written == {community_report_id("c1")}
    assert done.kept == {community_report_id("p")}
    (parent,) = vector.read_community_reports([community_report_id("p")])
    assert parent.summary == "stale"


class _FailingChild(_Reports):
    """The report model, failing c1's report (Acme alone) while ``fail``."""

    def __init__(self, fail: bool) -> None:
        super().__init__()
        self.fail = fail

    def __call__(self, system: str, human: str) -> str:
        members = [n for n in _NAMES if re.search(rf"\b{n}\b", human)]
        if self.fail and members == ["Acme"]:
            self.inputs.append(human)
            return "no report"
        return super().__call__(system, human)


@pytest.mark.parametrize("rolls_up", [True, False])
def test_a_parent_waits_for_a_child_that_failed_and_converges(rolls_up: bool) -> None:
    # RV1-B2: c1's report failed and p was written without its roll-up,
    # unflagged; the next run wrote c1 alone, so p kept the stale input for
    # good. p now waits, flagged, and the next run writes both as a run
    # whose model never failed does.
    def run(
        model: _Reports, stores: tuple[Any, Any] | None = None, ids=_OUTDATED
    ) -> tuple[Any, Any, Any]:
        graph, vector = stores or _stores(config)
        done = _refresh(config, graph, vector, model)("default", ids, {"acme"})
        return done, graph, vector

    config = _config()
    # The parent's context is truncated: it rolls its children's reports up.
    config.graph.community_detection.report_generation.max_report_context_tokens = 40
    config.graph.community_detection.report_generation.enable_sub_community_rollup = (
        rolls_up
    )
    failing = _FailingChild(fail=True)
    first, graph, vector = run(failing)
    if not rolls_up:
        # Without the roll-up p's input reads no child report: it is written.
        assert first.written == {community_report_id("p")}
        assert first.kept == {community_report_id("c1")}
        return
    assert first.written == set()
    assert first.kept == _OUTDATED
    # p was not asked: its input would lack c1's report.
    assert len(failing.inputs) == 1 and "Carrier" not in failing.inputs[0]
    (parent,) = vector.read_community_reports([community_report_id("p")])
    assert parent.summary == "stale"

    recovered = _FailingChild(fail=False)
    second, _, vector = run(recovered, (graph, vector), first.kept)
    assert second.written == _OUTDATED and not second.kept

    one_shot = _FailingChild(fail=False)
    _, _, clean = run(one_shot)
    assert sorted(recovered.inputs) == sorted(one_shot.inputs)
    assert {
        r.community_id: r.summary for r in vector.read_community_reports(_OUTDATED)
    } == {r.community_id: r.summary for r in clean.read_community_reports(_OUTDATED)}


def test_a_community_without_members_left_loses_its_report() -> None:
    config = _config()
    graph, vector = _stores(config)
    graph.delete_by_id(["acme"])
    vector.upsert_community_reports([_report("c1", ["t1"], 0, parent="p")])
    model = _Reports()
    report = community_report_id("c1")
    done = _refresh(config, graph, vector, model)("default", {report})
    assert done.deleted == {report} and not model.inputs
    assert report not in vector.ids("community_reports")


def test_a_written_report_reads_the_stored_claims_about_its_members() -> None:
    config = Config()
    config.processing.claim_extraction.enabled = True
    graph, vector = _stores(config)
    vector.upsert_claims(
        [
            Claim(
                id="claim-1",
                subject_id="acme",
                subject_name="Acme",
                object_id="carrier",
                object_name="Carrier",
                type="CONTRACT",
                status="TRUE",
                description="Acme hired Carrier for the harbour route",
                text_unit_ids=["t2"],
            )
        ]
    )
    model = _Reports()
    report = community_report_id("c1")
    _refresh(config, graph, vector, model)("default", {report}, {"acme"})
    (text,) = model.inputs
    assert "Acme hired Carrier for the harbour route" in text


# --- Failure paths: every one keeps the reports flagged (RV6-10) -----------
#
# A report the refresh could not write again must stay listed as kept: the
# run's build record names it, so the next run tries again. Never written,
# its stored version stays as it was.


def _assert_kept_as_they_were(done: Any, vector: Any, model: _Reports) -> None:
    assert done.kept == _OUTDATED
    assert not done.written and not done.deleted
    assert not model.inputs
    stored = {
        r.community_id: r.summary for r in vector.read_community_reports(_OUTDATED)
    }
    assert stored == {"c1": "stale", "p": "stale"}


def test_a_refresh_that_raises_keeps_every_report(caplog) -> None:
    config = _config()
    graph, vector = _stores(config)
    model = _Reports()

    def fail(*_: Any, **__: Any) -> Any:
        raise RuntimeError("graph store down")

    graph.find_ids_citing = fail
    done = _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})

    _assert_kept_as_they_were(done, vector, model)
    assert any("stay flagged outdated" in r.getMessage() for r in caplog.records)


def test_a_lost_lease_is_raised_not_kept() -> None:
    from unified_kg_rag.shared import NamespaceLeaseError

    config = _config()
    graph, vector = _stores(config)

    def lost(*_: Any, **__: Any) -> Any:
        raise NamespaceLeaseError("lease lost")

    vector.read_community_reports = lost
    with pytest.raises(NamespaceLeaseError):
        _refresh(config, graph, vector, _Reports())("default", _OUTDATED, {"acme"})


def test_nothing_to_refresh_reads_nothing() -> None:
    config = _config()
    graph, vector = _stores(config)
    vector.read_community_reports = MagicMock()
    done = _refresh(config, graph, vector, _Reports())("default", set())
    assert done == ({*()}, {*()}, {*()}, {})
    vector.read_community_reports.assert_not_called()


def test_a_store_that_cannot_read_reports_by_id_keeps_them(caplog) -> None:
    config = _config()
    graph, vector = _stores(config)
    model = _Reports()
    read = vector.read_community_reports
    vector.read_community_reports = lambda ids, suffix=None: None
    done = _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})
    vector.read_community_reports = read

    _assert_kept_as_they_were(done, vector, model)
    assert any(
        "cannot read community reports" in r.getMessage() for r in caplog.records
    )


def test_a_graph_that_cannot_find_what_units_state_keeps_them(caplog) -> None:
    config = _config()
    graph, vector = _stores(config)
    model = _Reports()
    graph.find_ids_citing = lambda units, suffix=None: None
    done = _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})

    _assert_kept_as_they_were(done, vector, model)
    assert any("cannot find what text units" in r.getMessage() for r in caplog.records)


def test_a_report_missing_below_keeps_its_ancestors(caplog) -> None:
    config = _config()
    graph, vector = _stores(config, child_report=False)
    model = _Reports()
    done = _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})
    assert done.kept == {community_report_id("p")}
    assert any("without a stored report" in r.getMessage() for r in caplog.records)


def test_a_store_that_rejects_the_rewrite_keeps_the_reports() -> None:
    from unified_kg_rag.ports.indexer import IndexingStats

    config = _config()
    graph, vector = _stores(config)
    model = _Reports()
    upsert = vector.upsert_community_reports
    # A write that failed without naming its items wrote none.
    vector.upsert_community_reports = lambda reports: IndexingStats(
        total_items=len(reports), failed_items=len(reports)
    )
    done = _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})
    vector.upsert_community_reports = upsert
    assert done.kept == _OUTDATED and not done.written
    assert done.results["vector_refreshed_reports_default"].failed_items == 2

    # One named failure keeps that report only.
    graph, vector = _stores(config)
    upsert = vector.upsert_community_reports

    def reject_parent(reports: list[CommunityReport]) -> Any:
        stats = upsert([r for r in reports if r.community_id != "p"])
        stats.failed_items += 1
        stats.failed_ids.append(community_report_id("p"))
        return stats

    vector.upsert_community_reports = reject_parent
    done = _refresh(config, graph, vector, _Reports())("default", _OUTDATED, {"acme"})
    assert done.written == {community_report_id("c1")}
    assert done.kept == {community_report_id("p")}


def test_an_emptied_report_whose_delete_failed_stays_kept() -> None:
    from unified_kg_rag.ports.indexer import IndexingStats

    config = _config()
    graph, vector = _stores(config)
    graph.delete_by_id(["acme"])
    report = community_report_id("c1")
    vector.upsert_community_reports([_report("c1", ["t1"], 0, parent="p")])
    vector.delete_by_id = lambda ids, prefix, suffix=None: IndexingStats(
        total_items=len(ids), failed_items=len(ids), failed_ids=list(ids)
    )
    done = _refresh(config, graph, vector, _Reports())("default", {report})
    assert done.kept == {report} and not done.deleted
