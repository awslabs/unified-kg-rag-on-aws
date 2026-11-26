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
from unified_kg_rag.ports.indexer import IndexingStats, VectorIndexer

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


def _vertex(
    community: str, units: list[str], level: int, children: list[str] = ()
) -> Community:
    """The graph store's vertex of ``community``, as the full build wrote it."""
    label = community_label(level, community)
    return Community(
        id=community,
        short_id=label,
        name=community_name(level, label),
        level=str(level),
        parent="p" if level == 0 else "",
        children=list(children),
        text_unit_ids=units,
    )


# The vertices of c1, c2 and p, as the full build before the removal wrote
# them: a community is stored whatever its report.
_VERTICES = [
    _vertex("c1", ["t1", "t2"], 0),
    _vertex("c2", ["t2", "t3"], 0),
    _vertex("p", ["t1", "t2", "t3"], 1, ["c1", "c2"]),
]


def _stores(
    config: Config,
    *,
    child_report: bool = True,
    vector_class: type[FakeVectorStore] = FakeVectorStore,
) -> tuple[Any, Any]:
    graph = FakeGraphStore()
    vector = vector_class(opensearch_config=config.indexing.opensearch)
    graph.upsert_entities([e.model_copy(deep=True) for e in _ENTITIES])
    graph.upsert_relationships([r.model_copy(deep=True) for r in _EDGES])
    graph.upsert_communities([c.model_copy(deep=True) for c in _VERTICES])
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
    # RV3-I4: the graph store's community vertices are written again with
    # the members they now have, as a full build's detector makes them.
    vertices = graph.data["communities"]
    c1, p = vertices[("default", "c1")], vertices[("default", "p")]
    assert (c1.entity_ids, c1.text_unit_ids, c1.size) == (["acme"], ["t2"], 1)
    assert c1.attributes["entity_names"] == ["Acme"]
    assert c1.attributes["num_relationships"] == 0
    assert p.text_unit_ids == ["t2", "t3"]
    assert p.attributes["entity_names"] == ["Acme", "Carrier", "Depot"]
    assert p.attributes["num_relationships"] == 2
    assert p.children == ["c1", "c2"]
    # c2 was not outdated: its vertex is as the full build wrote it.
    assert vertices[("default", "c2")].text_unit_ids == ["t2", "t3"]
    assert vertices[("default", "c2")].attributes is None


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


def _full_build_inputs(
    config: Config, communities: list[Community], sub_reports: dict[str, Any]
) -> list[str]:
    """The report inputs a full build's detector packs for ``communities``
    over the surviving graph (``_ENTITIES``, ``_EDGES``)."""
    full = _Reports()
    providers = Providers(
        config, boto_session=MagicMock(), llm_factory=ScriptedLLMFactory(full)
    )
    detector = CommunityDetector(config, show_progress=False, providers=providers)
    detector.report_on(GraphBuilder(list(_ENTITIES), list(_EDGES)).build())
    detector.generate_reports(communities, sub_reports=sub_reports)
    return full.inputs


def _parent(children: list[str]) -> Community:
    label = community_label(1, "p")
    return Community(
        id="p",
        short_id=label,
        name=community_name(1, label),
        level="1",
        parent="",
        children=children,
        entity_ids=["acme", "carrier", "depot"],
        relationship_ids=["acme-carrier", "carrier-depot"],
        size=3,
    )


def test_a_child_stored_without_a_report_counts_through_its_members() -> None:
    # B-1: c2 is stored without a report (a full build stores a seed-only
    # community, or one whose report failed, so); p lists it, as
    # attach_report_hierarchy lists every child. p was kept outdated for
    # good ("a community below them without a stored report"). A full
    # build packs p from all its members, a child without a report
    # substituting nothing: so does the refresh.
    config = _config()
    graph, vector = _stores(config, child_report=False)
    model = _Reports()
    done = _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})
    assert done.written == _OUTDATED and not done.kept
    reports = {r.community_id: r for r in vector.read_community_reports(_OUTDATED)}
    assert reports["p"].summary == "Members: Acme, Carrier, Depot."
    # c2 is still stored: p still lists it.
    assert reports["p"].child_community_ids == ["c1", "c2"]
    parent_input = model.inputs[-1]
    assert parent_input in _full_build_inputs(config, [_parent(["c1", "c2"])], {})


def test_a_reportless_child_reads_its_level0_communities_through_its_vertex() -> None:
    # A child without a report that is not a level-0 community (a seed-only
    # one over level-0 seed-only ones): its members name the communities
    # below it, which only its graph vertex lists.
    config = _config()
    graph, vector = _stores(config, child_report=False)
    graph.upsert_communities([_vertex("m", ["t2", "t3"], 1, ["c2"])])
    vector.upsert_community_reports(
        [_report("p", ["t1", "t2", "t3"], 2, children=["c1", "m"])]
    )
    done = _refresh(config, graph, vector, _Reports())("default", _OUTDATED, {"acme"})
    assert done.written == _OUTDATED
    (parent,) = vector.read_community_reports([community_report_id("p")])
    assert parent.summary == "Members: Acme, Carrier, Depot."
    assert parent.child_community_ids == ["c1", "m"]


@pytest.mark.parametrize("unread", ["read_community_children", "read_communities"])
def test_a_child_the_graph_store_cannot_read_keeps_its_parent(
    caplog, unread: str
) -> None:
    # A backend that cannot tell a community's children (the port's
    # default; the fake's read_community_children, as the port's, reads
    # through read_communities, so a backend without either cannot walk):
    # p's members cannot be told, so p is kept, and the warning names the
    # remedy that writes it.
    config = _config()
    graph, vector = _stores(config, child_report=False)
    setattr(graph, unread, lambda ids, suffix=None: None)
    model = _Reports()
    done = _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})
    assert done.written == {community_report_id("c1")}
    assert done.kept == {community_report_id("p")}
    # Only a reset writes it (r8diff-2: the build record says so).
    assert done.reset_only == {community_report_id("p")}
    assert len(model.inputs) == 1
    (warning,) = [
        r.getMessage() for r in caplog.records if "cannot read" in r.getMessage()
    ]
    assert "indexing.reset: true" in warning


def test_a_child_the_removal_deleted_leaves_its_parent_written() -> None:
    # B-1 (arch/final-B b7_stuck): a removal deleted c2 (none of its text
    # units left: its report, its vertex, and its id from its members'
    # community_ids). p, outdated, still listed it and was kept for good.
    # c2 holds no member: p is written from c1's, and no longer lists it.
    config = _config()
    graph, vector = _stores(config, child_report=False)
    graph.delete_by_id(["c2", "carrier", "depot", "carrier-depot", "acme-carrier"])
    model = _Reports()
    done = _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})
    assert done.written == _OUTDATED and not done.kept
    reports = {r.community_id: r for r in vector.read_community_reports(_OUTDATED)}
    assert reports["p"].summary == "Members: Acme."
    assert reports["p"].text_unit_ids == ["t2"]
    assert reports["p"].child_community_ids == ["c1"]
    vertex = graph.data["communities"][("default", "p")]
    assert (vertex.children, vertex.text_unit_ids) == (["c1"], ["t2"])
    # A later refresh of p reads no deleted child.
    vector.mark_reports_outdated([community_report_id("p")], "default")
    again = _refresh(config, graph, vector, _Reports())(
        "default", {community_report_id("p")}
    )
    assert again.written == {community_report_id("p")}


def test_a_seed_only_child_is_emptied_and_its_parent_written_again() -> None:
    # G-2: c2's members are stated by a seed graph only. A full build has
    # no report for such a community: its report is deleted (M27, the
    # seed_only check at the emptying, survived every test). Its community
    # stays, so a later refresh of p reads its members through its vertex,
    # and writes p as a full build packs it (B-1's second producer).
    config = _config()
    graph, vector = _stores(config)
    seed = "sg-1"
    graph.upsert_entities(
        [_entity("Carrier", [seed], ["c2"]), _entity("Depot", [seed], ["c2"])]
    )
    graph.upsert_relationships(
        [_edge("carrier", "depot", [seed]), _edge("acme", "carrier", ["t2"])]
    )
    vector.upsert_text_units([TextUnit(id=seed, text="s", attributes={})])
    c2 = community_report_id("c2")
    model = _Reports()
    outdated = {*_OUTDATED, c2}
    first = _refresh(config, graph, vector, model)(
        "default", outdated, {"acme", "carrier", "depot"}
    )
    assert first.deleted == {c2}
    assert first.written == _OUTDATED
    assert c2 not in vector.ids("community_reports")
    # Its vertex is written again from its members, as a refreshed one.
    vertex = graph.data["communities"][("default", "c2")]
    assert (vertex.text_unit_ids, vertex.size) == ([seed], 2)
    (parent,) = vector.read_community_reports([community_report_id("p")])
    assert parent.summary == "Members: Acme, Carrier, Depot."
    assert parent.child_community_ids == ["c1", "c2"]

    p = community_report_id("p")
    second = _refresh(config, graph, vector, _Reports())("default", {p}, {"acme"})
    assert second.written == {p} and not second.kept


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
    # A failure the next run may not meet: it tries again.
    assert not done.reset_only


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
    assert done == ({*()}, {*()}, {*()}, {}, {*()})
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
    # The backend cannot: no incremental run writes them (r8diff-2).
    assert done.reset_only == _OUTDATED


def test_a_graph_that_cannot_find_what_units_state_keeps_them(caplog) -> None:
    config = _config()
    graph, vector = _stores(config)
    model = _Reports()
    graph.find_ids_citing = lambda units, suffix=None: None
    done = _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})

    _assert_kept_as_they_were(done, vector, model)
    (warning,) = [
        r.getMessage()
        for r in caplog.records
        if "cannot find what text units" in r.getMessage()
    ]
    assert "until indexing.reset: true" in warning
    assert done.reset_only == _OUTDATED


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
    assert not done.reset_only


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


# --- The outdated-report note (RV3-I1) --------------------------------------


def _removal_stores() -> tuple[Config, IndexingManager, Any]:
    config = _config()
    graph, vector = _stores(config)
    vector.upsert_text_units([TextUnit(id="t1", text="t1")])
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    return config, manager, vector


def test_a_removal_notes_the_reports_it_flags_before_flagging_them(mocker) -> None:
    from unified_kg_rag.application.storage.stored_items import OUTDATED_REPORTS

    _, manager, vector = _removal_stores()
    vector.write_extraction_records(
        {OUTDATED_REPORTS: {"report_ids": ["r-earlier"], "candidates": ["e-earlier"]}}
    )
    flag = mocker.spy(vector, "mark_reports_outdated")
    order: list[str] = []
    write = vector.write_extraction_records
    mocker.patch.object(
        vector,
        "write_extraction_records",
        side_effect=lambda records, **kw: order.append("note") or write(records, **kw),
    )
    flag.side_effect = lambda *a, **k: order.append("flag") or IndexingStats()

    manager.remove_text_units("default", ["t1"])

    assert order[:2] == ["note", "flag"]
    note = vector.read_extraction_records([OUTDATED_REPORTS])[OUTDATED_REPORTS]
    # The earlier note's ids are kept beside this removal's.
    assert note["report_ids"] == sorted({"r-earlier", *_OUTDATED})
    assert "e-earlier" in note["candidates"]
    assert manager.read_outdated_note("default")[0] == {"r-earlier", *_OUTDATED}


def test_a_note_that_cannot_be_written_fails_the_removal(mocker) -> None:
    _, manager, vector = _removal_stores()
    failed = IndexingStats(total_items=1)
    failed.add_error("store down")
    mocker.patch.object(vector, "write_extraction_records", return_value=failed)
    flag = mocker.spy(vector, "mark_reports_outdated")

    results = manager.remove_text_units("default", ["t1"])

    # Reported failed (the caller keeps the documents' records and retries),
    # nothing flagged or stripped.
    assert results["vector_outdated_reports_default"].failed_items == len(_OUTDATED)
    flag.assert_not_called()
    assert not manager.outdated_reports


def test_only_a_note_the_run_read_or_wrote_is_cleared(mocker) -> None:
    from unified_kg_rag.application.storage.stored_items import OUTDATED_REPORTS

    _, manager, vector = _removal_stores()
    delete = mocker.spy(vector, "delete_extraction_records")
    manager.clear_outdated_note("default")
    delete.assert_not_called()

    vector.write_extraction_records({OUTDATED_REPORTS: {"report_ids": ["r1"]}})
    assert manager.read_outdated_note("default") == ({"r1"}, set(), set())
    manager.clear_outdated_note("default")
    assert OUTDATED_REPORTS not in vector.read_extraction_records([OUTDATED_REPORTS])
    # A run starts knowing of none.
    manager._outdated_notes.add("default")
    manager.reset_run_state()
    assert not manager._outdated_notes


def test_a_note_merges_into_the_runs_outdated_reports(mocker) -> None:
    from unified_kg_rag.application.storage.stored_items import OUTDATED_REPORTS

    _, manager, vector = _removal_stores()
    vector.write_extraction_records(
        {OUTDATED_REPORTS: {"report_ids": ["r1"], "candidates": ["e1"]}}
    )
    manager.outdated_reports["default"] = {"r0"}
    assert manager.merge_outdated_note("default") == {"e1"}
    assert manager.outdated_reports["default"] == {"r0", "r1"}

    # A note this run read and a commit that wrote no report keeps.
    manager.keep_outdated_notes()
    manager.clear_outdated_note("default")
    assert OUTDATED_REPORTS in vector.read_extraction_records([OUTDATED_REPORTS])

    # A note that cannot be read is kept too, though this run wrote it.
    manager.note_outdated_reports("default", ["r2"], [])
    mocker.patch.object(
        vector, "read_extraction_records", side_effect=RuntimeError("store down")
    )
    with pytest.raises(RuntimeError):
        manager.merge_outdated_note("default")
    delete = mocker.spy(vector, "delete_extraction_records")
    manager.clear_outdated_note("default")
    delete.assert_not_called()


class _CannotReadUnits(FakeVectorStore):
    """A custom store that leaves ``read_text_unit_attributes`` at the
    port's default (``{}``), whatever else it implements."""

    read_text_unit_attributes = VectorIndexer.read_text_unit_attributes


_STORED_LINEAGE = {
    "c1": (["doc-t1", "doc-t2"], {"file": ["t1.txt", "t2.txt"]}),
    "p": (["doc-t1", "doc-t2", "doc-t3"], {"file": ["t1.txt", "t2.txt", "t3.txt"]}),
}


def _lineage_stores(vector_class: type[FakeVectorStore]) -> tuple[Config, Any, Any]:
    """The stores, each outdated report written with the documents and
    document filters a full build gave it (``attach_report_lineage``)."""
    config = _config()
    graph, vector = _stores(config, vector_class=vector_class)
    vector.upsert_text_units(
        [
            TextUnit(
                id=u,
                text=u,
                attributes={"document_id": f"doc-{u}", "filters": {"file": f"{u}.txt"}},
            )
            for u in ("t1", "t2", "t3", "t9")
        ]
    )
    reports = vector.read_community_reports(sorted(_OUTDATED))
    for report in reports:
        documents, filters = _STORED_LINEAGE[report.community_id]
        report.document_ids = list(documents)
        report.attributes = {**(report.attributes or {}), "filters": filters}
    vector.upsert_community_reports(reports)
    return config, graph, vector


def _lineage(vector: Any) -> dict[str, tuple[list[str], Any]]:
    return {
        r.community_id: (r.document_ids, (r.attributes or {}).get("filters"))
        for r in vector.read_community_reports(sorted(_OUTDATED))
    }


def test_a_removal_narrows_the_reports_lineage_to_the_units_left() -> None:
    config, graph, vector = _lineage_stores(FakeVectorStore)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    manager.remove_text_units("default", ["t1"])

    lineage = _lineage(vector)
    assert lineage["c1"][0] == ["doc-t2"]
    assert lineage["p"][0] == ["doc-t2", "doc-t3"]


def test_a_store_that_cannot_read_unit_attributes_keeps_the_reports_lineage(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    # R4A-5 / R4B-6: the default {} read as "no unit stored" narrowed every
    # kept report to no documents and no filters, so each attribute filter
    # excluded it (fail-closed) and its lineage was lost from the store.
    from unified_kg_rag.shared import logging as shared_logging

    monkeypatch.setattr(shared_logging, "_warned_once", set())
    config, graph, vector = _lineage_stores(_CannotReadUnits)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    manager.remove_text_units("default", ["t1"])

    assert _lineage(vector) == _STORED_LINEAGE
    # R5A-1: the text units are narrowed all the same (that needs no unit
    # attributes), so a report the refresh does not write again cites no
    # removed one.
    assert {
        r.community_id: r.text_unit_ids
        for r in vector.read_community_reports(sorted(_OUTDATED))
    } == {"c1": ["t2"], "p": ["t2", "t3"]}
    assert manager.outdated_reports["default"] == _OUTDATED
    assert any(
        "_CannotReadUnits leaves VectorIndexer.read_text_unit_attributes" in m
        for m in caplog.messages
    )


def test_a_refresh_on_a_store_that_cannot_read_unit_attributes_keeps_the_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The refresh wrote each report again with the lineage of the units it
    # read: none from the default, so no documents and no filters.
    from unified_kg_rag.shared import logging as shared_logging

    monkeypatch.setattr(shared_logging, "_warned_once", set())
    config, graph, vector = _lineage_stores(_CannotReadUnits)
    model = _Reports()

    done = _refresh(config, graph, vector, model)("default", _OUTDATED, {"acme"})

    assert done.written == _OUTDATED
    reports = {r.community_id: r for r in vector.read_community_reports(_OUTDATED)}
    assert reports["p"].summary == "Members: Acme, Carrier, Depot."
    # The text units are the community's current ones; the documents and
    # filters those stored with the report (the store cannot say less).
    assert reports["p"].text_unit_ids == ["t2", "t3"]
    assert _lineage(vector) == _STORED_LINEAGE


def test_a_community_without_a_report_has_its_vertex_written_from_its_members() -> None:
    # r8diff-6: c2 is stored without a report (seed-only, or its report
    # failed) and its vertex cites t1, which a removal took: the refresh
    # writes its vertex from its members (no model call), and deletes the
    # vertex of one with no member left.
    config = _config()
    graph, vector = _stores(config, child_report=False)
    graph.upsert_communities(
        [_vertex("c2", ["t1", "t2", "t3"], 0), _vertex("c8", ["t1"], 0)]
    )
    model = _Reports()
    done = _refresh(config, graph, vector, model)("default", (), (), {"c2", "c8"})
    assert not (done.written | done.deleted | done.kept)
    assert not model.inputs
    vertex = graph.data["communities"][("default", "c2")]
    assert vertex.text_unit_ids == ["t2", "t3"]
    assert (vertex.size, vertex.level, vertex.parent) == (2, "0", "p")
    assert vertex.attributes["entity_names"] == ["Carrier", "Depot"]
    assert ("default", "c8") not in graph.data["communities"]


def test_a_parent_written_again_no_longer_lists_a_child_deleted_beside_it() -> None:
    # r9diff-2: c8, stored without a report, keeps t9 but no member: the
    # refresh deletes its vertex. p, refreshed in the same run, read its
    # place before the deletion was known and still listed c8 in its
    # report's child_community_ids and its vertex's children.
    config = _config()
    graph, vector = _stores(config, child_report=False)
    graph.upsert_communities(
        [
            _vertex("c8", ["t1", "t9"], 0),
            _vertex("p", ["t1", "t2", "t3", "t9"], 1, ["c1", "c2", "c8"]),
        ]
    )
    vector.upsert_community_reports(
        [_report("p", ["t1", "t2", "t3", "t9"], 1, children=["c1", "c2", "c8"])]
    )
    done = _refresh(config, graph, vector, _Reports())(
        "default", _OUTDATED, {"acme"}, {"c2", "c8"}
    )
    assert done.written == _OUTDATED and not done.kept
    assert ("default", "c8") not in graph.data["communities"]
    assert graph.data["communities"][("default", "p")].children == ["c1", "c2"]
    (report,) = vector.read_community_reports([community_report_id("p")])
    assert report.child_community_ids == ["c1", "c2"]


def test_a_graph_that_cannot_read_vertices_keeps_them(caplog) -> None:
    config = _config()
    graph, vector = _stores(config, child_report=False)
    graph.upsert_communities([_vertex("c2", ["t1", "t2", "t3"], 0)])
    graph.read_communities = lambda ids, suffix=None: None
    done = _refresh(config, graph, vector, _Reports())("default", (), (), {"c2"})
    assert not (done.written | done.deleted | done.kept)
    assert graph.data["communities"][("default", "c2")].text_unit_ids == [
        "t1",
        "t2",
        "t3",
    ]
    (warning,) = [
        r.getMessage()
        for r in caplog.records
        if "cannot read community vertices" in r.getMessage()
    ]
    assert "indexing.reset: true" in warning


def test_a_removal_lists_the_communities_it_leaves_without_an_outdated_report(
    mocker,
) -> None:
    from unified_kg_rag.application.storage.stored_items import OUTDATED_REPORTS

    _, manager, vector = _removal_stores()
    graph = manager.graph_indexer
    # s cites t1 and a unit left, and has no report; g cites only t1.
    graph.upsert_communities([_vertex("s", ["t1", "t2"], 0), _vertex("g", ["t1"], 0)])
    manager.remove_text_units("default", ["t1"])
    # c1 and p have outdated reports (written again with their vertices).
    assert manager.stale_communities == {"default": {"s"}}
    note = vector.read_extraction_records([OUTDATED_REPORTS])[OUTDATED_REPORTS]
    assert note["community_ids"] == ["s"]
    # A successor of a run killed before its commit refreshes them too.
    manager.reset_run_state()
    assert not manager.stale_communities
    manager.merge_outdated_note("default")
    assert manager.stale_communities == {"default": {"s"}}


def test_a_suffix_with_only_stale_communities_merges_its_note_and_refreshes() -> None:
    # r9diff-3: a removal can leave communities without an outdated report
    # citing removed text and flag no report (a seed-only cluster with no
    # document entity): its suffix's note (here an earlier killed run's,
    # naming only communities) is merged, and the refresh runs for those
    # communities alone. Both paths were left to no test.
    from tests.fixtures.fakes.doc_status import FakeDocStatusStore
    from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
    from unified_kg_rag.application.ingestion.report_refresh import RefreshedReports
    from unified_kg_rag.application.storage.stored_items import OUTDATED_REPORTS

    _, manager, vector = _removal_stores()
    vector.write_extraction_records(
        {OUTDATED_REPORTS: {"community_ids": ["s1"], "candidates": ["e1"]}},
        suffix="other",
    )
    # This run's own removal in "other" (the run's namespace is "default").
    manager.stale_communities["other"] = {"s2"}
    incremental = IncrementalIndexer(FakeDocStatusStore(), manager)
    candidates = incremental._merge_outdated_notes()
    assert manager.stale_communities == {"other": {"s1", "s2"}}
    assert candidates["other"] == {"e1"}

    calls: list[tuple[str, set[str], set[str]]] = []

    def refresh(suffix, ids, members, stale):  # type: ignore[no-untyped-def]
        calls.append((suffix, set(ids), set(stale)))
        return RefreshedReports(set(), set(), set(), {}, set())

    left = incremental._refresh_reports(refresh, [], candidates)  # type: ignore[arg-type]
    assert calls == [("other", set(), {"s1", "s2"})]
    assert not left


def test_a_failed_delete_of_communities_left_empty_warns_and_keeps_them(
    caplog,
) -> None:
    # r9base-1: deleting the vertices of communities with no member left
    # is best-effort (only a removal's search reads them): a failed delete
    # warns, records no stats, and the rest of the refresh stands.
    config = _config()
    graph, vector = _stores(config, child_report=False)
    graph.upsert_communities(
        [_vertex("c2", ["t1", "t2", "t3"], 0), _vertex("c8", ["t1"], 0)]
    )

    def fail(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("graph store down")

    graph.delete_by_id = fail
    done = _refresh(config, graph, vector, _Reports())("default", (), (), {"c2", "c8"})
    assert "graph_emptied_communities_default" not in done.results
    assert "graph_refreshed_communities_default" in done.results
    assert ("default", "c8") in graph.data["communities"]
    assert graph.data["communities"][("default", "c2")].text_unit_ids == ["t2", "t3"]
    (warning,) = [
        r.getMessage()
        for r in caplog.records
        if "with no member left from the graph store failed" in r.getMessage()
    ]
    assert "1 communities of 'default'" in warning
