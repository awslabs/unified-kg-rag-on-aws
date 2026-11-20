# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A community report written again after a removal reads what a full
build's detector reads for the same community (``ReportRefresh``, B-2).

Random level-0 communities under one parent, random members citing random
text units, random edges inside and across communities, and a random set of
removed text units. The stores hold what the removal and its commit leave
(an entity or edge keeps the units not removed and goes with none left), the
reports citing a removed unit are outdated, and each is written again. The
model is asked, for every report written again, with the input a detector
over the whole surviving graph packs for the community's surviving members
(its children's reports rolled up as written), and a community without a
member left loses its report: the convergence law of the summary tree, for
reports.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.fixtures.fakes.chat_models import ScriptedLLMFactory
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from tests.unit.test_report_refresh import _edge, _entity, _report, _Reports
from unified_kg_rag.adapters.ingestion.community_detector import CommunityDetector
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion.report_refresh import ReportRefresh
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.community_clustering import (
    community_label,
    community_name,
)
from unified_kg_rag.domain.ingestion.community_hierarchy import community_report_id
from unified_kg_rag.domain.ingestion.graph_builder import GraphBuilder
from unified_kg_rag.domain.models import Community, Config, TextUnit

pytestmark = pytest.mark.property

_NAMES = ("Acme", "Buyer", "Carrier", "Depot", "Xeno")
_UNITS = [f"t{i}" for i in range(6)]


@st.composite
def _corpus(draw: st.DrawFn) -> dict[str, Any]:
    communities = draw(st.integers(1, 3))
    members = {
        name: (
            f"c{draw(st.integers(0, communities - 1))}",
            sorted(draw(st.sets(st.sampled_from(_UNITS), min_size=1, max_size=3))),
        )
        for name in _NAMES
    }
    pairs = [(a, b) for i, a in enumerate(_NAMES) for b in _NAMES[i + 1 :]]
    edges = {
        pair: sorted(draw(st.sets(st.sampled_from(_UNITS), min_size=1, max_size=2)))
        for pair in draw(st.sets(st.sampled_from(pairs), max_size=6))
    }
    removed = draw(st.sets(st.sampled_from(_UNITS), min_size=1, max_size=3))
    budget = draw(st.sampled_from([40, 4000]))
    return {
        "communities": communities,
        "members": members,
        "edges": edges,
        "removed": removed,
        "budget": budget,
    }


def _community(cid: str, level: int, members: list[str], graph: Any, children=()):
    label = community_label(level, cid)
    inside = graph.subgraph(members)
    return Community(
        id=cid,
        short_id=label,
        name=community_name(level, label),
        level=str(level),
        parent="p" if level == 0 else "",
        children=list(children),
        entity_ids=sorted(members),
        size=len(members),
        relationship_ids=sorted(
            rel["id"]
            for _, _, data in inside.edges(data=True)
            for rel in data["relationships"]
        ),
    )


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(corpus=_corpus())
def test_a_written_report_reads_what_a_full_build_reads(corpus: dict) -> None:
    config = Config()
    config.processing.claim_extraction.enabled = False
    config.graph.community_detection.report_generation.max_report_context_tokens = (
        corpus["budget"]
    )
    removed = corpus["removed"]
    before = {
        name.lower(): (community, units)
        for name, (community, units) in corpus["members"].items()
    }
    # The level-0 communities a member was clustered into.
    leaves = sorted({community for community, _ in before.values()})
    # What the removal and its commit leave.
    entities = [
        _entity(name, kept, [community])
        for name, (community, units) in corpus["members"].items()
        if (kept := [u for u in units if u not in removed])
    ]
    alive = {e.id for e in entities}
    edges = [
        _edge(a.lower(), b.lower(), kept)
        for (a, b), units in corpus["edges"].items()
        if (kept := [u for u in units if u not in removed])
        and {a.lower(), b.lower()} <= alive
    ]
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    graph.upsert_entities([e.model_copy(deep=True) for e in entities])
    graph.upsert_relationships([r.model_copy(deep=True) for r in edges])
    vector.upsert_text_units(
        [TextUnit(id=u, text=u) for u in _UNITS if u not in removed]
    )
    # The reports as the full build before the removal wrote them.
    units_of = {
        c: sorted({u for cid, units in before.values() if cid == c for u in units})
        for c in leaves
    }
    reports = [_report(c, units_of[c], 0, parent="p") for c in leaves]
    parent_units = sorted({u for units in units_of.values() for u in units})
    reports.append(_report("p", parent_units, 1, children=leaves))
    vector.upsert_community_reports(reports)
    outdated = {r.id for r in reports if removed & set(r.text_unit_ids or [])}
    stripped = {e.id for e in entities if set(before[e.id][1]) & removed}

    refreshed = _Reports()
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    providers = Providers(
        config, boto_session=MagicMock(), llm_factory=ScriptedLLMFactory(refreshed)
    )
    stored_children = {
        r.community_id: r
        for r in vector.read_community_reports([community_report_id(c) for c in leaves])
        if r.id not in outdated
    }
    done = ReportRefresh(manager, providers)("default", outdated, stripped)

    # A full build's detector over the surviving graph, for the same
    # communities with their surviving members.
    whole = GraphBuilder(entities, edges).build()
    full = _Reports()
    detector = CommunityDetector(
        config,
        show_progress=False,
        providers=Providers(
            config, boto_session=MagicMock(), llm_factory=ScriptedLLMFactory(full)
        ),
    ).report_on(whole)
    members = {c: [e.id for e in entities if e.community_ids == [c]] for c in leaves}
    expected = [
        _community(c, 0, members[c], whole)
        for c in leaves
        if community_report_id(c) in outdated and members[c]
    ]
    if community_report_id("p") in outdated and alive:
        expected.append(_community("p", 1, sorted(alive), whole, leaves))
    detector.generate_reports(expected, sub_reports=stored_children)

    # The calls of one level run concurrently: compared as a multiset.
    assert sorted(refreshed.inputs) == sorted(full.inputs)
    assert done.written == {community_report_id(c.id) for c in expected}
    assert done.deleted == outdated - done.written
    assert not done.kept
    stored = vector.ids("community_reports")
    assert not done.deleted & stored
