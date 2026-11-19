# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differential tests against the original LightRAG and MS GraphRAG code
(docs/design/retrieval-graphrag-lightrag.md#6-retrieval-methodologies).

The originals' answers on synthetic inputs
(``tests/fixtures/graphrag_lightrag_fidelity.py``) are recorded in
``tests/fixtures/graphrag_lightrag/upstream.json``; these tests compare ours
with them without the originals installed:

- LightRAG's weighted polling, exact: ``_pick_by_weighted_polling`` picks
  ``pick_by_weighted_polling``'s chunks in its order;
- MS GraphRAG's report context order, exact: a report input lists the
  relationships and entities in ``sort_context``'s order (no token cap), and
  the entities no relationship links come after them, where upstream leaves
  them out;
- which relationships reach it (a documented deviation): upstream hands
  ``sort_context`` one relationship per entity, here every relationship
  among the community's entities is packed.

``test_the_recorded_answers_are_the_originals`` runs the originals from
checkouts named by ``GRAPHRAG_SRC`` and ``LIGHTRAG_SRC`` (the weekly
``fidelity`` workflow, at the pinned commits); it is skipped otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures import graphrag_lightrag_fidelity as shared
from unified_kg_rag.adapters.ingestion.community_detector import CommunityDetector
from unified_kg_rag.adapters.search_strategies.lightrag_search import (
    _pick_by_weighted_polling,
)
from unified_kg_rag.domain.ingestion.graph_builder import GraphBuilder
from unified_kg_rag.domain.models import Community, Config, Entity, Relationship

pytestmark = pytest.mark.unit

UPSTREAM = json.loads(shared.FIXTURE.read_text(encoding="utf-8"))
_POLLING = {case["case"]: case for case in shared.polling_cases()}
_REPORTS = {case["case"]: case for case in shared.report_cases()}


@pytest.mark.parametrize("answer", UPSTREAM["polling"], ids=lambda a: a["case"])
def test_weighted_polling_picks_lightrags_chunks(answer: dict[str, Any]) -> None:
    case = _POLLING[answer["case"]]
    assert (
        _pick_by_weighted_polling(
            case["lineages"], case["max_related_chunks"], case["min_related_chunks"]
        )
        == answer["selected"]
    )


def _report_input(case: dict[str, Any]) -> dict[str, Any]:
    """Our report input for the case's community: every entity and
    relationship, no token or entity cap in the way."""
    config = Config()
    report = config.graph.community_detection.report_generation
    report.enabled = False
    report.max_report_context_tokens = 1_000_000
    report.max_entities_per_report = case["size"]
    detector = CommunityDetector(config, show_progress=False)
    names = [f"E{i:02d}" for i in range(case["size"])]
    detector.graph = GraphBuilder(
        [Entity(id=name, name=name, text_unit_ids=["t1"]) for name in names],
        [
            Relationship(
                id=f"R{j:02d}",
                source_id=names[source],
                target_id=names[target],
                type="RELATED_TO",
                description=f"R{j:02d}",
                text_unit_ids=["t1"],
            )
            for j, (source, target) in enumerate(case["edges"])
        ],
    ).build()
    community = Community(
        id="L0_C0",
        name="Level 0 Community 0",
        level="0",
        parent="",
        children=[],
        entity_ids=names,
        text_unit_ids=[],
        size=len(names),
    )
    return detector._prepare_report_input(community)


@pytest.mark.parametrize("answer", UPSTREAM["sort_context"], ids=lambda a: a["case"])
def test_a_report_input_is_packed_in_sort_contexts_order(
    answer: dict[str, Any],
) -> None:
    case = _REPORTS[answer["case"]]
    packed = _report_input(case)
    assert [r["id"] for r in packed["relationships"]] == [
        f"R{j:02d}" for j in answer["relationships"]
    ]
    entities = [e["id"] for e in packed["entities"]]
    linked = [f"E{i:02d}" for i in answer["entities"]]
    # Upstream lists the entities its relationships reach; the others come
    # last here (degree 0 in these graphs, so by id).
    assert entities[: len(linked)] == linked
    assert entities[len(linked) :] == sorted(set(entities) - set(linked))


def test_upstream_reports_see_one_relationship_per_entity_ours_every_one() -> None:
    # _prepare_reports_at_level aggregates each entity's relationships with
    # "first", so sort_context receives at most one relationship per entity.
    dropped = 0
    for answer in UPSTREAM["report_edges"]:
        case = _REPORTS[answer["case"]]
        assert len(answer["relationships"]) <= case["size"]
        dropped += len(case["edges"]) - len(answer["relationships"])
        packed = _report_input(case)
        assert len(packed["relationships"]) == len(case["edges"])
    assert dropped > 0


def test_the_recorded_answers_are_the_originals() -> None:
    graphrag, lightrag = os.environ.get("GRAPHRAG_SRC"), os.environ.get("LIGHTRAG_SRC")
    if not graphrag or not lightrag:
        pytest.skip("set GRAPHRAG_SRC and LIGHTRAG_SRC to the originals' checkouts")
    pytest.importorskip("pandas")
    assert shared.record(Path(graphrag), Path(lightrag)) == UPSTREAM
