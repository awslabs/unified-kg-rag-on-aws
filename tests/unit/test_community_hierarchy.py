# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A community report's place in the hierarchy, and the cuts global search maps.

The hierarchy is built bottom-up, so a report's depth from the root is derived
from the parent links; a cut at depth ``D`` is every community at ``D`` plus the
leaves above it (MS GraphRAG's ``read_indexer_reports`` roll-up). The cut is
one range filter on two indexed fields, checked here on the embedded engine.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.community_hierarchy import (
    LEAF_CUT_DEPTH,
    attach_report_hierarchy,
    community_depths,
    cut_filter,
)
from unified_kg_rag.domain.models import Community, CommunityReport, Config

pytestmark = pytest.mark.unit

# Built bottom-up: level 0 is the finest. C2 has one child (C2x) a level
# further down, A and B are carried up to different levels, B is a leaf.
#            R0 (L3)                 R1 (L3)
#          /        \                  |
#      A (L2)      B (L0)            C (L2)
#     /     \                       /     \
#  A1 (L0) A2 (L0)              C1 (L0)  C2 (L1)
#                                           |
#                                        C2x (L0)
HIERARCHY: dict[str, tuple[int, str, list[str]]] = {
    "R0": (3, "", ["A", "B"]),
    "R1": (3, "", ["C"]),
    "A": (2, "R0", ["A1", "A2"]),
    "B": (0, "R0", []),
    "C": (2, "R1", ["C1", "C2"]),
    "A1": (0, "A", []),
    "A2": (0, "A", []),
    "C1": (0, "C", []),
    "C2": (1, "C", ["C2x"]),
    "C2x": (0, "C2", []),
}


def _communities() -> list[Community]:
    return [
        Community(
            id=cid,
            name=cid,
            level=str(level),
            parent=parent,
            children=children,
        )
        for cid, (level, parent, children) in HIERARCHY.items()
    ]


def _reports() -> list[CommunityReport]:
    return [
        CommunityReport(id=f"r-{cid}", name=cid, community_id=cid, full_content=cid)
        for cid in HIERARCHY
    ]


def test_depth_counts_from_the_root_not_the_level() -> None:
    assert community_depths(_communities()) == {
        "R0": 0,
        "R1": 0,
        "A": 1,
        "B": 1,
        "C": 1,
        "A1": 2,
        "A2": 2,
        "C1": 2,
        "C2": 2,
        "C2x": 3,
    }


def test_a_missing_parent_makes_a_root() -> None:
    orphan = Community(id="x", name="x", level="0", parent="gone", children=[])
    assert community_depths([orphan]) == {"x": 0}


def test_reports_carry_level_depth_parent_and_children() -> None:
    reports = {r.community_id: r for r in _reports()}
    attach_report_hierarchy(list(reports.values()), _communities())
    a, b = reports["A"], reports["B"]
    assert (a.level, a.depth, a.parent_community_id) == (2, 1, "R0")
    assert a.child_community_ids == ["A1", "A2"]
    assert (b.level, b.depth, b.child_community_ids) == (0, 1, [])
    assert reports["R0"].parent_community_id is None


def test_the_indexing_manager_attaches_the_hierarchy_on_write() -> None:
    # At write time, so reports restored from a stage cache get it too.
    reports = _reports()
    manager = IndexingManager.__new__(IndexingManager)
    manager.config = Config()
    manager.opensearch_indexer = _Recorder()
    manager.neptune_indexer = _Recorder()
    manager._log_completion_summary = lambda *_: None  # type: ignore[method-assign]
    manager.index_delta(communities=_communities(), community_reports=reports)
    assert {r.community_id: r.depth for r in reports}["C2x"] == 3


class _Recorder:
    def __getattr__(self, name: str):
        return lambda *args, **kwargs: None


def _indexed_cut(tmp_path: Path, depth: int) -> set[str]:
    reports = _reports()
    attach_report_hierarchy(reports, _communities())
    config = Config()
    indexer = OpenSearchIndexer.__new__(OpenSearchIndexer)
    indexer.config = config
    indexer.opensearch_config = config.indexing.opensearch
    indexer.analyzer = "standard"
    indexer._embedding_dimension = 2
    engine = SQLiteSearchEngine(tmp_path / "vector.sqlite3")
    engine.create_index("reports", indexer._get_community_reports_mapping())
    engine.bulk_index(
        "reports",
        [indexer._prepare_community_report_doc(r, ([1.0, 0.0],) * 3) for r in reports],
    )
    clauses = [{"range": {k: v}} for k, v in cut_filter(depth).items()]
    hits = engine.search(
        index="reports",
        body={"size": 100, "query": {"bool": {"filter": clauses}}},
    )["hits"]["hits"]
    return {hit["_source"]["community_id"] for hit in hits}


@pytest.mark.parametrize(
    ("depth", "cut"),
    [
        (0, {"R0", "R1"}),
        (1, {"A", "B", "C"}),
        # B is a leaf above depth 2: it stands for its part of the corpus.
        (2, {"A1", "A2", "B", "C1", "C2"}),
        (3, {"A1", "A2", "B", "C1", "C2x"}),
        (LEAF_CUT_DEPTH, {"A1", "A2", "B", "C1", "C2x"}),
    ],
)
def test_a_cut_covers_every_entity_once(tmp_path: Path, depth: int, cut) -> None:
    assert _indexed_cut(tmp_path, depth) == cut
