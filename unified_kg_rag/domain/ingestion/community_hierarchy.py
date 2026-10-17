# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where each community report sits in the community hierarchy.

Global search maps over the reports of one *cut* of the hierarchy, as MS
GraphRAG does (``read_indexer_reports``): the communities at depth ``D`` from
the root, plus the leaves that stop above ``D``, so every entity is covered
once. The hierarchy is built bottom-up (``Community.level`` 0 is the finest),
while MS GraphRAG counts its ``community_level`` from the root, so the depth is
derived here from the parent links and stored on each report.

A report belongs to cut ``D`` when ``depth <= D <= max_cut_depth``, where
``max_cut_depth`` is the report's own depth for a community with children and
:data:`LEAF_CUT_DEPTH` for a leaf: one range filter per field, which every
store applies.
"""

from __future__ import annotations

from unified_kg_rag.domain.models import Community, CommunityReport

# A leaf stands for every deeper cut.
LEAF_CUT_DEPTH = 1_000_000


def community_depths(communities: list[Community]) -> dict[str, int]:
    """Each community's distance from its root (a community with no parent).

    A community whose parent is not in ``communities`` counts as a root.
    """
    parents = {c.id: c.parent for c in communities}
    depths: dict[str, int] = {}

    def depth(community_id: str) -> int:
        chain: list[str] = []
        current = community_id
        while current not in depths:
            parent = parents.get(current)
            if not parent or parent not in parents or parent in chain:
                depths[current] = 0
                break
            chain.append(current)
            current = parent
        found = depths[current]
        for node in reversed(chain):
            found += 1
            depths[node] = found
        return depths[community_id]

    for community in communities:
        depth(community.id)
    return depths


def attach_report_hierarchy(
    reports: list[CommunityReport] | None, communities: list[Community] | None
) -> None:
    """Record each report's level, depth, parent and children (in place).

    A report whose community is not in ``communities`` is left as it is.
    """
    if not reports or not communities:
        return
    by_id = {c.id: c for c in communities}
    depths = community_depths(communities)
    for report in reports:
        community = by_id.get(report.community_id)
        if community is None:
            continue
        try:
            report.level = int(community.level)
        except (TypeError, ValueError):
            report.level = None
        report.depth = depths[community.id]
        parent = community.parent
        report.parent_community_id = parent if parent in by_id else None
        report.child_community_ids = [c for c in community.children if c in by_id]


def max_cut_depth(report: CommunityReport) -> int | None:
    """The deepest cut the report belongs to (see the module docstring)."""
    if report.depth is None:
        return None
    return report.depth if report.child_community_ids else LEAF_CUT_DEPTH


def cut_filter(depth: int) -> dict[str, dict[str, int]]:
    """The store filter selecting the reports of the depth-``depth`` cut."""
    return {"depth": {"lte": depth}, "max_cut_depth": {"gte": depth}}
