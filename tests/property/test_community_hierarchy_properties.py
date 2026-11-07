# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariants of the community hierarchy (``build_hierarchy``).

Split communities stay the parents of their pieces, so the tree is no
longer one Leiden pass per level; whatever its shape, every level and
every cut global search maps (``community_depth``, §6.1.1) must cover each
clustered entity exactly once, a parent must be the union of its children
one level or more below it, and the leaves are the level-0 communities an
incremental run's drift and roll-up start from.
"""

from __future__ import annotations

import logging

import networkx as nx
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from unified_kg_rag.domain.ingestion.community_clustering import (
    build_hierarchy,
    hierarchical_communities,
    partition_tree,
    small_component_nodes,
)
from unified_kg_rag.domain.ingestion.community_hierarchy import (
    LEAF_CUT_DEPTH,
    community_depths,
)
from unified_kg_rag.domain.models import Community
from unified_kg_rag.domain.models.config import CommunityDetectionConfig

pytestmark = pytest.mark.property


@st.composite
def cases(draw: st.DrawFn) -> tuple[nx.Graph, CommunityDetectionConfig]:
    groups = draw(st.integers(1, 8))
    size = draw(st.integers(2, 14))
    graph = nx.planted_partition_graph(
        groups,
        size,
        draw(st.floats(0.3, 0.95)),
        draw(st.floats(0.01, 0.2)),
        seed=draw(st.integers(0, 10_000)),
    )
    graph = nx.relabel_nodes(graph, lambda n: f"e{n:03d}")
    weights = draw(
        st.lists(
            st.integers(1, 10),
            min_size=graph.number_of_edges(),
            max_size=graph.number_of_edges(),
        )
    )
    for (u, v), w in zip(graph.edges(), weights, strict=True):
        graph[u][v]["weight"] = float(w)
    max_size = draw(st.integers(2, 12))
    config = CommunityDetectionConfig(
        max_cluster_size=max_size,
        min_community_size=draw(st.integers(1, min(3, max_size))),
        max_levels=draw(st.integers(1, 5)),
        random_state=draw(st.integers(1, 1000)),
    )
    return graph, config


def _assert_covers_once(groups: list[set[str]], entities: set[str]) -> None:
    assert sum(len(g) for g in groups) == len(entities)
    assert set().union(*groups) == entities


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(cases())
def test_levels_and_cuts_cover_every_clustered_entity_once(
    case: tuple[nx.Graph, CommunityDetectionConfig],
) -> None:
    graph, config = case
    logging.disable(logging.WARNING)
    try:
        unclustered = small_component_nodes(graph, config.min_community_size)
        partitions, _ = build_hierarchy(graph, config)
        communities, node_to_leaf = hierarchical_communities(partitions, unclustered)
    finally:
        logging.disable(logging.NOTSET)

    assert 1 <= len(partitions) <= config.max_levels
    # Each level partitions the level below.
    _assert_covers_once([set(p) for p in partitions[0]], set(graph.nodes))
    for below, level in zip(partitions, partitions[1:], strict=False):
        _assert_covers_once(level, set(range(len(below))))

    clustered = set(graph.nodes) - unclustered
    if not clustered:
        assert not communities
        return
    leaves = [c for c in communities.values() if not c.children_ids]
    assert all(c.level == 0 for c in leaves)
    _assert_covers_once([c.nodes for c in leaves], clustered)
    assert set(node_to_leaf) == clustered
    for community in communities.values():
        if community.children_ids:
            children = [communities[i] for i in community.children_ids]
            assert len(children) >= 2
            assert all(c.parent_id == community.community_id for c in children)
            assert all(c.level < community.level for c in children)
            _assert_covers_once([c.nodes for c in children], community.nodes)

    records = [
        Community(
            id=c.community_id,
            name=c.label,
            level=str(c.level),
            parent=c.parent_id or "",
            children=c.children_ids,
        )
        for c in communities.values()
    ]
    depths = community_depths(records)
    for depth in range(max(depths.values()) + 2):
        # The cut global search maps at ``community_depth`` = depth.
        cut = [
            communities[cid].nodes
            for cid, d in depths.items()
            if d <= depth <= (d if communities[cid].children_ids else LEAF_CUT_DEPTH)
        ]
        _assert_covers_once(cut, clustered)


@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(cases())
def test_partition_tree_parents_are_their_childrens_union(
    case: tuple[nx.Graph, CommunityDetectionConfig],
) -> None:
    # After the small-leaf merge moves entities between split communities,
    # a parent's members (which its cluster-graph node is built from) must
    # be re-derived from the children it kept.
    graph, config = case
    logging.disable(logging.WARNING)
    try:
        roots = partition_tree(graph, config)
    finally:
        logging.disable(logging.NOTSET)
    _assert_covers_once([root.members for root in roots], set(graph.nodes))
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node.children:
            assert len(node.children) >= 2
            _assert_covers_once([c.members for c in node.children], node.members)
            stack.extend(node.children)
