# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The community hierarchy against graphs with a known partition.

A community over ``max_cluster_size`` is split and stays the parent of its
pieces, as graspologic's ``hierarchical_leiden`` (MS GraphRAG's clustering)
keeps it. Regrouping the pieces afresh (one Leiden pass over their cluster
graph) recovered the planted groups worse than the first pass had: a
level-1 purity of 0.55 against 0.85 before the split.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter
from typing import Any

import graspologic_native
import networkx as nx
import pytest

from unified_kg_rag.domain.ingestion.community_clustering import (
    build_hierarchy,
    hierarchical_communities,
    leiden_communities,
    partition_tree,
)
from unified_kg_rag.domain.models.config import CommunityDetectionConfig

pytestmark = pytest.mark.unit


def _nmi(labels: dict[Any, int], truth: dict[Any, int]) -> float:
    n = len(labels)
    a = Counter(labels.values())
    b = Counter(truth[x] for x in labels)
    joint = Counter((labels[x], truth[x]) for x in labels)
    mutual = sum(
        c / n * math.log((c / n) / ((a[i] / n) * (b[j] / n)))
        for (i, j), c in joint.items()
    )

    def entropy(counts: Counter[int]) -> float:
        return -sum(v / n * math.log(v / n) for v in counts.values())

    total = entropy(a) + entropy(b)
    return 2 * mutual / total if total else 1.0


def _purity(labels: dict[Any, int], truth: dict[Any, int]) -> float:
    groups: dict[int, list[int]] = {}
    for node, label in labels.items():
        groups.setdefault(label, []).append(truth[node])
    return sum(Counter(v).most_common(1)[0][1] for v in groups.values()) / len(labels)


def _weighted(graph: nx.Graph, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    for u, v in graph.edges():
        graph[u][v]["weight"] = float(rng.randint(1, 10))
    return graph


def _planted(k: int, size: int, p_in: float, p_out: float) -> tuple[nx.Graph, dict]:
    graph = nx.relabel_nodes(
        nx.planted_partition_graph(k, size, p_in, p_out, seed=1), lambda n: f"e{n:05d}"
    )
    truth = {f"e{n:05d}": n // size for n in range(k * size)}
    return _weighted(graph, 3), truth


def _lfr(seed: int) -> tuple[nx.Graph, dict]:
    raw = nx.LFR_benchmark_graph(
        2000,
        2.5,
        1.5,
        0.1,
        average_degree=6,
        max_degree=80,
        min_community=30,
        max_community=200,
        seed=seed,
    )
    truth = {f"e{n:05d}": min(raw.nodes[n]["community"]) for n in raw}
    graph = nx.Graph()
    graph.add_edges_from((f"e{u:05d}", f"e{v:05d}") for u, v in raw.edges() if u != v)
    return _weighted(graph, seed), truth


GRAPHS = {
    "planted-40x50": lambda: _planted(40, 50, 0.25, 0.004),
    "planted-60x40": lambda: _planted(60, 40, 0.3, 0.003),
    "lfr-0": lambda: _lfr(0),
    "lfr-2": lambda: _lfr(2),
}


def _levels(partitions: list[list[set[Any]]]) -> list[dict[Any, int]]:
    """Each level of ``build_hierarchy``'s result as an entity -> label map."""
    sets = [list(partitions[0])]
    for partition in partitions[1:]:
        sets.append([set().union(*(sets[-1][i] for i in group)) for group in partition])
    return [
        {node: i for i, members in enumerate(level) for node in members}
        for level in sets
    ]


@pytest.fixture(autouse=True)
def _quiet(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)


@pytest.mark.timeout(300)
@pytest.mark.parametrize("name", sorted(GRAPHS))
def test_a_level_keeps_the_pre_split_partition(name: str) -> None:
    graph, truth = GRAPHS[name]()
    config = CommunityDetectionConfig(max_cluster_size=20, random_state=42)
    first_pass = leiden_communities(graph, config.resolution, config)
    labels = {node: i for i, members in first_pass.items() for node in members}
    pre_purity, pre_nmi = _purity(labels, truth), _nmi(labels, truth)

    levels = _levels(build_hierarchy(graph, config)[0])[1:]
    # The default min_community_size merges a few small leaves across the
    # first pass's boundaries, hence the tolerance.
    assert max(_purity(level, truth) for level in levels) >= pre_purity - 0.01
    assert max(_nmi(level, truth) for level in levels) >= pre_nmi - 0.01


def test_split_communities_are_the_parents_of_their_pieces() -> None:
    graph, _ = _planted(60, 40, 0.3, 0.003)
    config = CommunityDetectionConfig(
        max_cluster_size=20, random_state=42, min_community_size=1
    )
    first_pass = list(leiden_communities(graph, config.resolution, config).values())
    communities, _ = hierarchical_communities(build_hierarchy(graph, config)[0])
    by_nodes = {frozenset(c.nodes): c for c in communities.values()}
    split = [members for members in first_pass if len(members) > 20]
    assert split
    for members in first_pass:
        assert frozenset(members) in by_nodes
    for members in split:
        parent = by_nodes[frozenset(members)]
        children = [communities[i] for i in parent.children_ids]
        assert len(children) >= 2
        assert set().union(*(c.nodes for c in children)) == members
    # Every community nests in one first-pass community or is a union of them.
    for community in communities.values():
        inside = [m for m in first_pass if community.nodes <= m]
        unions = [m for m in first_pass if m <= community.nodes]
        assert inside or set().union(*unions) == community.nodes


@pytest.mark.parametrize("name", ["planted-40x50", "planted-60x40"])
def test_partition_tree_matches_graspologic_hierarchical_leiden(name: str) -> None:
    """Differential check against ``graspologic_native.hierarchical_leiden``
    with MS GraphRAG's arguments (``graphrag/graphs/hierarchical_leiden.py``).

    The pass over the whole graph is the same partition; the pieces of a
    split community come from Leiden runs whose random state differs from
    the native recursion's, so those are compared by quality and shape.
    """
    graph, _ = GRAPHS[name]()
    config = CommunityDetectionConfig(
        max_cluster_size=20,
        random_state=42,
        trials=1,
        extra_forced_iterations=0,
        min_community_size=1,
    )
    roots = partition_tree(graph, config)
    edges = sorted(
        (min(u, v), max(u, v), float(d["weight"])) for u, v, d in graph.edges(data=True)
    )
    native = graspologic_native.hierarchical_leiden(
        edges=edges,
        max_cluster_size=20,
        seed=42,
        starting_communities=None,
        resolution=1.0,
        randomness=0.001,
        use_modularity=True,
        iterations=1,
    )
    native_level_0: dict[int, set[str]] = {}
    native_final: dict[int, set[str]] = {}
    for entry in native:
        if entry.level == 0:
            native_level_0.setdefault(entry.cluster, set()).add(entry.node)
        if entry.is_final_cluster:
            native_final.setdefault(entry.cluster, set()).add(entry.node)

    assert {frozenset(r.members) for r in roots} == {
        frozenset(m) for m in native_level_0.values()
    }
    leaves: list[set[str]] = []
    stack = list(roots)
    while stack:
        node = stack.pop()
        if node.children:
            assert set().union(*(c.members for c in node.children)) == node.members
            stack.extend(node.children)
        else:
            leaves.append(node.members)
    assert max(len(leaf) for leaf in leaves) <= 20
    ours = nx.community.modularity(graph, leaves)
    theirs = nx.community.modularity(graph, list(native_final.values()))
    assert ours == pytest.approx(theirs, abs=0.02)
    assert len(leaves) == pytest.approx(len(native_final), rel=0.1)
