# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariants of the Personalized PageRank (``domain.retrieval.ppr``).

It is a distribution, it equals the exact solution of its linear system,
raising a node's reset mass never lowers its score, and the PPR of a graph
cut to the ``h``-hop ball around the seeds differs from the whole graph's by
at most ``2 d^(h+1)`` in L1: the bound docs/design.md §6.4 rests on.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from unified_kg_rag.domain.retrieval.ppr import personalized_pagerank

pytestmark = pytest.mark.property

# Enough steps that the power iteration is within 1e-12 of the fixed point.
_EXACT = {"max_iterations": 200, "tolerance": 1e-13}


@st.composite
def graphs(draw: st.DrawFn, max_nodes: int = 12):
    """Nodes ``n0..``, weighted edges and a non-empty reset vector."""
    n = draw(st.integers(min_value=2, max_value=max_nodes))
    nodes = [f"n{i:02d}" for i in range(n)]
    pairs = draw(
        st.lists(
            st.tuples(
                st.integers(0, n - 1),
                st.integers(0, n - 1),
                st.floats(0.1, 5.0, allow_nan=False),
            ),
            max_size=3 * n,
        )
    )
    edges = [(nodes[i], nodes[j], w) for i, j, w in pairs]
    seeds = draw(
        st.dictionaries(
            st.sampled_from(nodes),
            st.floats(0.01, 1.0, allow_nan=False),
            min_size=1,
            max_size=max(1, n // 2),
        )
    )
    damping = draw(st.floats(0.1, 0.9))
    return nodes, edges, seeds, damping


def _exact(nodes, edges, reset, damping) -> dict[str, float]:
    """``p`` with ``p (I - d P') = c r``, normalized: the fixed point the
    power iteration converges to (dangling mass back to the reset)."""
    index = {node: i for i, node in enumerate(sorted(nodes))}
    n = len(index)
    weight = np.zeros((n, n))
    for u, v, w in edges:
        if u != v:
            weight[index[u], index[v]] += w
            weight[index[v], index[u]] += w
    strength = weight.sum(axis=1)
    transition = np.divide(
        weight,
        strength[:, None],
        out=np.zeros_like(weight),
        where=strength[:, None] > 0,
    )
    r = np.zeros(n)
    for node, value in reset.items():
        r[index[node]] += value
    r /= r.sum()
    p = np.linalg.solve((np.eye(n) - damping * transition).T, r)
    p /= p.sum()
    return {node: float(p[i]) for node, i in index.items()}


@given(graphs())
def test_scores_are_a_distribution_equal_to_the_exact_solution(graph) -> None:
    nodes, edges, reset, damping = graph
    result = personalized_pagerank(nodes, edges, reset, damping=damping, **_EXACT)
    assert all(score >= 0 for score in result.scores.values())
    assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-9)
    assert result.scores == pytest.approx(
        _exact(nodes, edges, reset, damping), abs=1e-8
    )


@given(graphs(), st.data())
def test_more_reset_mass_on_a_node_never_lowers_its_score(graph, data) -> None:
    nodes, edges, reset, damping = graph
    node = data.draw(st.sampled_from(nodes))
    extra = data.draw(st.floats(0.01, 2.0))
    before = personalized_pagerank(nodes, edges, reset, damping=damping, **_EXACT)
    raised = {**reset, node: reset.get(node, 0.0) + extra}
    after = personalized_pagerank(nodes, edges, raised, damping=damping, **_EXACT)
    assert after.scores[node] >= before.scores[node] - 1e-9


def _ball(nodes, edges, sources, hops) -> set[str]:
    neighbours: dict[str, set[str]] = {node: set() for node in nodes}
    for u, v, _ in edges:
        neighbours[u].add(v)
        neighbours[v].add(u)
    seen = dict.fromkeys(sources, 0)
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        if seen[node] == hops:
            continue
        for other in neighbours[node]:
            if other not in seen:
                seen[other] = seen[node] + 1
                queue.append(other)
    return set(seen)


@settings(max_examples=200)
@given(graphs(max_nodes=20), st.integers(min_value=0, max_value=4))
def test_a_truncated_neighbourhood_drifts_at_most_2_d_to_the_h_plus_1(
    graph, hops
) -> None:
    nodes, edges, reset, damping = graph
    whole = personalized_pagerank(nodes, edges, reset, damping=damping, **_EXACT)
    ball = _ball(nodes, edges, list(reset), hops)
    inside = [(u, v, w) for u, v, w in edges if u in ball and v in ball]
    cut = personalized_pagerank(ball, inside, reset, damping=damping, **_EXACT)
    drift = sum(abs(whole.scores[n] - cut.scores.get(n, 0.0)) for n in nodes)
    assert drift <= 2 * damping ** (hops + 1) + 1e-9
    # The mass farther than ``hops`` from every seed is at most d^(h+1).
    outside = sum(whole.scores[n] for n in nodes if n not in ball)
    assert outside <= damping ** (hops + 1) + 1e-9
