# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""HippoRAG 2's Personalized PageRank and seeding (``domain.retrieval.ppr``)."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import GraphEdge
from unified_kg_rag.domain.retrieval.ppr import (
    PassageGraph,
    cap_edges_per_vertex,
    min_max,
    passage_weights,
    personalized_pagerank,
    seed_weights,
)

pytestmark = pytest.mark.unit


def test_a_path_from_one_seed_matches_the_closed_form() -> None:
    # 0 -1- 1 -2- 2, reset on 0, d = 0.5: p0 = 5/9, p1 = 1/3, p2 = 1/9
    # (python-igraph's personalized_pagerank gives the same).
    result = personalized_pagerank(
        ["0", "1", "2"], [("0", "1", 1.0), ("1", "2", 2.0)], {"0": 1.0}
    )
    assert result.converged
    assert result.scores == pytest.approx(
        {"0": 5 / 9, "1": 1 / 3, "2": 1 / 9}, abs=1e-6
    )
    assert sum(result.scores.values()) == pytest.approx(1.0)


def test_a_node_without_edges_returns_its_mass_to_the_reset_vector() -> None:
    # The isolated seed's walk restarts on the reset vector (half on itself,
    # half on "a"): p_lone = 1/4 + p_lone/4, p_a = 1/4 + p_b/2 + p_lone/4,
    # p_b = p_a/2.
    result = personalized_pagerank(
        ["a", "b", "lone"], [("a", "b", 1.0)], {"a": 1.0, "lone": 1.0}
    )
    assert result.scores == pytest.approx(
        {"lone": 1 / 3, "a": 4 / 9, "b": 2 / 9}, abs=1e-6
    )


def test_parallel_edges_add_up_and_self_loops_are_dropped() -> None:
    once = personalized_pagerank([], [("a", "b", 2.0), ("a", "c", 1.0)], {"a": 1.0})
    twice = personalized_pagerank(
        [],
        [("a", "b", 1.0), ("b", "a", 1.0), ("a", "c", 1.0), ("c", "c", 5.0)],
        {"a": 1.0},
    )
    assert twice.scores == pytest.approx(once.scores)


def test_the_order_of_the_input_does_not_change_the_scores() -> None:
    edges = [("a", "b", 1.0), ("b", "c", 1.0), ("c", "d", 3.0), ("a", "d", 0.5)]
    forward = personalized_pagerank("abcd", edges, {"a": 1.0, "c": 0.2})
    backward = personalized_pagerank(
        "dcba", list(reversed(edges)), {"c": 0.2, "a": 1.0}
    )
    assert forward.scores == backward.scores


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"damping": 1.0}, "damping"),
        ({"damping": 0.0}, "damping"),
        ({"reset": {"a": 0.0}}, "no mass"),
        ({"reset": {"a": -1.0}}, "reset weight"),
        ({"reset": {"a": float("nan")}}, "reset weight"),
    ],
)
def test_an_invalid_damping_or_reset_is_refused(kwargs: dict, match: str) -> None:
    arguments = {"reset": {"a": 1.0}, **kwargs}
    with pytest.raises(ValueError, match=match):
        personalized_pagerank(["a", "b"], [("a", "b", 1.0)], **arguments)


def test_max_iterations_bounds_the_run() -> None:
    result = personalized_pagerank(
        "abc", [("a", "b", 1.0), ("b", "c", 1.0)], {"a": 1.0}, max_iterations=2
    )
    assert (result.iterations, result.converged) == (2, False)
    assert sum(result.scores.values()) == pytest.approx(1.0)


def test_seed_weights_follow_hipporag_2() -> None:
    # Mean over the facts naming an entity of fact score / passages cited.
    facts = [(0.9, ["a", "b"]), (0.5, ["b", "c"]), (0.0, ["d", "a"])]
    weights = seed_weights(facts, {"a": 3, "b": 1, "c": 0}, link_top_k=None)
    assert weights == pytest.approx(
        {"a": (0.9 / 3 + 0.0) / 2, "b": (0.9 + 0.5) / 2, "c": 0.5}
    )
    # link_top_k keeps the best, ties by id.
    assert list(seed_weights(facts, {"a": 3, "b": 1}, link_top_k=2)) == ["b", "c"]


def test_passage_weights_scale_min_max_scores() -> None:
    assert min_max({}) == {}
    assert min_max({"x": 2.0, "y": 2.0}) == {"x": 1.0, "y": 1.0}
    assert passage_weights({"x": 3.0, "y": 1.0, "z": 2.0}, 0.05) == pytest.approx(
        {"x": 0.05, "y": 0.0, "z": 0.025}
    )


def _edge(edge_id: str, source: str, target: str) -> GraphEdge:
    return GraphEdge(id=edge_id, source_id=source, target_id=target)


def test_cap_edges_per_vertex_keeps_each_vertex_first_edges_by_id() -> None:
    edges = [
        _edge("r3", "hub", "c"),
        _edge("r1", "hub", "a"),
        _edge("r2", "b", "hub"),
        _edge("r4", "a", "out"),  # leaves the set
        _edge("r1", "hub", "a"),  # repeated
    ]
    kept = cap_edges_per_vertex(edges, ["hub", "a", "b", "c"], per_vertex=2)
    # hub's first two (r1, r2); c keeps r3 through its own slot.
    assert [e.id for e in kept] == ["r2", "r1", "r3"]
    assert [e.id for e in cap_edges_per_vertex(edges, ["hub", "a"], 1)] == ["r1"]
    assert cap_edges_per_vertex(edges, ["hub", "a"], 0) == []


def test_a_passage_graph_ranks_the_bridge_passage_above_a_hub_passage() -> None:
    # Two seeds, Ann and Bob, each cite their own passage; the bridge
    # passage is cited by both, the hub passage by a third entity only one
    # seed reaches. The bridge gathers mass from both seeds.
    graph = PassageGraph()
    graph.add_relationship("ann", "bob")
    graph.add_relationship("ann", "hub")
    graph.link_passages("ann", ["p-ann", "p-bridge"])
    graph.link_passages("bob", ["p-bob", "p-bridge"])
    graph.link_passages("hub", ["p-hub"])
    ranked, result = graph.rank_passages({"ann": 1.0, "bob": 1.0}, {})
    assert result is not None and result.converged
    order = [unit for unit, _ in ranked]
    assert order[0] == "p-bridge"
    assert order.index("p-bridge") < order.index("p-hub")
    assert graph.passages == {"p-ann", "p-bob", "p-bridge", "p-hub"}


def test_entities_and_passages_with_one_id_stay_apart() -> None:
    graph = PassageGraph()
    graph.link_passages("x", ["x"])
    ranked, _ = graph.rank_passages({"x": 1.0}, {})
    assert [unit for unit, _ in ranked] == ["x"]
    assert graph.node_count == 2


def test_a_passage_graph_without_seed_mass_ranks_nothing() -> None:
    graph = PassageGraph()
    graph.link_passages("x", ["p"])
    assert graph.rank_passages({"x": 0.0}, {}) == ([], None)
    assert PassageGraph().rank_passages({"x": 1.0}, {}) == ([], None)
    # A passage seed alone ranks: it is added as a node.
    ranked, _ = PassageGraph().rank_passages({}, {"p": 0.05})
    assert [unit for unit, _ in ranked] == ["p"]
