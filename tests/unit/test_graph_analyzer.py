# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for GraphAnalyzer — the pure NetworkX graph-analysis core used by
the visualization/ingestion paths (AWS-free).

These exercise real centrality/statistics computation on small hand-built
graphs plus the empty-graph and caching edge cases, so a regression in the
NetworkX wiring or the export shape would fail here.
"""

from __future__ import annotations

import networkx as nx
import pytest

from aws_graphrag.domain.ingestion.graph_analyzer import GraphAnalyzer
from aws_graphrag.domain.models import Config

pytestmark = pytest.mark.unit


def _triangle_graph() -> nx.Graph:
    """A->B->C->A triangle with names; fully connected, density 1.0."""
    g = nx.Graph()
    g.add_node("a", name="Alice", node_type="entity")
    g.add_node("b", name="Bob", node_type="entity")
    g.add_node("c", name="Carol", node_type="entity")
    g.add_edge("a", "b", weight=1.0)
    g.add_edge("b", "c", weight=1.0)
    g.add_edge("c", "a", weight=1.0)
    return g


def _star_graph() -> nx.Graph:
    """Hub 'h' connected to three leaves; the hub is the most central node."""
    g = nx.Graph()
    g.add_node("h", name="Hub")
    for leaf in ("l1", "l2", "l3"):
        g.add_node(leaf, name=leaf.upper())
        g.add_edge("h", leaf)
    return g


class TestCentrality:
    def test_empty_graph_returns_empty_dict(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=nx.Graph())
        assert analyzer.calculate_centrality() == {}

    def test_none_graph_returns_empty_dict(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=None)
        assert analyzer.calculate_centrality() == {}

    def test_degree_centrality_on_triangle_is_uniform(self) -> None:
        # Every node in a triangle has degree 2 of 2 possible -> degree centrality 1.0.
        analyzer = GraphAnalyzer(Config(), graph=_triangle_graph())
        metrics = analyzer.calculate_centrality()
        assert set(metrics) == {"a", "b", "c"}
        assert all(m.degree == pytest.approx(1.0) for m in metrics.values())

    def test_hub_has_highest_degree_centrality_in_star(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=_star_graph())
        metrics = analyzer.calculate_centrality()
        hub = metrics["h"].degree
        leaves = [metrics[n].degree for n in ("l1", "l2", "l3")]
        assert all(hub > leaf for leaf in leaves)

    def test_node_name_pulled_from_attributes(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=_triangle_graph())
        metrics = analyzer.calculate_centrality()
        assert metrics["a"].node_name == "Alice"

    def test_claim_node_name_is_composed_triple(self) -> None:
        g = nx.Graph()
        g.add_node(
            "claim1",
            node_type="claim",
            source_name="Alice",
            target_name="Acme",
            type="WORKS_AT",
        )
        g.add_node("b", name="Bob")
        g.add_edge("claim1", "b")
        analyzer = GraphAnalyzer(Config(), graph=g)
        metrics = analyzer.calculate_centrality()
        assert metrics["claim1"].node_name == "Alice -> WORKS_AT -> Acme"

    def test_disabled_metric_stays_none(self) -> None:
        config = Config()
        # closeness/eigenvector default to off; degree/betweenness/pagerank on.
        analyzer = GraphAnalyzer(config, graph=_triangle_graph())
        metrics = analyzer.calculate_centrality()
        node = metrics["a"]
        assert node.closeness is None
        assert node.eigenvector is None
        assert node.degree is not None
        assert node.pagerank is not None

    def test_enabling_closeness_populates_it(self) -> None:
        config = Config()
        config.graph.analysis.centrality.calculate_closeness = True
        analyzer = GraphAnalyzer(config, graph=_triangle_graph())
        metrics = analyzer.calculate_centrality()
        assert metrics["a"].closeness == pytest.approx(1.0)

    def test_centrality_cache_populated_for_enabled_methods(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=_triangle_graph())
        analyzer.calculate_centrality()
        # degree/betweenness/pagerank are enabled by default.
        assert "degree" in analyzer.centrality_cache
        assert "pagerank" in analyzer.centrality_cache
        assert "closeness" not in analyzer.centrality_cache


class TestStatistics:
    def test_empty_graph_zero_counts(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=nx.Graph())
        stats = analyzer.get_graph_statistics()
        assert stats.num_nodes == 0
        assert stats.num_edges == 0
        assert stats.density is None

    def test_none_graph_returns_zero_stats(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=None)
        stats = analyzer.get_graph_statistics()
        assert stats.num_nodes == 0
        assert stats.num_edges == 0

    def test_triangle_node_and_edge_counts(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=_triangle_graph())
        stats = analyzer.get_graph_statistics()
        assert stats.num_nodes == 3
        assert stats.num_edges == 3

    def test_fully_connected_density_is_one(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=_triangle_graph())
        stats = analyzer.get_graph_statistics()
        assert stats.density == pytest.approx(1.0)

    def test_connected_components_single_for_triangle(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=_triangle_graph())
        stats = analyzer.get_graph_statistics()
        assert stats.num_connected_components == 1
        assert stats.largest_component_size == 3

    def test_two_components_reported(self) -> None:
        g = nx.Graph()
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        g.add_node("e")  # isolated -> third component
        analyzer = GraphAnalyzer(Config(), graph=g)
        stats = analyzer.get_graph_statistics()
        assert stats.num_connected_components == 3
        assert stats.largest_component_size == 2

    def test_diameter_only_when_enabled_and_connected(self) -> None:
        config = Config()
        config.graph.analysis.statistics.calculate_diameter = True
        # path a-b-c has diameter 2.
        g = nx.Graph()
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        analyzer = GraphAnalyzer(config, graph=g)
        stats = analyzer.get_graph_statistics()
        assert stats.diameter == 2

    def test_diameter_skipped_when_disconnected(self) -> None:
        config = Config()
        config.graph.analysis.statistics.calculate_diameter = True
        g = nx.Graph()
        g.add_edge("a", "b")
        g.add_edge("c", "d")
        analyzer = GraphAnalyzer(config, graph=g)
        stats = analyzer.get_graph_statistics()
        # >1 component -> diameter computation is skipped, stays None.
        assert stats.diameter is None

    def test_statistics_are_cached(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=_triangle_graph())
        first = analyzer.get_graph_statistics()
        second = analyzer.get_graph_statistics()
        assert first is second  # cached instance returned

    def test_setting_graph_clears_cache(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=_triangle_graph())
        analyzer.calculate_centrality()
        analyzer.get_graph_statistics()
        assert analyzer.centrality_cache
        analyzer.graph = _star_graph()
        assert analyzer.centrality_cache == {}
        assert analyzer._statistics_cache is None


class TestExportGraphData:
    def test_empty_graph_export_shape(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=None)
        data = analyzer.export_graph_data()
        assert data == {"nodes": [], "edges": [], "statistics": {}}

    def test_export_shape_and_counts(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=_triangle_graph())
        data = analyzer.export_graph_data()
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 3
        assert data["statistics"]["num_nodes"] == 3

    def test_node_ids_are_strings_with_attributes(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=_triangle_graph())
        data = analyzer.export_graph_data()
        node = next(n for n in data["nodes"] if n["id"] == "a")
        assert isinstance(node["id"], str)
        assert node["attributes"]["name"] == "Alice"

    def test_edges_have_source_and_target(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=_triangle_graph())
        data = analyzer.export_graph_data()
        for edge in data["edges"]:
            assert "source" in edge
            assert "target" in edge
            assert isinstance(edge["source"], str)

    def test_export_includes_centrality_after_calculation(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=_triangle_graph())
        analyzer.calculate_centrality()  # fills centrality cache
        data = analyzer.export_graph_data()
        node = data["nodes"][0]
        # cache keys are surfaced as "<name>_centrality" on each node.
        assert "degree_centrality" in node
        assert node["degree_centrality"] == pytest.approx(1.0)

    def test_export_without_centrality_has_no_centrality_keys(self) -> None:
        analyzer = GraphAnalyzer(Config(), graph=_triangle_graph())
        data = analyzer.export_graph_data()  # no calculate_centrality call
        node = data["nodes"][0]
        assert not any(k.endswith("_centrality") for k in node)
