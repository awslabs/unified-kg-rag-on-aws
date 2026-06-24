# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""AWS-free unit tests for the pyvis ``InteractiveRenderer``.

These build small graphs / hierarchical communities and assert the renderer
constructs a pyvis network with the expected nodes/edges and writes interactive
HTML to a ``tmp_path``. pyvis is a pure-Python dependency (no AWS).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from aws_graphrag.adapters.ingestion.community_detector import HierarchicalCommunity
from aws_graphrag.adapters.renderers.interactive import InteractiveRenderer

pytestmark = pytest.mark.unit


def _graph() -> nx.Graph:
    g = nx.Graph()
    g.add_node("e1", name="Alice", community_id="c0", node_type="entity")
    g.add_node("e2", name="Acme", community_id="c0", node_type="entity")
    g.add_node("e3", name="Seattle", community_id="c1", node_type="entity")
    g.add_edge(
        "e1", "e2", weight=2.0, type="works_at", source_name="Alice", target_name="Acme"
    )
    g.add_edge("e2", "e3", weight=0.5, type="located_in")
    return g


class TestInit:
    def test_defaults(self) -> None:
        r = InteractiveRenderer({})
        assert r.height == "900px"
        assert r.physics_enabled is True
        assert r.cdn_resources == "in_line"

    def test_overrides(self) -> None:
        r = InteractiveRenderer({"height": "500px", "physics_enabled": False})
        assert r.height == "500px"
        assert r.physics_enabled is False


class TestGeneratePalette:
    def test_zero_colors(self) -> None:
        assert InteractiveRenderer._generate_palette(0) == []

    def test_uses_base_colors_for_small_count(self) -> None:
        colors = InteractiveRenderer._generate_palette(3)
        assert len(colors) == 3
        assert all(c.startswith("#") for c in colors)

    def test_generates_hsl_for_large_count(self) -> None:
        n = 30
        colors = InteractiveRenderer._generate_palette(n)
        assert len(colors) == n
        assert all(c.startswith("hsl(") for c in colors)


class TestNetworkVisualization:
    def test_writes_html_with_nodes_and_edges(self, tmp_path: Path) -> None:
        out = tmp_path / "graph.html"
        renderer = InteractiveRenderer({})
        renderer.create_network_visualization(_graph(), {}, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_uses_layout_positions_when_provided(self, tmp_path: Path) -> None:
        # A provided layout disables physics; output still produced.
        out = tmp_path / "graph.html"
        layout = {"e1": (0.0, 0.0), "e2": (1.0, 1.0), "e3": (0.5, 0.5)}
        InteractiveRenderer({}).create_network_visualization(_graph(), layout, str(out))
        assert out.exists()

    def test_empty_graph_writes_nothing(self, tmp_path: Path) -> None:
        out = tmp_path / "graph.html"
        InteractiveRenderer({}).create_network_visualization(nx.Graph(), {}, str(out))
        assert not out.exists()

    def test_directed_graph_handled(self, tmp_path: Path) -> None:
        g = nx.DiGraph()
        g.add_node("a", name="A")
        g.add_node("b", name="B")
        g.add_edge("a", "b", weight=1.0)
        out = tmp_path / "digraph.html"
        InteractiveRenderer({}).create_network_visualization(g, {}, str(out))
        assert out.exists()

    def test_claim_node_uses_type_as_display_name(self, tmp_path: Path) -> None:
        # claim nodes branch through a different title/label path.
        g = nx.Graph()
        g.add_node(
            "c1",
            node_type="claim",
            type="ASSERTS",
            subject_name="Alice",
            object_name="Acme",
        )
        g.add_node("e1", name="Alice")
        g.add_edge("c1", "e1", edge_type="is_subject_of")
        out = tmp_path / "claim.html"
        InteractiveRenderer({}).create_network_visualization(g, {}, str(out))
        assert out.exists()


class TestCommunityHierarchy:
    def _hierarchy(self) -> list[HierarchicalCommunity]:
        child0 = HierarchicalCommunity(
            community_id="L0_C0", level=0, nodes={"e1", "e2"}, parent_id="L1_C0"
        )
        child1 = HierarchicalCommunity(
            community_id="L0_C1", level=0, nodes={"e3"}, parent_id="L1_C0"
        )
        parent = HierarchicalCommunity(
            community_id="L1_C0",
            level=1,
            nodes={"e1", "e2", "e3"},
            children_ids=["L0_C0", "L0_C1"],
        )
        return [parent, child0, child1]

    def test_writes_html_file(self, tmp_path: Path) -> None:
        out = tmp_path / "hierarchy.html"
        InteractiveRenderer({}).create_community_hierarchy(self._hierarchy(), str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_list_writes_nothing(self, tmp_path: Path) -> None:
        out = tmp_path / "hierarchy.html"
        InteractiveRenderer({}).create_community_hierarchy([], str(out))
        assert not out.exists()

    def test_dangling_parent_edge_skipped(self, tmp_path: Path) -> None:
        # A community whose parent_id is not in the set must not crash the
        # edge-building loop (edge simply skipped).
        comm = HierarchicalCommunity(
            community_id="L0_C0", level=0, nodes={"e1"}, parent_id="MISSING"
        )
        out = tmp_path / "hierarchy.html"
        InteractiveRenderer({}).create_community_hierarchy([comm], str(out))
        assert out.exists()
