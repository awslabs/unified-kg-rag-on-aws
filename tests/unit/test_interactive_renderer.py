# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for the pyvis ``InteractiveRenderer``.

These build small graphs / hierarchical communities and assert the renderer
constructs a pyvis network with the expected nodes/edges and writes interactive
HTML to a ``tmp_path``. pyvis is a pure-Python dependency (no AWS).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from unified_kg_rag.adapters.ingestion.community_detector import HierarchicalCommunity
from unified_kg_rag.adapters.renderers.interactive import InteractiveRenderer

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


class TestEdgeWeightNormalization:
    @staticmethod
    def _weighted_graph(weights: list[float]) -> nx.Graph:
        g = nx.Graph()
        for i, w in enumerate(weights):
            g.add_edge(f"a{i}", f"b{i}", weight=w)
        return g

    def test_strength_scale_weights_span_the_full_range(self) -> None:
        # Relationship strengths are 1-10; the old min(weight, 1.0) cap mapped
        # every one of them to 1.0 (max width).
        g = self._weighted_graph([1.0, 5.0, 10.0])
        norm = InteractiveRenderer._normalized_edge_weights(g)
        values = [norm[("a0", "b0")], norm[("a1", "b1")], norm[("a2", "b2")]]
        assert values[0] == pytest.approx(0.0)
        assert values[2] == pytest.approx(1.0)
        assert 0.0 < values[1] < 1.0
        assert values == sorted(values)

    def test_log_scaling_dampens_large_merge_counts(self) -> None:
        g = self._weighted_graph([1.0, 2.0, 100.0])
        norm = InteractiveRenderer._normalized_edge_weights(g)
        # Linear min-max would put weight 2 at ~0.01; log scaling keeps it visible.
        assert norm[("a1", "b1")] > 0.05

    def test_uniform_weights_render_at_mid_width(self) -> None:
        g = self._weighted_graph([3.0, 3.0])
        norm = InteractiveRenderer._normalized_edge_weights(g)
        assert set(norm.values()) == {0.5}

    def test_invalid_weights_default_to_one(self) -> None:
        g = nx.Graph()
        g.add_edge("a", "b", weight="n/a")
        g.add_edge("c", "d")
        norm = InteractiveRenderer._normalized_edge_weights(g)
        assert set(norm.values()) == {0.5}

    def test_rendered_edge_widths_differ(self, mocker) -> None:  # noqa: ANN001
        renderer = InteractiveRenderer({})
        net = renderer._init_network()
        g = self._weighted_graph([1.0, 10.0])
        for node in g.nodes:
            net.add_node(node)
        spy = mocker.spy(net, "add_edge")
        renderer._add_edges(net, g)
        widths = sorted(call.kwargs["width"] for call in spy.call_args_list)
        assert widths[0] == pytest.approx(InteractiveRenderer.MIN_EDGE_WIDTH)
        assert widths[1] == pytest.approx(InteractiveRenderer.MAX_EDGE_WIDTH)
