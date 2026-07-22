# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the standalone visualization CLI + renderer registry (M4)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from unified_kg_rag.adapters.renderers import (
    BaseRenderer,
    RenderContext,
    get_renderer_class,
    register_renderer,
    registered_renderers,
)
from unified_kg_rag.application.cli.run_visualization import (
    load_render_context,
    run_visualization,
)
from unified_kg_rag.domain.models import Config

pytestmark = pytest.mark.unit


@pytest.fixture
def export_json(tmp_path: Path) -> Path:
    # Mirrors the real export_visualization_data shape: graph_analyzer's
    # nodes/edges + community_detector's {"hierarchy": [...]} under "communities".
    data = {
        "nodes": [
            {"id": "e1", "attributes": {"name": "Alice"}},
            {"id": "e2", "attributes": {"name": "Acme"}},
        ],
        "edges": [{"source": "e1", "target": "e2", "attributes": {"weight": 1.0}}],
        "layout": {"e1": [0.0, 0.0], "e2": [1.0, 1.0]},
        "communities": {
            "resolution": 1.0,
            "hierarchy": [
                {
                    "community_id": "L0_C0",
                    "level": 0,
                    "nodes": ["e1", "e2"],
                    "size": 2,
                    "parent": None,
                    "children": [],
                }
            ],
        },
        # Export shape: node_id -> CentralityMetrics dict.
        "centrality": {
            "e1": {"node_id": "e1", "node_name": "Alice", "degree": 1.0},
            "e2": {"node_id": "e2", "node_name": "Bob", "degree": 0.5},
        },
    }
    path = tmp_path / "viz.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestLoadRenderContext:
    def test_reconstructs_graph(self, export_json: Path) -> None:
        ctx = load_render_context(export_json)
        assert ctx.graph.number_of_nodes() == 2
        assert ctx.graph.number_of_edges() == 1
        assert ctx.graph.nodes["e1"]["name"] == "Alice"
        assert ctx.layout == {"e1": [0.0, 0.0], "e2": [1.0, 1.0]}

    def test_rehydrates_typed_centrality(self, export_json: Path) -> None:
        # Centrality must round-trip into CentralityMetrics so the standalone
        # static renderer can draw the centrality comparison plot.
        ctx = load_render_context(export_json)
        assert set(ctx.centrality) == {"e1", "e2"}
        assert ctx.centrality["e1"].node_name == "Alice"
        assert ctx.centrality["e1"].degree == 1.0

    def test_rehydrates_typed_communities(self, export_json: Path) -> None:
        ctx = load_render_context(export_json)
        # Static renderer needs Community objects with .size and str level.
        assert len(ctx.communities) == 1
        assert ctx.communities[0].id == "L0_C0"
        assert ctx.communities[0].size == 2
        assert ctx.communities[0].level == "0"
        # Interactive renderer needs HierarchicalCommunity objects with .nodes.
        assert len(ctx.community_hierarchy) == 1
        assert ctx.community_hierarchy[0].community_id == "L0_C0"
        assert ctx.community_hierarchy[0].nodes == {"e1", "e2"}

    def test_handles_list_communities_fallback(self, tmp_path: Path) -> None:
        # Tolerate a bare-list communities payload too.
        data = {
            "nodes": [],
            "edges": [],
            "communities": [
                {"community_id": "x", "level": 0, "nodes": [], "children": []}
            ],
        }
        path = tmp_path / "v.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        ctx = load_render_context(path)
        assert ctx.community_hierarchy[0].community_id == "x"


class TestRunVisualization:
    def test_drives_registered_renderer(
        self, export_json: Path, tmp_path: Path, config: Config
    ) -> None:
        calls: list[RenderContext] = []

        @register_renderer("fake_for_cli_test")
        class _FakeRenderer(BaseRenderer):
            def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
                calls.append(context)
                out = output_dir / "fake.txt"
                out.write_text("ok", encoding="utf-8")
                return [out]

        out_dir = tmp_path / "out"
        written = run_visualization(export_json, out_dir, ["fake_for_cli_test"], config)
        assert len(calls) == 1
        assert calls[0].graph.number_of_nodes() == 2
        assert written == [out_dir / "fake.txt"]
        assert (out_dir / "fake.txt").read_text() == "ok"


class TestRendererRegistry:
    def test_builtin_renderers_registered(self) -> None:
        assert {"interactive", "static"} <= set(registered_renderers())

    def test_unknown_renderer_raises(self) -> None:
        with pytest.raises(ValueError, match="No renderer registered"):
            get_renderer_class("does-not-exist")
