# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the standalone visualization CLI + renderer registry (M4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import pytest

from tests.fixtures.extras import requires_viz
from unified_kg_rag.adapters.renderers import (
    BaseRenderer,
    RenderContext,
    get_renderer_class,
    register_renderer,
    registered_renderers,
    render_all,
)
from unified_kg_rag.application.cli import run_visualization as cli_module
from unified_kg_rag.application.cli.run_visualization import (
    load_render_context,
    run_visualization,
)
from unified_kg_rag.domain.models import Config
from unified_kg_rag.shared import ConfigurationError

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
    @pytest.mark.usefixtures("restore_renderer_registry")
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

    @pytest.mark.usefixtures("restore_renderer_registry")
    def test_checks_each_renderer_once_before_loading(
        self, tmp_path: Path, config: Config
    ) -> None:
        checks: list[str] = []

        @register_renderer("checked_for_cli_test")
        class _Checked(BaseRenderer):
            @classmethod
            def check_dependencies(cls) -> None:
                checks.append("checked")

            def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
                return []

        # Checked before the data is read: a missing file fails after it.
        with pytest.raises(FileNotFoundError):
            run_visualization(
                tmp_path / "missing.json", tmp_path, ["checked_for_cli_test"], config
            )
        assert checks == ["checked"]
        checks.clear()
        data = tmp_path / "viz.json"
        data.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
        run_visualization(data, tmp_path / "out", ["checked_for_cli_test"], config)
        assert checks == ["checked"]

    @requires_viz
    def test_data_only_export_gets_a_layout_and_centrality(
        self, tmp_path: Path, config: Config
    ) -> None:
        # The file ingestion writes (graph.visualization.export_data).
        from unified_kg_rag.domain.ingestion.visualization_data import graph_data

        graph = nx.Graph()
        graph.add_edge("e1", "e2")
        graph.add_edge("e2", "e3")
        path = tmp_path / "data.json"
        path.write_text(
            json.dumps(graph_data(graph, {"hierarchy": []})), encoding="utf-8"
        )
        written = run_visualization(
            path, tmp_path / "out", ["interactive", "static"], config
        )
        assert written
        context = load_render_context(path)
        cli_module._fill_missing_analysis(context, config)
        assert set(context.layout) == {"e1", "e2", "e3"}
        assert context.centrality["e2"].degree == 1.0

    def test_exported_edges_do_not_repeat_their_relationships(self) -> None:
        # Regression: every edge carried a copy of each relationship it merges
        # under attributes.relationships, about half of the edge bytes, though
        # no renderer reads it.
        from unified_kg_rag.domain.ingestion.graph_builder import GraphBuilder
        from unified_kg_rag.domain.ingestion.visualization_data import graph_data
        from unified_kg_rag.domain.models import Entity, Relationship

        entities = [
            Entity(id="e1", name="Vendor", embedding=[0.1]),
            Entity(id="e2", name="Buyer"),
        ]
        relationships = [
            Relationship(id=f"r{i}", source_id="e1", target_id="e2", type="SUPPLIES")
            for i in range(2)
        ]
        graph = GraphBuilder(entities, relationships).build()
        assert len(graph.edges["e1", "e2"]["relationships"]) == 2  # live graph
        data = graph_data(graph, {"hierarchy": []})
        (edge,) = data["edges"]
        assert "relationships" not in edge["attributes"]
        assert edge["attributes"]["type"] == "SUPPLIES"
        assert edge["attributes"]["weight"] == 2.0
        assert "embedding" not in data["nodes"][0]["attributes"]

    def test_data_only_export_lays_out_only_the_drawn_nodes(
        self, tmp_path: Path, config: Config
    ) -> None:
        from unified_kg_rag.domain.ingestion.visualization_data import graph_data

        graph = nx.star_graph(5)
        path = tmp_path / "data.json"
        path.write_text(
            json.dumps(graph_data(graph, {"hierarchy": []})), encoding="utf-8"
        )
        config.graph.visualization.interactive = {"max_nodes": 3}
        context = load_render_context(path)
        cli_module._fill_missing_analysis(context, config)
        # The hub (degree 5) and two leaves: the nodes the page draws.
        assert len(context.layout) == 3 and "0" in context.layout


class TestRendererRegistry:
    def test_builtin_renderers_registered(self) -> None:
        assert {"interactive", "static"} <= set(registered_renderers())

    def test_unknown_renderer_raises(self) -> None:
        with pytest.raises(ValueError, match="No renderer registered"):
            get_renderer_class("does-not-exist")

    @pytest.mark.usefixtures("restore_renderer_registry")
    def test_render_all_skips_a_failing_renderer_and_passes_each_its_block(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The one render loop the manager and run-visualization share.
        seen: dict[str, object] = {}

        @register_renderer("broken_for_test")
        class _Broken(BaseRenderer):
            def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
                raise RuntimeError("cannot draw")

        @register_renderer("ok_for_test")
        class _Ok(BaseRenderer):
            def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
                seen["config"] = self.config
                return [output_dir / "ok.txt"]

        viz_config = type("V", (), {"ok_for_test": {"size": 3}})()
        written = render_all(
            RenderContext(graph=nx.Graph()),
            tmp_path,
            ["broken_for_test", "ok_for_test"],
            viz_config,
        )
        assert written == [tmp_path / "ok.txt"]
        assert seen["config"] == {"size": 3}
        assert "Renderer 'broken_for_test' failed: cannot draw" in caplog.text
        with pytest.raises(ValueError, match="No renderer registered"):
            render_all(RenderContext(graph=nx.Graph()), tmp_path, ["nope"], None)


def _write_empty_export(tmp_path: Path) -> Path:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    return path


@requires_viz
class TestBuiltinAdaptersReportOnlyWrittenFiles:
    @staticmethod
    def _empty_context() -> RenderContext:
        return RenderContext(graph=nx.Graph(), layout={})

    def test_interactive_empty_graph_reports_nothing(self, tmp_path: Path) -> None:
        renderer = get_renderer_class("interactive")({})
        assert renderer.render(self._empty_context(), tmp_path) == []
        assert not (tmp_path / "interactive_graph.html").exists()

    def test_stale_output_is_not_reported(self, tmp_path: Path) -> None:
        # A file left by an earlier run must not be reported as freshly written.
        stale = tmp_path / "interactive_graph.html"
        stale.write_text("old", encoding="utf-8")
        renderer = get_renderer_class("interactive")({})
        assert renderer.render(self._empty_context(), tmp_path) == []
        assert not stale.exists()

    def test_reported_paths_all_exist(
        self, export_json: Path, tmp_path: Path, config: Config
    ) -> None:
        written = run_visualization(
            export_json, tmp_path / "out", ["interactive", "static"], config
        )
        assert written
        assert all(p.is_file() for p in written)

    def test_empty_export_renders_nothing(self, tmp_path: Path, config: Config) -> None:
        written = run_visualization(
            _write_empty_export(tmp_path),
            tmp_path / "out",
            ["interactive", "static"],
            config,
        )
        assert written == []

    def test_main_returns_nonzero_when_nothing_rendered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        argv = [
            "run-visualization",
            "--data-path",
            str(_write_empty_export(tmp_path)),
            "--output-directory",
            str(tmp_path / "out"),
            # Explicit built-ins: other tests register extra renderers globally.
            "--renderers",
            "interactive",
            "static",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(cli_module, "load_cli_config", lambda _args: Config())
        assert cli_module.main() == 1


def _without_module(
    monkeypatch: pytest.MonkeyPatch, package: str, *loaded: str
) -> None:
    """Make ``import package`` fail as on an install without it, and forget
    the renderer modules (``loaded``) that already imported it."""
    for name in [m for m in sys.modules if m == package or m.startswith(f"{package}.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, package, None)
    for name in loaded:
        monkeypatch.delitem(sys.modules, name, raising=False)


class TestWithoutTheVizExtra:
    @pytest.mark.parametrize(
        ("renderer", "package"), [("interactive", "pyvis"), ("static", "bokeh")]
    )
    def test_a_builtin_renderer_names_the_extra(
        self, renderer: str, package: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _without_module(
            monkeypatch, package, f"unified_kg_rag.adapters.renderers.{renderer}"
        )
        renderer_cls = get_renderer_class(renderer)
        with pytest.raises(ConfigurationError, match=rf"'{package}'.*--extra viz"):
            renderer_cls.check_dependencies()
        with pytest.raises(ConfigurationError, match="--extra viz"):
            renderer_cls({}).render(RenderContext(graph=nx.Graph()), Path("."))

    def test_render_all_fails_before_rendering_anything(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Not logged and skipped like a failing renderer: the static charts
        # are not written either when the interactive page cannot be.
        _without_module(
            monkeypatch, "pyvis", "unified_kg_rag.adapters.renderers.interactive"
        )
        graph = nx.Graph([("a", "b")])
        with pytest.raises(ConfigurationError, match="--extra viz"):
            render_all(
                RenderContext(graph=graph), tmp_path, ["static", "interactive"], None
            )
        assert list(tmp_path.iterdir()) == []

    def test_main_exits_2_naming_the_extra(
        self,
        export_json: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _without_module(
            monkeypatch, "bokeh", "unified_kg_rag.adapters.renderers.static"
        )
        argv = [
            "run-visualization",
            "--data-path",
            str(export_json),
            "--output-directory",
            str(tmp_path / "out"),
            "--renderers",
            "static",
            "--output-format",
            "json",
        ]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(cli_module, "load_cli_config", lambda _args: Config())
        assert cli_module.main() == 2
        assert "--extra viz" in json.loads(capsys.readouterr().out)["error"]
        assert not (tmp_path / "out").exists()
