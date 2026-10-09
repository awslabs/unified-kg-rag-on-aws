# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for the Bokeh ``StaticRenderer``.

These build small NetworkX graphs / centrality / community inputs and assert
the renderer constructs Bokeh figures and writes HTML output files to a
``tmp_path``. Bokeh is a pure-Python dependency (no AWS).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest
from bokeh.plotting import figure

from unified_kg_rag.adapters.renderers.static import StaticRenderer
from unified_kg_rag.domain.ingestion.graph_analyzer import CentralityMetrics
from unified_kg_rag.domain.models import Community

pytestmark = pytest.mark.unit


def _graph() -> nx.Graph:
    g = nx.Graph()
    for i in range(6):
        g.add_node(f"n{i}", name=f"N{i}")
    g.add_edge("n0", "n1", weight=1.0)
    g.add_edge("n1", "n2", weight=2.0)
    g.add_edge("n0", "n2", weight=0.5)
    g.add_edge("n3", "n4")
    g.add_edge("n4", "n5")
    return g


def _community(cid: str, level: str, size: int) -> Community:
    return Community(
        id=cid,
        name=cid,
        level=level,
        parent="",
        children=[],
        entity_ids=[],
        text_unit_ids=[],
        size=size,
    )


class TestInit:
    def test_defaults_applied(self) -> None:
        r = StaticRenderer({})
        assert r.figure_width == 900
        assert r.figure_height == 600
        assert r.log_scale_x is False

    def test_config_overrides(self) -> None:
        r = StaticRenderer({"figure_width": 400, "log_scale_x": True})
        assert r.figure_width == 400
        assert r.log_scale_x is True


class TestCreateFigure:
    def test_returns_bokeh_figure_with_labels(self) -> None:
        r = StaticRenderer({})
        p = r._create_figure("Title", "X label", "Y label")
        assert isinstance(p, figure)
        assert p.xaxis[0].axis_label == "X label"
        assert p.yaxis[0].axis_label == "Y label"


class TestDegreeDistribution:
    def test_writes_html_file(self, tmp_path: Path) -> None:
        out = tmp_path / "degree.html"
        StaticRenderer({}).plot_degree_distribution(_graph(), str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_log_scale_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "degree_log.html"
        StaticRenderer({"log_scale_x": True}).plot_degree_distribution(
            _graph(), str(out)
        )
        assert out.exists()

    def test_empty_graph_writes_nothing(self, tmp_path: Path) -> None:
        out = tmp_path / "empty.html"
        StaticRenderer({}).plot_degree_distribution(nx.Graph(), str(out))
        assert not out.exists()


class TestCentralityComparison:
    def _centrality(self) -> dict[str, CentralityMetrics]:
        return {
            "n0": CentralityMetrics(
                node_id="n0",
                node_name="N0",
                degree=1.0,
                betweenness=0.5,
                pagerank=0.3,
            ),
            "n1": CentralityMetrics(
                node_id="n1",
                node_name="N1",
                degree=0.5,
                betweenness=0.2,
                pagerank=0.1,
            ),
        }

    def test_writes_html_file(self, tmp_path: Path) -> None:
        out = tmp_path / "centrality.html"
        StaticRenderer({}).plot_centrality_comparison(self._centrality(), str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_data_writes_nothing(self, tmp_path: Path) -> None:
        out = tmp_path / "centrality.html"
        StaticRenderer({}).plot_centrality_comparison({}, str(out))
        assert not out.exists()

    def test_duplicate_node_names_get_unique_factors(
        self, tmp_path: Path, mocker, caplog
    ) -> None:
        # Two claims with the same subject/type/object share a display name;
        # repeated x-axis factors are a Bokeh DUPLICATE_FACTORS error.
        name = "Vendor -> Buyer (obligation)"
        data = {
            f"c{i}": CentralityMetrics(
                node_id=f"c{i}", node_name=name, degree=1.0 - i / 10
            )
            for i in range(3)
        }
        save = mocker.spy(StaticRenderer, "_save_plot")
        out = tmp_path / "centrality.html"
        StaticRenderer({}).plot_centrality_comparison(data, str(out))
        assert out.exists()
        factors = list(save.call_args.args[0].x_range.factors)
        assert factors == [name, f"{name} (2)", f"{name} (3)"]
        assert "DUPLICATE_FACTORS" not in caplog.text

    def test_unique_factors_skips_suffixes_already_taken(self) -> None:
        out = StaticRenderer._unique_factors(["A", "A", "A (2)", "B"])
        assert out == ["A", "A (2)", "A (2) (2)", "B"]
        assert len(set(out)) == len(out)

    def test_no_named_nodes_writes_nothing(self, tmp_path: Path) -> None:
        # All nodes lack a node_name -> nothing to plot.
        data = {"n0": CentralityMetrics(node_id="n0", node_name=None, degree=1.0)}
        out = tmp_path / "centrality.html"
        StaticRenderer({}).plot_centrality_comparison(data, str(out))
        assert not out.exists()


class TestCommunitySizeDistribution:
    def test_writes_html_file(self, tmp_path: Path) -> None:
        communities = [
            _community("L0_C0", "0", 3),
            _community("L0_C1", "0", 5),
            _community("L0_C2", "0", 2),
        ]
        out = tmp_path / "comm.html"
        StaticRenderer({}).plot_community_size_distribution(communities, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_list_writes_nothing(self, tmp_path: Path) -> None:
        out = tmp_path / "comm.html"
        StaticRenderer({}).plot_community_size_distribution([], str(out))
        assert not out.exists()

    def test_only_level0_communities_counted(self, tmp_path: Path) -> None:
        # Only level-0 communities are plotted; a level-1-only input has no
        # valid sizes and writes nothing.
        communities = [_community("L1_C0", "1", 10)]
        out = tmp_path / "comm.html"
        StaticRenderer({}).plot_community_size_distribution(communities, str(out))
        assert not out.exists()


class TestSavePlot:
    def test_save_plot_writes_to_path(self, tmp_path: Path) -> None:
        r = StaticRenderer({})
        p = r._create_figure("T", "x", "y")
        out = tmp_path / "plot.html"
        r._save_plot(p, str(out), "T")
        assert out.exists()
        assert "<html" in out.read_text(encoding="utf-8").lower()
