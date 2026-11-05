# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for the HTML/SVG ``StaticRenderer``.

These build small NetworkX graphs / centrality / community inputs, render
them to a ``tmp_path`` and parse the pages with ``html.parser``. The renderer
needs no optional extra, so these run on a core install too.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

import networkx as nx
import pytest

from unified_kg_rag.adapters.renderers.static import (
    TOP_CENTRALITY_NODES,
    StaticRenderer,
)
from unified_kg_rag.domain.ingestion.graph_analyzer import CentralityMetrics
from unified_kg_rag.domain.models import Community

pytestmark = pytest.mark.unit

_HOSTILE = '<script>alert("x")</script> & Co'


@dataclass
class _Element:
    tag: str
    attrs: dict[str, str | None]
    text: str = ""
    children: list[_Element] = field(default_factory=list)

    def title(self) -> str:
        return next((c.text for c in self.children if c.tag == "title"), "")


class _Page(HTMLParser):
    """Every element of a page, with the text directly inside it."""

    def __init__(self, html: str) -> None:
        super().__init__()
        self.elements: list[_Element] = []
        self._open: list[_Element] = []
        self.feed(html)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element = _Element(tag, dict(attrs))
        if self._open:
            self._open[-1].children.append(element)
        self.elements.append(element)
        self._open.append(element)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self._open.pop()

    def handle_endtag(self, tag: str) -> None:
        while self._open:
            if self._open.pop().tag == tag:
                break

    def handle_data(self, data: str) -> None:
        if self._open:
            self._open[-1].text += data

    def find(self, tag: str, **attrs: str) -> list[_Element]:
        return [
            e
            for e in self.elements
            if e.tag == tag and all(e.attrs.get(k) == v for k, v in attrs.items())
        ]

    def bars(self, metric: str | None = None) -> list[_Element]:
        bars = self.find("rect", **{"class": "bar"})
        if metric is None:
            return bars
        return [b for b in bars if b.attrs.get("data-metric") == metric]


def _page(path: Path) -> _Page:
    return _Page(path.read_text(encoding="utf-8"))


def _counted(page: _Page) -> int:
    """The sum of the histogram bars' counts, read from their tooltips."""
    total = 0
    for bar in page.bars():
        match = re.search(r"Count: ([\d,]+)", bar.title())
        assert match, bar.title()
        total += int(match.group(1).replace(",", ""))
    return total


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


class TestPage:
    def test_one_self_contained_svg_chart(self, tmp_path: Path) -> None:
        out = tmp_path / "degree.html"
        StaticRenderer({"figure_width": 640}).plot_degree_distribution(
            _graph(), str(out)
        )
        page = _page(out)
        (svg,) = page.find("svg")
        assert svg.attrs["data-chart"] == "degree_distribution"
        assert svg.attrs["width"] == "640"
        assert page.find("title")[0].text == "Degree Distribution"
        # Nothing to load: no script, stylesheet or remote reference.
        assert not page.find("script") and not page.find("link")
        html = out.read_text(encoding="utf-8")
        for namespace in ("http://www.w3.org/2000/svg", "http://www.w3.org/1999/xhtml"):
            html = html.replace(f'xmlns="{namespace}"', "")
        assert "http" not in html

    def test_configured_font_and_colour_are_escaped(self, tmp_path: Path) -> None:
        out = tmp_path / "degree.html"
        hostile = '"><script>alert(1)</script>'
        StaticRenderer(
            {"font": hostile, "background_color": hostile}
        ).plot_degree_distribution(_graph(), str(out))
        page = _page(out)
        assert not page.find("script")
        assert page.find("svg")[0].attrs["font-family"] == hostile
        assert page.find("rect", fill=hostile)


class TestDegreeDistribution:
    def test_bars_count_every_node(self, tmp_path: Path) -> None:
        out = tmp_path / "degree.html"
        StaticRenderer({}).plot_degree_distribution(_graph(), str(out))
        page = _page(out)
        # Degrees 2, 2, 2, 1, 2, 1: two bins with nodes.
        assert len(page.bars()) == 2
        assert _counted(page) == 6
        labels = [e.text for e in page.find("div", **{"class": "axis-label"})]
        assert labels == ["Number of Nodes", "Degree"]

    def test_log_scale_writes_file(self, tmp_path: Path) -> None:
        out = tmp_path / "degree_log.html"
        graph = nx.barabasi_albert_graph(200, 2, seed=1)
        StaticRenderer({"log_scale_x": True}).plot_degree_distribution(graph, str(out))
        page = _page(out)
        assert _counted(page) == 200
        ticks = [e.text for e in page.find("div", **{"class": "tick-label"})]
        assert "10" in ticks

    def test_log_scale_with_one_degree_value(self, tmp_path: Path) -> None:
        # Every node has degree 2: no log range, so the axis is linear.
        out = tmp_path / "degree_log.html"
        StaticRenderer({"log_scale_x": True}).plot_degree_distribution(
            nx.cycle_graph(5), str(out)
        )
        assert _counted(_page(out)) == 5

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

    @staticmethod
    def _labels(page: _Page) -> list[str]:
        return [e.text for e in page.find("div", **{"class": "tick-label"})][
            # The y-axis ticks come first.
            len(page.find("line", **{"stroke-dasharray": "4 4"})) :
        ]

    def test_three_metric_bars_per_node(self, tmp_path: Path) -> None:
        out = tmp_path / "centrality.html"
        StaticRenderer({}).plot_centrality_comparison(self._centrality(), str(out))
        page = _page(out)
        assert page.find("svg")[0].attrs["data-chart"] == "centrality_comparison"
        for metric in ("degree", "betweenness", "pagerank"):
            assert len(page.bars(metric)) == 2
        assert self._labels(page) == ["N0", "N1"]
        legend = [e.text for e in page.find("div", **{"class": "legend-label"})]
        assert legend == ["Degree", "Betweenness", "Pagerank"]
        assert page.bars("pagerank")[0].title() == "N0\nPagerank: 0.300"

    def test_names_are_escaped(self, tmp_path: Path) -> None:
        data = {"n0": CentralityMetrics(node_id="n0", node_name=_HOSTILE, degree=1.0)}
        out = tmp_path / "centrality.html"
        StaticRenderer({}).plot_centrality_comparison(data, str(out))
        page = _page(out)
        assert not page.find("script")
        assert self._labels(page) == [_HOSTILE]
        assert page.find("div", **{"class": "tick-label"})[-1].attrs["title"] == (
            _HOSTILE
        )
        assert page.bars("degree")[0].title() == f"{_HOSTILE}\nDegree: 1.000"
        assert "&lt;script&gt;" in out.read_text(encoding="utf-8")

    def test_a_long_name_ends_in_an_ellipsis_with_a_tooltip(
        self, tmp_path: Path
    ) -> None:
        name = "Vendor " * 20
        data = {"n0": CentralityMetrics(node_id="n0", node_name=name, degree=1.0)}
        out = tmp_path / "centrality.html"
        StaticRenderer({}).plot_centrality_comparison(data, str(out))
        page = _page(out)
        assert self._labels(page) == [name]
        label = page.find("div", **{"class": "tick-label"})[-1]
        assert label.attrs["title"] == name
        assert "text-overflow:ellipsis" in (label.attrs["style"] or "")

    def test_a_large_graph_shows_the_top_nodes_by_degree(self, tmp_path: Path) -> None:
        data = {
            f"n{i}": CentralityMetrics(
                node_id=f"n{i}", node_name=f"N{i}", degree=i / 1000, pagerank=0.1
            )
            for i in range(1000)
        }
        out = tmp_path / "centrality.html"
        StaticRenderer({}).plot_centrality_comparison(data, str(out))
        page = _page(out)
        assert TOP_CENTRALITY_NODES == 25
        assert self._labels(page) == [f"N{i}" for i in range(999, 974, -1)]
        assert len(page.bars("degree")) == len(page.bars("pagerank")) == 25

    def test_empty_data_writes_nothing(self, tmp_path: Path) -> None:
        out = tmp_path / "centrality.html"
        StaticRenderer({}).plot_centrality_comparison({}, str(out))
        assert not out.exists()

    def test_duplicate_node_names_get_unique_labels(self, tmp_path: Path) -> None:
        # Two claims with the same subject/type/object share a display name;
        # two groups of one label would read as one node.
        name = "Vendor -> Buyer (obligation)"
        data = {
            f"c{i}": CentralityMetrics(
                node_id=f"c{i}", node_name=name, degree=1.0 - i / 10
            )
            for i in range(3)
        }
        out = tmp_path / "centrality.html"
        StaticRenderer({}).plot_centrality_comparison(data, str(out))
        assert self._labels(_page(out)) == [name, f"{name} (2)", f"{name} (3)"]

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

    def test_a_missing_metric_is_not_plotted_as_zero(self, tmp_path: Path) -> None:
        # Betweenness failed for every node and pagerank for one: both were
        # drawn as 0.0 bars, read as measured zeros.
        data = self._centrality()
        for metrics in data.values():
            metrics.betweenness = None
        data["n1"].pagerank = None
        out = tmp_path / "centrality.html"
        StaticRenderer({}).plot_centrality_comparison(data, str(out))
        page = _page(out)
        assert page.bars("betweenness") == []
        assert [b.title() for b in page.bars("degree")] == [
            "N0\nDegree: 1.000",
            "N1\nDegree: 0.500",
        ]
        assert [b.title() for b in page.bars("pagerank")] == ["N0\nPagerank: 0.300"]
        legend = [e.text for e in page.find("div", **{"class": "legend-label"})]
        assert legend == ["Degree", "Pagerank"]

    def test_no_metric_values_writes_nothing(self, tmp_path: Path) -> None:
        data = {"n0": CentralityMetrics(node_id="n0", node_name="N0")}
        out = tmp_path / "centrality.html"
        StaticRenderer({}).plot_centrality_comparison(data, str(out))
        assert not out.exists()


class TestCommunitySizeDistribution:
    def test_bars_count_every_level0_community(self, tmp_path: Path) -> None:
        communities = [
            _community("L0_C0", "0", 3),
            _community("L0_C1", "0", 5),
            _community("L0_C2", "0", 2),
            _community("L1_C0", "1", 10),
        ]
        out = tmp_path / "comm.html"
        StaticRenderer({}).plot_community_size_distribution(communities, str(out))
        page = _page(out)
        assert page.find("svg")[0].attrs["data-chart"] == (
            "community_size_distribution"
        )
        assert _counted(page) == 3

    def test_log_scale(self, tmp_path: Path) -> None:
        communities = [
            _community(f"L0_C{i}", "0", s) for i, s in enumerate([1, 2, 3, 40, 100])
        ]
        out = tmp_path / "comm.html"
        StaticRenderer({"log_scale_x": True}).plot_community_size_distribution(
            communities, str(out)
        )
        page = _page(out)
        assert _counted(page) == 5
        ticks = [e.text for e in page.find("div", **{"class": "tick-label"})]
        assert {"1", "10", "100"} <= set(ticks)

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
