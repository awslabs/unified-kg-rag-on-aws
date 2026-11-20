# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One graph layout and one node cap for every surface that draws a graph:
``run-visualization``'s pages and the console's Graph page
(``domain.ingestion.visualization_data.spring_layout``, capped by
``graph.visualization.interactive.max_nodes``)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pytest

from tests.fixtures.graph_exports import write_export
from unified_kg_rag.domain.ingestion.visualization_data import spring_layout
from unified_kg_rag.domain.models import Config
from unified_kg_rag.domain.models.config import InteractiveGraphSettings

pytestmark = pytest.mark.unit


def _islands() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from(
        [("a", "b"), ("b", "c"), ("c", "a"), ("c", "d"), ("x", "y"), ("y", "z")]
    )
    graph.add_node("solo")
    return graph


# The layout of _islands(), to 4 decimals: the console's packed layout,
# now in the graph's own node order (it was a set's, so the 3-node island
# moved with the process's hash seed).
_GOLDEN = {
    "a": [1.6009, 1.2745],
    "b": [1.5395, 0.6188],
    "c": [0.8596, 1.0131],
    "d": [0.0, 1.0936],
    "solo": [0.5, 2.9],
    "x": [3.3779, 1.7321],
    "y": [3.2658, 0.8645],
    "z": [3.1543, 0.0015],
}


def test_the_layout_is_the_golden_one() -> None:
    layout = spring_layout(_islands(), None)
    assert {k: [round(c, 4) for c in v] for k, v in layout.items()} == _GOLDEN


@pytest.mark.parametrize("seed", ["1", "2", "3"])
def test_the_layout_does_not_move_with_the_hash_seed(seed: str) -> None:
    script = (
        "import json, networkx as nx\n"
        "from unified_kg_rag.domain.ingestion.visualization_data import "
        "spring_layout\n"
        "g = nx.Graph()\n"
        "g.add_edges_from([('a','b'),('b','c'),('c','a'),('c','d'),"
        "('x','y'),('y','z')])\n"
        "g.add_node('solo')\n"
        "print(json.dumps({k: [round(c, 4) for c in v] "
        "for k, v in spring_layout(g, None).items()}))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": seed},
    ).stdout
    assert json.loads(out) == _GOLDEN


def test_each_component_is_laid_out_in_a_box_of_its_own() -> None:
    # One spring layout of a disconnected graph flings its islands about
    # and over the main component; packed, their boxes are disjoint.
    layout = spring_layout(_islands(), None)
    boxes = []
    for component in nx.connected_components(_islands()):
        xs = [layout[n][0] for n in component]
        ys = [layout[n][1] for n in component]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    for i, (ax0, ay0, ax1, ay1) in enumerate(boxes):
        for bx0, by0, bx1, by1 in boxes[i + 1 :]:
            assert ax1 < bx0 or bx1 < ax0 or ay1 < by0 or by1 < ay0


def test_only_the_best_connected_nodes_are_laid_out() -> None:
    layout = spring_layout(nx.star_graph(5), 3)
    assert len(layout) == 3 and "0" in layout
    assert spring_layout(nx.Graph(), 3) == {}


def test_the_cap_has_one_default_for_every_surface() -> None:
    # Measured: laying out 600 nodes takes about 3 s, 2000 about 15 s
    # (networkx spring layout, a scale-free graph; the console waits for it).
    assert InteractiveGraphSettings().max_nodes == 600


def test_the_console_and_run_visualization_draw_the_same_layout(
    tmp_path: Path,
) -> None:
    from unified_kg_rag.application.cli import run_visualization as cli
    from unified_kg_rag.application.cli.console import graph_api

    config = Config()
    config.graph.visualization.interactive = InteractiveGraphSettings(max_nodes=40)
    path = write_export(tmp_path / "v.json", 60)
    context = cli.load_render_context(path)
    cli._fill_missing_analysis(context, config, centrality=False)
    page = graph_api.cached_graph(path, max_nodes=40).page
    assert page["layout"] == context.layout
    assert len(page["layout"]) == 40


def test_the_console_draws_the_configured_cap(tmp_path: Path) -> None:
    from unified_kg_rag.application.cli.console import graph_api

    path = write_export(tmp_path / "v.json", 30)
    assert len(graph_api.cached_graph(path, max_nodes=10).page["layout"]) == 10
    # Another cap is another page, not the cached one.
    assert len(graph_api.cached_graph(path, max_nodes=20).page["layout"]) == 20
    # 0 or None draws every node.
    assert len(graph_api.cached_graph(path, max_nodes=None).page["layout"]) == 30
    data = json.loads(path.read_text())
    assert len(data["nodes"]) == 30
