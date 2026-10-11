# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the graph-analysis HTML report (AWS-free)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from unified_kg_rag.domain.ingestion.graph_analyzer import GraphStatistics
from unified_kg_rag.visualization.exporters.html_exporter import HTMLExporter

pytestmark = pytest.mark.unit


def _card(html: str, title: str) -> str:
    match = re.search(
        rf"<h3>{re.escape(title)}</h3>\s*<p class=\"value\">([^<]*)", html
    )
    assert match, title
    return match.group(1)


def test_zero_statistics_shown_not_na() -> None:
    stats = GraphStatistics(
        num_nodes=2, num_edges=0, density=0.0, average_clustering=0.0
    )
    html = HTMLExporter()._generate_html({"graph_stats": stats})
    assert _card(html, "Density") == "0.0000"
    assert _card(html, "Avg. Clustering") == "0.0000"


def test_missing_community_metrics_are_na_not_zero() -> None:
    html = HTMLExporter()._generate_html(
        {"graph_stats": GraphStatistics(num_nodes=1, num_edges=0)}
    )
    assert _card(html, "Communities") == "N/A"
    assert _card(html, "Modularity") == "N/A"
    assert _card(html, "Density") == "N/A"


def test_links_only_to_written_files(tmp_path: Path) -> None:
    written = [
        tmp_path / "interactive_graph.html",
        tmp_path / "degree_distribution.html",
    ]
    html = HTMLExporter()._generate_html({"visualizations": written})
    assert re.findall(r'href="([^"]+)"', html) == [
        "interactive_graph.html",
        "degree_distribution.html",
    ]
    assert "Interactive Graph" in html
    assert "community_hierarchy.html" not in html


def test_no_written_files_says_so() -> None:
    html = HTMLExporter()._generate_html({})
    assert 'href="' not in html
    assert "No visualizations were rendered." in html
