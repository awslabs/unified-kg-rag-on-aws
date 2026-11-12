# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Graph page's server side: what ``/api/graphs/data`` reads from an
exported ``visualization_data.json`` (no app, no stores)."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.graph_exports import write_export
from unified_kg_rag.application.cli.console import graph_api

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("extra", [1, 400])
def test_a_large_graph_lays_out_only_the_drawn_nodes(
    tmp_path: Path, extra: int
) -> None:
    total = graph_api.MAX_DRAWN_NODES + extra
    data = graph_api.load_graph(write_export(tmp_path / "v.json", total))
    assert len(data["layout"]) == graph_api.MAX_DRAWN_NODES
    assert "n0" in data["layout"]  # the best-connected is drawn
    assert data["stats"]["nodes"] == total


def test_an_exported_layout_larger_than_the_cap_is_cut_to_it(
    tmp_path: Path,
) -> None:
    total = graph_api.MAX_DRAWN_NODES + 50
    layout = {f"n{i}": [float(i), 0.0] for i in range(total)}
    data = graph_api.load_graph(write_export(tmp_path / "v.json", total, layout=layout))
    assert len(data["layout"]) == graph_api.MAX_DRAWN_NODES
    assert data["layout"]["n0"] == [0.0, 0.0]
