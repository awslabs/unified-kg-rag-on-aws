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
    data = graph_api.cached_graph(write_export(tmp_path / "v.json", total)).page
    assert len(data["layout"]) == graph_api.MAX_DRAWN_NODES
    assert "n0" in data["layout"]  # the best-connected is drawn
    assert data["stats"]["nodes"] == total


def test_an_exported_layout_larger_than_the_cap_is_cut_to_it(
    tmp_path: Path,
) -> None:
    total = graph_api.MAX_DRAWN_NODES + 50
    layout = {f"n{i}": [float(i), 0.0] for i in range(total)}
    data = graph_api.cached_graph(
        write_export(tmp_path / "v.json", total, layout=layout)
    ).page
    assert len(data["layout"]) == graph_api.MAX_DRAWN_NODES
    assert data["layout"]["n0"] == [0.0, 0.0]


def test_the_page_gets_the_drawn_entities_in_full_and_the_rest_by_name(
    tmp_path: Path,
) -> None:
    total = graph_api.MAX_DRAWN_NODES + 30
    data = graph_api.cached_graph(write_export(tmp_path / "v.json", total)).page
    drawn = {n["id"] for n in data["nodes"]}
    assert drawn == set(data["layout"])
    assert len(data["others"]) == total - graph_api.MAX_DRAWN_NODES
    assert all(set(o) == {"id", "name", "degree"} for o in data["others"])
    assert not drawn & {o["id"] for o in data["others"]}
    degrees = [o["degree"] for o in data["others"]]
    assert degrees == sorted(degrees, reverse=True)
    assert data["edges"] and all(
        e["source"] in drawn and e["target"] in drawn for e in data["edges"]
    )
    # The analysis is of the whole graph.
    assert data["stats"]["nodes"] == total
    assert data["stats"]["edges"] > len(data["edges"])


def test_an_undrawn_entity_is_read_with_every_edge_that_touches_it(
    tmp_path: Path,
) -> None:
    total = graph_api.MAX_DRAWN_NODES + 30
    path = write_export(tmp_path / "v.json", total)
    undrawn = graph_api.cached_graph(path).page["others"][0]["id"]
    found = graph_api.entity(path, undrawn)
    assert found is not None and found["node"]["id"] == undrawn
    assert found["node"]["name"] == f"Entity {undrawn}"
    assert len(found["edges"]) == found["node"]["degree"]
    assert found["names"]["n0"] == "Entity n0"
    assert graph_api.entity(path, "missing") is None


def test_a_graph_is_read_again_only_when_its_file_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[Path] = []
    read = graph_api._read_graph

    def counted(path: Path) -> graph_api.LoadedGraph:
        reads.append(path)
        return read(path)

    monkeypatch.setattr(graph_api, "_read_graph", counted)
    path = write_export(tmp_path / "v.json", 20)
    first = graph_api.cached_graph(path)
    assert graph_api.cached_graph(path) is first
    assert len(reads) == 1
    write_export(path, 21)
    assert graph_api.cached_graph(path).page["stats"]["nodes"] == 21
    assert len(reads) == 2
