# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Graph page's server side: what ``/api/graphs/data`` reads from an
exported ``visualization_data.json`` (no app, no stores)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.graph_exports import write_export
from unified_kg_rag.application.cli.console import graph_api
from unified_kg_rag.shared.exceptions import StoreConnectionError

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


def test_a_graph_file_is_named_by_its_namespace_not_the_last_run(
    tmp_path: Path,
) -> None:
    # Regression (rv25): an export was labelled with the pipeline id of
    # the run that wrote it ("console-extra-..."), though it holds the
    # whole namespace's graph.
    import json

    from unified_kg_rag.application.cli.console.paths import Root, Roots

    cache = tmp_path / "cache"
    named = write_export(
        cache / "console-extra-1a2b" / "visualization" / "visualization_data.json", 3
    )
    data = json.loads(named.read_text())
    head = {"namespace": "tenant-a", "built_at": "2026-11-01T00:00:00Z"}
    named.write_text(json.dumps({**head, **data}), encoding="utf-8")
    write_export(cache / "older-run" / "visualization" / "visualization_data.json", 2)
    found = {
        g["run"]: g
        for g in graph_api.find_graphs(Roots.build(Root("cache", cache)), ("cache",))
    }
    assert found["console-extra-1a2b"]["label"] == "tenant-a"
    assert found["console-extra-1a2b"]["namespace"] == "tenant-a"
    assert found["console-extra-1a2b"]["built_at"] == "2026-11-01T00:00:00Z"
    # An export of an earlier release names no namespace: its run then.
    assert found["older-run"]["label"] == "older-run"
    assert found["older-run"]["namespace"] is None


def test_an_export_older_than_its_namespace_s_build_is_outdated() -> None:
    def export(namespace: str | None, built_at: str | None) -> dict[str, object]:
        return {"namespace": namespace, "built_at": built_at}

    graphs = [
        export("default", "2026-11-02T00:00:00Z"),
        export("default", "2026-11-01T00:00:00Z"),
        export("tenant-a", "2026-11-01T00:00:00Z"),
        export("tenant-a", "2026-10-01T00:00:00Z"),
        export(None, None),
    ]
    # The build record names a later build than any export of "default".
    current = graph_api.CurrentGraph("current", "default", "2026-11-03T00:00:00Z")
    graph_api.mark_outdated(graphs, current)
    assert [g["outdated"] for g in graphs] == [True, True, False, True, False]
    graph_api.mark_outdated(graphs, graph_api.CurrentGraph("failed", None))
    assert [g["outdated"] for g in graphs] == [False, True, False, True, False]


def test_the_graph_page_builds_one_provider_bundle_per_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening the page again reuses the bundle (rv26 architecture: it was
    built, and logged, on every open); a saved config builds another."""
    from unified_kg_rag.application import bootstrap
    from unified_kg_rag.domain.models import Config

    built: list[object] = []
    monkeypatch.setattr(
        bootstrap, "build_providers", lambda config: built.append(config) or config
    )
    read: list[object] = []

    def current(config: object, cache: Path, providers: Any) -> object:
        read.append(providers())
        return graph_api.CurrentGraph("unrecorded", "default")

    monkeypatch.setattr(graph_api, "current_graph", current)
    builder = graph_api.GraphBuilder(tmp_path)
    first, second = Config(), Config()
    for config in (first, first, second):
        assert builder.current(lambda c=config: c).state == "unrecorded"
    assert built == [first, second]
    assert read == [first, first, second]

    def broken() -> Config:
        raise ValueError("no config")

    assert builder.current(broken) == graph_api.CurrentGraph(
        "failed", None, reason="config"
    )

    def unreadable(config: object, cache: Path, providers: Any) -> object:
        raise StoreConnectionError("graph store down")

    monkeypatch.setattr(graph_api, "current_graph", unreadable)
    assert builder.current(lambda: first) == graph_api.CurrentGraph(
        "failed", "default", reason="read"
    )


def test_a_failed_build_keeps_the_recorded_build_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """rv27: a build that raised (an unwritable output folder) dropped the
    build record's time, so the older export the page then shows was not
    marked outdated."""
    from unified_kg_rag.application.storage import visualization_export
    from unified_kg_rag.domain.models import Config

    monkeypatch.setattr(
        visualization_export,
        "stored_export",
        lambda bundle, suffix: ("2026-11-15T00:00:00Z", None),
    )

    def unwritable(*args: Any) -> Any:
        raise visualization_export.ExportWriteError("read-only folder")

    monkeypatch.setattr(visualization_export, "current_visualization_data", unwritable)
    current = graph_api.current_graph(Config(), tmp_path, lambda: object())
    assert current == graph_api.CurrentGraph(
        "failed", "default", "2026-11-15T00:00:00Z", reason="write"
    )

    def unreadable(*args: Any) -> Any:
        raise PermissionError("the store's own file")

    monkeypatch.setattr(visualization_export, "current_visualization_data", unreadable)
    current = graph_api.current_graph(Config(), tmp_path, lambda: object())
    assert current.reason == "read" and current.view()["reason"] == "read"
    graphs: list[dict[str, Any]] = [
        {"namespace": "default", "built_at": "2026-11-13T00:00:00Z"}
    ]
    graph_api.mark_outdated(graphs, current)
    assert graphs[0]["outdated"] is True
