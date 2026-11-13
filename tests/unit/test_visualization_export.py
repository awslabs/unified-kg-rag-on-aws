# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``visualization_data.json`` read from an index namespace's stores at the
end of a run: the whole graph and its communities, written where the run
says, found again by ``run-visualization`` (newest, or of a namespace)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.storage import visualization_export
from unified_kg_rag.application.storage.visualization_export import (
    export_build_time,
    export_namespace,
    newest_export,
    write_visualization_data,
)
from unified_kg_rag.domain.ingestion.visualization_data import (
    stored_visualization_data,
)
from unified_kg_rag.domain.models import (
    Claim,
    Config,
    DocumentDelta,
    Entity,
    Relationship,
)

pytestmark = pytest.mark.unit

_ENTITIES = [
    Entity(id="e2", name="Buyer", type="org", community_ids=["c0", "c1"]),
    Entity(id="e1", name="Vendor", type="org", community_ids=["c0", "c1"]),
    Entity(id="e3", name="창고", type="place", community_ids=["c2"]),
]
_RELATIONSHIPS = [
    Relationship(
        id="r1",
        source_id="e1",
        target_id="e2",
        source_name="Vendor",
        target_name="Buyer",
        type="SUPPLIES",
        weight=2.0,
    )
]
_REPORTS = [
    {"community_id": "c1", "level": 1, "parent_community_id": "c0"},
    {"community_id": "c0", "level": 0, "child_community_ids": ["c1"]},
    {"community_id": "c2", "level": "0"},
]


def test_the_stored_graph_and_its_communities_are_one_deterministic_file() -> None:
    claim = Claim(
        id="k1",
        subject_id="e1",
        subject_name="Vendor",
        object_name="Buyer",
        type="DELIVERY",
        description="Vendor ships on time.",
    )
    data = stored_visualization_data(
        _ENTITIES, _RELATIONSHIPS, [claim], _REPORTS, namespace="tenant-a"
    )
    assert next(iter(data)) == "namespace" and data["namespace"] == "tenant-a"
    assert [n["id"] for n in data["nodes"]] == ["e1", "e2", "e3", "k1"]
    by_id = {n["id"]: n["attributes"] for n in data["nodes"]}
    # Colored by the level-0 community; claims are nodes of their own.
    assert by_id["e1"]["community_id"] == "c0" and by_id["e3"]["community_id"] == "c2"
    assert by_id["k1"]["node_type"] == "claim"
    assert [
        (h["community_id"], h["level"], h["nodes"])
        for h in data["communities"]["hierarchy"]
    ] == [
        ("c0", 0, ["e1", "e2"]),
        ("c2", 0, ["e3"]),
        ("c1", 1, ["e1", "e2"]),
    ]
    assert data["communities"]["hierarchy"][0]["children"] == ["c1"]
    assert data["communities"]["hierarchy"][2]["parent"] == "c0"
    again = stored_visualization_data(
        list(reversed(_ENTITIES)),
        _RELATIONSHIPS,
        [claim],
        list(reversed(_REPORTS)),
        namespace="tenant-a",
    )
    assert json.dumps(again, default=str) == json.dumps(data, default=str)


def _providers(vector: Any, graph: Any) -> Any:
    return SimpleNamespace(
        build_vector_indexer=lambda: vector, build_graph_indexer=lambda: graph
    )


def _stores(*, relationships_index: bool) -> tuple[FakeVectorStore, FakeGraphStore]:
    vector, graph = FakeVectorStore(), FakeGraphStore()
    # The vector store keeps no community ids; the graph store does.
    vector.index_entities(
        [e.model_copy(update={"community_ids": None}) for e in _ENTITIES]
    )
    if relationships_index:
        vector.index_relationships(_RELATIONSHIPS)
    graph.index_entities(_ENTITIES)
    graph.index_relationships(_RELATIONSHIPS)
    return vector, graph


@pytest.mark.parametrize("relationships_index", [True, False])
def test_the_export_reads_communities_and_edges_from_the_stores(
    tmp_path: Path, relationships_index: bool
) -> None:
    vector, graph = _stores(relationships_index=relationships_index)
    path = tmp_path / "viz" / "visualization_data.json"

    counts = write_visualization_data(_providers(vector, graph), "default", path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert counts == {"nodes": 3, "edges": 1, "communities": 0}
    assert data["namespace"] == "default" and export_namespace(path) == "default"
    assert {n["attributes"]["community_ids"][0] for n in data["nodes"]} == {"c0", "c2"}
    assert [(e["source"], e["target"]) for e in data["edges"]] == [("e1", "e2")]


def test_the_export_closes_both_stores_even_when_a_read_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vector, graph = _stores(relationships_index=True)
    closed: list[str] = []
    monkeypatch.setattr(vector, "close", lambda: closed.append("vector"))
    monkeypatch.setattr(graph, "close", lambda: closed.append("graph"))
    write_visualization_data(_providers(vector, graph), "default", tmp_path / "v.json")
    assert sorted(closed) == ["graph", "vector"]

    closed.clear()

    def broken(*_: Any, **__: Any) -> Any:
        raise RuntimeError("read failed")

    monkeypatch.setattr(vector, "read_build_record", broken)
    with pytest.raises(RuntimeError):
        write_visualization_data(
            _providers(vector, graph), "default", tmp_path / "w.json"
        )
    assert sorted(closed) == ["graph", "vector"]


def test_newest_export_of_a_namespace(tmp_path: Path) -> None:
    def write(name: str, namespace: str | None, mtime: int) -> Path:
        path = tmp_path / name / "visualization_data.json"
        path.parent.mkdir(parents=True)
        head = {"namespace": namespace} if namespace else {}
        path.write_text(json.dumps({**head, "nodes": [], "edges": []}))
        os.utime(path, (mtime, mtime))
        return path

    old = write("a", "tenant-a", 1_000)
    legacy = write("b", None, 2_000)  # written before exports named one
    other = write("c", "tenant-b", 3_000)
    assert newest_export([tmp_path, tmp_path / "missing"]) == other
    assert newest_export([tmp_path], "tenant-a") == old
    assert newest_export([tmp_path], "tenant-c") is None
    assert export_namespace(legacy) is None


def _pipeline(config: Config, executed: set[str], cache: Path | None) -> Any:
    pipeline = object.__new__(DataIngestionPipeline)
    pipeline.config = config
    pipeline.pipeline_config = SimpleNamespace(
        local_directory=cache, s3_sync_enabled=False
    )
    pipeline.providers = None
    pipeline._executed_stages = frozenset(executed)
    return pipeline


@pytest.mark.parametrize(
    ("configured", "cache", "expected"),
    [
        ("custom/viz", "cache", "custom/viz"),
        (None, "cache", "cache/pid/visualization"),
        (None, None, "outputs/visualization"),
    ],
)
def test_the_export_goes_where_the_config_says(
    configured: str | None, cache: str | None, expected: str
) -> None:
    config = Config()
    config.graph.visualization.outputs_directory = configured
    pipeline = _pipeline(config, set(), Path(cache) if cache else None)
    path = pipeline._visualization_path("pid")
    assert path == Path(expected) / "visualization_data.json"


@pytest.mark.parametrize(
    ("export_data", "executed", "delta", "written"),
    [
        (True, {"indexing"}, None, True),
        (False, {"indexing"}, None, False),
        (True, {"graph_extraction"}, None, False),
        # A first run into an empty namespace is a full build.
        (True, {"indexing"}, DocumentDelta(new=["d1"]), True),
        # An incremental run exports nothing (rv24 perf 2): run-visualization
        # builds the file from the stores.
        (True, {"indexing"}, DocumentDelta(new=["d2"], unchanged=["d1"]), False),
    ],
)
def test_a_full_build_exports_once_it_indexed_and_never_fails_for_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    export_data: bool,
    executed: set[str],
    delta: DocumentDelta | None,
    written: bool,
) -> None:
    calls: list[Path] = []

    def write(_providers: Any, _suffix: Any, path: Path) -> dict[str, int]:
        calls.append(path)
        return {"nodes": 1, "edges": 0, "communities": 0}

    monkeypatch.setattr(
        "unified_kg_rag.application.ingestion.pipeline.write_visualization_data", write
    )
    config = Config()
    config.graph.visualization.export_data = export_data
    pipeline = _pipeline(config, executed, tmp_path)
    context = SimpleNamespace(
        pipeline_id="pid",
        visualization_data_path=None,
        stage_results=[],
        incremental_delta=delta,
    )
    pipeline._export_visualization(context)
    assert bool(calls) is written
    assert (context.visualization_data_path is not None) is written

    def fail(*_: Any) -> dict[str, int]:
        raise OSError("disk full")

    monkeypatch.setattr(
        "unified_kg_rag.application.ingestion.pipeline.write_visualization_data", fail
    )
    context = SimpleNamespace(
        pipeline_id="pid",
        visualization_data_path=None,
        stage_results=[],
        incremental_delta=delta,
    )
    pipeline._export_visualization(context)  # logged, not raised
    assert context.visualization_data_path is None


def test_a_vector_store_that_cannot_page_is_refused(tmp_path: Path) -> None:
    vector = SimpleNamespace(
        scan_items=lambda *_a, **_k: None,
        read_build_record=lambda _suffix: None,
        close=lambda: None,
    )
    graph = SimpleNamespace(close=lambda: None)
    with pytest.raises(visualization_export.ConfigurationError, match="cannot page"):
        write_visualization_data(
            _providers(vector, graph), "default", tmp_path / "x.json"
        )


def test_an_export_names_the_build_it_shows(tmp_path: Path) -> None:
    def write(name: str, data: dict[str, Any]) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return path

    built = stored_visualization_data(
        [], [], [], [], namespace="tenant-a", built_at="2026-11-09T01:02:03Z"
    )
    unbuilt = stored_visualization_data([], [], [], [], namespace="tenant-a")
    legacy = write("legacy.json", {"namespace": "tenant-a", "nodes": []})
    for path, stamp in (
        (write("built.json", built), "2026-11-09T01:02:03Z"),
        (write("unbuilt.json", unbuilt), None),
        (legacy, None),
    ):
        assert export_namespace(path) == "tenant-a"
        assert export_build_time(path) == stamp


@pytest.mark.parametrize("inside", [True, False])
def test_an_export_in_the_cache_directory_is_uploaded(
    tmp_path: Path, inside: bool
) -> None:
    uploads: list[Path] = []
    pipeline = _pipeline(Config(), {"indexing"}, tmp_path)
    pipeline.pipeline_config.s3_sync_enabled = True
    pipeline.cache_manager = SimpleNamespace(
        get_pipeline_cache_dir=lambda pid: tmp_path / pid
    )
    pipeline.s3_cache_manager = SimpleNamespace(
        upload_file=lambda _pid, _cache, path: uploads.append(path)
    )
    path = (tmp_path / "pid" if inside else tmp_path / "elsewhere") / "viz.json"

    pipeline._upload_export("pid", path)

    assert uploads == ([path] if inside else [])
