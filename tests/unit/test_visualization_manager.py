# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GraphVisualizationManager: export wiring, lazy embedder, degraded layout.

AWS-free: a real ``GraphAnalyzer`` over a small synthetic graph, a fake
community detector exposing only what the manager reads, and an injected fake
embedder. ``boto_session`` is a stand-in object (never used for a call).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pytest

from unified_kg_rag.adapters.ingestion.community_detector import HierarchicalCommunity
from unified_kg_rag.application.cli.run_visualization import load_render_context
from unified_kg_rag.domain.ingestion.graph_analyzer import GraphAnalyzer
from unified_kg_rag.domain.models import Community, Config
from unified_kg_rag.visualization import base as viz_base
from unified_kg_rag.visualization.base import (
    VISUALIZATION_DATA_FILENAME,
    GraphVisualizationManager,
)
from unified_kg_rag.visualization.embeddings.node2vec import NodeEmbeddings

pytestmark = pytest.mark.unit


def _graph() -> nx.Graph:
    g = nx.Graph()
    for node_id, name in [("e1", "Vendor"), ("e2", "Buyer"), ("e3", "Warehouse")]:
        g.add_node(
            node_id,
            name=name,
            description=f"{name} in a synthetic supply chain",
            description_embedding=[0.1] * 64,
            embedding=[0.2] * 64,
            node_type="entity",
        )
    g.add_edge("e1", "e2", weight=8.0, description="supplies", embedding=[0.3] * 64)
    g.add_edge("e2", "e3", weight=2.0, description="stores goods at")
    return g


class _FakeCommunityDetector:
    def __init__(self) -> None:
        self.node_to_community_l0 = {"e1": "L0_C0", "e2": "L0_C0", "e3": "L0_C1"}
        self.all_communities = {
            "L0_C0": HierarchicalCommunity(
                community_id="L0_C0", level=0, nodes={"e1", "e2"}
            ),
            "L0_C1": HierarchicalCommunity(community_id="L0_C1", level=0, nodes={"e3"}),
        }

    def get_community_metrics(self) -> None:
        return None

    def generate_community_objects(self) -> list[Community]:
        return [
            Community.model_validate(
                {
                    "id": c.community_id,
                    "name": c.community_id,
                    "level": str(c.level),
                    "parent": "",
                    "children": [],
                    "size": len(c.nodes),
                }
            )
            for c in self.all_communities.values()
        ]

    def export_community_data(self) -> dict[str, Any]:
        return {
            "resolution": 1.0,
            "hierarchy": [
                {
                    "community_id": c.community_id,
                    "level": c.level,
                    "nodes": sorted(c.nodes),
                    "size": len(c.nodes),
                    "parent": None,
                    "children": [],
                }
                for c in self.all_communities.values()
            ],
        }


class _FakeEmbedder:
    def __init__(self, *, degraded: bool = False) -> None:
        self.degraded = degraded
        self.calls = 0

    def generate_embeddings(self, graph: nx.Graph) -> NodeEmbeddings:
        self.calls += 1
        if self.degraded:
            return NodeEmbeddings(nodes=[], embeddings={}, degraded=True)
        nodes = [str(n) for n in graph.nodes()]
        return NodeEmbeddings(
            nodes=nodes,
            embeddings={
                n: np.array([float(i), float(i) ** 2]) for i, n in enumerate(nodes)
            },
        )


def _manager(
    config: Config,
    tmp_path: Path,
    *,
    embedder: _FakeEmbedder | None = None,
    embedding_method: str = "none",
) -> GraphVisualizationManager:
    config.graph.visualization.embedding_method = embedding_method
    config.graph.visualization.layout_method = "pca"
    return GraphVisualizationManager(
        config=config,
        graph_analyzer=GraphAnalyzer(config, graph=_graph()),
        community_detector=_FakeCommunityDetector(),  # type: ignore[arg-type]
        outputs_dir=tmp_path / "viz",
        boto_session=object(),  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
    )


def _export(tmp_path: Path) -> dict[str, Any]:
    path = tmp_path / "viz" / VISUALIZATION_DATA_FILENAME
    assert path.is_file()
    return json.loads(path.read_text(encoding="utf-8"))


def test_run_writes_visualization_data_for_the_cli(
    config: Config, tmp_path: Path
) -> None:
    _manager(config, tmp_path).run()

    data = _export(tmp_path)
    assert {n["id"] for n in data["nodes"]} == {"e1", "e2", "e3"}
    assert set(data["layout"]) == {"e1", "e2", "e3"}
    assert data["layout_degraded"] is False
    # Round-trips through the standalone CLI loader.
    ctx = load_render_context(tmp_path / "viz" / VISUALIZATION_DATA_FILENAME)
    assert ctx.graph.number_of_nodes() == 3
    assert ctx.graph.number_of_edges() == 2
    assert {c.community_id for c in ctx.community_hierarchy} == {"L0_C0", "L0_C1"}


def test_export_strips_embedding_attributes(config: Config, tmp_path: Path) -> None:
    _manager(config, tmp_path).run()

    data = _export(tmp_path)
    for item in data["nodes"] + data["edges"]:
        assert not any(
            k == "embedding" or k.endswith("_embedding") for k in item["attributes"]
        )
    node = next(n for n in data["nodes"] if n["id"] == "e1")
    assert node["attributes"]["name"] == "Vendor"
    assert node["attributes"]["description"].startswith("Vendor")


def test_export_does_not_mutate_the_live_graph(config: Config, tmp_path: Path) -> None:
    manager = _manager(config, tmp_path)
    manager.run()
    assert "description_embedding" in manager.analyzer.graph.nodes["e1"]


def test_embedder_not_constructed_when_embedding_method_none(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*_a: Any, **_k: Any) -> None:
        raise AssertionError("BedrockNodeEmbedder must not be constructed")

    monkeypatch.setattr(viz_base, "BedrockNodeEmbedder", _boom)
    _manager(config, tmp_path, embedding_method="none").run()
    assert _export(tmp_path)["layout_degraded"] is False


def test_embedding_layout_uses_injected_embedder_once(
    config: Config, tmp_path: Path
) -> None:
    embedder = _FakeEmbedder()
    _manager(config, tmp_path, embedder=embedder, embedding_method="node2vec").run()
    # The export reuses the run's layout instead of embedding a second time.
    assert embedder.calls == 1
    data = _export(tmp_path)
    assert data["layout_degraded"] is False
    assert set(data["layout"]) == {"e1", "e2", "e3"}


def test_degraded_embeddings_mark_layout_degraded(
    config: Config, tmp_path: Path
) -> None:
    manager = _manager(
        config,
        tmp_path,
        embedder=_FakeEmbedder(degraded=True),
        embedding_method="node2vec",
    )
    manager.run()
    assert manager.layout_degraded is True
    data = _export(tmp_path)
    assert data["layout_degraded"] is True
    # Topology-based fallback still positions every node.
    assert set(data["layout"]) == {"e1", "e2", "e3"}
