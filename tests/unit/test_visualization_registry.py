# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for the renderer registry / RenderContext and the pure
``DimensionalityReducer`` (PCA/t-SNE/fallback paths).

The node2vec embedder is Bedrock-coupled and excluded here; the reducer itself
operates on plain numpy arrays so its PCA/t-SNE/fallback paths are AWS-free.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from unified_kg_rag.adapters.renderers import (
    BaseRenderer,
    InteractiveRendererAdapter,
    RenderContext,
    StaticRendererAdapter,
    get_renderer_class,
    register_renderer,
    registered_renderers,
)
from unified_kg_rag.visualization.embeddings.dimensionality import (
    DimensionalityReducer,
)
from unified_kg_rag.visualization.embeddings.node2vec import NodeEmbeddings

pytestmark = pytest.mark.unit


class TestRegistry:
    def test_builtin_renderers_registered(self) -> None:
        assert {"interactive", "static"} <= set(registered_renderers())

    def test_get_renderer_class_resolves_builtins(self) -> None:
        assert get_renderer_class("static") is StaticRendererAdapter
        assert get_renderer_class("interactive") is InteractiveRendererAdapter

    def test_unknown_renderer_raises_with_available_list(self) -> None:
        with pytest.raises(ValueError, match="No renderer registered"):
            get_renderer_class("nope")

    def test_register_then_resolve(self) -> None:
        @register_renderer("registry_test_renderer")
        class _R(BaseRenderer):
            def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
                return []

        assert "registry_test_renderer" in registered_renderers()
        assert get_renderer_class("registry_test_renderer") is _R

    def test_reregister_same_class_is_idempotent(self) -> None:
        @register_renderer("registry_idempotent")
        class _R(BaseRenderer):
            def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
                return []

        # Decorating the same class under the same name again is allowed.
        register_renderer("registry_idempotent")(_R)
        assert get_renderer_class("registry_idempotent") is _R

    def test_reregister_different_class_raises(self) -> None:
        @register_renderer("registry_conflict")
        class _A(BaseRenderer):
            def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
                return []

        with pytest.raises(ValueError, match="already registered"):

            @register_renderer("registry_conflict")
            class _B(BaseRenderer):
                def render(
                    self, context: RenderContext, output_dir: Path
                ) -> list[Path]:
                    return []


class TestRenderContext:
    def test_defaults_are_empty_collections(self) -> None:
        ctx = RenderContext(graph=nx.Graph())
        assert ctx.layout == {}
        assert ctx.communities == []
        assert ctx.community_hierarchy == []
        assert ctx.centrality == {}

    def test_carries_supplied_fields(self) -> None:
        g = nx.Graph()
        g.add_node("a")
        ctx = RenderContext(graph=g, layout={"a": [0.0, 0.0]}, centrality={"a": 1})
        assert ctx.graph.number_of_nodes() == 1
        assert ctx.layout == {"a": [0.0, 0.0]}
        assert ctx.centrality == {"a": 1}


class TestRendererAdaptersDriveRenderContext:
    def _ctx(self) -> RenderContext:
        g = nx.Graph()
        g.add_node("e1", name="Alice", community_id="c0")
        g.add_node("e2", name="Acme", community_id="c0")
        g.add_edge("e1", "e2", weight=1.0)
        return RenderContext(graph=g)

    def test_static_adapter_writes_degree_plot(self, tmp_path: Path) -> None:
        written = StaticRendererAdapter({}).render(self._ctx(), tmp_path)
        assert (tmp_path / "degree_distribution.html") in written
        assert (tmp_path / "degree_distribution.html").exists()

    def test_interactive_adapter_writes_graph_html(self, tmp_path: Path) -> None:
        written = InteractiveRendererAdapter({}).render(self._ctx(), tmp_path)
        assert (tmp_path / "interactive_graph.html") in written
        assert (tmp_path / "interactive_graph.html").exists()


def _embeddings(n: int, dim: int = 8) -> NodeEmbeddings:
    rng = np.random.default_rng(0)
    nodes = [f"n{i}" for i in range(n)]
    return NodeEmbeddings(
        nodes=nodes,
        embeddings={node: rng.normal(size=dim) for node in nodes},
    )


class TestDimensionalityReducer:
    def test_empty_embeddings_returns_empty(self) -> None:
        reducer = DimensionalityReducer({})
        out = reducer.reduce_dimensions(NodeEmbeddings(nodes=[], embeddings={}))
        assert out == {}

    def test_single_node_returns_origin(self) -> None:
        reducer = DimensionalityReducer({})
        emb = NodeEmbeddings(nodes=["n0"], embeddings={"n0": np.zeros(8)})
        out = reducer.reduce_dimensions(emb)
        assert out == {"n0": (0.0, 0.0)}

    def test_pca_produces_2d_layout(self) -> None:
        reducer = DimensionalityReducer({})
        out = reducer.reduce_dimensions(_embeddings(10), method="pca")
        assert len(out) == 10
        assert all(len(coord) == 2 for coord in out.values())
        assert all(
            isinstance(x, float) and isinstance(y, float) for x, y in out.values()
        )

    def test_tsne_produces_2d_layout(self) -> None:
        reducer = DimensionalityReducer({"tsne": {"perplexity": 5}})
        out = reducer.reduce_dimensions(_embeddings(20), method="tsne")
        assert len(out) == 20
        assert all(len(coord) == 2 for coord in out.values())

    def test_too_few_points_falls_back_to_pca(self) -> None:
        # With < n_components+1 points the reducer downgrades to PCA but still
        # produces a 2D layout for every node.
        reducer = DimensionalityReducer({})
        out = reducer.reduce_dimensions(_embeddings(3), method="umap")
        assert len(out) == 3
        assert all(len(coord) == 2 for coord in out.values())
