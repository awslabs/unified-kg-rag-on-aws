# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for the renderer registry and RenderContext."""

from __future__ import annotations

from pathlib import Path

import networkx as nx
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

pytestmark = pytest.mark.unit


@pytest.mark.usefixtures("restore_renderer_registry")
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
