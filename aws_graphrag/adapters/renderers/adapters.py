# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Registered renderer adapters wrapping the concrete renderers.

These present the existing ``InteractiveRenderer`` / ``StaticRenderer`` (whose
methods differ) behind the uniform :class:`BaseRenderer.render` interface so the
manager and the standalone CLI drive them through the registry.
"""

from __future__ import annotations

from pathlib import Path

from aws_graphrag.shared import get_logger

from .base import BaseRenderer, RenderContext, register_renderer
from .interactive import InteractiveRenderer
from .static import StaticRenderer

logger = get_logger(__name__)


@register_renderer("interactive")
class InteractiveRendererAdapter(BaseRenderer):
    """pyvis interactive network + community hierarchy."""

    def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
        renderer = InteractiveRenderer(self.config)
        written: list[Path] = []

        graph_path = output_dir / "interactive_graph.html"
        renderer.create_network_visualization(
            context.graph, context.layout, str(graph_path)
        )
        written.append(graph_path)

        if context.community_hierarchy:
            hierarchy_path = output_dir / "community_hierarchy.html"
            renderer.create_community_hierarchy(
                context.community_hierarchy, str(hierarchy_path)
            )
            written.append(hierarchy_path)

        return written


@register_renderer("static")
class StaticRendererAdapter(BaseRenderer):
    """Bokeh static plots: degree / centrality / community-size distributions."""

    def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
        renderer = StaticRenderer(self.config)
        written: list[Path] = []

        degree_path = output_dir / "degree_distribution.html"
        renderer.plot_degree_distribution(context.graph, str(degree_path))
        written.append(degree_path)

        if context.centrality:
            centrality_path = output_dir / "centrality_comparison.html"
            renderer.plot_centrality_comparison(
                context.centrality, str(centrality_path)
            )
            written.append(centrality_path)

        if context.communities:
            community_path = output_dir / "community_size_distribution.html"
            renderer.plot_community_size_distribution(
                context.communities, str(community_path)
            )
            written.append(community_path)

        return written
