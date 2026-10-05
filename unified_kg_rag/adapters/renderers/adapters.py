# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registered renderer adapters wrapping the concrete renderers.

These present the existing ``InteractiveRenderer`` / ``StaticRenderer`` (whose
methods differ) behind the uniform :class:`BaseRenderer.render` interface so the
manager and the standalone CLI drive them through the registry.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from unified_kg_rag.shared import get_logger

from .base import BaseRenderer, RenderContext, register_renderer
from .interactive import InteractiveRenderer
from .static import StaticRenderer

logger = get_logger(__name__)


def _render_file(path: Path, draw: Callable[[str], None]) -> list[Path]:
    """Run ``draw(path)`` and report ``path`` only if it actually wrote it.

    The concrete renderers return early (writing nothing) for empty inputs. A
    previous run's file at ``path`` is removed first (it would be overwritten
    anyway), so an existing file afterwards always reflects this render and
    the output directory never mixes stale and fresh artifacts.
    """
    path.unlink(missing_ok=True)
    draw(str(path))
    if not path.is_file():
        logger.warning("Renderer produced no output for '%s'", path.name)
        return []
    return [path]


@register_renderer("interactive")
class InteractiveRendererAdapter(BaseRenderer):
    """pyvis interactive network + community hierarchy."""

    def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
        renderer = InteractiveRenderer(self.config)
        written: list[Path] = []

        written += _render_file(
            output_dir / "interactive_graph.html",
            lambda p: renderer.create_network_visualization(
                context.graph, context.layout, p
            ),
        )

        if context.community_hierarchy:
            written += _render_file(
                output_dir / "community_hierarchy.html",
                lambda p: renderer.create_community_hierarchy(
                    context.community_hierarchy, p
                ),
            )

        return written


@register_renderer("static")
class StaticRendererAdapter(BaseRenderer):
    """Bokeh static plots: degree / centrality / community-size distributions."""

    def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
        renderer = StaticRenderer(self.config)
        written: list[Path] = []

        written += _render_file(
            output_dir / "degree_distribution.html",
            lambda p: renderer.plot_degree_distribution(context.graph, p),
        )

        if context.centrality:
            written += _render_file(
                output_dir / "centrality_comparison.html",
                lambda p: renderer.plot_centrality_comparison(context.centrality, p),
            )

        if context.communities:
            written += _render_file(
                output_dir / "community_size_distribution.html",
                lambda p: renderer.plot_community_size_distribution(
                    context.communities, p
                ),
            )

        return written
