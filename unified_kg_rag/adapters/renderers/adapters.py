# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registered renderer adapters wrapping the concrete renderers.

These present the existing ``InteractiveRenderer`` / ``StaticRenderer`` (whose
methods differ) behind the uniform :class:`BaseRenderer.render` interface so
``run-visualization`` drives them through the registry. The interactive
renderer imports pyvis (the ``viz`` extra), so each adapter imports its
renderer on use: registering the adapters needs no extra. The static renderer
writes HTML and SVG itself and needs none.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from unified_kg_rag.shared import ConfigurationError, get_logger

from .base import VIZ_EXTRA_HINT, BaseRenderer, RenderContext, register_renderer

if TYPE_CHECKING:
    from .interactive import InteractiveRenderer
    from .static import StaticRenderer

logger = get_logger(__name__)


def _missing_extra(renderer: str, error: ImportError) -> ConfigurationError:
    package = (error.name or "").split(".")[0]
    return ConfigurationError(
        f"The '{renderer}' renderer needs '{package}', which is not "
        f"installed: {VIZ_EXTRA_HINT}"
    )


def _interactive_renderer() -> type[InteractiveRenderer]:
    try:
        from .interactive import InteractiveRenderer
    except ImportError as e:
        raise _missing_extra("interactive", e) from e
    return InteractiveRenderer


def _static_renderer() -> type[StaticRenderer]:
    from .static import StaticRenderer

    return StaticRenderer


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


@register_renderer("interactive", requires=("pyvis", VIZ_EXTRA_HINT))
class InteractiveRendererAdapter(BaseRenderer):
    """pyvis interactive network + community hierarchy."""

    def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
        renderer = _interactive_renderer()(self.config)
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
    """Static HTML/SVG charts: degree / centrality / community-size
    distributions. Dependency-free, so it renders on a core install."""

    needs_centrality = True

    def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
        renderer = _static_renderer()(self.config)
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
