# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderer abstraction + registry for graph visualization.

``run-visualization`` drives a registry of :class:`BaseRenderer` adapters. A
new renderer (e.g. a Cytoscape or D3 exporter) is added by subclassing
``BaseRenderer`` and decorating it with ``@register_renderer(...)`` — no edit
to the CLI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from unified_kg_rag.shared import ConfigurationError, get_logger
from unified_kg_rag.shared.extras import extra_hint
from unified_kg_rag.shared.registry import Registry

if TYPE_CHECKING:
    import networkx as nx

logger = get_logger(__name__)

# How to install what the interactive renderer imports (pyvis).
VIZ_EXTRA_HINT = f"install {extra_hint('viz')}"


@dataclass
class RenderContext:
    """Everything a renderer may need, assembled once by the caller."""

    graph: nx.Graph
    layout: dict[str, Any] = field(default_factory=dict)
    communities: list[Any] = field(default_factory=list)
    community_hierarchy: list[Any] = field(default_factory=list)
    centrality: dict[str, Any] = field(default_factory=dict)


class BaseRenderer(ABC):
    """A renderer turns a :class:`RenderContext` into one or more output files."""

    # Whether :meth:`render` reads ``RenderContext.centrality``: the caller
    # computes it (betweenness alone is O(V*E)) only for a renderer that does.
    needs_centrality: ClassVar[bool] = False

    def __init__(self, config: Any) -> None:
        self.config = config

    @abstractmethod
    def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
        """Produce visualization files; return the paths written."""
        ...


_RENDERERS: Registry[str, type[BaseRenderer]] = Registry("renderer")
_REGISTRY = _RENDERERS.entries

RendererT = TypeVar("RendererT", bound="type[BaseRenderer]")


def register_renderer(
    name: str, *, requires: tuple[str, str] | None = None
) -> Callable[[RendererT], RendererT]:
    """Class decorator registering a renderer under ``name``, under the one
    registry rule (``shared.registry``: case-insensitive, a later
    registration replaces). ``requires`` is the ``(module, how to install
    it)`` of the optional package :meth:`BaseRenderer.render` imports:
    :func:`resolve_renderers` fails naming it before any rendering."""
    _RENDERERS.check_name(name)

    def decorator(cls: RendererT) -> RendererT:
        key = _RENDERERS.register(name, cls, requires=requires)
        logger.debug("Registered renderer '%s' -> %s", key, cls.__name__)
        return cls

    return decorator


def registered_renderers() -> tuple[str, ...]:
    return tuple(_RENDERERS)


def get_renderer_class(name: str) -> type[BaseRenderer]:
    renderer = _RENDERERS.get(name)
    if renderer is None:
        available = ", ".join(_RENDERERS.names()) or "(none)"
        raise ValueError(
            f"No renderer registered for '{name}'. Available: {available}."
        )
    return renderer


def check_renderer_dependencies(name: str) -> None:
    """Raise ``ConfigurationError`` when the package renderer ``name``
    declared (``requires``) is not installed."""
    missing = _RENDERERS.missing(name)
    if missing is not None:
        module, install = missing
        raise ConfigurationError(
            f"The '{name}' renderer needs '{module}', which is not installed: "
            f"{install}"
        )


def resolve_renderers(
    renderer_names: Iterable[str],
) -> list[tuple[str, type[BaseRenderer]]]:
    """The renderer class of each name, its dependencies checked: an unknown
    name raises ``ValueError`` and a missing dependency
    ``ConfigurationError`` (:func:`check_renderer_dependencies`)."""
    renderers = [(name, get_renderer_class(name)) for name in renderer_names]
    for name, _ in renderers:
        check_renderer_dependencies(name)
    return renderers


def render_resolved(
    context: RenderContext,
    output_dir: Path,
    renderers: Iterable[tuple[str, type[BaseRenderer]]],
    viz_config: Any,
    *,
    failures: dict[str, str] | None = None,
) -> list[Path]:
    """Run renderers :func:`resolve_renderers` returned over ``context``;
    return the files written.

    A renderer's config is the ``viz_config`` attribute named after it (an
    empty dict when there is none), so a newly registered renderer needs no
    dispatch edit. A renderer that fails is logged and skipped so the others
    still render; ``failures``, when given, gets its name and error.
    """
    written: list[Path] = []
    for name, renderer_cls in renderers:
        renderer_config = getattr(viz_config, name, None) or {}
        try:
            paths = renderer_cls(renderer_config).render(context, output_dir)
        except ConfigurationError:
            raise
        except Exception as e:  # noqa: BLE001 - one renderer must not sink the rest
            logger.error("Renderer '%s' failed: %s", name, e)
            if failures is not None:
                failures[name] = str(e) or type(e).__name__
            continue
        if paths:
            logger.info("Renderer '%s' wrote %d file(s)", name, len(paths))
        else:
            logger.warning("Renderer '%s' wrote no files", name)
        written.extend(paths)
    return written
