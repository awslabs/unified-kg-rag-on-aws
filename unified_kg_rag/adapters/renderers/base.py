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
from typing import TYPE_CHECKING, Any, TypeVar

from unified_kg_rag.shared import ConfigurationError, get_logger

if TYPE_CHECKING:
    import networkx as nx

logger = get_logger(__name__)

# How to install what the built-in renderers import (pyvis, Bokeh).
VIZ_EXTRA_HINT = (
    "install the viz extra (`uv sync --extra viz`, or `pip install -e "
    "'.[viz]'` in a source checkout)"
)


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

    def __init__(self, config: Any) -> None:
        self.config = config

    @classmethod
    def check_dependencies(cls) -> None:
        """Raise ``ConfigurationError`` when a package :meth:`render` needs
        is not installed (nothing by default).

        Called before any rendering, so a missing optional dependency fails
        the run naming its remedy instead of leaving that renderer's files
        out.
        """
        return None

    @abstractmethod
    def render(self, context: RenderContext, output_dir: Path) -> list[Path]:
        """Produce visualization files; return the paths written."""
        ...


_REGISTRY: dict[str, type[BaseRenderer]] = {}

RendererT = TypeVar("RendererT", bound="type[BaseRenderer]")


def register_renderer(name: str) -> Callable[[RendererT], RendererT]:
    """Class decorator registering a renderer under ``name``."""

    def decorator(cls: RendererT) -> RendererT:
        if name in _REGISTRY and _REGISTRY[name] is not cls:
            raise ValueError(f"Renderer '{name}' already registered")
        _REGISTRY[name] = cls
        logger.debug("Registered renderer '%s' -> %s", name, cls.__name__)
        return cls

    return decorator


def registered_renderers() -> tuple[str, ...]:
    return tuple(_REGISTRY)


def get_renderer_class(name: str) -> type[BaseRenderer]:
    try:
        return _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(
            f"No renderer registered for '{name}'. Available: {available}."
        ) from None


def resolve_renderers(
    renderer_names: Iterable[str],
) -> list[tuple[str, type[BaseRenderer]]]:
    """The renderer class of each name, its dependencies checked: an unknown
    name raises ``ValueError`` and a missing dependency
    ``ConfigurationError`` (:meth:`BaseRenderer.check_dependencies`)."""
    renderers = [(name, get_renderer_class(name)) for name in renderer_names]
    for _, renderer_cls in renderers:
        renderer_cls.check_dependencies()
    return renderers


def render_all(
    context: RenderContext,
    output_dir: Path,
    renderer_names: Iterable[str],
    viz_config: Any,
) -> list[Path]:
    """Run each named renderer over ``context``; return the files written.

    The names are resolved and checked (:func:`resolve_renderers`) before
    anything renders; see :func:`render_resolved` for the rest.
    """
    return render_resolved(
        context, output_dir, resolve_renderers(renderer_names), viz_config
    )


def render_resolved(
    context: RenderContext,
    output_dir: Path,
    renderers: Iterable[tuple[str, type[BaseRenderer]]],
    viz_config: Any,
) -> list[Path]:
    """Run renderers :func:`resolve_renderers` returned over ``context``;
    return the files written.

    A renderer's config is the ``viz_config`` attribute named after it (an
    empty dict when there is none), so a newly registered renderer needs no
    dispatch edit. A renderer that fails is logged and skipped so the others
    still render.
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
            continue
        if paths:
            logger.info("Renderer '%s' wrote %d file(s)", name, len(paths))
        else:
            logger.warning("Renderer '%s' wrote no files", name)
        written.extend(paths)
    return written
