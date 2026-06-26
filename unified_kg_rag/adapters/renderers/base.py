# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderer abstraction + registry for graph visualization.

Replaces the hardcoded renderer calls in ``GraphVisualizationManager`` with a
registry of :class:`BaseRenderer` adapters. A new renderer (e.g. a Cytoscape or
D3 exporter) is added by subclassing ``BaseRenderer`` and decorating it with
``@register_renderer(...)`` — no edit to the manager.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from unified_kg_rag.shared import get_logger

if TYPE_CHECKING:
    import networkx as nx

logger = get_logger(__name__)


@dataclass
class RenderContext:
    """Everything a renderer may need, assembled once by the manager."""

    graph: nx.Graph
    layout: dict[str, Any] = field(default_factory=dict)
    communities: list[Any] = field(default_factory=list)
    community_hierarchy: list[Any] = field(default_factory=list)
    centrality: dict[str, Any] = field(default_factory=dict)


class BaseRenderer(ABC):
    """A renderer turns a :class:`RenderContext` into one or more output files."""

    def __init__(self, config: Any) -> None:
        self.config = config

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
