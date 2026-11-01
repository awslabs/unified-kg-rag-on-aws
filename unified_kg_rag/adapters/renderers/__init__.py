# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph renderers and their registry.

Importing this package registers the built-in renderers without loading pyvis
or Bokeh (the ``viz`` extra): each adapter imports its concrete renderer when
it renders, and fails fast naming the extra when it is missing.
"""

from typing import Any

# Importing adapters registers them with the renderer registry.
from .adapters import InteractiveRendererAdapter, StaticRendererAdapter
from .base import (
    VIZ_EXTRA_HINT,
    BaseRenderer,
    RenderContext,
    get_renderer_class,
    register_renderer,
    registered_renderers,
    render_all,
)

__all__ = [
    "VIZ_EXTRA_HINT",
    "BaseRenderer",
    "InteractiveRenderer",
    "InteractiveRendererAdapter",
    "RenderContext",
    "StaticRenderer",
    "StaticRendererAdapter",
    "get_renderer_class",
    "register_renderer",
    "registered_renderers",
    "render_all",
]


def __getattr__(name: str) -> Any:
    # The concrete renderers import pyvis / Bokeh: load them on first access.
    if name == "InteractiveRenderer":
        from .interactive import InteractiveRenderer

        return InteractiveRenderer
    if name == "StaticRenderer":
        from .static import StaticRenderer

        return StaticRenderer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
