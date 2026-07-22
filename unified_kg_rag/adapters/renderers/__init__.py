# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Importing adapters registers them with the renderer registry.
from .adapters import InteractiveRendererAdapter, StaticRendererAdapter
from .base import (
    BaseRenderer,
    RenderContext,
    get_renderer_class,
    register_renderer,
    registered_renderers,
)
from .interactive import InteractiveRenderer
from .static import StaticRenderer

__all__ = [
    "BaseRenderer",
    "InteractiveRenderer",
    "InteractiveRendererAdapter",
    "RenderContext",
    "StaticRenderer",
    "StaticRendererAdapter",
    "get_renderer_class",
    "register_renderer",
    "registered_renderers",
]
