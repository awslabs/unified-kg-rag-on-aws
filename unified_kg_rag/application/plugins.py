# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugins the CLIs, the MCP server and the console load without a fork.

A plugin registers its extensions on import, through the public registries
(``register_provider``, ``register_schema_source``, ``register_seed_source``,
``register_renderer``, ``ParserFactory.register_loader``). The library
imports only the built-in ones, so a plugin is imported here, once per
process, before a CLI loads its configuration and before any provider bundle
is built (``bootstrap.build_providers``):

- every entry point of the ``unified_kg_rag.plugins`` group an installed
  distribution declares (``[project.entry-points."unified_kg_rag.plugins"]``,
  ``my_backend = "my_package.plugin"``), or
- with ``UKR_PLUGINS`` set, the comma-separated modules it names instead
  (``UKR_PLUGINS=my_package.plugin,other.plugin``; empty: none).

An entry may name a module (importing it registers) or a callable in one
(``my_package.plugin:register``, called with no argument). A plugin that
cannot be loaded raises ``ConfigurationError`` naming it: a run that needs
it would fail later on a kind nothing registered.
"""

from __future__ import annotations

import importlib
import os
import threading
from importlib.metadata import EntryPoint, entry_points
from types import ModuleType

from unified_kg_rag.shared import ConfigurationError, get_logger

logger = get_logger(__name__)

ENTRY_POINT_GROUP = "unified_kg_rag.plugins"
PLUGINS_VARIABLE = "UKR_PLUGINS"

_lock = threading.Lock()
_loaded: set[str] = set()


def _configured() -> list[tuple[str, str]]:
    """``(name, target)`` of each plugin to load: ``UKR_PLUGINS``'s when it
    is set, else the installed entry points'."""
    named = os.environ.get(PLUGINS_VARIABLE)
    if named is not None:
        targets = [part.strip() for part in named.split(",") if part.strip()]
        return [(target, target) for target in targets]
    return [
        (point.name, point.value)
        for point in sorted(entry_points(group=ENTRY_POINT_GROUP), key=_key)
    ]


def _key(point: EntryPoint) -> tuple[str, str]:
    return point.name, point.value


def _load(target: str) -> None:
    module_name, _, attribute = target.partition(":")
    # The operator's own code: an installed distribution's entry point or a
    # module the process environment names (UKR_PLUGINS), the trust of the
    # process itself, never a request's input.
    # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import - plugin modules
    module: ModuleType = importlib.import_module(module_name.strip())
    if not attribute:
        return
    loaded: object = module
    for part in attribute.strip().split("."):
        loaded = getattr(loaded, part)
    if not callable(loaded):
        raise TypeError(f"'{target}' is not callable")
    loaded()


def load_plugins() -> list[str]:
    """Load the plugins not loaded yet (see the module docstring); the
    names of those this call loaded.

    Raises:
        ConfigurationError: A plugin cannot be imported, or its callable
            raised.
    """
    loaded: list[str] = []
    with _lock:
        for name, target in _configured():
            if target in _loaded:
                continue
            try:
                _load(target)
            except Exception as e:
                raise ConfigurationError(
                    f"The plugin '{name}' ({target}) could not be loaded: "
                    f"{type(e).__name__}: {e}. Install it, fix it, or leave it "
                    f"out ({PLUGINS_VARIABLE} names the plugins to load)"
                ) from e
            _loaded.add(target)
            loaded.append(name)
    if loaded:
        logger.info("Loaded plugins: %s", ", ".join(loaded))
    return loaded
