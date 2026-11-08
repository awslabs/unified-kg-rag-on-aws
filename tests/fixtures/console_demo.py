# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console demo's modules (``console/demo/``: its corpus, the fake model
server and the workspace builder), loaded from their files for the console
tests, which run the console against them as a user's demo would."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

DEMO = Path(__file__).resolve().parents[2] / "console" / "demo"

CONSOLE_INSTALLED = all(
    importlib.util.find_spec(m) is not None for m in ("fastapi", "uvicorn", "multipart")
)
requires_console = pytest.mark.skipif(
    not CONSOLE_INSTALLED, reason="needs the console extra (uv sync --extra console)"
)


def load(name: str) -> ModuleType:
    """``console/demo/<name>.py`` as a module (once per process)."""
    if name in sys.modules:
        return sys.modules[name]
    # The demo's modules import each other by name, as demo.py runs them.
    if str(DEMO) not in sys.path:
        sys.path.insert(0, str(DEMO))
    spec = importlib.util.spec_from_file_location(name, DEMO / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
