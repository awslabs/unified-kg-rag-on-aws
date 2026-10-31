# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skip markers for tests that need an optional extra.

CI installs every extra these name (and checks they are importable), so the
markers skip only on a core install, where the suite still runs the rest.
"""

from __future__ import annotations

import importlib.util

import pytest


def _installed(*modules: str) -> bool:
    return all(importlib.util.find_spec(module) is not None for module in modules)


# The LLM judges' packages (the eval extra).
EVAL_INSTALLED = _installed("ragas", "datasets", "pandas", "langchain_openai")
# The langchain judge's alone (the dev group installs it for the local adapters).
LANGCHAIN_JUDGE_INSTALLED = _installed("langchain_openai")
# The built-in renderers' packages (the viz extra).
VIZ_INSTALLED = _installed("pyvis", "bokeh")

requires_eval = pytest.mark.skipif(
    not EVAL_INSTALLED, reason="needs the eval extra (uv sync --extra eval)"
)
requires_viz = pytest.mark.skipif(
    not VIZ_INSTALLED, reason="needs the viz extra (uv sync --extra viz)"
)
