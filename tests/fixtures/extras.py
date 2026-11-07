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


# The ragas judge's packages (the eval extra).
EVAL_INSTALLED = _installed("ragas", "datasets", "pandas")
# The interactive renderer's package (the viz extra).
VIZ_INSTALLED = _installed("pyvis")
# The RDF reader of the ontology schema sources (the ontology extra).
ONTOLOGY_INSTALLED = _installed("rdflib")
# The MCP SDK of run-mcp-server (the mcp extra).
MCP_INSTALLED = _installed("mcp")

requires_eval = pytest.mark.skipif(
    not EVAL_INSTALLED, reason="needs the eval extra (uv sync --extra eval)"
)
requires_viz = pytest.mark.skipif(
    not VIZ_INSTALLED, reason="needs the viz extra (uv sync --extra viz)"
)
requires_ontology = pytest.mark.skipif(
    not ONTOLOGY_INSTALLED, reason="needs the ontology extra (uv sync --extra ontology)"
)
requires_mcp = pytest.mark.skipif(
    not MCP_INSTALLED, reason="needs the mcp extra (uv sync --extra mcp)"
)
