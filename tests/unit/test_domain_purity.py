# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture guard: the domain/ layer stays technology-agnostic.

CLAUDE.md/design.md advertise "No boto3/LangChain/backend imports" in domain/ as
a grep-verifiable invariant. This test statically scans every domain module's
imports and fails on a forbidden one, so an accidental infra import into a pure
model or algorithm is caught. There are no carve-outs: LangChain documents are
converted to the domain ``Document`` in ``shared.utils.document_converter``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_DOMAIN_ROOT = Path(__file__).resolve().parents[2] / "unified_kg_rag" / "domain"

# Infra/backend packages the pure domain core must not import.
_FORBIDDEN_PREFIXES = (
    "boto3",
    "botocore",
    "langchain",  # covers langchain, langchain_core, langchain_aws, ...
    "opensearchpy",
    "gremlin_python",
    "tqdm",
)


def _imported_modules(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            mods.add(node.module)
    return mods


def test_domain_has_no_infra_imports() -> None:
    violations: list[str] = []
    for py_file in sorted(_DOMAIN_ROOT.rglob("*.py")):
        rel = py_file.relative_to(_DOMAIN_ROOT).as_posix()
        for mod in _imported_modules(py_file):
            match = next((p for p in _FORBIDDEN_PREFIXES if mod.startswith(p)), None)
            if match:
                violations.append(f"{rel} imports '{mod}' (forbidden: {match})")
    assert not violations, "domain purity violated:\n" + "\n".join(violations)
