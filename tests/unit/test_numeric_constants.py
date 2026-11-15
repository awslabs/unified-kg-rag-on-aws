# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""No new bare numeric module constant without a reason.

A static scan of every package module for a module-level name bound to a
number (``THRESHOLD = 0.85``, ``LIMIT = 4 * 1024``; a bool is no number).
Each must be listed in ``tests/fixtures/numeric_constants.py`` with why it
is a constant and not a configuration field with an evidence tag
(``domain/models/evidence.py``), and every entry there must still exist.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.fixtures.numeric_constants import ALLOWED

pytestmark = pytest.mark.unit

PACKAGE = Path(__file__).resolve().parents[2] / "unified_kg_rag"


def _is_number(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int | float) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp):
        return _is_number(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_number(node.left) and _is_number(node.right)
    return False


def numeric_constants(root: Path = PACKAGE) -> set[str]:
    """``<module path>:<NAME>`` of every module-level numeric binding."""
    found: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        module = path.relative_to(root).as_posix()
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if not _is_number(value):
                continue
            found.update(
                f"{module}:{target.id}"
                for target in targets
                if isinstance(target, ast.Name)
            )
    return found


def test_scan_finds_a_new_constant(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text(
        "THRESHOLD = 0.85\nLIMIT: int = 4 * 1024\nFLAG = True\nNAME = 'x'\n"
        "def f():\n    LOCAL = 3\n",
        encoding="utf-8",
    )
    assert numeric_constants(tmp_path) == {"mod.py:THRESHOLD", "mod.py:LIMIT"}


def test_every_numeric_constant_has_a_reason() -> None:
    unlisted = sorted(numeric_constants() - set(ALLOWED))
    assert not unlisted, (
        "numeric module constants not in tests/fixtures/numeric_constants.py: "
        f"{unlisted}. A value that tunes what is retrieved, extracted or "
        "answered belongs in the configuration with an evidence tag; else "
        "list it there with its reason."
    )


def test_every_listed_constant_exists() -> None:
    stale = sorted(set(ALLOWED) - numeric_constants())
    assert not stale, f"listed constants that are gone, remove them: {stale}"
