# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""No new bare numeric constant without a reason.

A static scan of every package module for a module-level name bound to a
number (``THRESHOLD = 0.85``, ``LIMIT = 4 * 1024``; a bool is no number)
and for an upper-case class attribute bound to one (``class Manager:
BUFFER_TOKENS = 512``, keyed ``<module>:<Class>.<NAME>``). Each must be listed in ``tests/fixtures/numeric_constants.py`` with why it
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


def _numeric_bindings(body: list[ast.stmt]) -> list[str]:
    """The names ``body`` binds to a number, in order."""
    names: list[str] = []
    for node in body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if _is_number(value):
            names.extend(t.id for t in targets if isinstance(t, ast.Name))
    return names


def numeric_constants(root: Path = PACKAGE) -> set[str]:
    """``<module path>:<NAME>`` of every module-level numeric binding and
    ``<module path>:<Class>.<NAME>`` of every upper-case numeric class
    attribute."""
    found: set[str] = set()
    for path in sorted(root.rglob("*.py")):
        module = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found.update(f"{module}:{name}" for name in _numeric_bindings(tree.body))
        found.update(
            f"{module}:{node.name}.{name}"
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            for name in _numeric_bindings(node.body)
            if name.lstrip("_").isupper()
        )
    return found


def test_scan_finds_a_new_constant(tmp_path: Path) -> None:
    (tmp_path / "mod.py").write_text(
        "THRESHOLD = 0.85\nLIMIT: int = 4 * 1024\nFLAG = True\nNAME = 'x'\n"
        "def f():\n    LOCAL = 3\n"
        "class Manager:\n    BUFFER_TOKENS = 512\n    _MAX_ITEMS: int = 8\n"
        "    ratio = 0.5\n    ENABLED = False\n",
        encoding="utf-8",
    )
    assert numeric_constants(tmp_path) == {
        "mod.py:THRESHOLD",
        "mod.py:LIMIT",
        "mod.py:Manager.BUFFER_TOKENS",
        "mod.py:Manager._MAX_ITEMS",
    }


def test_every_numeric_constant_has_a_reason() -> None:
    unlisted = sorted(numeric_constants() - set(ALLOWED))
    assert not unlisted, (
        "numeric constants not in tests/fixtures/numeric_constants.py: "
        f"{unlisted}. A value that tunes what is retrieved, extracted or "
        "answered belongs in the configuration with an evidence tag; else "
        "list it there with its reason."
    )


def test_every_listed_constant_exists() -> None:
    stale = sorted(set(ALLOWED) - numeric_constants())
    assert not stale, f"listed constants that are gone, remove them: {stale}"
