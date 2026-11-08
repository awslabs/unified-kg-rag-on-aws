# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structure guard for the ``domain/models/config/`` package.

The configuration tree is split into one private module per section, and the
package root re-exports every name, so callers keep importing
``from unified_kg_rag.domain.models.config import ...``. These checks keep a
module from growing back into one large file and keep the re-exports complete.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import unified_kg_rag.domain.models.config as config_package

pytestmark = pytest.mark.unit

_PACKAGE = Path(config_package.__file__).resolve().parent
_SECTION_MODULES = sorted(p for p in _PACKAGE.glob("_*.py") if p.name != "__init__.py")

# A section module past this size is due to be split again.
_MAX_LINES = 1100


def _top_level_names(path: Path) -> list[str]:
    names: list[str] = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return names


def test_package_has_section_modules() -> None:
    assert len(_SECTION_MODULES) > 1


@pytest.mark.parametrize("path", _SECTION_MODULES, ids=lambda p: p.name)
def test_section_module_stays_under_line_cap(path: Path) -> None:
    lines = len(path.read_text(encoding="utf-8").splitlines())
    assert lines <= _MAX_LINES, (
        f"{path.name} has {lines} lines (cap {_MAX_LINES}): split the section "
        "into another private module and re-export it from __init__.py"
    )


def test_every_public_section_name_is_reexported() -> None:
    exported = set(config_package.__all__)
    missing = sorted(
        f"{path.stem}.{name}"
        for path in _SECTION_MODULES
        for name in _top_level_names(path)
        if not name.startswith("_") and name not in exported
    )
    assert not missing, f"add to config/__init__.py __all__: {missing}"


def test_every_exported_name_resolves() -> None:
    assert len(config_package.__all__) == len(set(config_package.__all__))
    for name in config_package.__all__:
        assert hasattr(config_package, name), name
