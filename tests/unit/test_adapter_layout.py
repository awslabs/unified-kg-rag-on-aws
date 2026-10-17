# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture guard: the adapter packages keep their two roles apart.

Vendor packages (``adapters/aws``, ``adapters/local``) hold technology
bindings; role packages (``storage``, ``retrievers``, ``doc_status``, ``guardrail``) hold the
port implementations built on them. A role package may import a vendor one,
never the reverse, so a binding never depends on a port implementation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ADAPTERS = Path(__file__).resolve().parents[2] / "unified_kg_rag" / "adapters"
_VENDOR_PACKAGES = ("aws", "local")
_ROLE_PACKAGES = ("storage", "retrievers", "doc_status", "guardrail")


def _imported_modules(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


@pytest.mark.parametrize("vendor", _VENDOR_PACKAGES)
def test_vendor_packages_do_not_import_role_packages(vendor: str) -> None:
    role_prefixes = tuple(f"unified_kg_rag.adapters.{role}" for role in _ROLE_PACKAGES)
    violations = sorted(
        f"{path.relative_to(_ADAPTERS)} imports {module}"
        for path in (_ADAPTERS / vendor).rglob("*.py")
        for module in _imported_modules(path)
        if any(module == p or module.startswith(f"{p}.") for p in role_prefixes)
    )
    assert violations == []


def test_role_packages_exist() -> None:
    # Guards the guard: a renamed role package would make the scan vacuous.
    for package in (*_VENDOR_PACKAGES, *_ROLE_PACKAGES):
        assert (_ADAPTERS / package / "__init__.py").is_file(), package
