# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture guard: the inner layers stay technology-agnostic.

CLAUDE.md/design.md advertise "No boto3/LangChain/backend imports" in domain/,
and the ports and the shared kernel beneath it import no infrastructure
either (the domain imports ``shared``; the ports are written in terms of the
domain models). Two checks enforce it, for each of ``domain/``, ``ports/``
and ``shared/``:

* a static scan of every file's own imports (catches function-local
  imports too), and
* a runtime check that imports every module of the layer in a clean
  interpreter and asserts no forbidden package was loaded *transitively*
  (e.g. through an eager re-export).

There are no carve-outs: LangChain documents are converted to the domain
``Document`` in ``adapters.ingestion.document_converter``. Which package
layers each layer may import is ``test_layering.py``'s concern.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "unified_kg_rag"
# The layers kept free of infrastructure. A function-local import in shared
# (``warn_if_langsmith_tracing``) loads its package only when called.
_PURE_LAYERS = ("domain", "ports", "shared")

# Infra/backend packages the pure domain core must not import.
_FORBIDDEN_PREFIXES = (
    "boto3",
    "botocore",
    "langchain",  # covers langchain, langchain_core, langchain_aws, ...
    "opensearchpy",
    "gremlin_python",
    "tqdm",
    # Tracing goes through shared.observability; the OTel adapter is opt-in.
    "opentelemetry",
)

# Top-level packages that must not be in sys.modules after importing the
# domain. Broader than the static list: also the heavy deps that the
# LangChain-coupled shared helpers pull in.
_FORBIDDEN_LOADED = (
    *_FORBIDDEN_PREFIXES,
    "langsmith",
    "lxml",
    "tenacity",
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


@pytest.mark.parametrize("layer", _PURE_LAYERS)
def test_layer_has_no_infra_imports(layer: str) -> None:
    root = _PACKAGE_ROOT / layer
    violations: list[str] = []
    for py_file in sorted(root.rglob("*.py")):
        rel = py_file.relative_to(root).as_posix()
        for mod in _imported_modules(py_file):
            match = next((p for p in _FORBIDDEN_PREFIXES if mod.startswith(p)), None)
            if match:
                violations.append(f"{rel} imports '{mod}' (forbidden: {match})")
    assert not violations, f"{layer} purity violated:\n" + "\n".join(violations)


def _module_names(root: Path) -> list[str]:
    # Walk files, not packages: domain/ingestion and domain/retrieval are
    # namespace packages (no __init__.py), which pkgutil.walk_packages skips.
    pkg_root = _PACKAGE_ROOT.parent
    names = []
    for py_file in sorted(root.rglob("*.py")):
        parts = py_file.relative_to(pkg_root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


@pytest.mark.parametrize("layer", _PURE_LAYERS)
def test_layer_import_loads_no_infra_packages(layer: str) -> None:
    modules = _module_names(_PACKAGE_ROOT / layer)
    assert len(modules) > 1  # the walk reached the layer's modules
    if layer == "domain":
        assert any(".ingestion." in m for m in modules)  # and the namespaces
    script = (
        "import importlib, json, sys\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
        f"forbidden = {_FORBIDDEN_LOADED!r}\n"
        "loaded = {m.split('.')[0] for m in sys.modules}\n"
        "print(json.dumps(sorted(t for t in loaded if t.startswith(forbidden))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    leaked = json.loads(result.stdout.strip().splitlines()[-1])
    assert not leaked, (
        f"importing unified_kg_rag.{layer}.* loaded {leaked}; find the chain with "
        "`python -X importtime -c 'import <module>'`"
    )
