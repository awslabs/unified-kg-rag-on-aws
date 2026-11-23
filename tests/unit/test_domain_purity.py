# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture guard: the inner layers stay technology-agnostic.

CLAUDE.md/design.md advertise "No boto3/LangChain/backend imports" in domain/,
and the ports and the shared kernel beneath it import no infrastructure
either (the domain imports ``shared``; the ports are written in terms of the
domain models). Two checks enforce it, for each of ``domain/``, ``ports/``
and ``shared/``:

* a static scan of every file's own imports (function-local and
  type-only ones too): ``test_layering.py``'s per-layer allowlist of
  third-party packages (``_THIRD_PARTY``), which this module checks names
  none of the forbidden packages below, and
* a runtime check that imports every module of the layer in a clean
  interpreter and asserts no forbidden package was loaded *transitively*
  (e.g. through an eager re-export).

LangChain documents are converted to the domain ``Document`` in
``adapters.ingestion.document_converter``. The one third-party package of a
pure layer outside its models and algorithms is langsmith, which
``shared.logging`` imports only when called (so the runtime check holds).
Which package layers each layer may import is ``test_layering.py``'s
concern.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.test_layering import _THIRD_PARTY

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


@pytest.mark.parametrize("layer", _PURE_LAYERS)
def test_layer_has_no_infra_imports(layer: str) -> None:
    # The static scan is test_layering's allowlist: the layer passes it
    # (test_each_layer_imports_only_its_third_party_packages), so it is
    # enough that the allowlist names no forbidden package.
    allowed = _THIRD_PARTY[layer]
    assert not {
        package for package in allowed if package.startswith(_FORBIDDEN_PREFIXES)
    }, f"{layer} may import a forbidden package"


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
