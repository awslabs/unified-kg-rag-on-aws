# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture guard: the domain/ layer stays technology-agnostic.

CLAUDE.md/design.md advertise "No boto3/LangChain/backend imports" in domain/.
Two checks enforce it:

* a static scan of every domain file's own imports (catches function-local
  imports too), and
* a runtime check that imports every domain module in a clean interpreter and
  asserts no forbidden package was loaded *transitively* (e.g. through an eager
  re-export in ``shared``).

There are no carve-outs: LangChain documents are converted to the domain
``Document`` in ``adapters.ingestion.document_converter``.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
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


def test_domain_has_no_infra_imports() -> None:
    violations: list[str] = []
    for py_file in sorted(_DOMAIN_ROOT.rglob("*.py")):
        rel = py_file.relative_to(_DOMAIN_ROOT).as_posix()
        for mod in _imported_modules(py_file):
            match = next((p for p in _FORBIDDEN_PREFIXES if mod.startswith(p)), None)
            if match:
                violations.append(f"{rel} imports '{mod}' (forbidden: {match})")
    assert not violations, "domain purity violated:\n" + "\n".join(violations)


def _domain_module_names() -> list[str]:
    # Walk files, not packages: domain/ingestion and domain/retrieval are
    # namespace packages (no __init__.py), which pkgutil.walk_packages skips.
    pkg_root = _DOMAIN_ROOT.parent.parent
    names = []
    for py_file in sorted(_DOMAIN_ROOT.rglob("*.py")):
        parts = py_file.relative_to(pkg_root).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


def test_domain_import_loads_no_infra_packages() -> None:
    modules = _domain_module_names()
    assert any(".ingestion." in m for m in modules)  # the walk reached them
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
        f"importing unified_kg_rag.domain.* loaded {leaked}; find the chain with "
        "`python -X importtime -c 'import <module>'`"
    )


# --- layer contracts (design.md §2.1) ----------------------------------------

_PACKAGE_ROOT = _DOMAIN_ROOT.parent

# Packages each layer must not import, at runtime or under TYPE_CHECKING
# (``application -> adapters -> ports -> domain``; ``shared`` is the kernel
# every layer uses, so it imports none of them). ``evaluation`` drives the RAG
# chain, so it sits between the chain (``application.retrieval``, built from
# ``application.bootstrap``) and the CLIs; ``visualization`` renders over the
# adapters and is driven by the ingestion stages and a CLI.
_LAYER_FORBIDDEN: dict[str, tuple[str, ...]] = {
    "domain": ("ports", "adapters", "application", "evaluation", "visualization"),
    "ports": ("adapters", "application", "evaluation", "visualization"),
    "shared": ("ports", "adapters", "application", "evaluation", "visualization"),
    "adapters": ("application", "evaluation", "visualization"),
    "evaluation": (
        "application.cli",
        "application.ingestion",
        "application.prompts",
        "application.storage",
        "visualization",
    ),
    "visualization": ("application", "evaluation"),
}

# Deliberate type-only exceptions (TYPE_CHECKING imports), documented in
# design.md §2.5.
_ALLOWED: set[tuple[str, str]] = set()


def _layer_violations() -> list[str]:
    violations = []
    for layer, forbidden in _LAYER_FORBIDDEN.items():
        for py_file in sorted((_PACKAGE_ROOT / layer).rglob("*.py")):
            rel = py_file.relative_to(_PACKAGE_ROOT).as_posix()
            for mod in _imported_modules(py_file):
                inner = mod.removeprefix("unified_kg_rag.")
                if (
                    inner != mod
                    and any(
                        inner == banned or inner.startswith(f"{banned}.")
                        for banned in forbidden
                    )
                    and (rel, mod) not in _ALLOWED
                ):
                    violations.append(f"{rel} imports '{mod}'")
    return violations


def test_layers_import_only_inward() -> None:
    violations = _layer_violations()
    assert not violations, "layer rule violated:\n" + "\n".join(violations)


def test_layer_scan_reaches_every_layer() -> None:
    for layer in _LAYER_FORBIDDEN:
        assert any((_PACKAGE_ROOT / layer).rglob("*.py")), layer


def _type_checking_imports(py_file: Path) -> set[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) in (
            "TYPE_CHECKING",
            "typing.TYPE_CHECKING",
        ):
            for child in node.body:
                if isinstance(child, ast.ImportFrom) and child.module:
                    mods.add(child.module)
    return mods


@pytest.mark.parametrize(("rel", "module"), sorted(_ALLOWED))
def test_allowed_exceptions_are_type_only(rel: str, module: str) -> None:
    assert module in _type_checking_imports(_PACKAGE_ROOT / rel)
