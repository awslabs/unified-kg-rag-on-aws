# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture guard: every import follows the layer rules (design.md §2.1).

A static scan of every module in the package, every import counted: module
level, inside a function (lazy) and under ``TYPE_CHECKING``, relative ones
resolved. A lazy or type-only import is still a dependency: it couples the
two modules' code, and a type checker loads the imported package.

The rules:

* **Layer order.** Layers are ordered outermost first (``LAYERS``); a module
  imports its own layer and the layers after it, never one before it.
* **Evaluation drives the chain.** Of the application layer, evaluation
  imports only the RAG chain and the composition root that builds its
  providers (``_EVALUATION_REACHES``), never the ingestion pipeline.
* **Backends are injected.** Outside the composition roots
  (``application/bootstrap.py`` and the CLIs) nothing above the adapters
  imports a vendor binding or a port implementation (``_BACKENDS``); they
  arrive through ``Providers``. Evaluation and visualization do not import
  the store adapters at all: evaluation reaches the stores through the
  chain, visualization renders the graph it is handed.
* **Vendor packages are leaves.** ``adapters/aws`` and ``adapters/local``
  import, of the adapters, only each other and the vendor-neutral kernel
  (``adapters/llm``, ``adapters/observability``), which imports no other
  adapter package: a role package (storage, retrievers, ...) wraps a vendor
  binding, never the reverse.
* **Plugins through their registry.** Outside its own package, a registered
  plugin (renderer, search strategy, evaluator) is reached through the
  package root or its base/registry module, never a concrete module; the
  evaluator type map (``EvaluationManager._resolve_evaluator_class``) is the
  one place that names the evaluators.
* **The root facade** is imported only for ``__version__``: it re-exports
  the public API lazily, so importing any other name from it inside the
  package would hide a dependency on the layer that defines it.

``_ALLOWED`` lists deliberate exceptions, each with its reason. It is empty:
keep it that way, or say there why an edge cannot follow the rules.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PACKAGE = "unified_kg_rag"
_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / PACKAGE

# Outermost first. Each layer is a set of module prefixes (relative to the
# package); the longest matching prefix decides a module's layer.
LAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # The console scripts: parse arguments, load config, wire and run.
    ("entry points", ("application.cli",)),
    # Drives the RAG chain over a dataset; the CLIs drive it.
    ("evaluation", ("evaluation",)),
    # Orchestration: ingestion pipeline, indexing manager, RAG chain,
    # composition root, config loading.
    ("application", ("application",)),
    # Renders a built graph through the renderer registry; the ingestion
    # stage and run-visualization drive it.
    ("visualization", ("visualization",)),
    ("adapters", ("adapters",)),
    ("ports", ("ports",)),
    ("domain", ("domain",)),
    # The cross-cutting kernel (logging, errors, metrics, tracing, pure
    # helpers): every layer uses it, so it imports no other layer.
    ("shared", ("shared",)),
)

# The application modules evaluation may import.
_EVALUATION_REACHES = ("application.retrieval", "application.bootstrap")

# Modules that build backends from config (composition roots).
_COMPOSITION_ROOTS = ("application.bootstrap", "application.cli")

# Vendor bindings and port implementations: built by the composition roots
# and the provider builders, injected everywhere else.
_BACKENDS = (
    "adapters.aws",
    "adapters.local",
    "adapters.doc_status",
    "adapters.guardrail",
    "adapters.retrievers",
    "adapters.storage.neptune_indexer",
    "adapters.storage.opensearch_indexer",
    "adapters.storage.sqlite_graph_indexer",
)
# Layers that see no store adapter at all (not even the filter schema).
_STORE_FREE = {"evaluation": "adapters.storage", "visualization": "adapters.storage"}

_VENDORS = ("adapters.aws", "adapters.local")
_ADAPTER_KERNEL = ("adapters.llm", "adapters.observability")

# Plugin package -> the modules of it others may import.
_PLUGINS: dict[str, tuple[str, ...]] = {
    "adapters.renderers": ("adapters.renderers", "adapters.renderers.base"),
    "adapters.search_strategies": ("adapters.search_strategies",),
    "adapters.evaluators": ("adapters.evaluators", "adapters.evaluators.base"),
}
# Plugin package -> the module holding its explicit type -> class map.
_PLUGIN_MAPS = {"adapters.evaluators": "evaluation.evaluation_manager"}

# (importing module, imported module) -> why the edge is allowed.
_ALLOWED: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class Import:
    source: str  # importing module, relative to the package
    target: str  # imported module, relative to the package ("" = the root)
    names: tuple[str, ...]  # the name a ``from`` import binds
    line: int

    def __str__(self) -> str:
        return f"{self.source}:{self.line} imports {self.target or PACKAGE}"


def _module_name(py_file: Path, root: Path) -> str:
    parts = py_file.relative_to(root).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    """The absolute dotted name a (possibly relative) ``from`` import names."""
    if not node.level:
        return node.module or ""
    base = f"{PACKAGE}.{module}" if module else PACKAGE
    parts = base.split(".")
    keep = len(parts) - node.level + (1 if is_package else 0)
    return ".".join([*parts[:keep], *([node.module] if node.module else [])])


def _is_module(root: Path, name: str) -> bool:
    if not name.startswith(f"{PACKAGE}."):
        return False
    path = root.joinpath(*name.split(".")[1:])
    return path.with_suffix(".py").is_file() or path.is_dir()


def package_imports(root: Path = _PACKAGE_ROOT) -> Iterator[Import]:
    """Every import of a package module by a package module, at any depth."""
    for py_file in sorted(root.rglob("*.py")):
        module = _module_name(py_file, root)
        is_package = py_file.name == "__init__.py"
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [(alias.name, ()) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = _resolve(node, module, is_package)
                # ``from pkg import mod`` imports the submodule ``pkg.mod``.
                targets = [
                    (
                        (
                            f"{base}.{alias.name}"
                            if _is_module(root, f"{base}.{alias.name}")
                            else base
                        ),
                        (alias.name,),
                    )
                    for alias in node.names
                ]
            else:
                continue
            for name, imported in targets:
                if name == PACKAGE or name.startswith(f"{PACKAGE}."):
                    target = name.removeprefix(PACKAGE).removeprefix(".")
                    yield Import(module, target, imported, node.lineno)


def _within(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == p or module.startswith(f"{p}.") for p in prefixes)


def layer_of(module: str) -> int | None:
    """Index in ``LAYERS`` of ``module``'s layer; None for the root facade."""
    best: tuple[int, int] | None = None  # (prefix length, layer index)
    for index, (_, prefixes) in enumerate(LAYERS):
        for prefix in prefixes:
            if _within(module, (prefix,)) and (best is None or len(prefix) > best[0]):
                best = (len(prefix), index)
    return None if best is None else best[1]


def _rule_broken(imp: Import) -> str | None:
    """Why ``imp`` breaks a rule, or None."""
    source, target = layer_of(imp.source), layer_of(imp.target)
    if source is None:
        return "module in no layer"
    if not imp.target:
        return None if imp.names == ("__version__",) else "import __version__ only"
    if target is None:
        return "target in no layer"
    importer, imported = LAYERS[source][0], LAYERS[target][0]
    if target < source:
        return f"{importer} must not import {imported}"
    if (
        _within(imp.target, _BACKENDS)
        and target != source
        and not _within(imp.source, _COMPOSITION_ROOTS)
    ):
        return "a backend; inject it (Providers)"
    if importer == "evaluation" and imported == "application":
        if not _within(imp.target, _EVALUATION_REACHES):
            return "evaluation reaches the application through the chain"
    if importer in _STORE_FREE and _within(imp.target, (_STORE_FREE[importer],)):
        return f"{importer} imports no store adapter"
    for plugins, public in _PLUGINS.items():
        if (
            _within(imp.target, (plugins,))
            and not _within(imp.source, (plugins,))
            and imp.target not in public
            and imp.source != _PLUGIN_MAPS.get(plugins)
        ):
            return "a plugin; go through its registry"
    if not _within(imp.target, ("adapters",)):
        return None
    if _within(imp.source, _VENDORS):
        if not _within(imp.target, (*_VENDORS, *_ADAPTER_KERNEL)):
            return "a vendor package imports no role package"
    if _within(imp.source, _ADAPTER_KERNEL) and not _within(
        imp.target, _ADAPTER_KERNEL
    ):
        return "the adapter kernel imports no other adapter"
    return None


def violations(root: Path = _PACKAGE_ROOT) -> list[str]:
    """One line per import statement that breaks a rule."""
    found = [
        f"{imp}: {reason}"
        for imp in package_imports(root)
        # The root facade itself re-exports every layer.
        if imp.source and (imp.source, imp.target) not in _ALLOWED
        if (reason := _rule_broken(imp))
    ]
    return list(dict.fromkeys(found))


def test_imports_follow_the_layer_rules() -> None:
    found = violations()
    assert not found, "layer rule violated:\n" + "\n".join(found)


def test_every_module_is_in_a_layer() -> None:
    modules = {_module_name(f, _PACKAGE_ROOT) for f in _PACKAGE_ROOT.rglob("*.py")}
    assert {m for m in modules if layer_of(m) is None} == {""}  # the facade


def test_every_layer_has_modules() -> None:
    modules = [_module_name(f, _PACKAGE_ROOT) for f in _PACKAGE_ROOT.rglob("*.py")]
    for index, (name, _) in enumerate(LAYERS):
        assert any(layer_of(m) == index for m in modules), name


def test_allowed_exceptions_are_still_needed() -> None:
    edges = {(imp.source, imp.target) for imp in package_imports()}
    assert set(_ALLOWED) <= edges, "remove the stale _ALLOWED entries"


def test_the_scan_sees_lazy_type_only_and_relative_imports(tmp_path: Path) -> None:
    domain = tmp_path / "domain"
    domain.mkdir()
    (domain / "__init__.py").write_text("from . import algo\n")
    (domain / "algo.py").write_text(
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from unified_kg_rag.ports import DocStatusPort\n"
        "def run():\n"
        "    import unified_kg_rag.adapters.aws.bedrock\n"
        "from ..application import bootstrap\n"
    )
    found = {(i.source, i.target) for i in package_imports(tmp_path)}
    assert found == {
        ("domain", "domain.algo"),
        ("domain.algo", "ports"),
        ("domain.algo", "adapters.aws.bedrock"),
        ("domain.algo", "application"),
    }


@pytest.mark.parametrize(
    ("source", "target", "rule"),
    [
        ("domain/x.py", "unified_kg_rag.adapters.retrieval.base", "must not import"),
        ("shared/x.py", "unified_kg_rag.domain.models", "must not import"),
        ("application/cli/x.py", "unified_kg_rag.evaluation", None),
        ("application/x.py", "unified_kg_rag.evaluation", "must not import"),
        ("application/x.py", "unified_kg_rag.adapters.aws.s3_cache", "a backend"),
        ("application/bootstrap.py", "unified_kg_rag.adapters.aws.s3_cache", None),
        ("evaluation/x.py", "unified_kg_rag.adapters.storage.filter_schema", "store"),
        ("visualization/x.py", "unified_kg_rag.application", "must not import"),
        ("evaluation/x.py", "unified_kg_rag.application.retrieval.rag_chain", None),
        ("evaluation/x.py", "unified_kg_rag.application.ingestion.pipeline", "chain"),
        ("adapters/aws/x.py", "unified_kg_rag.adapters.store_errors", "vendor"),
        ("adapters/llm/x.py", "unified_kg_rag.adapters.local.models", "kernel"),
        ("adapters/storage/x.py", "unified_kg_rag.adapters.aws.neptune", None),
        ("adapters/x.py", "unified_kg_rag", "__version__ only"),
        ("visualization/x.py", "unified_kg_rag.adapters.renderers.static", "registry"),
        ("visualization/x.py", "unified_kg_rag.adapters.renderers", None),
        ("adapters/renderers/x.py", "unified_kg_rag.adapters.renderers.static", None),
        (
            "evaluation/x.py",
            "unified_kg_rag.adapters.evaluators.ragas_evaluator",
            "regis",
        ),
        (
            "evaluation/evaluation_manager.py",
            "unified_kg_rag.adapters.evaluators.ragas_evaluator",
            None,
        ),
    ],
)
def test_each_rule_catches_its_violation(
    tmp_path: Path, source: str, target: str, rule: str | None
) -> None:
    path = tmp_path / source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"from {target} import thing\n")
    found = violations(tmp_path)
    if rule is None:
        assert found == []
    else:
        assert len(found) == 1 and rule in found[0], found
