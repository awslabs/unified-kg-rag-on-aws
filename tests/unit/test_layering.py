# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Architecture guard: every import follows the layer rules (design.md §2.1).

A static scan of every module in the package, every import counted: module
level, inside a function (lazy) and under ``TYPE_CHECKING``, relative ones
resolved, and ``importlib.import_module`` / ``__import__`` of a literal
module name. A lazy or type-only import is still a dependency: it couples
the two modules' code, and a type checker loads the imported package.
Importing ``pkg.mod`` also runs ``pkg/__init__.py`` (and its parents'), so
the imports a package's ``__init__`` makes count as made by every module
that imports from the package (from outside it).

The rules:

* **Layer order.** Layers are ordered outermost first (``LAYERS``); a module
  imports its own layer and the layers after it, never one before it.
* **Evaluation drives the chain.** Of the application layer, evaluation
  imports only the RAG chain and the composition root that builds its
  providers (``_EVALUATION_REACHES``), never the ingestion pipeline.
* **Backends are injected.** Outside the composition roots
  (``application/bootstrap.py`` and the CLIs) nothing above the adapters
  imports a vendor binding or a port implementation (``_BACKENDS``); they
  arrive through ``Providers``. Inside the adapters, the search strategies,
  the retrieval kernel and the ingestion adapters
  (``_BACKEND_FREE_ADAPTERS``) receive them too. Evaluation does not import
  the store adapters at all: it reaches the stores through the chain.
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
* **Imports are static.** Only the facade and the console-script launcher
  (``_DYNAMIC_IMPORTERS``) import a package module by a name string: any
  other dynamic import would be an edge a reader cannot see. A dynamic
  import whose name the scan cannot read (a variable, a concatenation, a
  relative name, an f-string that could name a package module), and code
  run or loaded by other means (``exec``, ``eval``, ``runpy``,
  ``spec_from_file_location``, ``exec_module``, ``load_module``,
  ``walk_packages``, ...), is allowed only where ``_RUNTIME_NAMED_IMPORTS``
  says what it imports. The import functions are recognised under an alias
  (``from importlib import import_module as load``), through
  ``getattr(importlib, "import_module")`` and namespace lookups
  (``vars(importlib)[...]``, ``importlib.__dict__[...]``), their name given
  positionally or as ``name=``; a reference to one that does not call it
  (``load = importlib.import_module``, ``functools.partial(...)``) counts as
  an import of a name the scan cannot read.

The scan sees edges, not what they load at run time: an allowed module can
pass a backend on (an adapter module importing a store adapter, imported by
the application). ``test_injected_modules_load_no_store_backend`` imports
every module that has its backends injected in a clean interpreter and
checks no backend is loaded.

``_ALLOWED`` lists deliberate exceptions, each with its reason. It is empty:
keep it that way, or say there why an edge cannot follow the rules.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
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
# Adapter packages that receive the backends as well.
_BACKEND_FREE_ADAPTERS = (
    "adapters.search_strategies",
    "adapters.retrieval",
    "adapters.ingestion",
)
# Layers that see no store adapter at all (not even the filter schema).
_STORE_FREE = {"evaluation": "adapters.storage"}

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

# The modules that may import a package module by a name string: the facade
# resolves its lazy names, the launcher imports the CLI its script names.
_DYNAMIC_IMPORTERS = ("", "application.cli.launch")
# importlib.import_module, __import__, pkgutil.resolve_name and
# importlib.util.find_spec (which imports the parent packages of the name).
_DYNAMIC_IMPORT_CALLS = {"import_module", "__import__", "resolve_name", "find_spec"}
# The calls that import the module they name (find_spec of a name in a
# variable only looks an optional extra up).
_NAME_IMPORTING_CALLS = _DYNAMIC_IMPORT_CALLS - {"find_spec"}
# Calls that run or load code the scan cannot follow, whatever they are given
# (``exec`` and ``eval`` only as the builtins: other objects have such methods).
_CODE_LOADING_CALLS = {
    "exec",
    "eval",
    "spec_from_file_location",
    "module_from_spec",
    "exec_module",
    "load_module",
    "run_module",
    "run_path",
    "walk_packages",
}
_BUILTIN_ONLY = {"exec", "eval"}
# The modules those functions come from, for ``from <module> import f as g``.
_DYNAMIC_IMPORT_MODULES = {
    "importlib",
    "importlib.util",
    "pkgutil",
    "builtins",
    "runpy",
}
# Names bound to those modules (and the builtins), for ``getattr(m, name)``
# and ``vars(m)[name]`` lookups of a function the scan cannot read.
_LOADER_NAMES = {"importlib", "pkgutil", "builtins", "runpy", "__builtins__"}
# What such a lookup names.
_UNREAD_FUNCTION = "<a function the scan cannot read>"
# A reference to one of these, not called where it is named (an alias by
# assignment, an argument of functools.partial), is flagged as unread.
_REFERENCED = _NAME_IMPORTING_CALLS | _CODE_LOADING_CALLS | {_UNREAD_FUNCTION}
# Module -> what its dynamic imports of a name the scan cannot read import.
_RUNTIME_NAMED_IMPORTS: dict[str, str] = {
    "adapters.local.models": (
        "import_optional: the [local]-extra module its caller names "
        "(langchain_ollama, langchain_openai, sentence_transformers)"
    ),
}
# What a dynamic import of a name the scan cannot read targets.
_UNREAD = "<a name the scan cannot read>"

# (importing module, imported module) -> why the edge is allowed.
_ALLOWED: dict[tuple[str, str], str] = {}


@dataclass(frozen=True)
class Import:
    source: str  # importing module, relative to the package
    target: str  # imported module, relative to the package ("" = the root)
    names: tuple[str, ...]  # the name a ``from`` import binds
    line: int
    dynamic: bool = False  # ``import_module``/``__import__`` of a literal name
    via: str | None = None  # the package whose ``__init__`` makes the edge

    def __str__(self) -> str:
        text = f"{self.source}:{self.line} imports {self.target or PACKAGE}"
        return f"{text} (through {self.via}/__init__)" if self.via else text


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


def _call_aliases(tree: ast.AST) -> dict[str, str]:
    """Local name -> the dynamic import function it is bound to
    (``from importlib import import_module as load``)."""
    functions = _DYNAMIC_IMPORT_CALLS | _CODE_LOADING_CALLS
    return {
        alias.asname or alias.name: alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module in _DYNAMIC_IMPORT_MODULES
        for alias in node.names
        if alias.name in functions
    }


def _looked_up(obj: ast.expr, key: ast.expr | None) -> str:
    """The function ``getattr(obj, key)`` or ``vars(obj)[key]`` names: a
    literal ``key``, or ``_UNREAD_FUNCTION`` for another key of a loader
    module (``getattr(importlib, "import_" + "module")``)."""
    if isinstance(key, ast.Constant) and isinstance(key.value, str):
        return key.value
    if isinstance(obj, ast.Name) and obj.id in _LOADER_NAMES:
        return _UNREAD_FUNCTION
    return ""


def _called_function(func: ast.expr, aliases: dict[str, str]) -> str:
    """The function name an expression names: an attribute, a name (or what
    its alias is bound to), ``getattr(module, name)``, or a lookup in a
    module's namespace (``vars(module)[name]``, ``module.__dict__[name]``,
    ``__builtins__[name]``)."""
    if isinstance(func, ast.Attribute):
        if func.attr in _BUILTIN_ONLY:
            namespace = func.value
            builtin = isinstance(namespace, ast.Name) and namespace.id in {
                "builtins",
                "__builtins__",
            }
            return func.attr if builtin else ""
        return func.attr
    if isinstance(func, ast.Name):
        return aliases.get(func.id, func.id)
    if (
        isinstance(func, ast.Call)
        and isinstance(func.func, ast.Name)
        and func.func.id == "getattr"
        and len(func.args) >= 2
    ):
        return _looked_up(func.args[0], func.args[1])
    if isinstance(func, ast.Subscript):
        namespace = func.value
        if (
            isinstance(namespace, ast.Call)
            and isinstance(namespace.func, ast.Name)
            and namespace.func.id == "vars"
            and len(namespace.args) == 1
        ):
            return _looked_up(namespace.args[0], func.slice)
        if isinstance(namespace, ast.Attribute) and namespace.attr == "__dict__":
            return _looked_up(namespace.value, func.slice)
        if isinstance(namespace, ast.Name) and namespace.id == "__builtins__":
            return _looked_up(namespace, func.slice)
    return ""


def _is_reference(node: ast.AST, called: set[int], aliases: dict[str, str]) -> bool:
    """Whether ``node`` names an import or code-loading function without
    calling it there (``load = importlib.import_module``)."""
    if id(node) in called:
        return False
    if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
        if not isinstance(node.ctx, ast.Load):
            return False
    elif not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
    ):
        return False
    return _called_function(node, aliases) in _REFERENCED


def _dynamic_target(node: ast.Call, aliases: dict[str, str]) -> str | None:
    """The module the call ``node`` imports, if it is a dynamic import.

    A literal name (``resolve_name``'s without its ``:attribute``) or an
    f-string's literal head up to its last dot is the module; ``_UNREAD``
    for a name that could be a package module the scan cannot read and for
    code run or loaded by other means; None for any other call, and for a
    name outside the package.
    """
    name = _called_function(node.func, aliases)
    if name in _CODE_LOADING_CALLS or name == _UNREAD_FUNCTION:
        return _UNREAD
    if name not in _DYNAMIC_IMPORT_CALLS:
        return None
    arg = next(
        iter(node.args or [k.value for k in node.keywords if k.arg == "name"]), None
    )
    if arg is None:  # the name in *args or **kwargs
        return _UNREAD if name in _NAME_IMPORTING_CALLS else None
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        if arg.value.startswith("."):
            return _UNREAD  # relative to a package given at run time
        return arg.value.partition(":")[0]
    if isinstance(arg, ast.JoinedStr) and arg.values:
        head = arg.values[0]
        if isinstance(head, ast.Constant) and isinstance(head.value, str):
            text = head.value
            if text.startswith(f"{PACKAGE}."):
                return text.rpartition(".")[0] or None
            # f"unstructured.partition.{x}" names no package module.
            if "." in text:
                return None
    return _UNREAD if name in _NAME_IMPORTING_CALLS else None


def package_imports(root: Path = _PACKAGE_ROOT) -> Iterator[Import]:
    """Every import of a package module by a package module, at any depth."""
    for py_file in sorted(root.rglob("*.py")):
        module = _module_name(py_file, root)
        is_package = py_file.name == "__init__.py"
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        aliases = _call_aliases(tree)
        called = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        for node in ast.walk(tree):
            dynamic = False
            if _is_reference(node, called, aliases):
                yield Import(module, _UNREAD, (), getattr(node, "lineno", 0), True)
                continue
            if isinstance(node, ast.Call):
                if (literal := _dynamic_target(node, aliases)) is None:
                    continue
                if literal == _UNREAD:
                    yield Import(module, _UNREAD, (), node.lineno, dynamic=True)
                    continue
                targets = [(literal, ())]
                dynamic = True
            elif isinstance(node, ast.Import):
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
                    yield Import(module, target, imported, node.lineno, dynamic)


def _packages_run(name: str, packages: set[str]) -> Iterator[str]:
    """The packages whose ``__init__`` importing ``name`` runs, outermost
    first (the root facade excluded: it loads nothing eagerly)."""
    parts = name.split(".") if name else []
    for end in range(1, len(parts) + 1):
        if (package := ".".join(parts[:end])) in packages:
            yield package


def imports_with_packages(root: Path = _PACKAGE_ROOT) -> Iterator[Import]:
    """``package_imports`` plus, for each, the imports the ``__init__`` of
    every package it loads makes (transitively through further package
    ``__init__``s); a package the importer is in is loaded already."""
    imports = list(package_imports(root))
    packages = {_module_name(f, root) for f in root.rglob("__init__.py")} - {""}
    by_package: dict[str, list[Import]] = {}
    for imp in imports:
        if imp.source in packages:
            by_package.setdefault(imp.source, []).append(imp)
    for imp in imports:
        yield imp
        seen: set[str] = set()
        pending = [imp.target]
        while pending:
            for package in _packages_run(pending.pop(), packages):
                if package in seen or _within(imp.source, (package,)):
                    continue
                seen.add(package)
                for edge in by_package.get(package, ()):
                    yield Import(
                        imp.source, edge.target, edge.names, imp.line, via=package
                    )
                    pending.append(edge.target)


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
    if imp.target == _UNREAD:
        if imp.source in _DYNAMIC_IMPORTERS or imp.source in _RUNTIME_NAMED_IMPORTS:
            return None
        return (
            "a dynamic import of a name the scan cannot read; import it "
            "statically, or say what it imports in _RUNTIME_NAMED_IMPORTS"
        )
    if imp.dynamic and imp.source not in _DYNAMIC_IMPORTERS:
        return "a dynamic import; import it statically"
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
    if _within(imp.source, _BACKEND_FREE_ADAPTERS) and _within(imp.target, _BACKENDS):
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
            # A plugin package root re-exports its plugins.
            and not (imp.via and _within(imp.via, (plugins,)))
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
        for imp in imports_with_packages(root)
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
    unread = {imp.source for imp in package_imports() if imp.target == _UNREAD}
    assert (
        set(_RUNTIME_NAMED_IMPORTS) <= unread
    ), "remove the stale _RUNTIME_NAMED_IMPORTS entries"


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
        ("evaluation/x.py", "unified_kg_rag.application.retrieval.rag_chain", None),
        ("evaluation/x.py", "unified_kg_rag.application.ingestion.pipeline", "chain"),
        ("adapters/aws/x.py", "unified_kg_rag.adapters.store_errors", "vendor"),
        ("adapters/llm/x.py", "unified_kg_rag.adapters.local.models", "kernel"),
        ("adapters/storage/x.py", "unified_kg_rag.adapters.aws.neptune", None),
        ("adapters/x.py", "unified_kg_rag", "__version__ only"),
        (
            "application/cli/x.py",
            "unified_kg_rag.adapters.renderers.static",
            "registry",
        ),
        ("application/cli/x.py", "unified_kg_rag.adapters.renderers", None),
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
        (
            "adapters/search_strategies/x.py",
            "unified_kg_rag.adapters.retrievers.neptune_retriever",
            "a backend",
        ),
        (
            "adapters/retrieval/x.py",
            "unified_kg_rag.adapters.aws.opensearch",
            "backend",
        ),
        ("adapters/ingestion/x.py", "unified_kg_rag.adapters.local.models", "backend"),
        ("adapters/providers.py", "unified_kg_rag.adapters.local.models", None),
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


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    "source", ["application/x.py", "adapters/search_strategies/x.py"]
)
def test_every_backend_is_injected(tmp_path: Path, backend: str, source: str) -> None:
    path = tmp_path / source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"from unified_kg_rag.{backend}.module import thing\n")
    found = violations(tmp_path)
    assert len(found) == 1 and "a backend; inject it" in found[0], found


@pytest.mark.parametrize(
    ("source", "call", "rule"),
    [
        (
            "application/x.py",
            "importlib.import_module('unified_kg_rag.adapters.aws.bedrock')",
            "a dynamic import",
        ),
        ("domain/x.py", "__import__('unified_kg_rag.domain.models')", "dynamic"),
        ("domain/x.py", "import_module(f'unified_kg_rag.adapters.{name}')", "dynamic"),
        ("domain/x.py", "importlib.import_module('httpx')", None),
        (
            "application/cli/launch.py",
            "importlib.import_module(f'unified_kg_rag.application.cli.{name}')",
            None,
        ),
        # Names the scan cannot read.
        (
            "domain/x.py",
            "importlib.import_module('.aws.bedrock', 'unified_kg_rag.adapters')",
            "cannot read",
        ),
        (
            "domain/x.py",
            "importlib.import_module('unified_kg_rag.' + 'adapters.aws.bedrock')",
            "cannot read",
        ),
        ("domain/x.py", "importlib.import_module(name)", "cannot read"),
        ("domain/x.py", "importlib.import_module(f'unified_kg_rag{name}')", "read"),
        ("domain/x.py", "__import__(name)", "cannot read"),
        ("domain/x.py", "pkgutil.resolve_name(name)", "cannot read"),
        (
            "domain/x.py",
            "pkgutil.resolve_name('unified_kg_rag.adapters.aws.bedrock:X')",
            "a dynamic import",
        ),
        (
            "domain/x.py",
            "importlib.util.find_spec('unified_kg_rag.adapters.aws.bedrock')",
            "a dynamic import",
        ),
        # Outside the package, or only looked up.
        ("domain/x.py", "import_module(f'unstructured.partition.{name}')", None),
        ("domain/x.py", "importlib.util.find_spec(name)", None),
        ("adapters/local/models.py", "importlib.import_module(name)", None),
    ],
)
def test_dynamic_imports_are_flagged(
    tmp_path: Path, source: str, call: str, rule: str | None
) -> None:
    path = tmp_path / source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"import importlib\nname = 'x'\nmodule = {call}\n")
    found = violations(tmp_path)
    if rule is None:
        assert found == []
    else:
        assert len(found) == 1 and rule in found[0], found


_BEDROCK = "'unified_kg_rag.adapters.aws.bedrock'"


@pytest.mark.parametrize(
    ("code", "rule"),
    [
        # Under an alias.
        (
            f"from importlib import import_module as load\nload({_BEDROCK})",
            "a dynamic import; import it statically",
        ),
        ("from importlib import import_module as load\nload(name)", "cannot read"),
        ("from builtins import __import__ as imp\nimp(name)", "cannot read"),
        # Through getattr.
        (
            f"getattr(importlib, 'import_module')({_BEDROCK})",
            "a dynamic import; import it statically",
        ),
        ("getattr(importlib, '__import__')(name)", "cannot read"),
        # The name as a keyword, or not given at all.
        (
            f"importlib.import_module(name={_BEDROCK})",
            "a dynamic import; import it statically",
        ),
        ("importlib.import_module(name=name)", "cannot read"),
        ("importlib.import_module(*names)", "cannot read"),
        # Code run or loaded by other means.
        (f"exec('import ' + {_BEDROCK})", "cannot read"),
        ("builtins.exec(code)", "cannot read"),
        ("importlib.util.spec_from_file_location('m', path)", "cannot read"),
        (
            "from importlib.util import spec_from_file_location as spec\n"
            "spec('m', path)",
            "cannot read",
        ),
        ("__builtins__.exec(code)", "cannot read"),
        (f"eval('__import__(' + {_BEDROCK} + ')')", "cannot read"),
        ("import runpy\nrunpy.run_module(name)", "cannot read"),
        ("from runpy import run_path as run\nrun(path)", "cannot read"),
        ("importlib.util.module_from_spec(spec)", "cannot read"),
        ("spec.loader.exec_module(module)", "cannot read"),
        ("SourceFileLoader('m', path).load_module()", "cannot read"),
        ("import pkgutil\nlist(pkgutil.walk_packages(paths))", "cannot read"),
        # Looked up in a module's namespace.
        (f"vars(importlib)['import_module']({_BEDROCK})", "import it statically"),
        (f"importlib.__dict__['import_module']({_BEDROCK})", "statically"),
        (f"__builtins__['__import__']({_BEDROCK})", "import it statically"),
        ("vars(importlib)[name](module)", "cannot read"),
        ("getattr(importlib, 'import_' + 'module')(name)", "cannot read"),
        # Named without being called there.
        ("load = importlib.import_module\nload(name)", "cannot read"),
        (
            "import functools\n"
            f"functools.partial(importlib.import_module, {_BEDROCK})()",
            "cannot read",
        ),
        ("load = getattr(importlib, '__import__')", "cannot read"),
        ("from pkgutil import resolve_name as rn\nrn(name)", "cannot read"),
        ("importlib.import_module(**options)", "cannot read"),
        # Not an import.
        ("cursor.exec(statement)", None),
        ("model.eval()", None),
        ("from importlib.util import find_spec as has\nhas(name)", None),
        ("getattr(importlib, 'reload')(module)", None),
        ("from mylib import import_module as fetch\nfetch(name)", None),
        ("check = importlib.util.find_spec", None),
        ("self.__dict__['counts'].append(1)", None),
        ("getattr(self, 'import_' + kind)()", None),
    ],
)
def test_disguised_dynamic_imports_are_flagged(
    tmp_path: Path, code: str, rule: str | None
) -> None:
    path = tmp_path / "domain" / "x.py"
    path.parent.mkdir(parents=True)
    path.write_text(f"import importlib\n{code}\n")
    found = violations(tmp_path)
    if rule is None:
        assert found == []
    else:
        assert len(found) == 1 and rule in found[0], found


@pytest.mark.parametrize(
    ("source", "statement", "rule"),
    [
        # The package's __init__ loads a backend for every importer.
        (
            "application/x.py",
            "from unified_kg_rag.adapters.storage.filter_schema import F",
            "a backend; inject it",
        ),
        (
            "evaluation/x.py",
            "from unified_kg_rag.adapters.llm import helper",
            "evaluation imports no store adapter",
        ),
        # Inside the package its __init__ is loaded already.
        ("adapters/storage/x.py", "from . import filter_schema", None),
        # The composition root may load backends.
        (
            "application/bootstrap.py",
            "from unified_kg_rag.adapters.storage.filter_schema import F",
            None,
        ),
        # A plugin package root re-exports its plugins.
        (
            "application/cli/x.py",
            "from unified_kg_rag.adapters.renderers.base import R",
            None,
        ),
    ],
)
def test_package_init_edges_count_for_the_importer(
    tmp_path: Path, source: str, statement: str, rule: str | None
) -> None:
    storage = tmp_path / "adapters" / "storage"
    storage.mkdir(parents=True)
    (tmp_path / "adapters" / "__init__.py").write_text("")
    (storage / "__init__.py").write_text("from .nested import thing\n")
    (storage / "filter_schema.py").write_text("")
    # A nested package's __init__ is followed as well.
    (storage / "nested").mkdir()
    (storage / "nested" / "__init__.py").write_text(
        "from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer\n"
    )
    (storage / "neptune_indexer.py").write_text("")
    llm = tmp_path / "adapters" / "llm"
    llm.mkdir()
    (llm / "__init__.py").write_text(
        "from unified_kg_rag.adapters.storage import filter_schema\n"
    )
    renderers = tmp_path / "adapters" / "renderers"
    renderers.mkdir()
    (renderers / "__init__.py").write_text("from .static import StaticRenderer\n")
    (renderers / "static.py").write_text("")
    path = tmp_path / source
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{statement}\n")
    found = [
        v for v in violations(tmp_path) if v.startswith(source[:-3].replace("/", "."))
    ]
    if rule is None:
        assert found == []
    else:
        assert any(rule in v for v in found), found
        assert all("/__init__)" in v for v in found), found


# Third-party store clients and the backends (vendor bindings and port
# implementations) that must stay unloaded by the modules that receive them
# through ``Providers``.
_STORE_MODULES = (
    "gremlin_python",
    "opensearchpy",
    *(f"{PACKAGE}.{backend}" for backend in _BACKENDS),
)
# The modules that have their backends injected: every module of these
# packages but the composition roots.
_INJECTED = (
    "application",
    "evaluation",
    *_BACKEND_FREE_ADAPTERS,
)


def _injected_modules(root: Path = _PACKAGE_ROOT) -> list[str]:
    modules = sorted(_module_name(f, root) for f in root.rglob("*.py"))
    return [
        f"{PACKAGE}.{m}"
        for m in modules
        if _within(m, _INJECTED) and not _within(m, _COMPOSITION_ROOTS)
    ]


# Imports each module in turn and reports the watched modules that import
# loaded first (one interpreter for all: a backend any of them loads is in
# sys.modules after it). A module whose optional dependency is missing (a
# core install without the eval extra) is skipped.
_LOADED_BY = """
import importlib, json, sys
watched, loaded = {watched!r}, {{}}
for module in {modules!r}:
    before = set(sys.modules)
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as e:
        if (e.name or "").startswith("unified_kg_rag"):
            raise
        continue
    new = set(sys.modules) - before
    hit = sorted(w for w in watched if any(m == w or m.startswith(w + ".") for m in new))
    if hit:
        loaded[module] = hit
print(json.dumps(loaded))
"""


def _loaded_by(modules: list[str]) -> dict[str, list[str]]:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _LOADED_BY.format(watched=_STORE_MODULES, modules=modules),
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return dict(json.loads(result.stdout.strip().splitlines()[-1]))


def test_injected_modules_load_no_store_backend() -> None:
    """The runtime side of the static rules: what a clean interpreter has
    loaded after importing each module that has its backends injected. The
    scan sees edges; this sees a backend an allowed module passes on."""
    modules = _injected_modules()
    assert f"{PACKAGE}.application.retrieval.rag_chain" in modules
    assert f"{PACKAGE}.evaluation" in modules
    assert not any(".application.cli" in m or m.endswith("bootstrap") for m in modules)
    loaded = _loaded_by(modules)
    assert not loaded, f"backends loaded by injected modules: {loaded}"


def test_every_injected_package_contributes_a_module() -> None:
    # A renamed or emptied package would drop out of the runtime check
    # silently.
    modules = _injected_modules()
    for package in _INJECTED:
        assert any(m.startswith(f"{PACKAGE}.{package}") for m in modules), package


def test_the_runtime_check_sees_a_backend_passed_on() -> None:
    # One module loads a store client (as an allowed adapter module passing
    # a backend on would); the check names it, not the clean ones.
    loaded = _loaded_by(["json", "gremlin_python.process.traversal", "csv"])
    assert loaded == {"gremlin_python.process.traversal": ["gremlin_python"]}
