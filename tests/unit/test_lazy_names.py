# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A CLI module's lazily imported names are bound before code reads them.

``lazy.lazy_names`` gives a CLI module a ``__getattr__`` (an attribute
access, ``run_evaluation.compared_pairs``, imports the name) and a loader
that binds every name as a global. A function reading one as a bare global
finds it only once the loader ran: in the CLI the entry point runs it
first, but a function called without it (a static method a test or a
library caller uses directly) raised ``NameError``, and only when no
earlier test in the process had loaded the names.

So every function that reads a lazy name as a global calls the loader
itself, or is an instance method (or nested in one) of a class whose
``__init__`` does.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PACKAGE = Path(__file__).resolve().parents[2] / "unified_kg_rag"

_Def = ast.FunctionDef | ast.AsyncFunctionDef


def _lazy(tree: ast.Module) -> tuple[set[str], str] | None:
    """The lazy names and the loader's name of a module, if it has them."""
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "lazy_names"
            and isinstance(node.targets[0], ast.Tuple)
        ):
            loader = node.targets[0].elts[1]
            assert isinstance(loader, ast.Name)
            return _keys(node.value.args[1]), loader.id
    return None


def _keys(names: ast.expr) -> set[str]:
    """The names of a ``{name: module}`` literal or of a
    ``dict.fromkeys((names...), module)``."""
    if isinstance(names, ast.Dict):
        keys = names.keys
    else:
        assert (
            isinstance(names, ast.Call) and ast.unparse(names.func) == "dict.fromkeys"
        )
        assert isinstance(names.args[0], ast.Tuple | ast.List)
        keys = list(names.args[0].elts)
    return {k.value for k in keys if isinstance(k, ast.Constant)}


def _calls(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(n, ast.Call) and getattr(n.func, "id", None) == name
        for n in ast.walk(node)
    )


def _reads(node: _Def, names: set[str]) -> set[str]:
    """The lazy names ``node``'s body reads as globals (its nested defs and
    the annotations aside: the modules postpone them, so none is read)."""
    found: set[str] = set()
    stack: list[ast.AST] = list(node.body)
    while stack:
        child = stack.pop()
        if isinstance(child, _Def | ast.Lambda | ast.ClassDef):
            if isinstance(child, ast.Lambda):
                stack.extend(ast.iter_child_nodes(child))
            continue
        if isinstance(child, ast.AnnAssign):
            stack.extend(n for n in (child.target, child.value) if n is not None)
            continue
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            if child.id in names:
                found.add(child.id)
        stack.extend(ast.iter_child_nodes(child))
    return found


def _unbound_reads(tree: ast.Module, names: set[str], loader: str) -> list[str]:
    """``function: names`` of each function that may read a lazy name
    before the loader ran."""
    problems: list[str] = []

    def visit(node: ast.AST, scope: str, loaded: bool, cls: ast.ClassDef | None):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{scope}{child.name}.", False, child)
            elif isinstance(child, _Def):
                bound = loaded or _calls(child, loader)
                if cls is not None and not loaded:
                    decorators = {getattr(d, "id", None) for d in child.decorator_list}
                    init = next(
                        (
                            n
                            for n in cls.body
                            if isinstance(n, _Def) and n.name == "__init__"
                        ),
                        None,
                    )
                    if (
                        not decorators & {"staticmethod", "classmethod"}
                        and init is not None
                        and _calls(init, loader)
                    ):
                        bound = True
                if (read := _reads(child, names)) and not bound:
                    problems.append(f"{scope}{child.name}: {sorted(read)}")
                visit(child, f"{scope}{child.name}.", bound, None)

    visit(tree, "", False, None)
    return problems


# Every module postpones its annotations, which the scan skips.
_FUTURE = "from __future__ import annotations"

_MODULES = sorted(
    path
    for path in (PACKAGE / "application" / "cli").glob("*.py")
    if path.name != "lazy.py" and "lazy_names(" in path.read_text(encoding="utf-8")
)


def test_some_cli_module_has_lazy_names() -> None:
    assert len(_MODULES) >= 3, _MODULES


@pytest.mark.parametrize("path", _MODULES, ids=lambda p: p.stem)
def test_every_lazy_name_is_bound_before_it_is_read(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert _FUTURE in text
    tree = ast.parse(text)
    lazy = _lazy(tree)
    assert lazy is not None
    names, loader = lazy
    assert _unbound_reads(tree, names, loader) == [], f"call {loader}() first"


def test_the_scan_sees_a_static_method_without_the_loader() -> None:
    source = """
__getattr__, _load = lazy_names(globals(), dict.fromkeys(("thing",), _module))


class Runner:
    def __init__(self):
        _load()

    def method(self):
        return thing

    @staticmethod
    def static():
        return thing

    @staticmethod
    def loaded():
        _load()
        return thing

    @staticmethod
    def annotated(value: thing) -> thing:
        kept: thing = value
        return kept


def free():
    def inner():
        return thing

    return inner


def entry():
    _load()

    def inner():
        return thing

    return inner
"""
    tree = ast.parse(source)
    assert _lazy(tree) == ({"thing"}, "_load")
    assert _unbound_reads(tree, {"thing"}, "_load") == [
        "Runner.static: ['thing']",
        "free.inner: ['thing']",
    ]
