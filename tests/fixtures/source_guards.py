# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-owner guards over the package source: one concept, one module.

A concept implemented in several places drifts: each copy is fixed on its
own, or not at all (the namespace rule, the file-name readers, the script
ranges). A ``SourceGuard`` names the pattern of a copy, the module(s) that
own the concept, and an allowlist of the other places that may hold the
pattern, each with its reason. Its test fails on a match outside the owners
and the allowlist (a new copy), and on an allowlist entry that matches
nothing (a copy removed, so the entry goes too): the allowlist only
shrinks.

A match is keyed by its module path (relative to the package) and its
enclosing function or class (``path::Class.method``, ``path::<module>``),
not by line, so editing a file does not move its entries.
"""

from __future__ import annotations

import ast
import functools
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "unified_kg_rag"

# (line, snippet) of each match in one module.
Finder = Callable[[ast.Module, str], Iterator[tuple[int, str]]]


@dataclass(frozen=True)
class Hit:
    """One match: where it is, the scope that holds it, what it reads."""

    path: str
    line: int
    scope: str
    snippet: str

    @property
    def key(self) -> str:
        return f"{self.path}::{self.scope}"

    def __str__(self) -> str:
        return f"{self.path}:{self.line} ({self.scope}) {self.snippet}"


@dataclass(frozen=True)
class SourceGuard:
    """A concept's single owner: ``find`` matches a copy of it; matches in
    ``owners`` (module paths or directory prefixes, relative to the
    package) are the concept itself, and ``allowed`` maps each other
    accepted ``path::scope`` to its reason. ``use`` says what a new copy
    should call instead. ``within`` limits the scan to these directories."""

    name: str
    use: str
    owners: tuple[str, ...]
    find: Finder
    allowed: Mapping[str, str] = field(default_factory=dict)
    within: tuple[str, ...] = ("",)

    def hits(self) -> list[Hit]:
        return list(self._hits)

    @functools.cached_property
    def _hits(self) -> tuple[Hit, ...]:
        found = []
        for path in _modules(self.within):
            relative = path.relative_to(PACKAGE).as_posix()
            text, tree, scopes = _parsed(path)
            for line, snippet in self.find(tree, text):
                found.append(Hit(relative, line, scopes.get(line, "<module>"), snippet))
        return tuple(found)

    def _owned(self, hit: Hit) -> bool:
        return any(
            hit.path == owner or hit.path.startswith(owner.rstrip("/") + "/")
            for owner in self.owners
        )

    def copies(self) -> list[Hit]:
        """The matches outside the owners and the allowlist."""
        return [
            hit
            for hit in self.hits()
            if not self._owned(hit) and hit.key not in self.allowed
        ]

    def stale(self) -> list[str]:
        """The allowlist entries no match needs any more."""
        keys = {hit.key for hit in self.hits() if not self._owned(hit)}
        return sorted(set(self.allowed) - keys)


@functools.cache
def _parsed(path: Path) -> tuple[str, ast.Module, dict[int, str]]:
    """A module's text, tree and line scopes, parsed once per process."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    return text, tree, _scopes(tree)


def _modules(within: Iterable[str]) -> list[Path]:
    paths: set[Path] = set()
    for part in within:
        path = PACKAGE / part
        paths.update([path] if path.is_file() else path.rglob("*.py"))
    return sorted(paths)


def _scopes(tree: ast.Module) -> dict[int, str]:
    """Line number to the qualified name of the innermost def or class."""
    scopes: dict[int, str] = {}

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{prefix}{child.name}"
                end = child.end_lineno or child.lineno
                for line in range(child.lineno, end + 1):
                    scopes[line] = name
                visit(child, f"{name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return scopes


@functools.cache
def _all_nodes(tree: ast.Module) -> tuple[ast.AST, ...]:
    """Every node of ``tree``, walked once (the guards share the walk)."""
    return tuple(ast.walk(tree))


@functools.cache
def _docstrings(tree: ast.Module) -> frozenset[int]:
    """The ids of the docstring constants (a module's, class's, def's)."""
    ids = set()
    for node in _all_nodes(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return frozenset(ids)


# --- Finders -----------------------------------------------------------------


def string_literals(*values: str) -> Finder:
    """A string constant equal to one of ``values`` (docstrings aside)."""

    def find(tree: ast.Module, _: str) -> Iterator[tuple[int, str]]:
        docstrings = _docstrings(tree)
        for node in _all_nodes(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in values
                and id(node) not in docstrings
            ):
                yield node.lineno, repr(node.value)

    return find


def calls(*names: str) -> Finder:
    """A call of one of ``names``, dotted (``json.loads``, ``os.replace``)."""
    wanted = {tuple(name.split(".")) for name in names}

    def find(tree: ast.Module, _: str) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(tree):
            if isinstance(node, ast.Call) and _dotted(node.func) in wanted:
                yield node.lineno, ast.unparse(node.func)

    return find


def _dotted(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return (node.id, *reversed(parts))
    return None


def key_reads(*keys: str) -> Finder:
    """A read of one of ``keys`` from a mapping: ``x.get("key"...)`` or
    ``x["key"]`` (a load, not a store)."""

    def find(tree: ast.Module, _: str) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in keys
            ):
                yield node.lineno, ast.unparse(node)[:80]
            elif (
                isinstance(node, ast.Subscript)
                and isinstance(node.ctx, ast.Load)
                and isinstance(node.slice, ast.Constant)
                and node.slice.value in keys
            ):
                yield node.lineno, ast.unparse(node)[:80]

    return find


def lines_matching(pattern: re.Pattern[str]) -> Finder:
    """A source line ``pattern`` matches, outside comments."""

    def find(_: ast.Module, text: str) -> Iterator[tuple[int, str]]:
        for number, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0] if "#" in line else line
            if pattern.search(code):
                yield number, line.strip()[:80]

    return find


def text_matching(pattern: re.Pattern[str]) -> Finder:
    """A match of ``pattern`` anywhere in the source, across lines (a copy
    written over several lines), at the line it starts on."""

    def find(_: ast.Module, text: str) -> Iterator[tuple[int, str]]:
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            yield line, " ".join(match.group(0).split())[:80]

    return find


def nodes(predicate: Callable[[ast.AST], bool]) -> Finder:
    """A node ``predicate`` accepts."""

    def find(tree: ast.Module, _: str) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(tree):
            if predicate(node):
                yield getattr(node, "lineno", 0), ast.unparse(node)[:80]

    return find


def any_of(*finders: Finder) -> Finder:
    """What any of ``finders`` matches."""

    def find(tree: ast.Module, text: str) -> Iterator[tuple[int, str]]:
        for finder in finders:
            yield from finder(tree, text)

    return find


def matches(finder: Finder, source: str) -> list[str]:
    """What ``finder`` matches in ``source``: the pin of a pattern."""
    return [snippet for _, snippet in finder(ast.parse(source), source)]
