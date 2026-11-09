# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-owner guards over the package source: one concept, one module.

A concept implemented in several places drifts: each copy is fixed on its
own, or not at all (the namespace rule, the file-name readers, the script
ranges). A ``SourceGuard`` names the pattern of a copy, the module(s) that
own the concept, and an allowlist of the other places that may hold the
pattern, each with how many matches it holds and why. Its tests fail on:

- a match outside the owners and the allowlist, or one more in an allowed
  place than its entry counts (a new copy);
- an entry that counts more matches than its place holds (a copy removed:
  the entry shrinks or goes, so the allowlist only shrinks);
- an owner the pattern does not match (the finder no longer sees the
  concept's own form, so it would miss a copy of it), and an owner or a
  ``within`` path that does not exist (a typo scans nothing).

A match is keyed by its module path (relative to the package) and its
scope: the enclosing function or class (``path::Class.method``), the name a
module- or class-level assignment binds (``path::_TABLE``,
``path::Class.FIELD``), else ``path::<module>``; not by line, so editing a
file does not move its entries. Names a module imports are resolved
(``import json as j`` makes ``j.loads`` a ``json.loads`` call, ``from os
import replace`` makes ``replace`` ``os.replace``), so an alias hides no
copy.
"""

from __future__ import annotations

import ast
import functools
import re
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "unified_kg_rag"

# (line, snippet) of each match in one module.
Finder = Callable[[ast.Module, str], Iterator[tuple[int, str]]]

# An allowlist entry: how many matches its place holds, and why.
Allowed = tuple[int, str]


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
    accepted ``path::scope`` to its match count and reason. ``use`` says
    what a new copy should call instead. ``within`` limits the scan to
    these paths (the owners are always scanned)."""

    name: str
    use: str
    owners: tuple[str, ...]
    find: Finder
    allowed: Mapping[str, Allowed] = field(default_factory=dict)
    within: tuple[str, ...] = ("",)

    def hits(self) -> list[Hit]:
        return list(self._hits)

    @functools.cached_property
    def _hits(self) -> tuple[Hit, ...]:
        found = []
        for path in _modules((*self.within, *self.owners)):
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

    def _counts(self) -> Counter[str]:
        return Counter(hit.key for hit in self.hits() if not self._owned(hit))

    def copies(self) -> list[Hit]:
        """The matches outside the owners and the allowlist, and every match
        of an allowed place that holds more than its entry counts."""
        counts = self._counts()
        return [
            hit
            for hit in self.hits()
            if not self._owned(hit)
            and counts[hit.key] > self.allowed.get(hit.key, (0, ""))[0]
        ]

    def stale(self) -> list[str]:
        """The allowlist entries that count more matches than their place
        holds (none: the entry goes; fewer: its count shrinks)."""
        counts = self._counts()
        return [
            f"{key}: {expected} allowed, {counts[key]} found"
            for key, (expected, _) in sorted(self.allowed.items())
            if counts[key] < expected
        ]

    def unmatched_owners(self) -> list[str]:
        """The owners the pattern matches nothing in."""
        held = {hit.path for hit in self.hits() if self._owned(hit)}
        return [
            owner
            for owner in self.owners
            if not any(
                p == owner or p.startswith(owner.rstrip("/") + "/") for p in held
            )
        ]

    def missing_paths(self) -> list[str]:
        """The owners, ``within`` paths and allowed modules that do not exist."""
        paths = (
            *self.owners,
            *(p for p in self.within if p),
            *(key.split("::", 1)[0] for key in self.allowed),
        )
        return [p for p in paths if not (PACKAGE / p).exists()]


@functools.cache
def _parsed(path: Path) -> tuple[str, ast.Module, dict[int, str]]:
    """A module's text, tree and line scopes, parsed once per process."""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    return text, tree, _scopes(tree)


def _modules(paths: Iterable[str]) -> list[Path]:
    found: set[Path] = set()
    for part in paths:
        path = PACKAGE / part
        found.update([path] if path.is_file() else path.rglob("*.py"))
    return sorted(found)


def _bound_name(node: ast.AST) -> str | None:
    """The name a module- or class-level statement binds (its first)."""
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign | ast.AugAssign):
        targets = [node.target]
    for target in targets:
        if isinstance(target, ast.Tuple | ast.List) and target.elts:
            target = target.elts[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def _scopes(tree: ast.Module) -> dict[int, str]:
    """Line number to its scope: the qualified name of the innermost def or
    class, or of the name a module- or class-level assignment binds."""
    scopes: dict[int, str] = {}

    def mark(node: ast.stmt, name: str) -> None:
        for line in range(node.lineno, (node.end_lineno or node.lineno) + 1):
            scopes[line] = name

    def visit(node: ast.AST, prefix: str, in_def: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = f"{prefix}{child.name}"
                mark(child, name)
                visit(
                    child,
                    f"{name}.",
                    in_def or not isinstance(child, ast.ClassDef),
                )
            elif (
                not in_def
                and isinstance(child, ast.stmt)
                and (bound := _bound_name(child)) is not None
            ):
                mark(child, f"{prefix}{bound}")
                visit(child, prefix, in_def)
            else:
                visit(child, prefix, in_def)

    visit(tree, "", False)
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


@functools.cache
def _imports(tree: ast.Module) -> Mapping[str, str]:
    """Each name a module's imports bind, to the dotted name it stands for
    (``j`` of ``import json as j``: ``json``; ``r`` of ``from os import
    replace as r``: ``os.replace``; a relative module keeps its dots)."""
    names: dict[str, str] = {}
    for node in _all_nodes(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    names[alias.asname] = alias.name
                else:
                    head = alias.name.split(".", 1)[0]
                    names[head] = head
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            for alias in node.names:
                names[alias.asname or alias.name] = f"{module}.{alias.name}"
    return names


def resolved(node: ast.AST, tree: ast.Module) -> str | None:
    """The dotted name ``node`` (a name or an attribute chain) stands for,
    its first part resolved through the module's imports."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    head = _imports(tree).get(node.id, node.id)
    return ".".join([head, *reversed(parts)])


def _names_in(node: ast.AST, tree: ast.Module) -> set[str]:
    """The dotted names the names and attribute chains under ``node`` stand
    for."""
    return {
        name
        for child in ast.walk(node)
        if isinstance(child, ast.Name | ast.Attribute)
        and (name := resolved(child, tree)) is not None
    }


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


def references(*names: str) -> Finder:
    """A use of one of the dotted ``names`` (a module's constant: the key a
    copy reads instead of the literal), however it was imported."""

    def find(tree: ast.Module, _: str) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(tree):
            if (
                isinstance(node, ast.Name | ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and resolved(node, tree) in names
            ):
                yield node.lineno, ast.unparse(node)

    return find


def calls(*names: str) -> Finder:
    """A call of one of the dotted ``names`` (``json.loads``,
    ``os.replace``), however the module imported it."""
    wanted = set(names)

    def find(tree: ast.Module, _: str) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(tree):
            if isinstance(node, ast.Call) and resolved(node.func, tree) in wanted:
                yield node.lineno, ast.unparse(node.func)

    return find


def calls_mentioning(callee: str, *names: str) -> Finder:
    """A call of ``callee`` whose arguments use one of the dotted ``names``
    (``Counter(... script_of(c) ...)``)."""

    def find(tree: ast.Module, _: str) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(tree):
            if isinstance(node, ast.Call) and resolved(node.func, tree) == callee:
                used = set().union(
                    *(_names_in(a, tree) for a in (*node.args, *node.keywords))
                )
                if used & set(names):
                    yield node.lineno, ast.unparse(node)[:80]

    return find


def key_reads(*keys: str, of: re.Pattern[str] | None = None) -> Finder:
    """A read of one of ``keys`` from a mapping or an object:
    ``x.get("key"...)``, ``x["key"]`` (a load, not a store),
    ``getattr(x, "key"...)``, or a ``.get(k)`` / ``[k]`` in a loop
    ``for k in ("key", ...)``; with ``of``, only from an ``x`` whose source
    ``of`` searches (``metadata``: a result's raw metadata)."""

    def literal(node: ast.AST | None) -> bool:
        return isinstance(node, ast.Constant) and node.value in keys

    def find(tree: ast.Module, _: str) -> Iterator[tuple[int, str]]:
        looped: set[tuple[int, str]] = set()  # (id of the loop body, its var)
        for node in _all_nodes(tree):
            if (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Tuple | ast.List | ast.Set)
                and any(literal(e) for e in node.iter.elts)
            ):
                for child in (n for b in node.body for n in ast.walk(b)):
                    looped.add((id(child), node.target.id))
        for node in _all_nodes(tree):
            arg: ast.AST | None = None
            source: ast.AST | None = None
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and node.args
            ):
                arg, source = node.args[0], node.func.value
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                arg, source = node.slice, node.value
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
            ):
                arg, source = node.args[1], node.args[0]
            if (
                literal(arg)
                or (isinstance(arg, ast.Name) and (id(node), arg.id) in looped)
            ) and (
                of is None or (source is not None and of.search(ast.unparse(source)))
            ):
                yield getattr(node, "lineno", 0), ast.unparse(node)[:80]

    return find


def lines_matching(pattern: re.Pattern[str]) -> Finder:
    """A source line ``pattern`` matches, outside comments (for code
    forms; what a string holds is ``string_values``')."""

    def find(_: ast.Module, text: str) -> Iterator[tuple[int, str]]:
        for number, line in enumerate(text.splitlines(), 1):
            if pattern.search(line.split("#", 1)[0]):
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


def string_values(pattern: re.Pattern[str]) -> Finder:
    """A string constant (docstrings aside) whose value ``pattern``
    searches: what a regex or a character table holds, escapes decoded,
    wherever a ``#`` or a line break falls in its source."""

    def find(tree: ast.Module, _: str) -> Iterator[tuple[int, str]]:
        docstrings = _docstrings(tree)
        for node in _all_nodes(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and pattern.search(node.value)
            ):
                yield node.lineno, repr(node.value)[:80]

    return find


def nodes(predicate: Callable[[ast.AST], bool]) -> Finder:
    """A node ``predicate`` accepts."""

    def find(tree: ast.Module, _: str) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(tree):
            if predicate(node):
                yield getattr(node, "lineno", 0), ast.unparse(node)[:80]

    return find


def module_nodes(predicate: Callable[[ast.AST, ast.Module], bool]) -> Finder:
    """A node ``predicate`` accepts, given its module (to resolve names)."""

    def find(tree: ast.Module, _: str) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(tree):
            if predicate(node, tree):
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
