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
import replace`` makes ``replace`` ``os.replace``, and a relative import is
resolved against the module's package), so an alias hides no copy.

``tests/fixtures/guard_copies`` holds planted copies the finders were not
written against, each naming the guard it must trip (or must not):
``test_single_owner_guards`` measures every guard on them.
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


@dataclass(frozen=True, eq=False)
class Source:
    """A parsed module: its tree, its text and the dotted package its
    relative imports resolve against (``unified_kg_rag.evaluation``)."""

    tree: ast.Module
    text: str
    package: str

    @classmethod
    def of(cls, text: str, path: str = "probe.py") -> Source:
        """``text`` parsed as the module at ``path`` (relative to the
        package)."""
        return cls(ast.parse(text), text, package_of(path))


def package_of(path: str) -> str:
    """The dotted package of the module at ``path`` (relative to the
    package): ``evaluation/report.py`` and ``evaluation/__init__.py`` are
    both in ``unified_kg_rag.evaluation``."""
    return ".".join((PACKAGE.name, *Path(path).parts[:-1]))


# (line, snippet) of each match in one module.
Finder = Callable[[Source], Iterator[tuple[int, str]]]

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
        found: list[Hit] = []
        for path in _modules((*self.within, *self.owners)):
            relative = path.relative_to(PACKAGE).as_posix()
            source, scopes = _parsed(path)
            found += _hits_of(self.find, relative, source, scopes)
        return tuple(found)

    def hits_in(self, path: str, text: str) -> list[Hit]:
        """What ``find`` matches in ``text`` read as the module at ``path``
        (a planted copy)."""
        source = Source.of(text, path)
        return _hits_of(self.find, path, source, _scopes(source.tree))

    def scans(self, path: str) -> bool:
        """Whether a copy at ``path`` would count: the guard reads the
        module, and no owner holds it."""
        return not self._owned(path) and any(_under(path, p) for p in self.within)

    def _owned(self, path: str) -> bool:
        return any(_under(path, owner) for owner in self.owners)

    def _counts(self) -> Counter[str]:
        return Counter(hit.key for hit in self.hits() if not self._owned(hit.path))

    def copies(self) -> list[Hit]:
        """The matches outside the owners and the allowlist, and every match
        of an allowed place that holds more than its entry counts."""
        counts = self._counts()
        return [
            hit
            for hit in self.hits()
            if not self._owned(hit.path)
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
        held = {hit.path for hit in self.hits() if self._owned(hit.path)}
        return [
            owner
            for owner in self.owners
            if not any(_under(path, owner) for path in held)
        ]

    def missing_paths(self) -> list[str]:
        """The owners, ``within`` paths and allowed modules that do not exist."""
        paths = (
            *self.owners,
            *(p for p in self.within if p),
            *(key.split("::", 1)[0] for key in self.allowed),
        )
        return [p for p in paths if not (PACKAGE / p).exists()]


def _under(path: str, prefix: str) -> bool:
    """Whether the module ``path`` is ``prefix`` or under it (``""``: the
    whole package)."""
    return not prefix or path == prefix or path.startswith(prefix.rstrip("/") + "/")


def _hits_of(
    find: Finder, path: str, source: Source, scopes: Mapping[int, str]
) -> list[Hit]:
    return [
        Hit(path, line, scopes.get(line, "<module>"), snippet)
        for line, snippet in find(source)
    ]


@functools.cache
def _parsed(path: Path) -> tuple[Source, dict[int, str]]:
    """A module's source and line scopes, parsed once per process."""
    text = path.read_text(encoding="utf-8")
    source = Source.of(text, path.relative_to(PACKAGE).as_posix())
    return source, _scopes(source.tree)


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


def _absolute(node: ast.ImportFrom, package: str) -> str:
    """The module an ``ImportFrom`` names, a relative one resolved against
    ``package`` (``from ..shared import x`` in ``unified_kg_rag.evaluation``:
    ``unified_kg_rag.shared``)."""
    if not node.level:
        return node.module or ""
    parts = package.split(".")
    base = parts[: len(parts) - (node.level - 1)]
    return ".".join([*base, *([node.module] if node.module else [])])


@functools.cache
def _imports(source: Source) -> Mapping[str, str]:
    """Each name a module's imports bind, to the dotted name it stands for
    (``j`` of ``import json as j``: ``json``; ``r`` of ``from os import
    replace as r``: ``os.replace``; ``k`` of ``from .degradation import
    KEY as k``: the absolute ``unified_kg_rag....degradation.KEY``)."""
    names: dict[str, str] = {}
    for node in _all_nodes(source.tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    names[alias.asname] = alias.name
                else:
                    head = alias.name.split(".", 1)[0]
                    names[head] = head
        elif isinstance(node, ast.ImportFrom):
            module = _absolute(node, source.package)
            for alias in node.names:
                names[alias.asname or alias.name] = f"{module}.{alias.name}"
    return names


@functools.cache
def _bindings(source: Source) -> Mapping[str, tuple[ast.expr, ...]]:
    """Each local name to the values bound to it (``meta = out.metadata``,
    ``with x as meta``, ``(meta := x)``), wherever in the module."""
    bound: dict[str, list[ast.expr]] = {}
    for node in _all_nodes(source.tree):
        pairs: list[tuple[ast.expr, ast.expr]] = []
        if isinstance(node, ast.Assign):
            pairs = [(target, node.value) for target in node.targets]
        elif isinstance(node, ast.AnnAssign | ast.NamedExpr) and node.value:
            pairs = [(node.target, node.value)]
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            pairs = [(node.optional_vars, node.context_expr)]
        for target, value in pairs:
            if isinstance(target, ast.Name):
                bound.setdefault(target.id, []).append(value)
    return {name: tuple(values) for name, values in bound.items()}


def resolved(node: ast.AST, source: Source) -> str | None:
    """The dotted name ``node`` (a name or an attribute chain) stands for,
    its first part resolved through the module's imports."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    head = _imports(source).get(node.id, node.id)
    return ".".join([head, *reversed(parts)])


def imported(node: ast.AST, source: Source) -> bool:
    """Whether ``node`` is a name the module imports (a module or one of
    its functions), not a local value."""
    return isinstance(node, ast.Name) and node.id in _imports(source)


def _names_in(node: ast.AST, source: Source) -> set[str]:
    """The dotted names the names and attribute chains under ``node`` stand
    for."""
    return {
        name
        for child in ast.walk(node)
        if isinstance(child, ast.Name | ast.Attribute)
        and (name := resolved(child, source)) is not None
    }


# --- Finders -----------------------------------------------------------------


def string_literals(*values: str) -> Finder:
    """A string constant equal to one of ``values`` (docstrings aside)."""

    def find(source: Source) -> Iterator[tuple[int, str]]:
        docstrings = _docstrings(source.tree)
        for node in _all_nodes(source.tree):
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

    def find(source: Source) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(source.tree):
            if (
                isinstance(node, ast.Name | ast.Attribute)
                and isinstance(node.ctx, ast.Load)
                and resolved(node, source) in names
            ):
                yield node.lineno, ast.unparse(node)

    return find


def uses(*names: str) -> Finder:
    """A use of one of the dotted ``names`` or of anything under one: a
    denylist of modules and their members (``yaml`` matches
    ``yaml.safe_load(s)``, ``json.loads`` matches it called, passed as a
    callback or bound to a name), however the module imported it. A
    chain is matched once, at its outermost use (``yaml.safe_load``, not
    ``yaml`` in it too)."""
    wanted = tuple(names)

    def denied(node: ast.AST, source: Source) -> bool:
        name = resolved(node, source)
        return name is not None and any(
            name == w or name.startswith(w + ".") for w in wanted
        )

    def find(source: Source) -> Iterator[tuple[int, str]]:
        inner: set[int] = set()
        for node in _all_nodes(source.tree):
            if id(node) in inner or not isinstance(node, ast.Name | ast.Attribute):
                continue
            if isinstance(node.ctx, ast.Load) and denied(node, source):
                inner.update(id(n) for n in ast.walk(node))
                yield node.lineno, ast.unparse(node)[:80]

    return find


def calls(*names: str) -> Finder:
    """A call of one of the dotted ``names`` (``json.loads``,
    ``os.replace``), however the module imported it."""
    wanted = set(names)

    def find(source: Source) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(source.tree):
            if isinstance(node, ast.Call) and resolved(node.func, source) in wanted:
                yield node.lineno, ast.unparse(node.func)

    return find


def method_calls(*methods: str) -> Finder:
    """A call of a method named one of ``methods`` on any value
    (``Model.model_validate_json(raw)``)."""

    def find(source: Source) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(source.tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in methods
            ):
                yield node.lineno, ast.unparse(node.func)[:80]

    return find


def calls_mentioning(callees: Iterable[str], *names: str) -> Finder:
    """A call of one of the dotted ``callees`` whose arguments use one of
    the dotted ``names`` (``Counter(... script_of(c) ...)``)."""
    wanted = set(callees)

    def find(source: Source) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(source.tree):
            if isinstance(node, ast.Call) and resolved(node.func, source) in wanted:
                used = set().union(
                    *(_names_in(a, source) for a in (*node.args, *node.keywords))
                )
                if used & set(names):
                    yield node.lineno, ast.unparse(node)[:80]

    return find


# The methods and ``operator`` getters that read a key.
_KEY_METHODS = ("get", "pop")
_GETTERS = ("operator.itemgetter", "operator.attrgetter")


def key_reads(
    *keys: str, of: re.Pattern[str] | None = None, constants: Iterable[str] = ()
) -> Finder:
    """A read of one of ``keys`` from a mapping or an object:
    ``x.get("key"...)``, ``x.pop("key"...)``, ``x["key"]`` (a load, not a
    store), ``getattr(x, "key"...)``, ``operator.itemgetter("key")`` (or
    ``attrgetter``), a ``case {"key": ...}`` pattern, the key through a
    name bound to it, or a ``.get(k)`` / ``[k]`` in a loop ``for k in
    ("key", ...)``; with ``of``, only from an
    ``x`` whose source ``of`` searches, or that is a name bound to such a
    value (``metadata``: a result's raw metadata, ``meta =
    out.metadata``). ``constants`` are the dotted names of constants that
    hold a key (``RELATIVE_PATH_KEY``): a read through one is a read of the
    key."""
    held = set(constants)

    def find(source: Source) -> Iterator[tuple[int, str]]:
        bound = _bindings(source)

        def literal(node: ast.AST | None) -> bool:
            if isinstance(node, ast.Constant):
                return node.value in keys
            if isinstance(node, ast.Name) and any(
                isinstance(v, ast.Constant) and v.value in keys
                for v in bound.get(node.id, ())
            ):
                return True  # the key held in a name (``_FIELD = "key"``)
            return node is not None and resolved(node, source) in held

        def read_of(value: ast.AST | None) -> bool:
            if of is None:
                return True
            if value is None:
                return False
            texts = [ast.unparse(value)]
            if isinstance(value, ast.Name):
                texts += [ast.unparse(v) for v in bound.get(value.id, ())]
            return any(of.search(text) for text in texts)

        looped: set[tuple[int, str]] = set()  # (id of the loop body, its var)
        for node in _all_nodes(source.tree):
            if (
                isinstance(node, ast.For)
                and isinstance(node.target, ast.Name)
                and isinstance(node.iter, ast.Tuple | ast.List | ast.Set)
                and any(literal(e) for e in node.iter.elts)
            ):
                for child in (n for b in node.body for n in ast.walk(b)):
                    looped.add((id(child), node.target.id))
        for node in _all_nodes(source.tree):
            if isinstance(node, ast.Match):
                for case in node.cases:
                    for pattern in ast.walk(case.pattern):
                        if isinstance(pattern, ast.MatchMapping) and any(
                            literal(key) for key in pattern.keys
                        ):
                            if read_of(node.subject):
                                yield pattern.lineno, ast.unparse(pattern)[:80]
                continue
            arg: ast.AST | None = None
            subject: ast.AST | None = None
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in _KEY_METHODS
                and node.args
            ):
                arg, subject = node.args[0], node.func.value
            elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
                arg, subject = node.slice, node.value
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) >= 2
            ):
                arg, subject = node.args[1], node.args[0]
            elif (
                isinstance(node, ast.Call)
                and resolved(node.func, source) in _GETTERS
                and node.args
            ):
                # The value it reads from is its call's argument, unseen.
                arg = next((a for a in node.args if literal(a)), node.args[0])
            if (
                literal(arg)
                or (isinstance(arg, ast.Name) and (id(node), arg.id) in looped)
            ) and read_of(subject):
                yield getattr(node, "lineno", 0), ast.unparse(node)[:80]

    return find


def lines_matching(pattern: re.Pattern[str]) -> Finder:
    """A source line ``pattern`` matches, outside comments (for code
    forms; what a string holds is ``string_values``')."""

    def find(source: Source) -> Iterator[tuple[int, str]]:
        for number, line in enumerate(source.text.splitlines(), 1):
            if pattern.search(line.split("#", 1)[0]):
                yield number, line.strip()[:80]

    return find


def text_matching(pattern: re.Pattern[str]) -> Finder:
    """A match of ``pattern`` anywhere in the source, across lines (a copy
    written over several lines), at the line it starts on."""

    def find(source: Source) -> Iterator[tuple[int, str]]:
        text = source.text
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            yield line, " ".join(match.group(0).split())[:80]

    return find


def string_values(
    pattern: re.Pattern[str], accept: Callable[[re.Match[str]], bool] | None = None
) -> Finder:
    """A string constant (docstrings aside) whose value ``pattern``
    searches: what a regex or a character table holds, escapes decoded,
    wherever a ``#`` or a line break falls in its source. With ``accept``,
    only a match it accepts (a range whose ends it checks)."""

    def find(source: Source) -> Iterator[tuple[int, str]]:
        docstrings = _docstrings(source.tree)
        for node in _all_nodes(source.tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and any(
                    accept is None or accept(match)
                    for match in pattern.finditer(node.value)
                )
            ):
                yield node.lineno, repr(node.value)[:80]

    return find


def nodes(predicate: Callable[[ast.AST], bool]) -> Finder:
    """A node ``predicate`` accepts."""

    def find(source: Source) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(source.tree):
            if predicate(node):
                yield getattr(node, "lineno", 0), ast.unparse(node)[:80]

    return find


def module_nodes(predicate: Callable[[ast.AST, Source], bool]) -> Finder:
    """A node ``predicate`` accepts, given its module (to resolve names)."""

    def find(source: Source) -> Iterator[tuple[int, str]]:
        for node in _all_nodes(source.tree):
            if predicate(node, source):
                yield getattr(node, "lineno", 0), ast.unparse(node)[:80]

    return find


def any_of(*finders: Finder) -> Finder:
    """What any of ``finders`` matches, a line one of them matched counted
    once (two finders seeing one copy find one copy)."""

    def find(source: Source) -> Iterator[tuple[int, str]]:
        seen: set[int] = set()
        for finder in finders:
            found = list(finder(source))
            yield from ((line, s) for line, s in found if line not in seen)
            seen.update(line for line, _ in found)

    return find


def matches(finder: Finder, source: str, path: str = "probe.py") -> list[str]:
    """What ``finder`` matches in ``source`` read as the module at ``path``:
    the pin of a pattern."""
    return [snippet for _, snippet in finder(Source.of(source, path))]
