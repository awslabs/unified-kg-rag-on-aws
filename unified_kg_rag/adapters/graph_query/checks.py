# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static checks of a compiled structured query, before it is sent
(docs/design.md §6.6, safety layer 4).

They read the compiled form only, not the plan or the compiler, so a
compiler bug cannot also blind them. A failure is a bug: it is logged at
ERROR and raised as ``GraphQueryCompileError``, and nothing is sent.

- :func:`check_gremlin`: every step of the bytecode (nested traversals
  included) is a read step of the allowlist; the traversal starts with
  ``V()`` followed by ``hasLabel(label)``; every step that produces a
  vertex is followed by ``hasLabel(label)``, and no other label is named;
  the only source instruction is ``OptionsStrategy`` with
  ``evaluationTimeout``.
- :func:`check_sql`: one ``SELECT`` or ``WITH`` statement without a quote,
  comment, statement separator, named parameter or write/administration
  keyword, reading only the ``vertices`` and ``edges`` tables and its own
  CTEs, calling only the allowlisted functions, without ``NOT``, ``IS``
  (but ``IS NOT NULL``), ``CASE``, ``BETWEEN`` or an outer, cross or
  natural join, with every ``vertices``
  and ``edges`` alias restricted by ``<alias>.label = ?`` as a top-level
  conjunct of its ``WHERE`` or ``ON`` clause where that parameter is the
  label.
"""

from __future__ import annotations

import bisect
import re
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import GraphQueryCompileError

if TYPE_CHECKING:
    from gremlin_python.process.traversal import Bytecode, Traversal

logger = get_logger(__name__)

# The read steps the compilers emit (bytecode names: "as", "not", "id").
GREMLIN_STEPS = frozenset(
    {
        "V",
        "hasLabel",
        "has",
        "hasNot",
        "hasKey",
        "where",
        "and",
        "or",
        "not",
        "outE",
        "inE",
        "bothE",
        "inV",
        "outV",
        "otherV",
        "as",
        "select",
        "project",
        "by",
        "values",
        "valueMap",
        "properties",
        "key",
        "value",
        "id",
        "label",
        "dedup",
        "count",
        "groupCount",
        "group",
        "fold",
        "unfold",
        "limit",
        "order",
        "union",
        "simplePath",
        "path",
        "coalesce",
        "constant",
        "identity",
        "local",
    }
)
# Named in the message when found: what a write or an escape would use.
GREMLIN_DENIED = frozenset(
    {
        "addV",
        "addE",
        "property",
        "drop",
        "mergeV",
        "mergeE",
        "inject",
        "sideEffect",
        "io",
        "call",
        "subgraph",
        "program",
        "withComputer",
        "E",
        "aggregate",
        "store",
        "cap",
        "sack",
        "math",
    }
)
# Steps that produce vertices: each must be followed by hasLabel(label).
_VERTEX_STEPS = frozenset({"V", "inV", "outV", "otherV", "bothV", "out", "in", "both"})
_SOURCE_OPTIONS = frozenset({"evaluationTimeout"})


def _refuse(kind: str, reason: str) -> GraphQueryCompileError:
    logger.error("Refused a compiled %s query (a compiler bug): %s", kind, reason)
    return GraphQueryCompileError(f"compiled {kind} query refused: {reason}")


def check_gremlin(traversal: Traversal | Bytecode, label: str) -> None:
    """Raise ``GraphQueryCompileError`` unless ``traversal`` is a read-only
    traversal of ``label``'s vertices (module docstring)."""
    # Here, not at the top: the SQLite graph reads this module without the
    # Gremlin client.
    from gremlin_python.process.strategies import OptionsStrategy
    from gremlin_python.process.traversal import Bytecode

    bytecode = traversal if isinstance(traversal, Bytecode) else traversal.bytecode
    for instruction in bytecode.source_instructions:
        name, args = instruction[0], instruction[1:]
        if name != "withStrategies" or not all(
            isinstance(arg, OptionsStrategy)
            and set(arg.configuration) <= _SOURCE_OPTIONS
            for arg in args
        ):
            raise _refuse("Gremlin", f"source instruction '{name}' is not allowed")
    steps = bytecode.step_instructions
    if not steps or steps[0][0] != "V":
        raise _refuse("Gremlin", "the traversal must start with V()")
    _check_steps(steps, label)


def _check_steps(steps: Sequence[Sequence[Any]], label: str) -> None:
    for index, step in enumerate(steps):
        name, args = str(step[0]), list(step[1:])
        if name in GREMLIN_DENIED or name not in GREMLIN_STEPS:
            raise _refuse("Gremlin", f"step '{name}' is not allowed")
        if name == "hasLabel" and args != [label]:
            raise _refuse("Gremlin", "hasLabel names another label")
        if name in _VERTEX_STEPS:
            if name == "V" and args:
                raise _refuse("Gremlin", "V() takes no ids")
            following = steps[index + 1] if index + 1 < len(steps) else None
            if following is None or following[0] != "hasLabel":
                raise _refuse("Gremlin", f"'{name}' is not followed by hasLabel(label)")
        for arg in args:
            _check_argument(arg, label)


def _check_argument(arg: Any, label: str) -> None:
    """Every argument by its type alone: scalars, predicates, the ordering
    tokens, nested traversals and collections of them. Anything else (a
    lambda, a strategy, a token such as ``T.label``) is refused, and so is a
    string naming a hidden token (``~label``, ``~id``)."""
    from gremlin_python.process.traversal import (
        Bytecode,
        Column,
        Order,
        P,
        Scope,
        Traversal,
    )

    if arg is None or isinstance(arg, bool | int | float):
        return
    if isinstance(arg, str):
        if arg.startswith("~"):
            raise _refuse("Gremlin", f"argument {arg!r} names a hidden token")
        return
    if isinstance(arg, Order | Scope | Column):
        return
    if isinstance(arg, Traversal):
        arg = arg.bytecode
    if isinstance(arg, Bytecode):
        if arg.source_instructions:
            raise _refuse("Gremlin", "a nested traversal has a source")
        _check_steps(arg.step_instructions, label)
    elif isinstance(arg, P):
        _check_argument(arg.value, label)
        if arg.other is not None:
            _check_argument(arg.other, label)
    elif isinstance(arg, list | tuple | set | frozenset):
        for item in arg:
            _check_argument(item, label)
    elif isinstance(arg, dict):
        for key, item in arg.items():
            _check_argument(key, label)
            _check_argument(item, label)
    else:
        raise _refuse(
            "Gremlin", f"an argument of type {type(arg).__name__} is not allowed"
        )


# --- SQL ---------------------------------------------------------------------------

SQL_TABLES = frozenset({"vertices", "edges"})
# The functions the SQLite compiler calls: the vertex, edge and group-key
# predicates it registers on the connection, and the aggregate count.
SQL_FUNCTIONS = frozenset(
    {"kg_node", "kg_edge", "kg_key", "kg_rank", "kg_sortable", "count"}
)
_SQL_KEYWORDS_BEFORE_PAREN = frozenset(
    {"in", "as", "from", "join", "on", "and", "or", "not", "where", "select", "all"}
)
_SQL_DENIED = re.compile(
    r"\b(insert|update|delete|replace|create|drop|alter|attach|detach|pragma|"
    r"vacuum|reindex|analyze|savepoint|release|begin|commit|rollback|"
    r"transaction|returning|trigger|view|load_extension|conflict|recursive|"
    r"or|having|window|over|values)\b",
    re.IGNORECASE,
)
# A numbered placeholder (?1) binds a parameter out of its position.
_NUMBERED = re.compile(r"\?\d")
# Operators that could negate or bypass a label restriction (NOT v.label = ?,
# (v.label = ?) IS 0, CASE WHEN, BETWEEN ... AND v.label = ?); the one IS
# the compiler writes, IS NOT NULL, is allowed.
_SQL_NEGATING = re.compile(r"\b(not|is|case|between)\b", re.IGNORECASE)
# An outer, cross or natural join keeps the rows its ON clause does not
# match (or has none): the compiler joins with an inner JOIN ... ON only.
_SQL_JOIN_KINDS = re.compile(
    r"\b(left|right|full|outer|cross|natural|using)\b", re.IGNORECASE
)
_IS_NOT_NULL = re.compile(r"\bis\s+not\s+null\b", re.IGNORECASE)
# What may stand before and after a top-level conjunct of a WHERE or ON
# clause.
_CONJUNCT_BEFORE = frozenset({"where", "on", "and"})
_CONJUNCT_AFTER = frozenset(
    {"and", "join", "where", "group", "order", "limit", "union"}
)
_CLAUSE_START = frozenset({"where", "on"})
_SQL_FORBIDDEN_CHARS = frozenset(";'\"`:@$[]\\")
_TABLE_REF = re.compile(r"\b(?:from|join)\s+([a-z_][a-z0-9_]*|\()", re.IGNORECASE)
_STORE_TABLE = re.compile(
    r"\b(vertices|edges)\b(\s+as\s+([a-z_][a-z0-9_]*))?", re.IGNORECASE
)
_CTE = re.compile(
    r"(?:\bwith|,)\s*([a-z_][a-z0-9_]*)\s*(?:\([^()]*\))?\s+as\s*\(", re.IGNORECASE
)
_CALL = re.compile(r"\b([a-z_][a-z0-9_]*)\s*\(", re.IGNORECASE)


def check_sql(sql: str, params: Sequence[Any], label: str) -> None:
    """Raise ``GraphQueryCompileError`` unless ``sql`` with ``params`` is a
    read-only query of ``label``'s rows (module docstring)."""
    text = sql.strip()
    if not re.match(r"(?is)^(with|select)\b", text):
        raise _refuse("SQL", "the statement must start with SELECT or WITH")
    if found := sorted(set(text) & _SQL_FORBIDDEN_CHARS):
        raise _refuse("SQL", f"characters {''.join(found)!r} are not allowed")
    if "--" in text or "/*" in text:
        raise _refuse("SQL", "comments are not allowed")
    if match := _SQL_DENIED.search(text):
        raise _refuse("SQL", f"keyword '{match.group(1)}' is not allowed")
    if _NUMBERED.search(text):
        raise _refuse("SQL", "numbered placeholders are not allowed")
    if match := _SQL_NEGATING.search(_IS_NOT_NULL.sub(" ", text)):
        raise _refuse("SQL", f"keyword '{match.group(1)}' is not allowed")
    if match := _SQL_JOIN_KINDS.search(text):
        raise _refuse("SQL", f"keyword '{match.group(1)}' is not allowed")
    if text.count("?") != len(params):
        raise _refuse("SQL", "the parameters do not match the placeholders")
    _check_from_clauses(text)
    ctes = {name.lower() for name in _CTE.findall(text)}
    _check_ctes(text, ctes)
    for ref in _TABLE_REF.findall(text):
        if ref != "(" and ref.lower() not in SQL_TABLES | ctes:
            raise _refuse("SQL", f"table '{ref}' is not allowed")
    for name in set(_CALL.findall(text)):
        lowered = name.lower()
        if lowered in _SQL_KEYWORDS_BEFORE_PAREN or lowered in ctes:
            continue
        if lowered not in SQL_FUNCTIONS:
            raise _refuse("SQL", f"function '{name}' is not allowed")
    aliases: dict[str, None] = {}
    for match in _STORE_TABLE.finditer(text):
        if match.group(3) is None:
            raise _refuse("SQL", f"'{match.group(1)}' is read without an alias")
        aliases[match.group(3)] = None
    if not aliases:
        raise _refuse("SQL", "the statement reads no store table")
    marks = [i for i, char in enumerate(text) if char == "?"]
    for alias in aliases:
        _check_label_bound(text, params, alias, label, marks)


def _check_from_clauses(text: str) -> None:
    """Refuse a comma join: in a FROM clause, a comma at the clause's own
    nesting level (the compiler joins with JOIN ... ON only)."""
    lowered = text.lower()
    for match in re.finditer(r"\bfrom\b", lowered):
        depth = 0
        index = match.end()
        while index < len(lowered):
            char = lowered[index]
            if char == "(":
                depth += 1
            elif char == ")":
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0:
                if char == ",":
                    raise _refuse("SQL", "a comma join is not allowed")
                if re.match(r"\b(where|group|order|limit|union)\b", lowered[index:]):
                    break
            index += 1


def _check_ctes(text: str, ctes: set[str]) -> None:
    """Refuse a CTE that reads itself (a recursive one: unbounded)."""
    for name in ctes:
        head = re.search(
            rf"(?:\bwith|,)\s*{re.escape(name)}\s*(?:\([^()]*\))?\s+as\s*\(",
            text,
            re.IGNORECASE,
        )
        if head is None:
            continue
        depth, index = 1, head.end()
        while index < len(text) and depth:
            depth += {"(": 1, ")": -1}.get(text[index], 0)
            index += 1
        body = text[head.end() : index]
        if re.search(rf"\b{re.escape(name)}\b", body, re.IGNORECASE):
            raise _refuse("SQL", f"CTE '{name}' reads itself")


def _check_label_bound(
    text: str, params: Sequence[Any], alias: str, label: str, marks: list[int]
) -> None:
    pattern = re.compile(rf"\b{re.escape(alias)}\.label\s*=\s*\?")
    matches = list(pattern.finditer(text))
    if not matches:
        raise _refuse("SQL", f"alias '{alias}' has no label = ? restriction")
    for match in matches:
        if not _is_top_level_conjunct(text, match.start(), match.end()):
            raise _refuse(
                "SQL",
                f"alias '{alias}' label = ? is not a top-level conjunct of its "
                "WHERE or ON clause",
            )
        index = bisect.bisect_left(marks, match.end() - 1)
        if params[index] != label:
            raise _refuse("SQL", f"alias '{alias}' is bound to another label")


def _is_top_level_conjunct(text: str, start: int, end: int) -> bool:
    """Whether ``text[start:end]`` is one of the AND-joined terms of a
    WHERE or ON clause at that clause's own nesting level (OR, NOT, IS and
    CASE are refused before), so no other operator can negate it: the word
    before it is WHERE, ON or AND, the token after it ends the term, and
    walking back to the clause keyword crosses no unclosed parenthesis."""
    before = re.search(r"([a-z_]+|\S)\s*$", text[:start], re.IGNORECASE)
    if before is None or before.group(1).lower() not in _CONJUNCT_BEFORE:
        return False
    after = re.match(r"\s*([a-z_]+|\S|$)", text[end:], re.IGNORECASE)
    token = after.group(1).lower() if after else ""
    if token not in _CONJUNCT_AFTER and token not in ("", ")"):
        return False
    depth = 0
    for word in reversed(re.findall(r"[a-z_]+|[()]", text[:start], re.IGNORECASE)):
        if word == ")":
            depth += 1
        elif word == "(":
            if depth == 0:
                return False
            depth -= 1
        elif depth == 0 and word.lower() in _CLAUSE_START:
            return True
    return False


def placeholders(values: Iterable[Any]) -> str:
    """One ``?`` per value, comma-separated."""
    return ",".join("?" for _ in values)
