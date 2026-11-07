# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A validated plan compiled to parameterised SQL over the embedded graph.

:class:`SQLiteQuery` writes one statement over ``SQLiteGraphStore``'s
``vertices`` and ``edges`` tables. The walk is a ``UNION ALL`` of
fixed-length branches, one per hop count and edge orientation of the plan
(a ``both`` hop is an out and an in branch), each a join of ``edges``
along the walk with ``vertices`` at the nodes; every alias carries
``label = ?`` bound to the label the retriever computed, and the walk's
vertices are pairwise distinct (§6.6). Each node's constraints, the
caller's vertex filter and the edge rule are evaluated by the functions the
statement calls (``kg_node``, ``kg_edge``, ``kg_key``), which the store
registers for this statement only: they run the domain's own predicates
(``node_holds``), so the embedded graph and the reference executor share
one definition. No identifier or JSON path is built from a plan value;
every value is a bound parameter, and the text holds no quote.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Sequence
from typing import Any

from unified_kg_rag.adapters.graph_query.checks import check_sql, placeholders
from unified_kg_rag.domain.models.graph_query import (
    GraphQueryRequest,
    GroupRow,
    PathEdge,
    PathRow,
    ResolvedEdge,
)
from unified_kg_rag.domain.retrieval.graph_query import (
    EdgePredicate,
    VertexPredicate,
    node_holds,
    value_kind,
    vertex_rank,
)
from unified_kg_rag.shared.utils.common import compute_hash

# An application-defined function: (argument count, implementation).
SQLFunction = tuple[int, Callable[..., Any]]

# Branches a statement may union at most (a plan within the grammar's caps
# needs at most 30: four "both" hops).
MAX_BRANCHES = 64


def _orientations(edge: ResolvedEdge) -> list[tuple[bool, ...]]:
    """Every sequence of step orientations (``True``: source to target) one
    edge pattern may take."""
    forward = {"out": (True,), "in": (False,), "both": (True, False)}[edge.direction]
    found: list[tuple[bool, ...]] = []
    for length in range(edge.min_hops, edge.max_hops + 1):
        found.extend(itertools.product(forward, repeat=length))
    return found


class SQLiteQuery:
    """The statement of one request over ``label``, the functions it calls
    and the decoding of its rows."""

    def __init__(
        self,
        request: GraphQueryRequest,
        label: str,
        *,
        vertex_ok: VertexPredicate | None,
        edge_ok: EdgePredicate | None,
    ) -> None:
        self.request = request
        self.label = label
        self._vertex_ok = vertex_ok
        self._edge_ok = edge_ok
        plan = request.plan
        self.max_steps = sum(edge.max_hops for edge in plan.edges)
        self.branches = list(
            itertools.product(*(_orientations(edge) for edge in plan.edges))
        )
        # Answers per (node, vertex id) and per edge id, within this
        # statement: a vertex is joined once per walk that reaches it.
        self._node_memo: dict[tuple[int, str], int] = {}
        self._edge_memo: dict[str, int] = {}
        self.sql, self.params = self._statement()
        check_sql(self.sql, self.params, label)

    @property
    def digest(self) -> str:
        return compute_hash(self.sql + json.dumps(self.params, default=str), length=16)

    # --- the functions the statement calls ---------------------------------

    def functions(self) -> dict[str, SQLFunction]:
        functions: dict[str, SQLFunction] = {"kg_node": (3, self._kg_node)}
        if self._edge_ok is not None:
            functions["kg_edge"] = (2, self._kg_edge)
        if self.request.group_key is not None:
            functions["kg_key"] = (1, self._kg_key)
            functions["kg_sortable"] = (1, _sortable)
        if self.request.operation == "tuples":
            functions["kg_rank"] = (1, _rank)
        return functions

    def _kg_node(self, index: int, vertex_id: str, props: str) -> int:
        key = (int(index), str(vertex_id))
        found = self._node_memo.get(key)
        if found is None:
            request = self.request
            found = int(
                node_holds(
                    request.plan.nodes[key[0]],
                    key[1],
                    json.loads(props),
                    linked=request.linked,
                    allowed_ids=request.allowed_ids,
                    vertex_ok=self._vertex_ok,
                )
            )
            self._node_memo[key] = found
        return found

    def _kg_edge(self, edge_id: str, props: str) -> int:
        found = self._edge_memo.get(edge_id)
        if found is None:
            assert self._edge_ok is not None
            found = int(self._edge_ok(json.loads(props)))
            self._edge_memo[edge_id] = found
        return found

    def _kg_key(self, props: str) -> str | None:
        """The group key of a vertex: its first scalar value of the key, as
        JSON (so a boolean stays one), or NULL."""
        key = self.request.group_key
        assert key is not None
        for value in json.loads(props).get(key) or ():
            if value_kind(value) is not None:
                return json.dumps(value)
        return None

    # --- the statement -----------------------------------------------------------

    def _columns(self) -> list[str]:
        nodes = [f"n{i}" for i in range(len(self.request.plan.nodes))]
        if self.request.operation != "paths":
            return nodes
        steps = range(self.max_steps)
        return (
            nodes
            + [f"p{i}" for i in range(self.max_steps + 1)]
            + [f"x{i}" for i in steps]
            + [f"t{i}" for i in steps]
            + [f"s{i}" for i in steps]
            + ["k"]
        )

    def _branch(
        self, orientations: tuple[tuple[bool, ...], ...]
    ) -> tuple[str, list[Any]]:
        plan, request, label = self.request.plan, self.request, self.label
        joins: list[str] = []
        join_params: list[Any] = []
        where = ["b0.label = ?"]
        where_params: list[Any] = [label]
        walk = ["b0.id"]
        edges: list[str] = []

        def node_terms(index: int, alias: str) -> None:
            node = plan.nodes[index]
            linked = request.linked.get(node.var) if node.mention is not None else None
            if linked is not None:
                where.append(
                    f"{alias}.id IN ({placeholders(linked)})" if linked else "0"
                )
                where_params.extend(linked)
            where.append(f"kg_node(?, {alias}.id, {alias}.props) = 1")
            where_params.append(index)

        node_terms(0, "b0")
        step = 0
        for index, (edge, steps) in enumerate(
            zip(plan.edges, orientations, strict=True)
        ):
            for forward in steps:
                alias = f"e{step}"
                near, far = (
                    ("source_id", "target_id")
                    if forward
                    else ("target_id", "source_id")
                )
                condition = f"{alias}.label = ? AND {alias}.{near} = {walk[-1]}"
                join_params.append(label)
                if edge.labels:
                    condition += f" AND {alias}.type IN ({placeholders(edge.labels)})"
                    join_params.extend(edge.labels)
                joins.append(f"JOIN edges AS {alias} ON {condition}")
                if self._edge_ok is not None:
                    where.append(f"kg_edge({alias}.id, {alias}.props) = 1")
                walk.append(f"{alias}.{far}")
                edges.append(alias)
                step += 1
            alias = f"b{index + 1}"
            joins.append(
                f"JOIN vertices AS {alias} ON {alias}.label = ? AND {alias}.id = {walk[-1]}"
            )
            join_params.append(label)
            node_terms(index + 1, alias)
        # The walk's vertices are pairwise distinct (Gremlin's simplePath()).
        where.extend(f"{a} <> {b}" for a, b in itertools.combinations(walk, 2))
        boundaries = [f"b{i}.id" for i in range(len(plan.nodes))]
        select = list(boundaries)
        if request.operation == "paths":
            pad = self.max_steps - len(edges)
            select += walk + ["NULL"] * pad
            select += [f"{e}.id" for e in edges] + ["NULL"] * pad
            select += [f"{e}.type" for e in edges] + ["NULL"] * pad
            select += [f"{e}.source_id" for e in edges] + ["NULL"] * pad
            select.append(str(len(edges)))
        sql = (
            f"SELECT {', '.join(select)} FROM vertices AS b0 "  # nosec B608 - compiled aliases and ? marks; values bound, statement checked
            + " ".join(joins)
            + " WHERE "
            + " AND ".join(where)
        )
        return sql, [*join_params, *where_params]

    def _statement(self) -> tuple[str, list[Any]]:
        if len(self.branches) > MAX_BRANCHES:
            raise ValueError(
                f"{len(self.branches)} walk branches; at most {MAX_BRANCHES}"
            )
        parts = [self._branch(orientations) for orientations in self.branches]
        columns = self._columns()
        walk = (
            f"WITH w({', '.join(columns)}) AS ("
            + " UNION ALL ".join(sql for sql, _ in parts)
            + ") "
        )
        params: list[Any] = [p for _, branch in parts for p in branch]
        request = self.request
        index = {var: i for i, var in enumerate(request.plan.vars)}
        picked = ", ".join(f"n{index[v]}" for v in request.vars)
        limit = max(1, request.limit) + 1
        if request.operation == "count":
            return (
                f"{walk}SELECT count(*) FROM (SELECT DISTINCT {picked} FROM w)",  # nosec B608 - compiled aliases and ? marks; values bound, statement checked
                params,
            )
        if request.operation == "tuples":
            # The store's cut keeps the first by the first node's rank, then
            # the ids (domain.retrieval.graph_query's order).
            ids = ", ".join(f"w.n{index[v]}" for v in request.vars)
            first = f"n{index[request.vars[0]]}"
            return (
                f"{walk}SELECT DISTINCT {ids}, kg_rank(v.props) AS r FROM w "  # nosec B608 - compiled aliases and ? marks; values bound, statement checked
                f"JOIN vertices AS v ON v.label = ? AND v.id = w.{first} "
                f"ORDER BY r DESC, {ids} LIMIT ?",
                [*params, self.label, limit],
            )
        if request.operation == "paths":
            cols = [c for c in columns if c[0] in "pxts"]
            return (
                f"{walk}SELECT DISTINCT {', '.join(cols)}, k FROM w "  # nosec B608 - compiled aliases and ? marks; values bound, statement checked
                f"ORDER BY k, {', '.join(cols)} LIMIT ?",
                [*params, limit],
            )
        assert request.group_var is not None
        group = f"n{index[request.group_var]}"
        counted = ", ".join(f"w.n{index[v]}" for v in request.vars)
        if request.group_key is None:
            # nosemgrep: python.django.security.injection.tainted-sql-string.tainted-sql-string - compiled aliases only
            inner = f"SELECT DISTINCT w.{group} AS g, {counted} FROM w"  # nosec B608 - compiled aliases and ? marks; values bound, statement checked
            inner_params: list[Any] = []
        else:
            inner = (
                f"SELECT DISTINCT kg_key(v.props) AS g, {counted} FROM w "  # nosec B608 - compiled aliases and ? marks; values bound, statement checked
                f"JOIN vertices AS v ON v.label = ? AND v.id = w.{group}"
            )
            inner_params = [self.label]
        direction = "DESC" if request.order == "desc" else "ASC"
        key = "q.g" if request.group_key is None else "kg_sortable(q.g)"
        return (
            f"{walk}SELECT q.g, count(*) FROM ({inner}) AS q WHERE q.g IS NOT NULL "  # nosec B608 - compiled aliases and ? marks; values bound, statement checked
            f"GROUP BY q.g ORDER BY count(*) {direction}, {key} LIMIT ?",
            [*params, *inner_params, limit],
        )

    # --- decoding -------------------------------------------------------------------

    def decode(self, rows: Sequence[Sequence[Any]]) -> dict[str, Any]:
        request = self.request
        keep = max(1, request.limit)
        if request.operation == "count":
            return {"count": int(rows[0][0]) if rows else 0}
        if request.operation == "group":
            groups = []
            for key, count in rows:
                value = key if request.group_key is None else json.loads(key)
                groups.append(GroupRow(key=value, count=int(count)))
            return {"groups": tuple(groups[:keep]), "truncated": len(groups) > keep}
        if request.operation == "paths":
            paths = [self._path(row) for row in rows]
            return {"paths": tuple(paths[:keep]), "truncated": len(paths) > keep}
        bindings = [
            {var: str(value) for var, value in zip(request.vars, row, strict=False)}
            for row in rows
        ]
        return {"rows": tuple(bindings[:keep]), "truncated": len(bindings) > keep}

    def _path(self, row: Sequence[Any]) -> PathRow:
        n = self.max_steps
        vertices = [v for v in row[: n + 1] if v is not None]
        ids, types, sources = (
            row[n + 1 : 2 * n + 1],
            row[2 * n + 1 : 3 * n + 1],
            row[3 * n + 1 : 4 * n + 1],
        )
        edges = []
        for step in range(len(vertices) - 1):
            here, there = vertices[step], vertices[step + 1]
            source = str(sources[step])
            edges.append(
                PathEdge(
                    id=str(ids[step]),
                    type=str(types[step]),
                    source_id=source,
                    target_id=there if source == here else here,
                )
            )
        return PathRow(vertex_ids=tuple(str(v) for v in vertices), edges=tuple(edges))


def _rank(props: str) -> float:
    """A vertex's rank as the domain orders rows by it (``vertex_rank``)."""
    return vertex_rank(json.loads(props))


def _sortable(key: str | None) -> Any:
    """A group key (JSON) as SQLite orders it like ``scalar_sort_key``:
    numbers (booleans as 0/1) before strings."""
    return None if key is None else json.loads(key)
