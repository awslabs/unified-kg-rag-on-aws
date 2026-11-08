# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A validated plan compiled to Gremlin bytecode (Neptune, Gremlin Server).

:class:`GremlinQuery` builds the traversal of one ``GraphQueryRequest``
from the connection's source and decodes its rows. The label is a
constructor argument the retriever computes from the suffix; the plan and
the request cannot name one. Every vertex step is followed by
``hasLabel(label)``; a variable-length hop is a ``union`` of fixed-length
chains, so each of its steps is label-checked too; values are bytecode
arguments. The match semantics are §6.6's: ``simplePath()`` over the whole
walk keeps its vertices distinct (at the end of the walk and at the end of
each union branch, since Neptune drops one that follows a ``union``), bindings are projected to the vertices'
``id`` properties and de-duplicated, so a vertex stored twice during a
vertex-id switch counts once.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from gremlin_python.process.graph_traversal import (
    GraphTraversal,
    GraphTraversalSource,
    __,
)
from gremlin_python.process.traversal import Column, Order, P, Scope, TextP, Traversal

from unified_kg_rag.adapters.graph_query.checks import check_gremlin
from unified_kg_rag.domain.models.graph_query import (
    GraphQueryRequest,
    GroupRow,
    PathEdge,
    PathRow,
    ResolvedCondition,
    ResolvedEdge,
    ResolvedNode,
)
from unified_kg_rag.domain.retrieval.filters import KindsRule, Range, Term
from unified_kg_rag.domain.retrieval.graph_query import value_kind
from unified_kg_rag.shared.utils.common import compute_hash

# The property every vertex and edge is looked up by.
ID = "id"
# Vertex properties the profile never reads (long, internal or lineage).
PROFILE_SKIPPED_KEYS = (
    "id",
    "name",
    "description",
    "text_unit_ids",
    "community_ids",
    "name_key",
    "build_tag",
    "attr_filters",
    "attr_index",
)
# Ids hydrated per request.
HYDRATE_BATCH = 200

Filters = Callable[[GraphTraversal], GraphTraversal]


def gremlin_vertex_filters(
    traversal: GraphTraversal, clauses: Sequence[Term | Range | KindsRule]
) -> GraphTraversal:
    """``traversal`` with a vertex's filter clauses (``vertex_clauses``) as
    Gremlin steps: a strict ``has`` per value (``within`` for several) or
    range bound; a ``source_kinds`` rule a ``where`` (``gremlin_kinds``)."""
    for clause in clauses:
        if isinstance(clause, KindsRule):
            traversal = traversal.where(gremlin_kinds(clause, clause.kinds))
        elif isinstance(clause, Range):
            for op, bound in clause.bounds:
                traversal = traversal.has(clause.field, getattr(P, op)(bound))
        elif len(clause.values) == 1 and not clause.listed:
            traversal = traversal.has(clause.field, clause.values[0])
        else:
            traversal = traversal.has(clause.field, P.within(list(clause.values)))
    return traversal


def gremlin_kinds(rule: KindsRule, stored: Sequence[str]) -> GraphTraversal:
    """The anonymous traversal an element passes under a ``source_kinds``
    rule: its property holds one of ``stored`` (a vertex's kinds, or the
    JSON encodings of an edge's: ``KindsRule.encodings``), or, when the rule
    names ``document``, it has no ``source_kinds`` at all."""
    values = list(stored)
    held = __.has(rule.field, values[0] if len(values) == 1 else P.within(values))
    if not rule.unstated_passes:
        return held
    return __.or_(held, __.hasNot(rule.field))


def _flipped(direction: str) -> str:
    return {"out": "in", "in": "out"}.get(direction, direction)


class GremlinQuery:
    """The traversal of one request over ``label``.

    ``apply_filters`` applies the caller's graph filters to a vertex step
    (the retriever's own graph-expansion filter code), ``edge_rule`` the
    caller's ``source_kinds`` rule to every edge (``kinds_rule``).
    """

    def __init__(
        self,
        request: GraphQueryRequest,
        label: str,
        *,
        apply_filters: Filters,
        edge_rule: KindsRule | None,
        timeout_ms: int,
    ) -> None:
        self.request = request
        self.label = label
        self._apply_filters = apply_filters
        self._edge_rule = edge_rule
        self.timeout_ms = max(1, int(timeout_ms))
        nodes = list(request.plan.nodes)
        edges = list(request.plan.edges)
        # Start from a linked node: a traversal cannot start in the middle,
        # and the last node's ids are as good a start as the first's.
        self.reversed = (
            len(nodes) > 1 and not self._linked(nodes[0]) and self._linked(nodes[-1])
        )
        if self.reversed:
            nodes.reverse()
            edges = [
                edge.model_copy(update={"direction": _flipped(edge.direction)})
                for edge in reversed(edges)
            ]
        self.nodes = nodes
        self.edges = edges

    def _linked(self, node: ResolvedNode) -> bool:
        return node.mention is not None

    # --- the traversal -----------------------------------------------------

    def build(self, g: GraphTraversalSource) -> Traversal:
        """The checked traversal (``check_gremlin``)."""
        traversal = self._operation(self._pattern(g))
        check_gremlin(traversal, self.label)
        return traversal

    def digest(self, g: GraphTraversalSource) -> str:
        """A hash of the bytecode, for logs (never the text)."""
        return compute_hash(str(self.build(g).bytecode), length=16)

    def _pattern(self, g: GraphTraversalSource) -> GraphTraversal:
        traversal = (
            g.with_("evaluationTimeout", self.timeout_ms).V().hasLabel(self.label)
        )
        traversal = self._constrain(traversal, self.nodes[0])
        for edge, node in zip(self.edges, self.nodes[1:], strict=True):
            traversal = self._segment(traversal, edge)
            traversal = self._constrain(traversal, node)
        return traversal.simplePath()

    def _constrain(
        self, traversal: GraphTraversal, node: ResolvedNode
    ) -> GraphTraversal:
        request = self.request
        if node.mention is not None:
            traversal = traversal.has(
                ID, P.within(list(request.linked.get(node.var, ())))
            )
        if request.allowed_ids is not None:
            traversal = traversal.has(ID, P.within(list(request.allowed_ids)))
        if node.types:
            traversal = traversal.has("type", P.within(list(node.types)))
        for condition in node.vertex_conditions:
            traversal = self._condition(traversal, condition)
        return self._apply_filters(traversal).as_(node.var)

    @staticmethod
    def _condition(
        traversal: GraphTraversal, condition: ResolvedCondition
    ) -> GraphTraversal:
        key, op, value = condition.prop.key, condition.op, condition.value
        if op == "exists":
            return traversal.has(key) if value is not False else traversal.hasNot(key)
        if op == "eq":
            return traversal.has(key, P.eq(value))
        if op == "neq":
            return traversal.has(key, P.neq(value))
        if op == "in":
            assert isinstance(value, tuple)
            return traversal.has(key, P.within(list(value)))
        if op == "contains":
            return traversal.has(key, TextP.containing(value))
        predicate = {"gt": P.gt, "gte": P.gte, "lt": P.lt, "lte": P.lte}[op]
        return traversal.has(key, predicate(value))

    def _step(self, start: Any, edge: ResolvedEdge) -> GraphTraversal:
        labels = list(edge.labels)
        direction = edge.direction
        incident = (
            start.outE(*labels)
            if direction == "out"
            else start.inE(*labels) if direction == "in" else start.bothE(*labels)
        )
        if self._edge_rule is not None:
            rule = self._edge_rule
            incident = incident.where(gremlin_kinds(rule, rule.encodings()))
        reached = (
            incident.inV()
            if direction == "out"
            else incident.outV() if direction == "in" else incident.otherV()
        )
        return reached.hasLabel(self.label)

    def _segment(self, traversal: GraphTraversal, edge: ResolvedEdge) -> GraphTraversal:
        if edge.min_hops == edge.max_hops:
            # One length is one chain, walked in place: Neptune rejects a
            # union() of a single simplePath() branch ("Exception processing
            # a script"), and the trailing simplePath() covers a plain chain.
            for _ in range(edge.min_hops):
                traversal = self._step(traversal, edge)
            return traversal
        chains = []
        for length in range(edge.min_hops, edge.max_hops + 1):
            chain = self._step(__, edge)
            for _ in range(length - 1):
                chain = self._step(chain, edge)
            # Neptune ignores a simplePath() after a union() of several
            # branches (Gremlin Server honours it): each branch checks the
            # walk so far itself.
            chains.append(chain.simplePath())
        return traversal.union(*chains)

    def _ids(self, traversal: GraphTraversal, keys: Sequence[str]) -> GraphTraversal:
        projected = traversal.project(*keys)
        for var in keys:
            projected = projected.by(__.select(var).values(ID))
        return projected

    def _operation(self, traversal: GraphTraversal) -> GraphTraversal:
        request = self.request
        limit = max(1, request.limit) + 1
        if request.operation == "count":
            return self._ids(traversal, request.vars).dedup().count()
        if request.operation == "group":
            var = request.group_var
            assert var is not None
            key = request.group_key
            if key is not None:
                traversal = traversal.where(__.select(var).has(key))
                group_by = __.select(var).values(key)
            else:
                group_by = __.select(var).values(ID)
            projected = traversal.project("_g", *request.vars).by(group_by)
            for counted in request.vars:
                projected = projected.by(__.select(counted).values(ID))
            # Cut in the domain's order: by count, then by key.
            direction = Order.desc if request.order == "desc" else Order.asc
            return (
                projected.dedup()
                .groupCount()
                .by(__.select("_g"))
                .order(Scope.local)
                .by(Column.values, direction)
                .by(Column.keys, Order.asc)
                .unfold()
                .limit(limit)
            )
        if request.operation == "paths":
            edge_projection = (
                __.project("id", "type", "source")
                .by(__.coalesce(__.values(ID), __.id_()))
                .by(__.label())
                .by(__.outV().hasLabel(self.label).values(ID))
            )
            # Shortest first, as the domain orders them.
            return (
                traversal.path()
                .by(__.values(ID))
                .by(edge_projection)
                .dedup()
                .order()
                .by(__.count(Scope.local), Order.asc)
                .limit(limit)
            )
        # By the first node's rank (desc), then the ids: the domain's order.
        keys = list(request.vars)
        projected = traversal.project(*keys, "_r")
        for var in keys:
            projected = projected.by(__.select(var).values(ID))
        projected = projected.by(
            __.select(keys[0]).coalesce(__.values("rank"), __.constant(0))
        )
        ordered = projected.dedup().order().by(__.select("_r"), Order.desc)
        for var in keys:
            ordered = ordered.by(__.select(var), Order.asc)
        return ordered.limit(limit)

    # --- decoding ------------------------------------------------------------------

    def decode(self, rows: list[Any]) -> dict[str, Any]:
        """``rows`` as ``GraphQueryResult`` fields (count, groups, rows or
        paths, truncated)."""
        request = self.request
        keep = max(1, request.limit)
        if request.operation == "count":
            return {"count": int(rows[0]) if rows else 0}
        if request.operation == "group":
            groups = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for key, count in row.items():
                    if value_kind(key) is not None:
                        groups.append(GroupRow(key=key, count=int(count)))
            return {"groups": tuple(groups[:keep]), "truncated": len(groups) > keep}
        if request.operation == "paths":
            paths = [p for p in (self._path(row) for row in rows) if p is not None]
            return {"paths": tuple(paths[:keep]), "truncated": len(paths) > keep}
        bindings = [
            {var: str(row[var]) for var in request.vars}
            for row in rows
            if isinstance(row, dict) and all(var in row for var in request.vars)
        ]
        return {"rows": tuple(bindings[:keep]), "truncated": len(bindings) > keep}

    def _path(self, row: Any) -> PathRow | None:
        objects = list(getattr(row, "objects", row) or [])
        vertices = [str(v) for v in objects[0::2]]
        projections = objects[1::2]
        if len(vertices) != len(projections) + 1:
            return None
        edges = []
        for index, projection in enumerate(projections):
            if not isinstance(projection, dict):
                return None
            here, there = vertices[index], vertices[index + 1]
            source = str(projection.get("source"))
            edges.append(
                PathEdge(
                    id=str(projection.get("id")),
                    type=str(projection.get("type")),
                    source_id=source,
                    target_id=there if source == here else here,
                )
            )
        if self.reversed:
            vertices.reverse()
            edges.reverse()
        return PathRow(vertex_ids=tuple(vertices), edges=tuple(edges))


def named_vertex_ids(fields: dict[str, Any], group_by_vertex: bool) -> list[str]:
    """The vertex ids a decoded result names (to hydrate)."""
    ids: dict[str, None] = {}
    for row in fields.get("rows", ()):
        ids.update(dict.fromkeys(row.values()))
    for path in fields.get("paths", ()):
        ids.update(dict.fromkeys(path.vertex_ids))
    if group_by_vertex:
        ids.update(dict.fromkeys(str(g.key) for g in fields.get("groups", ())))
    return list(ids)


def hydrate_traversal(
    g: GraphTraversalSource,
    label: str,
    ids: Sequence[str],
    keys: Sequence[str],
    timeout_ms: int,
) -> Traversal:
    """The value maps (``keys``) of ``label``'s vertices among ``ids``."""
    traversal = (
        g.with_("evaluationTimeout", max(1, int(timeout_ms)))
        .V()
        .hasLabel(label)
        .has(ID, P.within(list(ids)))
        .project("id", "props")
        .by(__.values(ID))
        .by(__.valueMap(*keys))
    )
    check_gremlin(traversal, label)
    return traversal


def profile_traversal(
    g: GraphTraversalSource,
    label: str,
    sample: int,
    edges_per_vertex: int,
    timeout_ms: int,
) -> Traversal:
    """``sample`` of ``label``'s vertices: their scalar properties (but the
    long and internal ones) and the label and target type of at most
    ``edges_per_vertex`` of their out-edges."""
    target_type = __.coalesce(__.inV().hasLabel(label).values("type"), __.constant(""))
    traversal = (
        g.with_("evaluationTimeout", max(1, int(timeout_ms)))
        .V()
        .hasLabel(label)
        .limit(max(1, sample))
        .project("props", "out")
        .by(
            __.properties()
            .hasKey(P.without(list(PROFILE_SKIPPED_KEYS)))
            .group()
            .by(__.key())
            .by(__.value().fold())
        )
        .by(
            __.outE()
            .limit(max(0, edges_per_vertex))
            .project("label", "target")
            .by(__.label())
            .by(target_type)
            .fold()
        )
    )
    check_gremlin(traversal, label)
    return traversal


def decode_profile(
    rows: list[Any],
) -> tuple[list[dict[str, list[Any]]], list[tuple[str, str, str]]]:
    """The profile traversal's rows as value maps and
    ``(source type, edge label, target type)`` triples."""
    vertices: list[dict[str, list[Any]]] = []
    edges: list[tuple[str, str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        props = row.get("props") or {}
        value_map = {
            str(key): list(values) if isinstance(values, list) else [values]
            for key, values in props.items()
        }
        vertices.append(value_map)
        source = str((value_map.get("type") or [""])[0])
        for edge in row.get("out") or ():
            if isinstance(edge, dict):
                edges.append((source, str(edge.get("label")), str(edge.get("target"))))
    return vertices, edges
