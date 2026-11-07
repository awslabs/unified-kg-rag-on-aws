# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured graph queries: the plan a model writes, the profile it is
validated against, and the store request and result (docs/design.md §6.6).

A :class:`GraphQueryPlan` is the only thing the ``structured`` strategy's
planner model writes: one linear pattern of node and edge patterns and one
return clause, closed (``extra="forbid"``, strict scalar types, capped
lengths). It names entity types, relationship types, property names and
literal values, never a vertex label, suffix, store, index or id: the
compilers add the scope. :func:`domain.retrieval.graph_query.validate_plan`
checks it against a :class:`GraphProfile` and resolves its names to the
stored spellings (:class:`ValidatedPlan`); a :class:`GraphQueryRequest`
asks a ``GraphQueryPort`` one of four store operations over it.

Pure: no store, model or framework import beyond ``shared``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    AllowInfNan,
    BaseModel,
    ConfigDict,
    Field,
    Strict,
    StringConstraints,
    model_validator,
)

# The grammar's caps (the validation adds max_hops from the config).
MAX_NODES = 4
MAX_CONDITIONS = 6
MAX_IN_VALUES = 50
MAX_STRING_CHARS = 256
MAX_TYPES = 8
MAX_HOPS = 4
MAX_PROPERTIES = 10
MAX_ENTITIES = 10
MAX_KEYWORDS = 5
MAX_LIMIT = 200

Var = Annotated[str, Strict(), StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,15}$")]
Name = Annotated[
    str,
    Strict(),
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_STRING_CHARS),
]
PropertyName = Annotated[
    str, Strict(), StringConstraints(strip_whitespace=True, min_length=1, max_length=80)
]
Text = Annotated[str, Strict(), StringConstraints(max_length=MAX_STRING_CHARS)]
Number = Annotated[float, Strict(), AllowInfNan(False)]
Integer = Annotated[int, Strict()]
Flag = Annotated[bool, Strict()]
# A literal value: no coercion ("1" stays a string, true stays a boolean).
Scalar = Flag | Integer | Number | Text

Operator = Literal["eq", "neq", "in", "gt", "gte", "lt", "lte", "contains", "exists"]
Direction = Literal["out", "in", "both"]
ReturnKind = Literal[
    "rows", "count", "count_distinct", "group_count", "aggregate", "exists", "path"
]
AggregateFunction = Literal["min", "max", "sum", "avg"]
Order = Literal["desc", "asc"]
# Where a property is read: the vertex itself, or its seed properties (one
# JSON string per vertex, decoded in code).
PropertySource = Literal["vertex", "seed"]
ValueKind = Literal["string", "number", "boolean"]


class _Closed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Condition(_Closed):
    """One condition on a node's property (``prop`` a name the profile
    lists). ``in`` takes a list, ``exists`` ``true``/``false`` (default
    true), ``contains`` a string (a case-sensitive substring), the others
    one value."""

    prop: PropertyName
    op: Operator
    value: Scalar | list[Scalar] | None = None

    @model_validator(mode="after")
    def _value_fits_operator(self) -> Condition:
        value = self.value
        if self.op == "in":
            if not isinstance(value, list) or not 1 <= len(value) <= MAX_IN_VALUES:
                raise ValueError(f"'in' takes a list of 1 to {MAX_IN_VALUES} values")
        elif self.op == "exists":
            if value is not None and not isinstance(value, bool):
                raise ValueError("'exists' takes true, false or no value")
        elif isinstance(value, list) or value is None:
            raise ValueError(f"'{self.op}' takes one value")
        elif self.op == "contains" and not isinstance(value, str):
            raise ValueError("'contains' takes a string")
        return self


class NodePattern(_Closed):
    """One node of the pattern: a variable, entity types (any when empty), a
    name the question mentions (linked to stored entities by code) and
    conditions (all must hold)."""

    var: Var
    types: list[Name] = Field(default_factory=list, max_length=MAX_TYPES)
    mention: Name | None = None
    where: list[Condition] = Field(default_factory=list, max_length=MAX_CONDITIONS)


class EdgePattern(_Closed):
    """The edge between two consecutive nodes: relationship types (any when
    empty), the direction from the earlier node to the later one, and the
    number of hops (``[min, max]``, a path through other entities when
    above 1)."""

    types: list[Name] = Field(default_factory=list, max_length=MAX_TYPES)
    direction: Direction = "out"
    hops: tuple[Integer, Integer] = (1, 1)

    @model_validator(mode="after")
    def _hops_in_range(self) -> EdgePattern:
        low, high = self.hops
        if not 1 <= low <= high <= MAX_HOPS:
            raise ValueError(
                f"hops must be [min, max] with 1 <= min <= max <= {MAX_HOPS}"
            )
        return self


class GroupKey(_Closed):
    """What ``group_count`` groups by: a node's vertex (no ``prop``) or one
    of its properties."""

    var: Var
    prop: PropertyName | None = None


class Aggregate(_Closed):
    """``aggregate``: a function over a numeric property of a node's
    distinct vertices."""

    fn: AggregateFunction
    var: Var
    prop: PropertyName


class ReturnClause(_Closed):
    """What the query returns.

    - ``rows``: the distinct vertices of ``vars`` (default: every node).
    - ``count``: the number of distinct matches; ``count_distinct`` of
      distinct vertices of ``vars[0]``.
    - ``group_count``: matches per ``group_by`` key (with one var in
      ``vars``: its distinct vertices per key), ``order``ed, ``limit`` of
      them.
    - ``aggregate``: ``min``/``max``/``sum``/``avg`` of a property.
    - ``exists``: whether anything matches.
    - ``path``: the paths between the first and the last node.
    """

    kind: ReturnKind
    vars: list[Var] = Field(default_factory=list, max_length=MAX_NODES)
    group_by: GroupKey | None = None
    aggregate: Aggregate | None = None
    order: Order = "desc"
    limit: Annotated[int, Strict(), Field(ge=1, le=MAX_LIMIT)] = 20
    properties: list[PropertyName] = Field(
        default_factory=list, max_length=MAX_PROPERTIES
    )


class GraphQueryPlan(_Closed):
    """The planner's whole answer (the grammar of docs/design.md §6.6).

    ``entities`` and ``keywords`` are the names and themes the question
    holds, as written: the fallback strategy reads them in place of a query
    extraction. ``unanswerable`` says the graph cannot answer the question.
    """

    nodes: list[NodePattern] = Field(default_factory=list, max_length=MAX_NODES)
    edges: list[EdgePattern] = Field(default_factory=list, max_length=MAX_NODES - 1)
    returns: ReturnClause | None = None
    unanswerable: Flag = False
    entities: list[Name] = Field(default_factory=list, max_length=MAX_ENTITIES)
    keywords: list[Name] = Field(default_factory=list, max_length=MAX_KEYWORDS)


# --- the profile the plan is validated against ---------------------------------


class TypeProfile(_Closed):
    """An entity type or a relationship type: the name the planner sees,
    its ``schema_key``, every stored spelling of it, how many items of the
    profile have it and whether the extraction schema declares it."""

    label: str
    key: str
    spellings: tuple[str, ...]
    count: int = 0
    declared: bool = False
    description: str = ""


class PatternProfile(_Closed):
    """An observed ``(source type, relationship type, target type)`` triple
    (by type key), with how often the profile saw it."""

    source: str
    edge: str
    target: str
    count: int


class PropertyProfile(_Closed):
    """A queryable property: the name the plan uses (``rank``,
    ``attr.role``, ``seed.country``), the stored key, where it is read, the
    value kinds and whether a vertex held several values, with up to a few
    sampled values of a low-cardinality string property."""

    name: str
    key: str
    source: PropertySource
    kinds: tuple[ValueKind, ...]
    multi: bool = False
    count: int = 0
    examples: tuple[Scalar, ...] = ()


class GraphProfile(_Closed):
    """What a graph label holds, as far as a plan may name it: types,
    relationship types, observed patterns and properties. ``sampled``: built
    from a sample of ``sample_size`` vertices, not every vertex."""

    entity_types: tuple[TypeProfile, ...] = ()
    edge_types: tuple[TypeProfile, ...] = ()
    patterns: tuple[PatternProfile, ...] = ()
    properties: tuple[PropertyProfile, ...] = ()
    sampled: bool = False
    sample_size: int = 0


# --- the validated plan ----------------------------------------------------------


class PropertyRef(_Closed):
    """A plan property resolved: its plan name, stored key and source."""

    name: str
    key: str
    source: PropertySource
    kinds: tuple[ValueKind, ...] = ()
    multi: bool = False


class ResolvedCondition(_Closed):
    prop: PropertyRef
    op: Operator
    value: Scalar | tuple[Scalar, ...] | None = None


class ResolvedNode(_Closed):
    """A node with its types as stored spellings (``()``: any type)."""

    var: str
    types: tuple[str, ...] = ()
    type_labels: tuple[str, ...] = ()
    mention: str | None = None
    conditions: tuple[ResolvedCondition, ...] = ()

    @property
    def vertex_conditions(self) -> tuple[ResolvedCondition, ...]:
        return tuple(c for c in self.conditions if c.prop.source == "vertex")

    @property
    def seed_conditions(self) -> tuple[ResolvedCondition, ...]:
        return tuple(c for c in self.conditions if c.prop.source == "seed")


class ResolvedEdge(_Closed):
    """An edge with its types as stored edge labels (``()``: any)."""

    labels: tuple[str, ...] = ()
    type_labels: tuple[str, ...] = ()
    direction: Direction = "out"
    min_hops: int = 1
    max_hops: int = 1


class ResolvedReturn(_Closed):
    kind: ReturnKind
    vars: tuple[str, ...]
    group_var: str | None = None
    group_prop: PropertyRef | None = None
    aggregate_fn: AggregateFunction | None = None
    aggregate_var: str | None = None
    aggregate_prop: PropertyRef | None = None
    order: Order = "desc"
    limit: int = 20
    properties: tuple[PropertyRef, ...] = ()


class ValidatedPlan(_Closed):
    """A plan whose every name the profile lists, resolved to what the
    store holds; ``plan`` is the model's plan as it was validated."""

    nodes: tuple[ResolvedNode, ...]
    edges: tuple[ResolvedEdge, ...]
    returns: ResolvedReturn
    plan: GraphQueryPlan

    @property
    def vars(self) -> tuple[str, ...]:
        return tuple(node.var for node in self.nodes)

    def node(self, var: str) -> ResolvedNode:
        return next(node for node in self.nodes if node.var == var)

    @property
    def reads_seed_properties(self) -> bool:
        """Whether a seed property is a condition, the group key or the
        aggregated value: then the store cannot evaluate the plan."""
        r = self.returns
        refs = [c.prop for n in self.nodes for c in n.conditions]
        refs += [p for p in (r.group_prop, r.aggregate_prop) if p is not None]
        return any(ref.source == "seed" for ref in refs)


# --- the store request and result -----------------------------------------------

# What a store computes for a plan (domain.retrieval.graph_query.plan_operation):
# count: the distinct bindings of ``vars``, exact; group: (key, distinct
# bindings of ``vars``) per key; tuples: the distinct bindings of ``vars``;
# paths: the walks. Every list is at most ``limit + 1`` long (one more than
# kept: the result is then ``truncated``), in no particular order.
GraphOperation = Literal["count", "group", "tuples", "paths"]


class GraphQueryRequest(_Closed):
    """One store operation over a validated plan.

    ``filters`` are the caller's filters the graph applies to a bound
    vertex (``source_kinds`` also to every edge of the walk), without
    ``id``: ``allowed_ids`` is the caller's ``id`` filter. ``linked`` holds
    each mention node's entity ids. ``hydrate`` names the vertex properties
    the result carries for every vertex it returns. No field names a label:
    the store derives it from ``suffix``.
    """

    plan: ValidatedPlan
    suffix: str | None = None
    operation: GraphOperation
    vars: tuple[str, ...] = ()
    group_var: str | None = None
    group_key: str | None = None
    limit: int = 1
    filters: dict[str, Any] | None = None
    allowed_ids: tuple[str, ...] | None = None
    linked: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    hydrate: tuple[str, ...] = ()
    timeout_seconds: float = 10.0


class GroupRow(_Closed):
    key: Scalar
    count: int


class PathEdge(_Closed):
    id: str
    type: str
    source_id: str
    target_id: str


class PathRow(_Closed):
    """A walk: its vertex ids in pattern order and the edge of each step."""

    vertex_ids: tuple[str, ...]
    edges: tuple[PathEdge, ...] = ()


class GraphQueryResult(_Closed):
    """A store operation's answer, with the stored properties
    (``hydrate``) of every vertex it names (``vertices``: id -> value map,
    every value a list)."""

    operation: GraphOperation
    count: int | None = None
    groups: tuple[GroupRow, ...] = ()
    rows: tuple[dict[str, str], ...] = ()
    paths: tuple[PathRow, ...] = ()
    truncated: bool = False
    vertices: dict[str, dict[str, list[Any]]] = Field(default_factory=dict)
    backend: str = ""
    compiled_digest: str = ""
    elapsed_ms: float = 0.0


class QueryAnswer(_Closed):
    """What a plan answers, computed from a store result or from scanned
    bindings (``domain.retrieval.graph_query``). ``exact``: computed by the
    store over every match (a count) or over every candidate in code; a
    ``truncated`` answer is a lower bound (a count) or a subset."""

    kind: ReturnKind
    count: int | None = None
    exists: bool | None = None
    value: float | None = None
    value_count: int = 0
    groups: tuple[GroupRow, ...] = ()
    rows: tuple[dict[str, str], ...] = ()
    paths: tuple[PathRow, ...] = ()
    truncated: bool = False

    @property
    def empty(self) -> bool:
        """No row, group, path or value: a count (0 too) and an ``exists``
        answer are never empty."""
        if self.kind in ("count", "count_distinct", "exists"):
            return False
        if self.kind == "aggregate":
            return self.value is None
        return not (self.rows or self.groups or self.paths)


# --- an in-memory property graph (the reference executor's input) ---------------


class GraphEdgeRecord(_Closed):
    """One stored edge: its relationship id, label (type), endpoints and
    properties (as the store holds them: lists are JSON strings)."""

    id: str
    type: str
    source_id: str
    target_id: str
    props: dict[str, Any] = Field(default_factory=dict)


class PropertyGraph(BaseModel):
    """One label's vertices (id -> value map, every value a list) and edges,
    for the reference executor (``domain.retrieval.graph_query``)."""

    vertices: dict[str, dict[str, list[Any]]] = Field(default_factory=dict)
    edges: list[GraphEdgeRecord] = Field(default_factory=list)
