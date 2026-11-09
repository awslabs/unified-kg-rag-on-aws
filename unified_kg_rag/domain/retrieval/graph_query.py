# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured graph queries in pure code (docs/design.md §6.6).

- :func:`build_profile` / :func:`merge_declared`: what a label holds, as a
  plan may name it.
- :func:`validate_plan`: a plan checked against a profile and resolved to
  the stored spellings, or an ``InvalidGraphQueryError`` naming the field.
- :func:`describe_plan`: the plan in English, written by code.
- :func:`plan_operation` / :func:`hydrate_keys`: the store operation a plan
  needs and the vertex properties its answer reads.
- :func:`answer_from_result` / :func:`aggregate_bindings`: the answer from
  a store result, or from scanned bindings checked in code.
- :func:`match_walks` / :func:`reference_execute`: the match semantics
  over a ``PropertyGraph`` and the structural benchmark's gold answers
  (the conformance suite's per-operation reference builds on them).

The match semantics are the one definition of §6.6: a walk over distinct
vertices, each edge pattern ``[min, max]`` steps of its types and
direction, each node's constraints on the vertex it binds, results over
distinct bindings (tuples of the bound vertices' ``id`` properties).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from unified_kg_rag.domain.ingestion.base_processor import SEED_PROPERTIES_ATTRIBUTE
from unified_kg_rag.domain.models import Constants
from unified_kg_rag.domain.models.graph_query import (
    GraphEdgeRecord,
    GraphOperation,
    GraphProfile,
    GraphQueryPlan,
    GraphQueryResult,
    GroupRow,
    PathEdge,
    PathRow,
    PatternProfile,
    PropertyGraph,
    PropertyProfile,
    PropertyRef,
    QueryAnswer,
    ResolvedCondition,
    ResolvedEdge,
    ResolvedNode,
    ResolvedReturn,
    Scalar,
    TypeProfile,
    ValidatedPlan,
    ValueKind,
)
from unified_kg_rag.domain.models.schema import ExtractionSchema, schema_key
from unified_kg_rag.domain.retrieval.filters import RANGE_BOUNDS, bound_holds
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.exceptions import InvalidGraphQueryError

logger = get_logger(__name__)

ATTRIBUTE_KEY_PREFIX = f"{Constants.ATTRIBUTE_PREFIX.value}_"
# The vertex property that holds the seed properties, as one JSON string.
SEED_PROPERTIES_KEY = f"{ATTRIBUTE_KEY_PREFIX}{SEED_PROPERTIES_ATTRIBUTE}"
# The vertex property that holds the entity's document filters (JSON).
FILTERS_KEY = f"{ATTRIBUTE_KEY_PREFIX}{Constants.FILTERS.value}"
# Plan-name prefixes of extracted attributes and seed properties.
ATTRIBUTE_NAME_PREFIX = "attr."
SEED_NAME_PREFIX = "seed."

# The core vertex properties a plan may name: kind and whether multi-valued.
CORE_PROPERTIES: dict[str, tuple[ValueKind, bool]] = {
    "name": ("string", False),
    "type": ("string", False),
    "description": ("string", False),
    "rank": ("number", False),
    "confidence": ("number", False),
    "in_schema": ("boolean", False),
    "type_iri": ("string", False),
    "iri": ("string", False),
    "source_kinds": ("string", True),
}
# Never a condition, a key or a shown property: ids, lineage and internals
# (the suffix, the document filters, the seed properties' JSON, extraction
# bookkeeping; an attribute whose name starts with "_" is internal too).
NEVER_QUERYABLE = frozenset(
    {
        "id",
        "name_key",
        "build_tag",
        "text_unit_ids",
        "community_ids",
        FILTERS_KEY,
        SEED_PROPERTIES_KEY,
        f"{ATTRIBUTE_KEY_PREFIX}{Constants.INDEX.value}",
        f"{ATTRIBUTE_KEY_PREFIX}text_unit_weights",
    }
)


def is_queryable_key(key: str) -> bool:
    """Whether a stored vertex key may be named by a plan: a core property
    or an extracted attribute that is not internal."""
    if key in NEVER_QUERYABLE:
        return False
    if key in CORE_PROPERTIES:
        return True
    return key.startswith(ATTRIBUTE_KEY_PREFIX) and not key[
        len(ATTRIBUTE_KEY_PREFIX) :
    ].startswith("_")


# What the store returns of every vertex an answer names: its projection as
# graph expansion builds it, plus the document filters (seed provenance).
PROJECTION_KEYS: tuple[str, ...] = (
    "id",
    "name",
    "type",
    "description",
    "rank",
    "text_unit_ids",
    "iri",
    "source_kinds",
    FILTERS_KEY,
)
# Sampled string values longer than this are not shown as examples.
_EXAMPLE_MAX_CHARS = 60
_NUMERIC_OPS = RANGE_BOUNDS

VertexMap = Mapping[str, Sequence[Any]]
VertexPredicate = Callable[[Any], bool]
EdgePredicate = Callable[[Mapping[str, Any]], bool]


# --- values ---------------------------------------------------------------------


def value_kind(value: Any) -> ValueKind | None:
    """The kind of a stored scalar (``None`` for anything else)."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int | float):
        return None if isinstance(value, float) and math.isnan(value) else "number"
    if isinstance(value, str):
        return "string"
    return None


def _same_kind(a: Any, b: Any) -> bool:
    kind = value_kind(a)
    return kind is not None and kind == value_kind(b)


def scalar_holds(stored: Any, op: str, value: Any) -> bool:
    """Whether one stored value satisfies ``op value``: comparisons within
    one kind (a number never equals a boolean); ``neq`` holds across kinds,
    as Gremlin's does."""
    if op == "eq":
        return _same_kind(stored, value) and bool(stored == value)
    if op == "neq":
        return not (_same_kind(stored, value) and stored == value)
    if op == "in":
        return any(_same_kind(stored, v) and stored == v for v in value)
    if op == "contains":
        return isinstance(stored, str) and isinstance(value, str) and value in stored
    if op in _NUMERIC_OPS:
        if not _same_kind(stored, value) or value_kind(stored) == "boolean":
            return False
        return bound_holds(op, stored, value)
    return False


def condition_holds(values: Sequence[Any], condition: ResolvedCondition) -> bool:
    """Whether a property's stored values satisfy ``condition``: any value
    for an operator, presence for ``exists``; no value fails every
    operator but ``exists: false``."""
    present = [v for v in values if v is not None]
    if condition.op == "exists":
        return bool(present) == (condition.value is not False)
    return any(scalar_holds(v, condition.op, condition.value) for v in present)


def seed_properties(vertex: VertexMap) -> dict[str, Any]:
    """A vertex's seed properties (decoded from their JSON string)."""
    merged: dict[str, Any] = {}
    for raw in vertex.get(SEED_PROPERTIES_KEY) or ():
        if isinstance(raw, dict):
            decoded: Any = raw
        else:
            try:
                decoded = json.loads(raw) if isinstance(raw, str) else None
            except ValueError:
                decoded = None
        if isinstance(decoded, dict):
            merged.update(decoded)
    return merged


def property_values(vertex: VertexMap, prop: PropertyRef) -> list[Any]:
    """The stored values of ``prop`` on ``vertex`` (a list; seed properties
    decoded)."""
    if prop.source == "seed":
        value = seed_properties(vertex).get(prop.key)
        return [] if value is None else [value]
    return [v for v in (vertex.get(prop.key) or ()) if v is not None]


def first_value(vertex: VertexMap, prop: PropertyRef) -> Any:
    values = property_values(vertex, prop)
    return values[0] if values else None


def vertex_rank(vertex: VertexMap | None) -> float:
    for value in (vertex or {}).get("rank") or ():
        if value_kind(value) == "number":
            return float(value)
    return 0.0


# Integral floats up to this magnitude are exact as ints.
_EXACT_INTEGRAL = 2**53


def canonical_scalar(value: Any) -> Any:
    """A stored scalar as one group key: an integral float as its int (5.0
    is 5, as every store's numeric comparison reads them; a JSON or Gremlin
    group of the raw values kept 5 and 5.0 apart), anything else as is."""
    if (
        isinstance(value, float)
        and value.is_integer()
        and abs(value) <= _EXACT_INTEGRAL
    ):
        return int(value)
    return value


def scalar_sort_key(value: Any) -> tuple[int, Any]:
    """One order over group keys of any kind: numbers (booleans as 0/1),
    then strings."""
    if isinstance(value, bool | int | float):
        return (0, float(value))
    return (1, str(value))


# --- the profile ------------------------------------------------------------------


def plan_name(key: str, source: str = "vertex") -> str:
    """The name a plan uses for a stored property key."""
    if source == "seed":
        return f"{SEED_NAME_PREFIX}{key}"
    if key.startswith(ATTRIBUTE_KEY_PREFIX):
        return f"{ATTRIBUTE_NAME_PREFIX}{key[len(ATTRIBUTE_KEY_PREFIX):]}"
    return key


@dataclass
class _PropertyStats:
    count: int = 0
    multi: bool = False
    kinds: set[str] | None = None
    values: set[Any] | None = None
    overflow: bool = False

    def add(self, values: Sequence[Any], max_examples: int) -> None:
        kinds = self.kinds if self.kinds is not None else set()
        seen = self.values if self.values is not None else set()
        scalars = [v for v in values if value_kind(v) is not None]
        if not scalars:
            return
        self.count += 1
        self.multi = self.multi or len(scalars) > 1
        for value in scalars:
            kinds.add(str(value_kind(value)))
            if isinstance(value, str) and not self.overflow:
                if len(value) > _EXAMPLE_MAX_CHARS:
                    self.overflow = True
                    continue
                seen.add(value)
                if len(seen) > max_examples:
                    self.overflow = True
        self.kinds, self.values = kinds, seen


def _decoded_attribute(values: Sequence[Any]) -> bool:
    """Whether an attribute's values are JSON objects or arrays (a dict or a
    list attribute's encoding), not scalars a plan can compare."""
    return any(isinstance(v, str) and v[:1] in ("{", "[") for v in values)


def build_profile(
    vertices: Iterable[VertexMap],
    edges: Iterable[tuple[str, str, str]],
    *,
    sampled: bool,
    max_examples: int = 10,
) -> GraphProfile:
    """The profile of sampled ``vertices`` (value maps) and their
    ``edges`` (``(source type, edge label, target type)``).

    Types and edge labels are grouped by ``schema_key``, each with every
    stored spelling and the most frequent as its label. A property is a core
    one, an extracted attribute (``attr.<key>``) or a seed property
    (``seed.<key>``); a string property with at most ``max_examples``
    distinct short values lists them. The core properties are always
    listed, sampled or not.
    """
    type_spellings: dict[str, Counter[str]] = defaultdict(Counter)
    edge_spellings: dict[str, Counter[str]] = defaultdict(Counter)
    patterns: Counter[tuple[str, str, str]] = Counter()
    stats: dict[tuple[str, str], _PropertyStats] = defaultdict(_PropertyStats)
    size = 0
    for vertex in vertices:
        size += 1
        for kind in vertex.get("type") or ():
            if isinstance(kind, str) and kind:
                type_spellings[schema_key(kind)][kind] += 1
        for key, values in vertex.items():
            if key == SEED_PROPERTIES_KEY:
                for seed_key, value in seed_properties(vertex).items():
                    stats[(seed_key, "seed")].add([value], max_examples)
                continue
            if not isinstance(key, str) or not is_queryable_key(key):
                continue
            values = list(values or ())
            if key.startswith(ATTRIBUTE_KEY_PREFIX) and _decoded_attribute(values):
                continue
            stats[(key, "vertex")].add(values, max_examples)
    for source, label, target in edges:
        if not label:
            continue
        edge_spellings[schema_key(label)][label] += 1
        if source and target:
            patterns[(schema_key(source), schema_key(label), schema_key(target))] += 1

    def types(spellings: dict[str, Counter[str]]) -> tuple[TypeProfile, ...]:
        found = [
            TypeProfile(
                label=counts.most_common(1)[0][0],
                key=key,
                spellings=tuple(sorted(counts)),
                count=sum(counts.values()),
            )
            for key, counts in spellings.items()
        ]
        return tuple(sorted(found, key=lambda t: (-t.count, t.label)))

    properties: list[PropertyProfile] = []
    for key, (kind, multi) in CORE_PROPERTIES.items():
        seen = stats.pop((key, "vertex"), None)
        properties.append(_property(key, "vertex", seen, (kind,), multi, max_examples))
    for (key, source), seen in sorted(
        stats.items(), key=lambda item: (-item[1].count, item[0])
    ):
        properties.append(_property(key, source, seen, (), False, max_examples))
    return GraphProfile(
        entity_types=types(type_spellings),
        edge_types=types(edge_spellings),
        patterns=tuple(
            PatternProfile(source=s, edge=e, target=t, count=n)
            for (s, e, t), n in sorted(
                patterns.items(), key=lambda item: (-item[1], item[0])
            )
        ),
        properties=tuple(properties),
        sampled=sampled,
        sample_size=size,
    )


def _property(
    key: str,
    source: str,
    seen: _PropertyStats | None,
    default_kinds: tuple[ValueKind, ...],
    default_multi: bool,
    max_examples: int,
) -> PropertyProfile:
    kinds: tuple[ValueKind, ...] = default_kinds
    examples: tuple[Scalar, ...] = ()
    if seen is not None and seen.kinds:
        order: tuple[ValueKind, ...] = ("string", "number", "boolean")
        kinds = tuple(k for k in order if k in seen.kinds)
        if (
            key not in ("name", "description")
            and not seen.overflow
            and seen.values
            and max_examples > 0
        ):
            examples = tuple(sorted(seen.values))
    return PropertyProfile(
        name=plan_name(key, source),
        key=key,
        source="seed" if source == "seed" else "vertex",
        kinds=kinds,
        multi=default_multi or bool(seen and seen.multi),
        count=seen.count if seen else 0,
        examples=examples,
    )


def merge_declared(
    profile: GraphProfile, schema: ExtractionSchema | None
) -> GraphProfile:
    """``profile`` with the extraction schema's declared types, relationship
    types and patterns: a declared type the sample missed is still allowed,
    under its declared label and its upper- and lower-case spellings (how
    extraction may have written it)."""
    if schema is None:
        return profile

    def merged(
        found: tuple[TypeProfile, ...], declared: Iterable[tuple[str, str]]
    ) -> tuple[TypeProfile, ...]:
        by_key = {t.key: t for t in found}
        for label, description in declared:
            key = schema_key(label)
            current = by_key.get(key)
            variants = {label, label.upper(), label.lower()}
            if current is None:
                by_key[key] = TypeProfile(
                    label=label,
                    key=key,
                    spellings=tuple(sorted(variants)),
                    declared=True,
                    description=description,
                )
            else:
                by_key[key] = current.model_copy(
                    update={
                        "label": label,
                        "declared": True,
                        "description": description,
                        "spellings": tuple(sorted(set(current.spellings) | variants)),
                    }
                )
        return tuple(sorted(by_key.values(), key=lambda t: (-t.count, t.label)))

    patterns = {(p.source, p.edge, p.target): p for p in profile.patterns}
    for pattern in schema.relationship_patterns:
        triple = (
            schema_key(pattern.source_type),
            schema_key(pattern.relation),
            schema_key(pattern.target_type),
        )
        patterns.setdefault(
            triple,
            PatternProfile(source=triple[0], edge=triple[1], target=triple[2], count=0),
        )
    return profile.model_copy(
        update={
            "entity_types": merged(
                profile.entity_types,
                ((t.label, t.description) for t in schema.entity_types),
            ),
            "edge_types": merged(
                profile.edge_types,
                ((t.label, t.description) for t in schema.relationship_types),
            ),
            "patterns": tuple(
                sorted(patterns.values(), key=lambda p: (-p.count, p.source, p.edge))
            ),
        }
    )


# --- validation -------------------------------------------------------------------


def _fail(field: str, reason: str) -> InvalidGraphQueryError:
    return InvalidGraphQueryError(field, reason)


# Gremlin names its hidden tokens (``~label``, ``~id``) with this prefix; a
# stored name or a value starting with it would reach the compiled query as
# such a token, which the bytecode checker refuses as a compiler bug.
HIDDEN_TOKEN_PREFIX = "~"


def _refuse_hidden_tokens(plan: ValidatedPlan) -> None:
    """Raise ``InvalidGraphQueryError`` (and log a WARNING: the data holds
    such a name) for a type, relationship type, property key or string
    value of ``plan`` that starts with ``HIDDEN_TOKEN_PREFIX``."""
    named: list[tuple[str, str]] = []
    for index, node in enumerate(plan.nodes):
        named += [(f"nodes[{index}].types", t) for t in node.types]
        for c_index, condition in enumerate(node.conditions):
            field = f"nodes[{index}].where[{c_index}]"
            named.append((f"{field}.prop", condition.prop.key))
            values = (
                condition.value
                if isinstance(condition.value, tuple)
                else (condition.value,)
            )
            named += [(f"{field}.value", v) for v in values if isinstance(v, str)]
    for index, edge in enumerate(plan.edges):
        named += [(f"edges[{index}].types", label) for label in edge.labels]
    r = plan.returns
    refs = [*r.properties, *(p for p in (r.group_prop, r.aggregate_prop) if p)]
    named += [("returns", ref.key) for ref in refs]
    for field, text in named:
        if text.startswith(HIDDEN_TOKEN_PREFIX):
            logger.warning(
                "Structured query refused: %s %r starts with %r, a Gremlin hidden "
                "token (rename it in the data)",
                field,
                text,
                HIDDEN_TOKEN_PREFIX,
            )
            raise _fail(
                field,
                f"'{text}' starts with '{HIDDEN_TOKEN_PREFIX}', which no query may "
                "name",
            )


def _listing(names: Iterable[str], limit: int = 20) -> str:
    shown = list(names)
    more = len(shown) - limit
    text = ", ".join(shown[:limit])
    return f"{text}, ... ({more} more)" if more > 0 else text or "(none)"


def _resolve_types(
    names: Sequence[str], known: tuple[TypeProfile, ...], field: str, what: str
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    by_key = {t.key: t for t in known}
    spellings: list[str] = []
    labels: list[str] = []
    for index, name in enumerate(names):
        found = by_key.get(schema_key(name))
        if found is None:
            raise _fail(
                f"{field}[{index}]",
                f"unknown {what} '{name}'; use one of: "
                + _listing(t.label for t in known),
            )
        labels.append(found.label)
        spellings.extend(s for s in found.spellings if s not in spellings)
    return tuple(spellings), tuple(labels)


def _resolve_property(name: str, profile: GraphProfile, field: str) -> PropertyRef:
    by_name = {p.name: p for p in profile.properties}
    found = by_name.get(name)
    if found is None:
        # A bare attribute or seed property name, when only one is so named.
        candidates = [
            p
            for p in profile.properties
            if p.name in (f"{ATTRIBUTE_NAME_PREFIX}{name}", f"{SEED_NAME_PREFIX}{name}")
        ]
        if len(candidates) == 1:
            found = candidates[0]
    if found is None or (found.source == "vertex" and not is_queryable_key(found.key)):
        raise _fail(
            field,
            f"unknown property '{name}'; use one of: "
            + _listing(p.name for p in profile.properties),
        )
    return PropertyRef(
        name=found.name,
        key=found.key,
        source=found.source,
        kinds=found.kinds,
        multi=found.multi,
    )


def _check_value_kind(prop: PropertyRef, op: str, value: Any, field: str) -> None:
    values = list(value) if isinstance(value, tuple | list) else [value]
    if op == "exists":
        return
    if op == "contains":
        # Only a property that holds strings alone: a store compares a
        # substring against a number by failing the query (Gremlin's
        # TextP.containing).
        if prop.kinds != ("string",):
            raise _fail(
                field,
                f"'contains' needs a property of string values only; {prop.name} "
                f"holds {'/'.join(prop.kinds) or 'unknown'} values",
            )
        return
    if op in _NUMERIC_OPS and value_kind(values[0]) == "boolean":
        raise _fail(field, f"'{op}' does not compare booleans")
    if op in _NUMERIC_OPS and len(prop.kinds) > 1:
        raise _fail(
            field,
            f"'{op}' compares values of one kind; {prop.name} holds "
            f"{'/'.join(prop.kinds)} values",
        )
    if not prop.kinds:
        return
    for item in values:
        if value_kind(item) not in prop.kinds:
            raise _fail(
                field,
                f"{prop.name} holds {'/'.join(prop.kinds)} values, not "
                f"{value_kind(item)} {item!r}",
            )


def validate_plan(
    plan: GraphQueryPlan, profile: GraphProfile, *, max_hops: int
) -> ValidatedPlan:
    """``plan`` checked against ``profile`` and resolved to stored names.

    Raises:
        InvalidGraphQueryError: The plan has no node or return clause, an
            edge count other than nodes - 1, a repeated or unknown variable,
            more than ``max_hops`` hops, a type, relationship type or
            property the profile does not list, a value of another kind than
            the property holds, a return clause its kind cannot use, or a
            name or string value starting with ``~`` (a Gremlin hidden
            token; logged as a WARNING).
    """
    if not plan.nodes:
        raise _fail("nodes", "a plan needs at least one node")
    if plan.returns is None:
        raise _fail("returns", "a plan needs a return clause")
    if len(plan.edges) != len(plan.nodes) - 1:
        raise _fail(
            "edges",
            f"{len(plan.nodes)} nodes need {len(plan.nodes) - 1} edges, "
            f"not {len(plan.edges)}",
        )
    seen: set[str] = set()
    for index, node in enumerate(plan.nodes):
        if node.var in seen:
            raise _fail(f"nodes[{index}].var", f"variable '{node.var}' is repeated")
        seen.add(node.var)
    total = sum(edge.hops[1] for edge in plan.edges)
    if total > max_hops:
        raise _fail(
            "edges", f"the pattern spans up to {total} hops; at most {max_hops}"
        )

    def resolve(
        names: Sequence[str], known: tuple[TypeProfile, ...], field: str, what: str
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        try:
            return _resolve_types(names, known, field, what)
        except InvalidGraphQueryError as e:
            if not profile.sampled:
                raise
            # A rare type the sample missed is refused too: say so, so the
            # repair picks a listed one and the operator can declare it.
            raise _fail(
                e.field,
                f"{e.reason} (the schema lists the types of a sample of "
                f"{profile.sample_size} entities and the declared ones)",
            ) from e

    nodes: list[ResolvedNode] = []
    for index, node in enumerate(plan.nodes):
        types, labels = resolve(
            node.types, profile.entity_types, f"nodes[{index}].types", "entity type"
        )
        conditions: list[ResolvedCondition] = []
        for c_index, condition in enumerate(node.where):
            field = f"nodes[{index}].where[{c_index}]"
            prop = _resolve_property(condition.prop, profile, f"{field}.prop")
            value: Any = condition.value
            _check_value_kind(prop, condition.op, value, f"{field}.value")
            conditions.append(
                ResolvedCondition(
                    prop=prop,
                    op=condition.op,
                    value=tuple(value) if isinstance(value, list) else value,
                )
            )
        nodes.append(
            ResolvedNode(
                var=node.var,
                types=types,
                type_labels=labels,
                mention=node.mention,
                conditions=tuple(conditions),
            )
        )
    edges = []
    for index, edge in enumerate(plan.edges):
        labels_, type_labels = resolve(
            edge.types, profile.edge_types, f"edges[{index}].types", "relationship type"
        )
        edges.append(
            ResolvedEdge(
                labels=labels_,
                type_labels=type_labels,
                direction=edge.direction,
                min_hops=edge.hops[0],
                max_hops=edge.hops[1],
            )
        )
    returns = _resolve_return(plan, profile, seen)
    validated = ValidatedPlan(
        nodes=tuple(nodes), edges=tuple(edges), returns=returns, plan=plan
    )
    _refuse_hidden_tokens(validated)
    return validated


def _resolve_return(
    plan: GraphQueryPlan, profile: GraphProfile, known: set[str]
) -> ResolvedReturn:
    clause = plan.returns
    assert clause is not None
    every = tuple(node.var for node in plan.nodes)

    def var(name: str, field: str) -> str:
        if name not in known:
            raise _fail(
                field, f"unknown variable '{name}'; nodes are {', '.join(every)}"
            )
        return name

    vars_ = tuple(var(v, f"returns.vars[{i}]") for i, v in enumerate(clause.vars))
    properties = tuple(
        _resolve_property(name, profile, f"returns.properties[{i}]")
        for i, name in enumerate(clause.properties)
    )
    fields: dict[str, Any] = {
        "kind": clause.kind,
        "order": clause.order,
        "limit": clause.limit,
        "properties": properties,
    }
    kind = clause.kind
    if kind == "rows":
        fields["vars"] = vars_ or every
    elif kind == "count_distinct":
        if len(vars_) > 1 or (not vars_ and len(every) > 1):
            raise _fail("returns.vars", "count_distinct counts the vertices of one var")
        fields["vars"] = vars_ or every
    elif kind == "group_count":
        if clause.group_by is None:
            raise _fail("returns.group_by", "group_count needs group_by")
        if len(vars_) > 1:
            raise _fail(
                "returns.vars", "group_count counts the vertices of at most one var"
            )
        fields["vars"] = vars_ or every
        fields["group_var"] = var(clause.group_by.var, "returns.group_by.var")
        if clause.group_by.prop is not None:
            prop = _resolve_property(
                clause.group_by.prop, profile, "returns.group_by.prop"
            )
            if prop.multi:
                raise _fail(
                    "returns.group_by.prop",
                    f"{prop.name} holds several values a vertex; group by a "
                    "single-valued property",
                )
            if len(prop.kinds) > 1:
                raise _fail(
                    "returns.group_by.prop",
                    f"{prop.name} holds values of several kinds",
                )
            fields["group_prop"] = prop
    elif kind == "aggregate":
        if clause.aggregate is None:
            raise _fail("returns.aggregate", "aggregate needs an aggregate clause")
        prop = _resolve_property(
            clause.aggregate.prop, profile, "returns.aggregate.prop"
        )
        if prop.kinds and "number" not in prop.kinds:
            raise _fail("returns.aggregate.prop", f"{prop.name} is not numeric")
        fields["vars"] = (var(clause.aggregate.var, "returns.aggregate.var"),)
        fields["aggregate_fn"] = clause.aggregate.fn
        fields["aggregate_var"] = clause.aggregate.var
        fields["aggregate_prop"] = prop
    elif kind == "path":
        if len(every) != 2:
            raise _fail(
                "returns.kind",
                "a path runs between two nodes joined by one edge pattern "
                "(its hops: [1, n])",
            )
        fields["vars"] = every
    else:  # count, exists
        fields["vars"] = every
    return ResolvedReturn(**fields)


# --- rendering -------------------------------------------------------------------


def _quoted(value: Any) -> str:
    if isinstance(value, str):
        text = value if len(value) <= 80 else value[:80] + "…"
        return json.dumps(text, ensure_ascii=False)
    return json.dumps(value)


def _condition_text(condition: ResolvedCondition) -> str:
    name, op, value = condition.prop.name, condition.op, condition.value
    if op == "exists":
        return f"has {name}" if value is not False else f"has no {name}"
    if op == "in":
        assert isinstance(value, tuple)
        return f"{name} in [{', '.join(_quoted(v) for v in value)}]"
    symbol = {
        "eq": "=",
        "neq": "!=",
        "gt": ">",
        "gte": ">=",
        "lt": "<",
        "lte": "<=",
        "contains": "contains",
    }[op]
    return f"{name} {symbol} {_quoted(value)}"


def _node_text(node: ResolvedNode, linked: Mapping[str, Sequence[str]]) -> str:
    parts = [
        node.var + (": " + " | ".join(node.type_labels) if node.type_labels else "")
    ]
    if node.mention is not None:
        names = list(dict.fromkeys(linked.get(node.var, ())))
        shown = ", ".join(_quoted(n) for n in names[:5])
        more = f" and {len(names) - 5} more" if len(names) > 5 else ""
        parts.append(
            f"named {_quoted(node.mention)}"
            + (f" = {shown}{more}" if names else " (linked to nothing)")
        )
    parts.extend(_condition_text(c) for c in node.conditions)
    return "(" + ", ".join(parts) + ")"


def _edge_text(edge: ResolvedEdge) -> str:
    types = "|".join(edge.type_labels) if edge.type_labels else "any"
    hops = "" if edge.max_hops == 1 else f" *{edge.min_hops}..{edge.max_hops}"
    inner = f"[{types}{hops}]"
    return {"out": f"-{inner}->", "in": f"<-{inner}-", "both": f"-{inner}-"}[
        edge.direction
    ]


def describe_plan(
    plan: ValidatedPlan, linked: Mapping[str, Sequence[str]] | None = None
) -> str:
    """The plan in English and pattern notation, written by code (the
    context, the logs and ``metadata.structured``); ``linked`` gives each
    mention node the names of the stored entities it was linked to."""
    linked = linked or {}
    pattern = _node_text(plan.nodes[0], linked)
    for edge, node in zip(plan.edges, plan.nodes[1:], strict=True):
        pattern += f" {_edge_text(edge)} {_node_text(node, linked)}"
    r = plan.returns
    if r.kind == "count":
        what = "the number of distinct matches"
    elif r.kind == "count_distinct":
        what = f"the number of distinct {r.vars[0]}"
    elif r.kind == "group_count":
        key = f"{r.group_var}.{r.group_prop.name}" if r.group_prop else r.group_var
        counted = f"distinct {r.vars[0]}" if len(r.vars) == 1 else "matches"
        what = (
            f"{counted} per {key}, {'most' if r.order == 'desc' else 'fewest'} "
            f"first, top {r.limit}"
        )
    elif r.kind == "aggregate":
        assert r.aggregate_prop is not None
        what = f"the {r.aggregate_fn} of {r.aggregate_var}.{r.aggregate_prop.name}"
    elif r.kind == "exists":
        what = "whether anything matches"
    elif r.kind == "path":
        what = f"the paths from {r.vars[0]} to {r.vars[-1]}, shortest first"
    else:
        what = f"the distinct {', '.join(r.vars)} (up to {r.limit})"
    shown = (
        f"; showing {', '.join(p.name for p in r.properties)}" if r.properties else ""
    )
    note = ""
    counts_matches = r.kind == "count" or (r.kind == "group_count" and len(r.vars) != 1)
    if counts_matches and any(edge.direction == "both" for edge in plan.edges):
        # A match is a binding of every node: (a, b) and (b, a) are two.
        note = (
            " An edge of either direction matches a pair once in each order"
            " both nodes fit, so a symmetric pattern counts each pair twice."
        )
    return f"Match {pattern}. Return {what}{shown}.{note}"


# --- the store operation ------------------------------------------------------------


@dataclass(frozen=True)
class Operation:
    """The store operation a plan needs (``plan_operation``)."""

    operation: GraphOperation
    vars: tuple[str, ...]
    limit: int
    group_var: str | None = None
    group_key: str | None = None
    scan: bool = False
    order: Literal["desc", "asc"] = "desc"


def plan_operation(plan: ValidatedPlan, *, scan: bool, cap: int) -> Operation:
    """What the store computes for ``plan``.

    Without ``scan`` the store aggregates: ``count`` (exact),
    ``group`` or ``tuples`` / ``paths`` of at most ``cap`` rows that code
    orders and cuts. With ``scan`` (the caller must check the bound vertices
    in code: document filters), or for a plan that reads seed properties,
    every plan is ``tuples`` of all nodes (``paths`` for a path), at most
    ``cap`` of them, aggregated in code.
    """
    r = plan.returns
    every = plan.vars
    if scan or plan.reads_seed_properties:
        if r.kind == "path":
            return Operation("paths", every, cap, scan=True)
        return Operation("tuples", every, cap, scan=True)
    if r.kind == "count":
        return Operation("count", every, 0)
    if r.kind == "count_distinct":
        return Operation("count", r.vars, 0)
    if r.kind == "exists":
        return Operation("tuples", every, 1)
    if r.kind == "group_count":
        assert r.group_var is not None
        return Operation(
            "group",
            r.vars,
            cap,
            group_var=r.group_var,
            group_key=r.group_prop.key if r.group_prop else None,
            order=r.order,
        )
    if r.kind == "path":
        return Operation("paths", every, cap)
    return Operation("tuples", r.vars, cap)


def hydrate_keys(plan: ValidatedPlan) -> tuple[str, ...]:
    """The vertex properties an answer to ``plan`` reads: the projection,
    and the properties it shows, groups by or aggregates (the seed
    properties' JSON when it reads one)."""
    keys = list(PROJECTION_KEYS)
    r = plan.returns
    refs = [*r.properties, *(p for p in (r.group_prop, r.aggregate_prop) if p)]
    refs += [c.prop for node in plan.nodes for c in node.conditions]
    for ref in refs:
        key = SEED_PROPERTIES_KEY if ref.source == "seed" else ref.key
        if key not in keys:
            keys.append(key)
    return tuple(keys)


# --- answers -------------------------------------------------------------------------


def _distinct(
    rows: Iterable[Mapping[str, str]], vars_: Sequence[str]
) -> list[dict[str, str]]:
    seen: dict[tuple[str, ...], dict[str, str]] = {}
    for row in rows:
        key = tuple(str(row.get(v, "")) for v in vars_)
        if key not in seen:
            seen[key] = {v: str(row.get(v, "")) for v in vars_}
    return list(seen.values())


def _ordered_rows(
    rows: list[dict[str, str]],
    vars_: Sequence[str],
    vertices: Mapping[str, VertexMap],
) -> list[dict[str, str]]:
    first = vars_[0]
    return sorted(
        rows,
        key=lambda row: (
            -vertex_rank(vertices.get(row[first])),
            tuple(row[v] for v in vars_),
        ),
    )


def _ordered_groups(groups: Iterable[GroupRow], order: str) -> list[GroupRow]:
    sign = -1 if order == "desc" else 1
    return sorted(groups, key=lambda g: (sign * g.count, scalar_sort_key(g.key)))


def _ordered_paths(paths: Iterable[PathRow]) -> list[PathRow]:
    return sorted(
        set(paths),
        key=lambda p: (len(p.vertex_ids), p.vertex_ids, tuple(e.id for e in p.edges)),
    )


def _aggregate(fn: str, values: list[float]) -> float | None:
    if not values:
        return None
    if fn == "min":
        return min(values)
    if fn == "max":
        return max(values)
    total = math.fsum(values)
    return total if fn == "sum" else total / len(values)


def answer_from_result(plan: ValidatedPlan, result: GraphQueryResult) -> QueryAnswer:
    """The answer of a store result of ``plan_operation(plan, scan=False)``.

    The store cuts groups, rows and paths in the plan's order, so the first
    ``limit`` of them are exact whenever its page holds that many: only a
    page shorter than the limit and cut (never with the default caps) is a
    truncation."""
    r = plan.returns

    def cut(found: int) -> bool:
        return result.truncated and found < r.limit

    if result.operation == "count":
        return QueryAnswer(kind=r.kind, count=result.count or 0)
    if result.operation == "group":
        groups = _ordered_groups(result.groups, r.order)
        return QueryAnswer(
            kind=r.kind,
            groups=tuple(groups[: r.limit]),
            truncated=cut(len(groups)),
            more=result.truncated or len(groups) > r.limit,
        )
    if result.operation == "paths":
        paths = _ordered_paths(result.paths)
        return QueryAnswer(
            kind=r.kind,
            paths=tuple(paths[: r.limit]),
            truncated=cut(len(paths)),
            more=result.truncated or len(paths) > r.limit,
        )
    if r.kind == "exists":
        return QueryAnswer(kind=r.kind, exists=bool(result.rows))
    if r.kind == "rows":
        answer = aggregate_bindings(
            plan, result.rows, result.vertices, truncated=False, filtered=False
        )
        return answer.model_copy(
            update={
                "truncated": cut(len(answer.rows)),
                "more": answer.more or result.truncated,
            }
        )
    return aggregate_bindings(
        plan, result.rows, result.vertices, truncated=result.truncated, filtered=False
    )


def aggregate_bindings(
    plan: ValidatedPlan,
    bindings: Iterable[Mapping[str, str]],
    vertices: Mapping[str, VertexMap],
    *,
    truncated: bool = False,
    paths: Iterable[PathRow] = (),
    filtered: bool = True,
) -> QueryAnswer:
    """The answer of ``plan`` computed in code from ``bindings`` (each a
    var -> id map) and the stored properties of their vertices.

    With ``filtered`` (a scan), each node's seed conditions are applied
    first (its vertex conditions the store applied). ``truncated``: the
    bindings are a subset (the scan stopped at its cap): a count is then a
    lower bound and a list a subset.
    """
    r = plan.returns
    rows = (
        matching_bindings(plan, bindings, vertices)
        if filtered
        else _distinct(bindings, r.vars)
    )
    if r.kind == "path":
        ends = {tuple(row[v] for v in plan.vars) for row in rows}
        kept = [p for p in paths if (p.vertex_ids[0], p.vertex_ids[-1]) in ends]
        walked = _ordered_paths(kept)
        return QueryAnswer(
            kind=r.kind,
            paths=tuple(walked[: r.limit]),
            truncated=truncated,
            more=len(walked) > r.limit,
        )
    if r.kind == "count":
        return QueryAnswer(kind=r.kind, count=len(rows), truncated=truncated)
    if r.kind == "count_distinct":
        count = len({row[r.vars[0]] for row in rows})
        return QueryAnswer(kind=r.kind, count=count, truncated=truncated)
    if r.kind == "exists":
        return QueryAnswer(
            kind=r.kind, exists=bool(rows), truncated=truncated and not rows
        )
    if r.kind == "group_count":
        assert r.group_var is not None
        counted: dict[Any, set[tuple[str, ...]]] = defaultdict(set)
        for row in rows:
            vertex = vertices.get(row[r.group_var]) or {}
            key: Any = (
                row[r.group_var]
                if r.group_prop is None
                else first_value(vertex, r.group_prop)
            )
            if key is None or value_kind(key) is None:
                continue
            counted[canonical_scalar(key)].add(tuple(row[v] for v in r.vars))
        groups = _ordered_groups(
            (GroupRow(key=key, count=len(ids)) for key, ids in counted.items()),
            r.order,
        )
        return QueryAnswer(
            kind=r.kind,
            groups=tuple(groups[: r.limit]),
            truncated=truncated,
            more=len(groups) > r.limit,
        )
    if r.kind == "aggregate":
        assert r.aggregate_var is not None and r.aggregate_prop is not None
        numbers = []
        for vertex_id in dict.fromkeys(row[r.aggregate_var] for row in rows):
            for value in property_values(
                vertices.get(vertex_id) or {}, r.aggregate_prop
            ):
                if value_kind(value) == "number":
                    numbers.append(float(value))
                    break
        return QueryAnswer(
            kind=r.kind,
            value=_aggregate(str(r.aggregate_fn), numbers),
            value_count=len(numbers),
            truncated=truncated,
        )
    projected = _distinct(rows, r.vars)
    ordered = _ordered_rows(projected, r.vars, vertices)
    return QueryAnswer(
        kind=r.kind,
        rows=tuple(ordered[: r.limit]),
        truncated=truncated,
        more=len(ordered) > r.limit,
    )


def matching_bindings(
    plan: ValidatedPlan,
    bindings: Iterable[Mapping[str, str]],
    vertices: Mapping[str, VertexMap],
) -> list[dict[str, str]]:
    """The distinct ``bindings`` of every node whose seed conditions hold
    (what a scan's answer is computed over)."""
    rows = _distinct(bindings, plan.vars)
    return [row for row in rows if _seed_conditions_hold(plan, row, vertices)]


def _seed_conditions_hold(
    plan: ValidatedPlan, row: Mapping[str, str], vertices: Mapping[str, VertexMap]
) -> bool:
    for node in plan.nodes:
        conditions = node.seed_conditions
        if not conditions:
            continue
        vertex = vertices.get(row[node.var]) or {}
        if not all(
            condition_holds(property_values(vertex, c.prop), c) for c in conditions
        ):
            return False
    return True


# --- the reference executor ------------------------------------------------------------


@dataclass(frozen=True)
class Walk:
    """A match: its vertices, the edge of each step and each node's
    vertex."""

    vertices: tuple[str, ...]
    edges: tuple[GraphEdgeRecord, ...]
    binding: tuple[str, ...]


def _types_hold(node: ResolvedNode, vertex: VertexMap) -> bool:
    if not node.types:
        return True
    return any(t in node.types for t in vertex.get("type") or ())


def node_holds(
    node: ResolvedNode,
    vertex_id: str,
    vertex: VertexMap,
    *,
    linked: Mapping[str, Sequence[str]],
    allowed_ids: Collection[str] | None,
    vertex_ok: VertexPredicate | None,
) -> bool:
    """Whether ``vertex`` may bind ``node``: its types, its vertex
    conditions, its linked ids, the caller's ``id`` filter and the caller's
    vertex filter (``vertex_ok``). Seed conditions are not read here: the
    stores cannot evaluate them, and neither does the reference."""
    if node.mention is not None and vertex_id not in linked.get(node.var, ()):
        return False
    if allowed_ids is not None and vertex_id not in allowed_ids:
        return False
    if not _types_hold(node, vertex):
        return False
    for condition in node.vertex_conditions:
        if not condition_holds(property_values(vertex, condition.prop), condition):
            return False
    return vertex_ok is None or vertex_ok(vertex)


def edge_holds(
    edge: ResolvedEdge, record: GraphEdgeRecord, edge_ok: EdgePredicate | None
) -> bool:
    if edge.labels and record.type not in edge.labels:
        return False
    return edge_ok is None or edge_ok(record.props)


def match_walks(
    plan: ValidatedPlan,
    graph: PropertyGraph,
    *,
    linked: Mapping[str, Sequence[str]] | None = None,
    allowed_ids: Sequence[str] | None = None,
    vertex_ok: VertexPredicate | None = None,
    edge_ok: EdgePredicate | None = None,
) -> Iterator[Walk]:
    """Every match of ``plan`` in ``graph``, in a deterministic order."""
    linked = linked or {}
    allowed = None if allowed_ids is None else frozenset(allowed_ids)
    out_edges: dict[str, list[GraphEdgeRecord]] = defaultdict(list)
    in_edges: dict[str, list[GraphEdgeRecord]] = defaultdict(list)
    for record in sorted(graph.edges, key=lambda e: e.id):
        out_edges[record.source_id].append(record)
        in_edges[record.target_id].append(record)

    def ok(index: int, vertex_id: str) -> bool:
        vertex = graph.vertices.get(vertex_id)
        return vertex is not None and node_holds(
            plan.nodes[index],
            vertex_id,
            vertex,
            linked=linked,
            allowed_ids=allowed,
            vertex_ok=vertex_ok,
        )

    def steps(edge: ResolvedEdge, at: str) -> Iterator[tuple[GraphEdgeRecord, str]]:
        if edge.direction in ("out", "both"):
            for record in out_edges.get(at, ()):
                if edge_holds(edge, record, edge_ok):
                    yield record, record.target_id
        if edge.direction in ("in", "both"):
            for record in in_edges.get(at, ()):
                if edge_holds(edge, record, edge_ok):
                    yield record, record.source_id

    def extend(
        index: int,
        walk: tuple[str, ...],
        edges: tuple[GraphEdgeRecord, ...],
        binding: tuple[str, ...],
    ) -> Iterator[Walk]:
        if index == len(plan.edges):
            yield Walk(walk, edges, binding)
            return
        edge = plan.edges[index]

        def hop(
            path: tuple[str, ...], taken: tuple[GraphEdgeRecord, ...], made: int
        ) -> Iterator[Walk]:
            if made >= edge.min_hops and ok(index + 1, path[-1]):
                yield from extend(index + 1, path, taken, (*binding, path[-1]))
            if made == edge.max_hops:
                return
            for record, nxt in steps(edge, path[-1]):
                if nxt in path or nxt not in graph.vertices:
                    continue
                yield from hop((*path, nxt), (*taken, record), made + 1)

        for record, nxt in steps(edge, walk[-1]):
            if nxt in walk or nxt not in graph.vertices:
                continue
            yield from hop((*walk, nxt), (*edges, record), 1)

    for vertex_id in sorted(graph.vertices):
        if ok(0, vertex_id):
            yield from extend(0, (vertex_id,), (), (vertex_id,))


def reference_execute(
    plan: ValidatedPlan,
    graph: PropertyGraph,
    *,
    linked: Mapping[str, Sequence[str]] | None = None,
) -> QueryAnswer:
    """The answer to ``plan`` over all of ``graph``, without filters: every
    match scanned and aggregated in code (the structural benchmark's gold
    answers)."""
    walks = list(match_walks(plan, graph, linked=linked))
    bindings = [dict(zip(plan.vars, w.binding, strict=True)) for w in walks]
    paths = [
        PathRow(
            vertex_ids=w.vertices,
            edges=tuple(
                PathEdge(
                    id=e.id, type=e.type, source_id=e.source_id, target_id=e.target_id
                )
                for e in w.edges
            ),
        )
        for w in walks
    ]
    return aggregate_bindings(plan, bindings, graph.vertices, paths=paths)


# --- the answer as context ------------------------------------------------------------


def answer_vertex_ids(plan: ValidatedPlan, answer: QueryAnswer) -> list[str]:
    """The vertices an answer names, in its order: each row's, each group
    of vertices', each path's."""
    ids: dict[str, None] = {}
    for row in answer.rows:
        ids.update(dict.fromkeys(row[v] for v in plan.returns.vars if v in row))
    if plan.returns.kind == "group_count" and plan.returns.group_prop is None:
        ids.update(dict.fromkeys(str(g.key) for g in answer.groups))
    for path in answer.paths:
        ids.update(dict.fromkeys(path.vertex_ids))
    return list(ids)


def literal_conditions(plan: ValidatedPlan) -> list[ResolvedCondition]:
    """The string literals ``plan`` compares by equality (``eq``, and each
    value of an ``in``), each once, as ``eq`` conditions: what a zero count
    or a no rests on the data spelling as the plan does."""
    found: dict[tuple[str, str, str], ResolvedCondition] = {}
    for node in plan.nodes:
        for condition in node.conditions:
            if condition.op not in ("eq", "in"):
                continue
            values = (
                condition.value
                if isinstance(condition.value, tuple)
                else (condition.value,)
            )
            for value in values:
                if isinstance(value, str):
                    key = (condition.prop.source, condition.prop.key, value)
                    found.setdefault(
                        key,
                        ResolvedCondition(prop=condition.prop, op="eq", value=value),
                    )
    return list(found.values())


def value_check_plan(
    plan: ValidatedPlan, condition: ResolvedCondition
) -> ValidatedPlan:
    """A plan asking whether any vertex (of any type) satisfies
    ``condition``: whether the data holds the value at all."""
    return ValidatedPlan(
        nodes=(ResolvedNode(var="x", conditions=(condition,)),),
        edges=(),
        returns=ResolvedReturn(kind="exists", vars=("x",)),
        plan=plan.plan,
    )


def answer_holders(
    plan: ValidatedPlan, answer: QueryAnswer
) -> tuple[str, ResolvedCondition] | None:
    """The node whose vertices hold an answer's value and the condition
    that selects them, for an answer that names no vertex: a ``min`` or
    ``max`` aggregate's value on its var, a group count by a property's
    listed keys on the group var; ``None`` for any other answer (a count,
    a yes, a sum or a mean rests on every match alike)."""
    r = plan.returns
    if (
        r.kind == "aggregate"
        and r.aggregate_fn in ("min", "max")
        and r.aggregate_var is not None
        and r.aggregate_prop is not None
        and answer.value is not None
    ):
        condition = ResolvedCondition(
            prop=r.aggregate_prop, op="eq", value=canonical_scalar(answer.value)
        )
        return r.aggregate_var, condition
    if (
        r.kind == "group_count"
        and r.group_var is not None
        and r.group_prop is not None
        and answer.groups
    ):
        keys = tuple(dict.fromkeys(canonical_scalar(g.key) for g in answer.groups))
        return r.group_var, ResolvedCondition(prop=r.group_prop, op="in", value=keys)
    return None


def with_condition(
    plan: ValidatedPlan, var: str, condition: ResolvedCondition
) -> ValidatedPlan:
    """``plan`` with ``condition`` added to the node of ``var``."""
    nodes = tuple(
        (
            node.model_copy(update={"conditions": (*node.conditions, condition)})
            if node.var == var
            else node
        )
        for node in plan.nodes
    )
    return plan.model_copy(update={"nodes": nodes})


def evidence_vertex_ids(
    plan: ValidatedPlan,
    answer: QueryAnswer,
    rows: Iterable[Mapping[str, str]],
    vertices: Mapping[str, VertexMap],
) -> list[str]:
    """The vertices of matching ``rows`` that evidence an answer naming
    none, each once: the rows whose vertex holds the answer's value first
    (``answer_holders``; by the listed groups' order), then by the first
    node's rank and the ids; in each row, the holding node's vertex first."""
    holders = answer_holders(plan, answer)
    var = holders[0] if holders else plan.vars[0]
    order = [var, *(v for v in plan.vars if v != var)]
    keys: dict[Any, int] = {}
    if holders is not None and isinstance(holders[1].value, tuple):
        keys = {key: index for index, key in enumerate(holders[1].value)}

    def place(row: Mapping[str, str]) -> int:
        if holders is None:
            return 0
        vertex = vertices.get(row[var]) or {}
        values = property_values(vertex, holders[1].prop)
        if not condition_holds(values, holders[1]):
            return len(keys) + 1
        return min((keys.get(canonical_scalar(v), 0) for v in values), default=0)

    ranked = sorted(
        _distinct(rows, plan.vars),
        key=lambda row: (
            place(row),
            -vertex_rank(vertices.get(row[plan.vars[0]])),
            tuple(row[v] for v in plan.vars),
        ),
    )
    return list(dict.fromkeys(row[v] for row in ranked for v in order))


def _vertex_label(vertex_id: str, vertices: Mapping[str, VertexMap]) -> str:
    vertex = vertices.get(vertex_id) or {}
    name = next(iter(vertex.get("name") or ()), vertex_id)
    kind = next(iter(vertex.get("type") or ()), None)
    return f"{name} ({kind})" if kind else str(name)


def _shown_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value)
    return text if len(text) <= 200 else text[:200] + "…"


def render_answer(
    plan: ValidatedPlan,
    answer: QueryAnswer,
    vertices: Mapping[str, VertexMap],
    *,
    linked: Mapping[str, Sequence[str]] | None = None,
    max_lines: int = 50,
) -> str:
    """The query result's context section, written by code: the plan, the
    answer (a count, groups, rows, a value, yes/no or paths, at most
    ``max_lines`` of them) and whether it is complete."""
    r = plan.returns
    lines = [f"Graph query: {describe_plan(plan, linked)}"]
    scope = (
        "computed by code over every match in the graph, within the request's "
        "filters"
        if not answer.truncated
        else "TRUNCATED: computed over the first matches only"
    )
    lines.append(f"Result ({scope}):")
    if r.kind in ("count", "count_distinct"):
        count = answer.count or 0
        lines.append(f"{'at least ' if answer.truncated else ''}{count}")
    elif r.kind == "exists":
        if answer.exists:
            lines.append("yes")
        else:
            lines.append(
                "no match among the matches checked" if answer.truncated else "no"
            )
    elif r.kind == "aggregate":
        assert r.aggregate_prop is not None
        value = "none" if answer.value is None else _shown_value(answer.value)
        lines.append(
            f"{r.aggregate_fn} of {r.aggregate_var}.{r.aggregate_prop.name}: "
            f"{value} (over {answer.value_count} values)"
        )
    elif r.kind == "group_count":
        key_name = r.group_prop.name if r.group_prop else r.group_var
        counted = f"distinct {r.vars[0]}" if len(r.vars) == 1 else "matches"
        lines.append(f"{key_name} | {counted}")
        for group in answer.groups[:max_lines]:
            key = (
                _vertex_label(str(group.key), vertices)
                if r.group_prop is None
                else _shown_value(group.key)
            )
            bound = "at least " if answer.truncated else ""
            lines.append(f"{key} | {bound}{group.count}")
    elif r.kind == "path":
        for index, path in enumerate(answer.paths[:max_lines], start=1):
            text = _vertex_label(path.vertex_ids[0], vertices)
            for step, edge in enumerate(path.edges):
                here, there = path.vertex_ids[step], path.vertex_ids[step + 1]
                arrow = (
                    f" -[{edge.type}]-> "
                    if edge.source_id == here
                    else f" <-[{edge.type}]- "
                )
                text += arrow + _vertex_label(there, vertices)
            lines.append(f"{index}. {text}")
    else:
        for index, row in enumerate(answer.rows[:max_lines], start=1):
            parts = []
            for var in r.vars:
                vertex = vertices.get(row[var]) or {}
                shown = [
                    f"{prop.name}: "
                    + ", ".join(_shown_value(v) for v in property_values(vertex, prop))
                    for prop in r.properties
                    if property_values(vertex, prop)
                ]
                text = f"{var} = {_vertex_label(row[var], vertices)}"
                if shown:
                    text += "; " + "; ".join(shown)
                parts.append(text)
            lines.append(f"{index}. " + ", ".join(parts))
    listed = max(len(answer.rows), len(answer.groups), len(answer.paths))
    if listed > max_lines:
        lines.append(f"... and {listed - max_lines} more")
    elif answer.more and listed:
        lines.append(f"(the first {listed} by the plan's order; more matched)")
    return "\n".join(lines)
