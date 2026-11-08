# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Invariants of the structured query plan (docs/design.md §6.6).

Any JSON a model may write either becomes a validated plan or is refused
with a plan error (never another exception); every validated plan is
described; and over random graphs the store operation a plan runs gives the
answer a recount of every match gives, so in-store aggregation is never
another semantics than the scan.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from tests.fixtures.graph_query import profile_of, property_graph, reference_operation
from unified_kg_rag.domain.models import Entity, Relationship
from unified_kg_rag.domain.models.graph_query import (
    GraphQueryPlan,
    GraphQueryRequest,
    PropertyGraph,
)
from unified_kg_rag.domain.retrieval.graph_query import (
    aggregate_bindings,
    answer_from_result,
    describe_plan,
    hydrate_keys,
    plan_operation,
    reference_execute,
    validate_plan,
)
from unified_kg_rag.shared.exceptions import InvalidGraphQueryError

pytestmark = pytest.mark.property

TYPES = ["VENDOR", "BUYER", "PRODUCT"]
EDGES = ["SUPPLIES", "MAKES", "BUYS"]

json_values = st.recursive(
    st.none()
    | st.booleans()
    | st.integers(-5, 300)
    | st.floats(allow_nan=True, allow_infinity=True)
    | st.sampled_from(
        ["v", "b", "VENDOR", "rank", "name", "eq", "in", "count", "rows", "out", ""]
    ),
    lambda children: st.lists(children, max_size=4)
    | st.dictionaries(
        st.sampled_from(
            [
                "nodes",
                "edges",
                "returns",
                "var",
                "types",
                "where",
                "prop",
                "op",
                "value",
                "kind",
                "vars",
                "hops",
                "direction",
                "group_by",
                "limit",
                "aggregate",
                "fn",
                "mention",
                "unanswerable",
                "label",
                "suffix",
            ]
        ),
        children,
        max_size=5,
    ),
    max_leaves=25,
)


@st.composite
def graphs(draw: st.DrawFn) -> PropertyGraph:
    n = draw(st.integers(2, 9))
    entities = [
        Entity(
            id=f"e{i}",
            name=f"Entity {i}",
            type=draw(st.sampled_from(TYPES)),
            rank=draw(st.integers(0, 5)),
            attributes=(
                # An int or the float it equals (two extraction runs).
                {
                    "seed_properties": {
                        "size": draw(st.integers(1, 4) | st.integers(1, 4).map(float))
                    }
                }
                if draw(st.booleans())
                else None
            ),
        )
        for i in range(n)
    ]
    pairs = draw(
        st.lists(
            st.tuples(
                st.integers(0, n - 1), st.integers(0, n - 1), st.sampled_from(EDGES)
            ),
            max_size=3 * n,
        )
    )
    relationships = [
        Relationship(id=f"r{k}", source_id=f"e{a}", target_id=f"e{b}", type=kind)
        for k, (a, b, kind) in enumerate(pairs)
    ]
    return property_graph(entities, relationships)


@st.composite
def plans(draw: st.DrawFn) -> dict[str, Any]:
    count = draw(st.integers(1, 3))
    nodes = []
    for i in range(count):
        node: dict[str, Any] = {"var": f"n{i}"}
        if draw(st.booleans()):
            node["types"] = draw(
                st.lists(st.sampled_from(TYPES), min_size=1, max_size=2)
            )
        if draw(st.integers(0, 3)) == 0:
            node["where"] = [
                {
                    "prop": "rank",
                    "op": draw(st.sampled_from(["gte", "lt", "neq"])),
                    "value": draw(st.integers(0, 5)),
                }
            ]
        if draw(st.integers(0, 4)) == 0:
            node.setdefault("where", []).append(
                {"prop": "seed.size", "op": "gte", "value": draw(st.integers(1, 4))}
            )
        nodes.append(node)
    edges = []
    for _ in range(count - 1):
        low = draw(st.integers(1, 2))
        edges.append(
            {
                "types": draw(st.lists(st.sampled_from(EDGES), max_size=2)),
                "direction": draw(st.sampled_from(["out", "in", "both"])),
                "hops": [low, draw(st.integers(low, 2))],
            }
        )
    kinds = ["rows", "count", "count_distinct", "group_count", "aggregate", "exists"]
    if count == 2:
        kinds.append("path")
    kind = draw(st.sampled_from(kinds))
    returns: dict[str, Any] = {"kind": kind, "limit": draw(st.integers(1, 30))}
    var = draw(st.sampled_from([n["var"] for n in nodes]))
    if kind == "count_distinct":
        returns["vars"] = [var]
    elif kind == "group_count":
        returns["group_by"] = {
            "var": var,
            "prop": draw(st.sampled_from([None, "type", "rank", "seed.size"])),
        }
    elif kind == "aggregate":
        returns["aggregate"] = {
            "fn": draw(st.sampled_from(["min", "max", "sum", "avg"])),
            "var": var,
            "prop": draw(st.sampled_from(["rank", "seed.size"])),
        }
    elif kind == "rows" and draw(st.booleans()):
        returns["vars"] = [var]
    return {"nodes": nodes, "edges": edges, "returns": returns}


@settings(max_examples=300)
@given(json_values)
def test_any_json_is_a_plan_or_a_plan_error(raw: Any) -> None:
    graph = property_graph(
        [
            Entity(id="a", name="A", type="VENDOR"),
            Entity(id="b", name="B", type="BUYER"),
        ],
        [Relationship(id="r", source_id="a", target_id="b", type="SUPPLIES")],
    )
    try:
        plan = GraphQueryPlan.model_validate(raw)
        validated = validate_plan(plan, profile_of(graph), max_hops=3)
    except (ValidationError, InvalidGraphQueryError):
        return
    describe_plan(validated)


@settings(max_examples=150)
@given(graphs(), plans())
def test_the_store_operation_recounts_every_match(
    graph: PropertyGraph, raw: dict[str, Any]
) -> None:
    try:
        plan = validate_plan(
            GraphQueryPlan.model_validate(raw), profile_of(graph), max_hops=4
        )
    except InvalidGraphQueryError:
        return  # a type, edge or seed property this graph lacks
    assert describe_plan(plan)
    op = plan_operation(plan, scan=False, cap=100_000)
    request = GraphQueryRequest(
        plan=plan,
        operation=op.operation,
        vars=op.vars,
        group_var=op.group_var,
        group_key=op.group_key,
        order=op.order,
        limit=op.limit,
        hydrate=hydrate_keys(plan),
    )
    result = reference_operation(request, graph)
    if op.scan:
        if op.operation == "paths":
            bindings = [
                dict(zip(plan.vars, (p.vertex_ids[0], p.vertex_ids[-1]), strict=True))
                for p in result.paths
            ]
        else:
            bindings = list(result.rows)
        answer = aggregate_bindings(plan, bindings, result.vertices, paths=result.paths)
    else:
        answer = answer_from_result(plan, result)
    assert answer == reference_execute(plan, graph)
    # One group per value: 2.0 is the group 2.
    assert not any(
        isinstance(g.key, float) and g.key.is_integer() for g in answer.groups
    )
