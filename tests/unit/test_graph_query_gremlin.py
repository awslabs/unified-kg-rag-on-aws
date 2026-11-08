# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Gremlin compiler and the bytecode checker (docs/design.md §6.6).

No server: the traversals are built from an unconnected source and read
as bytecode. The suites over Gremlin Server and Neptune are
``tests/integration/test_graph_query_conformance.py``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from gremlin_python.process.graph_traversal import __
from gremlin_python.process.traversal import Bytecode
from gremlin_python.structure.graph import Graph
from hypothesis import given, settings

from tests.fixtures.graph_query import CONFORMANCE_PLANS, validated
from tests.property.test_graph_query_properties import plans
from unified_kg_rag.adapters.graph_query.checks import GREMLIN_DENIED, check_gremlin
from unified_kg_rag.adapters.graph_query.gremlin import (
    GremlinQuery,
    hydrate_traversal,
    profile_traversal,
)
from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.domain.models.graph_query import GraphQueryPlan, GraphQueryRequest
from unified_kg_rag.domain.retrieval.filters import KindsRule
from unified_kg_rag.domain.retrieval.graph_query import (
    hydrate_keys,
    plan_operation,
    validate_plan,
)
from unified_kg_rag.shared.exceptions import (
    GraphQueryCompileError,
    InvalidGraphQueryError,
)

pytestmark = pytest.mark.unit

LABEL = "Entity-default"
g = Graph().traversal()


def _query(plan_name_or_raw: str | dict, **request: Any) -> GremlinQuery:
    raw = (
        CONFORMANCE_PLANS[plan_name_or_raw]
        if isinstance(plan_name_or_raw, str)
        else plan_name_or_raw
    )
    plan = validated(raw)
    op = plan_operation(plan, scan=False, cap=50)
    fields = {
        "operation": op.operation,
        "vars": op.vars,
        "group_var": op.group_var,
        "group_key": op.group_key,
        "order": op.order,
        "limit": op.limit,
        "hydrate": hydrate_keys(plan),
        **request,
    }
    filters = fields.pop("filters", None)
    # As NeptuneRetriever.aexecute scopes them.
    scoped = NeptuneRetriever._scope_filters_to_label(LABEL, filters)
    return GremlinQuery(
        GraphQueryRequest(plan=plan, filters=filters, **fields),
        LABEL,
        apply_filters=lambda t: NeptuneRetriever._apply_filters(t, scoped),
        edge_rule=None,
        timeout_ms=5000,
    )


def _steps(bytecode: Bytecode) -> Iterator[list[Any]]:
    """Every step instruction, nested ones included."""
    for step in bytecode.step_instructions:
        yield list(step)
        for arg in step[1:]:
            if isinstance(arg, Bytecode):
                yield from _steps(arg)


def _assert_scoped(bytecode: Bytecode) -> None:
    """The invariants the checker enforces, asserted independently of it."""
    names = [s[0] for s in _steps(bytecode)]
    assert not set(names) & GREMLIN_DENIED
    assert bytecode.step_instructions[0] == ["V"]

    def walk(code: Bytecode) -> None:
        steps = code.step_instructions
        for index, step in enumerate(steps):
            if step[0] in ("V", "inV", "outV", "otherV"):
                assert steps[index + 1] == ["hasLabel", LABEL], steps
            if step[0] == "hasLabel":
                assert step[1:] == [LABEL]
            for arg in step[1:]:
                if isinstance(arg, Bytecode):
                    walk(arg)

    walk(bytecode)
    options = bytecode.source_instructions
    assert len(options) == 1 and options[0][0] == "withStrategies"
    assert options[0][1].configuration == {"evaluationTimeout": 5000}


@pytest.mark.parametrize("name", sorted(CONFORMANCE_PLANS))
def test_every_compiled_plan_is_scoped_and_read_only(name: str) -> None:
    _assert_scoped(_query(name).build(g).bytecode)


def test_a_count_is_a_label_scoped_walk_over_distinct_vertices() -> None:
    steps = _query("count_supply_pairs").build(g).bytecode.step_instructions
    names = [s[0] for s in steps]
    assert names[:4] == ["V", "hasLabel", "has", "as"]
    assert steps[2][1] == "type"
    assert ["outE", "SUPPLIES"] in steps
    assert names[-4:] == ["by", "by", "dedup", "count"]
    assert "simplePath" in names


def test_a_variable_hop_is_a_union_of_fixed_chains() -> None:
    steps = _query("two_hop_any").build(g).bytecode.step_instructions
    union = next(s for s in steps if s[0] == "union")
    chains = [arg.step_instructions for arg in union[1:]]
    assert [sum(1 for s in c if s[0] == "otherV") for c in chains] == [1, 2, 3]


def test_every_union_branch_keeps_the_walk_simple() -> None:
    # Neptune ignores a simplePath() that follows a union(), so each branch
    # ends with its own; the trailing one still covers the later hops.
    steps = _query("two_hop_any").build(g).bytecode.step_instructions
    union = next(s for s in steps if s[0] == "union")
    chains = [arg.step_instructions for arg in union[1:]]
    assert chains and all(c[-1] == ["simplePath"] for c in chains)
    assert "simplePath" in [s[0] for s in steps[steps.index(union) + 1 :]]


def test_a_fixed_multi_hop_edge_is_one_chain_without_a_union() -> None:
    # Neptune 1.4 rejects union(chain.simplePath()) with one branch followed
    # by simplePath() ("Exception processing a script"); a fixed length walks
    # its hops in place and the trailing simplePath() keeps it simple.
    plan = {
        "nodes": [
            {
                "var": "a",
                "where": [{"prop": "name", "op": "eq", "value": "Hooli Retail"}],
            },
            {"var": "z", "types": ["PRODUCT"]},
        ],
        "edges": [{"direction": "both", "hops": [2, 2]}],
        "returns": {"kind": "rows", "vars": ["z"]},
    }
    steps = _query(plan).build(g).bytecode.step_instructions
    names = [s[0] for s in steps]
    assert "union" not in names
    assert names.count("otherV") == 2
    assert "simplePath" in names[names.index("otherV") :]


def test_a_linked_last_node_starts_the_walk() -> None:
    query = _query(
        {
            "nodes": [
                {"var": "p", "types": ["PRODUCT"]},
                {"var": "x", "mention": "Acme"},
            ],
            "edges": [{"types": ["MAKES"], "direction": "in"}],
            "returns": {"kind": "rows", "vars": ["p"]},
        },
        linked={"x": ("v1",)},
    )
    assert query.reversed
    steps = query.build(g).bytecode.step_instructions
    # The walk starts at x's linked ids and runs the edge the other way.
    assert steps[2][0] == "has" and steps[2][1] == "id"
    assert ["outE", "MAKES"] in steps


def test_values_travel_as_arguments() -> None:
    injected = "x') .drop() //"
    query = _query(
        {
            "nodes": [
                {"var": "v", "where": [{"prop": "name", "op": "eq", "value": injected}]}
            ],
            "returns": {"kind": "count"},
        }
    )
    steps = query.build(g).bytecode.step_instructions
    has = next(s for s in steps if s[0] == "has" and s[1] == "name")
    assert has[2].value == injected
    assert "drop" not in [s[0] for s in _steps(query.build(g).bytecode)]


def test_the_caller_filters_apply_to_every_bound_vertex() -> None:
    query = _query(
        "count_supply_pairs", filters={"attributes.category": "a", "type": "VENDOR"}
    )
    steps = query.build(g).bytecode.step_instructions
    # A strict has() per filter and node: an entity's own attribute is its
    # attr_<key> property.
    assert sum(1 for s in steps if s[0] == "where") == 0
    assert sum(1 for s in steps if s[:2] == ["has", "attr_category"]) == 2
    assert sum(1 for s in steps if s[:2] == ["has", "type"]) == 4


def test_hydration_and_profile_traversals_are_scoped() -> None:
    _assert_scoped(hydrate_traversal(g, LABEL, ["v1"], ["name"], 5000).bytecode)
    _assert_scoped(profile_traversal(g, LABEL, 10, 5, 5000).bytecode)


def _bad(build) -> Bytecode:
    return build(g.with_("evaluationTimeout", 5000)).bytecode


@pytest.mark.parametrize(
    "build",
    [
        pytest.param(lambda s: s.V().hasLabel(LABEL).drop(), id="drop"),
        pytest.param(
            lambda s: s.V().hasLabel(LABEL).property("name", "x"), id="property"
        ),
        pytest.param(lambda s: s.addV(LABEL), id="addV"),
        pytest.param(lambda s: s.E(), id="E-start"),
        pytest.param(lambda s: s.V(), id="no-label"),
        pytest.param(lambda s: s.V().hasLabel("Entity-other"), id="other-label"),
        pytest.param(lambda s: s.V().hasLabel(LABEL, "Entity-other"), id="two-labels"),
        pytest.param(lambda s: s.V("Entity-default:v1").hasLabel(LABEL), id="V-ids"),
        pytest.param(lambda s: s.V().hasLabel(LABEL).out(), id="out-unscoped"),
        pytest.param(lambda s: s.V().hasLabel(LABEL).outE().inV(), id="inV-unscoped"),
        pytest.param(
            lambda s: s.V().hasLabel(LABEL).union(__.outE().otherV()),
            id="nested-unscoped",
        ),
        pytest.param(
            lambda s: s.V().hasLabel(LABEL).where(__.V().hasLabel("Entity-x")),
            id="nested-V",
        ),
        pytest.param(
            lambda s: s.V().hasLabel(LABEL).sideEffect(__.drop()), id="side-effect"
        ),
        pytest.param(lambda s: s.V().hasLabel(LABEL).inject(1), id="inject"),
        pytest.param(
            lambda s: s.V().hasLabel(LABEL).coalesce(__.addE("X")), id="nested-write"
        ),
        pytest.param(
            lambda s: s.V()
            .hasLabel(LABEL)
            .project("x")
            .by(lambda: ("it.property('x', 'y'); 1", "gremlin-groovy")),
            id="lambda",
        ),
        pytest.param(
            lambda s: s.V().hasLabel(LABEL).has("~label", "Entity-other"),
            id="hidden-token",
        ),
        pytest.param(
            lambda s: s.V().hasLabel(LABEL).has("name", object()), id="object-arg"
        ),
        # Neither allowed nor named as denied: the allowlist refuses it (an
        # unbounded walk out of the step caps).
        pytest.param(
            lambda s: s.V().hasLabel(LABEL).repeat(__.both().hasLabel(LABEL)),
            id="repeat",
        ),
        pytest.param(
            lambda s: s.V()
            .hasLabel(LABEL)
            .union(__.repeat(__.both().hasLabel(LABEL)).times(2)),
            id="nested-repeat",
        ),
    ],
)
def test_the_checker_refuses_writes_and_escapes(build) -> None:
    with pytest.raises(GraphQueryCompileError):
        check_gremlin(_bad(build), LABEL)


def test_the_checker_refuses_another_source_instruction() -> None:
    with pytest.raises(GraphQueryCompileError):
        check_gremlin(g.withSideEffect("a", 1).V().hasLabel(LABEL), LABEL)
    with pytest.raises(GraphQueryCompileError):
        check_gremlin(g.with_("x", 1).V().hasLabel(LABEL), LABEL)
    check_gremlin(g.with_("evaluationTimeout", 1).V().hasLabel(LABEL), LABEL)


@settings(max_examples=200)
@given(plans())
def test_every_valid_plan_compiles_to_a_checked_traversal(raw: dict) -> None:
    from tests.fixtures.graph_query import procurement_graph, profile_of

    try:
        plan = validate_plan(
            GraphQueryPlan.model_validate(raw),
            profile_of(procurement_graph()),
            max_hops=4,
        )
    except InvalidGraphQueryError:
        return
    for scan in (False, True):
        op = plan_operation(plan, scan=scan, cap=10)
        query = GremlinQuery(
            GraphQueryRequest(
                plan=plan,
                operation=op.operation,
                vars=op.vars,
                group_var=op.group_var,
                group_key=op.group_key,
                order=op.order,
                limit=op.limit,
            ),
            LABEL,
            apply_filters=lambda t: t,
            edge_rule=KindsRule(("seed_graph", "document")),
            timeout_ms=5000,
        )
        _assert_scoped(query.build(g).bytecode)


def test_groups_of_an_integral_float_and_its_int_are_one() -> None:
    """The server groups 5 and 5.0 apart; the decoded groups merge them."""
    query = _query(
        {
            "nodes": [{"var": "v", "types": ["VENDOR"]}],
            "returns": {
                "kind": "group_count",
                "group_by": {"var": "v", "prop": "rank"},
            },
        }
    )
    decoded = query.decode([{5.0: 2}, {7.5: 2}, {5: 1}, {True: 9}])
    assert [(g.key, g.count) for g in decoded["groups"]] == [
        (5, 3),
        (7.5, 2),
        (True, 9),
    ]
    assert type(decoded["groups"][0].key) is int
