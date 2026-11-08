# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The structured query plan in pure code: its grammar, its validation
against a profile, its rendering, the operation a store runs for it, the
answers and the reference executor (docs/design.md §6.6)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from tests.fixtures.graph_query import (
    CONFORMANCE_PLANS,
    procurement_entities,
    procurement_graph,
    procurement_relationships,
    profile_of,
    property_graph,
    reference_operation,
    validated,
)
from unified_kg_rag.adapters.retrievers.graph_expansion import edge_predicate
from unified_kg_rag.adapters.retrievers.sqlite_graph_retriever import vertex_filter
from unified_kg_rag.domain.models import Entity
from unified_kg_rag.domain.models.graph_query import (
    GraphQueryPlan,
    GraphQueryRequest,
    GroupRow,
    PropertyRef,
    QueryAnswer,
    ResolvedCondition,
)
from unified_kg_rag.domain.models.schema import ExtractionSchema
from unified_kg_rag.domain.retrieval.graph_query import (
    aggregate_bindings,
    answer_from_result,
    build_profile,
    condition_holds,
    describe_plan,
    hydrate_keys,
    is_queryable_key,
    merge_declared,
    plan_operation,
    reference_execute,
    scalar_holds,
    validate_plan,
)
from unified_kg_rag.shared.exceptions import InvalidGraphQueryError

pytestmark = pytest.mark.unit


def _plan(**fields: Any) -> GraphQueryPlan:
    return GraphQueryPlan.model_validate(fields)


def _answer(name: str) -> QueryAnswer:
    return reference_execute(validated(CONFORMANCE_PLANS[name]), procurement_graph())


def _names(rows: tuple[dict[str, str], ...], var: str) -> list[str]:
    graph = procurement_graph()
    return [graph.vertices[row[var]]["name"][0] for row in rows]


# --- grammar -------------------------------------------------------------------


def test_the_grammar_is_closed() -> None:
    base = {"nodes": [{"var": "v"}], "returns": {"kind": "count"}}
    # No field names a label, a suffix, an index or an id.
    for extra in (
        {"label": "Entity-other"},
        {"suffix": "other"},
        {"index": "x"},
        {"query": "g.V().drop()"},
    ):
        with pytest.raises(ValidationError):
            GraphQueryPlan.model_validate({**base, **extra})
    with pytest.raises(ValidationError):
        GraphQueryPlan.model_validate(
            {"nodes": [{"var": "v", "label": "Entity-x"}], "returns": {"kind": "count"}}
        )
    with pytest.raises(ValidationError):
        GraphQueryPlan.model_validate(
            {"nodes": [{"var": "V-1"}], "returns": {"kind": "count"}}
        )


def test_literal_values_are_not_coerced() -> None:
    plan = _plan(
        nodes=[{"var": "v", "where": [{"prop": "rank", "op": "eq", "value": "5"}]}],
        returns={"kind": "count"},
    )
    assert plan.nodes[0].where[0].value == "5"
    true = _plan(
        nodes=[
            {"var": "v", "where": [{"prop": "in_schema", "op": "eq", "value": True}]}
        ],
        returns={"kind": "count"},
    )
    assert true.nodes[0].where[0].value is True
    with pytest.raises(ValidationError):
        _plan(
            nodes=[
                {
                    "var": "v",
                    "where": [{"prop": "rank", "op": "eq", "value": {"$gt": 1}}],
                }
            ],
            returns={"kind": "count"},
        )
    with pytest.raises(ValidationError):
        _plan(
            nodes=[
                {
                    "var": "v",
                    "where": [{"prop": "rank", "op": "gt", "value": float("nan")}],
                }
            ],
            returns={"kind": "count"},
        )


@pytest.mark.parametrize(
    "condition",
    [
        {"prop": "name", "op": "in", "value": "x"},
        {"prop": "name", "op": "in", "value": []},
        {"prop": "name", "op": "eq"},
        {"prop": "name", "op": "exists", "value": "yes"},
        {"prop": "name", "op": "contains", "value": 3},
        {"prop": "name", "op": "eq", "value": "x" * 257},
        {"prop": "name", "op": "in", "value": ["x"] * 51},
    ],
)
def test_a_condition_takes_the_value_its_operator_needs(condition: dict) -> None:
    with pytest.raises(ValidationError):
        _plan(nodes=[{"var": "v", "where": [condition]}], returns={"kind": "count"})


@pytest.mark.parametrize("hops", [[0, 1], [2, 1], [1, 5], [1]])
def test_hops_are_a_bounded_range(hops: list[int]) -> None:
    with pytest.raises(ValidationError):
        _plan(
            nodes=[{"var": "a"}, {"var": "b"}],
            edges=[{"hops": hops}],
            returns={"kind": "count"},
        )


def test_the_caps_hold() -> None:
    with pytest.raises(ValidationError):
        _plan(nodes=[{"var": f"n{i}"} for i in range(5)], returns={"kind": "count"})
    with pytest.raises(ValidationError):
        _plan(
            nodes=[
                {"var": "v", "where": [{"prop": "rank", "op": "gt", "value": 1}] * 7}
            ],
            returns={"kind": "count"},
        )
    with pytest.raises(ValidationError):
        _plan(nodes=[{"var": "v"}], returns={"kind": "rows", "limit": 201})


# --- validation ----------------------------------------------------------------


def test_types_resolve_to_every_stored_spelling() -> None:
    profile = build_profile(
        [
            {"id": ["a"], "type": ["VENDOR"]},
            {"id": ["b"], "type": ["vendor"]},
            {"id": ["c"], "type": ["Buyer"]},
        ],
        [("VENDOR", "supplies", "Buyer"), ("vendor", "SUPPLIES", "Buyer")],
        sampled=False,
    )
    plan = validate_plan(
        _plan(
            nodes=[{"var": "v", "types": ["Vendor"]}, {"var": "b", "types": ["BUYER"]}],
            edges=[{"types": ["Supplies"]}],
            returns={"kind": "count"},
        ),
        profile,
        max_hops=3,
    )
    assert set(plan.nodes[0].types) == {"VENDOR", "vendor"}
    assert plan.nodes[1].types == ("Buyer",)
    assert set(plan.edges[0].labels) == {"supplies", "SUPPLIES"}


def test_an_unknown_name_is_refused_naming_the_field_and_the_choices() -> None:
    with pytest.raises(InvalidGraphQueryError) as error:
        validated(
            {
                "nodes": [{"var": "v", "types": ["SUPPLIER"]}],
                "returns": {"kind": "count"},
            }
        )
    assert error.value.field == "nodes[0].types[0]"
    assert "VENDOR" in error.value.reason and "BUYER" in error.value.reason
    with pytest.raises(InvalidGraphQueryError) as error:
        validated(
            {
                "nodes": [{"var": "a"}, {"var": "b"}],
                "edges": [{"types": ["OWNS"]}],
                "returns": {"kind": "count"},
            }
        )
    assert error.value.field == "edges[0].types[0]"
    assert "SUPPLIES" in error.value.reason
    with pytest.raises(InvalidGraphQueryError, match="unknown property 'revenue'"):
        validated(
            {
                "nodes": [
                    {"var": "v", "where": [{"prop": "revenue", "op": "gt", "value": 1}]}
                ],
                "returns": {"kind": "count"},
            }
        )


@pytest.mark.parametrize(
    "prop",
    ["id", "text_unit_ids", "community_ids", "name_key", "attr.index", "attr.filters"],
)
def test_ids_lineage_and_internals_are_never_queryable(prop: str) -> None:
    with pytest.raises(InvalidGraphQueryError):
        validated(
            {
                "nodes": [{"var": "v", "where": [{"prop": prop, "op": "exists"}]}],
                "returns": {"kind": "count"},
            }
        )


@pytest.mark.parametrize(
    ("key", "queryable"),
    [
        ("name", True),
        ("rank", True),
        ("attr_role", True),
        # Neither a core property nor an extracted attribute.
        ("role", False),
        ("label", False),
        ("attrs_role", False),
        # An internal attribute ("_" first) and the store's bookkeeping.
        ("attr__internal", False),
        ("attr_", True),
        ("build_tag", False),
        ("attr_text_unit_weights", False),
    ],
)
def test_only_core_properties_and_extracted_attributes_are_plan_keys(
    key: str, queryable: bool
) -> None:
    assert is_queryable_key(key) is queryable
    profile = build_profile(
        [{"id": ["v"], "type": ["VENDOR"], key: ["x"]}], [], sampled=False
    )
    listed = {p.key for p in profile.properties}
    assert (key in listed) is queryable


@pytest.mark.parametrize(
    "where",
    [
        {"prop": "attr.code", "op": "eq", "value": "~id"},
        {"prop": "attr.code", "op": "in", "value": ["A1", "~label"]},
    ],
)
def test_a_value_naming_a_hidden_token_is_refused_with_a_warning(
    where: dict, caplog: pytest.LogCaptureFixture
) -> None:
    graph = procurement_graph()
    for vertex in graph.vertices.values():
        vertex["attr_code"] = ["A1"]
    plan = _plan(nodes=[{"var": "v", "where": [where]}], returns={"kind": "count"})
    with caplog.at_level("WARNING"), pytest.raises(InvalidGraphQueryError, match="~"):
        validate_plan(plan, profile_of(graph), max_hops=3)
    assert "hidden token" in caplog.text


def test_a_type_naming_a_hidden_token_is_refused_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stored type or relationship type starting with "~" reached the
    compiled Gremlin as a hidden token, refused there as a compiler bug."""
    entities = [
        Entity(id="t1", name="T1", type="~VENDOR"),
        Entity(id="t2", name="T2", type="BUYER"),
    ]
    graph = property_graph(entities, [])
    profile = build_profile(
        graph.vertices.values(), [("~VENDOR", "~SUPPLIES", "BUYER")], sampled=False
    )
    for raw, field in [
        ({"nodes": [{"var": "v", "types": ["~VENDOR"]}]}, "nodes[0].types"),
        (
            {
                "nodes": [{"var": "v"}, {"var": "b", "types": ["BUYER"]}],
                "edges": [{"types": ["~SUPPLIES"]}],
            },
            "edges[0].types",
        ),
    ]:
        caplog.clear()
        plan = _plan(**raw, returns={"kind": "count"})
        with caplog.at_level("WARNING"), pytest.raises(InvalidGraphQueryError) as e:
            validate_plan(plan, profile, max_hops=3)
        assert e.value.field == field
        assert "hidden token" in caplog.text


def test_the_profile_lists_attributes_and_seed_properties_by_namespace() -> None:
    names = {p.name: p for p in profile_of(procurement_graph()).properties}
    assert names["attr.role"].examples == ("supplier",)
    assert names["seed.country"].kinds == ("string",)
    assert set(names["seed.country"].examples) == {"Norland", "Sudmark"}
    assert names["seed.employees"].kinds == ("number",)
    assert names["source_kinds"].multi
    assert "attr.index" not in names and "attr.filters" not in names
    # A bare name resolves when one attribute or seed property has it.
    plan = validated(
        {
            "nodes": [
                {
                    "var": "v",
                    "where": [{"prop": "country", "op": "eq", "value": "Norland"}],
                }
            ],
            "returns": {"kind": "count"},
        }
    )
    assert plan.nodes[0].conditions[0].prop.name == "seed.country"
    assert plan.reads_seed_properties


def test_values_must_be_of_the_kind_the_property_holds() -> None:
    with pytest.raises(InvalidGraphQueryError, match="number"):
        validated(
            {
                "nodes": [
                    {
                        "var": "v",
                        "where": [{"prop": "rank", "op": "eq", "value": "high"}],
                    }
                ],
                "returns": {"kind": "count"},
            }
        )
    with pytest.raises(InvalidGraphQueryError, match="booleans"):
        validated(
            {
                "nodes": [
                    {
                        "var": "v",
                        "where": [{"prop": "in_schema", "op": "gt", "value": True}],
                    }
                ],
                "returns": {"kind": "count"},
            }
        )


@pytest.mark.parametrize(
    "plan,field",
    [
        (
            {"nodes": [{"var": "a"}, {"var": "b"}], "returns": {"kind": "count"}},
            "edges",
        ),
        (
            {
                "nodes": [{"var": "a"}, {"var": "a"}],
                "edges": [{}],
                "returns": {"kind": "count"},
            },
            "nodes[1].var",
        ),
        (
            {
                "nodes": [{"var": "a"}, {"var": "b"}, {"var": "c"}],
                "edges": [{"hops": [1, 2]}, {"hops": [1, 2]}],
                "returns": {"kind": "count"},
            },
            "edges",
        ),
        (
            {"nodes": [{"var": "a"}], "returns": {"kind": "rows", "vars": ["b"]}},
            "returns.vars[0]",
        ),
        (
            {"nodes": [{"var": "a"}], "returns": {"kind": "group_count"}},
            "returns.group_by",
        ),
        (
            {
                "nodes": [{"var": "a"}],
                "returns": {
                    "kind": "group_count",
                    "group_by": {"var": "a", "prop": "source_kinds"},
                },
            },
            "returns.group_by.prop",
        ),
        (
            {
                "nodes": [{"var": "a"}],
                "returns": {
                    "kind": "aggregate",
                    "aggregate": {"fn": "max", "var": "a", "prop": "name"},
                },
            },
            "returns.aggregate.prop",
        ),
        ({"nodes": [{"var": "a"}], "returns": {"kind": "path"}}, "returns.kind"),
        ({"nodes": [{"var": "a"}]}, "returns"),
        ({"nodes": [], "returns": {"kind": "count"}}, "nodes"),
    ],
)
def test_an_inconsistent_plan_names_its_field(plan: dict, field: str) -> None:
    with pytest.raises(InvalidGraphQueryError) as error:
        validated(plan, max_hops=3)
    assert error.value.field == field


def test_declared_types_are_allowed_when_the_sample_missed_them() -> None:
    schema = ExtractionSchema.from_items(
        entity_types=["WAREHOUSE: A storage site"],
        relationship_types=["STORES"],
    )
    profile = merge_declared(profile_of(procurement_graph()), schema)
    plan = validate_plan(
        _plan(
            nodes=[{"var": "w", "types": ["warehouse"]}, {"var": "p"}],
            edges=[{"types": ["stores"]}],
            returns={"kind": "count"},
        ),
        profile,
        max_hops=3,
    )
    assert "WAREHOUSE" in plan.nodes[0].types and "warehouse" in plan.nodes[0].types
    declared = {t.label: t for t in profile.entity_types}
    assert declared["WAREHOUSE"].declared and declared["WAREHOUSE"].count == 0
    assert declared["VENDOR"].count == 4


# --- rendering and operations ------------------------------------------------------


def test_the_plan_is_described_by_code() -> None:
    plan = validated(CONFORMANCE_PLANS["buyers_per_vendor"])
    text = describe_plan(plan)
    assert text.startswith("Match (v: VENDOR) -[SUPPLIES]-> (b: BUYER).")
    assert "distinct b per v, most first, top 3" in text
    mention = validated(
        {
            "nodes": [
                {"var": "x", "mention": "가나다 상사"},
                {"var": "p", "types": ["PRODUCT"]},
            ],
            "edges": [{"types": ["MAKES"], "hops": [1, 2]}],
            "returns": {"kind": "rows", "vars": ["p"]},
        }
    )
    text = describe_plan(mention, {"x": ["가나다 상사"]})
    # The stored names a mention linked to are shown, so the answer sees
    # which entities were counted.
    assert '(x, named "가나다 상사" = "가나다 상사")' in text
    assert "(linked to nothing)" in describe_plan(mention, {})
    assert "-[MAKES *1..2]->" in text


def test_the_store_aggregates_unless_code_must_check_the_bindings() -> None:
    count = validated(CONFORMANCE_PLANS["count_supply_pairs"])
    assert plan_operation(count, scan=False, cap=10).operation == "count"
    scanned = plan_operation(count, scan=True, cap=10)
    assert (scanned.operation, scanned.vars, scanned.limit) == (
        "tuples",
        ("v", "b"),
        10,
    )
    seed = validated(CONFORMANCE_PLANS["seed_country_count"])
    assert plan_operation(seed, scan=False, cap=10).operation == "tuples"
    group = plan_operation(
        validated(CONFORMANCE_PLANS["types_by_count"]), scan=False, cap=7
    )
    assert (group.operation, group.group_key, group.limit) == ("group", "type", 7)
    path = plan_operation(
        validated(CONFORMANCE_PLANS["paths_between"]), scan=True, cap=5
    )
    assert path.operation == "paths"
    exists = plan_operation(
        validated(CONFORMANCE_PLANS["exists_supply"]), scan=False, cap=5
    )
    assert (exists.operation, exists.limit) == ("tuples", 1)
    assert "attr_seed_properties" in hydrate_keys(seed)
    assert "attr_seed_properties" not in hydrate_keys(count)


# --- answers ---------------------------------------------------------------------


def test_reference_answers_on_the_procurement_graph() -> None:
    assert _answer("count_vendors_supplying").count == 3
    assert _answer("count_supply_pairs").count == 6
    groups = _answer("buyers_per_vendor").groups
    assert groups == (
        GroupRow(key="v3", count=3),
        GroupRow(key="v1", count=2),
        GroupRow(key="v2", count=1),
    )
    assert _answer("types_by_count").groups == (
        GroupRow(key="VENDOR", count=4),
        GroupRow(key="BUYER", count=3),
        GroupRow(key="PRODUCT", count=3),
    )
    assert _names(_answer("products_of_suppliers_of_buyer").rows, "p") == [
        "Valve",
        "Pump",
    ]
    assert _names(_answer("two_hop_any").rows, "z") == ["Valve", "Gasket"]
    assert _answer("exists_supply").exists is False
    assert _answer("max_rank_of_buyers").value == 4
    paths = _answer("paths_between").paths
    assert [p.vertex_ids for p in paths] == [
        ("v2", "b1", "v1", "b2"),
        ("v2", "b1", "v3", "b2"),
    ]
    assert [e.id for e in paths[0].edges] == ["r3", "r1", "r2"]
    assert _names(_answer("korean_vendor_products").rows, "p") == ["Valve"]
    assert _names(_answer("role_attribute").rows, "v") == ["Acme Supplies"]
    assert _answer("seed_country_count").count == 3
    assert _answer("rank_range").count == 5


@pytest.mark.parametrize("name", sorted(CONFORMANCE_PLANS))
def test_the_store_operation_answers_as_the_scan_does(name: str) -> None:
    """``answer_from_result`` over the operation a store runs equals the
    answer computed in code from every binding: the in-store aggregation
    is a recount, never another semantics."""
    plan = validated(CONFORMANCE_PLANS[name])
    graph = procurement_graph()
    op = plan_operation(plan, scan=False, cap=1000)
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
        rows = result.rows if op.operation == "tuples" else ()
        paths = result.paths
        bindings = rows or tuple(
            {"a": p.vertex_ids[0], "z": p.vertex_ids[-1]} for p in paths
        )
        answer = aggregate_bindings(plan, bindings, result.vertices, paths=paths)
    else:
        answer = answer_from_result(plan, result)
    assert answer == reference_execute(plan, graph)


def test_the_caller_filters_apply_to_bound_vertices_and_every_edge() -> None:
    plan = validated(CONFORMANCE_PLANS["count_supply_pairs"])
    graph = procurement_graph()
    filters = {"source_kinds": "document"}
    keep = vertex_filter(filters)
    edge_ok = edge_predicate(filters)
    assert edge_ok is not None

    request = GraphQueryRequest(plan=plan, operation="count", vars=plan.vars)
    # The seed-only edge v3 -> b1 is left out.
    assert reference_operation(request, graph, edge_ok=edge_ok).count == 5
    only_seed = vertex_filter({"source_kinds": "seed_graph"})
    assert reference_operation(request, graph, vertex_ok=only_seed).count == 0
    assert keep is not None
    assert reference_operation(request, graph, vertex_ok=keep).count == 6
    # The caller's id filter keeps only the vertices it names.
    allowed = request.model_copy(update={"allowed_ids": ("v1", "b1", "b2")})
    assert reference_operation(allowed, graph).count == 2


def test_a_scan_past_its_cap_is_a_lower_bound() -> None:
    plan = validated(CONFORMANCE_PLANS["count_supply_pairs"])
    request = GraphQueryRequest(
        plan=plan,
        operation="tuples",
        vars=plan.vars,
        limit=4,
        hydrate=hydrate_keys(plan),
    )
    result = reference_operation(request, procurement_graph())
    assert result.truncated and len(result.rows) == 4
    answer = aggregate_bindings(plan, result.rows, result.vertices, truncated=True)
    assert answer.count == 4 and answer.truncated


def test_linked_mentions_bind_only_their_entities() -> None:
    plan = validated(
        {
            "nodes": [
                {"var": "x", "mention": "Acme"},
                {"var": "b", "types": ["BUYER"]},
            ],
            "edges": [{"types": ["SUPPLIES"]}],
            "returns": {"kind": "rows", "vars": ["b"]},
        }
    )
    request = GraphQueryRequest(
        plan=plan,
        operation="tuples",
        vars=("b",),
        limit=10,
        linked={"x": ("v1",)},
        hydrate=hydrate_keys(plan),
    )
    result = reference_operation(request, procurement_graph())
    assert sorted(row["b"] for row in result.rows) == ["b1", "b2"]
    # A mention linked to nothing matches nothing.
    unlinked = request.model_copy(update={"linked": {}})
    assert reference_operation(unlinked, procurement_graph()).rows == ()


def test_suffixes_are_separate_graphs() -> None:
    """The executor reads one label's graph: another suffix's entities are
    another ``PropertyGraph`` (the stores' label), never mixed in."""
    other = property_graph(
        procurement_entities("other"), procurement_relationships("other")
    )
    assert other.vertices.keys() == procurement_graph().vertices.keys()
    assert other.vertices["v1"]["attr_index"] == ["other"]


def test_an_empty_answer_is_one_without_rows_but_never_a_count() -> None:
    assert QueryAnswer(kind="rows").empty
    assert QueryAnswer(kind="aggregate").empty
    assert not QueryAnswer(kind="count", count=0).empty
    assert not QueryAnswer(kind="exists", exists=False).empty


def test_text_and_order_comparisons_need_one_value_kind() -> None:
    """A property holding strings and numbers: Gremlin fails the query on a
    substring test of a number, so the plan is refused before it runs."""
    graph = property_graph(
        [
            Entity(id="a", name="A", type="VENDOR", attributes={"score": 3}),
            Entity(id="b", name="B", type="VENDOR", attributes={"score": "s4"}),
        ],
        [],
    )
    profile = profile_of(graph)
    for op, value in (("contains", "s"), ("gt", 2)):
        with pytest.raises(InvalidGraphQueryError, match="attr.score"):
            validate_plan(
                _plan(
                    nodes=[
                        {
                            "var": "v",
                            "where": [{"prop": "attr.score", "op": op, "value": value}],
                        }
                    ],
                    returns={"kind": "count"},
                ),
                profile,
                max_hops=3,
            )
    # Equality across kinds stays allowed (no store fails it).
    validate_plan(
        _plan(
            nodes=[
                {
                    "var": "v",
                    "where": [{"prop": "attr.score", "op": "eq", "value": "s4"}],
                }
            ],
            returns={"kind": "count"},
        ),
        profile,
        max_hops=3,
    )


@pytest.mark.parametrize(
    ("stored", "op", "value", "holds"),
    [
        # eq: within one kind; a number never equals a boolean.
        (5, "eq", 5, True),
        (5, "eq", 5.0, True),
        (5, "eq", 6, False),
        (1, "eq", True, False),
        (True, "eq", 1, False),
        ("Acme", "eq", "Acme", True),
        ("Acme", "eq", "acme", False),
        ("5", "eq", 5, False),
        # neq holds across kinds, as Gremlin's does.
        (5, "neq", 5, False),
        (5, "neq", 6, True),
        (True, "neq", 1, True),
        ("Acme", "neq", 5, True),
        ("Acme", "neq", "Acme", False),
        # in: any listed value of the same kind.
        (5, "in", (1, 5), True),
        (5, "in", (1, 2), False),
        (1, "in", (True,), False),
        ("b", "in", ("a", "b"), True),
        # contains: case-sensitive, strings only.
        ("Acme Supplies", "contains", "Supp", True),
        ("Acme Supplies", "contains", "supp", False),
        (55, "contains", "5", False),
        # Order comparisons: within one kind, never booleans.
        (5, "gt", 4, True),
        (5, "gt", 5, False),
        (5, "gte", 5, True),
        (4, "gte", 5, False),
        (4, "lt", 5, True),
        (5, "lt", 5, False),
        (5, "lte", 5, True),
        (6, "lte", 5, False),
        (5.5, "gt", 5, True),
        ("b", "gt", "a", True),
        ("a", "gte", "b", False),
        (5, "gt", "4", False),
        (True, "gt", False, False),
        (True, "lte", True, False),
        (5, "exists", True, False),
    ],
)
def test_one_stored_value_against_each_operator(
    stored: Any, op: str, value: Any, holds: bool
) -> None:
    """The comparison semantics the reference and SQLite share: each
    operator pinned on its own, so a store answering alike from the same
    wrong predicate cannot hide it (a neq read as eq, a gt as gte)."""
    assert scalar_holds(stored, op, value) is holds


@pytest.mark.parametrize(
    ("values", "op", "value", "holds"),
    [
        ([], "exists", True, False),
        ([], "exists", False, True),
        ([None], "exists", True, False),
        ([None], "exists", False, True),
        ([0], "exists", True, True),
        ([0], "exists", False, False),
        ([False], "exists", True, True),
        # Any stored value; no value fails every operator but exists: false.
        (["a", "b"], "eq", "b", True),
        ([1, 9], "gt", 5, True),
        ([1, 2], "gt", 5, False),
        ([], "neq", "a", False),
        ([None], "eq", None, False),
    ],
)
def test_a_property_holds_a_condition_by_any_of_its_values(
    values: list[Any], op: str, value: Any, holds: bool
) -> None:
    prop = PropertyRef(name="p", key="p", source="vertex", kinds=(), multi=True)
    condition = ResolvedCondition(prop=prop, op=op, value=value)
    assert condition_holds(values, condition) is holds
