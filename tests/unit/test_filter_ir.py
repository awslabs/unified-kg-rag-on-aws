# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The backend-neutral filter form: catalog, clauses and the three compilers.

The answers each store gives are pinned by
``tests/integration/test_filter_conformance.py``; this module pins the
pieces: how a key is classified, how a value becomes a clause, and that each
compiler refuses what its store cannot apply.
"""

from __future__ import annotations

from typing import Any

import pytest
from gremlin_python.process.graph_traversal import __

from unified_kg_rag.adapters.graph_query.gremlin import (
    gremlin_kinds,
    gremlin_vertex_filters,
)
from unified_kg_rag.adapters.retrievers.graph_expansion import vertex_clauses
from unified_kg_rag.adapters.retrievers.opensearch_retriever import (
    opensearch_filter_clauses,
)
from unified_kg_rag.adapters.retrievers.sqlite_graph_retriever import vertex_filter
from unified_kg_rag.domain.retrieval.filters import (
    GRAPH_ENTITY_FILTER_FIELDS,
    ITEM_FILTER_FIELDS,
    AnyOf,
    FieldKind,
    KindsRule,
    Range,
    Term,
    field_kind,
    kinds_rule,
    loose_values,
    own_attribute,
    parse_any_of,
    parse_filters,
)
from unified_kg_rag.shared import InvalidFilterError

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("key", "kind"),
    [
        ("id", FieldKind.KEYWORD),
        ("source_kind", FieldKind.KEYWORD),
        ("name", FieldKind.NAME),
        ("name.keyword", FieldKind.NAME),
        ("description", FieldKind.TEXT),
        ("translated_text_ja", FieldKind.TEXT),
        ("rank", FieldKind.NUMBER),
        ("level", FieldKind.INTEGER),
        ("in_schema", FieldKind.BOOLEAN),
        ("source_kinds", FieldKind.SOURCE_KINDS),
        ("attr_category", FieldKind.DOCUMENT),
        ("attributes.filters.category", FieldKind.DOCUMENT),
        ("attributes.seed_properties", FieldKind.UNINDEXED),
        ("attributes.seed_properties.color", FieldKind.UNINDEXED),
        ("attributes.seed_propertiesx", FieldKind.ATTRIBUTE),
        ("attributes.role", FieldKind.ATTRIBUTE),
        ("attributes.address.city", FieldKind.ATTRIBUTE),
    ],
)
def test_each_key_has_one_kind(key: str, kind: FieldKind) -> None:
    assert field_kind(key) is kind


def test_an_own_attribute_is_a_top_level_one() -> None:
    assert own_attribute("attributes.role") == "role"
    for key in (
        "attributes.address.city",
        "attributes.filters.category",
        "attributes.index",
        "attributes._weights",
        "attr_role",
    ):
        assert own_attribute(key) is None, key


def test_the_catalogs_declare_no_refused_key() -> None:
    for fields in (*ITEM_FILTER_FIELDS.values(), GRAPH_ENTITY_FILTER_FIELDS):
        for key in fields.fields:
            assert field_kind(key) not in (FieldKind.TEXT, FieldKind.UNINDEXED)
        assert not fields.declares("description")
        assert not fields.declares("attributes.seed_properties.color")
    entities = ITEM_FILTER_FIELDS["entities"]
    assert entities.declares("attributes.address.city")
    assert not GRAPH_ENTITY_FILTER_FIELDS.declares("attributes.address.city")
    assert not GRAPH_ENTITY_FILTER_FIELDS.declares("attributes.filters.category")
    assert GRAPH_ENTITY_FILTER_FIELDS.declares("attributes.role")


def test_parse_types_and_shapes_the_clauses() -> None:
    assert parse_filters(
        {
            "rank": {"gte": "2"},
            "type": ["org"],
            "in_schema": "no",
            "source_kinds": ["Document", "document"],
            "name": "가나다 상사",
        }
    ) == (
        Range("rank", (("gte", 2),), FieldKind.NUMBER),
        Term("type", ("org",), FieldKind.KEYWORD, listed=True),
        Term("in_schema", (False,), FieldKind.BOOLEAN),
        KindsRule(("document",)),
        Term("name", ("가나다 상사",), FieldKind.NAME),
    )
    assert parse_filters(None) == ()
    # A strategy's empty scope parses; a caller's is refused.
    assert parse_filters({"id": []}, allow_empty_lists=True) == (
        Term("id", (), FieldKind.KEYWORD, listed=True),
    )
    with pytest.raises(InvalidFilterError, match="empty list"):
        parse_filters({"id": []})


@pytest.mark.parametrize(
    "filters",
    [
        {"description": "x"},
        {"translated_text_ko": "x"},
        {"attributes.seed_properties.color": "red"},
        {"rank": {"bogus": 1}},
        {"in_schema": "maybe"},
        {"source_kinds": "web"},
    ],
)
def test_parse_refuses_what_no_store_applies(filters: dict[str, Any]) -> None:
    with pytest.raises(InvalidFilterError):
        parse_filters(filters)


def test_the_kinds_rule_is_one_rule_for_every_store() -> None:
    document = KindsRule(("document",))
    seed = KindsRule(("seed_graph",))
    assert document.passes(None) and document.passes(["document", "seed_graph"])
    assert not document.passes(["seed_graph"])
    assert not seed.passes(None) and seed.passes(["seed_graph"])
    assert document.narrows and seed.narrows
    assert not KindsRule(("document", "seed_graph")).narrows
    assert seed.encodings() == ['["seed_graph"]', '["document", "seed_graph"]']
    assert kinds_rule({"source_kinds": "SEED_GRAPH"}) == seed
    assert kinds_rule({"type": "org"}) is None
    # The OpenSearch, Gremlin and SQLite renderings of one rule.
    assert opensearch_filter_clauses([document]) == [
        {
            "bool": {
                "should": [
                    {"term": {"source_kinds": "document"}},
                    {"bool": {"must_not": [{"exists": {"field": "source_kinds"}}]}},
                ]
            }
        }
    ]
    assert (
        gremlin_kinds(document, ["document"]).bytecode
        == __.or_(
            __.has("source_kinds", "document"), __.hasNot("source_kinds")
        ).bytecode
    )
    keep = vertex_filter({"source_kinds": "document"})
    assert keep is not None
    assert keep({}) and not keep({"source_kinds": ["seed_graph"]})


def test_an_own_attribute_matches_either_spelling_on_a_graph() -> None:
    assert loose_values([1999]) == (1999, "1999")
    assert loose_values(["1999"]) == ("1999", 1999)
    assert loose_values(["2.5", True]) == ("2.5", 2.5, True, "true")
    assert loose_values(["vendor"]) == ("vendor",)
    (clause,) = vertex_clauses({"attributes.founded": "1999"})
    assert clause == Term(
        "attr_founded", ("1999", 1999), FieldKind.ATTRIBUTE, listed=True
    )
    keep = vertex_filter({"attributes.founded": "1999"})
    assert keep is not None
    assert keep({"attr_founded": [1999]}) and not keep({"attr_founded": [199]})
    # Gremlin's equality: a boolean is no number.
    keep = vertex_filter({"attributes.flag": 1})
    assert keep is not None and not keep({"attr_flag": [True]})


def test_vertex_clauses_name_the_vertex_properties() -> None:
    assert vertex_clauses({"id": ["a"], "name.keyword": "A", "rank": "2"}) == (
        Term("name", ("A",), FieldKind.NAME),
        Term("rank", (2,), FieldKind.NUMBER),
    )
    steps = gremlin_vertex_filters(
        __.identity(), vertex_clauses({"name": "A", "type": ["x", "y"]})
    ).bytecode.step_instructions
    assert [s[:2] for s in steps[1:]] == [["has", "name"], ["has", "type"]]


@pytest.mark.parametrize(
    "key", ["attr_category", "attributes.filters.category", "attributes.a.b"]
)
def test_a_graph_compiler_refuses_what_no_vertex_holds(key: str) -> None:
    # The graph's scoping drops these first; a compiler given one refuses
    # rather than matching every vertex or none.
    with pytest.raises(InvalidFilterError, match="graph vertex"):
        vertex_clauses({key: "x"})


def test_the_opensearch_compiler_refuses_text() -> None:
    with pytest.raises(InvalidFilterError):
        opensearch_filter_clauses([Term("description", ("x",), FieldKind.TEXT)])


def test_any_of_is_one_should_of_the_groups() -> None:
    clause = parse_any_of([{"source_id": ["a"]}, {"target_id": ["a"]}])
    assert clause == AnyOf(
        (
            (Term("source_id", ("a",), FieldKind.KEYWORD, listed=True),),
            (Term("target_id", ("a",), FieldKind.KEYWORD, listed=True),),
        )
    )
    assert opensearch_filter_clauses([clause]) == [
        {
            "bool": {
                "should": [
                    {"bool": {"filter": [{"terms": {"source_id": ["a"]}}]}},
                    {"bool": {"filter": [{"terms": {"target_id": ["a"]}}]}},
                ]
            }
        }
    ]
    assert parse_any_of(None) is None
    with pytest.raises(InvalidFilterError, match="empty"):
        parse_any_of([{}])
