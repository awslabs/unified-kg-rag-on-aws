# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A synthetic procurement graph for the structured-query tests.

Vendors supply buyers and make products, buyers buy products; some names
are Korean and Japanese, some vertices carry seed properties and an
extracted attribute, one edge only a seed graph states. Everything is
made up. :func:`property_graph` encodes it with the real Neptune codec
(what every graph backend stores), so the reference executor reads what
the stores hold.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from unified_kg_rag.adapters.storage.neptune_codec import (
    entity_properties,
    relationship_label,
    relationship_properties,
)
from unified_kg_rag.domain.models import Entity, Relationship
from unified_kg_rag.domain.models.graph_query import (
    GraphEdgeRecord,
    GraphProfile,
    GraphQueryPlan,
    PropertyGraph,
    ValidatedPlan,
)
from unified_kg_rag.domain.retrieval.graph_query import (
    build_profile,
    validate_plan,
)

MAX_LENGTH = 1000


def _entity(
    entity_id: str,
    name: str,
    kind: str,
    *,
    suffix: str | None = None,
    rank: int = 1,
    seed: dict[str, Any] | None = None,
    role: str | None = None,
    kinds: list[str] | None = None,
    units: list[str] | None = None,
) -> Entity:
    attributes: dict[str, Any] = {}
    if suffix:
        attributes["index"] = suffix
    if seed:
        attributes["seed_properties"] = seed
        attributes["filters"] = {"seed_graph": ["procure"]}
    if role:
        attributes["role"] = role
    return Entity(
        id=entity_id,
        name=name,
        type=kind,
        description=f"{name} is a {kind.lower()}.",
        rank=rank,
        text_unit_ids=units or [f"tu-{entity_id}"],
        attributes=attributes or None,
        source_kinds=kinds,
    )


def _rel(
    rel_id: str,
    source: str,
    target: str,
    kind: str,
    *,
    suffix: str | None = None,
    kinds: list[str] | None = None,
) -> Relationship:
    return Relationship(
        id=rel_id,
        source_id=source,
        target_id=target,
        type=kind,
        description=f"{source} {kind.lower()} {target}",
        text_unit_ids=[f"tu-{rel_id}"],
        attributes={"index": suffix} if suffix else None,
        source_kinds=kinds,
    )


def procurement_entities(suffix: str | None = None) -> list[Entity]:
    """Four vendors (one Korean), three buyers (one Japanese), three
    products."""
    seed = ["document", "seed_graph"]
    return [
        _entity(
            "v1",
            "Acme Supplies",
            "VENDOR",
            suffix=suffix,
            rank=5,
            seed={"country": "Norland", "employees": 120},
            role="supplier",
            kinds=seed,
        ),
        _entity(
            "v2",
            "Globex Parts",
            "VENDOR",
            suffix=suffix,
            rank=4,
            seed={"country": "Sudmark", "employees": 45},
            kinds=seed,
        ),
        _entity(
            "v3",
            "가나다 상사",
            "VENDOR",
            suffix=suffix,
            rank=3,
            seed={"country": "Norland", "employees": 300},
            kinds=seed,
        ),
        _entity("v4", "Initech Tools", "VENDOR", suffix=suffix, rank=1),
        _entity("b1", "Umbrella Buyer", "BUYER", suffix=suffix, rank=4),
        _entity("b2", "株式会社サクラ", "BUYER", suffix=suffix, rank=3),
        _entity("b3", "Hooli Retail", "BUYER", suffix=suffix, rank=2),
        _entity("p1", "Valve", "PRODUCT", suffix=suffix, rank=3),
        _entity("p2", "Pump", "PRODUCT", suffix=suffix, rank=2),
        _entity("p3", "Gasket", "PRODUCT", suffix=suffix, rank=1),
    ]


def procurement_relationships(suffix: str | None = None) -> list[Relationship]:
    return [
        _rel("r1", "v1", "b1", "SUPPLIES", suffix=suffix),
        _rel("r2", "v1", "b2", "SUPPLIES", suffix=suffix),
        _rel("r3", "v2", "b1", "SUPPLIES", suffix=suffix),
        _rel("r4", "v3", "b2", "SUPPLIES", suffix=suffix),
        _rel("r5", "v3", "b3", "SUPPLIES", suffix=suffix),
        _rel("r6", "v3", "b1", "SUPPLIES", suffix=suffix, kinds=["seed_graph"]),
        _rel("r7", "v1", "p1", "MAKES", suffix=suffix),
        _rel("r8", "v2", "p2", "MAKES", suffix=suffix),
        _rel("r9", "v3", "p1", "MAKES", suffix=suffix),
        _rel("r10", "b1", "p1", "BUYS", suffix=suffix),
        _rel("r11", "b2", "p3", "BUYS", suffix=suffix),
        _rel("r12", "v4", "p3", "MAKES", suffix=suffix),
    ]


def property_graph(
    entities: Iterable[Entity], relationships: Iterable[Relationship]
) -> PropertyGraph:
    """What a graph store holds of ``entities`` and ``relationships`` (the
    Neptune codec's encoding; vertex values as lists)."""
    vertices: dict[str, dict[str, list[Any]]] = {}
    for entity in entities:
        value_map: dict[str, list[Any]] = {"id": [entity.id]}
        for key, value in entity_properties(entity, MAX_LENGTH).items():
            values = value if isinstance(value, list) else [value]
            value_map[key] = list(dict.fromkeys(values))
        vertices[entity.id] = value_map
    edges = [
        GraphEdgeRecord(
            id=rel.id,
            type=relationship_label(rel),
            source_id=rel.source_id,
            target_id=rel.target_id,
            props={"id": rel.id, **relationship_properties(rel, MAX_LENGTH)},
        )
        for rel in relationships
        if rel.source_id in vertices and rel.target_id in vertices
    ]
    return PropertyGraph(vertices=vertices, edges=edges)


def procurement_graph() -> PropertyGraph:
    return property_graph(procurement_entities(), procurement_relationships())


def profile_of(graph: PropertyGraph) -> GraphProfile:
    types = {
        vertex_id: (vertex.get("type") or [""])[0]
        for vertex_id, vertex in graph.vertices.items()
    }
    return build_profile(
        graph.vertices.values(),
        [(types[e.source_id], e.type, types[e.target_id]) for e in graph.edges],
        sampled=False,
    )


def validated(
    plan: dict[str, Any] | GraphQueryPlan, max_hops: int = 4
) -> ValidatedPlan:
    """``plan`` validated against the procurement graph's profile."""
    parsed = (
        plan
        if isinstance(plan, GraphQueryPlan)
        else GraphQueryPlan.model_validate(plan)
    )
    return validate_plan(parsed, profile_of(procurement_graph()), max_hops=max_hops)


# Plans every executor answers alike (the conformance suite's cases), with
# the answers over the procurement graph.
CONFORMANCE_PLANS: dict[str, dict[str, Any]] = {
    "count_vendors_supplying": {
        "nodes": [{"var": "v", "types": ["VENDOR"]}, {"var": "b", "types": ["BUYER"]}],
        "edges": [{"types": ["SUPPLIES"]}],
        "returns": {"kind": "count_distinct", "vars": ["v"]},
    },
    "count_supply_pairs": {
        "nodes": [{"var": "v", "types": ["vendor"]}, {"var": "b", "types": ["BUYER"]}],
        "edges": [{"types": ["supplies"]}],
        "returns": {"kind": "count"},
    },
    "buyers_per_vendor": {
        "nodes": [{"var": "v", "types": ["VENDOR"]}, {"var": "b", "types": ["BUYER"]}],
        "edges": [{"types": ["SUPPLIES"]}],
        "returns": {
            "kind": "group_count",
            "group_by": {"var": "v"},
            "vars": ["b"],
            "limit": 3,
        },
    },
    "types_by_count": {
        "nodes": [{"var": "x"}],
        "returns": {"kind": "group_count", "group_by": {"var": "x", "prop": "type"}},
    },
    "products_of_suppliers_of_buyer": {
        "nodes": [
            {
                "var": "b",
                "types": ["BUYER"],
                "where": [{"prop": "name", "op": "eq", "value": "Umbrella Buyer"}],
            },
            {"var": "v", "types": ["VENDOR"]},
            {"var": "p", "types": ["PRODUCT"]},
        ],
        "edges": [{"types": ["SUPPLIES"], "direction": "in"}, {"types": ["MAKES"]}],
        "returns": {"kind": "rows", "vars": ["p"]},
    },
    "two_hop_any": {
        "nodes": [
            {
                "var": "a",
                "where": [{"prop": "name", "op": "eq", "value": "Hooli Retail"}],
            },
            {"var": "z", "types": ["PRODUCT"]},
        ],
        "edges": [{"direction": "both", "hops": [1, 3]}],
        "returns": {"kind": "rows", "vars": ["z"]},
    },
    # A fixed length of two hops: one chain, no union (Neptune rejects a
    # one-branch union followed by simplePath()).
    "two_hop_exact": {
        "nodes": [
            {
                "var": "a",
                "where": [{"prop": "name", "op": "eq", "value": "Hooli Retail"}],
            },
            {"var": "z", "types": ["PRODUCT"]},
        ],
        "edges": [{"direction": "both", "hops": [2, 2]}],
        "returns": {"kind": "rows", "vars": ["z"]},
    },
    "exists_supply": {
        "nodes": [
            {
                "var": "v",
                "where": [{"prop": "name", "op": "eq", "value": "Initech Tools"}],
            },
            {"var": "b", "types": ["BUYER"]},
        ],
        "edges": [{"types": ["SUPPLIES"]}],
        "returns": {"kind": "exists"},
    },
    "max_rank_of_buyers": {
        "nodes": [{"var": "b", "types": ["BUYER"]}],
        "returns": {
            "kind": "aggregate",
            "aggregate": {"fn": "max", "var": "b", "prop": "rank"},
        },
    },
    "paths_between": {
        "nodes": [
            {
                "var": "a",
                "where": [{"prop": "name", "op": "eq", "value": "Globex Parts"}],
            },
            {
                "var": "z",
                "where": [{"prop": "name", "op": "eq", "value": "株式会社サクラ"}],
            },
        ],
        "edges": [{"direction": "both", "hops": [1, 3]}],
        "returns": {"kind": "path", "limit": 50},
    },
    "korean_vendor_products": {
        "nodes": [
            {
                "var": "v",
                "where": [{"prop": "name", "op": "contains", "value": "가나다"}],
            },
            {"var": "p", "types": ["PRODUCT"]},
        ],
        "edges": [{"types": ["MAKES"]}],
        "returns": {"kind": "rows", "vars": ["p"], "properties": ["rank"]},
    },
    "role_attribute": {
        "nodes": [
            {
                "var": "v",
                "where": [{"prop": "attr.role", "op": "eq", "value": "supplier"}],
            }
        ],
        "returns": {"kind": "rows"},
    },
    "seed_country_count": {
        "nodes": [
            {
                "var": "v",
                "types": ["VENDOR"],
                "where": [{"prop": "seed.country", "op": "eq", "value": "Norland"}],
            },
            {"var": "b", "types": ["BUYER"]},
        ],
        "edges": [{"types": ["SUPPLIES"]}],
        "returns": {"kind": "count_distinct", "vars": ["b"]},
    },
    "rank_range": {
        "nodes": [
            {
                "var": "x",
                "where": [
                    {"prop": "rank", "op": "gte", "value": 3},
                    {"prop": "type", "op": "in", "value": ["VENDOR", "BUYER"]},
                ],
            }
        ],
        "returns": {"kind": "count"},
    },
}
