# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One ``GraphIndexer`` contract for every graph backend.

Runs on the fake (the reference), the embedded SQLite store, Gremlin Server
(``LOCAL_STORES=1``) and Neptune (``-m aws``), the last two also with custom
vertex ids (``indexing.neptune.vertex_ids: custom``); see
``tests/fixtures/store_backends.py``. A difference between them is a bug in
one of them, whichever is right.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.store_backends import (
    CUSTOM_IDS,
    GRAPH_BACKENDS,
    StoreHarness,
    open_harness,
)
from unified_kg_rag.domain.ingestion.seed_graph import (
    annotate_sources,
    seed_unit_id,
)
from unified_kg_rag.domain.models import Community, Entity, Relationship
from unified_kg_rag.ports.indexer import StoredAnnotation
from unified_kg_rag.shared import NamespaceLeaseError

pytestmark = pytest.mark.integration


@pytest.fixture(params=GRAPH_BACKENDS)
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[StoreHarness]:
    yield from open_harness(request.param, tmp_path)


def _entity(entity_id: str, suffix: str | None = None, **fields: Any) -> Entity:
    attributes = dict(fields.pop("attributes", None) or {})
    if suffix:
        attributes["index"] = suffix
    fields.setdefault("name", entity_id.upper())
    return Entity(id=entity_id, attributes=attributes or None, **fields)


def _rel(
    rel_id: str, source: str, target: str, suffix: str | None = None, **fields: Any
) -> Relationship:
    attributes = dict(fields.pop("attributes", None) or {})
    if suffix:
        attributes["index"] = suffix
    return Relationship(
        id=rel_id,
        source_id=source,
        target_id=target,
        attributes=attributes or None,
        **fields,
    )


def _community(
    comm_id: str, members: list[str] | None, suffix: str | None = None
) -> Community:
    return Community(
        id=comm_id,
        name=comm_id,
        level="0",
        parent="-1",
        children=[],
        entity_ids=members,
        attributes={"index": suffix} if suffix else None,
    )


def _graph(h: StoreHarness, edges: list[tuple[str, str]], suffix: str | None = None):
    ids = sorted({end for edge in edges for end in edge})
    entities = [_entity(i, suffix) for i in ids]
    assert h.graph_indexer.upsert_entities(entities).failed_items == 0
    stats = h.graph_indexer.upsert_relationships(
        [_rel(f"{a}-{b}", a, b, suffix) for a, b in edges]
    )
    assert stats.failed_items == 0, stats.errors


def test_entity_and_relationship_round_trip_through_the_codec(
    harness: StoreHarness,
) -> None:
    graph = harness.graph_indexer
    entity = _entity(
        "e1",
        type="ORG",
        description="Vendor of widgets",
        rank=3,
        confidence=0.5,
        text_unit_ids=["t2", "t1"],
        community_ids=["c1"],
        attributes={"role": "vendor", "meta": {"k": 1}},
    )
    rel = _rel(
        "r1",
        "e1",
        "e2",
        type="SUPPLIES",
        source_name="E1",
        target_name="E2",
        weight=2.0,
        description="supplies",
        rank=4,
        text_unit_ids=["t1", "t1", "t3"],
        attributes={"meta": {"k": 2}},
    )
    assert graph.index_entities([entity, _entity("e2")]).failed_items == 0
    assert graph.index_relationships([rel]).successful_items == 1

    (read,) = graph.read_entities(["e1"])
    assert read.model_dump(exclude={"text_unit_ids"}) == entity.model_dump(
        exclude={"text_unit_ids"}
    )
    # Vertex lists are sets; edge lists are one JSON value (order and
    # duplicates survive).
    assert sorted(read.text_unit_ids or []) == ["t1", "t2"]
    assert graph.read_relationships(["r1"]) == [rel]


def test_list_properties_are_sets_and_an_upsert_replaces_them(
    harness: StoreHarness,
) -> None:
    graph = harness.graph_indexer
    graph.upsert_entities([_entity("e1", text_unit_ids=["a", "a", "b"])])
    (read,) = graph.read_entities(["e1"])
    assert sorted(read.text_unit_ids or []) == ["a", "b"]
    graph.upsert_entities([_entity("e1", text_unit_ids=["c"])])
    (read,) = graph.read_entities(["e1"])
    assert read.text_unit_ids == ["c"]


def test_a_hub_entity_keeps_every_text_unit(harness: StoreHarness) -> None:
    # Neptune writes a long set property in bounded requests (a hub's 5,000
    # values in one failed with QueryTooLargeException): every value lands,
    # and a rewrite replaces them all.
    graph = harness.graph_indexer
    units = [f"t{i:05d}" for i in range(2_500)]
    stats = graph.upsert_entities([_entity("hub", text_unit_ids=units)])
    assert stats.failed_items == 0, stats.errors
    (read,) = graph.read_entities(["hub"])
    assert sorted(read.text_unit_ids or []) == units
    stats = graph.index_entities([_entity("hub", text_unit_ids=units[:1_200])])
    assert stats.failed_items == 0, stats.errors
    (read,) = graph.read_entities(["hub"])
    assert sorted(read.text_unit_ids or []) == units[:1_200]


def test_upsert_is_idempotent_and_replaces_the_properties(
    harness: StoreHarness,
) -> None:
    graph = harness.graph_indexer
    entities = [_entity("a", description="kept", rank=2), _entity("b")]
    for _ in range(2):
        assert graph.upsert_entities(entities).failed_items == 0
        assert graph.upsert_relationships([_rel("ab", "a", "b")]).failed_items == 0
        stats = graph.upsert_communities([_community("c1", ["a", "b"])])
        assert stats.failed_items == 0
    assert harness.stored_ids("graph.entities") == {"a", "b"}
    assert harness.stored_ids("graph.relationships") == {"ab"}
    assert harness.stored_ids("graph.communities") == {"c1"}

    # The write replaces the stored properties: one the item no longer has
    # (a value now None, a removed attribute) does not keep its old value.
    graph.upsert_entities(
        [_entity("a", type="ORG", attributes={"role": "vendor", "region": "x"})]
    )
    graph.upsert_entities([_entity("a", description="new", attributes={"role": "r"})])
    (read,) = graph.read_entities(["a"])
    assert (read.description, read.type) == ("new", None)
    assert read.attributes == {"role": "r"}


def test_edge_with_a_missing_endpoint_is_a_failed_id(harness: StoreHarness) -> None:
    graph = harness.graph_indexer
    graph.upsert_entities([_entity("a")])
    stats = graph.upsert_relationships([_rel("r1", "a", "ghost"), _rel("r2", "a", "a")])
    assert stats.failed_ids == ["r1"]
    assert stats.successful_items == 1
    assert [r.id for r in graph.read_relationships(["r1", "r2"])] == ["r2"]


def test_readding_an_edge_replaces_it_unless_it_cannot_be_written(
    harness: StoreHarness,
) -> None:
    graph = harness.graph_indexer
    graph.upsert_entities([_entity("a"), _entity("b")])
    graph.index_relationships([_rel("r1", "a", "b", description="old")])
    graph.upsert_relationships([_rel("r1", "b", "a", description="new")])
    (rel,) = graph.read_relationships(["r1"])
    assert (rel.source_id, rel.target_id, rel.description) == ("b", "a", "new")

    # An edge re-pointed at a missing endpoint (a re-keyed entity whose
    # write failed) fails and the stored edge stays: an unchanged document
    # may still state it.
    stats = graph.upsert_relationships([_rel("r1", "a", "ghost")])
    assert stats.failed_ids == ["r1"]
    (rel,) = graph.read_relationships(["r1"])
    assert (rel.source_id, rel.target_id, rel.description) == ("b", "a", "new")
    assert harness.stored_ids("graph.relationships") == {"r1"}


def test_deleting_a_vertex_drops_its_edges(harness: StoreHarness) -> None:
    graph = harness.graph_indexer
    _graph(harness, [("a", "b"), ("b", "c"), ("a", "c")])
    assert graph.find_incident_relationship_ids(["b"], "default") == ["a-b", "b-c"]
    assert graph.delete_by_id(["b"], suffix="default").successful_items == 1
    assert sorted(graph.read_entity_names()) == [("a", "A"), ("c", "C")]
    assert harness.stored_ids("graph.relationships") == {"a-c"}
    # Edges and community vertices by id too.
    graph.upsert_communities([_community("c1", ["a"])])
    graph.delete_by_id(["a-c", "c1"], suffix="default")
    assert harness.stored_ids("graph.relationships") == set()
    assert harness.stored_ids("graph.communities") == set()


def test_a_delete_without_a_suffix_stays_in_the_default(harness: StoreHarness) -> None:
    # No suffix is the default suffix for deletes too: an unscoped delete
    # removed another tenant's vertex with the same content-derived id.
    graph = harness.graph_indexer
    e1, e2 = f"{harness.tag}-e1", f"{harness.tag}-e2"
    tenant = harness.suffix("s1")
    graph.upsert_entities([_entity(e1), _entity(e1, tenant), _entity(e2, tenant)])
    graph.upsert_relationships([_rel("r1", e1, e2, tenant)])
    graph.delete_by_id([e1], suffix=tenant)
    assert [e.id for e in graph.read_entities([e1])] == [e1]
    assert graph.read_entities([e1], suffix=tenant) == []
    assert graph.read_relationships(["r1"], suffix=tenant) == []

    graph.upsert_entities([_entity(e1, tenant)])
    graph.upsert_relationships([_rel("r1", e1, e2, tenant)])
    assert graph.find_incident_relationship_ids([e1], suffix=None) == []
    assert graph.find_incident_relationship_ids([e1], suffix=tenant) == ["r1"]
    graph.upsert_entities([_entity(e1)])
    graph.delete_by_id([e1])
    assert graph.read_entities([e1]) == []
    assert [e.id for e in graph.read_entities([e1], suffix=tenant)] == [e1]
    assert [e.id for e in graph.read_entities([e2], suffix=tenant)] == [e2]


def test_suffix_and_additional_suffix_isolate(harness: StoreHarness) -> None:
    graph = harness.graph_indexer
    tenant = harness.suffix("s1")
    versioned = harness.derive(
        lambda c: setattr(
            c.indexing, "additional_suffix", f"{c.indexing.additional_suffix or ''}v2"
        )
    )
    graph.upsert_entities([_entity("e1", description="t"), _entity("e1", tenant)])
    versioned.graph_indexer.upsert_entities([_entity("e1", description="v")])
    assert [e.description for e in graph.read_entities(["e1"])] == ["t"]
    assert [e.description for e in versioned.graph_indexer.read_entities(["e1"])] == [
        "v"
    ]
    assert [e.id for e in graph.read_entities(["e1"], suffix=tenant)] == ["e1"]
    assert graph.read_entities(["e1"], suffix=harness.suffix("s2")) == []
    assert versioned.graph_indexer.read_entities(["e1"], suffix=tenant) == []


def test_clear_keeps_sibling_namespaces(harness: StoreHarness) -> None:
    # The cleared namespace, a suffix extending it, and the same suffix under
    # an additional suffix: Neptune labels and OpenSearch names share a prefix.
    tag = harness.suffix("clr")
    extended = harness.suffix("clr-eu")
    sibling = harness.derive(
        lambda c: setattr(
            c.indexing, "additional_suffix", f"{c.indexing.additional_suffix or ''}x"
        )
    )
    for store, suffix in (
        (harness, tag),
        (harness, extended),
        (sibling, tag),
    ):
        _graph(store, [("a", "b")], suffix)
        communities = [_community("c1", ["a"], suffix)]
        assert store.graph_indexer.upsert_communities(communities).failed_items == 0
    harness.graph_indexer.clear([tag])
    for store, suffix, left in (
        (harness, tag, set()),
        (harness, extended, {"a", "b"}),
        (sibling, tag, {"a", "b"}),
    ):
        assert store.stored_ids("graph.entities", suffix) == left
        assert store.stored_ids("graph.communities", suffix) == (
            {"c1"} if left else set()
        )
        assert store.stored_ids("graph.relationships", suffix) == (
            {"a-b"} if left else set()
        )


def test_a_full_build_with_suffixes_empties_them(harness: StoreHarness) -> None:
    graph = harness.graph_indexer
    tenant = harness.suffix("s1")
    _graph(harness, [("a", "b")], tenant)
    graph.upsert_communities([_community("c1", ["a"], tenant)])
    _graph(harness, [("k", "l")])

    # A run whose rebuild of the suffix has no items still replaces it.
    graph.index_entities([], suffixes=[tenant])
    graph.index_communities([], suffixes=[tenant])
    assert harness.stored_ids("graph.entities", tenant) == set()
    assert harness.stored_ids("graph.relationships", tenant) == set()
    assert harness.stored_ids("graph.communities", tenant) == set()
    assert harness.stored_ids("graph.entities") == {"k", "l"}

    # A full build replaces the suffix's entities (and their edges).
    graph.index_entities([_entity("m")])
    assert harness.stored_ids("graph.entities") == {"m"}
    assert harness.stored_ids("graph.relationships") == set()


def test_upsert_communities_never_clears(harness: StoreHarness) -> None:
    graph = harness.graph_indexer
    graph.upsert_entities([_entity("a"), _entity("b")])
    graph.index_communities([_community("c1", ["a"]), _community("c2", ["b"])])
    assert graph.upsert_communities([_community("c3", ["a"])]).failed_items == 0
    assert harness.stored_ids("graph.communities") == {"c1", "c2", "c3"}
    graph.index_communities([_community("c9", ["a"])])
    assert harness.stored_ids("graph.communities") == {"c9"}


def test_neighbor_incident_and_name_reads(harness: StoreHarness) -> None:
    graph = harness.graph_indexer
    tenant = harness.suffix("s1")
    _graph(harness, [("a", "b"), ("c", "a"), ("a", "a")])
    graph.upsert_entities([_entity("x", tenant), _entity("a", tenant)])
    graph.upsert_relationships([_rel("s-r", "a", "x", tenant)])
    # The port leaves the neighbour order open (the rank counts them).
    neighbors = graph.read_neighbor_ids(["a", "ghost", "b"])
    assert {k: sorted(v) for k, v in neighbors.items()} == {
        "a": ["a", "b", "c"],
        "b": ["a"],
    }
    assert graph.read_neighbor_ids(["a"], suffix=tenant) == {"a": ["x"]}
    # The count is the length of the neighbour list, unread entities absent.
    assert graph.count_neighbors(["a", "ghost", "b"]) == {"a": 3, "b": 1}
    assert graph.count_neighbors(["a"], suffix=tenant) == {"a": 1}
    assert graph.count_neighbors(["a", "b"], at_most=2) == {"a": 2, "b": 1}
    assert graph.find_incident_relationship_ids(["b"], "default") == ["a-b"]
    assert graph.find_incident_relationship_ids(["a"], suffix=tenant) == ["s-r"]
    assert sorted(graph.read_entity_names()) == [("a", "A"), ("b", "B"), ("c", "C")]
    assert sorted(graph.read_entity_names(tenant)) == [("a", "A"), ("x", "X")]
    assert graph.read_entities(["ghost"]) == []
    assert graph.read_relationships(["ghost"]) == []


def test_document_backed_neighbor_reads(harness: StoreHarness) -> None:
    graph = harness.graph_indexer
    seed = seed_unit_id("seed-doc", 0)
    graph.upsert_entities([_entity(i) for i in ("a", "b", "c", "d", "e")])
    edges = [
        # Seed-only: not followed.
        _rel("a-b", "a", "b", text_unit_ids=[seed]),
        # Document and seed evidence: followed.
        _rel("a-c", "a", "c", text_unit_ids=["u1", seed]),
        # No source_kinds stored (a document-only edge): followed.
        _rel("a-d", "a", "d", text_unit_ids=["u2"]),
        # A seed-only edge beside a document one: the pair stays linked.
        _rel("e-b", "e", "b", text_unit_ids=[seed]),
        _rel("b-e", "b", "e", type="PAYS", text_unit_ids=["u3"]),
    ]
    # As indexing annotates them: the graph codec writes source_kinds.
    stats = graph.upsert_relationships(
        [e.model_copy(update={"source_kinds": annotate_sources(e)}) for e in edges]
    )
    assert stats.failed_items == 0
    backed = graph.read_neighbor_ids(["a", "b", "ghost"], document_backed=True)
    assert {k: sorted(v) for k, v in backed.items()} == {"a": ["c", "d"], "b": ["e"]}
    every = graph.read_neighbor_ids(["a", "b"])
    assert {k: sorted(v) for k, v in every.items()} == {
        "a": ["b", "c", "d"],
        "b": ["a", "e"],
    }
    # The community walk counts every edge (Leiden clusters seeds too).
    assert graph.count_neighbors(["a", "b"]) == {"a": 3, "b": 2}


def test_find_ids_citing_reads_entities_and_communities_of_a_suffix(
    harness: StoreHarness,
) -> None:
    graph = harness.graph_indexer
    tenant = harness.suffix("s1")
    graph.upsert_entities(
        [
            _entity("a", text_unit_ids=["t1", "t2"]),
            _entity("b", text_unit_ids=["t3"]),
            _entity("x", tenant, text_unit_ids=["t1"]),
        ]
    )
    community = _community("c1", None).model_copy(update={"text_unit_ids": ["t2"]})
    graph.upsert_communities([community])

    found = graph.find_ids_citing(["t1", "t9"], suffix="default")
    assert {i: sorted(units) for i, units in found.items()} == {"a": ["t1", "t2"]}
    assert graph.find_ids_citing(["t1"], suffix=tenant) == {"x": ["t1"]}
    assert graph.find_ids_citing(["t3"], suffix="default", communities=True) == {}
    cited = graph.find_ids_citing(["t2"], suffix="default", communities=True)
    assert cited == {"c1": ["t2"]}
    assert graph.find_ids_citing([], suffix="default") == {}


def test_find_relationship_ids_citing_reads_the_edges_of_a_suffix(
    harness: StoreHarness,
) -> None:
    graph = harness.graph_indexer
    tenant = harness.suffix("s1")
    graph.upsert_entities(
        [
            _entity("a"),
            _entity("b"),
            _entity("c"),
            _entity("x", tenant),
            _entity("y", tenant),
        ]
    )
    stats = graph.upsert_relationships(
        [
            _rel("ab", "a", "b", text_unit_ids=["t1", "t2"]),
            _rel("bc", "b", "c", text_unit_ids=["t3"]),
            _rel("ca", "c", "a"),
            _rel("xy", "x", "y", tenant, text_unit_ids=["t1"]),
        ]
    )
    assert stats.failed_items == 0, stats.errors

    found = graph.find_relationship_ids_citing(["t1", "t9"], suffix="default")
    assert {i: sorted(units) for i, units in found.items()} == {"ab": ["t1", "t2"]}
    assert graph.find_relationship_ids_citing(["t1"], suffix=tenant) == {"xy": ["t1"]}
    # A unit id that only prefixes a cited one is not cited.
    assert graph.find_relationship_ids_citing(["t"], suffix="default") == {}
    both = graph.find_relationship_ids_citing(["t2", "t3"], suffix="default")
    assert sorted(both) == ["ab", "bc"]
    assert graph.find_relationship_ids_citing([], suffix="default") == {}


def test_find_entity_ids_by_name_key_is_scoped_to_a_suffix(
    harness: StoreHarness,
) -> None:
    from unified_kg_rag.domain.ingestion.base_resolver import compact_name_key

    graph = harness.graph_indexer
    tenant = harness.suffix("s1")
    graph.upsert_entities(
        [
            _entity("a", name="Acme Inc"),
            _entity("b", name="Buyer"),
            _entity("x", tenant, name="Acme Inc"),
        ]
    )

    key = compact_name_key("Acme Inc")
    assert graph.find_entity_ids_by_name_key([key], suffix="default") == ["a"]
    assert graph.find_entity_ids_by_name_key([key], suffix=tenant) == ["x"]
    assert graph.find_entity_ids_by_name_key(["nobody"], suffix="default") == []


def test_a_rebuild_drops_an_edge_it_did_not_write(harness: StoreHarness) -> None:
    # Both endpoints are written again; the edge between them is not: a
    # sweep that dropped only stale vertices would keep it.
    graph = harness.graph_indexer
    _graph(harness, [("a", "b")])
    assert graph.begin_rebuild(["default"])
    graph.index_entities([_entity("a"), _entity("b")], suffixes=["default"])
    graph.end_rebuild(["default"])

    assert harness.stored_ids("graph.entities") == {"a", "b"}
    assert harness.stored_ids("graph.relationships") == set()


def test_schema_annotations_round_trip_and_an_upsert_clears_them(
    harness: StoreHarness,
) -> None:
    graph = harness.graph_indexer
    iri = "https://example.org/onto#Organization"
    annotated = [
        _entity("a", type="organization", in_schema=True, type_iri=iri),
        _entity("b", type="vendor", in_schema=False),
    ]
    edge = _rel(
        "ab",
        "a",
        "b",
        type="SUPPLIES",
        in_schema=True,
        predicate_iri="https://example.org/onto#supplies",
    )
    assert graph.upsert_entities(annotated).failed_items == 0
    assert graph.upsert_relationships([edge]).failed_items == 0
    # A store returns the vertices in no particular order.
    assert sorted(graph.read_entities(["a", "b"]), key=lambda e: e.id) == annotated
    assert graph.read_relationships(["ab"]) == [edge]

    # Written without them (the schema turned off), they are gone.
    plain = [
        e.model_copy(update={"in_schema": None, "type_iri": None}) for e in annotated
    ]
    graph.upsert_entities(plain)
    graph.upsert_relationships(
        [edge.model_copy(update={"in_schema": None, "predicate_iri": None})]
    )
    read = graph.read_entities(["a", "b"])
    assert [(e.in_schema, e.type_iri) for e in read] == [(None, None)] * 2
    (rel,) = graph.read_relationships(["ab"])
    assert (rel.in_schema, rel.predicate_iri) == (None, None)


def test_seed_graph_fields_round_trip_and_an_upsert_clears_them(
    harness: StoreHarness,
) -> None:
    graph = harness.graph_indexer
    iri = "https://example.org/procure#a"
    seeded = [
        _entity("a", iri=iri, source_kinds=["document", "seed_graph"]),
        _entity("b", source_kinds=["seed_graph"]),
    ]
    edge = _rel("ab", "a", "b", type="SUPPLIES", source_kinds=["seed_graph"])
    assert graph.upsert_entities(seeded).failed_items == 0
    assert graph.upsert_relationships([edge]).failed_items == 0
    assert sorted(graph.read_entities(["a", "b"]), key=lambda e: e.id) == seeded
    assert graph.read_relationships(["ab"]) == [edge]

    # Written without them (no seed unit cited any more), they are gone.
    plain = [e.model_copy(update={"iri": None, "source_kinds": None}) for e in seeded]
    graph.upsert_entities(plain)
    graph.upsert_relationships([edge.model_copy(update={"source_kinds": None})])
    read = graph.read_entities(["a", "b"])
    assert [(e.iri, e.source_kinds) for e in read] == [(None, None)] * 2
    (rel,) = graph.read_relationships(["ab"])
    assert rel.source_kinds is None


def test_schema_annotations_are_listed_and_rewritten_alone(
    harness: StoreHarness,
) -> None:
    graph = harness.graph_indexer
    tenant = harness.suffix("ann")
    iri = "https://example.org/onto#Organization"
    graph.upsert_entities(
        [
            _entity("a", type="organization", in_schema=True, type_iri=iri),
            _entity("b", type="vendor", description="kept"),
            _entity("a", tenant, type="organization", in_schema=True, type_iri=iri),
        ]
    )
    graph.upsert_relationships(
        [_rel("ab", "a", "b", type="SUPPLIES", in_schema=False, description="d")]
    )
    assert graph.read_schema_annotations("entities") == {
        "a": StoredAnnotation(type="organization", in_schema=True, iri=iri),
        "b": StoredAnnotation(type="vendor"),
    }
    assert graph.read_schema_annotations("relationships") == {
        "ab": StoredAnnotation(type="SUPPLIES", in_schema=False)
    }

    company = "https://example.org/onto#Company"
    supplies = "https://example.org/onto#supplies"
    stats = graph.write_schema_annotations(
        "entities", {"a": (None, company), "b": (False, None), "gone": (True, None)}
    )
    assert stats.failed_items == 0
    stats = graph.write_schema_annotations("relationships", {"ab": (True, supplies)})
    assert stats.failed_items == 0

    a, b = sorted(graph.read_entities(["a", "b"]), key=lambda e: e.id)
    assert (a.in_schema, a.type_iri, a.type) == (None, company, "organization")
    assert (b.in_schema, b.type_iri, b.description) == (False, None, "kept")
    (rel,) = graph.read_relationships(["ab"])
    assert (rel.in_schema, rel.predicate_iri, rel.type, rel.description) == (
        True,
        supplies,
        "SUPPLIES",
        "d",
    )
    assert graph.read_entities(["gone"]) == []
    # Another suffix keeps its own.
    (other,) = graph.read_entities(["a"], suffix=tenant)
    assert (other.in_schema, other.type_iri) == (True, iri)
    assert graph.read_schema_annotations("relationships", suffix=tenant) == {}


def test_relationship_annotations_are_kept_per_suffix(harness: StoreHarness) -> None:
    # Relationship ids are suffix-independent: another suffix's edge with
    # the same id is neither read nor rewritten.
    graph = harness.graph_indexer
    tenant = harness.suffix("ann")
    graph.upsert_entities(
        [_entity("a"), _entity("b"), _entity("a", tenant), _entity("b", tenant)]
    )
    graph.upsert_relationships(
        [
            _rel("ab", "a", "b", type="SUPPLIES", in_schema=False),
            _rel("ab", "a", "b", tenant, type="SUPPLIES", in_schema=False),
        ]
    )
    stats = graph.write_schema_annotations("relationships", {"ab": (True, "urn:x:s")})
    assert stats.failed_items == 0
    assert graph.read_schema_annotations("relationships") == {
        "ab": StoredAnnotation(type="SUPPLIES", in_schema=True, iri="urn:x:s")
    }
    assert graph.read_schema_annotations("relationships", suffix=tenant) == {
        "ab": StoredAnnotation(type="SUPPLIES", in_schema=False)
    }
    # A None removes the IRI again.
    graph.write_schema_annotations("relationships", {"ab": (True, None)})
    assert graph.read_schema_annotations("relationships") == {
        "ab": StoredAnnotation(type="SUPPLIES", in_schema=True)
    }


def test_schema_annotations_are_read_in_bounded_requests(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = harness.graph_indexer
    ids = [f"e{i}" for i in range(5)]
    graph.upsert_entities([_entity(i, type="ORG", in_schema=True) for i in ids])
    graph.upsert_relationships(
        [
            _rel(f"r{i}", a, b, type="SUPPLIES")
            for i, (a, b) in enumerate(zip(ids, ids[1:], strict=False))
        ]
    )
    rows: list[int] = []
    if hasattr(graph, "_execute_with_retries"):
        # Gremlin: at most batch_size items' properties per request.
        monkeypatch.setattr(graph.neptune_config, "batch_size", 2)
        execute = graph._execute_with_retries

        def record(build: Any, operation_name: str, **kwargs: Any) -> Any:
            result = execute(build, operation_name, **kwargs)
            if operation_name == "read_schema_annotations":
                rows.append(len(result))
            return result

        monkeypatch.setattr(graph, "_execute_with_retries", record)
    assert graph.read_schema_annotations("entities") == {
        i: StoredAnnotation(type="ORG", in_schema=True) for i in ids
    }
    assert set(graph.read_schema_annotations("relationships") or {}) == {
        "r0",
        "r1",
        "r2",
        "r3",
    }
    if hasattr(graph, "_execute_with_retries"):
        assert rows == [2, 2, 1, 2, 2]


@pytest.mark.parametrize("harness", ["sqlite"], indirect=True)
def test_rewritten_vertex_annotations_match_the_vertex_filters(
    harness: StoreHarness,
) -> None:
    # Stored as a vertex's one-value lists, as every other vertex property.
    from unified_kg_rag.adapters.retrievers.sqlite_graph_retriever import (
        vertex_filter,
    )

    graph = harness.graph_indexer
    graph.upsert_entities([_entity("a", type="organization")])
    graph.write_schema_annotations("entities", {"a": (False, "urn:x:org")})
    with graph.store.read("test") as connection:
        (props,) = connection.execute(
            "SELECT props FROM vertices WHERE id = 'a'"
        ).fetchone()
    value_map = json.loads(props)
    for filters in ({"in_schema": False}, {"type_iri": "urn:x:org"}):
        keep = vertex_filter(filters)
        assert keep is not None and keep(value_map)


@pytest.mark.parametrize(
    "harness", ["container", f"container+{CUSTOM_IDS}"], indirect=True
)
def test_a_failed_annotation_write_is_a_failed_item(
    harness: StoreHarness, monkeypatch: pytest.MonkeyPatch
) -> None:
    graph = harness.graph_indexer
    graph.upsert_entities([_entity("a"), _entity("b")])

    def refused(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("write refused")

    def lost(*args: Any, **kwargs: Any) -> Any:
        raise NamespaceLeaseError("lease lost")

    # Undone before the harness clears the stores.
    with monkeypatch.context() as patch:
        patch.setattr(graph, "_execute_with_retries", refused)
        stats = graph.write_schema_annotations(
            "entities", {"a": (True, None), "b": (True, None)}
        )
        assert (stats.successful_items, stats.failed_items) == (0, 2)
        assert sorted(stats.failed_ids) == ["a", "b"]

        patch.setattr(graph, "_execute_with_retries", lost)
        with pytest.raises(NamespaceLeaseError):
            graph.write_schema_annotations("entities", {"a": (True, None)})


def test_holds_any_item_tells_a_suffix_with_items(harness: StoreHarness) -> None:
    graph = harness.graph_indexer
    tenant, other = harness.suffix("held"), harness.suffix("none")
    assert graph.holds_any_item(suffix=tenant) is False
    graph.upsert_entities([_entity("a", tenant)])
    assert graph.holds_any_item(suffix=tenant) is True
    assert graph.holds_any_item(suffix=other) is False


def test_holds_any_relationship_tells_a_suffix_with_edges(
    harness: StoreHarness,
) -> None:
    # Entities and their community memberships are no relationship; an
    # edge between two of the suffix's entities is.
    graph = harness.graph_indexer
    tenant, other = harness.suffix("edges"), harness.suffix("none")
    graph.upsert_entities([_entity("a", tenant), _entity("b", tenant)])
    graph.upsert_communities([_community("c1", ["a", "b"], tenant)])
    assert graph.holds_any_item(suffix=tenant) is True
    assert graph.holds_any_relationship(suffix=tenant) is False
    graph.upsert_relationships([_rel("a-b", "a", "b", tenant)])
    assert graph.holds_any_relationship(suffix=tenant) is True
    assert graph.holds_any_relationship(suffix=other) is False


def test_relationship_ranks_are_set_in_place(harness: StoreHarness) -> None:
    # A delta rewrites a hub's stored edges for their rank alone: only the
    # rank changes, in the suffix's edges; an edge not stored fails by id.
    graph = harness.graph_indexer
    rel = _rel(
        "a-b",
        "a",
        "b",
        "t1",
        type="SUPPLIES",
        description="supplies",
        weight=2.0,
        rank=2,
        text_unit_ids=["u1"],
    )
    assert graph.upsert_entities([_entity("a", "t1"), _entity("b", "t1")])
    assert graph.upsert_entities([_entity("a", "t2"), _entity("b", "t2")])
    other = rel.model_copy(update={"attributes": {"index": "t2"}})
    assert graph.upsert_relationships([rel, other]).failed_items == 0
    (before,) = graph.read_relationships(["a-b"], suffix="t1")

    stats = graph.update_relationship_ranks(
        [
            rel.model_copy(update={"rank": 7}),
            _rel("x-y", "a", "b", "t1", rank=3),
        ]
    )

    assert (stats.successful_items, stats.failed_ids) == (1, ["x-y"])
    (after,) = graph.read_relationships(["a-b"], suffix="t1")
    assert after.rank == 7
    assert after.model_dump(exclude={"rank"}) == before.model_dump(exclude={"rank"})
    (untouched,) = graph.read_relationships(["a-b"], suffix="t2")
    assert untouched.rank == 2
    # An id given twice is one edge, written with its last rank.
    stats = graph.update_relationship_ranks(
        [rel.model_copy(update={"rank": 4}), rel.model_copy(update={"rank": 5})]
    )
    assert (stats.total_items, stats.successful_items, stats.failed_items) == (1, 1, 0)
    (twice,) = graph.read_relationships(["a-b"], suffix="t1")
    assert twice.rank == 5


def test_a_census_is_the_same_on_every_backend(harness: StoreHarness) -> None:
    """``census_graph`` (``run-describe``'s ``graph``, the load test's
    convergence check) reads through the port alone: the same graph gives
    the same counts and digests on every backend, scoped to its suffix."""
    from unified_kg_rag.application.storage.inspection import _Digest, census_graph

    def _digest(rows: Any) -> str:
        digest = _Digest()
        for row in rows:
            digest.add(row)
        return digest.hexdigest()

    graph = harness.graph_indexer
    tenant = "census_t"
    entities = [
        _entity(
            "a",
            tenant,
            type="ORGANIZATION",
            rank=2,
            description="공급업체",
            text_unit_ids=["t2", "t1"],
        ),
        _entity("b", tenant, type="조직", rank=1, text_unit_ids=["t1"]),
        _entity("c", tenant, rank=1),
    ]
    rels = [
        _rel(
            "a-b",
            "a",
            "b",
            tenant,
            type="SUPPLIES",
            rank=3,
            weight=2.5,
            description="供应",
            text_unit_ids=["t2", "t1"],
        ),
        _rel("a-c", "a", "c", tenant, type="SHIPS_TO", rank=3),
    ]
    assert graph.upsert_entities(entities).failed_items == 0
    assert graph.upsert_relationships(rels).failed_items == 0
    # Another suffix's items are not counted.
    _graph(harness, [("x", "y")], "census_other")

    census = census_graph(graph, tenant, page_size=1)

    assert census is not None
    assert census.vertices.count == 3 and census.edges.count == 2
    assert census.vertices.by_type == {"(untyped)": 1, "ORGANIZATION": 1, "조직": 1}
    assert census.edges.by_type == {"SHIPS_TO": 1, "SUPPLIES": 1}
    assert census.vertices.digest == _digest(
        [["a", "ORGANIZATION"], ["b", "조직"], ["c", "(untyped)"]]
    )
    assert census.edges.rank_digest == _digest([["a-b", 3], ["a-c", 3]])
    assert census.vertices.rank_digest == _digest([["a", 2], ["b", 1], ["c", 1]])
    # Provenance sorted, descriptions as stored, endpoints and weights (an
    # absent weight reads as the model's default, as the full reads do).
    assert census.vertices.provenance_digest == _digest(
        [["a", ["t1", "t2"]], ["b", ["t1"]], ["c", []]]
    )
    assert census.edges.provenance_digest == _digest(
        [["a-b", ["t1", "t2"]], ["a-c", []]]
    )
    assert census.vertices.description_digest == _digest(
        [["a", "공급업체"], ["b", None], ["c", None]]
    )
    assert census.edges.description_digest == _digest([["a-b", "供应"], ["a-c", None]])
    assert census.edges.topology_digest == _digest(
        [["a-b", "a", "b", 2.5], ["a-c", "a", "c", 1.0]]
    )
    assert census.vertices.topology_digest is None
    # A rewritten weight, endpoint or provenance changes only its digest.
    for update, changed in (
        ({"weight": 9.0}, "topology_digest"),
        ({"target_id": "c"}, "topology_digest"),
        ({"text_unit_ids": ["t1"]}, "provenance_digest"),
        ({"description": "다른 설명"}, "description_digest"),
    ):
        assert (
            graph.upsert_relationships([rels[0].model_copy(update=update)]).failed_items
            == 0
        )
        edited = census_graph(graph, tenant)
        assert edited is not None and edited.edges.count == 2, update
        differs = {
            name for name, value in edited.edges if value != getattr(census.edges, name)
        }
        assert differs == {changed}, update
        assert graph.upsert_relationships([rels[0]]).failed_items == 0
        assert census_graph(graph, tenant) == census
    graph.delete_by_id(["a-c"], suffix=tenant)
    after = census_graph(graph, tenant)
    assert after is not None and after.edges.count == 1
    assert after.edges.digest != census.edges.digest
    assert after.vertices == census.vertices


def test_the_sqlite_census_projects_every_census_property() -> None:
    from unified_kg_rag.adapters.storage import sqlite_graph_indexer as sqlite
    from unified_kg_rag.adapters.storage.neptune_codec import CENSUS_PROPERTIES

    assert ("id", *sqlite._CENSUS_KEYS) == CENSUS_PROPERTIES
    paths = ", ".join(f"'$.{k}'" for k in sqlite._CENSUS_KEYS)
    for query in sqlite._CENSUS_QUERIES.values():
        assert f"json_extract(props, {paths})" in query


def test_a_census_counts_every_stored_copy(harness: StoreHarness) -> None:
    """A Gremlin store can hold two edges of one id (a retried rewrite
    without transactions): the census counts rows, not distinct ids, so the
    duplicate shows (SQLite keys an edge by its id: it cannot hold one)."""
    from gremlin_python.process.graph_traversal import __

    from unified_kg_rag.application.storage.inspection import census_graph

    graph = harness.graph_indexer
    if not hasattr(graph, "_execute_with_retries"):
        pytest.skip("only a Gremlin store can hold a duplicate edge")
    tenant = "census_dup"
    _graph(harness, [("a", "b"), ("a", "c")], tenant)
    census = census_graph(graph, tenant)
    assert census is not None and census.edges.count == 2
    label = graph._label(graph.neptune_config.entity_label_prefix, tenant)
    (edge,) = graph.read_relationships(["a-b"], tenant)
    graph._execute_with_retries(
        lambda g: g.V()
        .has("id", "a")
        .hasLabel(label)
        .addE(edge.type or "RELATED")
        .to(__.V().has("id", "b").hasLabel(label))
        .property("id", "a-b")
        .property("rank", edge.rank),
        "duplicate edge",
        results=True,
    )

    doubled = census_graph(graph, tenant, page_size=1)

    assert doubled is not None
    assert doubled.edges.count == 3
    assert doubled.edges.digest != census.edges.digest
    assert doubled.vertices == census.vertices
