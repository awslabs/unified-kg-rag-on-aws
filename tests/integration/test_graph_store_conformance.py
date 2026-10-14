# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One ``GraphIndexer`` contract for every graph backend.

Runs on the fake (the reference), the embedded SQLite store, Gremlin Server
(``LOCAL_STORES=1``) and Neptune (``-m aws``); see
``tests/fixtures/store_backends.py``. A difference between them is a bug in
one of them, whichever is right.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.store_backends import BACKENDS, StoreHarness, open_harness
from unified_kg_rag.domain.models import Community, Entity, Relationship

pytestmark = pytest.mark.integration


@pytest.fixture(params=BACKENDS)
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


def test_upsert_is_idempotent_and_keeps_absent_properties(
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

    # A property the upsert's item lacks keeps its stored value.
    graph.upsert_entities([_entity("a", type="ORG")])
    (read,) = graph.read_entities(["a"])
    assert (read.description, read.type) == ("kept", "ORG")


def test_edge_with_a_missing_endpoint_is_a_failed_id(harness: StoreHarness) -> None:
    graph = harness.graph_indexer
    graph.upsert_entities([_entity("a")])
    stats = graph.upsert_relationships([_rel("r1", "a", "ghost"), _rel("r2", "a", "a")])
    assert stats.failed_ids == ["r1"]
    assert stats.successful_items == 1
    assert [r.id for r in graph.read_relationships(["r1", "r2"])] == ["r2"]


def test_readding_an_edge_replaces_it_even_when_it_cannot_be_written(
    harness: StoreHarness,
) -> None:
    graph = harness.graph_indexer
    graph.upsert_entities([_entity("a"), _entity("b")])
    graph.index_relationships([_rel("r1", "a", "b", description="old")])
    graph.upsert_relationships([_rel("r1", "b", "a", description="new")])
    (rel,) = graph.read_relationships(["r1"])
    assert (rel.source_id, rel.target_id, rel.description) == ("b", "a", "new")

    # The stored edge of the id is dropped before the new one is written, so
    # an edge re-added towards a missing endpoint leaves no edge behind.
    stats = graph.upsert_relationships([_rel("r1", "a", "ghost")])
    assert stats.failed_ids == ["r1"]
    assert graph.read_relationships(["r1"]) == []
    assert harness.stored_ids("graph.relationships") == set()


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


def test_unscoped_delete_reaches_every_suffix(harness: StoreHarness) -> None:
    graph = harness.graph_indexer
    # Ids of this run only: an unscoped delete is graph-wide.
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
    assert graph.find_incident_relationship_ids([e1], suffix=None) == ["r1"]
    graph.delete_by_id([e1])
    assert graph.read_entities([e1]) == []
    assert graph.read_entities([e1], suffix=tenant) == []
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
    assert harness.graph_indexer.clear([tag])
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
    assert graph.find_incident_relationship_ids(["b"], "default") == ["a-b"]
    assert graph.find_incident_relationship_ids(["a"], suffix=tenant) == ["s-r"]
    assert sorted(graph.read_entity_names()) == [("a", "A"), ("b", "B"), ("c", "C")]
    assert sorted(graph.read_entity_names(tenant)) == [("a", "A"), ("x", "X")]
    assert graph.read_entities(["ghost"]) == []
    assert graph.read_relationships(["ghost"]) == []
