# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The seed-graph fields in the stores: an entity's ``iri`` and the
``source_kinds`` of entities and relationships.

Written only on items a seed graph states, so what no seed graph states is
stored exactly as before the fields existed; the filters on them; the vector
copy comparison that decides what is embedded again.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.local.search_engine.query import LoadedIndex, evaluate
from unified_kg_rag.adapters.retrievers.sqlite_graph_retriever import vertex_filter
from unified_kg_rag.adapters.storage.neptune_codec import (
    entity_from_vertex,
    entity_properties,
    relationship_from_edge,
    relationship_properties,
)
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.storage import indexing_manager as manager_module
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.seed_graph import seed_unit_id
from unified_kg_rag.domain.models import Config, Entity, Relationship
from unified_kg_rag.shared.exceptions import StoreRequestRejectedError

pytestmark = pytest.mark.unit

SEED_UNIT = seed_unit_id("seed-doc", 0)
IRI = "https://example.org/procure#" + "vendor-" * 20
BOTH = ["document", "seed_graph"]


def _entity(entity_id: str = "e1", **fields: Any) -> Entity:
    fields.setdefault("text_unit_ids", ["tu-1"])
    return Entity(
        id=entity_id,
        name="Vendor",
        type="ORGANIZATION",
        description="Supplies widgets",
        rank=2.0,
        confidence=0.9,
        **fields,
    )


def _rel(rel_id: str = "r1", **fields: Any) -> Relationship:
    fields.setdefault("text_unit_ids", ["tu-1"])
    fields.setdefault("source_id", "e1")
    fields.setdefault("target_id", "e2")
    return Relationship(
        id=rel_id,
        source_name="Vendor",
        target_name="Buyer",
        type="SUPPLIES",
        description="Vendor supplies Buyer",
        weight=1.0,
        rank=3.0,
        **fields,
    )


def _value_map(props: dict[str, Any]) -> dict[str, list[Any]]:
    """A vertex valueMap of ``props`` (every value a list)."""
    return {k: v if isinstance(v, list) else [v] for k, v in props.items()}


# --------------------------------------------------------------------------- #
# Without a seed graph, every store holds what it held before
# --------------------------------------------------------------------------- #


def test_the_graph_properties_of_an_unseeded_item_are_unchanged() -> None:
    assert entity_properties(_entity(), max_length=100) == {
        "name_key": "vendor",
        "name": "Vendor",
        "type": "ORGANIZATION",
        "description": "Supplies widgets",
        "rank": 2.0,
        "confidence": 0.9,
        "text_unit_ids": ["tu-1"],
    }
    assert relationship_properties(_rel(), max_length=100) == {
        "source_name": "Vendor",
        "target_name": "Buyer",
        "weight": 1.0,
        "description": "Vendor supplies Buyer",
        "rank": 3.0,
        "text_unit_ids": '["tu-1"]',
    }


def test_the_vector_documents_of_an_unseeded_item_are_unchanged() -> None:
    config = Config()
    indexer = OpenSearchIndexer.__new__(OpenSearchIndexer)
    object.__setattr__(indexer, "config", config)
    object.__setattr__(indexer, "opensearch_config", config.indexing.opensearch)
    vectors = ([0.1], [0.2])
    assert indexer._prepare_entity_doc(_entity(), vectors) == {
        "id": "e1",
        "name": "Vendor",
        "name_embedding": [0.1],
        "description": "Supplies widgets",
        "description_embedding": [0.2],
        "type": "ORGANIZATION",
        "rank": 2.0,
        "confidence": 0.9,
        "text_unit_ids": ["tu-1"],
    }
    assert OpenSearchIndexer._prepare_relationship_doc(_rel(), vectors) == {
        "id": "r1",
        "source_id": "e1",
        "target_id": "e2",
        "source_name": "Vendor",
        "target_name": "Buyer",
        "description": "Vendor supplies Buyer",
        "description_embedding": [0.1],
        "weight": 1.0,
        "rank": 3.0,
        "text_unit_ids": ["tu-1"],
    }
    # An empty list is no evidence either.
    assert "source_kinds" not in OpenSearchIndexer._prepare_relationship_doc(
        _rel(source_kinds=[]), vectors
    )


# --------------------------------------------------------------------------- #
# The codec and the vector documents of a seeded item
# --------------------------------------------------------------------------- #


def test_the_codec_writes_and_reads_the_seed_graph_fields() -> None:
    entity = _entity(iri=IRI, source_kinds=BOTH)
    props = entity_properties(entity, max_length=20)
    # A cut IRI would name another resource: written whole. The kinds are a
    # multi-valued vertex property.
    assert props["iri"] == IRI
    assert props["source_kinds"] == BOTH
    # A set-cardinality property keeps no order.
    value_map = _value_map({"id": "e1", **props})
    value_map["source_kinds"] = ["seed_graph", "document"]
    read = entity_from_vertex(value_map)
    assert read is not None and (read.iri, read.source_kinds) == (IRI, BOTH)

    edge = relationship_properties(_rel(source_kinds=["seed_graph"]), max_length=20)
    # Exactly this string for an edge only a seed graph states.
    assert edge["source_kinds"] == '["seed_graph"]'
    both = relationship_properties(
        _rel(source_kinds=["seed_graph", "document"]), max_length=20
    )
    assert json.loads(both["source_kinds"]) == BOTH
    back = relationship_from_edge({"id": "r1", **both}, "SUPPLIES", "e1", "e2")
    assert back is not None and back.source_kinds == BOTH


def test_the_codec_reads_no_seed_graph_field_from_an_earlier_item() -> None:
    props = entity_properties(_entity(), max_length=100)
    read = entity_from_vertex(_value_map({"id": "e1", **props}))
    assert read is not None and (read.iri, read.source_kinds) == (None, None)
    edge = relationship_properties(_rel(), max_length=100)
    back = relationship_from_edge({"id": "r1", **edge}, "SUPPLIES", "e1", "e2")
    assert back is not None and back.source_kinds is None
    # A value of another type than written is left out.
    odd = entity_from_vertex({"id": ["e1"], "name": ["V"], "iri": [3]})
    assert odd is not None and odd.iri is None


def test_the_vector_documents_carry_the_seed_graph_fields_when_set() -> None:
    config = Config()
    indexer = OpenSearchIndexer.__new__(OpenSearchIndexer)
    object.__setattr__(indexer, "config", config)
    object.__setattr__(indexer, "opensearch_config", config.indexing.opensearch)
    doc = indexer._prepare_entity_doc(
        _entity(iri=IRI, source_kinds=BOTH), ([0.1], [0.2])
    )
    assert (doc["iri"], doc["source_kinds"]) == (IRI, BOTH)
    edge = OpenSearchIndexer._prepare_relationship_doc(
        _rel(source_kinds=["seed_graph"]), ([0.1],)
    )
    assert edge["source_kinds"] == ["seed_graph"]
    assert "iri" not in edge


# --------------------------------------------------------------------------- #
# IndexingManager: source_kinds set at every write
# --------------------------------------------------------------------------- #


def _manager() -> tuple[IndexingManager, FakeVectorStore, FakeGraphStore]:
    config = Config()
    graph = FakeGraphStore(config)
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    return (
        IndexingManager(config, vector_indexer=vector, graph_indexer=graph),
        vector,
        graph,
    )


def _items() -> tuple[list[Entity], list[Relationship]]:
    entities = [
        _entity("e1"),
        _entity("e2", text_unit_ids=["tu-1", SEED_UNIT], iri=IRI),
        _entity("e3", text_unit_ids=[SEED_UNIT]),
    ]
    relationships = [
        _rel("r1"),
        _rel("r2", text_unit_ids=[SEED_UNIT], source_id="e2", target_id="e3"),
    ]
    return entities, relationships


_KINDS = {"e1": None, "e2": BOTH, "e3": ["seed_graph"]}
_EDGE_KINDS = {"r1": None, "r2": ["seed_graph"]}


@pytest.mark.parametrize("path", ["full", "delta"])
def test_the_manager_sets_the_source_kinds_of_what_it_writes(path: str) -> None:
    manager, vector, graph = _manager()
    entities, relationships = _items()
    write = manager.index_all_data if path == "full" else manager.index_delta
    write(entities=entities, relationships=relationships)
    # The graph copies read back through the codec.
    graph_entities = graph.read_entities(sorted(graph.ids("entities")))
    graph_edges = graph.read_relationships(sorted(graph.ids("relationships")))
    for stored_entities, stored_edges in (
        (graph_entities, graph_edges),
        (vector.data["entities"].values(), vector.data["relationships"].values()),
    ):
        assert {e.id: e.source_kinds for e in stored_entities} == _KINDS
        assert {r.id: r.source_kinds for r in stored_edges} == _EDGE_KINDS
    assert {e.id: e.iri for e in graph_entities}["e2"] == IRI
    assert vector.data["entities"]["e2"].iri == IRI
    # The caller's items are left as they were.
    assert all(e.source_kinds is None for e in entities)


def test_a_stale_source_kinds_is_cleared_once_no_seed_unit_is_cited() -> None:
    manager, vector, graph = _manager()
    stale = _entity("e1", source_kinds=["seed_graph"])
    manager.index_delta(entities=[stale], relationships=[])
    (read,) = graph.read_entities(["e1"])
    assert read.source_kinds is None
    assert vector.data["entities"]["e1"].source_kinds is None


def test_an_unseeded_item_is_never_stale_against_an_earlier_copy() -> None:
    # A copy written before the fields existed holds neither: an item no
    # seed graph states compares equal, so it is not embedded again.
    _, vector, _ = _manager()
    entity, rel = _entity(), _rel()
    vector.upsert_entities([entity])
    vector.upsert_relationships([rel])
    copies = vector.read_vector_copies("entities", ["e1"])
    copies |= vector.read_vector_copies("relationships", ["r1"])
    assert not {"iri", "source_kinds"} & (set(copies["e1"]) | set(copies["r1"]))
    seeded = manager_module._with_sources(entity)
    assert seeded is entity
    assert not manager_module._vector_stale(seeded, copies["e1"])
    assert not manager_module._vector_stale(rel, copies["r1"])
    # A seed item whose copy lacks them is.
    cited = _entity(text_unit_ids=["tu-1", SEED_UNIT])
    assert manager_module._vector_stale(
        manager_module._with_sources(cited), copies["e1"]
    )
    assert manager_module._vector_stale(_entity(iri=IRI), copies["e1"])


def test_a_rank_only_edge_is_written_in_full_only_when_its_kinds_change() -> None:
    manager, vector, _ = _manager()
    plain, seeded = _rel("r1"), _rel("r2", text_unit_ids=["tu-1", SEED_UNIT])
    vector.upsert_relationships([plain, seeded])  # copies with no source_kinds
    updated: list[str] = []
    upserted: list[str] = []
    lineage, upsert = vector.update_relationship_lineage, vector.upsert_relationships

    def record_lineage(rels: list[Relationship]) -> Any:
        updated.extend(r.id for r in rels)
        return lineage(rels)

    def record_upsert(rels: list[Relationship]) -> Any:
        upserted.extend(r.id for r in rels)
        return upsert(rels)

    vector.update_relationship_lineage = record_lineage  # type: ignore[method-assign]
    vector.upsert_relationships = record_upsert  # type: ignore[method-assign]
    manager._rank_only = {"r1", "r2"}
    manager._upsert_vector_relationships(
        manager._annotated_relationships([plain, seeded])
    )
    # The unseeded edge is updated in place, not embedded again.
    assert updated == ["r1"] and upserted == ["r2"]
    assert vector.data["relationships"]["r2"].source_kinds == BOTH


# --------------------------------------------------------------------------- #
# The filters on the embedded stores
# --------------------------------------------------------------------------- #


def _index(sources: dict[str, dict[str, Any]]) -> LoadedIndex:
    return LoadedIndex(
        name="entities",
        generation=0,
        mappings={"properties": {"source_kinds": {"type": "keyword"}}},
        settings={},
        sources=sources,
        vectors={},
    )


def test_the_embedded_engine_matches_exists_as_opensearch() -> None:
    index = _index(
        {
            "doc": {"id": "doc"},
            "empty": {"id": "empty", "source_kinds": []},
            "null": {"id": "null", "source_kinds": None},
            "seed": {"id": "seed", "source_kinds": ["seed_graph"]},
        }
    )
    assert set(evaluate(index, {"exists": {"field": "source_kinds"}})) == {"seed"}
    assert set(evaluate(index, {"exists": {"field": "unmapped"}})) == set()
    with pytest.raises(StoreRequestRejectedError):
        evaluate(index, {"exists": {"field": "source_kinds", "boost": 2}})


@pytest.mark.parametrize(
    ("value", "kept"),
    [
        ("seed_graph", {"both", "seed"}),
        ("document", {"doc", "both"}),
        (["document", "seed_graph"], {"doc", "both", "seed"}),
    ],
)
def test_the_source_kinds_filter_reads_alike_on_the_embedded_stores(
    value: Any, kept: set[str]
) -> None:
    from unified_kg_rag.adapters.retrievers.opensearch_retriever import (
        OpenSearchRetriever,
    )

    kinds = {"doc": None, "both": BOTH, "seed": ["seed_graph"]}
    index = _index(
        {i: {"id": i, **({"source_kinds": k} if k else {})} for i, k in kinds.items()}
    )
    (clause,) = OpenSearchRetriever._build_filter_clauses({"source_kinds": value})
    assert set(evaluate(index, {"bool": {"filter": [clause]}})) == kept
    keep = vertex_filter({"source_kinds": value})
    assert keep is not None
    vertices = {
        i: {"id": [i], **({"source_kinds": k} if k else {})} for i, k in kinds.items()
    }
    assert {i for i, v in vertices.items() if keep(v)} == kept
