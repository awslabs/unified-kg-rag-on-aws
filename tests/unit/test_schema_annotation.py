# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The schema annotations stored with entities and relationships.

``in_schema`` and ``type_iri`` (entities), ``in_schema`` and
``predicate_iri`` (relationships): computed at write time from the stored
type and the schema, absent from every dump, record and store document
while ``None``, and carried by the Neptune codec and the OpenSearch
documents and mappings.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.storage.neptune_codec import (
    entity_from_vertex,
    entity_properties,
    relationship_from_edge,
    relationship_properties,
)
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.contributions import record_payload
from unified_kg_rag.domain.ingestion.schema_annotation import SchemaAnnotator
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    Relationship,
    TextUnitExtraction,
)
from unified_kg_rag.domain.models.schema import (
    EntityTypeSpec,
    ExtractionSchema,
    RelationshipTypeSpec,
)

pytestmark = pytest.mark.unit

ORG = "https://example.org/onto#Organization"
SUPPLIES = "https://example.org/onto#supplies"
_ENTITY_TYPES: list[Any] = [
    EntityTypeSpec(label="ORGANIZATION", description="Companies", iri=ORG),
    "PRODUCT: Goods",
]
_RELATIONSHIP_TYPES: list[Any] = [
    RelationshipTypeSpec(label="SUPPLIES", iri=SUPPLIES),
    "STOCKS",
]


def _schema(mode: str = "align", **keys: Any) -> ExtractionSchema:
    keys.setdefault("entity_types", _ENTITY_TYPES)
    keys.setdefault("relationship_types", _RELATIONSHIP_TYPES)
    return ExtractionSchema.from_items(mode=mode, **keys)  # type: ignore[arg-type]


def _config(**extraction: Any) -> Config:
    return Config.model_validate({"processing": {"graph_extraction": extraction}})


def _entity(entity_id: str, type_: str | None, **fields: Any) -> Entity:
    return Entity(id=entity_id, name=entity_id.title(), type=type_, **fields)


def _rel(rel_id: str, type_: str | None, **fields: Any) -> Relationship:
    fields.setdefault("source_id", "a")
    fields.setdefault("target_id", "b")
    return Relationship(id=rel_id, type=type_, **fields)


# --------------------------------------------------------------------------- #
# The annotation function
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("mode", ["align", "strict"])
def test_declared_types_are_in_schema_with_their_iri(mode: str) -> None:
    annotator = SchemaAnnotator(_schema(mode))
    assert annotator.active
    # The stored type is the lowercased vote: matched by schema_key.
    assert annotator.entity_annotation("organization") == (True, ORG)
    assert annotator.entity_annotation(" Organization ") == (True, ORG)
    assert annotator.entity_annotation("product") == (True, None)
    assert annotator.relationship_annotation("supplies") == (True, SUPPLIES)
    assert annotator.relationship_annotation("STOCKS") == (True, None)


def test_undeclared_and_untyped_items_are_off_schema() -> None:
    annotator = SchemaAnnotator(_schema())
    for label in ("vendor", "", None, "ORGANIZATIONS", "Org"):
        assert annotator.entity_annotation(label) == (False, None), label
    for label in ("mentions", "", None, "SUPPLY"):
        assert annotator.relationship_annotation(label) == (False, None), label


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(_schema("off"), id="off"),
        # Entity types alone (every configuration from before the schema
        # keys) leave the records and the stores as they were.
        pytest.param(_schema(relationship_types=[]), id="align-entity-types-only"),
        pytest.param(ExtractionSchema.from_items(), id="no-schema"),
    ],
)
def test_no_annotation_where_no_schema_applies(schema: ExtractionSchema) -> None:
    annotator = SchemaAnnotator(schema)
    assert not annotator.active
    assert annotator.entity_annotation("organization") == (None, None)
    assert annotator.relationship_annotation("supplies") == (None, None)


def test_a_kind_without_declared_types_is_not_annotated() -> None:
    strict = SchemaAnnotator(_schema("strict", relationship_types=[]))
    assert strict.entity_annotation("organization") == (True, ORG)
    assert strict.relationship_annotation("supplies") == (None, None)
    relationships_only = SchemaAnnotator(_schema(entity_types=[]))
    assert relationships_only.entity_annotation("organization") == (None, None)
    assert relationships_only.relationship_annotation("supplies") == (True, SUPPLIES)


def test_an_item_is_copied_only_when_its_annotation_changes() -> None:
    annotator = SchemaAnnotator(_schema())
    annotated = _entity("v", "organization", in_schema=True, type_iri=ORG)
    assert annotator.entity(annotated) is annotated
    plain = _entity("v", "organization")
    copy = annotator.entity(plain)
    assert copy is not plain and plain.in_schema is None
    assert (copy.in_schema, copy.type_iri) == (True, ORG)
    rel = annotator.relationship(_rel("r", "SUPPLIES"))
    assert (rel.in_schema, rel.predicate_iri) == (True, SUPPLIES)


def test_a_stale_annotation_is_cleared_where_no_schema_applies() -> None:
    off = SchemaAnnotator(_schema("off"))
    entity = off.entity(_entity("v", "organization", in_schema=True, type_iri=ORG))
    assert (entity.in_schema, entity.type_iri) == (None, None)
    rel = off.relationship(_rel("r", "x", in_schema=False, predicate_iri=SUPPLIES))
    assert (rel.in_schema, rel.predicate_iri) == (None, None)


# --------------------------------------------------------------------------- #
# Dumps and records
# --------------------------------------------------------------------------- #


def test_a_dump_leaves_unset_annotations_out() -> None:
    entity, rel = _entity("v", "org"), _rel("r", "SUPPLIES")
    for dump in (entity.model_dump(), entity.model_dump(mode="json")):
        assert not {"in_schema", "type_iri"} & set(dump)
    assert not {"in_schema", "predicate_iri"} & set(rel.model_dump())
    assert "in_schema" not in entity.model_dump_json()
    set_ = entity.model_copy(update={"in_schema": False})
    assert set_.model_dump()["in_schema"] is False
    assert "type_iri" not in set_.model_dump()
    assert Entity.model_validate(set_.model_dump()) == set_


def test_an_extraction_record_never_holds_annotations() -> None:
    record = TextUnitExtraction(
        text_unit_id="t1",
        entities=[_entity("v", "org", in_schema=True, type_iri=ORG)],
        relationships=[_rel("r", "SUPPLIES", in_schema=True, predicate_iri=SUPPLIES)],
    )
    payload = record_payload(record)
    assert not {"in_schema", "type_iri"} & set(payload["entities"][0])
    assert not {"in_schema", "predicate_iri"} & set(payload["relationships"][0])


# --------------------------------------------------------------------------- #
# Stores: the Neptune codec and the OpenSearch documents
# --------------------------------------------------------------------------- #


def _value_map(props: dict[str, Any]) -> dict[str, list[Any]]:
    """A vertex valueMap of ``props`` (every value a list)."""
    return {
        key: value if isinstance(value, list) else [value]
        for key, value in props.items()
    }


def test_the_codec_writes_and_reads_the_annotations() -> None:
    long_iri = "https://example.org/" + "x" * 50
    entity = _entity("v", "org", in_schema=False, type_iri=long_iri)
    props = entity_properties(entity, max_length=20)
    # A cut IRI would name another resource: written whole.
    assert (props["in_schema"], props["type_iri"]) == (False, long_iri)
    read = entity_from_vertex(_value_map({"id": "v", **props}))
    assert read is not None and (read.in_schema, read.type_iri) == (False, long_iri)

    rel = _rel("r", "SUPPLIES", in_schema=True, predicate_iri=SUPPLIES)
    edge = relationship_properties(rel, max_length=20)
    assert (edge["in_schema"], edge["predicate_iri"]) == (True, SUPPLIES)
    back = relationship_from_edge({"id": "r", **edge}, "SUPPLIES", "a", "b")
    assert back is not None and (back.in_schema, back.predicate_iri) == (
        True,
        SUPPLIES,
    )


def test_the_codec_writes_no_annotation_while_unset() -> None:
    props = entity_properties(_entity("v", "org"), max_length=100)
    assert not {"in_schema", "type_iri"} & set(props)
    edge = relationship_properties(_rel("r", "SUPPLIES"), max_length=100)
    assert not {"in_schema", "predicate_iri"} & set(edge)
    read = entity_from_vertex(_value_map({"id": "v", **props}))
    assert read is not None and (read.in_schema, read.type_iri) == (None, None)


def test_the_codec_ignores_an_annotation_of_another_type() -> None:
    read = entity_from_vertex(
        {"id": ["v"], "name": ["V"], "in_schema": ["true"], "type_iri": [3]}
    )
    assert read is not None and (read.in_schema, read.type_iri) == (None, None)


def test_opensearch_documents_carry_the_annotations_only_when_set() -> None:
    vectors = ([0.1], [0.2])
    plain = OpenSearchIndexer._prepare_relationship_doc(_rel("r", "S"), vectors)
    assert not {"in_schema", "predicate_iri"} & set(plain)
    annotated = OpenSearchIndexer._prepare_relationship_doc(
        _rel("r", "S", in_schema=False, predicate_iri=SUPPLIES), vectors
    )
    assert (annotated["in_schema"], annotated["predicate_iri"]) == (False, SUPPLIES)


def test_opensearch_mappings_type_the_annotations() -> None:
    config = Config()
    indexer = OpenSearchIndexer.__new__(OpenSearchIndexer)
    object.__setattr__(indexer, "config", config)
    object.__setattr__(indexer, "opensearch_config", config.indexing.opensearch)
    object.__setattr__(indexer, "_embedding_dimension", 8)
    object.__setattr__(indexer, "analyzer", "standard")
    entities = indexer._get_entities_mapping()["mappings"]["properties"]
    relationships = indexer._get_relationships_mapping()["mappings"]["properties"]
    assert entities["in_schema"] == relationships["in_schema"] == {"type": "boolean"}
    assert entities["type_iri"] == relationships["predicate_iri"] == {"type": "keyword"}
    entity = _entity("v", "org", in_schema=True, type_iri=ORG)
    doc = indexer._prepare_entity_doc(entity, ([0.1], [0.2]))
    assert (doc["in_schema"], doc["type_iri"]) == (True, ORG)
    assert "in_schema" not in indexer._prepare_entity_doc(
        _entity("v", "org"), ([0.1], [0.2])
    )


# --------------------------------------------------------------------------- #
# IndexingManager: annotated at write time on every path
# --------------------------------------------------------------------------- #


_SCHEMA_CONFIG = {
    "schema_mode": "align",
    "entity_types": _ENTITY_TYPES,
    "relationship_types": _RELATIONSHIP_TYPES,
}


def _manager(config: Config) -> tuple[IndexingManager, FakeVectorStore, FakeGraphStore]:
    graph = FakeGraphStore(config)
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config, vector_indexer=vector, graph_indexer=graph)
    return manager, vector, graph


def _items() -> tuple[list[Entity], list[Relationship]]:
    entities = [_entity("a", "organization"), _entity("b", "vendor")]
    relationships = [
        _rel("ab", "SUPPLIES", source_name="A", target_name="B"),
        _rel("ba", "mentions", source_id="b", target_id="a"),
    ]
    return entities, relationships


def _stored(
    vector: FakeVectorStore, graph: FakeGraphStore
) -> tuple[dict[str, Any], ...]:
    graph_entities = graph.read_entities(sorted(graph.ids("entities")))
    graph_edges = graph.read_relationships(sorted(graph.ids("relationships")))
    return (
        {e.id: (e.in_schema, e.type_iri) for e in graph_entities},
        {r.id: (r.in_schema, r.predicate_iri) for r in graph_edges},
        {
            i: (e.in_schema, e.type_iri)
            for i, e in vector.data.get("entities", {}).items()
        },
        {
            i: (r.in_schema, r.predicate_iri)
            for i, r in vector.data.get("relationships", {}).items()
        },
    )


_ANNOTATED = (
    {"a": (True, ORG), "b": (False, None)},
    {"ab": (True, SUPPLIES), "ba": (False, None)},
    {"a": (True, ORG), "b": (False, None)},
    {"ab": (True, SUPPLIES), "ba": (False, None)},
)


@pytest.mark.parametrize("path", ["full", "delta"])
def test_the_manager_annotates_what_it_writes(path: str) -> None:
    manager, vector, graph = _manager(_config(**_SCHEMA_CONFIG))
    entities, relationships = _items()
    write = manager.index_all_data if path == "full" else manager.index_delta
    write(entities=entities, relationships=relationships)
    assert _stored(vector, graph) == _ANNOTATED
    # The caller's items are left as they were.
    assert all(e.in_schema is None for e in entities)


@pytest.mark.parametrize("path", ["full", "delta"])
def test_with_no_schema_the_manager_writes_no_annotation(path: str) -> None:
    manager, vector, graph = _manager(Config())
    entities, relationships = _items()
    stale = [e.model_copy(update={"in_schema": True}) for e in entities]
    write = manager.index_all_data if path == "full" else manager.index_delta
    write(entities=stale, relationships=relationships)
    none = {"a": (None, None), "b": (None, None)}
    edges = {"ab": (None, None), "ba": (None, None)}
    assert _stored(vector, graph) == (none, edges, none, edges)


def test_a_rank_only_edge_with_another_annotation_is_written_in_full() -> None:
    manager, vector, _ = _manager(_config(**_SCHEMA_CONFIG))
    _, relationships = _items()
    vector.upsert_relationships(relationships)  # written with no schema
    manager._rank_only = {"ab"}
    annotated = manager._annotator.relationships(relationships)
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
    manager._upsert_vector_relationships(annotated)
    # Its stored copy has no annotation: a lineage update would keep it so.
    assert updated == []
    assert sorted(upserted) == ["ab", "ba"]
    assert vector.data["relationships"]["ab"].in_schema is True

    # Once the copy holds it, a rank-only edge is updated in place again.
    manager._rank_only = {"ab"}
    upserted.clear()
    manager._upsert_vector_relationships(annotated)
    assert updated == ["ab"] and upserted == ["ba"]
