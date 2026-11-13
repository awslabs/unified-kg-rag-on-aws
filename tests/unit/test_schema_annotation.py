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

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from tests.fixtures.schema_annotation import (
    UNANNOTATED_DIGEST,
    schema_annotation_digest,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.adapters.schema_sources import SchemaSourceContext
from unified_kg_rag.adapters.schema_sources import registry as schema_registry
from unified_kg_rag.adapters.storage.neptune_codec import (
    entity_from_vertex,
    entity_properties,
    relationship_from_edge,
    relationship_properties,
)
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.ingestion import incremental as incremental_module
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    annotations_due,
    reannotate_on_schema_change,
    record_rebuilt_annotations,
)
from unified_kg_rag.application.ingestion.pipeline import _Borrowed
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.cache_keys import (
    extraction_settings,
    stage_input_fingerprint,
)
from unified_kg_rag.domain.ingestion.contributions import record_payload
from unified_kg_rag.domain.ingestion.schema_annotation import (
    REANNOTATING_DIGEST,
    SchemaAnnotator,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatusRecord,
    DocumentDelta,
    Entity,
    PipelineStageType,
    Relationship,
    SchemaDocument,
    TextUnitExtraction,
)
from unified_kg_rag.domain.models.schema import (
    EntityTypeSpec,
    ExtractionSchema,
    RelationshipTypeSpec,
)
from unified_kg_rag.ports.indexer import (
    GraphIndexer,
    IndexingStats,
    rewrites_schema_annotations,
)
from unified_kg_rag.shared import (
    AWSServiceError,
    ConfigurationError,
    NamespaceLeaseError,
)
from unified_kg_rag.shared.write_fence import install_write_fence, remove_write_fence

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
    assert annotator.fields == {"in_schema", "type_iri", "predicate_iri"}
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
        # Entity types alone and without IRIs (every configuration from
        # before the schema keys) leave the records and the stores as they
        # were.
        pytest.param(
            _schema(relationship_types=[], entity_types=["ORGANIZATION", "PRODUCT"]),
            id="align-entity-types-only",
        ),
        pytest.param(ExtractionSchema.from_items(), id="no-schema"),
    ],
)
def test_no_annotation_where_no_schema_applies(schema: ExtractionSchema) -> None:
    annotator = SchemaAnnotator(schema)
    assert annotator.fields == frozenset()
    assert annotator.entity_annotation("organization") == (None, None)
    assert annotator.relationship_annotation("supplies") == (None, None)


def test_a_declared_iri_is_written_where_in_schema_is_not() -> None:
    # align with entity types alone: no in_schema (extraction leaves the
    # records as written), but a declared class IRI still links the type.
    annotator = SchemaAnnotator(_schema(relationship_types=[]))
    assert annotator.fields == {"type_iri"}
    assert annotator.entity_annotation("organization") == (None, ORG)
    assert annotator.entity_annotation(" Organization ") == (None, ORG)
    assert annotator.entity_annotation("product") == (None, None)
    assert annotator.entity_annotation("vendor") == (None, None)
    assert annotator.entity_annotation(None) == (None, None)
    assert annotator.relationship_annotation("supplies") == (None, None)
    entity = annotator.entity(_entity("v", "organization"))
    assert (entity.in_schema, entity.type_iri) == (None, ORG)
    # An undeclared type has its stale IRI cleared.
    stale = annotator.entity(_entity("v", "vendor", type_iri=ORG))
    assert (stale.in_schema, stale.type_iri) == (None, None)


def test_off_writes_no_iri() -> None:
    annotator = SchemaAnnotator(_schema("off", relationship_types=[]))
    assert annotator.fields == frozenset()
    assert annotator.entity_annotation("organization") == (None, None)


def test_every_item_of_a_kind_without_declared_types_is_in_schema() -> None:
    # Nothing of that kind is off schema (extraction checks none of it, and
    # strict kept every one): in_schema true, so a filter on it keeps the
    # kind instead of emptying it (all relationships under strict with
    # entity types alone).
    strict = SchemaAnnotator(_schema("strict", relationship_types=[]))
    assert strict.fields == {"in_schema", "type_iri"}
    assert strict.entity_annotation("organization") == (True, ORG)
    assert strict.entity_annotation("vendor") == (False, None)
    for label in ("supplies", "anything", "", None):
        assert strict.relationship_annotation(label) == (True, None), label
    relationships_only = SchemaAnnotator(_schema(entity_types=[]))
    assert relationships_only.fields == {"in_schema", "predicate_iri"}
    for label in ("organization", "", None):
        assert relationships_only.entity_annotation(label) == (True, None), label
    assert relationships_only.relationship_annotation("supplies") == (True, SUPPLIES)
    assert relationships_only.relationship_annotation("mentions") == (False, None)


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
    # With the vector copy the derivation read for it.
    copies = {
        ("default", "ab"): vector.read_vector_copies("relationships", ["ab"])["ab"]
    }
    manager._rank_only_copies = dict(copies)
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
    manager._rank_only_copies = {
        ("default", "ab"): vector.read_vector_copies("relationships", ["ab"])["ab"]
    }
    upserted.clear()
    manager._upsert_vector_relationships(annotated)
    assert updated == ["ab"] and upserted == ["ba"]


# --------------------------------------------------------------------------- #
# Re-annotation: a schema edit only indexing sees
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "extraction",
    [
        {},
        {"entity_types": ["PERSON"]},
        {"entity_types": ["PERSON"], "relationship_types": ["R"], "schema_mode": "off"},
        {"entity_types": [{"label": "PERSON", "iri": "urn:p"}], "schema_mode": "off"},
    ],
)
def test_a_schema_that_annotates_nothing_has_the_unannotated_digest(
    extraction: dict[str, Any],
) -> None:
    assert schema_annotation_digest(_config(**extraction)) == UNANNOTATED_DIGEST


def test_the_digest_follows_the_annotations_only() -> None:
    base = schema_annotation_digest(_config(**_SCHEMA_CONFIG))
    # Descriptions, patterns, spellings of a label and the item form: the
    # same annotations.
    same = _config(
        schema_mode="align",
        entity_types=[
            {"label": "organization", "iri": ORG},
            {"label": "PRODUCT", "description": "Other words"},
        ],
        relationship_types=[{"label": "supplies", "iri": SUPPLIES}, "STOCKS: x"],
        relationship_patterns=["ORGANIZATION -SUPPLIES-> ORGANIZATION"],
    )
    assert schema_annotation_digest(same) == base
    other_iri = {**_SCHEMA_CONFIG, "entity_types": [{"label": "ORGANIZATION"}]}
    assert schema_annotation_digest(_config(**other_iri)) != base
    strict = {**_SCHEMA_CONFIG, "schema_mode": "strict"}
    assert schema_annotation_digest(_config(**strict)) == base  # same annotations
    off = {**_SCHEMA_CONFIG, "schema_mode": "off"}
    assert schema_annotation_digest(_config(**off)) == UNANNOTATED_DIGEST


def test_the_digest_follows_the_type_keys_and_every_iri() -> None:
    strict = {"schema_mode": "strict", "relationship_types": ["SUPPLIES"]}
    assert schema_annotation_digest(
        _config(**strict, entity_types=["ORGANIZATION"])
    ) != schema_annotation_digest(_config(**strict, entity_types=["PRODUCT"]))
    assert schema_annotation_digest(
        _config(**{**strict, "relationship_types": ["SUPPLIES"]})
    ) != schema_annotation_digest(
        _config(**{**strict, "relationship_types": ["STOCKS"]})
    )
    # A predicate IRI alone: the digest and the indexing stage's key.
    configs = [
        _config(relationship_types=[{"label": "SUPPLIES", "iri": iri}])
        for iri in ("urn:x:a", "urn:x:b")
    ]
    assert len({schema_annotation_digest(c) for c in configs}) == 2
    assert (
        len(
            {
                stage_input_fingerprint(c, PipelineStageType.INDEXING, "corpus")
                for c in configs
            }
        )
        == 2
    )


def test_the_digest_and_the_indexing_key_need_the_schema_loaded() -> None:
    config = _config(schema_source={"kind": "file", "path": "schema.yaml"})
    with pytest.raises(ConfigurationError, match="not loaded yet"):
        schema_annotation_digest(config)
    with pytest.raises(ConfigurationError, match="not loaded yet"):
        stage_input_fingerprint(config, PipelineStageType.INDEXING, "corpus")


def _written_without_a_schema() -> tuple[FakeVectorStore, FakeGraphStore]:
    manager, vector, graph = _manager(Config())
    entities, relationships = _items()
    manager.index_all_data(entities=entities, relationships=relationships)
    return vector, graph


def test_reannotate_rewrites_what_a_full_build_writes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    vector, graph = _written_without_a_schema()
    # A vector copy without its edge: no type to annotate it by.
    orphan = _rel("orphan", "SUPPLIES")
    vector.upsert_relationships([orphan])
    manager = IndexingManager(
        _config(**_SCHEMA_CONFIG), vector_indexer=vector, graph_indexer=graph
    )
    with caplog.at_level(logging.INFO):
        result = manager.reannotate({"default"})
    assert result.complete
    assert (
        "checked 2 stored entities and 2 stored relationships in 'default' "
        "and rewrote in_schema and the IRIs of 2 and 2 of them"
    ) in caplog.text
    assert (result.entities, result.entities_rewritten) == (2, 2)
    assert (result.relationships, result.relationships_rewritten) == (2, 2)
    stored = _stored(vector, graph)
    assert stored[3].pop("orphan") == (None, None)
    assert stored == _ANNOTATED

    # Annotated already: nothing written.
    writes: list[str] = []
    for store in (vector, graph):
        write = store.write_schema_annotations

        def record(kind: Any, annotations: Any, suffix: Any = None, w=write) -> Any:
            writes.append(kind)
            return w(kind, annotations, suffix)

        store.write_schema_annotations = record  # type: ignore[method-assign]
    again = manager.reannotate({"default"})
    assert writes == []
    assert (again.entities_rewritten, again.relationships_rewritten) == (0, 0)


def test_reannotate_skips_a_store_that_cannot_list_its_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector, graph = _written_without_a_schema()
    monkeypatch.setattr(graph, "read_schema_annotations", lambda kind, suffix: None)
    manager = IndexingManager(
        _config(**_SCHEMA_CONFIG), vector_indexer=vector, graph_indexer=graph
    )
    result = manager.reannotate({"default"})
    assert result.unsupported and not result.complete
    graph_entities, graph_edges, vector_entities, vector_edges = _stored(vector, graph)
    assert set(graph_entities.values()) == {(None, None)}
    # The vector entities hold their type; its relationships do not.
    assert vector_entities == _ANNOTATED[2]
    assert set(vector_edges.values()) == {(None, None)}


_NAMESPACE = "ns"
# The registry holds earlier documents of the namespace.
_EXISTING = DocumentDelta(unchanged=["ns#earlier"], stored_scopes=["ns|A"])


def _incremental(
    config: Config,
    vector: FakeVectorStore,
    graph: Any,
    store: FakeDocStatusStore | None = None,
    scope: str = "ns|A",
) -> IncrementalIndexer:
    settings = extraction_settings(config)
    return IncrementalIndexer(
        store if store is not None else FakeDocStatusStore(),
        IndexingManager(config, vector_indexer=vector, graph_indexer=graph),
        scope=scope,
        extraction_fingerprint=settings.fingerprint,
        extraction_settings=settings,
    )


def _index(incremental: IncrementalIndexer, delta: DocumentDelta = _EXISTING) -> bool:
    """What ``index_delta`` does about the annotations: whether a pass is
    due, then the pass."""
    due = annotations_due(incremental, delta)
    if due:
        reannotate_on_schema_change(incremental, [])
    return due


def _digest(config: Config) -> str:
    return schema_annotation_digest(config)


def _reads(*stores: Any) -> list[str]:
    """Records every ``read_schema_annotations`` call of ``stores``."""
    reads: list[str] = []
    for store in stores:
        read = store.read_schema_annotations

        def record(kind: Any, suffix: Any = None, r=read) -> Any:
            reads.append(kind)
            return r(kind, suffix)

        store.read_schema_annotations = record
    return reads


def _full_build(config: Config) -> tuple[dict[str, Any], ...]:
    manager, vector, graph = _manager(config)
    entities, relationships = _items()
    manager.index_all_data(entities=entities, relationships=relationships)
    return _stored(vector, graph)


def test_a_changed_digest_marks_the_namespace_then_records_the_pass() -> None:
    vector, graph = _written_without_a_schema()
    config = _config(**_SCHEMA_CONFIG)
    incremental = _incremental(config, vector, graph)
    store = incremental.doc_status
    assert isinstance(store, FakeDocStatusStore)
    store.put_annotation_digest(_NAMESPACE, UNANNOTATED_DIGEST)

    # Marked before the run writes anything; recorded once rewritten.
    assert annotations_due(incremental, _EXISTING)
    assert store.get_annotation_digest(_NAMESPACE) == REANNOTATING_DIGEST
    reannotate_on_schema_change(incremental, [])
    assert _stored(vector, graph) == _ANNOTATED
    assert store.get_annotation_digest(_NAMESPACE) == _digest(config)

    # The next run, of any scope, has nothing to read.
    reads = _reads(vector, graph)
    other = _incremental(config, vector, graph, store, scope="ns|B")
    assert not _index(other)
    assert reads == []


def test_a_new_scope_of_an_annotated_namespace_reads_nothing() -> None:
    config = _config(**_SCHEMA_CONFIG)
    manager, vector, graph = _manager(config)
    entities, relationships = _items()
    manager.index_all_data(entities=entities, relationships=relationships)
    store = FakeDocStatusStore()
    store.put_annotation_digest(_NAMESPACE, _digest(config))
    reads = _reads(vector, graph)
    incremental = _incremental(config, vector, graph, store, scope="ns|B")
    assert not _index(incremental, DocumentDelta(new=["ns#b"], stored_scopes=["ns|A"]))
    assert reads == []


S1 = {
    "schema_mode": "align",
    "entity_types": ["ORGANIZATION", "PRODUCT"],
    "relationship_types": ["SUPPLIES"],
}
S2 = {**S1, "entity_types": [{"label": "ORGANIZATION", "iri": "urn:x:org"}, "PRODUCT"]}


def test_a_schema_reverted_on_another_scope_is_rewritten() -> None:
    # Scope A annotates the namespace by S2, scope B then runs with S1: the
    # namespace holds S2's annotations, not those B's scope last wrote.
    first, second = _config(**S1), _config(**S2)
    manager, vector, graph = _manager(first)
    entities, relationships = _items()
    manager.index_all_data(entities=entities, relationships=relationships)
    store = FakeDocStatusStore()
    store.put_annotation_digest(_NAMESPACE, _digest(first))

    assert _index(_incremental(second, vector, graph, store, scope="ns|A"))
    assert _stored(vector, graph) == _full_build(second)
    assert _index(_incremental(first, vector, graph, store, scope="ns|B"))
    assert _stored(vector, graph) == _full_build(first)
    assert store.get_annotation_digest(_NAMESPACE) == _digest(first)


def test_a_pass_stopped_midway_is_made_again_under_any_schema() -> None:
    first, second = _config(**S1), _config(**S2)
    manager, vector, graph = _manager(first)
    entities, relationships = _items()
    manager.index_all_data(entities=entities, relationships=relationships)
    store = FakeDocStatusStore()
    store.put_annotation_digest(_NAMESPACE, _digest(first))

    # Killed after the vector store was rewritten for S2, before the graph.
    write = vector.write_schema_annotations

    def killed(kind: Any, annotations: Any, suffix: Any = None) -> Any:
        write(kind, annotations, suffix)
        raise KeyboardInterrupt

    vector.write_schema_annotations = killed  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        _index(_incremental(second, vector, graph, store))
    vector.write_schema_annotations = write  # type: ignore[method-assign]
    assert store.get_annotation_digest(_NAMESPACE) == REANNOTATING_DIGEST

    # Back to S1, the schema the record held before: rewritten all the same.
    assert _index(_incremental(first, vector, graph, store))
    assert _stored(vector, graph) == _full_build(first)
    assert store.get_annotation_digest(_NAMESPACE) == _digest(first)


def test_a_failed_rewrite_keeps_the_namespace_marked(
    caplog: pytest.LogCaptureFixture,
) -> None:
    vector, graph = _written_without_a_schema()
    config = _config(**_SCHEMA_CONFIG)
    store = FakeDocStatusStore()
    store.put_annotation_digest(_NAMESPACE, UNANNOTATED_DIGEST)
    graph.write_schema_annotations = (  # type: ignore[method-assign]
        lambda kind, annotations, suffix=None: IndexingStats(
            total_items=len(annotations), failed_items=len(annotations)
        )
    )
    with caplog.at_level(logging.INFO):
        assert _index(_incremental(config, vector, graph, store))
    assert "4 writes failed" in caplog.text
    assert "the next run tries again" in caplog.text
    # No count of a pass that did not complete.
    assert "Schema annotations: checked" not in caplog.text
    assert store.get_annotation_digest(_NAMESPACE) == REANNOTATING_DIGEST
    # And the next run, with the same schema, tries again.
    assert annotations_due(_incremental(config, vector, graph, store), _EXISTING)


def test_a_store_error_in_the_pass_warns_and_leaves_it_to_the_next_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    vector, graph = _written_without_a_schema()
    config = _config(**_SCHEMA_CONFIG)
    store = FakeDocStatusStore()
    store.put_annotation_digest(_NAMESPACE, UNANNOTATED_DIGEST)

    def unreachable(kind: Any, suffix: Any = None) -> Any:
        raise AWSServiceError("Neptune read_schema_annotations failed: timeout")

    graph.read_schema_annotations = unreachable  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING):
        assert _index(_incremental(config, vector, graph, store))
    assert "a store failed: AWSServiceError" in caplog.text
    assert store.get_annotation_digest(_NAMESPACE) == REANNOTATING_DIGEST


def test_a_lost_lease_stops_the_pass() -> None:
    vector, graph = _written_without_a_schema()
    store = FakeDocStatusStore()
    store.put_annotation_digest(_NAMESPACE, UNANNOTATED_DIGEST)

    def lost(kind: Any, annotations: Any, suffix: Any = None) -> Any:
        raise NamespaceLeaseError("lease lost")

    graph.write_schema_annotations = lost  # type: ignore[method-assign]
    with pytest.raises(NamespaceLeaseError):
        _index(_incremental(_config(**_SCHEMA_CONFIG), vector, graph, store))
    assert store.get_annotation_digest(_NAMESPACE) == REANNOTATING_DIGEST


class _UnlistedGraph(FakeGraphStore):
    """A graph store with the port's default annotation methods."""

    read_schema_annotations = GraphIndexer.read_schema_annotations  # type: ignore[assignment]
    write_schema_annotations = GraphIndexer.write_schema_annotations  # type: ignore[assignment]


def test_a_store_that_cannot_list_its_items_lets_no_store_be_read() -> None:
    config = _config(**_SCHEMA_CONFIG)
    graph = _UnlistedGraph(Config())
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    reads = _reads(vector)
    result = IndexingManager(
        config, vector_indexer=vector, graph_indexer=graph
    ).reannotate({"default"})
    assert result.unsupported and not result.complete
    assert reads == []
    assert rewrites_schema_annotations(FakeGraphStore(config))
    assert not rewrites_schema_annotations(graph)


@pytest.mark.parametrize(
    "inherited", ["read_schema_annotations", "write_schema_annotations"]
)
def test_a_store_overriding_one_annotation_method_cannot_rewrite(
    inherited: str,
) -> None:
    half = type(
        "Half", (FakeGraphStore,), {inherited: getattr(GraphIndexer, inherited)}
    )
    assert not rewrites_schema_annotations(half(Config()))
    assert rewrites_schema_annotations(FakeVectorStore())


def test_a_borrowed_store_answers_for_the_store_it_wraps() -> None:
    # The pipeline wraps the indexers it does not own (_Borrowed): the
    # wrapper's attributes are the store's, and so is the answer.
    assert not rewrites_schema_annotations(_Borrowed(_UnlistedGraph(Config())))
    assert rewrites_schema_annotations(_Borrowed(FakeGraphStore(Config())))


def test_a_namespace_without_a_record_is_read_once() -> None:
    # Written before the record existed (or by #482, which recorded it per
    # scope): what the stores hold is unknown, so one pass, then recorded.
    vector, graph = _written_without_a_schema()
    config = _config(**_SCHEMA_CONFIG)
    store = FakeDocStatusStore()
    assert _index(_incremental(config, vector, graph, store))
    assert _stored(vector, graph) == _ANNOTATED
    assert store.get_annotation_digest(_NAMESPACE) == _digest(config)
    assert not _index(_incremental(config, vector, graph, store))


def test_a_new_namespace_records_its_digest_without_reading() -> None:
    config = _config(**_SCHEMA_CONFIG)
    _, vector, graph = _manager(Config())
    store = FakeDocStatusStore()
    # Only the run's own new documents are recorded (open_attempts).
    store.put(DocStatusRecord(doc_id="ns#new", content_hash="pending", scope="ns|A"))
    reads = _reads(vector, graph)
    new = DocumentDelta(new=["ns#new"])
    assert not _index(_incremental(config, vector, graph, store), new)
    assert reads == []
    assert store.get_annotation_digest(_NAMESPACE) == _digest(config)

    # A registry that does not report stored_scopes, holding another
    # scope's document: not new.
    store = FakeDocStatusStore()
    store.put(DocStatusRecord(doc_id="ns#other", content_hash="h", scope="ns|B"))
    assert _index(_incremental(config, vector, graph, store), new)


@pytest.mark.parametrize("held_by", ["graph", "vector"])
def test_a_lost_registry_is_no_proof_the_stores_are_empty(held_by: str) -> None:
    # The registry was lost or replaced, the stores kept: its only records
    # are the run's new documents, yet the stores hold items written
    # without a schema. They are read once, as a namespace without a
    # record is, and hold what a full build writes.
    config = _config(**_SCHEMA_CONFIG)
    vector, graph = _written_without_a_schema()
    if held_by == "graph":
        vector.data.clear()
    else:
        graph.data.clear()
    store = FakeDocStatusStore()
    store.put(DocStatusRecord(doc_id="ns#new", content_hash="pending", scope="ns|A"))
    new = DocumentDelta(new=["ns#new"])
    incremental = _incremental(config, vector, graph, store)
    assert annotations_due(incremental, new)
    assert store.get_annotation_digest(_NAMESPACE) == REANNOTATING_DIGEST
    reannotate_on_schema_change(incremental, [])
    assert store.get_annotation_digest(_NAMESPACE) == _digest(config)
    if held_by == "graph":
        assert _stored(vector, graph)[:2] == _ANNOTATED[:2]
    else:  # Vector relationships without an edge keep theirs.
        assert _stored(vector, graph)[2] == _ANNOTATED[2]


def test_a_failed_store_probe_reads_the_namespace(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = _config(**_SCHEMA_CONFIG)
    _, vector, graph = _manager(Config())
    store = FakeDocStatusStore()

    def unreachable(suffix: Any = None) -> Any:
        raise AWSServiceError("Neptune holds_any_item failed: timeout")

    graph.holds_any_item = unreachable  # type: ignore[method-assign]
    with caplog.at_level(logging.WARNING):
        assert annotations_due(
            _incremental(config, vector, graph, store), DocumentDelta(new=["ns#n"])
        )
    assert "Could not tell whether the stores hold items" in caplog.text
    assert store.get_annotation_digest(_NAMESPACE) == REANNOTATING_DIGEST

    def lost(suffix: Any = None) -> Any:
        raise NamespaceLeaseError("lease lost")

    graph.holds_any_item = lost  # type: ignore[method-assign]
    with pytest.raises(NamespaceLeaseError):
        annotations_due(
            _incremental(config, vector, graph, FakeDocStatusStore()),
            DocumentDelta(new=["ns#n"]),
        )


def test_stores_that_cannot_rewrite_are_not_probed() -> None:
    # A pass could not read them: the new namespace records its digest, as
    # before the probe.
    config = _config(**_SCHEMA_CONFIG)
    graph = _UnlistedGraph(Config())
    graph.holds_any_item = None  # type: ignore[assignment,method-assign]
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    store = FakeDocStatusStore()
    assert not annotations_due(
        _incremental(config, vector, graph, store), DocumentDelta(new=["ns#n"])
    )
    assert store.get_annotation_digest(_NAMESPACE) == _digest(config)


@pytest.mark.parametrize("kind", ["entities", "relationships"])
def test_the_ports_default_probe_lists_the_items(kind: str) -> None:
    graph = _UnlistedGraph(Config())
    assert GraphIndexer.holds_any_item(graph) is None
    listing = type(
        "Listing",
        (_UnlistedGraph,),
        {
            "read_schema_annotations": lambda self, k, suffix=None: (
                {"x": object()} if k == kind else {}
            )
        },
    )
    assert GraphIndexer.holds_any_item(listing(Config())) is True
    empty = type(
        "Empty",
        (_UnlistedGraph,),
        {"read_schema_annotations": lambda self, k, suffix=None: {}},
    )
    assert GraphIndexer.holds_any_item(empty(Config())) is False
    # A relationship is an edge listed; an entity alone is none.
    assert GraphIndexer.holds_any_relationship(graph) is None
    assert GraphIndexer.holds_any_relationship(listing(Config())) is (
        kind == "relationships"
    )
    assert GraphIndexer.holds_any_relationship(empty(Config())) is False


class _LostLease:
    def check(self) -> None:
        raise NamespaceLeaseError("lease lost")


@contextmanager
def _lease_lost() -> Iterator[None]:
    token = install_write_fence(_LostLease())
    try:
        yield
    finally:
        remove_write_fence(token)


def test_a_lost_lease_records_no_digest_for_a_new_namespace() -> None:
    config = _config(**_SCHEMA_CONFIG)
    _, vector, graph = _manager(Config())
    store = FakeDocStatusStore()
    store.put(DocStatusRecord(doc_id="ns#new", content_hash="pending", scope="ns|A"))
    with _lease_lost(), pytest.raises(NamespaceLeaseError):
        annotations_due(
            _incremental(config, vector, graph, store), DocumentDelta(new=["ns#new"])
        )
    assert store.get_annotation_digest(_NAMESPACE) is None


def test_a_lost_lease_records_no_rebuilt_digest() -> None:
    manager, _, _ = _manager(_config(**_SCHEMA_CONFIG))
    store = FakeDocStatusStore()
    store.put_annotation_digest("a", REANNOTATING_DIGEST)
    with _lease_lost(), pytest.raises(NamespaceLeaseError):
        record_rebuilt_annotations(store, manager, {"a"}, rebuilt=True)
    assert store.get_annotation_digest("a") == REANNOTATING_DIGEST


def test_a_lost_lease_after_the_pass_keeps_the_namespace_marked() -> None:
    config = _config(**_SCHEMA_CONFIG)
    vector, graph = _written_without_a_schema()
    store = FakeDocStatusStore()
    store.put_annotation_digest(_NAMESPACE, UNANNOTATED_DIGEST)
    incremental = _incremental(config, vector, graph, store)
    assert annotations_due(incremental, _EXISTING)
    with _lease_lost(), pytest.raises(NamespaceLeaseError):
        reannotate_on_schema_change(incremental, [])
    assert store.get_annotation_digest(_NAMESPACE) == REANNOTATING_DIGEST


def test_a_doc_id_another_scope_holds_is_no_new_namespace() -> None:
    # The diff read that scope's record (stored_scopes): the namespace held
    # documents, even though every record it lists is one of the run's new.
    config = _config(**_SCHEMA_CONFIG)
    _, vector, graph = _manager(Config())
    store = FakeDocStatusStore()
    store.put(DocStatusRecord(doc_id="ns#x", content_hash="h", scope="ns|B"))
    delta = DocumentDelta(new=["ns#x"], stored_scopes=["ns|B"])
    assert annotations_due(_incremental(config, vector, graph, store), delta)
    assert store.get_annotation_digest(_NAMESPACE) == REANNOTATING_DIGEST


def test_a_marked_namespace_with_only_new_documents_is_read() -> None:
    # A killed reset left the mark: the pass is made, not skipped as for a
    # namespace without a record.
    config = _config(**_SCHEMA_CONFIG)
    _, vector, graph = _manager(Config())
    store = FakeDocStatusStore()
    store.put_annotation_digest(_NAMESPACE, REANNOTATING_DIGEST)
    store.put(DocStatusRecord(doc_id="ns#new", content_hash="pending", scope="ns|A"))
    reads = _reads(vector, graph)
    incremental = _incremental(config, vector, graph, store)
    assert annotations_due(incremental, DocumentDelta(new=["ns#new"]))
    reannotate_on_schema_change(incremental, [])
    assert reads
    assert store.get_annotation_digest(_NAMESPACE) == _digest(config)


def test_index_delta_marks_the_namespace_before_the_write_ahead(mocker) -> None:
    order: list[str] = []

    class _Stop(Exception):
        pass

    def write_ahead(*_: Any, **__: Any) -> None:
        order.append("write_ahead")
        raise _Stop

    incremental = SimpleNamespace(
        suffix="default",
        check_graph_holds_registry=lambda delta: None,
        write_ahead=write_ahead,
    )
    mocker.patch.object(
        incremental_module, "run_incremental_indexer", return_value=incremental
    )
    mocker.patch.object(incremental_module, "context_lineages", return_value=[])
    mocker.patch.object(
        incremental_module,
        "annotations_due",
        side_effect=lambda *_: order.append("due") or False,
    )
    manager = SimpleNamespace(
        vector_indexer=SimpleNamespace(check_index_settings=lambda s: None)
    )
    context = SimpleNamespace(incremental_delta=DocumentDelta())
    with pytest.raises(_Stop):
        incremental_module.index_delta(
            FakeDocStatusStore, manager, context, [], [], [], [], [], []  # type: ignore[arg-type]
        )
    assert order == ["due", "write_ahead"]


def test_a_rebuild_marks_its_namespaces_until_it_wrote_everything() -> None:
    config = _config(**_SCHEMA_CONFIG)
    manager, _, _ = _manager(config)
    store = FakeDocStatusStore()
    store.put_annotation_digest("a", _digest(config))
    store.put_annotation_digest("b", UNANNOTATED_DIGEST)
    record_rebuilt_annotations(store, manager, {"a", "b", "c"}, rebuilt=False)
    assert store.annotation_digests == {
        "a": _digest(config),
        "b": REANNOTATING_DIGEST,
        "c": REANNOTATING_DIGEST,
    }
    record_rebuilt_annotations(store, manager, {"a", "b", "c"}, rebuilt=True)
    assert set(store.annotation_digests.values()) == {_digest(config)}


def test_a_manager_built_with_a_schema_file_loads_it(tmp_path: Path) -> None:
    schema = tmp_path / "schema.yaml"
    schema.write_text(
        "entity_types:\n"
        "  - {label: ORGANIZATION, iri: 'https://example.org/onto#Organization'}\n"
        "relationship_types: [SUPPLIES]\n",
        encoding="utf-8",
    )
    config = _config(schema_source={"kind": "file", "path": str(schema)})
    manager, vector, graph = _manager(config)
    extraction = manager.config.processing.graph_extraction
    assert extraction.schema_source.kind == "inline"
    assert [str(t) for t in extraction.relationship_types] == ["SUPPLIES"]
    entities, relationships = _items()
    manager.index_all_data(entities=entities, relationships=relationships)
    graph_entities, graph_edges, _, _ = _stored(vector, graph)
    assert graph_entities == {"a": (True, ORG), "b": (False, None)}
    assert graph_edges == {"ab": (True, None), "ba": (False, None)}


def test_a_manager_loads_a_schema_source_with_the_run_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sessions: list[Any] = []

    def load(context: SchemaSourceContext) -> SchemaDocument:
        sessions.append(context.boto_session())
        return SchemaDocument(relationship_types=["SUPPLIES"])

    monkeypatch.setitem(schema_registry._KINDS.entries, "c69_session", load)
    config = _config(schema_source={"kind": "c69_session"})
    session = object()
    providers = Providers(config, boto_session=session)  # type: ignore[arg-type]
    graph = FakeGraphStore(config)
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    IndexingManager(
        config, vector_indexer=vector, graph_indexer=graph, providers=providers
    )
    assert sessions == [session]


def test_stores_written_under_the_earlier_rule_are_reannotated_once() -> None:
    # strict with entity types alone: before #481 the relationships got no
    # in_schema (the store as an upgrade finds it, with no namespace
    # record); now every one is in schema. The first run after the upgrade
    # rewrites them, the next has nothing to rewrite.
    config = _config(schema_mode="strict", entity_types=["ORGANIZATION", "VENDOR"])
    manager, vector, graph = _manager(config)
    entities, relationships = _items()
    manager.index_all_data(entities=entities, relationships=relationships)
    earlier = [r.model_copy(update={"in_schema": None}) for r in relationships]
    graph.upsert_relationships(earlier)
    vector.upsert_relationships(earlier)
    store = FakeDocStatusStore()

    assert _index(_incremental(config, vector, graph, store))
    stored = _stored(vector, graph)
    edges = {"ab": (True, None), "ba": (True, None)}
    assert (stored[1], stored[3]) == (edges, edges)
    assert not _index(_incremental(config, vector, graph, store))
