# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The schema annotations stored with entities and relationships.

Indexing writes, on every entity, ``in_schema`` (whether its type is a
declared entity type) and ``type_iri`` (the ``iri`` of that type), and on
every relationship ``in_schema`` and ``predicate_iri`` (of its declared
relationship type). Each is a pure function of the item's stored type (the
merged, voted one) and the schema, computed when the item is written, so a
full build and an incremental run store the same annotations. Whether a
relationship fits a pattern is not stored: it depends on its endpoints'
types, which other runs' documents can change without touching the edge.

An annotation is ``None`` where no schema applies: under ``off``, under
``align`` with entity types alone (extraction then leaves every record as
before the schema keys, ``SchemaConformance.applies``), and for one kind of
item when the schema declares no type of that kind (relationships, with
entity types only). An IRI is ``None`` for a type that declares none.
"""

from __future__ import annotations

from collections.abc import Iterable

from unified_kg_rag.domain.ingestion.schema_conformance import SchemaConformance
from unified_kg_rag.domain.models import Entity, Relationship
from unified_kg_rag.domain.models.schema import ExtractionSchema, schema_key

# (in_schema, iri) of one item.
Annotation = tuple[bool | None, str | None]
_NONE: Annotation = (None, None)


class SchemaAnnotator:
    """The annotations an extraction schema gives entities and relationships."""

    def __init__(self, schema: ExtractionSchema) -> None:
        applies = SchemaConformance(schema).applies
        # Type key -> the declared type's IRI; None: that kind is not annotated.
        self._entity_iris: dict[str, str | None] | None = (
            {schema_key(spec.label): spec.iri for spec in schema.entity_types}
            if applies and schema.entity_types
            else None
        )
        self._relationship_iris: dict[str, str | None] | None = (
            {schema_key(spec.label): spec.iri for spec in schema.relationship_types}
            if applies and schema.relationship_types
            else None
        )

    @property
    def active(self) -> bool:
        """Whether any item gets an annotation."""
        return self._entity_iris is not None or self._relationship_iris is not None

    @staticmethod
    def _annotation(
        iris: dict[str, str | None] | None, label: str | None
    ) -> Annotation:
        if iris is None:
            return _NONE
        key = schema_key(label or "")
        if key not in iris:
            return False, None
        return True, iris[key]

    def entity_annotation(self, entity_type: str | None) -> Annotation:
        """``(in_schema, type_iri)`` of an entity of ``entity_type`` (empty
        included: an untyped entity is off schema)."""
        return self._annotation(self._entity_iris, entity_type)

    def relationship_annotation(self, relationship_type: str | None) -> Annotation:
        """``(in_schema, predicate_iri)`` of a relationship of
        ``relationship_type``."""
        return self._annotation(self._relationship_iris, relationship_type)

    def entity(self, entity: Entity) -> Entity:
        """``entity`` with its annotations; itself when they already hold."""
        in_schema, iri = self.entity_annotation(entity.type)
        if (entity.in_schema, entity.type_iri) == (in_schema, iri):
            return entity
        return entity.model_copy(update={"in_schema": in_schema, "type_iri": iri})

    def relationship(self, rel: Relationship) -> Relationship:
        """``rel`` with its annotations; itself when they already hold."""
        in_schema, iri = self.relationship_annotation(rel.type)
        if (rel.in_schema, rel.predicate_iri) == (in_schema, iri):
            return rel
        return rel.model_copy(update={"in_schema": in_schema, "predicate_iri": iri})

    def entities(self, entities: Iterable[Entity]) -> list[Entity]:
        """Each of ``entities`` annotated (:meth:`entity`)."""
        return [self.entity(e) for e in entities]

    def relationships(
        self, relationships: Iterable[Relationship]
    ) -> list[Relationship]:
        """Each of ``relationships`` annotated (:meth:`relationship`)."""
        return [self.relationship(r) for r in relationships]
