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

``in_schema`` is ``None`` where the schema does not apply: under ``off``
and under ``align`` with entity types alone (extraction then leaves every
record as before the schema keys, ``SchemaConformance.applies``). Where it
applies, both kinds get it: a kind the schema declares no type of has no
off-schema item (extraction checks none of it, and ``strict`` kept every
one), so all its items are in schema. A filter on ``in_schema`` then never
empties one kind for want of the field. An IRI is written whenever the
item's type names a declared type with an ``iri``, unless the mode is
``off``: under ``align`` with entity types alone too, so declaring a class
IRI is enough to link the entities of that type. It is ``None`` for any
other type. :attr:`SchemaAnnotator.fields` names the fields some item can
get, which the query side checks a filter key against.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from unified_kg_rag.domain.ingestion.schema_conformance import SchemaConformance
from unified_kg_rag.domain.models import Entity, Relationship
from unified_kg_rag.domain.models.schema import ExtractionSchema, schema_key

# (in_schema, iri) of one item.
Annotation = tuple[bool | None, str | None]

# Bump when the annotation rules change for the same schema: every recorded
# digest then differs, and the next incremental run rewrites the annotations.
# 2: a kind with no declared type is in schema wherever the schema applies.
ANNOTATION_VERSION = 2


class SchemaAnnotator:
    """The annotations an extraction schema gives entities and relationships."""

    def __init__(self, schema: ExtractionSchema) -> None:
        self._applies = SchemaConformance(schema).applies
        # The declared type keys of each kind; empty: every type is in schema.
        self._entity_keys = frozenset(
            schema_key(spec.label) for spec in schema.entity_types
        )
        self._relationship_keys = frozenset(
            schema_key(spec.label) for spec in schema.relationship_types
        )
        # Type key -> the IRI of a declared type that has one.
        on = schema.mode != "off"
        self._entity_iris: dict[str, str] = {
            schema_key(spec.label): spec.iri
            for spec in schema.entity_types
            if on and spec.iri
        }
        self._relationship_iris: dict[str, str] = {
            schema_key(spec.label): spec.iri
            for spec in schema.relationship_types
            if on and spec.iri
        }

    @property
    def fields(self) -> frozenset[str]:
        """The annotation fields some item can get: ``in_schema`` where the
        schema applies, ``type_iri`` / ``predicate_iri`` where a declared type
        of that kind has an IRI. A filter on another one matches nothing."""
        return frozenset(
            name
            for name, written in (
                ("in_schema", self._applies),
                ("type_iri", bool(self._entity_iris)),
                ("predicate_iri", bool(self._relationship_iris)),
            )
            if written
        )

    @property
    def digest(self) -> str:
        """Identifies the annotations this annotator gives: two with one
        digest annotate every item alike (SHA-256, hex, of the type keys
        that get ``in_schema`` where the schema applies, and the IRI of each
        type key). Recorded with what an incremental run indexed, so a
        schema edit that changes only the annotations rewrites them without
        extracting again (``IndexingManager.reannotate``)."""
        state = [
            ANNOTATION_VERSION,
            sorted(self._entity_keys) if self._applies else None,
            sorted(self._relationship_keys) if self._applies else None,
            sorted(self._entity_iris.items()),
            sorted(self._relationship_iris.items()),
        ]
        payload = json.dumps(state, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _annotation(
        self, keys: frozenset[str], iris: dict[str, str], label: str | None
    ) -> Annotation:
        key = schema_key(label or "")
        in_schema = (not keys or key in keys) if self._applies else None
        return in_schema, iris.get(key)

    def entity_annotation(self, entity_type: str | None) -> Annotation:
        """``(in_schema, type_iri)`` of an entity of ``entity_type`` (empty
        included: an untyped entity is off schema)."""
        return self._annotation(self._entity_keys, self._entity_iris, entity_type)

    def relationship_annotation(self, relationship_type: str | None) -> Annotation:
        """``(in_schema, predicate_iri)`` of a relationship of
        ``relationship_type``."""
        return self._annotation(
            self._relationship_keys, self._relationship_iris, relationship_type
        )

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


# The digest of an annotator that annotates nothing: what every item written
# without schema annotations holds (no schema, or one that gives none).
UNANNOTATED_DIGEST = SchemaAnnotator(ExtractionSchema()).digest
