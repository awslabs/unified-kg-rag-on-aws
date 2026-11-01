# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""How extraction applies the extraction schema to what the model returns.

The schema (``domain.models.schema.ExtractionSchema``) shapes the prompt in
every mode. Once it applies (:attr:`SchemaConformance.applies`), a type
label naming a declared type is written as declared (:meth:`entity_type`,
:meth:`relationship_type`) before any id is built, so a relationship's id
and edge label are the same whichever spelling the model chose. Labels are
compared by ``schema_key`` only; nothing is remapped by similarity.

:meth:`SchemaConformance.conform` then checks one text unit's statements:

- an entity is off schema when entity types are declared and its type
  (empty included) is none of them;
- a relationship is off schema when relationship types are declared and
  its type is none of them;
- a relationship of a type with patterns is off pattern when the unit lists
  both endpoints with a declared type and no pattern of the type joins them
  in that direction (a reversed pattern is not flipped). An endpoint the
  unit does not list, or lists with an undeclared type, is unknown, and an
  unknown endpoint passes.

``align`` keeps everything and counts. ``strict`` drops an off-schema entity
(its key is rejected for the unit as ``off_schema`` unless the unit also
states it with a declared type), the unit's relationships naming it, and
the off-schema and off-pattern relationships. A check is per unit, so a full
build and an incremental run drop the same statements.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from pydantic import BaseModel, Field

from unified_kg_rag.domain.models import Entity, Relationship
from unified_kg_rag.domain.models.schema import ExtractionSchema, schema_key
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import entity_key

logger = get_logger(__name__)

# The counters of a check, in the order the summaries report them.
SCHEMA_COUNTERS = (
    "entities_off_schema",
    "entities_dropped_off_schema",
    "relationships_off_schema",
    "relationships_dropped_off_schema",
    "relationships_dropped_off_schema_endpoint",
    "relationships_off_pattern",
    "relationships_dropped_off_pattern",
)


class UnitConformance(BaseModel):
    """One text unit's statements after the schema check, and its counts."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    off_schema_keys: set[str] = Field(
        default_factory=set,
        description="entity_key of each entity strict dropped as off schema "
        "and the unit does not also state with a declared type",
    )
    off_schema_labels: Counter[str] = Field(
        default_factory=Counter,
        description="The off-schema type labels and how often (for the debug log)",
    )
    entities_off_schema: int = 0
    entities_dropped_off_schema: int = 0
    relationships_off_schema: int = 0
    relationships_dropped_off_schema: int = 0
    relationships_dropped_off_schema_endpoint: int = 0
    relationships_off_pattern: int = 0
    relationships_dropped_off_pattern: int = 0


def unit_entity_types(entities: Iterable[Entity]) -> dict[str, str]:
    """The type each entity key of a unit gets: the most frequent of its
    statements' types, lowercased, the smallest on a tie (the record's merge
    rule, ``contributions.merge_entity_contributions``)."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for entity in entities:
        if value := (entity.type or "").lower():
            counts[entity_key(entity.name)][value] += 1
    return {
        key: min(votes, key=lambda label: (-votes[label], label))
        for key, votes in counts.items()
    }


class SchemaConformance:
    """The extraction schema as extraction applies it."""

    def __init__(self, schema: ExtractionSchema) -> None:
        self.schema = schema
        # Entity types alone, as every configuration had before the schema
        # keys, only shape the prompt: align then leaves the records as they
        # were, so no record keyed before changes content.
        self.applies = schema.mode == "strict" or (
            schema.mode == "align" and bool(schema.relationship_types)
        )
        self.strict = schema.mode == "strict"
        self._entity_labels = {
            schema_key(spec.label): spec.label for spec in schema.entity_types
        }
        self._relationship_labels = {
            schema_key(spec.label): spec.label for spec in schema.relationship_types
        }
        # Relationship type key -> the (source, target) type keys allowed.
        self._patterns: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for pattern in schema.relationship_patterns:
            self._patterns[schema_key(pattern.relation)].add(
                (schema_key(pattern.source_type), schema_key(pattern.target_type))
            )

    def entity_type(self, label: str) -> str:
        """``label`` as declared when it names a declared entity type and the
        schema applies; else as written."""
        if not self.applies or not label:
            return label
        return self._entity_labels.get(schema_key(label), label)

    def relationship_type(self, label: str) -> str:
        """``label`` as declared when it names a declared relationship type
        and the schema applies; else as written."""
        if not self.applies or not label:
            return label
        return self._relationship_labels.get(schema_key(label), label)

    def entity_off_schema(self, label: str | None) -> bool:
        """Whether the schema applies, declares entity types, and ``label``
        (empty included) is none of them."""
        return (
            self.applies
            and bool(self._entity_labels)
            and schema_key(label or "") not in self._entity_labels
        )

    def relationship_off_schema(self, label: str | None) -> bool:
        """Whether the schema applies, declares relationship types, and
        ``label`` is none of them."""
        return (
            self.applies
            and bool(self._relationship_labels)
            and schema_key(label or "") not in self._relationship_labels
        )

    def off_pattern(
        self,
        relation: str | None,
        source_type: str | None,
        target_type: str | None,
    ) -> bool:
        """Whether a ``relation`` from an entity of ``source_type`` to one of
        ``target_type`` breaks the relation's patterns.

        False when the schema does not apply, the relation has no pattern, or
        an endpoint type is unknown (``None``, or not a declared type).
        """
        if not self.applies:
            return False
        allowed = self._patterns.get(schema_key(relation or ""))
        if not allowed or source_type is None or target_type is None:
            return False
        source, target = schema_key(source_type), schema_key(target_type)
        if source not in self._entity_labels or target not in self._entity_labels:
            return False
        return (source, target) not in allowed

    def conform(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        *,
        unit: str | None = None,
    ) -> UnitConformance:
        """Check one unit's statements (:func:`conform_entities`, then
        :func:`conform_relationships`); ``unit`` names it in the debug log."""
        result = self.conform_entities(entities, unit=unit)
        return self.conform_relationships(relationships, result, unit=unit)

    def conform_entities(
        self, entities: list[Entity], *, unit: str | None = None
    ) -> UnitConformance:
        """The unit's entities checked: strict drops the off-schema ones."""
        result = UnitConformance(entities=list(entities))
        if not self.applies or not self._entity_labels:
            return result
        kept: list[Entity] = []
        for entity in entities:
            if not self.entity_off_schema(entity.type):
                kept.append(entity)
                continue
            result.entities_off_schema += 1
            result.off_schema_labels[entity.type or ""] += 1
            if not self.strict:
                kept.append(entity)
                continue
            result.entities_dropped_off_schema += 1
            result.off_schema_keys.add(entity_key(entity.name))
            logger.debug(
                "Dropping entity '%s' of off-schema type '%s' in chunk '%s'",
                entity.name,
                entity.type,
                unit,
            )
        # A statement of the entity with a declared type keeps it listed.
        result.off_schema_keys -= {entity_key(entity.name) for entity in kept}
        result.entities = kept
        return result

    def conform_relationships(
        self,
        relationships: list[Relationship],
        result: UnitConformance,
        *,
        unit: str | None = None,
    ) -> UnitConformance:
        """``result`` with the unit's relationships checked against its
        checked entities (``result.entities``)."""
        if not self.applies:
            result.relationships = list(relationships)
            return result
        types = unit_entity_types(result.entities)
        kept: list[Relationship] = []
        for rel in relationships:
            source = entity_key(rel.source_name or "")
            target = entity_key(rel.target_name or "")
            if self.strict and {source, target} & result.off_schema_keys:
                result.relationships_dropped_off_schema_endpoint += 1
                logger.debug(
                    "Dropping relationship '%s': an endpoint was dropped as off "
                    "schema in chunk '%s'",
                    rel.short_id,
                    unit,
                )
                continue
            if self.relationship_off_schema(rel.type):
                result.relationships_off_schema += 1
                result.off_schema_labels[rel.type or ""] += 1
                if self.strict:
                    result.relationships_dropped_off_schema += 1
                    logger.debug(
                        "Dropping relationship '%s' of off-schema type '%s' in "
                        "chunk '%s'",
                        rel.short_id,
                        rel.type,
                        unit,
                    )
                    continue
            elif self.off_pattern(rel.type, types.get(source), types.get(target)):
                result.relationships_off_pattern += 1
                if self.strict:
                    result.relationships_dropped_off_pattern += 1
                    logger.debug(
                        "Dropping relationship '%s' (%s)-[%s]->(%s): no pattern "
                        "allows it, in chunk '%s'",
                        rel.short_id,
                        types.get(source),
                        rel.type,
                        types.get(target),
                        unit,
                    )
                    continue
            kept.append(rel)
        result.relationships = kept
        return result
