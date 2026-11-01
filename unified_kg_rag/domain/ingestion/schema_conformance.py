# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""How extraction applies the extraction schema to what the model returns.

The schema (``domain.models.schema.ExtractionSchema``) shapes the prompt in
every mode. Once it applies (:attr:`SchemaConformance.applies`), a type
label naming a declared type is written as declared (:meth:`entity_type`,
:meth:`relationship_type`) before any id is built, so a relationship's id
and edge label are the same whichever spelling the model chose. Labels are
compared by ``schema_key`` only; nothing is remapped by similarity.
"""

from __future__ import annotations

from unified_kg_rag.domain.models.schema import ExtractionSchema, schema_key


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
