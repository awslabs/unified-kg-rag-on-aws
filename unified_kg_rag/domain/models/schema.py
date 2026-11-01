# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The extraction schema: the entity and relationship types graph extraction
works with, and the patterns (source type, relation, target type) allowed
between them.

The configuration holds the schema inline (``processing.graph_extraction``:
``entity_types``, ``relationship_types``, ``relationship_patterns``) or names
a source that fills those keys at start-up (``schema_source``); either way
:meth:`ExtractionSchema.from_config` reads it from the inline keys. A label
is compared by :func:`schema_key` only: case, surrounding whitespace and the
spelling of word separators do not matter, anything else does (no synonyms,
plurals or edit distance).

An optional ``iri`` names the type in an external ontology (a class for an
entity type, a property for a relationship type); indexing stores it on the
items of the type (``domain.ingestion.schema_annotation``), so they can be
linked to it.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    StringConstraints,
    Tag,
    model_validator,
)

if TYPE_CHECKING:
    from .config import GraphExtractionConfig

# off: the schema only shapes the prompt. align (default): every item is
# kept and in-schema labels are written as declared. strict: items without a
# declared type, and relationships no pattern allows, are dropped. How
# extraction applies them: domain.ingestion.schema_conformance.
SchemaMode = Literal["off", "align", "strict"]

_SEPARATORS = re.compile(r"[\s_]+")


def schema_key(label: str) -> str:
    """The key a type label is compared by.

    NFKC-normalized, trimmed, each run of whitespace and underscores made one
    ``_``, then case-folded and normalized again (folding can leave a
    decomposed sequence, "ß" + U+0301 → "ss" + U+0301, which would key apart
    from its composed spelling): ``"works for"``, ``"WORKS_FOR"`` and
    ``" Works__For "`` share a key, and a key is its own key. Nothing
    looser: ``"PERSONS"`` is not ``"PERSON"``, ``"Org"`` is not
    ``"ORGANIZATION"`` and ``"works-for"`` is not ``"works for"``.
    """
    spaced = _SEPARATORS.sub("_", unicodedata.normalize("NFKC", label).strip())
    return unicodedata.normalize("NFKC", spaced.casefold())


_Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_Description = Annotated[str, StringConstraints(strip_whitespace=True)]
# An IRI has no whitespace (RFC 3987); its form is otherwise not checked.
_Iri = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, pattern=r"^\S+$")
]


class _TypeSpec(BaseModel):
    # Forbid: list items escape the config loader's unknown-key check, so a
    # misspelled key fails here instead of being dropped.
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: _Label = Field(
        description="The type's name, as the extraction prompt shows it; "
        "compared case- and separator-insensitively"
    )
    description: _Description = Field(
        default="", description="What the type covers, shown to the model"
    )


class EntityTypeSpec(_TypeSpec):
    """An entity type of the extraction schema."""

    iri: _Iri | None = Field(
        default=None,
        description="IRI of the ontology class this type maps to, stored as "
        "type_iri on its entities",
    )


class RelationshipTypeSpec(_TypeSpec):
    """A relationship type of the extraction schema."""

    iri: _Iri | None = Field(
        default=None,
        description="IRI of the ontology property this type maps to, stored as "
        "predicate_iri on its relationships",
    )


# "SOURCE -RELATION-> TARGET": whitespace before the opening '-'; the
# relation runs to the last '->'.
_PATTERN = re.compile(
    r"^\s*(?P<source>\S.*?)\s+-\s*(?P<relation>\S.*?)\s*->\s*(?P<target>\S.*?)\s*$"
)


class RelationshipPattern(BaseModel):
    """A relationship type allowed from one entity type to another."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_type: _Label = Field(description="Entity type label of the source")
    relation: _Label = Field(description="Relationship type label")
    target_type: _Label = Field(description="Entity type label of the target")

    @classmethod
    def parse(cls, text: str) -> RelationshipPattern:
        """The pattern written ``"SOURCE -RELATION-> TARGET"``.

        Raises:
            ValueError: ``text`` is not of that form.
        """
        match = _PATTERN.match(text)
        if match is None:
            raise ValueError(
                f"relationship pattern {text!r} is not of the form "
                "'SOURCE -RELATION-> TARGET'"
            )
        return cls(
            source_type=match["source"],
            relation=match["relation"],
            target_type=match["target"],
        )

    def __str__(self) -> str:
        return f"{self.source_type} -{self.relation}-> {self.target_type}"


EntityTypeItem = str | EntityTypeSpec
RelationshipTypeItem = str | RelationshipTypeSpec
RelationshipPatternItem = str | RelationshipPattern


def _item_form(value: Any) -> str:
    return "text" if isinstance(value, str) else "spec"


# The configured forms: a string or a mapping, told apart by type, so a
# mapping's error names its own fields and not also "not a string".
EntityTypeField = Annotated[
    Annotated[str, Tag("text")] | Annotated[EntityTypeSpec, Tag("spec")],
    Discriminator(_item_form),
]
RelationshipTypeField = Annotated[
    Annotated[str, Tag("text")] | Annotated[RelationshipTypeSpec, Tag("spec")],
    Discriminator(_item_form),
]
RelationshipPatternField = Annotated[
    Annotated[str, Tag("text")] | Annotated[RelationshipPattern, Tag("spec")],
    Discriminator(_item_form),
]


class SchemaDocument(BaseModel):
    """What a schema source supplies: the inline schema keys it fills, in
    the forms the config takes. ``entity_types`` ``None``: the config's
    stay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_types: list[EntityTypeField] | None = None
    relationship_types: list[RelationshipTypeField] = Field(default_factory=list)
    relationship_patterns: list[RelationshipPatternField] = Field(default_factory=list)


def _split_label(item: str) -> tuple[str, str]:
    """``(label, description)`` of a ``"LABEL: description"`` item."""
    label, _, description = item.partition(":")
    return label.strip(), description.strip()


def entity_type_spec(item: EntityTypeItem) -> EntityTypeSpec | None:
    """The spec of a configured entity type; ``None`` for a string item with
    no label (``": text"``), which the prompt has always skipped."""
    if isinstance(item, EntityTypeSpec):
        return item
    label, description = _split_label(item)
    return EntityTypeSpec(label=label, description=description) if label else None


def relationship_type_spec(item: RelationshipTypeItem) -> RelationshipTypeSpec:
    """The spec of a configured relationship type.

    Raises:
        ValueError: A string item has no label.
    """
    if isinstance(item, RelationshipTypeSpec):
        return item
    label, description = _split_label(item)
    if not label:
        raise ValueError(f"relationship type {item!r} has no label")
    return RelationshipTypeSpec(label=label, description=description)


def relationship_pattern(item: RelationshipPatternItem) -> RelationshipPattern:
    """The pattern of a configured item (``"SOURCE -RELATION-> TARGET"``).

    Raises:
        ValueError: A string item is not of that form.
    """
    if isinstance(item, RelationshipPattern):
        return item
    return RelationshipPattern.parse(item)


def _duplicates(labels: Iterable[str]) -> list[str]:
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for label in labels:
        key = schema_key(label)
        if key in seen:
            duplicates.append(f"{seen[key]!r} and {label!r}")
        else:
            seen[key] = label
    return duplicates


def schema_problems(
    entity_types: Sequence[EntityTypeSpec],
    relationship_types: Sequence[RelationshipTypeSpec],
    relationship_patterns: Sequence[RelationshipPattern],
) -> list[str]:
    """What makes a schema inconsistent, one message each (empty: none).

    Two types of one kind may not share a :func:`schema_key`, and a pattern
    names a declared entity type at each end and a declared relationship
    type.
    """
    problems = [
        f"entity_types: {pair} name the same type"
        for pair in _duplicates(spec.label for spec in entity_types)
    ] + [
        f"relationship_types: {pair} name the same type"
        for pair in _duplicates(spec.label for spec in relationship_types)
    ]
    if not relationship_patterns:
        return problems
    entities = {schema_key(spec.label) for spec in entity_types}
    relations = {schema_key(spec.label) for spec in relationship_types}
    for pattern in relationship_patterns:
        unknown = [
            f"entity type {label!r}"
            for label in (pattern.source_type, pattern.target_type)
            if schema_key(label) not in entities
        ]
        if schema_key(pattern.relation) not in relations:
            unknown.append(f"relationship type {pattern.relation!r}")
        if unknown:
            problems.append(
                f"relationship_patterns: '{pattern}' names the undeclared "
                + " and the undeclared ".join(unknown)
            )
    return problems


class ExtractionSchema(BaseModel):
    """The extraction schema in effect: types, patterns and mode.

    Built from the inline configuration (:meth:`from_config`). ``digest``
    identifies the content (types, descriptions, IRIs and patterns, in
    order), not the mode or where it was loaded from.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_types: tuple[EntityTypeSpec, ...] = ()
    relationship_types: tuple[RelationshipTypeSpec, ...] = ()
    relationship_patterns: tuple[RelationshipPattern, ...] = ()
    mode: SchemaMode = "align"

    @model_validator(mode="after")
    def _consistent(self) -> ExtractionSchema:
        problems = schema_problems(
            self.entity_types, self.relationship_types, self.relationship_patterns
        )
        if problems:
            raise ValueError("; ".join(problems))
        return self

    @classmethod
    def from_items(
        cls,
        entity_types: Iterable[EntityTypeItem] = (),
        relationship_types: Iterable[RelationshipTypeItem] = (),
        relationship_patterns: Iterable[RelationshipPatternItem] = (),
        mode: SchemaMode = "align",
    ) -> ExtractionSchema:
        """The schema of configured items, in either of their forms.

        Raises:
            ValueError: An item does not parse or the schema is inconsistent
                (:func:`schema_problems`).
        """
        return cls(
            entity_types=tuple(
                spec for item in entity_types if (spec := entity_type_spec(item))
            ),
            relationship_types=tuple(
                relationship_type_spec(item) for item in relationship_types
            ),
            relationship_patterns=tuple(
                relationship_pattern(item) for item in relationship_patterns
            ),
            mode=mode,
        )

    @classmethod
    def from_config(cls, config: GraphExtractionConfig) -> ExtractionSchema:
        """The schema ``processing.graph_extraction`` holds inline."""
        return cls.from_items(
            config.entity_types,
            config.relationship_types,
            config.relationship_patterns,
            config.schema_mode,
        )

    @property
    def digest(self) -> str:
        """A content hash (SHA-256, hex) of the types and patterns."""
        content = self.model_dump(mode="json", exclude={"mode"})
        payload = json.dumps(content, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def entity_type(self, label: str) -> EntityTypeSpec | None:
        """The declared entity type ``label`` names (:func:`schema_key`)."""
        key = schema_key(label)
        return next(
            (spec for spec in self.entity_types if schema_key(spec.label) == key),
            None,
        )

    def relationship_type(self, label: str) -> RelationshipTypeSpec | None:
        """The declared relationship type ``label`` names (:func:`schema_key`)."""
        key = schema_key(label)
        return next(
            (spec for spec in self.relationship_types if schema_key(spec.label) == key),
            None,
        )
