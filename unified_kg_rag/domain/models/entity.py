# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import ClassVar

from pydantic import Field

from .base import Named, SchemaAnnotated


class Entity(Named, SchemaAnnotated):
    _annotation_fields: ClassVar[tuple[str, ...]] = (
        "in_schema",
        "type_iri",
        "iri",
        "source_kinds",
    )

    type: str | None = Field(None, description="Type of the entity")
    description: str | None = Field(None, description="Description of the entity")
    description_embedding: list[float] | None = Field(
        None, description="The semantic embedding of the entity description"
    )
    text_unit_ids: list[str] | None = Field(
        None, description="List of text unit IDs in which the entity appears"
    )
    community_ids: list[str] | None = Field(
        None, description="The community IDs of the entity"
    )
    rank: int | None = Field(
        default=1,
        description="Graph degree of the entity (the number of distinct entities "
        "it is related to), set at graph resolution and on incremental merges; "
        "higher is more important",
    )
    confidence: float | None = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the entity extraction (0.0-1.0). Higher values indicate more reliable extraction from source text.",
    )
    in_schema: bool | None = Field(
        default=None,
        description="Whether the entity's type is a declared entity type of the "
        "extraction schema (any type, when it declares none), set at indexing; "
        "None when no schema applies",
    )
    type_iri: str | None = Field(
        default=None,
        description="IRI of the ontology class the entity's declared type maps "
        "to, set at indexing unless schema_mode is off; None when the type "
        "declares none",
    )
    iri: str | None = Field(
        default=None,
        description="IRI of the entity itself, as a seed graph states it (the "
        "smallest when its statements give several); None when none does",
    )
    source_kinds: list[str] | None = Field(
        default=None,
        description="The kinds of the text units the entity cites, sorted "
        "(document, seed_graph), set at indexing; None when it cites no seed "
        "graph unit (document-only)",
    )
