# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import ClassVar

from pydantic import Field

from .base import Identified, SchemaAnnotated


class Relationship(Identified, SchemaAnnotated):
    _annotation_fields: ClassVar[tuple[str, ...]] = ("in_schema", "predicate_iri")

    source_id: str = Field(..., description="The source entity ID")
    source_name: str | None = Field(None, description="The source entity name")
    target_id: str = Field(..., description="The target entity ID")
    target_name: str | None = Field(None, description="The target entity name")
    type: str | None = Field(None, description="The relationship type")
    weight: float | None = Field(1.0, description="The edge weight")
    description: str | None = Field(
        None, description="A description of the relationship"
    )
    description_embedding: list[float] | None = Field(
        None, description="The semantic embedding for the relationship description"
    )
    text_unit_ids: list[str] | None = Field(
        None, description="List of text unit IDs in which the relationship appears"
    )
    rank: int | None = Field(
        default=1,
        description="Sum of the endpoints' graph degrees, set with the entity "
        "rank; higher is more important",
    )
    in_schema: bool | None = Field(
        default=None,
        description="Whether the relationship's type is a declared relationship "
        "type of the extraction schema, set at indexing; None when no schema "
        "applies",
    )
    predicate_iri: str | None = Field(
        default=None,
        description="IRI of the ontology property the relationship's declared "
        "type maps to, set at indexing; None when the type declares none",
    )
