# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A seed graph: a knowledge graph a user brings, as a seed source reads it.

A seed source (JSONL, CSV, an RDF A-Box, ...) turns its input into a
:class:`SeedGraph` of :class:`SeedNode` and :class:`SeedEdge` statements.
Its content depends only on the source bytes and options, never on their
order: the conversion into virtual documents
(``domain.ingestion.seed_graph.convert_seed``) sorts everything.
"""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from unified_kg_rag.shared.utils import clean_display_name

# A property value: a JSON scalar. ``None`` is accepted and dropped.
SeedPropertyValue = str | int | float | bool


def _scalar_properties(value: Any) -> dict[str, SeedPropertyValue]:
    if not isinstance(value, dict):
        return value  # type: ignore[no-any-return]  # the field type rejects it
    kept: dict[str, Any] = {}
    for key, item in value.items():
        if item is None:
            continue
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"property {key!r} is not a finite number")
        kept[key] = item
    return kept


class _SeedStatement(BaseModel):
    # Forbid: a misspelled key from a loader fails instead of being dropped.
    model_config = ConfigDict(extra="forbid", frozen=True)

    description: str | None = Field(
        default=None, description="What the source says about the item"
    )
    properties: dict[str, SeedPropertyValue] = Field(
        default_factory=dict,
        description="Scalar properties, kept under attributes.seed_properties; "
        "None values are dropped",
    )

    @field_validator("properties", mode="before")
    @classmethod
    def _drop_null_properties(cls, value: Any) -> Any:
        return _scalar_properties(value)


class SeedNode(_SeedStatement):
    """An entity a seed graph states."""

    id: str = Field(
        min_length=1,
        description="Identifier of the node within the seed (what edges name)",
    )
    name: str = Field(
        default="",
        description="Display name; the id when the source gives none. Entity "
        "identity is the name's entity_key, as for extracted entities",
    )
    type: str | None = Field(default=None, description="Type label, as written")
    type_iri: str | None = Field(
        default=None,
        description="IRI of the node's class, mapped to a schema type through "
        "the schema's entity type IRIs",
    )
    iri: str | None = Field(default=None, description="IRI of the node itself")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence of the statement; None: the seed's",
    )

    @model_validator(mode="before")
    @classmethod
    def _name_defaults_to_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("id"), str):
            name = data.get("name")
            if not isinstance(name, str) or not clean_display_name(name):
                data = {**data, "name": data["id"]}
        return data

    @field_validator("name")
    @classmethod
    def _display_name(cls, value: str) -> str:
        return clean_display_name(value)


class SeedEdge(_SeedStatement):
    """A relationship a seed graph states, between two of its nodes."""

    source: str = Field(min_length=1, description="Id of the source node")
    target: str = Field(min_length=1, description="Id of the target node")
    type: str | None = Field(default=None, description="Type label, as written")
    predicate_iri: str | None = Field(
        default=None,
        description="IRI of the edge's property, mapped to a schema type "
        "through the schema's relationship type IRIs",
    )
    weight: float = Field(
        default=1.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Edge weight (added to the weight of the relationship)",
    )


class SeedLoadReport(BaseModel):
    """What a seed source read and left out, as counts only."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    files: int = Field(default=0, ge=0, description="Files read")
    bytes_read: int = Field(default=0, ge=0, description="Bytes read")
    skipped: dict[str, int] = Field(
        default_factory=dict,
        description="Statements the source skipped, by reason (blank nodes, "
        "literals, ...)",
    )


class SeedGraph(BaseModel):
    """The nodes and edges a seed source read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    nodes: tuple[SeedNode, ...] = ()
    edges: tuple[SeedEdge, ...] = ()
    report: SeedLoadReport = Field(default_factory=SeedLoadReport)
