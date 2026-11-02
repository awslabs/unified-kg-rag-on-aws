# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A seed graph: a knowledge graph a user brings, as a seed source reads it.

A seed source (JSONL, CSV, an RDF A-Box, ...) turns its input into a
:class:`SeedGraph` of :class:`SeedNode` and :class:`SeedEdge` statements.
Its content depends only on the source bytes and options, never on their
order: the conversion into virtual documents
(``domain.ingestion.seed_graph.convert_seed``) sorts everything.

Every text of a statement is valid Unicode: a lone surrogate (a JSON
``"\\ud800"`` escape, say) fails validation instead of failing a store write
later. Names, IRIs and type labels are capped (:data:`MAX_NAME_CHARS`,
:data:`MAX_IRI_CHARS`, :data:`MAX_TYPE_CHARS`) so that each, at 4 UTF-8 bytes
a character, stays far under the 32 766-byte limit of an indexed keyword
(an OpenSearch ``name.keyword`` or ``iri`` field).
"""

from __future__ import annotations

import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from unified_kg_rag.shared.utils import clean_display_name

# A property value: a JSON scalar. ``None`` is accepted and dropped.
SeedPropertyValue = str | int | float | bool

# The longest node name (a qualified name included): at most 4 000 UTF-8
# bytes, a fraction of an indexed keyword's 32 766 (name.keyword); far over
# any display name.
MAX_NAME_CHARS = 1000
# The longest IRI (iri, type_iri, predicate_iri): at most 16 000 UTF-8
# bytes, under an indexed keyword's 32 766 (iri); twice the 8 000 octets
# RFC 9110 recommends a URI support.
MAX_IRI_CHARS = 4000
# The longest type label: types are short labels (a mapped ontology type is
# at most 64 characters).
MAX_TYPE_CHARS = 256

_SURROGATE = re.compile(r"[\ud800-\udfff]")


def reject_lone_surrogates(value: str, what: str) -> str:
    """``value``, unless it holds a lone surrogate (a surrogate pair decodes
    to one character, so any surrogate left is lone): no UTF-8 encodes it.

    Raises:
        ValueError: It does.
    """
    if match := _SURROGATE.search(value):
        raise ValueError(
            f"{what} holds a lone surrogate (U+{ord(match.group()):04X}) at "
            f"character {match.start()}: it is not valid Unicode"
        )
    return value


def _scalar_properties(value: Any) -> dict[str, SeedPropertyValue]:
    if not isinstance(value, dict):
        return value  # type: ignore[no-any-return]  # the field type rejects it
    kept: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(key, str):
            reject_lone_surrogates(key, "a property key")
        if item is None:
            continue
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError(f"property {key!r} is not a finite number")
        if isinstance(item, str):
            reject_lone_surrogates(item, f"property {key!r}")
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

    @field_validator("*", mode="after")
    @classmethod
    def _valid_unicode(cls, value: Any) -> Any:
        if isinstance(value, str):
            return reject_lone_surrogates(value, "the text")
        return value


class SeedNode(_SeedStatement):
    """An entity a seed graph states."""

    id: str = Field(
        min_length=1,
        description="Identifier of the node within the seed (what edges name)",
    )
    name: str = Field(
        default="",
        max_length=MAX_NAME_CHARS,
        description="Display name; the id when the source gives none. Entity "
        "identity is the name's entity_key, as for extracted entities",
    )
    type: str | None = Field(
        default=None, max_length=MAX_TYPE_CHARS, description="Type label, as written"
    )
    type_iri: str | None = Field(
        default=None,
        max_length=MAX_IRI_CHARS,
        description="IRI of the node's class, mapped to a schema type through "
        "the schema's entity type IRIs",
    )
    iri: str | None = Field(
        default=None, max_length=MAX_IRI_CHARS, description="IRI of the node itself"
    )
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
    type: str | None = Field(
        default=None, max_length=MAX_TYPE_CHARS, description="Type label, as written"
    )
    predicate_iri: str | None = Field(
        default=None,
        max_length=MAX_IRI_CHARS,
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
