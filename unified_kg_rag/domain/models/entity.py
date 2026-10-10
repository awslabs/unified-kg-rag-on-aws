# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Literal

from pydantic import BaseModel, Field

from .base import Named


class Entity(Named):
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
        1,
        description="Rank of the entity, used for sorting. Higher rank indicates more important entity",
    )
    frequency: int | None = Field(
        None,
        description="Number of text units supporting the entity (recomputed on merge)",
    )
    confidence: float | None = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score of the entity extraction (0.0-1.0). Higher values indicate more reliable extraction from source text.",
    )


class RejectedEntity(BaseModel):
    """An entity graph extraction rejected for one text unit.

    Extraction drops an entity whose evidence is not in its chunk (entity
    grounding). Gleaning runs later on the same chunk and must not bring it
    back, through a gleaned relationship that names it, so the stage hands
    these on (and caches them with its output).
    """

    text_unit_id: str = Field(description="The text unit the entity came from")
    entity_key: str = Field(description="Identity key of the entity's name")
    reason: Literal["ungrounded"] = Field(
        description="ungrounded: the grounding guard dropped it"
    )
