# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from pydantic import Field

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
