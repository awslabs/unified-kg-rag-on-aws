# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from pydantic import Field

from .base import Identified


class Relationship(Identified):
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
        1,
        description="Rank of the relationship, used for sorting. Higher rank indicates more important relationship",
    )
