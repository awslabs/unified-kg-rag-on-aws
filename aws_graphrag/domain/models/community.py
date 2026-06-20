# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from pydantic import Field

from .base import Named


class Community(Named):
    level: str = Field(..., description="Community level")
    parent: str = Field(
        ..., description="Community ID of the parent node of this community"
    )
    children: list[str] = Field(
        ..., description="List of community IDs of the child nodes of this community"
    )
    entity_ids: list[str] | None = Field(
        None, description="List of entity IDs related to the community"
    )
    relationship_ids: list[str] | None = Field(
        None, description="List of relationship IDs related to the community"
    )
    covariate_ids: dict[str, list[str]] | None = Field(
        None,
        description="Dictionary of different types of covariates related to the community",
    )
    text_unit_ids: list[str] | None = Field(
        None, description="List of text unit IDs related to the community"
    )
    size: int | None = Field(
        None, description="The size of the community (Amount of text units)"
    )
    period: str | None = Field(None, description="The period of the community")
