from typing import Any

from pydantic import Field, model_validator

from .base import Identified


class Covariate(Identified):
    covariate_type: str = Field(default="claim", description="The covariate type")
    subject_id: str = Field(..., description="The subject ID")
    subject_name: str = Field(..., description="The subject name")
    subject_type: str = Field(default="entity", description="The subject type")
    text_unit_ids: list[str] | None = Field(
        None, description="List of text unit IDs in which the covariate info appears"
    )


class Claim(Covariate):
    object_id: str = Field(..., description="The object ID")
    object_name: str = Field(..., description="The object name")
    type: str = Field(..., description="The claim type")
    status: str | None = Field(
        None, description="The claim status (e.g., TRUE, FALSE, DISPUTED, UNKNOWN)"
    )
    start_date: str | None = Field(
        None, description="The start date of the claim (YYYY-MM-DD format)"
    )
    end_date: str | None = Field(
        None, description="The end date of the claim (YYYY-MM-DD format)"
    )
    description: str | None = Field(None, description="Description of the claim")
    description_embedding: list[float] | None = Field(
        None, description="The semantic embedding of the claim description"
    )
    source_text: str | None = Field(
        None, description="The source text that supports this claim"
    )

    @model_validator(mode="before")
    @classmethod
    def set_covariate_type(cls, data: dict[str, Any]) -> dict[str, Any]:
        if isinstance(data, dict):
            data["covariate_type"] = "claim"
        return data
