from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Identified(BaseModel):
    id: str = Field(..., description="The unique identifier of the item")
    short_id: str | None = Field(None, description="Human readable ID")
    attributes: dict[str, Any] | None = Field(
        None, description="Additional attributes associated with the item"
    )
    created_at: datetime | None = Field(
        None, description="The timestamp when the item was created"
    )
    updated_at: datetime | None = Field(
        None, description="The timestamp when the item was updated"
    )


class Named(Identified):
    name: str = Field(..., description="The name/title of the item")
    name_embedding: list[float] | None = Field(
        None, description="The semantic embedding of the item name"
    )
