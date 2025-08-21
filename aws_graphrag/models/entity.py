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
