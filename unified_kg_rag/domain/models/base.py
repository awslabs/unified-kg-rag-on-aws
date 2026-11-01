# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from datetime import datetime
from typing import Any, ClassVar

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
)


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


class SchemaAnnotated(BaseModel):
    """An item that can carry schema annotations (``in_schema`` and an IRI).

    Indexing sets them from the item's type and the extraction schema
    (``domain.ingestion.schema_annotation``); with no schema in effect they
    are ``None`` and a dump leaves them out, so the stores, the extraction
    records and the caches hold what they held before the fields existed.
    """

    # The annotation fields a dump leaves out while they are None.
    _annotation_fields: ClassVar[tuple[str, ...]] = ()

    @model_serializer(mode="wrap")
    def _without_unset_annotations(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        data: dict[str, Any] = handler(self)
        for name in self._annotation_fields:
            if name in data and data[name] is None:
                del data[name]
        return data
