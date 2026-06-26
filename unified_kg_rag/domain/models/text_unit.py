# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from .base import Identified


class TextUnit(Identified):
    text: str = Field(..., description="The text content of the unit")
    translated_texts: dict[str, str] | None = Field(
        default=None, description="Translated text content by language code"
    )
    entity_ids: list[str] | None = Field(
        None, description="List of entity IDs related to the text unit"
    )
    relationship_ids: list[str] | None = Field(
        None, description="List of relationship IDs related to the text unit"
    )
    covariate_ids: dict[str, list[str]] | None = Field(
        None,
        description="Dictionary of different types of covariates related to the text unit",
    )
    community_ids: list[str] | None = Field(
        None, description="List of community IDs related to the text unit"
    )
    document_ids: list[str] | None = Field(
        None, description="List of document IDs in which the text unit appears"
    )
    n_tokens: int | None = Field(None, description="The number of tokens in the text")
