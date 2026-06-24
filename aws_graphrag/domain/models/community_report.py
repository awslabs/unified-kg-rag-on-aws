# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import Field

from .base import Named


class CommunityReport(Named):
    community_id: str = Field(
        description="The ID of the community this report is associated with"
    )
    summary: str = Field("", description="Summary of the report")
    summary_embedding: list[float] | None = Field(
        None, description="The semantic embedding of the summary"
    )
    full_content: str = Field("", description="Full content of the report")
    full_content_embedding: list[float] | None = Field(
        None, description="The semantic embedding of the full report content"
    )
    rank: int | None = Field(
        1,
        description="Rank of the report, used for sorting. Higher means more important",
    )
    size: int | None = Field(
        None, description="The size of the report (Amount of text units)"
    )
    period: str | None = Field(None, description="The period of the report")
