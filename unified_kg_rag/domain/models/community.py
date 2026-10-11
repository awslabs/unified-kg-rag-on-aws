# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, Field

from .base import Named


class CommunityMetrics(BaseModel):
    """Graph-level community-structure statistics (detector output, domain model)."""

    modularity: float = Field(
        description="Modularity score measuring the quality of community division (higher values indicate better community structure)"
    )
    num_communities: int = Field(
        description="Total number of distinct communities identified in the graph"
    )
    average_community_size: float = Field(
        description="Mean number of nodes per community across all detected communities"
    )
    largest_community_size: int = Field(
        description="Number of nodes in the most populous community"
    )
    smallest_community_size: int = Field(
        description="Number of nodes in the least populous community"
    )
    community_size_distribution: dict[int, int] = Field(
        description="Histogram mapping community sizes to their frequency counts (size -> number of communities with that size)"
    )


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
    text_unit_ids: list[str] | None = Field(
        None, description="List of text unit IDs related to the community"
    )
    size: int | None = Field(
        None, description="The size of the community (Amount of text units)"
    )
    period: str | None = Field(None, description="The period of the community")


class HierarchicalCommunity(BaseModel):
    """One community of the Leiden hierarchy (detector output, domain model)."""

    community_id: str = Field(
        description="Unique string identifier for this community within the hierarchical structure"
    )
    level: int = Field(
        description="Depth level in the community hierarchy where 0 represents the base level with finest granularity"
    )
    nodes: set[str] = Field(
        description="Collection of graph node identifiers that are members of this community"
    )
    parent_id: str | None = Field(
        None,
        description="Identifier of the parent community at the next higher hierarchical level, if any",
    )
    children_ids: list[str] = Field(
        default_factory=list,
        description="Ordered list of child community identifiers at the next lower hierarchical level",
    )
    label: str = Field(
        default="",
        description="Human-readable positional label (e.g. 'L0_C3') for display; "
        "unlike community_id it is not stable across runs",
    )

    def __hash__(self) -> int:
        return hash(self.community_id)
