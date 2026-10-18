# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""How far incremental runs have moved the communities from a full build's.

An incremental run clusters only its delta and appends the communities next
to the stored ones (re-clustering the whole graph would cost a full build).
A delta entity that already existed then sits in two level-0 communities, and
its earlier communities' reports do not reflect what the delta added. Each
entity records the ids of the level-0 communities it belongs to
(``community_ids``); a full build gives a clustered entity one, and the
cross-run merge unions the stored ids with the delta's, so the count of an
entity's level-0 communities is the number of clusterings that placed it.
"""

from collections.abc import Iterable

from pydantic import BaseModel, Field

from unified_kg_rag.domain.models import Community, Entity


class CommunityDrift(BaseModel):
    """Community drift an incremental run measured on the entities it wrote."""

    delta_communities: int = Field(
        description="Communities (all levels) this run appended"
    )
    clustered_entities: int = Field(
        description="Entities in this run's level-0 communities"
    )
    overlapping_entities: int = Field(
        description="Of those, entities that also belong to an earlier "
        "level-0 community"
    )
    stale_communities: int = Field(
        description="Earlier level-0 communities of the overlapping entities: "
        "their reports (and their parents') miss what this run added"
    )
    added_roots: int = Field(
        description="Root communities this run appended. A root is in every "
        "cut of the hierarchy, so global search maps this many more reports "
        "until a rebuild, however few entities they hold"
    )
    overlap_share: float = Field(
        description="overlapping_entities / clustered_entities"
    )
    rebuild_recommended: bool = Field(
        description="overlap_share reached indexing.community_drift_threshold"
    )


def _level0(communities: Iterable[Community]) -> list[Community]:
    return [c for c in communities if str(c.level) == "0"]


def with_level0_community_ids(
    entities: list[Entity], communities: list[Community]
) -> list[Entity]:
    """``entities`` with the ids of their level-0 communities added.

    Copies only the entities that gain an id; the others are returned as is.
    """
    added: dict[str, list[str]] = {}
    for community in _level0(communities):
        for entity_id in community.entity_ids or []:
            added.setdefault(entity_id, []).append(community.id)
    result: list[Entity] = []
    for entity in entities:
        ids = added.get(entity.id)
        current = entity.community_ids or []
        new = [i for i in ids or [] if i not in current]
        result.append(
            entity.model_copy(update={"community_ids": [*current, *new]})
            if new
            else entity
        )
    return result


def measure_community_drift(
    entities: list[Entity], communities: list[Community], threshold: float
) -> CommunityDrift | None:
    """Drift of a delta run, from its merged entities and its communities.

    ``entities`` are the run's entities after the cross-run merge (their
    ``community_ids`` hold the stored ids too). ``None`` when the run
    appended no communities.
    """
    if not communities:
        return None
    level0 = _level0(communities)
    delta_ids = {c.id for c in level0}
    members = {eid for c in level0 for eid in c.entity_ids or []}
    by_id = {e.id: e for e in entities}
    stale: set[str] = set()
    overlapping = 0
    for entity_id in members:
        entity = by_id.get(entity_id)
        earlier = set(entity.community_ids or []) - delta_ids if entity else set()
        if earlier:
            overlapping += 1
            stale |= earlier
    share = overlapping / len(members) if members else 0.0
    ids = {c.id for c in communities}
    return CommunityDrift(
        delta_communities=len(communities),
        clustered_entities=len(members),
        overlapping_entities=overlapping,
        stale_communities=len(stale),
        added_roots=sum(1 for c in communities if c.parent not in ids),
        overlap_share=round(share, 4),
        rebuild_recommended=bool(members) and share >= threshold,
    )
