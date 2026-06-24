# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure merge functions for incremental indexing.

See package docstring for the porting rationale. Each function takes the
existing (``old``) artifacts plus the freshly computed ``delta`` and returns the
merged set, preserving the old item's id where the natural key matches so graph
references stay stable.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from aws_graphrag.domain.models import Community, CommunityReport, Entity, Relationship
from aws_graphrag.shared import get_logger
from aws_graphrag.shared.utils.common import normalize_name

logger = get_logger(__name__)


class DeltaMergeResult(BaseModel):
    """Outcome of merging delta artifacts into the existing index."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    communities: list[Community] = Field(default_factory=list)
    community_reports: list[CommunityReport] = Field(default_factory=list)
    # Maps a delta entity id to the existing entity id it merged into, so callers
    # can remap relationship/text-unit references onto the surviving id.
    entity_id_remap: dict[str, str] = Field(default_factory=dict)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _merge_descriptions(old: str | None, new: str | None) -> str | None:
    """Combine two descriptions, dropping duplicates and empties."""
    parts = [p for p in (old, new) if p]
    if not parts:
        return None
    deduped = _dedupe_preserve_order(parts)
    return "\n".join(deduped)


def merge_entities(
    old: list[Entity], delta: list[Entity]
) -> tuple[list[Entity], dict[str, str]]:
    """Merge delta entities into old ones by normalized name.

    Returns the merged entity list and ``{delta_id: surviving_id}`` for entities
    that merged into an existing one (so relationships can be remapped).
    """
    by_key: dict[str, Entity] = {}
    id_remap: dict[str, str] = {}

    for entity in old:
        by_key[normalize_name(entity.name)] = entity.model_copy(deep=True)

    for entity in delta:
        key = normalize_name(entity.name)
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = entity.model_copy(deep=True)
            continue

        # Merge into the surviving (old) entity; keep its id.
        id_remap[entity.id] = existing.id
        existing.description = _merge_descriptions(
            existing.description, entity.description
        )
        existing.text_unit_ids = _dedupe_preserve_order(
            (existing.text_unit_ids or []) + (entity.text_unit_ids or [])
        )
        existing.community_ids = _dedupe_preserve_order(
            (existing.community_ids or []) + (entity.community_ids or [])
        )
        # Frequency tracks the number of supporting text units (MS GraphRAG keeps
        # this separate from rank/degree, which reflects graph importance).
        existing.frequency = len(existing.text_unit_ids)
        if entity.type and not existing.type:
            existing.type = entity.type

    merged = list(by_key.values())
    logger.info(
        "Merged entities: %d old + %d delta -> %d (%d merged by name)",
        len(old),
        len(delta),
        len(merged),
        len(id_remap),
    )
    return merged, id_remap


def merge_relationships(
    old: list[Relationship],
    delta: list[Relationship],
    entity_id_remap: dict[str, str] | None = None,
) -> list[Relationship]:
    """Merge delta relationships into old ones by (source, target, type).

    ``entity_id_remap`` (from :func:`merge_entities`) is applied to delta
    relationship endpoints first so edges point at surviving entity ids.

    The merge key includes the normalized relationship ``type`` so a delta edge
    of a *different* type between the same endpoints stays a distinct edge (the
    full-build resolver groups by (source, target, type) too — keying on
    endpoints alone here would silently collapse them and drop the delta type).
    """
    remap = entity_id_remap or {}

    def _endpoints(rel: Relationship) -> tuple[str, str]:
        return remap.get(rel.source_id, rel.source_id), remap.get(
            rel.target_id, rel.target_id
        )

    def _key(source_id: str, target_id: str, rel: Relationship) -> tuple[str, str, str]:
        return source_id, target_id, (rel.type or "").strip().lower()

    by_key: dict[tuple[str, str, str], Relationship] = {}

    for rel in old:
        by_key[_key(rel.source_id, rel.target_id, rel)] = rel.model_copy(deep=True)

    for rel in delta:
        source_id, target_id = _endpoints(rel)
        key = _key(source_id, target_id, rel)
        existing = by_key.get(key)
        if existing is None:
            new_rel = rel.model_copy(deep=True)
            new_rel.source_id = source_id
            new_rel.target_id = target_id
            by_key[key] = new_rel
            continue

        existing.description = _merge_descriptions(
            existing.description, rel.description
        )
        existing.text_unit_ids = _dedupe_preserve_order(
            (existing.text_unit_ids or []) + (rel.text_unit_ids or [])
        )
        # Sum supporting weights (MS GraphRAG semantics). Additive merge is
        # order-independent and associative, unlike a running pairwise mean
        # which depends on merge order and is non-deterministic across runs.
        old_weight = existing.weight if existing.weight is not None else 1.0
        new_weight = rel.weight if rel.weight is not None else 1.0
        existing.weight = old_weight + new_weight

    merged = list(by_key.values())
    logger.info(
        "Merged relationships: %d old + %d delta -> %d",
        len(old),
        len(delta),
        len(merged),
    )
    return merged


def merge_communities(old: list[Community], delta: list[Community]) -> list[Community]:
    """Append delta communities to old ones (MS-style id-offset append).

    Incremental runs do not re-cluster globally; delta communities whose ids
    collide with existing ones are kept distinct by suffixing the delta id.
    Re-merging the same delta is idempotent: a delta whose (already-suffixed) id
    is present is skipped rather than re-suffixed into ``id-delta-delta``.
    """
    existing_ids = {community.id for community in old}
    merged = [community.model_copy(deep=True) for community in old]

    for community in delta:
        if community.id in existing_ids and community.id.endswith("-delta"):
            # Already-merged delta item re-presented; skip (idempotent).
            continue
        new_community = community.model_copy(deep=True)
        if new_community.id in existing_ids:
            new_community.id = f"{new_community.id}-delta"
            if new_community.id in existing_ids:
                continue
        existing_ids.add(new_community.id)
        merged.append(new_community)

    logger.info(
        "Merged communities: %d old + %d delta -> %d",
        len(old),
        len(delta),
        len(merged),
    )
    return merged


def merge_community_reports(
    old: list[CommunityReport], delta: list[CommunityReport]
) -> list[CommunityReport]:
    """Append delta community reports (mirrors :func:`merge_communities`)."""
    existing_ids = {report.id for report in old}
    merged = [report.model_copy(deep=True) for report in old]

    for report in delta:
        if report.id in existing_ids and report.id.endswith("-delta"):
            continue
        new_report = report.model_copy(deep=True)
        if new_report.id in existing_ids:
            new_report.id = f"{new_report.id}-delta"
            if new_report.id in existing_ids:
                continue
        existing_ids.add(new_report.id)
        merged.append(new_report)

    return merged
