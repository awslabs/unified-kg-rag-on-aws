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

from unified_kg_rag.domain.models import (
    Community,
    CommunityReport,
    Entity,
    Relationship,
)
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.common import normalize_name

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


def _relationship_weight(rel: Relationship, supporting_text_units: list[str]) -> float:
    """Derive an edge weight from the count of distinct supporting text units.

    The full-build resolver sums per-instance weights across a (source, target,
    type) group, and each extracted instance carries ~1.0 — so the summed weight
    tracks the number of supporting occurrences. Deriving the weight from the
    *deduplicated* supporting-text-unit count makes the incremental merge
    converge to the same value AND idempotent: re-applying a delta unions the
    same ids, so the count (and weight) is unchanged. An edge carrying no
    text-unit lineage falls back to its own weight (default 1.0).
    """
    if supporting_text_units:
        return float(len(supporting_text_units))
    return rel.weight if rel.weight is not None else 1.0


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

    The merge is idempotent: weight is derived from the deduplicated
    supporting-text-unit union (not summed), and an edge whose remapped endpoints
    collapse onto the same entity is dropped as a self-loop — matching the
    full-build :class:`RelationshipResolver`, which removes self-referencing
    edges and would otherwise diverge from this path.
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
        # The full-build resolver drops self-referencing edges, so the merged
        # output must never contain one — including any that slipped into the
        # stored set.
        if rel.source_id == rel.target_id:
            continue
        existing = rel.model_copy(deep=True)
        existing.weight = _relationship_weight(existing, existing.text_unit_ids or [])
        by_key[_key(rel.source_id, rel.target_id, rel)] = existing

    for rel in delta:
        source_id, target_id = _endpoints(rel)
        if source_id == target_id:
            # Either an inherent self-loop or one the remap created by collapsing
            # both endpoints onto one entity; the full-build resolver drops these,
            # so the incremental path must too.
            continue
        key = _key(source_id, target_id, rel)
        match = by_key.get(key)
        if match is None:
            new_rel = rel.model_copy(deep=True)
            new_rel.source_id = source_id
            new_rel.target_id = target_id
            new_rel.weight = _relationship_weight(new_rel, new_rel.text_unit_ids or [])
            by_key[key] = new_rel
            continue

        match.description = _merge_descriptions(match.description, rel.description)
        match.text_unit_ids = _dedupe_preserve_order(
            (match.text_unit_ids or []) + (rel.text_unit_ids or [])
        )
        match.weight = _relationship_weight(match, match.text_unit_ids or [])

    merged = list(by_key.values())
    logger.info(
        "Merged relationships: %d old + %d delta -> %d",
        len(old),
        len(delta),
        len(merged),
    )
    return merged


def _community_content_key(community: Community) -> tuple:
    """Identity signature for a community (excludes the id, which we reassign).

    Includes every membership/content field so two communities that differ in
    any of them are treated as distinct: keying on a subset would make a delta
    community that genuinely differs only in (say) its relationship/text-unit
    membership collide with an existing one and be silently dropped as
    "already merged".
    """
    return (
        community.name,
        str(community.level),
        community.parent,
        tuple(sorted(community.entity_ids or [])),
        tuple(sorted(community.relationship_ids or [])),
        tuple(sorted(community.text_unit_ids or [])),
    )


def _report_content_key(report: CommunityReport) -> tuple:
    """Identity signature for a community report (excludes the id).

    Includes ``full_content`` and ``rank`` so a regenerated report that differs
    only in its body or importance is not collapsed onto the old one.
    """
    return (
        report.name,
        report.community_id,
        report.summary,
        report.full_content,
        report.rank,
    )


def _placed_id(
    base_id: str, content_key: tuple, existing: dict[str, tuple]
) -> str | None:
    """Resolve where a colliding delta item should go, or ``None`` to skip.

    Walks ``base_id``, ``base_id-delta``, ``base_id-delta-2``, … :
    - if a candidate id is free, return it (disambiguated, never dropped);
    - if a candidate id is taken by an item with the SAME content, return
      ``None`` — this delta was already merged (re-application is idempotent);
    - if taken by DIFFERENT content, advance to the next candidate.
    """
    if base_id not in existing:
        return base_id
    if existing[base_id] == content_key:
        return None
    candidate = f"{base_id}-delta"
    counter = 2
    while candidate in existing:
        if existing[candidate] == content_key:
            return None
        candidate = f"{base_id}-delta-{counter}"
        counter += 1
    return candidate


def merge_communities(old: list[Community], delta: list[Community]) -> list[Community]:
    """Append delta communities to old ones (MS-style id-offset append).

    Incremental runs do not re-cluster globally; a delta community whose id
    collides with an existing one is kept distinct by a unique
    ``-delta``/``-delta-N`` suffix (never silently dropped). Re-merging the same
    delta is idempotent: a collision whose content matches an already-merged
    community is skipped rather than re-appended (matched by content, not by a
    fragile id-suffix string).
    """
    existing: dict[str, tuple] = {c.id: _community_content_key(c) for c in old}
    merged = [community.model_copy(deep=True) for community in old]

    for community in delta:
        key = _community_content_key(community)
        placed = _placed_id(community.id, key, existing)
        if placed is None:
            continue  # already merged (idempotent re-application)
        new_community = community.model_copy(deep=True)
        new_community.id = placed
        existing[placed] = key
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
    existing: dict[str, tuple] = {r.id: _report_content_key(r) for r in old}
    merged = [report.model_copy(deep=True) for report in old]

    for report in delta:
        key = _report_content_key(report)
        placed = _placed_id(report.id, key, existing)
        if placed is None:
            continue
        new_report = report.model_copy(deep=True)
        new_report.id = placed
        existing[placed] = key
        merged.append(new_report)

    return merged
