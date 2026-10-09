# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure merge functions for incremental indexing.

See package docstring for the porting rationale. Each function takes the
existing (``old``) artifacts plus the freshly computed ``delta`` and returns the
merged set, preserving the old item's id where the id or natural key matches so
graph references stay stable.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
    overlay_weights,
    text_unit_weights,
)
from unified_kg_rag.domain.models import (
    Community,
    CommunityReport,
    Constants,
    Entity,
    Relationship,
)
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.common import entity_key

if TYPE_CHECKING:
    from unified_kg_rag.domain.ingestion.base_resolver import FuzzyMatcher

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


def merge_descriptions(descriptions: Iterable[str | None]) -> str | None:
    """Combine descriptions, dropping duplicate and blank lines (order kept).

    The one description-merge rule of every build path (extraction,
    full-build resolution, cross-run merge), so a full build and an
    incremental build of the same corpus converge on the same text.

    A stored description is the newline join of earlier merges, so the dedupe
    compares lines: comparing whole strings would append a re-applied delta's
    description again on every run.
    """
    lines = [
        line
        for part in descriptions
        if part
        for line in part.split("\n")
        if line.strip()
    ]
    if not lines:
        return None
    return "\n".join(_dedupe_preserve_order(lines))


def _merge_attributes(
    old: dict[str, Any] | None, new: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Union two attribute maps, the newer value winning on a shared key.

    Same rule as the full-build resolver's ``_merge_attributes``. Attributes
    carry the index suffix the item is written under, so dropping them would
    move a merged item into the default suffix.
    """
    if not old and not new:
        return None
    return {**(old or {}), **(new or {})}


def _merge_entity_fields(surviving: Entity, incoming: Entity) -> None:
    """Fold ``incoming``'s evidence into ``surviving`` in place (id preserved).

    Shared by the exact-name and fuzzy-match paths so both converge to the same
    field semantics as the full-build resolver (``EntityResolver._merge_entities``):
    description union, text-unit/community-id union, frequency = #text-units,
    max confidence/rank, first-known type, attribute union.
    """
    surviving.description = merge_descriptions(
        (surviving.description, incoming.description)
    )
    surviving.attributes = _merge_attributes(surviving.attributes, incoming.attributes)
    surviving.text_unit_ids = _dedupe_preserve_order(
        (surviving.text_unit_ids or []) + (incoming.text_unit_ids or [])
    )
    surviving.community_ids = _dedupe_preserve_order(
        (surviving.community_ids or []) + (incoming.community_ids or [])
    )
    # Frequency tracks the number of supporting text units (MS GraphRAG keeps
    # this separate from rank/degree, which reflects graph importance).
    surviving.frequency = len(surviving.text_unit_ids)
    # Confidence and rank are monotonic in evidence: take the max so a more
    # confident / higher-ranked delta reinforces the surviving entity. This
    # matches the full-build resolver (max over the group), keeping a full
    # rebuild and an incremental build convergent on these fields too — they
    # feed report ranking and the confidence threshold filter.
    surviving.confidence = max(
        surviving.confidence if surviving.confidence is not None else 0.0,
        incoming.confidence if incoming.confidence is not None else 0.0,
    )
    surviving.rank = max(
        surviving.rank if surviving.rank is not None else 1,
        incoming.rank if incoming.rank is not None else 1,
    )
    if incoming.type and not surviving.type:
        surviving.type = incoming.type


def merge_entities(
    old: list[Entity],
    delta: list[Entity],
    fuzzy_matcher: FuzzyMatcher | None = None,
) -> tuple[list[Entity], dict[str, str]]:
    """Merge delta entities into old ones by id, else by identity key.

    The identity key is ``entity_key(name)``. Matching the id first matters
    when a gleaning correction renamed a stored entity and kept its id.

    Returns the merged entity list and ``{delta_id: surviving_id}`` for entities
    that merged into an existing one under a different id (so relationships can
    be remapped). A delta entity that matched its own stored id is merged but not
    listed: there is nothing to remap, and callers treat a non-empty remap as a
    reason to read back and rewrite the surviving entities' edges.

    When ``fuzzy_matcher`` is supplied (built over the *old* entity names), a
    delta entity whose normalized name does not exactly match an old one is
    additionally matched fuzzily against the old names — so near-duplicate
    surface forms ("Acme Corp" vs "Acme Corporation") converge across runs the
    way the full-build ``EntityResolver`` groups them, instead of accumulating
    as separate entities. Without a matcher the merge is exact-name-only (its
    long-standing behaviour). Fuzzy matches only ever collapse a delta onto an
    *existing old* entity (never delta-onto-delta), which keeps the result
    order-independent and idempotent.
    """
    by_id: dict[str, Entity] = {}
    # Identity key -> id of the first entity with that key.
    id_by_key: dict[str, str] = {}
    # Old display name -> its id, so a fuzzy hit on an old name can find the
    # surviving entity to merge into.
    old_id_by_name: dict[str, str] = {}
    id_remap: dict[str, str] = {}
    merged_count = 0

    for entity in old:
        by_id[entity.id] = entity.model_copy(deep=True)
        id_by_key.setdefault(entity_key(entity.name), entity.id)
        old_id_by_name[entity.name] = entity.id

    for entity in delta:
        key = entity_key(entity.name)
        # Id first: a gleaning correction renames an entity in place and keeps
        # the id derived from its old name, so a later delta naming the old form
        # carries the stored id under a different key.
        existing = by_id.get(entity.id) or by_id.get(id_by_key.get(key, ""))
        if existing is None and fuzzy_matcher is not None:
            existing = _find_fuzzy_old_match(
                entity, fuzzy_matcher, old_id_by_name, by_id
            )
        if existing is None:
            by_id[entity.id] = entity.model_copy(deep=True)
            id_by_key.setdefault(key, entity.id)
            continue

        # Merge into the surviving (old) entity; keep its id.
        merged_count += 1
        if entity.id != existing.id:
            id_remap[entity.id] = existing.id
        _merge_entity_fields(existing, entity)

    merged = list(by_id.values())
    logger.info(
        "Merged entities: %d old + %d delta -> %d (%d merged)",
        len(old),
        len(delta),
        len(merged),
        merged_count,
    )
    return merged, id_remap


def _find_fuzzy_old_match(
    entity: Entity,
    fuzzy_matcher: FuzzyMatcher,
    old_id_by_name: dict[str, str],
    by_id: dict[str, Entity],
) -> Entity | None:
    """Return the best old entity fuzzy-matching ``entity``'s name, or None.

    Considers only *old* candidate names (delta entities are never matcher
    candidates), so the collapse target is stable regardless of delta ordering.
    Ties break on the higher score, then the lexicographically smaller name, so
    the choice is deterministic. Old entities with an incompatible type are
    skipped, matching the full-build resolver's grouping guard (identifier
    conflicts are already filtered by ``find_all_matches``).
    """
    from unified_kg_rag.domain.ingestion.base_resolver import (
        entity_types_compatible,
    )

    matches = [
        (name, score)
        for name, score in fuzzy_matcher.find_all_matches(entity.name)
        if name in old_id_by_name
        and entity_types_compatible(entity.type, by_id[old_id_by_name[name]].type)
    ]
    if not matches:
        return None
    best_name, _ = max(matches, key=lambda m: (m[1], -len(m[0]), m[0]))
    return by_id.get(old_id_by_name[best_name])


def _remapped_endpoints(
    rel: Relationship, entity_id_remap: dict[str, str]
) -> tuple[str, str]:
    return entity_id_remap.get(rel.source_id, rel.source_id), entity_id_remap.get(
        rel.target_id, rel.target_id
    )


def _relationship_key(
    source_id: str, target_id: str, rel: Relationship
) -> tuple[str, str, str]:
    # An untyped relationship is stored (and read back) as the default type,
    # so both forms must share a key.
    rel_type = rel.type or Constants.DEFAULT_RELATIONSHIP_TYPE.value
    return source_id, target_id, rel_type.strip().lower()


def merge_relationships(
    old: list[Relationship],
    delta: list[Relationship],
    entity_id_remap: dict[str, str] | None = None,
) -> list[Relationship]:
    """Merge delta relationships into old ones by id, else (source, target, type).

    ``entity_id_remap`` (from :func:`merge_entities`) is applied to delta
    relationship endpoints first so edges point at surviving entity ids.

    The merge key includes the normalized relationship ``type`` so a delta edge
    of a *different* type between the same endpoints stays a distinct edge (the
    full-build resolver groups by (source, target, type) too — keying on
    endpoints alone here would silently collapse them and drop the delta type).

    Weight is the sum of the per-text-unit strengths (see
    ``relationship_weights``); a delta entry for a text unit replaces the stored
    one, so re-applying a delta is idempotent and the result equals a full build
    over the union of the text units. Stored edges the delta does not touch are
    returned unchanged. An edge whose remapped endpoints collapse onto the same
    entity is dropped as a self-loop, matching the full-build
    :class:`RelationshipResolver`.
    """
    remap = entity_id_remap or {}
    by_id: dict[str, Relationship] = {}
    # (source, target, type) -> id of the first edge with that key.
    id_by_key: dict[tuple[str, str, str], str] = {}

    for rel in old:
        # The full-build resolver drops self-referencing edges, so the merged
        # output must never contain one — including any that slipped into the
        # stored set.
        if rel.source_id == rel.target_id:
            continue
        by_id[rel.id] = rel.model_copy(deep=True)
        id_by_key.setdefault(
            _relationship_key(rel.source_id, rel.target_id, rel), rel.id
        )

    for rel in delta:
        source_id, target_id = _remapped_endpoints(rel, remap)
        if source_id == target_id:
            # Either an inherent self-loop or one the remap created by collapsing
            # both endpoints onto one entity; the full-build resolver drops these,
            # so the incremental path must too.
            continue
        key = _relationship_key(source_id, target_id, rel)
        # Id first: a gleaning correction changes an edge's type or direction in
        # place and keeps its id, so a later delta of the uncorrected edge
        # carries the stored id under a different key.
        match = by_id.get(rel.id) or by_id.get(id_by_key.get(key, ""))
        if match is None:
            new_rel = rel.model_copy(deep=True)
            new_rel.source_id = source_id
            new_rel.target_id = target_id
            apply_text_unit_weights(new_rel, text_unit_weights(new_rel))
            by_id[rel.id] = new_rel
            id_by_key.setdefault(key, rel.id)
            continue

        weights = overlay_weights(text_unit_weights(match), text_unit_weights(rel))
        match.description = merge_descriptions((match.description, rel.description))
        match.attributes = _merge_attributes(match.attributes, rel.attributes)
        match.text_unit_ids = _dedupe_preserve_order(
            (match.text_unit_ids or []) + (rel.text_unit_ids or [])
        )
        apply_text_unit_weights(match, weights)

    merged = list(by_id.values())
    logger.info(
        "Merged relationships: %d old + %d delta -> %d",
        len(old),
        len(delta),
        len(merged),
    )
    return merged


def relationship_id_remap(
    delta: list[Relationship],
    merged: list[Relationship],
    entity_id_remap: dict[str, str] | None = None,
) -> dict[str, str]:
    """Map each delta relationship id to the id :func:`merge_relationships` kept.

    A delta edge that merged into an existing edge (same remapped source,
    target and type) survives under the existing edge's id; callers recording
    lineage need that id, not the delta's. Self-loops the merge dropped have
    no entry.
    """
    remap = entity_id_remap or {}
    kept_ids = {r.id for r in merged}
    kept: dict[tuple[str, str, str], str] = {}
    for r in merged:
        kept.setdefault(_relationship_key(r.source_id, r.target_id, r), r.id)
    result: dict[str, str] = {}
    for rel in delta:
        if rel.id in kept_ids:
            # Matched by id (or kept as new): it survives under its own id.
            continue
        source_id, target_id = _remapped_endpoints(rel, remap)
        kept_id = kept.get(_relationship_key(source_id, target_id, rel))
        if kept_id is not None and kept_id != rel.id:
            result[rel.id] = kept_id
    return result


def remove_text_units(
    entities: list[Entity],
    relationships: list[Relationship],
    text_unit_ids: Collection[str],
) -> tuple[list[Entity], list[Relationship]]:
    """Strip removed text units from kept artifacts, recomputing derived counts.

    Used when a changed or deleted document's artifacts are shared with
    surviving documents: they stay, but must stop citing the removed chunks.
    Returns updated copies of only the items that referenced one of
    ``text_unit_ids``. Frequency and weight follow the same rules as the merge
    (frequency = number of supporting text units, weight = sum of the remaining
    per-text-unit strengths). Descriptions are left as they are:
    removing a document's contribution would need an LLM re-summary.
    """
    removed = set(text_unit_ids)
    updated_entities: list[Entity] = []
    for entity in entities:
        if removed.isdisjoint(entity.text_unit_ids or []):
            continue
        kept = [t for t in entity.text_unit_ids or [] if t not in removed]
        updated_entities.append(
            entity.model_copy(
                deep=True, update={"text_unit_ids": kept, "frequency": len(kept)}
            )
        )
    updated_relationships: list[Relationship] = []
    for rel in relationships:
        if removed.isdisjoint(rel.text_unit_ids or []):
            continue
        weights = {
            tu: w for tu, w in text_unit_weights(rel).items() if tu not in removed
        }
        kept = [t for t in rel.text_unit_ids or [] if t not in removed]
        updated = rel.model_copy(deep=True, update={"text_unit_ids": kept})
        apply_text_unit_weights(updated, weights)
        updated_relationships.append(updated)
    return updated_entities, updated_relationships


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
