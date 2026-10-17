# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure merge functions for incremental indexing.

See package docstring for the porting rationale. Each function takes the
existing (``old``) artifacts plus the freshly computed ``delta`` and returns the
merged set, preserving the old item's id where the id or natural key matches so
graph references stay stable.
"""

from __future__ import annotations

from collections.abc import Collection

from unified_kg_rag.domain.ingestion.base_processor import merge_attributes
from unified_kg_rag.domain.ingestion.base_resolver import (
    FuzzyMatcher,
    entity_types_compatible,
)
from unified_kg_rag.domain.ingestion.descriptions import merge_descriptions
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
    overlay_weights,
    text_unit_weights,
)
from unified_kg_rag.domain.models import Claim, Constants, Entity, Relationship
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.common import entity_key

logger = get_logger(__name__)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _merge_entity_fields(surviving: Entity, incoming: Entity) -> None:
    """Fold ``incoming``'s evidence into ``surviving`` in place (id preserved).

    Shared by the exact-name and fuzzy-match paths so both converge to the same
    field semantics as the full-build resolver (``EntityResolver._merge_entities``):
    description union, text-unit/community-id union, max confidence/rank, first-known type, attribute union.

    The name stays the surviving one (``incoming`` may spell the same key
    otherwise, or be a name variant the matcher joined): edges carry a copy
    of their endpoints' names, and the stored edges the delta does not
    rewrite would keep naming the old one. A full rebuild takes the name
    cited by the most text units instead.

    The type stays the surviving one too (``incoming``'s only fills a missing
    type), while a full build keeps the most frequent type of a name's
    mentions, so an incremental run can keep a type a full rebuild replaces,
    like a name.
    """
    surviving.description = merge_descriptions(
        (surviving.description, incoming.description)
    )
    surviving.attributes = merge_attributes((surviving.attributes, incoming.attributes))
    surviving.text_unit_ids = _dedupe_preserve_order(
        (surviving.text_unit_ids or []) + (incoming.text_unit_ids or [])
    )
    surviving.community_ids = _dedupe_preserve_order(
        (surviving.community_ids or []) + (incoming.community_ids or [])
    )
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

    The identity key is ``entity_key(name)``, which extraction derives the id
    from; matching the key as well covers items whose id was assigned
    otherwise (the direct API).

    Returns the merged entity list and ``{delta_id: surviving_id}`` for entities
    that merged into an existing one under a different id (so relationships can
    be remapped). A delta entity that matched its own stored id is merged but not
    listed: there is nothing to remap, and callers treat a non-empty remap as a
    reason to read back and rewrite the surviving entities' edges.

    When ``fuzzy_matcher`` is supplied (built over the *old* entity names with
    the full build's settings), a delta entity whose normalized name does not
    exactly match an old one is additionally matched against the old names
    with ``find_all_matches`` — so the surface forms the full-build
    ``EntityResolver`` groups ("Acme Inc" vs "Acme", and similar names when
    fuzzy resolution is on) converge across runs too, instead of accumulating
    as separate entities. Without a matcher the merge is exact-name-only.
    Matches only ever collapse a delta onto an *existing old* entity (never
    delta-onto-delta), which keeps the result order-independent and
    idempotent.
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
        # By id (the same statement), else by key: an edge re-pointed at a
        # merged entity keeps the id its extracted names derive.
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
        match.attributes = merge_attributes((match.attributes, rel.attributes))
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
    ``text_unit_ids``. Weight follows the same rule as the merge (the sum of
    the remaining per-text-unit strengths). Descriptions are left as they are:
    removing a document's contribution would need an LLM re-summary. So are
    the attributes: the document ``filters`` values the merge unioned in
    (``merge_attributes``) stay on the shared item until a full
    rebuild.
    """
    removed = set(text_unit_ids)
    updated_entities: list[Entity] = []
    for entity in entities:
        if removed.isdisjoint(entity.text_unit_ids or []):
            continue
        kept = [t for t in entity.text_unit_ids or [] if t not in removed]
        updated_entities.append(
            entity.model_copy(deep=True, update={"text_unit_ids": kept})
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


def merge_claims(old: list[Claim], delta: list[Claim]) -> list[Claim]:
    """Fold the stored evidence of each delta claim into it, matched by id.

    A claim's id derives from its subject, object, type and dates, not from
    its document, so the same claim extracted from another document has the
    stored claim's id. As in the full-build ``ClaimResolver`` merge, the
    descriptions and source texts are unioned (stored first) and the text
    units are the sorted union. The status is the stored one when it has
    one. ``ClaimResolver`` takes the most common status of a group (the first
    on a tie), which for one stored and one delta claim is the same rule; the
    stored claim keeps no per-status counts, though, so over three or more
    documents the two can differ (stored ``SUSPECTED`` stays when two later
    documents say ``TRUE``). The attributes are unioned with the document
    ``filters`` of both (see ``merge_attributes``). Every other field is the
    delta's. Returns the delta claims, merged.
    """
    stored = {claim.id: claim for claim in old}
    merged: list[Claim] = []
    for claim in delta:
        match = stored.get(claim.id)
        if match is None:
            merged.append(claim)
            continue
        merged.append(
            claim.model_copy(
                update={
                    "description": merge_descriptions(
                        (match.description, claim.description)
                    ),
                    "source_text": merge_descriptions(
                        (match.source_text, claim.source_text)
                    ),
                    "text_unit_ids": sorted(
                        {*(match.text_unit_ids or []), *(claim.text_unit_ids or [])}
                    ),
                    "status": match.status or claim.status,
                    "attributes": merge_attributes(
                        (match.attributes, claim.attributes)
                    ),
                }
            )
        )
    return merged


def remove_claim_text_units(
    claims: list[Claim], text_unit_ids: Collection[str]
) -> list[Claim]:
    """Strip removed text units from kept claims (see :func:`remove_text_units`).

    Returns updated copies of only the claims that cited one of
    ``text_unit_ids``; descriptions are left as they are.
    """
    removed = set(text_unit_ids)
    return [
        claim.model_copy(
            update={
                "text_unit_ids": [
                    t for t in claim.text_unit_ids or [] if t not in removed
                ]
            }
        )
        for claim in claims
        if not removed.isdisjoint(claim.text_unit_ids or [])
    ]
