# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stored items' content derived from their text units' current records.

An incremental run merges its delta into the stored graph and strips removed
text units from shared items (``merge/merger.py``); that keeps each item's
structure (id, text units, per-unit weights) equal to a full build's, but
folding fields into stored ones keeps text a removed or edited document
contributed. Here the content of every item a run touches is recomputed from
the records of the text units it cites now, by the full build's own code: the
records are merged (``merge_contributions``) and resolved (``GraphResolver``,
``ClaimResolver``) as a full build does, and each stored item takes the
name, type, description, confidence, attributes (and, for a claim, status
and source text) of the resolved item it corresponds to.

An item cites every text unit any of its member names is stated in, so the
records of its own text units hold all of its contributions: the result is
what a full build over the current corpus gives it, id included. An entity
whose derived name has another id than its stored one (a removed document
held the spelling that won) takes the derived id, and so do its edges and
claims; ``Derived.id_remap`` maps each stored id that changed to its new id,
so the caller re-points what else references it and deletes the old one.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from unified_kg_rag.domain.ingestion.claim_resolver import (
    ClaimResolver,
    resolved_claim_id,
)
from unified_kg_rag.domain.ingestion.contributions import (
    merge_claim_contributions,
    merge_contributions,
)
from unified_kg_rag.domain.ingestion.graph_resolver import (
    ENTITY_KEYS_ATTRIBUTE,
    GraphResolver,
    entity_keys,
)
from unified_kg_rag.domain.ingestion.relationship_weights import (
    TEXT_UNIT_WEIGHTS_ATTRIBUTE,
    apply_text_unit_weights,
    overlay_weights,
    text_unit_weights,
)
from unified_kg_rag.domain.models import (
    Claim,
    Config,
    Constants,
    Entity,
    Relationship,
    TextUnitExtraction,
)
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)


class Derived(BaseModel):
    """The touched items with their derived content."""

    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    # Ids of the items a text unit's record was missing for: kept as they
    # were (they cannot be derived).
    underived: list[str] = Field(default_factory=list)
    # Stored id -> derived id of every item whose id changed.
    id_remap: dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _type_key(value: str | None) -> str:
    return (value or Constants.DEFAULT_RELATIONSHIP_TYPE.value).strip().lower()


def _union(lists: Iterable[list[str] | None]) -> list[str]:
    return list(dict.fromkeys(i for ids in lists for i in ids or []))


def _with_entity_ids(claim: Claim, ids: Mapping[str, str]) -> Claim:
    """``claim`` with its subject and object mapped through ``ids``."""
    return claim.model_copy(
        update={
            "subject_id": ids.get(claim.subject_id, claim.subject_id),
            "object_id": (
                ids.get(claim.object_id, claim.object_id) if claim.object_id else None
            ),
        }
    )


def _merge_same_claims(claims: list[Claim]) -> list[Claim]:
    """One claim per id: given claims that derive to one fact (a stored claim
    and the delta's, stated about two spellings of its subject) are one claim
    citing the text units of both, as in a full build."""
    merged: dict[str, Claim] = {}
    for claim in claims:
        kept = merged.get(claim.id)
        if kept is None:
            merged[claim.id] = claim
            continue
        kept.text_unit_ids = list(
            dict.fromkeys([*(kept.text_unit_ids or []), *(claim.text_unit_ids or [])])
        )
    return list(merged.values())


def derive(
    config: Config,
    entities: Sequence[Entity],
    relationships: Sequence[Relationship],
    claims: Sequence[Claim],
    records: Mapping[str, TextUnitExtraction],
    claim_records: Mapping[str, list[Claim]] | None = None,
    rekeyed: Mapping[str, str] | None = None,
    blocked: Collection[str] = (),
) -> Derived:
    """Derive the content of the given stored items from ``records``.

    ``rekeyed`` maps stored entity ids an earlier derivation of this run
    replaced to their new ids (``Derived.id_remap``): an item still pointing
    at an old id follows it, and a claim keyed by an old id is matched.
    ``blocked`` ids are held by stored items outside the given ones: no item
    is re-keyed to one (it keeps its id rather than overwrite another).

    ``records`` (and ``claim_records``, the per-unit claims) are by text
    unit id and must hold every text unit the items cite; an item one of
    whose text units has none keeps its content and is listed in
    ``underived`` (its id changes only to follow a re-keyed endpoint).
    Given claims that derive to one fact are returned as one. Returns copies.
    """
    unit_ids = {
        unit_id
        for items in (entities, relationships, claims)
        for item in items
        for unit_id in item.text_unit_ids or []
    }
    underived: list[str] = []

    def complete(item: Entity | Relationship | Claim) -> bool:
        ok = all(unit_id in records for unit_id in item.text_unit_ids or [])
        if not ok:
            underived.append(item.id)
        return ok

    contributions, edges = merge_contributions(
        records[unit_id] for unit_id in unit_ids if unit_id in records
    )
    resolver = GraphResolver(config, max_workers=1, use_process_pool=False)
    resolved, _ = resolver.resolve_graph(contributions, edges)

    rekeyed = rekeyed or {}
    old_of = {new: old for old, new in rekeyed.items()}
    id_remap: dict[str, str] = {}

    # Each stored entity is the resolved entity sharing most of its keys.
    # The stored entities of one resolved entity (two spellings indexed
    # apart, or the old and new copies an interrupted re-key left) become
    # one, under the resolved entity's id unless another given entity holds
    # it or it is ``blocked``.
    taken = {entity.id for entity in entities} | set(blocked)
    groups: dict[str, tuple[Entity, list[Entity]]] = {}
    order: list[Entity | str] = []
    for entity in entities:
        keys = entity_keys(entity)
        candidates = [r for r in resolved["entities"] if keys & entity_keys(r)]
        if not complete(entity) or not candidates:
            order.append(entity.model_copy(deep=True))
            continue
        match = max(
            candidates,
            key=lambda r: (
                len(set(r.text_unit_ids or []) & set(entity.text_unit_ids or [])),
                r.id == entity.id,
            ),
        )
        if match.id not in groups:
            groups[match.id] = (match, [])
            order.append(match.id)
        groups[match.id][1].append(entity)
    # Resolved id -> the derived id, and every id a given claim may name
    # the entity by (its stored copies' ids, its derived id, the id an
    # earlier re-key of this run replaced).
    derived_of: dict[str, str] = {}
    aliases: dict[str, set[str]] = {}
    derived_entities: list[Entity] = []
    for item in order:
        if isinstance(item, Entity):
            derived_entities.append(item)
            continue
        match, group = groups[item]
        ids = [e.id for e in group]
        new_id = match.id if match.id in ids or match.id not in taken else ids[0]
        id_remap.update({i: new_id for i in ids if i != new_id})
        derived_of[match.id] = new_id
        aliases[match.id] = {*ids, new_id, old_of.get(new_id, new_id)}
        derived_entities.append(
            group[0].model_copy(
                deep=True,
                update={
                    "id": new_id,
                    "name": match.name,
                    "type": match.type,
                    "description": match.description,
                    "confidence": match.confidence,
                    "text_unit_ids": _union(e.text_unit_ids for e in group),
                    "community_ids": _union(e.community_ids for e in group),
                    "attributes": {
                        **(match.attributes or {}),
                        ENTITY_KEYS_ATTRIBUTE: sorted(entity_keys(match)),
                    },
                },
            )
        )
    names = {e.id: e.name for e in derived_entities}

    def entity_id(stored_id: str) -> str:
        return id_remap.get(stored_id) or rekeyed.get(stored_id, stored_id)

    resolved_edges: dict[tuple[str, str, str], Relationship] = {}
    for rel in resolved["relationships"]:
        source = derived_of.get(rel.source_id, rel.source_id)
        target = derived_of.get(rel.target_id, rel.target_id)
        resolved_edges.setdefault((source, target, _type_key(rel.type)), rel)
    # Stored edges of one resolved edge become one, like the entities.
    edge_taken = {rel.id for rel in relationships} | set(blocked)
    edge_groups: dict[tuple[str, str, str], list[Relationship]] = {}
    edge_order: list[Relationship | tuple[str, str, str]] = []
    for rel in relationships:
        source, target = entity_id(rel.source_id), entity_id(rel.target_id)
        key = (source, target, _type_key(rel.type))
        if not complete(rel) or key not in resolved_edges:
            # Kept as stored, re-pointed at a re-keyed endpoint.
            moved: dict[str, str | None] = {}
            if rel.source_id in id_remap:
                moved.update(source_id=source, source_name=names.get(source))
            if rel.target_id in id_remap:
                moved.update(target_id=target, target_name=names.get(target))
            edge_order.append(rel.model_copy(deep=True, update=moved))
            continue
        if key not in edge_groups:
            edge_groups[key] = []
            edge_order.append(key)
        edge_groups[key].append(rel)
    derived_relationships: list[Relationship] = []
    for edge in edge_order:
        if isinstance(edge, Relationship):
            derived_relationships.append(edge)
            continue
        match_rel, same = resolved_edges[edge], edge_groups[edge]
        ids = [r.id for r in same]
        new_rel_id = (
            match_rel.id
            if match_rel.id in ids or match_rel.id not in edge_taken
            else ids[0]
        )
        id_remap.update({i: new_rel_id for i in ids if i != new_rel_id})
        source, target, _ = edge
        merged = same[0].model_copy(
            deep=True,
            update={
                "id": new_rel_id,
                "source_id": source,
                "target_id": target,
                "source_name": names.get(source, same[0].source_name),
                "target_name": names.get(target, same[0].target_name),
                "type": match_rel.type,
                "description": match_rel.description,
                # The stored per-unit strengths stay: they follow the
                # stored order of the text units.
                "attributes": {
                    **(match_rel.attributes or {}),
                    TEXT_UNIT_WEIGHTS_ATTRIBUTE: (same[0].attributes or {}).get(
                        TEXT_UNIT_WEIGHTS_ATTRIBUTE
                    ),
                },
            },
        )
        if len(same) > 1:
            # The strengths a full build gives the edge in these text units.
            units = set(_union(r.text_unit_ids for r in same))
            stored: dict[str, float] = {}
            for rel in same:
                stored = overlay_weights(stored, text_unit_weights(rel))
            resolved_weights = text_unit_weights(match_rel)
            apply_text_unit_weights(
                merged,
                {
                    unit: resolved_weights.get(unit, weight)
                    for unit, weight in stored.items()
                    if unit in units
                },
            )
        derived_relationships.append(merged)

    derived_claims: list[Claim] = []
    if claims:
        claim_records = claim_records or {}
        statements = merge_claim_contributions(
            [
                claim.model_copy(deep=True)
                for unit_id in sorted(unit_ids)
                for claim in claim_records.get(unit_id, [])
            ]
        )
        resolved_claims, _ = ClaimResolver(
            config, max_workers=1, use_process_pool=False
        ).resolve(statements, resolved["entities"])
        # Matched to the given claims by the id they have under the stored
        # entity ids or under the derived ones (a claim the caller already
        # re-pointed); each takes the id it has under the derived ones.
        by_id: dict[str, Claim] = {}
        for claim in resolved_claims:
            for subject in sorted(aliases.get(claim.subject_id, {claim.subject_id})):
                objects = (
                    aliases.get(claim.object_id, {claim.object_id})
                    if claim.object_id
                    else {None}
                )
                for obj in sorted(objects, key=str):
                    stated = claim.model_copy(
                        update={"subject_id": subject, "object_id": obj}
                    )
                    by_id.setdefault(resolved_claim_id(stated), claim)
        for claim in claims:
            match_claim = by_id.get(claim.id)
            endpoints = {
                "subject_id": entity_id(claim.subject_id),
                "object_id": entity_id(claim.object_id) if claim.object_id else None,
            }
            if not all(u in claim_records for u in claim.text_unit_ids or []):
                underived.append(claim.id)
                match_claim = None
            if match_claim is None:
                derived_claims.append(claim.model_copy(deep=True, update=endpoints))
                continue
            match_claim = _with_entity_ids(match_claim, derived_of)
            new_claim_id = resolved_claim_id(match_claim)
            if new_claim_id in blocked:
                new_claim_id = claim.id
            if new_claim_id != claim.id:
                id_remap[claim.id] = new_claim_id
            derived_claims.append(
                claim.model_copy(
                    deep=True,
                    update={
                        "id": new_claim_id,
                        "subject_id": match_claim.subject_id,
                        "object_id": match_claim.object_id,
                        "type": match_claim.type,
                        "status": match_claim.status,
                        "description": match_claim.description,
                        "source_text": match_claim.source_text,
                        "subject_name": names.get(
                            match_claim.subject_id, match_claim.subject_name
                        ),
                    },
                )
            )

        derived_claims = _merge_same_claims(derived_claims)

    if underived:
        logger.warning(
            "%d stored items cite a text unit without an extraction record; "
            "they keep their stored content until a full rebuild",
            len(underived),
        )
    return Derived(
        entities=derived_entities,
        relationships=derived_relationships,
        claims=derived_claims,
        underived=underived,
        id_remap=id_remap,
    )
