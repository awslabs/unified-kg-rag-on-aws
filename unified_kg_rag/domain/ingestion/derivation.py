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
what a full build over the current corpus gives it. Ids stay the stored ones.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _type_key(value: str | None) -> str:
    return (value or Constants.DEFAULT_RELATIONSHIP_TYPE.value).strip().lower()


def derive(
    config: Config,
    entities: Sequence[Entity],
    relationships: Sequence[Relationship],
    claims: Sequence[Claim],
    records: Mapping[str, TextUnitExtraction],
    claim_records: Mapping[str, list[Claim]] | None = None,
) -> Derived:
    """Derive the content of the given stored items from ``records``.

    ``records`` (and ``claim_records``, the per-unit claims) are by text
    unit id and must hold every text unit the items cite; an item one of
    whose text units has none keeps its content and is listed in
    ``underived``. Returns copies.
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

    # Each stored entity is the resolved entity sharing most of its keys.
    stored_of: dict[str, str] = {}
    derived_entities: list[Entity] = []
    for entity in entities:
        keys = entity_keys(entity)
        candidates = [r for r in resolved["entities"] if keys & entity_keys(r)]
        if not complete(entity) or not candidates:
            derived_entities.append(entity.model_copy(deep=True))
            continue
        match = max(
            candidates,
            key=lambda r: (
                len(set(r.text_unit_ids or []) & set(entity.text_unit_ids or [])),
                r.id == entity.id,
            ),
        )
        stored_of[match.id] = entity.id
        derived_entities.append(
            entity.model_copy(
                deep=True,
                update={
                    "name": match.name,
                    "type": match.type,
                    "description": match.description,
                    "confidence": match.confidence,
                    "attributes": {
                        **(match.attributes or {}),
                        ENTITY_KEYS_ATTRIBUTE: sorted(entity_keys(match)),
                    },
                },
            )
        )
    names = {e.id: e.name for e in derived_entities}

    resolved_edges: dict[tuple[str, str, str], Relationship] = {}
    for rel in resolved["relationships"]:
        source = stored_of.get(rel.source_id, rel.source_id)
        target = stored_of.get(rel.target_id, rel.target_id)
        resolved_edges.setdefault((source, target, _type_key(rel.type)), rel)
    derived_relationships: list[Relationship] = []
    for rel in relationships:
        match_rel = resolved_edges.get(
            (rel.source_id, rel.target_id, _type_key(rel.type))
        )
        if not complete(rel) or match_rel is None:
            derived_relationships.append(rel.model_copy(deep=True))
            continue
        derived_relationships.append(
            rel.model_copy(
                deep=True,
                update={
                    "type": match_rel.type,
                    "description": match_rel.description,
                    # The stored per-unit strengths stay: they follow the
                    # stored order of the text units.
                    "attributes": {
                        **(match_rel.attributes or {}),
                        TEXT_UNIT_WEIGHTS_ATTRIBUTE: (rel.attributes or {}).get(
                            TEXT_UNIT_WEIGHTS_ATTRIBUTE
                        ),
                    },
                    "source_name": names.get(rel.source_id, rel.source_name),
                    "target_name": names.get(rel.target_id, rel.target_name),
                },
            )
        )

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
        by_id: dict[str, Claim] = {}
        for claim in resolved_claims:
            remapped = claim.model_copy(
                update={
                    "subject_id": stored_of.get(claim.subject_id, claim.subject_id),
                    "object_id": (
                        stored_of.get(claim.object_id, claim.object_id)
                        if claim.object_id
                        else None
                    ),
                }
            )
            by_id.setdefault(resolved_claim_id(remapped), remapped)
        for claim in claims:
            match_claim = by_id.get(claim.id)
            if not all(u in claim_records for u in claim.text_unit_ids or []):
                underived.append(claim.id)
                match_claim = None
            if match_claim is None:
                derived_claims.append(claim.model_copy(deep=True))
                continue
            derived_claims.append(
                claim.model_copy(
                    deep=True,
                    update={
                        "type": match_claim.type,
                        "status": match_claim.status,
                        "description": match_claim.description,
                        "source_text": match_claim.source_text,
                        "subject_name": names.get(
                            claim.subject_id, match_claim.subject_name
                        ),
                    },
                )
            )

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
    )
