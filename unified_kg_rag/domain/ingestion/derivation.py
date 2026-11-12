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
name, type, description, confidence, IRI, attributes (and, for a claim,
status and source text) of the resolved item it corresponds to.

An item cites every text unit any of its member names is stated in, so the
records of its own text units hold all of its contributions: the result is
what a full build over the current corpus gives it, id included. An entity
whose derived name has another id than its stored one (a removed document
held the spelling that won) takes the derived id, and so do its edges and
claims; ``Derived.id_remap`` maps each stored id that changed to its new id,
so the caller re-points what else references it and deletes the old one.

A stored entity may hold what a full build keeps apart (two spellings merged
when their types agreed; a removed or edited unit changed the type of one):
each resolved entity sharing a key and a text unit with it is derived, the
best match under its id and every other one as an entity of its own
(``Derived.split_from``), with the edges and claims that resolve to it.
"""

from __future__ import annotations

from collections import ChainMap, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from unified_kg_rag.domain.ingestion.claim_resolver import (
    ClaimResolver,
    claim_identity_suffix,
    resolved_claim_id,
)
from unified_kg_rag.domain.ingestion.contributions import (
    merge_contributions,
)
from unified_kg_rag.domain.ingestion.graph_resolver import (
    ENTITY_KEYS_ATTRIBUTE,
    GraphResolver,
    entity_keys,
)
from unified_kg_rag.domain.ingestion.names import compact_name_key
from unified_kg_rag.domain.ingestion.relationship_weights import (
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
from unified_kg_rag.shared.utils import entity_key

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
    # Id of each derived entity a stored one held part of but a full build
    # keeps apart -> the stored entity's id (its stored edges and claims may
    # now belong to it).
    split_from: dict[str, str] = Field(default_factory=dict)
    # Ids of the given edges a full build does not draw (their endpoints now
    # resolve to one entity), and of the given claims no claim of their text
    # units corresponds to: the caller deletes them.
    dropped: list[str] = Field(default_factory=list)
    # Stored entities whose stored edges may now belong to another derived
    # entity (they were re-keyed or split): the caller derives those edges
    # with them.
    rewired: list[str] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)


def _type_key(value: str | None) -> str:
    return (value or Constants.DEFAULT_RELATIONSHIP_TYPE.value).strip().lower()


def _union(lists: Iterable[list[str] | None]) -> list[str]:
    return list(dict.fromkeys(i for ids in lists for i in ids or []))


def with_entity_ids(claim: Claim, ids: Mapping[str, str]) -> Claim:
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


def _ordered_units(stored: Iterable[str], units: Sequence[str]) -> list[str]:
    """``units`` in the order ``stored`` lists them, the others after."""
    wanted = set(units)
    first = [u for u in dict.fromkeys(stored) if u in wanted]
    seen = set(first)
    return [*first, *(u for u in units if u not in seen)]


def _correspondence(
    given: Sequence[Entity | Relationship],
    candidates_of: Mapping[str, list[Entity | Relationship]],
    complete: Mapping[str, bool],
) -> tuple[dict[str, list[Any]], dict[str, list[Any]], dict[str, list[str]]]:
    """Match given items to the resolved items they correspond to.

    ``candidates_of`` holds each given item's candidates (by its id). Its
    match is the one of its own id, else the one sharing most of its text
    units (the smallest id on a tie);
    every other one sharing a text unit with it is a part a full build keeps
    apart. Returns, by resolved id, the given items it is the match of and
    those it is a part of, and by given id, the resolved ids it derives to
    (its match first).
    """
    matched: dict[str, list[Any]] = defaultdict(list)
    parts: dict[str, list[Any]] = defaultdict(list)
    derives_to: dict[str, list[str]] = {}
    for item in given:
        # In id order: a tie goes to the smallest id, whatever order the
        # candidates were collected in (an entity's keys are a set).
        candidates = sorted(candidates_of.get(item.id, []), key=lambda r: r.id)
        if not complete[item.id] or not candidates:
            continue
        units = set(item.text_unit_ids or [])
        match = max(
            candidates,
            key=lambda r: (r.id == item.id, len(set(r.text_unit_ids or []) & units)),
        )
        matched[match.id].append(item)
        derives_to[item.id] = [match.id]
        for other in candidates:
            if other is not match and units & set(other.text_unit_ids or []):
                parts[other.id].append(item)
                derives_to[item.id].append(other.id)
    return matched, parts, derives_to


@dataclass(frozen=True)
class _EntityDerivation:
    """What deriving the given entities decided, which their edges and
    claims follow (see :func:`_derive_entities`)."""

    # The derived entities, in the given entities' order.
    entities: list[Entity]
    # Given entity id -> the resolved ids it derives to (its match first).
    derives_to: dict[str, list[str]]
    # Resolved entity id -> its derived id.
    derived_of: dict[str, str]
    # Resolved entity id -> every id a given claim may name it by.
    aliases: dict[str, set[str]]
    # Derived id -> the given entity, of an entity no given one matched.
    created: dict[str, str]
    # Given entity id -> whether all of its text units have a record.
    complete: dict[str, bool]
    # Derived id -> derived name.
    names: dict[str, str]
    # See ``Derived.split_from``.
    parted_from: dict[str, str]
    # The given entities a part split off.
    split_members: set[str]
    # The derived ids of the resolved entities.
    derived_ids: set[str]
    # The given entities one of whose text units has no record.
    incomplete: set[str]


@dataclass(frozen=True)
class _ClaimDerivation:
    """The derived claims, and what deriving them adds to ``Derived``."""

    claims: list[Claim] = field(default_factory=list)
    underived: list[str] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    id_remap: dict[str, str] = field(default_factory=dict)


def _entity_ids(
    id_remap: Mapping[str, str], rekeyed: Mapping[str, str]
) -> Callable[[str], str]:
    """The id a stored entity has now: its re-key by this derivation (as
    ``id_remap`` holds it when asked), else by an earlier one of this run."""

    def entity_id(stored_id: str) -> str:
        return id_remap.get(stored_id) or rekeyed.get(stored_id, stored_id)

    return entity_id


def derive(
    config: Config,
    entities: Sequence[Entity],
    relationships: Sequence[Relationship],
    claims: Sequence[Claim],
    records: Mapping[str, TextUnitExtraction],
    claim_records: Mapping[str, list[Claim]] | None = None,
    rekeyed: Mapping[str, str] | None = None,
    blocked: Collection[str] = (),
    split_from: Mapping[str, str] | None = None,
    stored_relationship_ids: Collection[str] | None = None,
) -> Derived:
    """Derive the content of the given stored items from ``records``.

    ``rekeyed`` maps stored entity ids an earlier derivation of this run
    replaced to their new ids (``Derived.id_remap``), and ``split_from`` the
    entities it split off to the stored ones they came from
    (``Derived.split_from``): an item still pointing at an old id follows
    it, and a claim keyed by an old id is matched. ``blocked`` ids are held
    by stored items outside the given ones: no item is re-keyed to one (it
    keeps its id rather than overwrite another), nor split off under one.
    ``stored_relationship_ids`` are the ids of the stored edges incident to
    the given entities, when the caller read them: an edge a full build
    draws between two derived entities that is neither stored nor given is
    derived too (an earlier merge of its endpoints dropped it as a
    self-loop).

    ``records`` (and ``claim_records``, the per-unit claims) are by text
    unit id and must hold every text unit the items cite; an item one of
    whose text units has none keeps its content and is listed in
    ``underived`` (its id changes only to follow a re-keyed endpoint).
    Given claims that derive to one fact are returned as one, and a claim
    their text units state that a given claim had merged (about another
    spelling a full build keeps apart) is returned as a claim of its own.
    Returns copies.
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
    resolver = GraphResolver(config)
    resolved, _ = resolver.resolve_graph(contributions, edges)

    rekeyed = rekeyed or {}
    id_remap: dict[str, str] = {}
    # Entities first (their edges and claims follow the ids they take), then
    # edges, then claims: ``underived``, ``dropped`` and ``id_remap`` list
    # each phase's items after the earlier phases'.
    entity_derivation = _derive_entities(
        entities, resolved["entities"], complete, id_remap, rekeyed, blocked, split_from
    )
    derived_relationships, dropped = _derive_relationships(
        entity_derivation,
        entities,
        relationships,
        resolved["relationships"],
        complete,
        id_remap,
        rekeyed,
        blocked,
        stored_relationship_ids,
    )
    claim_derivation = _derive_claims(
        config,
        entity_derivation,
        claims,
        claim_records,
        unit_ids,
        resolved["entities"],
        id_remap,
        rekeyed,
        blocked,
    )
    derived_entities = entity_derivation.entities
    derived_claims = claim_derivation.claims
    parted_from = entity_derivation.parted_from
    split_members = entity_derivation.split_members
    underived.extend(claim_derivation.underived)
    dropped.extend(claim_derivation.dropped)
    id_remap.update(claim_derivation.id_remap)

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
        split_from=parted_from,
        dropped=dropped,
        rewired=sorted(
            {
                *split_members,
                *(e.id for e in entities if e.id in id_remap),
                # The entities another one merged into: their stored edges
                # may now equal one the merge drew.
                *(id_remap[e.id] for e in entities if e.id in id_remap),
            }
        ),
    )


def _derive_entities(
    entities: Sequence[Entity],
    resolved_entity_list: list[Entity],
    complete: Callable[[Entity], bool],
    id_remap: dict[str, str],
    rekeyed: Mapping[str, str],
    blocked: Collection[str],
    split_from: Mapping[str, str] | None,
) -> _EntityDerivation:
    """Derive the given stored entities from the resolved ones (see
    :func:`derive`); adds their re-keys to ``id_remap``."""
    old_of = {new: old for old, new in rekeyed.items()} | dict(split_from or {})
    created: dict[str, str] = {}

    # Each stored entity derives to the resolved entities sharing one of its
    # keys (an index, not a scan: a first build derives every entity). The
    # stored entities matching one resolved entity (two spellings indexed
    # apart, or the old and new copies an interrupted re-key left) become
    # one, under the resolved entity's id unless another given entity holds
    # it or it is ``blocked``.
    by_key: dict[str, list[Entity]] = defaultdict(list)
    for r in resolved_entity_list:
        for key in entity_keys(r):
            by_key[key].append(r)
    resolved_entities = {r.id: r for r in resolved_entity_list}
    entity_candidates: dict[str, list[Entity | Relationship]] = {}
    for entity in entities:
        found = {r.id: r for key in entity_keys(entity) for r in by_key.get(key, ())}
        entity_candidates[entity.id] = list(found.values())
    entity_complete = {e.id: complete(e) for e in entities}
    matched, parts, derives_to = _correspondence(
        entities, entity_candidates, entity_complete
    )
    taken = {entity.id for entity in entities} | set(blocked)
    # The text units and keys of a part split off under the id of a given
    # entity kept as stored (see below), by that id.
    gained: dict[str, list[str]] = defaultdict(list)
    gained_keys: dict[str, set[str]] = defaultdict(set)
    # Resolved id -> the derived id, and every id a given claim may name
    # the entity by (its stored copies' ids, its derived id, the id an
    # earlier re-key of this run replaced or it was split off from).
    derived_of: dict[str, str] = {}
    aliases: dict[str, set[str]] = {}
    derived_entities: list[Entity] = []
    emitted: set[str] = set()
    for entity in entities:
        targets = derives_to.get(entity.id)
        if targets is None:
            derived_entities.append(entity.model_copy(deep=True))
            continue
        for resolved_id in targets:
            if resolved_id in emitted:
                continue
            match = resolved_entities[resolved_id]
            group = matched.get(resolved_id, [])
            if group:
                ids = [e.id for e in group]
                new_id = (
                    match.id if match.id in ids or match.id not in taken else ids[0]
                )
                id_remap.update({i: new_id for i in ids if i != new_id})
                base = group[0]
            elif match.id in taken:
                # Split off under an id another item holds: kept merged. A
                # given holder kept as stored (a unit of it has no record:
                # a removal killed between the stores) takes the part's text
                # units, which the match no longer cites, and its keys (the
                # names those units state), so a later derivation of the
                # holder finds them and the entities they resolve to.
                if match.id in entity_complete and not entity_complete[match.id]:
                    gained[match.id].extend(match.text_unit_ids or [])
                    gained_keys[match.id] |= entity_keys(match)
                continue
            else:
                new_id = match.id
                created[new_id] = entity.id
                base = match
            emitted.add(resolved_id)
            members = [*group, *parts.get(resolved_id, [])]
            derived_of[match.id] = new_id
            aliases[match.id] = {
                *(e.id for e in members),
                new_id,
                old_of.get(new_id, new_id),
            }
            derived_entities.append(
                base.model_copy(
                    deep=True,
                    update={
                        "id": new_id,
                        "name": match.name,
                        "type": match.type,
                        "description": match.description,
                        "confidence": match.confidence,
                        "iri": match.iri,
                        "text_unit_ids": _ordered_units(
                            (u for e in group for u in e.text_unit_ids or []),
                            match.text_unit_ids or [],
                        ),
                        "community_ids": _union(e.community_ids for e in members),
                        "attributes": {
                            **(match.attributes or {}),
                            ENTITY_KEYS_ATTRIBUTE: sorted(entity_keys(match)),
                        },
                    },
                )
            )
    for derived_entity in derived_entities:
        if derived_entity.id in gained:
            derived_entity.text_unit_ids = list(
                dict.fromkeys(
                    [*(derived_entity.text_unit_ids or []), *gained[derived_entity.id]]
                )
            )
            derived_entity.attributes = {
                **(derived_entity.attributes or {}),
                ENTITY_KEYS_ATTRIBUTE: sorted(
                    entity_keys(derived_entity) | gained_keys[derived_entity.id]
                ),
            }
    names = {e.id: e.name for e in derived_entities}
    # Each part a stored entity lost (split off on its own, or into another
    # derived entity) -> that stored entity (the first, when several did),
    # and every stored entity a part split off: each may hold an edge the
    # part states, so all are rewired.
    split_parts = {
        resolved_id: members
        for resolved_id, members in parts.items()
        if resolved_id in derived_of
        and derived_of[resolved_id] not in {m.id for m in members}
    }
    parted_from = {
        derived_of[resolved_id]: members[0].id
        for resolved_id, members in split_parts.items()
    }
    split_members = {m.id for members in split_parts.values() for m in members}
    derived_ids = set(derived_of.values())
    incomplete = {e.id for e in entities if not entity_complete[e.id]}
    return _EntityDerivation(
        entities=derived_entities,
        derives_to=derives_to,
        derived_of=derived_of,
        aliases=aliases,
        created=created,
        complete=entity_complete,
        names=names,
        parted_from=parted_from,
        split_members=split_members,
        derived_ids=derived_ids,
        incomplete=incomplete,
    )


def _derive_relationships(
    entity_derivation: _EntityDerivation,
    entities: Sequence[Entity],
    relationships: Sequence[Relationship],
    resolved_relationships: list[Relationship],
    complete: Callable[[Relationship], bool],
    id_remap: dict[str, str],
    rekeyed: Mapping[str, str],
    blocked: Collection[str],
    stored_relationship_ids: Collection[str] | None,
) -> tuple[list[Relationship], list[str]]:
    """Derive the given stored edges, and those a full build draws that no
    stored edge is, between the derived entities (see :func:`derive`).

    Returns them and the ids of the given edges that go; adds the edges'
    re-keys to ``id_remap``.
    """
    derived_entities = entity_derivation.entities
    derives_to = entity_derivation.derives_to
    derived_of = entity_derivation.derived_of
    created = entity_derivation.created
    names = entity_derivation.names
    derived_ids = entity_derivation.derived_ids
    entity_id = _entity_ids(id_remap, rekeyed)

    # A stored endpoint may also be another spelling of its name stated in
    # the edge's text units: a run that split the two and died before their
    # edges were written left an edge of one on the other, and the stored
    # entity no longer holds the other's key (nor, always, the edge's units).
    by_compact_key: dict[str, list[Entity]] = defaultdict(list)
    for derived_entity in derived_entities:
        by_compact_key[compact_name_key(derived_entity.name)].append(derived_entity)
    names_of = {entity.id: entity.name for entity in entities}

    def endpoint_ids(stored_id: str, units: Collection[str] = ()) -> list[str]:
        """The derived ids a stored endpoint may now be (its match first),
        then its other spellings stated in ``units``."""
        targets = [
            derived_of[r] for r in derives_to.get(stored_id, []) if r in derived_of
        ] or [entity_id(stored_id)]
        name = names_of.get(stored_id)
        spelled = (
            [
                other.id
                for other in by_compact_key.get(compact_name_key(name), ())
                if set(units) & set(other.text_unit_ids or [])
            ]
            if name is not None
            else []
        )
        return list(dict.fromkeys([*targets, *spelled]))

    resolved_edges: dict[tuple[str, str, str], Relationship] = {}
    for rel in resolved_relationships:
        source = derived_of.get(rel.source_id, rel.source_id)
        target = derived_of.get(rel.target_id, rel.target_id)
        resolved_edges.setdefault((source, target, _type_key(rel.type)), rel)
    edge_key = {rel.id: key for key, rel in resolved_edges.items()}
    resolved_by_id = {rel.id: rel for rel in resolved_edges.values()}
    # Stored edges of one resolved edge become one, like the entities; a
    # stored edge between entities that split derives to the edges between
    # their parts it shares text units with.
    edge_candidates: dict[str, list[Entity | Relationship]] = {}
    for rel in relationships:
        type_ = _type_key(rel.type)
        edge_candidates[rel.id] = [
            resolved_edges[(source, target, type_)]
            for source in endpoint_ids(rel.source_id, rel.text_unit_ids or ())
            for target in endpoint_ids(rel.target_id, rel.text_unit_ids or ())
            if (source, target, type_) in resolved_edges
        ]
    edge_complete = {r.id: complete(r) for r in relationships}
    edge_matched, edge_parts, edge_derives_to = _correspondence(
        relationships, edge_candidates, edge_complete
    )
    edge_taken = {rel.id for rel in relationships} | set(blocked)
    derived_relationships: list[Relationship] = []
    dropped: list[str] = []
    edges_emitted: set[str] = set()
    for rel in relationships:
        targets = edge_derives_to.get(rel.id)
        if (
            targets is None
            and edge_complete[rel.id]
            and rel.source_id in derives_to
            and rel.target_id in derives_to
        ):
            # A full build over its text units, which hold both endpoints'
            # statements, draws no such edge (its endpoints now resolve to
            # one entity): it goes.
            dropped.append(rel.id)
            continue
        if targets is None:
            # Kept as stored, re-pointed at a re-keyed endpoint.
            source, target = entity_id(rel.source_id), entity_id(rel.target_id)
            moved: dict[str, str | None] = {}
            if rel.source_id in id_remap:
                moved.update(source_id=source, source_name=names.get(source))
            if rel.target_id in id_remap:
                moved.update(target_id=target, target_name=names.get(target))
            derived_relationships.append(rel.model_copy(deep=True, update=moved))
            continue
        for resolved_id in targets:
            if resolved_id in edges_emitted:
                continue
            match_rel = resolved_by_id[resolved_id]
            same = edge_matched.get(resolved_id, [])
            if same:
                ids = [r.id for r in same]
                new_rel_id = (
                    match_rel.id
                    if match_rel.id in ids or match_rel.id not in edge_taken
                    else ids[0]
                )
                id_remap.update({i: new_rel_id for i in ids if i != new_rel_id})
                base_rel = same[0]
            elif match_rel.id in edge_taken:
                continue
            else:
                new_rel_id = match_rel.id
                base_rel = match_rel
            edges_emitted.add(resolved_id)
            source, target, _ = edge_key[resolved_id]
            merged = base_rel.model_copy(
                deep=True,
                update={
                    "id": new_rel_id,
                    "source_id": source,
                    "target_id": target,
                    "source_name": names.get(source, base_rel.source_name),
                    "target_name": names.get(target, base_rel.target_name),
                    "type": match_rel.type,
                    "description": match_rel.description,
                    "attributes": dict(match_rel.attributes or {}),
                },
            )
            # The strengths a full build gives the edge in its text units (a
            # stored or delta edge may hold another merge's), in the stored
            # order of the units.
            stored: dict[str, float] = {}
            for r in same:
                stored = overlay_weights(stored, text_unit_weights(r))
            resolved_weights = text_unit_weights(match_rel)
            apply_text_unit_weights(
                merged,
                {
                    unit: resolved_weights.get(unit, stored.get(unit, 0.0))
                    for unit in _ordered_units(
                        _union(r.text_unit_ids for r in same),
                        match_rel.text_unit_ids or [],
                    )
                },
            )
            derived_relationships.append(merged)
    # The edges a full build draws between two derived entities that no
    # stored edge is (a part an entity split into has none; an earlier merge
    # of the endpoints dropped their edge as a self-loop). Their text units
    # are all here: an edge cites only units its endpoints cite.
    for (source, target, _), rel in resolved_edges.items():
        unstored = (
            source in created
            or target in created
            or (
                stored_relationship_ids is not None
                and rel.id not in stored_relationship_ids
            )
        )
        if (
            unstored
            and rel.id not in edges_emitted
            and rel.id not in edge_taken
            and source in derived_ids
            and target in derived_ids
        ):
            derived_relationships.append(
                rel.model_copy(
                    deep=True,
                    update={
                        "source_id": source,
                        "target_id": target,
                        "source_name": names.get(source, rel.source_name),
                        "target_name": names.get(target, rel.target_name),
                    },
                )
            )
    return derived_relationships, dropped


def _compact_names(claim: Claim) -> tuple[str, str | None]:
    """The compact keys of a claim's subject and object names."""
    return (
        compact_name_key(claim.subject_name),
        compact_name_key(claim.object_name) if claim.object_name else None,
    )


def _fact(claim: Claim) -> tuple[str, str]:
    """What tells two statements about the same names apart: the type and
    the dates, as :func:`resolved_claim_id` keys them (with the statement,
    which ``_derive_claims`` compares after them)."""
    return (
        (claim.type or "").casefold(),
        claim_identity_suffix(claim.start_date, claim.end_date, None),
    )


def _derive_claims(
    config: Config,
    entity_derivation: _EntityDerivation,
    claims: Sequence[Claim],
    claim_records: Mapping[str, list[Claim]] | None,
    unit_ids: Collection[str],
    resolved_entity_list: list[Entity],
    prior_id_remap: Mapping[str, str],
    rekeyed: Mapping[str, str],
    blocked: Collection[str],
) -> _ClaimDerivation:
    """Derive the given stored claims from their text units' statements,
    resolved against ``resolved_entity_list`` (see :func:`derive`).

    ``prior_id_remap`` holds the re-keys of the entities and edges; the
    claims' own come back in the result, which nothing given is changed for.
    """
    if not claims:
        return _ClaimDerivation()
    derived_of = entity_derivation.derived_of
    aliases = entity_derivation.aliases
    names = entity_derivation.names
    derived_ids = entity_derivation.derived_ids
    incomplete = entity_derivation.incomplete
    # Reads see the earlier phases' re-keys and the claims' own; writes go
    # to the claims' own only.
    claim_remap: dict[str, str] = {}
    id_remap = ChainMap(claim_remap, dict(prior_id_remap))
    entity_id = _entity_ids(id_remap, rekeyed)
    underived: list[str] = []
    dropped: list[str] = []
    derived_claims: list[Claim] = []

    claim_records = claim_records or {}
    statements = [
        claim.model_copy(deep=True)
        for unit_id in sorted(unit_ids)
        for claim in claim_records.get(unit_id, [])
    ]
    resolved_claims, _ = ClaimResolver(
        config, max_workers=1, use_process_pool=False
    ).resolve(statements, resolved_entity_list)
    # Matched to the given claims by the id they have under the stored
    # entity ids or under the derived ones (a claim the caller already
    # re-pointed); each takes the id it has under the derived ones.
    by_id: dict[str, Claim] = {}
    for claim in resolved_claims:
        exact = with_entity_ids(claim, derived_of)
        by_id.setdefault(resolved_claim_id(exact), claim)
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
    # The derived claim of each given one (None: dropped), in order.
    kept_for: list[Claim | None] = []
    for claim in claims:
        match_claim = by_id.get(claim.id)
        endpoints = {
            "subject_id": entity_id(claim.subject_id),
            "object_id": entity_id(claim.object_id) if claim.object_id else None,
        }
        units = set(claim.text_unit_ids or [])
        if not all(u in claim_records for u in units):
            underived.append(claim.id)
            match_claim = None
        elif match_claim is None and units:
            # No resolved claim has its id under any spelling (its subject
            # or object resolves to an entity of a third spelling): it is a
            # resolved claim stated in its text units, else a full build has
            # no such claim. Of those, one about the names it holds (by
            # their compact keys: a legal form or a dense-script space does
            # not tell a spelling apart), not another statement of its unit
            # about another entity; of those, one of its type and dates; then
            # then the same statement; then the one stated in most of its
            # units.
            sharing = [r for r in resolved_claims if units & set(r.text_unit_ids or [])]
            if not sharing:
                dropped.append(claim.id)
                kept_for.append(None)
                continue
            own_names, own_fact = _compact_names(claim), _fact(claim)
            own_statement = entity_key(claim.description)
            match_claim = max(
                sharing,
                key=lambda r: (
                    _compact_names(r) == own_names,
                    _fact(r) == own_fact,
                    entity_key(r.description) == own_statement,
                    len(units & set(r.text_unit_ids or [])),
                    resolved_claim_id(with_entity_ids(r, derived_of)),
                ),
            )
        if match_claim is None:
            derived_claims.append(claim.model_copy(deep=True, update=endpoints))
            kept_for.append(derived_claims[-1])
            continue
        match_claim = with_entity_ids(match_claim, derived_of)
        # An endpoint no given entity derives to that an earlier derivation
        # of this run re-keyed follows it.
        moved = {
            key: entity_id(value)
            for key in ("subject_id", "object_id")
            if (value := getattr(match_claim, key))
            and value not in derived_ids
            and entity_id(value) != value
        }
        if moved:
            match_claim = match_claim.model_copy(update=moved)
        # An endpoint whose stored entity could not be derived (one of its
        # text units has no record) resolved from the claim's units alone
        # to an entity no derived or stored item is: keep the stored one.
        kept = {
            key: endpoints[key]
            for key in ("subject_id", "object_id")
            if getattr(match_claim, key)
            and getattr(match_claim, key) not in derived_ids
            and endpoints[key] in incomplete
        }
        if kept:
            match_claim = match_claim.model_copy(update=kept)
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
                    # Part of its new id (resolved_claim_id): a given
                    # claim of other dates kept them under the matched
                    # claim's id.
                    "start_date": match_claim.start_date,
                    "end_date": match_claim.end_date,
                    "description": match_claim.description,
                    "source_text": match_claim.source_text,
                    # The units its statements are in now: a given claim
                    # may hold another spelling's statement merged in.
                    "text_unit_ids": _ordered_units(
                        claim.text_unit_ids or [], match_claim.text_unit_ids or []
                    ),
                    # Its text units' placement and document filters, as
                    # a relationship's: a stored claim's are a superset
                    # after a removal (or none, read back without them).
                    "attributes": dict(match_claim.attributes or {}),
                    "subject_name": names.get(
                        match_claim.subject_id, match_claim.subject_name
                    ),
                    "object_name": (
                        names.get(match_claim.object_id, match_claim.object_name)
                        if match_claim.object_id
                        else match_claim.object_name
                    ),
                },
            )
        )
        kept_for.append(derived_claims[-1])

    # Claims a full build keeps apart that a given claim merged (its
    # subject merged with another spelling): derived as claims of their own.
    given_units = {u for c in claims for u in c.text_unit_ids or []}
    emitted_ids = {c.id for c in derived_claims}
    # Units stating a claim kept apart about an entity not derived (a
    # spelling an underived stored entity holds): no claim of its own
    # can be emitted yet, so the given claim keeps them.
    unemitted: set[str] = set()
    for claim in resolved_claims:
        split = with_entity_ids(claim, derived_of)
        split_id = resolved_claim_id(split)
        if (
            split_id in emitted_ids
            or split_id in blocked
            or not given_units & set(claim.text_unit_ids or [])
        ):
            continue
        if split.subject_id not in derived_ids or (
            split.object_id and split.object_id not in derived_ids
        ):
            unemitted |= given_units & set(claim.text_unit_ids or [])
            continue
        emitted_ids.add(split_id)
        derived_claims.append(
            split.model_copy(
                deep=True,
                update={
                    "id": split_id,
                    "subject_name": names.get(split.subject_id, split.subject_name),
                    "object_name": (
                        names.get(split.object_id, split.object_name)
                        if split.object_id
                        else split.object_name
                    ),
                },
            )
        )
    if unemitted:
        cited = {u for c in derived_claims for u in c.text_unit_ids or []}
        for given, derived_claim in zip(claims, kept_for, strict=True):
            if derived_claim is None:
                continue
            if extra := [
                u
                for u in given.text_unit_ids or []
                if u in unemitted and u not in cited
            ]:
                derived_claim.text_unit_ids = [
                    *(derived_claim.text_unit_ids or []),
                    *extra,
                ]
                cited.update(extra)
    derived_claims = _merge_same_claims(derived_claims)
    return _ClaimDerivation(
        claims=derived_claims,
        underived=underived,
        dropped=dropped,
        id_remap=claim_remap,
    )
