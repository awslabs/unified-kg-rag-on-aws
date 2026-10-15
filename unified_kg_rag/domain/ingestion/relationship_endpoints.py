# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Relationship endpoints cite the relationship's text units.

A chunk that names an entity in a relationship mentions it, so every endpoint
cites the text units of each relationship it is an endpoint of, whichever stage
produced the relationship (first-pass extraction, gleaning) and whichever
chunk extracted the entity itself. An entity's text units then depend only on
the chunks that name it, not on which chunks share a batch: an incremental run
(a smaller batch merged into the stored graph) cites the same text units as a
full build, and removing a document strips an entity down to exactly the text
units of the surviving documents that name it. The document lineage attributes
endpoints to their relationships' documents on the same rule.

Microsoft GraphRAG and LightRAG cite only the chunks that list the entity; this
is a deliberate divergence for incremental convergence (see docs/design.md).
"""

from __future__ import annotations

from collections.abc import Sequence

from unified_kg_rag.domain.models import Constants, Entity, Relationship

# Relationship attributes a stub endpoint inherits: the ones that place an item
# in an index (its suffix) and scope it by filter. Without them a stub lands in
# the default index while its edge is written under the relationship's suffix,
# where the edge's endpoint vertex does not exist.
_PLACEMENT_ATTRIBUTES = (Constants.INDEX.value, Constants.FILTERS.value)


def _placement_attributes(rel: Relationship) -> dict[str, object]:
    attributes = rel.attributes or {}
    return {key: attributes[key] for key in _PLACEMENT_ATTRIBUTES if key in attributes}


def cite_relationship_endpoints(
    entities: Sequence[Entity],
    relationships: Sequence[Relationship],
) -> tuple[list[Entity], int]:
    """Make every endpoint an entity citing its relationships' text units.

    Endpoint ids are name-derived, so an endpoint no entity in ``entities``
    has (the model named it only inside a relationship, or another chunk
    extracted it) becomes a minimal stub entity rather than leaving an orphan
    edge for the graph builder to drop. A stub takes the relationship's index
    and filter attributes, so it is written under the same suffix as its edge.
    Entities sharing an id each gain the
    endpoint citations, so a later duplicate merge keeps their own text units.

    Pure: inputs are not mutated; an entity whose text units change is
    copied. Idempotent.

    Returns:
        The entities (in input order, stubs appended) and the number of stubs.
    """
    known_ids = {e.id for e in entities}
    # Insertion-ordered dicts as ordered sets: membership is O(1), so an
    # entity with many edges costs linear rather than quadratic time.
    cited_by_edges: dict[str, dict[str, None]] = {}
    stubs: dict[str, Entity] = {}
    for rel in relationships:
        for ent_id, name in (
            (rel.source_id, rel.source_name),
            (rel.target_id, rel.target_name),
        ):
            if not ent_id:
                continue
            if ent_id not in known_ids and ent_id not in stubs:
                if not name:
                    continue
                stub: dict[str, object] = {
                    "id": ent_id,
                    "name": name,
                    "text_unit_ids": [],
                }
                if placement := _placement_attributes(rel):
                    stub["attributes"] = placement
                stubs[ent_id] = Entity.model_validate(stub)
            cited_by_edges.setdefault(ent_id, {}).update(
                dict.fromkeys(rel.text_unit_ids or [])
            )

    result: list[Entity] = []
    for entity in [*entities, *stubs.values()]:
        own = list(entity.text_unit_ids or [])
        own_set = set(own)
        units = own + [t for t in cited_by_edges.get(entity.id, {}) if t not in own_set]
        if units != own:
            entity = entity.model_copy(update={"text_unit_ids": units})
        result.append(entity)
    return result, len(stubs)


def sync_relationship_endpoint_names(
    relationships: Sequence[Relationship], entities: Sequence[Entity]
) -> int:
    """Rewrite each edge's endpoint names from the entities' id -> name map.

    ``source_name`` / ``target_name`` are denormalised copies of the entity
    names, and the indexers, the gleaning prompt and retrieval context read
    those copies rather than resolving the id. Renaming an entity in place
    (a gleaning correction) or re-pointing an edge at a merged entity's id
    (gleaning, graph resolution) leaves the copies naming the old entity, so
    they are re-derived from the ids the edge references. An endpoint id with
    no entity keeps its name. Mutates ``relationships``; returns the number of
    edges changed.
    """
    name_by_id = {entity.id: entity.name for entity in entities}
    resynced = 0
    for rel in relationships:
        source_name = name_by_id.get(rel.source_id, rel.source_name)
        target_name = name_by_id.get(rel.target_id, rel.target_name)
        if (source_name, target_name) != (rel.source_name, rel.target_name):
            rel.source_name, rel.target_name = source_name, target_name
            resynced += 1
    return resynced
