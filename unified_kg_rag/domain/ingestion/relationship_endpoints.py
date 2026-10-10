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

from unified_kg_rag.domain.models import Entity, Relationship


def cite_relationship_endpoints(
    entities: Sequence[Entity],
    relationships: Sequence[Relationship],
) -> tuple[list[Entity], int]:
    """Make every endpoint an entity citing its relationships' text units.

    Endpoint ids are name-derived, so an endpoint no entity in ``entities``
    has (the model named it only inside a relationship, or another chunk
    extracted it) becomes a minimal stub entity rather than leaving an orphan
    edge for the graph builder to drop. Entities sharing an id each gain the
    endpoint citations, so a later duplicate merge keeps their own text units.

    Pure: inputs are not mutated; an entity whose text units change is copied
    with ``frequency`` set to its text-unit count. Idempotent.

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
                stubs[ent_id] = Entity.model_validate(
                    {"id": ent_id, "name": name, "text_unit_ids": []}
                )
            cited_by_edges.setdefault(ent_id, {}).update(
                dict.fromkeys(rel.text_unit_ids or [])
            )

    result: list[Entity] = []
    for entity in [*entities, *stubs.values()]:
        own = list(entity.text_unit_ids or [])
        own_set = set(own)
        units = own + [t for t in cited_by_edges.get(entity.id, {}) if t not in own_set]
        if units != own:
            entity = entity.model_copy(
                update={"text_unit_ids": units, "frequency": len(units)}
            )
        result.append(entity)
    return result, len(stubs)
