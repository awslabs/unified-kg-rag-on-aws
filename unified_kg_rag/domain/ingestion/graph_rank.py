# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-importance rank of entities and relationships (MS GraphRAG parity).

An entity's rank is its degree, the number of distinct entities it is
related to; a relationship's rank is the sum of its endpoints' degrees.
Retrieval orders entities by rank (``entity_importance_source="rank"``).
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping

from unified_kg_rag.domain.models import Entity, Relationship


def with_graph_ranks(
    entities: list[Entity],
    relationships: list[Relationship],
    stored_neighbors: Mapping[str, Iterable[str]] | None = None,
) -> tuple[list[Entity], list[Relationship]]:
    """Copies of ``entities`` and ``relationships`` with their ranks set.

    Degrees count the neighbours ``relationships`` link plus, per entity id,
    ``stored_neighbors`` (the stored graph's, for an incremental merge), each
    neighbour once.
    """
    neighbors: dict[str, set[str]] = defaultdict(set)
    for entity_id, ids in (stored_neighbors or {}).items():
        neighbors[entity_id].update(i for i in ids if i != entity_id)
    for rel in relationships:
        if rel.source_id != rel.target_id:
            neighbors[rel.source_id].add(rel.target_id)
            neighbors[rel.target_id].add(rel.source_id)

    def degree(entity_id: str) -> int:
        return len(neighbors.get(entity_id, ()))

    return (
        [e.model_copy(update={"rank": degree(e.id)}) for e in entities],
        [
            r.model_copy(update={"rank": degree(r.source_id) + degree(r.target_id)})
            for r in relationships
        ],
    )
