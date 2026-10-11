# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from typing import Any

import networkx as nx

from unified_kg_rag.domain.models import Claim, Entity, Relationship
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)


def edge_relationships(data: dict[str, Any]) -> list[dict[str, Any]]:
    """The relationships an edge of the built graph carries (see GraphBuilder).

    An edge added elsewhere (one relationship's attributes) is its own list.
    """
    relationships = data.get("relationships")
    return list(relationships) if relationships else [data]


class GraphBuilder:
    def __init__(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        claims: list[Claim] | None = None,
    ) -> None:
        self.entities = entities
        self.relationships = relationships
        self.claims = claims or []
        self.graph = nx.Graph()
        self._entity_lookup = self._build_entity_lookup()

    def build(self) -> nx.Graph:
        logger.info(
            "Building knowledge graph with %s entities, %s relationships, and %s claims",
            len(self.entities),
            len(self.relationships),
            len(self.claims),
        )

        self._add_entity_nodes()
        self._add_relationship_edges()

        if self.claims:
            self._add_claims()

        logger.info(
            "Knowledge graph built successfully: %s nodes, %s edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    def _build_entity_lookup(self) -> dict[str, Entity]:
        return {entity.id: entity for entity in self.entities}

    def _add_entity_nodes(self) -> None:
        for entity in self.entities:
            self.graph.add_node(entity.id, **entity.model_dump(), node_type="entity")

    def _add_relationship_edges(self) -> None:
        """Add one undirected edge per entity pair, carrying every relationship.

        Two entities can be linked by several relationships (other types, or
        both directions); a plain ``nx.Graph`` keeps one edge per pair. The
        edge weight is the sum of their weights and ``relationships`` lists
        each one (by id), so community membership and report context keep all
        of them and Leiden sees the pair's total strength whatever the input
        order. The other edge attributes are the smallest-id relationship's.
        """
        skipped_count = 0
        by_pair: dict[frozenset[str], list[Relationship]] = defaultdict(list)
        for rel in self.relationships:
            if self._both_nodes_exist(rel.source_id, rel.target_id):
                by_pair[frozenset((rel.source_id, rel.target_id))].append(rel)
            else:
                skipped_count += 1

        for group in by_pair.values():
            group.sort(key=lambda rel: rel.id)
            first = group[0]
            self.graph.add_edge(
                first.source_id,
                first.target_id,
                **{
                    **first.model_dump(),
                    "weight": sum(
                        rel.weight if rel.weight is not None else 1.0 for rel in group
                    ),
                },
                relationships=[rel.model_dump() for rel in group],
                edge_type="relationship",
            )

        if skipped_count > 0:
            logger.warning(
                "Skipped %s relationships due to missing entity nodes", skipped_count
            )

    def _both_nodes_exist(self, source_id: str, target_id: str) -> bool:
        return bool(self.graph.has_node(source_id) and self.graph.has_node(target_id))

    def _add_claims(self) -> None:
        stats = {
            "connected": 0,
            "partially_connected": 0,
            "subject_connections": 0,
            "object_connections": 0,
        }

        for claim in self.claims:
            self.graph.add_node(claim.id, **claim.model_dump(), node_type="claim")
            connected_roles = self._connect_claim_to_entities(claim, stats)
            self._update_connection_stats(connected_roles, stats)

        self._log_claim_statistics(stats)

    def _connect_claim_to_entities(
        self, claim: Claim, stats: dict[str, int]
    ) -> list[str]:
        connected_roles = []
        connections = [
            ("subject", claim.subject_id, "is_subject_of"),
            ("object", claim.object_id, "is_object_of"),
        ]

        for role, entity_id, edge_type in connections:
            if self._add_claim_entity_connection(claim.id, entity_id, edge_type):
                stats[f"{role}_connections"] += 1
                connected_roles.append(role)

        return connected_roles

    def _add_claim_entity_connection(
        self, claim_id: str, entity_id: str | None, edge_type: str
    ) -> bool:
        entity = self._resolve_entity(entity_id)
        if entity:
            self.graph.add_edge(entity.id, claim_id, edge_type=edge_type)
            return True
        return False

    def _resolve_entity(self, entity_identifier: str | None) -> Entity | None:
        if not entity_identifier:
            return None
        return self._entity_lookup.get(entity_identifier)

    @staticmethod
    def _update_connection_stats(
        connected_roles: list[str], stats: dict[str, int]
    ) -> None:
        if len(connected_roles) == 2:
            stats["connected"] += 1
        elif len(connected_roles) == 1:
            stats["partially_connected"] += 1

    def _log_claim_statistics(self, stats: dict[str, int]) -> None:
        total_claims = len(self.claims)
        if total_claims == 0:
            return

        total_connected = stats["connected"] + stats["partially_connected"]
        total_connected_rate = (total_connected / total_claims) * 100

        logger.info(
            "Claims integration: %s/%s (%.1f%%) claims connected to entities",
            total_connected,
            total_claims,
            total_connected_rate,
        )

        disconnected_count = total_claims - total_connected
        if disconnected_count > 0:
            logger.warning(
                "%s claims could not be connected to any entities", disconnected_count
            )
