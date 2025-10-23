import networkx as nx

from aws_graphrag.core import get_logger
from aws_graphrag.models import Claim, Entity, Relationship

logger = get_logger(__name__)


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
            f"Building knowledge graph with {len(self.entities)} entities, "
            f"{len(self.relationships)} relationships, and {len(self.claims)} claims"
        )

        self._add_entity_nodes()
        self._add_relationship_edges()

        if self.claims:
            self._add_claims()

        logger.info(
            f"Knowledge graph built successfully: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        return self.graph

    def _build_entity_lookup(self) -> dict[str, Entity]:
        return {entity.id: entity for entity in self.entities}

    def _add_entity_nodes(self) -> None:
        for entity in self.entities:
            self.graph.add_node(entity.id, **entity.model_dump(), node_type="entity")

    def _add_relationship_edges(self) -> None:
        skipped_count = 0

        for rel in self.relationships:
            if self._both_nodes_exist(rel.source_id, rel.target_id):
                self.graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    **rel.model_dump(),
                    edge_type="relationship",
                )
            else:
                skipped_count += 1

        if skipped_count > 0:
            logger.warning(
                f"Skipped {skipped_count} relationships due to missing entity nodes"
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
            f"Claims integration: {total_connected}/{total_claims} "
            f"({total_connected_rate:.1f}%) claims connected to entities"
        )

        disconnected_count = total_claims - total_connected
        if disconnected_count > 0:
            logger.warning(
                f"{disconnected_count} claims could not be connected to any entities"
            )
