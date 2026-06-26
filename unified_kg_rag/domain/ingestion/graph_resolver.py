# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field
from tqdm import tqdm

from unified_kg_rag.domain.ingestion.base_resolver import BaseResolver, FuzzyMatcher
from unified_kg_rag.domain.models import Config, Entity, Relationship
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)


def find_all_matches_for_entity_task(
    entity_name: str, fuzzy_matcher: FuzzyMatcher
) -> list[str]:
    match_result = fuzzy_matcher.find_all_matches(entity_name)
    return [match[0] for match in match_result]


class EntityResolutionStats(BaseModel):
    original_entities: int = 0
    resolved_entities: int = 0
    entity_groups_created: int = 0
    processing_time: float = 0.0

    @property
    def reduction_rate(self) -> float:
        if self.original_entities == 0:
            return 0.0
        return (
            (self.original_entities - self.resolved_entities) / self.original_entities
        ) * 100


class RelationshipResolutionStats(BaseModel):
    original_relationships: int = 0
    resolved_relationships: int = 0
    self_referencing_removed: int = 0
    relationship_groups_created: int = 0
    processing_time: float = 0.0

    @property
    def reduction_rate(self) -> float:
        if self.original_relationships == 0:
            return 0.0
        return (
            (self.original_relationships - self.resolved_relationships)
            / self.original_relationships
        ) * 100


class GraphResolutionStats(BaseModel):
    entity_stats: EntityResolutionStats = Field(default_factory=EntityResolutionStats)
    relationship_stats: RelationshipResolutionStats = Field(
        default_factory=RelationshipResolutionStats
    )
    total_processing_time: float = 0.0

    @property
    def total_original_items(self) -> int:
        return (
            self.entity_stats.original_entities
            + self.relationship_stats.original_relationships
        )

    @property
    def total_resolved_items(self) -> int:
        return (
            self.entity_stats.resolved_entities
            + self.relationship_stats.resolved_relationships
        )

    @property
    def overall_reduction_rate(self) -> float:
        if self.total_original_items == 0:
            return 0.0
        return (
            (self.total_original_items - self.total_resolved_items)
            / self.total_original_items
        ) * 100


class EntityResolver(BaseResolver):
    def resolve(
        self, entities: list[Entity], *args: Any, **kwargs: Any
    ) -> tuple[list[Entity], dict[str, str], EntityResolutionStats]:
        logger.info("Starting entity resolution for %s entities", len(entities))
        return self._resolve_entities(entities)

    def _resolve_entities(
        self, entities: list[Entity]
    ) -> tuple[list[Entity], dict[str, str], EntityResolutionStats]:
        start_time = time.time()
        stats = EntityResolutionStats(original_entities=len(entities))

        entity_groups = self._group_similar_entities(entities)
        stats.entity_groups_created = len(entity_groups)

        resolved_entities = []
        entity_mapping = {}
        for group in entity_groups:
            if not group:
                continue
            merged_entity = self._merge_entities(group)
            resolved_entities.append(merged_entity)
            for original_entity in group:
                entity_mapping[original_entity.id] = merged_entity.id

        stats.resolved_entities = len(resolved_entities)
        stats.processing_time = time.time() - start_time

        self._log_completion_summary(stats)
        return resolved_entities, entity_mapping, stats

    @staticmethod
    def _log_completion_summary(stats: EntityResolutionStats) -> None:
        logger.info(
            "Entity resolution completed: %s -> "
            "%s entities "
            "(%.2f%% reduction) in %.2fs",
            stats.original_entities,
            stats.resolved_entities,
            stats.reduction_rate,
            stats.processing_time,
        )

    def _group_similar_entities(self, entities: list[Entity]) -> list[list[Entity]]:
        if not entities or len(entities) < 2:
            return [[e] for e in entities]

        logger.info(
            "Grouping %s entities using %s method",
            len(entities),
            self.config.processing.resolution_method.value,
        )

        # Map name -> ALL entities with that name (not last-writer-wins): two
        # distinct entities sharing a surface name (e.g. "Mercury" the planet vs
        # the element) must both survive grouping. Collapsing to one silently
        # dropped the others' type/description/text_unit_ids.
        entity_map: dict[str, list[Entity]] = defaultdict(list)
        for entity in entities:
            entity_map[entity.name].append(entity)
        entity_names = list(entity_map.keys())

        fuzzy_matcher = self._create_fuzzy_matcher(candidate_texts=entity_names)

        adjacency_list = defaultdict(set)
        executor_class = (
            ProcessPoolExecutor if self.use_process_pool else ThreadPoolExecutor
        )

        with executor_class(max_workers=self.max_workers) as executor:
            future_to_name = {
                executor.submit(
                    find_all_matches_for_entity_task, name, fuzzy_matcher
                ): name
                for name in entity_names
            }
            for future in tqdm(
                as_completed(future_to_name),
                total=len(entity_names),
                desc="Resolving Entities",
                disable=not self.show_progress,
            ):
                original_name = future_to_name[future]
                try:
                    matched_names = future.result()
                    for matched_name in matched_names:
                        if original_name != matched_name:
                            adjacency_list[original_name].add(matched_name)
                            adjacency_list[matched_name].add(original_name)
                except Exception as e:
                    logger.warning(
                        "Failed to find matches for entity '%s': %s", original_name, e
                    )

        groups = []
        visited = set()
        for name in entity_names:
            if name not in visited:
                current_group_names = set()
                q = [name]
                visited.add(name)
                head = 0
                while head < len(q):
                    u = q[head]
                    head += 1
                    current_group_names.add(u)
                    for v in adjacency_list[u]:
                        if v not in visited:
                            visited.add(v)
                            q.append(v)
                groups.append([e for n in current_group_names for e in entity_map[n]])

        logger.info("Created %s entity groups", len(groups))
        return groups

    def _merge_entities(self, entities: list[Entity]) -> Entity:
        if len(entities) == 1:
            return entities[0]

        canonical_name = self._get_most_common_value([e.name for e in entities])
        primary_entity = next(
            (e for e in entities if e.name == canonical_name), entities[0]
        )

        # Max, not mean: confidence is monotonic in evidence (consistent with the
        # extractor merge); averaging dilutes a well-supported entity.
        confidences = [e.confidence for e in entities if e.confidence is not None]
        merged_confidence = max(confidences) if confidences else 1.0

        merged_text_unit_ids = self._merge_lists(
            [e.text_unit_ids for e in entities if e.text_unit_ids]
        )

        return Entity(
            id=primary_entity.id,
            short_id=primary_entity.short_id,
            name=canonical_name,
            name_embedding=primary_entity.name_embedding,
            type=primary_entity.type,
            description=self._merge_descriptions(
                [e.description for e in entities if e.description]
            ),
            description_embedding=primary_entity.description_embedding,
            text_unit_ids=merged_text_unit_ids,
            community_ids=self._merge_lists(
                [e.community_ids for e in entities if e.community_ids]
            ),
            rank=max((e.rank for e in entities if e.rank is not None), default=1),
            # Recompute from text-unit support (pre-existing frequencies are
            # typically None in a full build), so frequency is a real signal.
            frequency=len(merged_text_unit_ids),
            confidence=merged_confidence,
            attributes=self._merge_attributes(
                [e.attributes for e in entities if e.attributes]
            ),
            created_at=min(
                (e.created_at for e in entities if e.created_at),
                default=datetime.now(),
            ),
            updated_at=datetime.now(),
        )


class RelationshipResolver(BaseResolver):
    def resolve(
        self,
        relationships: list[Relationship],
        entity_mapping: dict[str, str],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[list[Relationship], RelationshipResolutionStats]:
        logger.info(
            "Starting relationship resolution for %s relationships", len(relationships)
        )
        return self._resolve_relationships(relationships, entity_mapping)

    def _resolve_relationships(
        self,
        relationships: list[Relationship],
        entity_mapping: dict[str, str],
    ) -> tuple[list[Relationship], RelationshipResolutionStats]:
        start_time = time.time()
        stats = RelationshipResolutionStats(original_relationships=len(relationships))

        updated_relationships = []
        for rel in relationships:
            source_resolved_id = entity_mapping.get(rel.source_id, rel.source_id)
            target_resolved_id = entity_mapping.get(rel.target_id, rel.target_id)

            if source_resolved_id == target_resolved_id:
                stats.self_referencing_removed += 1
                continue

            updated_relationships.append(
                rel.model_copy(
                    update={
                        "source_id": source_resolved_id,
                        "target_id": target_resolved_id,
                    }
                )
            )

        if stats.self_referencing_removed > 0:
            logger.info(
                "Removed %s self-referencing relationships",
                stats.self_referencing_removed,
            )

        relationship_groups = self._group_similar_relationships(updated_relationships)
        stats.relationship_groups_created = len(relationship_groups)

        resolved_relationships = []
        for group in relationship_groups:
            if group:
                merged_relationship = self._merge_relationships(group)
                resolved_relationships.append(merged_relationship)

        stats.resolved_relationships = len(resolved_relationships)
        stats.processing_time = time.time() - start_time

        self._log_completion_summary(stats)
        return resolved_relationships, stats

    @staticmethod
    def _log_completion_summary(
        stats: RelationshipResolutionStats,
    ) -> None:
        logger.info(
            "Relationship resolution completed: "
            "%s -> %s relationships "
            "(%.2f%% reduction) in %.2fs",
            stats.original_relationships,
            stats.resolved_relationships,
            stats.reduction_rate,
            stats.processing_time,
        )

    @staticmethod
    def _group_similar_relationships(
        relationships: list[Relationship],
    ) -> list[list[Relationship]]:
        if not relationships:
            return []
        groups_dict = defaultdict(list)
        for rel in relationships:
            # Normalize the type so case/whitespace variants of the same relation
            # ("WORKS_FOR" vs "works_for") merge instead of staying split
            # (consistent with the gleaner and incremental merge_relationships).
            rel_type = (rel.type or "").strip().lower()
            key = (rel.source_id, rel.target_id, rel_type)
            groups_dict[key].append(rel)
        return list(groups_dict.values())

    def _merge_relationships(self, relationships: list[Relationship]) -> Relationship:
        if len(relationships) == 1:
            return relationships[0]
        primary_rel = relationships[0]
        return Relationship(
            id=primary_rel.id,
            short_id=primary_rel.short_id,
            source_id=primary_rel.source_id,
            source_name=primary_rel.source_name,
            target_id=primary_rel.target_id,
            target_name=primary_rel.target_name,
            type=primary_rel.type,
            weight=sum(r.weight for r in relationships if r.weight is not None) or 1.0,
            description=self._merge_descriptions(
                [r.description for r in relationships if r.description]
            ),
            description_embedding=primary_rel.description_embedding,
            text_unit_ids=self._merge_lists(
                [r.text_unit_ids for r in relationships if r.text_unit_ids]
            ),
            rank=max((r.rank for r in relationships if r.rank is not None), default=1),
            attributes=self._merge_attributes(
                [r.attributes for r in relationships if r.attributes]
            ),
            created_at=min(
                (r.created_at for r in relationships if r.created_at),
                default=datetime.now(),
            ),
            updated_at=datetime.now(),
        )


class GraphResolver:
    def __init__(
        self,
        config: Config,
        max_workers: int | None = None,
        use_process_pool: bool = True,
    ):
        self.entity_resolver = EntityResolver(config, max_workers, use_process_pool)
        self.relationship_resolver = RelationshipResolver(
            config, max_workers, use_process_pool
        )

    def resolve_graph(
        self, entities: list[Entity], relationships: list[Relationship]
    ) -> tuple[dict[str, Any], GraphResolutionStats]:
        start_time = time.time()
        logger.info(
            "Starting graph resolution with %s entities and %s relationships",
            len(entities),
            len(relationships),
        )

        (
            resolved_entities,
            entity_mapping,
            entity_stats,
        ) = self.entity_resolver.resolve(entities)

        (
            resolved_relationships,
            relationship_stats,
        ) = self.relationship_resolver.resolve(relationships, entity_mapping)

        stats = GraphResolutionStats(
            entity_stats=entity_stats,
            relationship_stats=relationship_stats,
            total_processing_time=time.time() - start_time,
        )

        self._log_completion_summary(stats)

        result = {
            "entities": resolved_entities,
            "relationships": resolved_relationships,
        }
        return result, stats

    @staticmethod
    def _log_completion_summary(stats: GraphResolutionStats) -> None:
        logger.info(
            "Graph resolution completed in %.2fs: "
            "%s -> %s items "
            "(%.2f%% reduction)",
            stats.total_processing_time,
            stats.total_original_items,
            stats.total_resolved_items,
            stats.overall_reduction_rate,
        )
