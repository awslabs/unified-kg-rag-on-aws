# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from unified_kg_rag.domain.ingestion.base_resolver import (
    BaseResolver,
    FuzzyMatcher,
    normalize_entity_type,
)
from unified_kg_rag.domain.models import Config, Entity, Relationship
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor

logger = get_logger(__name__)

# Emit a progress log line every N completed items in the resolver's parallel
# loop. The resolver was historically the stage that could run for hours while
# looking like a hang; periodic logging keeps "slow" distinguishable from
# "stuck" without pulling a terminal progress-bar (tqdm) dependency into the
# technology-agnostic domain layer.
_RESOLVE_PROGRESS_EVERY = 2000


# One FuzzyMatcher per worker process, populated once by the pool initializer.
# ProcessPoolExecutor pickles every submit() argument and ships it to the worker
# on each task; passing the matcher (its MinHash LSH tables + per-candidate
# MinHashes run to tens of MB at scale) per task turned an O(ms) similarity
# lookup into an O(0.8s) serialization transfer, so a real corpus (~18k entity
# names => ~1.4 TB cumulative transfer) never finished. The initializer sends
# the matcher ONCE per worker instead of once per task.
_worker_fuzzy_matcher: FuzzyMatcher | None = None


def _init_worker_fuzzy_matcher(fuzzy_matcher: FuzzyMatcher) -> None:
    global _worker_fuzzy_matcher
    _worker_fuzzy_matcher = fuzzy_matcher


def find_all_matches_for_entity_task(entity_name: str) -> list[tuple[str, float]]:
    if _worker_fuzzy_matcher is None:
        raise RuntimeError(
            "FuzzyMatcher not initialized: run inside a pool started with "
            "_init_worker_fuzzy_matcher"
        )
    return _worker_fuzzy_matcher.find_all_matches(entity_name)


class _TypeAwareUnionFind:
    """Union-find over entity names that refuses type-conflicting unions.

    Each component tracks the type every typed member shares: ``None`` while
    no member carries a known type, otherwise the set of types common to all
    typed members (empty once members disagree, e.g. a surface name extracted
    as both ``organization`` and ``person``). A union is allowed only while
    that common set stays non-empty, so every pair of typed members drawn from
    different names in a finished group has the same type. This blocks the
    transitive chain the previous BFS allowed (``organization`` ~ untyped ~
    ``person`` collapsing into one entity). Identifier (discriminator)
    conflicts need no tracking here: the matcher only links names with
    *identical* discriminator tokens, and equality is transitive.
    """

    def __init__(self, common_types: dict[str, frozenset[str] | None]) -> None:
        self._parent = {name: name for name in common_types}
        self._common = dict(common_types)

    def find(self, name: str) -> str:
        root = name
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[name] != root:
            self._parent[name], name = root, self._parent[name]
        return root

    def union(self, a: str, b: str) -> bool:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return True
        common_a, common_b = self._common[root_a], self._common[root_b]
        if common_a is None or common_b is None:
            merged = common_a if common_b is None else common_b
        else:
            merged = common_a & common_b
            if not merged:
                return False
        # Deterministic root choice keeps grouping order-independent.
        root, child = sorted((root_a, root_b))
        self._parent[child] = root
        self._common[root] = merged
        return True


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

        # Best score per undirected name pair; the matcher already drops pairs
        # whose identifier tokens differ ("purchase order 1001" vs "... 1002").
        pair_scores: dict[tuple[str, str], float] = {}
        executor_class = (
            ProcessPoolExecutor if self.use_process_pool else ContextThreadPoolExecutor
        )

        # Send the matcher to each worker ONCE via the pool initializer, then
        # submit only the (tiny) entity name per task. This holds for both the
        # ProcessPoolExecutor (initializer runs once per process) and the
        # thread-pool path (initializer runs once per thread; the matcher
        # is shared in-process anyway). The thread pool accepts the same
        # initializer/initargs signature.
        with executor_class(
            max_workers=self.max_workers,
            initializer=_init_worker_fuzzy_matcher,
            initargs=(fuzzy_matcher,),
        ) as executor:
            future_to_name = {
                executor.submit(find_all_matches_for_entity_task, name): name
                for name in entity_names
            }
            total = len(entity_names)
            for done, future in enumerate(as_completed(future_to_name), start=1):
                original_name = future_to_name[future]
                if self.show_progress and done % _RESOLVE_PROGRESS_EVERY == 0:
                    logger.info("  ...resolved %s/%s entities", done, total)
                try:
                    for matched_name, score in future.result():
                        if matched_name == original_name:
                            continue
                        pair = (
                            (original_name, matched_name)
                            if original_name < matched_name
                            else (matched_name, original_name)
                        )
                        pair_scores[pair] = max(score, pair_scores.get(pair, 0.0))
                except Exception as e:
                    logger.warning(
                        "Failed to find matches for entity '%s': %s", original_name, e
                    )

        # Union strongest links first (ties broken by name for determinism),
        # rejecting any union that would put two type-incompatible members in
        # one group. Entities sharing an exact name always stay together.
        common_types: dict[str, frozenset[str] | None] = {}
        for name, members in entity_map.items():
            known = {t for t in (normalize_entity_type(e.type) for e in members) if t}
            common_types[name] = (
                None
                if not known
                else frozenset(known) if len(known) == 1 else frozenset()
            )
        union_find = _TypeAwareUnionFind(common_types)
        rejected = 0
        for (name_a, name_b), _score in sorted(
            pair_scores.items(), key=lambda item: (-item[1], item[0])
        ):
            if not union_find.union(name_a, name_b):
                rejected += 1
        if rejected:
            logger.info(
                "Skipped %s fuzzy entity links between incompatible types", rejected
            )

        names_by_root: dict[str, list[str]] = defaultdict(list)
        for name in entity_names:
            names_by_root[union_find.find(name)].append(name)
        groups = [
            [e for n in group_names for e in entity_map[n]]
            for group_names in names_by_root.values()
        ]

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
            # Keep the earliest known creation time; leave it None (truthfully
            # unknown) rather than fabricating wall-clock time when no member
            # carries one, so the merge stays a pure, reproducible function of
            # its inputs. updated_at is the one genuinely time-valued field.
            created_at=min(
                (e.created_at for e in entities if e.created_at),
                default=None,
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
        merged_text_unit_ids = self._merge_lists(
            [r.text_unit_ids for r in relationships if r.text_unit_ids]
        )
        # Weight tracks the count of distinct supporting text units (evidence
        # count). This is idempotent and order-independent, so the full-build and
        # incremental (merge_relationships) paths converge to the same weight; a
        # summed LLM "strength" would diverge and double-count on incremental
        # re-application. Falls back to the primary edge's own weight when no
        # text-unit lineage exists.
        merged_weight = (
            float(len(merged_text_unit_ids))
            if merged_text_unit_ids
            else (primary_rel.weight if primary_rel.weight is not None else 1.0)
        )
        return Relationship(
            id=primary_rel.id,
            short_id=primary_rel.short_id,
            source_id=primary_rel.source_id,
            source_name=primary_rel.source_name,
            target_id=primary_rel.target_id,
            target_name=primary_rel.target_name,
            type=primary_rel.type,
            weight=merged_weight,
            description=self._merge_descriptions(
                [r.description for r in relationships if r.description]
            ),
            description_embedding=primary_rel.description_embedding,
            text_unit_ids=merged_text_unit_ids,
            rank=max((r.rank for r in relationships if r.rank is not None), default=1),
            attributes=self._merge_attributes(
                [r.attributes for r in relationships if r.attributes]
            ),
            # See _merge_entities: leave created_at None when unknown so the
            # merge is a pure function of its inputs.
            created_at=min(
                (r.created_at for r in relationships if r.created_at),
                default=None,
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
