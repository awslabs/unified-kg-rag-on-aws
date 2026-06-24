# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, NamedTuple

from aws_graphrag.adapters.ingestion.description_summarizer import DescriptionSummarizer
from aws_graphrag.adapters.storage.neptune_indexer import NeptuneIndexer
from aws_graphrag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from aws_graphrag.domain.ingestion.merge import merge_entities, merge_relationships
from aws_graphrag.domain.models import (
    Claim,
    Community,
    CommunityReport,
    Config,
    Entity,
    Relationship,
    TextUnit,
)
from aws_graphrag.ports.indexer import BaseIndexer, IndexingStats
from aws_graphrag.shared import get_logger

logger = get_logger(__name__)


class IndexingTask(NamedTuple):
    fn: Callable
    args: list
    key: str


class IndexingManager:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.opensearch_indexer = OpenSearchIndexer(config=config)
        self.neptune_indexer = NeptuneIndexer(config=config)
        # Built lazily on first cross-run merge so a manager only used for
        # indexing (no incremental read-back) never constructs a Bedrock client.
        self._description_summarizer: DescriptionSummarizer | None = None

    @property
    def description_summarizer(self) -> DescriptionSummarizer:
        if self._description_summarizer is None:
            self._description_summarizer = DescriptionSummarizer(self.config)
        return self._description_summarizer

    def clear_all_data(self, text_units: list[TextUnit]) -> bool:
        suffixes = self._discover_suffixes(text_units)
        if not suffixes:
            logger.warning("No suffixes found to clear data")
            return True

        logger.info("Clearing data for suffixes: '%s'", suffixes)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                os_future = executor.submit(self.opensearch_indexer.clear, suffixes)
                neptune_future = executor.submit(self.neptune_indexer.clear, suffixes)
                opensearch_success = os_future.result()
                neptune_success = neptune_future.result()

            success = opensearch_success and neptune_success
            if not success:
                logger.error(
                    "Clear operation failed - OpenSearch: %s, Neptune: %s",
                    opensearch_success,
                    neptune_success,
                )
            return success
        except Exception as e:
            logger.error("Clear operation failed: %s", e)
            return False

    @staticmethod
    def _discover_suffixes(items: list[Any] | None) -> list[str]:
        if not items:
            return []
        return list({BaseIndexer.get_suffix(item) for item in items})

    def initialize(self) -> bool:
        try:
            opensearch_ok = self.opensearch_indexer.initialize()
            neptune_ok = self.neptune_indexer.initialize()

            if not opensearch_ok or not neptune_ok:
                logger.error("Failed to initialize indexers")
                return False

            return True
        except Exception as e:
            logger.error("Indexer initialization failed: %s", e)
            return False

    def index_all_data(
        self,
        text_units: list[TextUnit] | None = None,
        entities: list[Entity] | None = None,
        relationships: list[Relationship] | None = None,
        communities: list[Community] | None = None,
        community_reports: list[CommunityReport] | None = None,
        claims: list[Claim] | None = None,
    ) -> dict[str, IndexingStats]:
        start_time = time.time()
        results: dict[str, IndexingStats] = {}

        self._enrich_text_units(text_units, communities)

        phase1_tasks = [
            IndexingTask(
                self.opensearch_indexer.index_text_units,
                [text_units],
                "opensearch_text_units",
            ),
            IndexingTask(
                self.opensearch_indexer.index_entities,
                [entities],
                "opensearch_entities",
            ),
            IndexingTask(
                self.opensearch_indexer.index_community_reports,
                [community_reports],
                "opensearch_community_reports",
            ),
            IndexingTask(
                self.opensearch_indexer.index_relationships,
                [relationships],
                "opensearch_relationships",
            ),
            IndexingTask(
                self.opensearch_indexer.index_claims,
                [claims],
                "opensearch_claims",
            ),
            IndexingTask(
                self.neptune_indexer.index_entities, [entities], "neptune_entities"
            ),
        ]

        logger.info("--- Starting Indexing Phase 1 ---")
        results.update(self._run_indexing_phase(phase1_tasks))

        phase2_tasks = [
            IndexingTask(
                self.neptune_indexer.index_relationships,
                [relationships],
                "neptune_relationships",
            ),
            IndexingTask(
                self.neptune_indexer.index_communities,
                [communities],
                "neptune_communities",
            ),
        ]

        logger.info("--- Starting Indexing Phase 2 ---")
        results.update(self._run_indexing_phase(phase2_tasks))

        elapsed_time = time.time() - start_time
        self._log_completion_summary(results, elapsed_time)
        return results

    def index_delta(
        self,
        text_units: list[TextUnit] | None = None,
        entities: list[Entity] | None = None,
        relationships: list[Relationship] | None = None,
        communities: list[Community] | None = None,
        community_reports: list[CommunityReport] | None = None,
        claims: list[Claim] | None = None,
    ) -> dict[str, IndexingStats]:
        """Idempotently upsert a delta set into the live stores (incremental run).

        Routes to the indexers' ``upsert_*`` methods instead of the full
        rebuild path, so only changed/new artifacts are written and existing
        data is preserved. Entities (graph + vector) are upserted before
        relationships, which depend on entity vertices existing.
        """
        start_time = time.time()
        results: dict[str, IndexingStats] = {}

        if self.config.indexing.cross_run_merge:
            entities, relationships = self._merge_with_existing_graph(
                entities, relationships
            )

        self._enrich_text_units(text_units, communities)

        phase1_tasks = [
            IndexingTask(
                self.opensearch_indexer.upsert_text_units,
                [text_units],
                "opensearch_text_units",
            ),
            IndexingTask(
                self.opensearch_indexer.upsert_entities,
                [entities],
                "opensearch_entities",
            ),
            IndexingTask(
                self.opensearch_indexer.index_community_reports,
                [community_reports],
                "opensearch_community_reports",
            ),
            IndexingTask(
                self.opensearch_indexer.upsert_relationships,
                [relationships],
                "opensearch_relationships",
            ),
            IndexingTask(
                self.opensearch_indexer.upsert_claims,
                [claims],
                "opensearch_claims",
            ),
            IndexingTask(
                self.neptune_indexer.upsert_entities, [entities], "neptune_entities"
            ),
        ]
        logger.info("--- Starting Delta Indexing Phase 1 (upsert) ---")
        results.update(self._run_indexing_phase(phase1_tasks))

        phase2_tasks = [
            IndexingTask(
                self.neptune_indexer.upsert_relationships,
                [relationships],
                "neptune_relationships",
            ),
            IndexingTask(
                self.neptune_indexer.upsert_communities,
                [communities],
                "neptune_communities",
            ),
        ]
        logger.info("--- Starting Delta Indexing Phase 2 (upsert) ---")
        results.update(self._run_indexing_phase(phase2_tasks))

        elapsed_time = time.time() - start_time
        self._log_completion_summary(results, elapsed_time)
        return results

    def _merge_with_existing_graph(
        self,
        entities: list[Entity] | None,
        relationships: list[Relationship] | None,
    ) -> tuple[list[Entity] | None, list[Relationship] | None]:
        """Union delta artifacts with existing graph state before upsert.

        Reads the existing entities/relationships the delta touches back from the
        graph store and merges (description/text_unit_ids union, frequency/weight
        recompute) via the pure merge functions, so a cross-run upsert accumulates
        rather than overwriting. If the adapter does not support read-back it
        returns ``[]`` and this degenerates to the existing overwrite behaviour.
        """
        merged_entities = entities
        if entities:
            existing = self.neptune_indexer.read_entities([e.id for e in entities])
            if existing:
                merged_entities, _ = merge_entities(existing, entities)
                # The cross-run merge concatenates descriptions, so an entity
                # seen across many runs can grow unbounded — re-summarize the
                # over-threshold ones (no-op below the threshold / when disabled).
                merged_entities = self.description_summarizer.summarize_entities(
                    merged_entities or []
                )
        merged_relationships = relationships
        if relationships:
            existing_rels = self.neptune_indexer.read_relationships(
                [r.id for r in relationships]
            )
            if existing_rels:
                merged_relationships = merge_relationships(existing_rels, relationships)
                merged_relationships = (
                    self.description_summarizer.summarize_relationships(
                        merged_relationships or []
                    )
                )
        return merged_entities, merged_relationships

    def delete_documents(
        self, ids_by_suffix: dict[str, list[str]]
    ) -> dict[str, IndexingStats]:
        """Delete artifacts for removed documents from both stores by id.

        ``ids_by_suffix`` maps a tenant/version suffix to the entity/relationship/
        text-unit/community ids that belonged only to deleted documents.
        """
        results: dict[str, IndexingStats] = {}
        for suffix, ids in ids_by_suffix.items():
            if not ids:
                continue
            results[f"neptune_delete_{suffix}"] = self.neptune_indexer.delete_by_id(
                ids, suffix=suffix
            )
            for prefix in (
                self.opensearch_indexer.opensearch_config.text_units_index_prefix,
                self.opensearch_indexer.opensearch_config.entities_index_prefix,
                self.opensearch_indexer.opensearch_config.relationships_index_prefix,
                self.opensearch_indexer.opensearch_config.claims_index_prefix,
                self.opensearch_indexer.opensearch_config.community_reports_index_prefix,
            ):
                key = f"opensearch_delete_{prefix}_{suffix}"
                results[key] = self.opensearch_indexer.delete_by_id(ids, prefix, suffix)
        return results

    def _run_indexing_phase(
        self, tasks: list[IndexingTask]
    ) -> dict[str, IndexingStats]:
        phase_results: dict[str, IndexingStats] = {}
        valid_tasks = [task for task in tasks if task.args and task.args[0]]

        if not valid_tasks:
            logger.info("No tasks to run in this phase.")
            return phase_results

        total_items = sum(len(task.args[0]) for task in valid_tasks)
        logger.info(
            "Executing %s tasks with %s total items...", len(valid_tasks), total_items
        )

        # Indexing tasks are IO-bound (OpenSearch/Neptune network writes), so size
        # the pool by the number of independent tasks rather than CPU count — a
        # cpu*0.8 cap (~1-2 on a 2-vCPU Fargate task) would needlessly serialize
        # the independent per-backend writes. Cap at 8 as a safety bound.
        pool_size = min(len(valid_tasks), 8)
        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            futures_map = {
                executor.submit(task.fn, *task.args): task.key for task in valid_tasks
            }

            for future in as_completed(futures_map):
                task_name = futures_map[future]
                try:
                    phase_results[task_name] = future.result()
                except Exception as e:
                    logger.error("Task '%s' failed: %s", task_name, e)
                    stats = IndexingStats()
                    stats.add_error(str(e))
                    phase_results[task_name] = stats

        return phase_results

    @staticmethod
    def _enrich_text_units(
        text_units: list[TextUnit] | None, communities: list[Community] | None
    ) -> None:
        if not text_units or not communities:
            return

        text_unit_map = {tu.id: tu for tu in text_units}
        for community in communities:
            if not community.text_unit_ids:
                continue
            for text_unit_id in community.text_unit_ids:
                if text_unit := text_unit_map.get(text_unit_id):
                    if text_unit.community_ids is None:
                        text_unit.community_ids = []
                    if community.id not in text_unit.community_ids:
                        text_unit.community_ids.append(community.id)

    @staticmethod
    def _log_completion_summary(
        results: dict[str, IndexingStats], elapsed_time: float
    ) -> None:
        total_items = total_successful = total_failed = 0

        for stats in results.values():
            if stats:
                total_items += stats.total_items
                total_successful += stats.successful_items
                total_failed += stats.failed_items

        success_rate = (total_successful / total_items * 100) if total_items > 0 else 0
        logger.info(
            "Indexing completed in %.2fs: %s/%s items (%.1f%%)",
            elapsed_time,
            total_successful,
            total_items,
            success_rate,
        )

        if total_failed > 0:
            logger.warning("Failed items: %s", total_failed)
            for data_type, stats in results.items():
                if stats and stats.errors:
                    logger.warning("%s errors: %s", data_type, stats.errors[:2])

        for task_name, stats in results.items():
            if stats and stats.total_items > 0:
                failure_rate = stats.failed_items / stats.total_items
                if failure_rate > 0.5:
                    logger.warning(
                        "High failure rate for '%s': %s/%s (%.1f%%)",
                        task_name,
                        stats.failed_items,
                        stats.total_items,
                        failure_rate * 100,
                    )
