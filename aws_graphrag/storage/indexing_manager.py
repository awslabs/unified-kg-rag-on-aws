# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import cpu_count
from typing import Any, NamedTuple

from aws_graphrag.adapters.storage.neptune_indexer import NeptuneIndexer
from aws_graphrag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from aws_graphrag.core import get_logger
from aws_graphrag.models import (
    Claim,
    Community,
    CommunityReport,
    Config,
    Entity,
    Relationship,
    TextUnit,
)
from aws_graphrag.ports.indexer import BaseIndexer, IndexingStats

logger = get_logger(__name__)


class IndexingTask(NamedTuple):
    fn: Callable
    args: list
    key: str


class IndexingManager:
    def __init__(self, config: Config, max_workers: int | None = None) -> None:
        self.opensearch_indexer = OpenSearchIndexer(config=config)
        self.neptune_indexer = NeptuneIndexer(config=config)
        self.max_workers = max_workers or max(1, int(cpu_count() * 0.8))

    def clear_all_data(self, text_units: list[TextUnit]) -> bool:
        suffixes = self._discover_suffixes(text_units)
        if not suffixes:
            logger.warning("No suffixes found to clear data")
            return True

        logger.info(f"Clearing data for suffixes: '{suffixes}'")
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                os_future = executor.submit(self.opensearch_indexer.clear, suffixes)
                neptune_future = executor.submit(self.neptune_indexer.clear, suffixes)
                opensearch_success = os_future.result()
                neptune_success = neptune_future.result()

            success = opensearch_success and neptune_success
            if not success:
                logger.error(
                    f"Clear operation failed - OpenSearch: {opensearch_success}, Neptune: {neptune_success}"
                )
            return success
        except Exception as e:
            logger.error(f"Clear operation failed: {e}")
            return False

    @staticmethod
    def _discover_suffixes(items: list[Any] | None) -> list[str]:
        if not items:
            return []
        return list({BaseIndexer.get_suffix(item) for item in items})

    def get_comprehensive_stats(self) -> dict[str, Any]:
        try:
            return {
                "opensearch": self.opensearch_indexer.get_stats(),
                "neptune": self.neptune_indexer.get_stats(),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Failed to retrieve stats: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def initialize(self) -> bool:
        try:
            opensearch_ok = self.opensearch_indexer.initialize()
            neptune_ok = self.neptune_indexer.initialize()

            if not opensearch_ok or not neptune_ok:
                logger.error("Failed to initialize indexers")
                return False

            return True
        except Exception as e:
            logger.error(f"Indexer initialization failed: {e}")
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
                self.neptune_indexer.index_communities,
                [communities],
                "neptune_communities",
            ),
        ]
        logger.info("--- Starting Delta Indexing Phase 2 (upsert) ---")
        results.update(self._run_indexing_phase(phase2_tasks))

        elapsed_time = time.time() - start_time
        self._log_completion_summary(results, elapsed_time)
        return results

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
            results[f"neptune_delete_{suffix}"] = self.neptune_indexer.delete_by_id(ids)
            for prefix in (
                self.opensearch_indexer.opensearch_config.text_units_index_prefix,
                self.opensearch_indexer.opensearch_config.entities_index_prefix,
                self.opensearch_indexer.opensearch_config.relationships_index_prefix,
                self.opensearch_indexer.opensearch_config.claims_index_prefix,
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
            f"Executing {len(valid_tasks)} tasks with {total_items} total items..."
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures_map = {
                executor.submit(task.fn, *task.args): task.key for task in valid_tasks
            }

            for future in as_completed(futures_map):
                task_name = futures_map[future]
                try:
                    phase_results[task_name] = future.result()
                except Exception as e:
                    logger.error(f"Task '{task_name}' failed: {e}")
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
            f"Indexing completed in {elapsed_time:.2f}s: {total_successful}/{total_items} items ({success_rate:.1f}%)"
        )

        if total_failed > 0:
            logger.warning(f"Failed items: {total_failed}")
            for data_type, stats in results.items():
                if stats and stats.errors:
                    logger.warning(f"{data_type} errors: {stats.errors[:2]}")

        for task_name, stats in results.items():
            if stats and stats.total_items > 0:
                failure_rate = stats.failed_items / stats.total_items
                if failure_rate > 0.5:
                    logger.warning(
                        f"High failure rate for '{task_name}': "
                        f"{stats.failed_items}/{stats.total_items} ({failure_rate:.1%})"
                    )

    def validate_indexing_integrity(self, text_units: list[TextUnit]) -> dict[str, Any]:
        suffixes = self._discover_suffixes(text_units)
        if not suffixes:
            return {
                "error": "No suffixes to validate",
                "timestamp": datetime.now().isoformat(),
            }

        try:
            os_entity_count = self.opensearch_indexer.get_entity_count(suffixes)
            neptune_entity_count = self.neptune_indexer.get_entity_count(suffixes)
            count_match = os_entity_count == neptune_entity_count
            count_diff = abs(os_entity_count - neptune_entity_count)

            if not count_match:
                logger.warning(
                    f"Entity count mismatch: OpenSearch({os_entity_count}) vs Neptune({neptune_entity_count})"
                )

            return {
                "consistency_checks": {
                    "entity_count_match": count_match,
                    "opensearch_entity_count": os_entity_count,
                    "neptune_entity_count": neptune_entity_count,
                    "entity_count_difference": count_diff,
                },
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Integrity validation failed: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
