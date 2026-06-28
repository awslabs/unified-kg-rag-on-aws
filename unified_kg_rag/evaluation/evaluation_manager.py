# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import statistics
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.runnables import Runnable

from unified_kg_rag.application.retrieval.rag_chain import RAGOutput
from unified_kg_rag.domain.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
    EvaluatorType,
)
from unified_kg_rag.shared import EvaluationException, get_logger
from unified_kg_rag.shared.utils import BatchProcessor

from .base import BaseEvaluator
from .graph_aware_evaluator import GraphAwareEvaluator

logger = get_logger(__name__)


class EvaluationManager:
    @staticmethod
    def _resolve_evaluator_class(
        evaluator_type: EvaluatorType,
    ) -> Callable[..., BaseEvaluator] | None:
        """Resolve an evaluator class lazily (registry, but import-on-use).

        The langchain/ragas adapter evaluators import `evaluation.base`, which
        runs this package's __init__; importing them at module load here would
        create a circular import (manager -> adapter -> evaluation.base ->
        __init__ -> manager). Resolving inside the method defers the import to
        instantiation time, when the package is fully initialized — while
        keeping the declarative type->class registry.
        """
        if evaluator_type is EvaluatorType.LANGCHAIN:
            from unified_kg_rag.adapters.evaluators.langchain_evaluator import (
                LangChainEvaluator,
            )

            return LangChainEvaluator
        if evaluator_type is EvaluatorType.RAGAS:
            from unified_kg_rag.adapters.evaluators.ragas_evaluator import (
                RagasEvaluator,
            )

            return RagasEvaluator
        if evaluator_type is EvaluatorType.GRAPH_AWARE:
            return GraphAwareEvaluator
        # Defensive: a future EvaluatorType with no mapping resolves to None and
        # is skipped by the caller. mypy sees the enum as exhaustive today, hence
        # the ignore — the branch is real once a new member is added.
        return None  # type: ignore[unreachable]

    def __init__(self, config: Config, rag_chain: Runnable | None = None) -> None:
        self.config = config
        if rag_chain is None:
            raise EvaluationException("RAG chain not provided for evaluation.")
        self.rag_chain = rag_chain
        self.evaluators: dict[EvaluatorType, BaseEvaluator] = {}
        self._initialize_evaluators()
        self.batch_processor = BatchProcessor()

    def _initialize_evaluators(self) -> None:
        enabled_count = 0
        for evaluator_type in self.config.evaluation.enabled_evaluators:
            evaluator_class = self._resolve_evaluator_class(evaluator_type)
            if not evaluator_class:
                logger.warning("Unknown evaluator type: '%s'", evaluator_type)
                continue

            try:
                evaluator = evaluator_class(
                    config=self.config, rag_chain=self.rag_chain
                )
                if evaluator.validate_config():
                    self.evaluators[evaluator_type] = evaluator
                    enabled_count += 1
                else:
                    logger.error(
                        "Invalid configuration for '%s' evaluator", evaluator_type.value
                    )
            except Exception as e:
                logger.error(
                    "Failed to initialize '%s' evaluator: %s", evaluator_type.value, e
                )

        if enabled_count == 0:
            logger.warning("No evaluators were successfully initialized")
        else:
            logger.info("Initialized %s evaluators", enabled_count)

    @staticmethod
    def load_data(
        eval_data_path: str | Path, base_metadata: dict[str, Any] | None = None
    ) -> tuple[list[EvaluationQuery], list[EvaluationGroundTruth]]:
        if not eval_data_path:
            raise ValueError("Evaluation data path is required.")

        if base_metadata is None:
            base_metadata = {}

        try:
            with open(eval_data_path, encoding="utf-8") as f:
                data = json.load(f)

            queries = []
            ground_truths = []

            cleaned_base_metadata = {
                k: v for k, v in base_metadata.items() if v is not None
            }

            for i, item in enumerate(data):
                if not isinstance(item, dict) or "question" not in item:
                    logger.warning("Skipping invalid item at index %s: '%s'", i, item)
                    continue

                final_metadata = cleaned_base_metadata.copy()
                final_metadata.update(item.get("metadata", {}))

                query_id = str(item.get("query_id", item.get("id", f"q_{i}")))
                query = EvaluationQuery(
                    query_id=query_id,
                    question=item["question"],
                    category=item.get("category"),
                    difficulty=item.get("difficulty"),
                    metadata=final_metadata,
                )
                queries.append(query)

                # Build a ground truth when ANY ground-truth signal is present —
                # not only a textual answer — so graph-aware evaluation works on
                # datasets that supply only expected_entities/relationships.
                answer = item.get("answer")
                expected_entities = item.get("expected_entities", [])
                expected_relationships = item.get("expected_relationships", [])
                reference_sources = item.get("reference_sources", [])
                if answer or expected_entities or expected_relationships:
                    gt = EvaluationGroundTruth(
                        query_id=query_id,
                        ground_truth=str(answer) if answer else "",
                        reference_sources=reference_sources,
                        expected_entities=expected_entities,
                        expected_relationships=expected_relationships,
                    )
                    ground_truths.append(gt)

            logger.info(
                "Loaded %s queries and %s ground truths from '%s'.",
                len(queries),
                len(ground_truths),
                eval_data_path,
            )
            return queries, ground_truths

        except FileNotFoundError:
            logger.error("Evaluation data file not found: '%s'", eval_data_path)
            raise
        except json.JSONDecodeError as e:
            logger.error("Error decoding JSON from '%s': %s", eval_data_path, e)
            raise
        except Exception as e:
            logger.error("Failed to load data from '%s': %s", eval_data_path, e)
            raise

    async def evaluate_dataset(
        self,
        queries: list[EvaluationQuery],
        ground_truths: list[EvaluationGroundTruth],
        show_progress: bool = True,
    ) -> tuple[list[EvaluationResult], list[EvaluationReport], EvaluationSummary]:
        start_time = datetime.now()
        logger.info("Starting evaluation for %s queries", len(queries))

        try:
            results = await self._generate_answers(queries, show_progress)
            reports = await self._evaluate_results(queries, results, ground_truths)
            end_time = datetime.now()
            summary = self._generate_summary(
                queries, results, reports, start_time, end_time
            )

            logger.info(
                "Evaluation completed: %s/%s queries processed",
                summary.successful_evaluations,
                summary.total_queries,
            )
            return results, reports, summary
        except Exception as e:
            logger.error("Dataset evaluation failed: %s", e)
            raise EvaluationException(f"Dataset evaluation failed: {e}") from e

    async def _generate_answers(
        self, queries: list[EvaluationQuery], show_progress: bool
    ) -> list[EvaluationResult]:
        def prepare_inputs(query_batch: list[EvaluationQuery]) -> list[dict[str, Any]]:
            return [{"query": q.question, **q.metadata} for q in query_batch]

        raw_results = await self.batch_processor.aexecute_with_fallback(
            items_to_process=queries,
            prepare_inputs_func=prepare_inputs,
            batch_func=self.rag_chain.abatch,
            sequential_func=self.rag_chain.ainvoke,
            task_name="Answer Generation",
            show_progress=show_progress,
        )

        results = []
        for query, raw_result in zip(queries, raw_results, strict=True):
            try:
                rag_metadata = self._extract_from_result(raw_result, "metadata", {})
                results.append(
                    EvaluationResult(
                        query_id=query.query_id,
                        question=query.question,
                        generated_answer=self._extract_from_result(
                            raw_result, "answer", ""
                        ),
                        ground_truth="",
                        retrieved_contexts=self._extract_from_result(
                            raw_result, "sources", []
                        ),
                        enable_thinking=rag_metadata.get("enable_thinking", False),
                        search_strategy=rag_metadata.get("search_strategy"),
                        response_time=rag_metadata.get("processing_time"),
                        search_type=query.metadata.get("search_type"),
                        top_k=query.metadata.get("top_k"),
                        retrieval_multiplier=query.metadata.get("retrieval_multiplier"),
                        metadata=query.metadata,
                    )
                )
            except Exception as e:
                logger.error(
                    "Failed to process result for query '%s': %s", query.query_id, e
                )
                results.append(
                    EvaluationResult(
                        query_id=query.query_id,
                        question=query.question,
                        generated_answer="",
                        ground_truth="",
                        metadata={"error": str(e)},
                    )
                )
        return results

    def create_lean_context_strings(
        self, sources_list: list[dict[str, Any]]
    ) -> list[str]:
        lean_contexts = []
        translated_key = (
            f"translated_text_{self.config.processing.translation.target_language}"
        )
        desired_fields = [
            "description",
            "full_content",
            "name",
            "summary",
            translated_key,
        ]

        for item in sources_list:
            payloads_to_search = self._get_payloads_to_search(item)
            has_translated_text = self._has_translated_text(
                payloads_to_search, translated_key
            )
            lean_item = self._extract_fields(
                payloads_to_search, desired_fields, has_translated_text
            )

            if lean_item:
                lean_contexts.append(str(lean_item))
            else:
                lean_contexts.append(self._create_minimal_info(item))

        return lean_contexts

    @staticmethod
    def _get_payloads_to_search(item: dict[str, Any]) -> list[dict[str, Any]]:
        payloads = []

        if isinstance(item.get("metadata", {}).get("attributes"), dict):
            payloads.append(item["metadata"]["attributes"])

        if isinstance(item.get("metadata"), dict):
            payloads.append(item["metadata"])

        payloads.append(item)
        return payloads

    @staticmethod
    def _has_translated_text(
        payloads: list[dict[str, Any]], translated_key: str
    ) -> bool:
        return any(
            translated_key in payload and payload[translated_key]
            for payload in payloads
        )

    @staticmethod
    def _extract_fields(
        payloads: list[dict[str, Any]],
        desired_fields: list[str],
        has_translated_text: bool,
    ) -> dict[str, Any]:
        lean_item = {}

        for field in desired_fields:
            if field in lean_item or (field == "text" and has_translated_text):
                continue

            for payload in payloads:
                if field in payload and payload[field]:
                    lean_item[field] = payload[field]
                    break

        return lean_item

    @staticmethod
    def _create_minimal_info(item: dict[str, Any]) -> str:
        minimal_info = {"source": item.get("source"), "score": item.get("score")}
        return str({k: v for k, v in minimal_info.items() if v is not None})

    def _extract_from_result(
        self, raw_result: Any, key: str, default: Any = None
    ) -> Any:
        if isinstance(raw_result, RAGOutput):
            if key == "sources" and hasattr(raw_result, key):
                sources_list = getattr(raw_result, key, [])
                return self.create_lean_context_strings(sources_list)
            return getattr(raw_result, key, default)

        if isinstance(raw_result, dict):
            value = raw_result.get(key, default)
            if key == "sources" and isinstance(value, list):
                return self.create_lean_context_strings(value)
            return value

        return default if key != "answer" else str(raw_result)

    async def _evaluate_results(
        self,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
        ground_truths: list[EvaluationGroundTruth],
    ) -> list[EvaluationReport]:
        all_reports = []
        ground_truth_map = {gt.query_id: gt.ground_truth for gt in ground_truths}
        gt_obj_map = {gt.query_id: gt for gt in ground_truths}

        for result in results:
            result.ground_truth = ground_truth_map.get(result.query_id, "")
            # Thread graph-aware expectations onto the result so the
            # GraphAwareEvaluator can score entity/relationship coverage without
            # changing the evaluator signature.
            if gt := gt_obj_map.get(result.query_id):
                # Copy so a downstream in-place mutation of result.metadata does
                # not corrupt the shared ground-truth lists.
                result.metadata["expected_entities"] = list(gt.expected_entities)
                result.metadata["expected_relationships"] = list(
                    gt.expected_relationships
                )

        gt_list = [res.ground_truth for res in results]

        # Graph-aware coverage is the project's headline differentiator over
        # text-similarity eval, but it silently emits nothing when the dataset
        # carries no expected_entities/relationships. Warn ONCE so a user does
        # not believe coverage was measured when it was not (the per-query path
        # just skips, producing zero metrics with no signal).
        if EvaluatorType.GRAPH_AWARE in self.evaluators and not any(
            gt.expected_entities or gt.expected_relationships for gt in ground_truths
        ):
            logger.warning(
                "graph_aware evaluator is enabled but no dataset row supplies "
                "expected_entities/expected_relationships — no coverage metric "
                "will be reported. Add them to the evaluation dataset to measure "
                "entity/relationship recall."
            )

        for evaluator_type, evaluator in self.evaluators.items():
            try:
                reports = await evaluator.aevaluate_batch(queries, results, gt_list)
                all_reports.extend(reports)
            except Exception as e:
                logger.error(
                    "Failed to run '%s' evaluation: %s", evaluator_type.value, e
                )

        return all_reports

    def _generate_summary(
        self,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
        reports: list[EvaluationReport],
        start_time: datetime,
        end_time: datetime,
    ) -> EvaluationSummary:
        successful_evaluations = sum(1 for r in results if r.generated_answer)
        response_times = [r.response_time for r in results if r.response_time]
        avg_response_time = (
            sum(response_times) / len(response_times) if response_times else 0.0
        )

        return EvaluationSummary(
            total_queries=len(queries),
            successful_evaluations=successful_evaluations,
            failed_evaluations=len(queries) - successful_evaluations,
            average_response_time=avg_response_time,
            metric_statistics=self._calculate_metric_statistics(reports),
            evaluation_start_time=start_time,
            evaluation_end_time=end_time,
            configuration=self.config.evaluation.model_dump(),
        )

    @staticmethod
    def _calculate_metric_statistics(
        reports: list[EvaluationReport],
    ) -> dict[str, dict[str, float]]:
        metric_values = defaultdict(list)
        for report in reports:
            for metric in report.metrics:
                metric_values[metric.metric_type].append(metric.value)

        statistics_dict: dict[str, dict[str, float]] = {}
        for metric_type, values in metric_values.items():
            if not values:
                continue
            statistics_dict[metric_type.value] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
                "min": min(values),
                "max": max(values),
                "count": len(values),
            }
        return statistics_dict

    def save_results(
        self,
        results: list[EvaluationResult],
        reports: list[EvaluationReport],
        summary: EvaluationSummary,
        outputs_dir: str | Path,
    ) -> None:
        if isinstance(outputs_dir, str):
            outputs_dir = Path(outputs_dir)
        outputs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.config.evaluation.save_detailed_results:
            self._save_json(
                outputs_dir / f"evaluation_results_{timestamp}.json",
                [r.model_dump() for r in results],
            )
            self._save_json(
                outputs_dir / f"evaluation_reports_{timestamp}.json",
                [r.model_dump() for r in reports],
            )

        self._save_json(
            outputs_dir / f"evaluation_summary_{timestamp}.json", summary.model_dump()
        )
        logger.info("Evaluation results saved to '%s'", outputs_dir)

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
