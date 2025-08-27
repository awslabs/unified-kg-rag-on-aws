import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.runnables import Runnable

from aws_graphrag.core import EvaluationException, get_logger
from aws_graphrag.models import (
    Config,
    EvaluationGroundTruth,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluationSummary,
    EvaluatorType,
    SearchStrategy,
)
from aws_graphrag.retrieval import RAGOutput
from aws_graphrag.utils import BatchProcessor

from .base import BaseEvaluator
from .langchain_evaluator import LangChainEvaluator
from .ragas_evaluator import RagasEvaluator

logger = get_logger(__name__)


class EvaluationManager:
    EVALUATOR_MAPPING = {
        EvaluatorType.LANGCHAIN: LangChainEvaluator,
        EvaluatorType.RAGAS: RagasEvaluator,
    }

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
            evaluator_class = self.EVALUATOR_MAPPING.get(evaluator_type)
            if not evaluator_class:
                logger.warning(f"Unknown evaluator type: '{evaluator_type}'")
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
                        f"Invalid configuration for '{evaluator_type.value}' evaluator"
                    )
            except Exception as e:
                logger.error(
                    f"Failed to initialize '{evaluator_type.value}' evaluator: {e}"
                )

        if enabled_count == 0:
            logger.warning("No evaluators were successfully initialized")
        else:
            logger.info(f"Initialized {enabled_count} evaluators")

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
                    logger.warning(f"Skipping invalid item at index {i}: '{item}'")
                    continue

                final_metadata = cleaned_base_metadata.copy()
                final_metadata.update(item.get("metadata", {}))
                if "search_strategy" not in final_metadata:
                    final_metadata["search_strategy"] = SearchStrategy.AUTO.value

                query_id = item.get("query_id", item.get("id", f"q_{i}"))
                query = EvaluationQuery(
                    query_id=query_id,
                    question=item["question"],
                    category=item.get("category"),
                    difficulty=item.get("difficulty"),
                    metadata=final_metadata,
                )
                queries.append(query)

                ground_truth_value = item.get("answer")
                if ground_truth_value:
                    gt = EvaluationGroundTruth(
                        query_id=query_id,
                        ground_truth=str(ground_truth_value),
                        reference_sources=item.get("reference_sources", []),
                        expected_entities=item.get("expected_entities", []),
                        expected_relationships=item.get("expected_relationships", []),
                    )
                    ground_truths.append(gt)

            logger.info(
                f"Loaded {len(queries)} queries and {len(ground_truths)} ground truths from '{eval_data_path}'."
            )
            return queries, ground_truths

        except FileNotFoundError:
            logger.error(f"Evaluation data file not found: '{eval_data_path}'")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from '{eval_data_path}': {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load data from '{eval_data_path}': {e}")
            raise

    async def evaluate_dataset(
        self,
        queries: list[EvaluationQuery],
        ground_truths: list[EvaluationGroundTruth],
        show_progress: bool = True,
    ) -> tuple[list[EvaluationResult], list[EvaluationReport], EvaluationSummary]:
        start_time = datetime.now()
        logger.info(f"Starting evaluation for {len(queries)} queries")

        try:
            results = await self._generate_answers(queries, show_progress)
            reports = await self._evaluate_results(queries, results, ground_truths)
            end_time = datetime.now()
            summary = self._generate_summary(
                queries, results, reports, start_time, end_time
            )

            logger.info(
                f"Evaluation completed: {summary.successful_evaluations}/{summary.total_queries} queries processed"
            )
            return results, reports, summary
        except Exception as e:
            logger.error(f"Dataset evaluation failed: {e}")
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
                    f"Failed to process result for query '{query.query_id}': {e}"
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

        for result in results:
            result.ground_truth = ground_truth_map.get(result.query_id, "")

        gt_list = [res.ground_truth for res in results]

        for evaluator_type, evaluator in self.evaluators.items():
            try:
                reports = await evaluator.aevaluate_batch(queries, results, gt_list)
                all_reports.extend(reports)
            except Exception as e:
                logger.error(f"Failed to run '{evaluator_type.value}' evaluation: {e}")

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

        statistics_dict = {}
        for metric_type, values in metric_values.items():
            if not values:
                continue
            statistics_dict[metric_type] = {
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
        logger.info(f"Evaluation results saved to '{outputs_dir}'")

    @staticmethod
    def _save_json(path: Path, data: Any) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
