# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from abc import ABC, abstractmethod
from collections.abc import Coroutine
from datetime import datetime
from typing import Any

from tqdm import tqdm

from aws_graphrag.domain.models import (
    Config,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
)
from aws_graphrag.shared import get_logger

logger = get_logger(__name__)


class BaseEvaluator(ABC):
    def __init__(
        self,
        config: Config,
        evaluator_type: EvaluatorType,
        show_progress: bool = True,
        **kwargs: Any,
    ) -> None:
        self.config = config
        self.evaluator_type = evaluator_type
        self.show_progress = show_progress
        self._initialize_evaluator(**kwargs)

    @abstractmethod
    def _initialize_evaluator(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def evaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        pass

    @abstractmethod
    async def aevaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        pass

    async def aevaluate_batch(
        self,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
        ground_truths: list[str],
        **kwargs: Any,
    ) -> list[EvaluationReport]:
        semaphore = asyncio.Semaphore(self.config.processing.max_concurrency)

        async def _run_with_semaphore(
            coro: Coroutine, index: int
        ) -> tuple[int, EvaluationReport | None, Exception | None]:
            async with semaphore:
                try:
                    result = await coro
                    return index, result, None
                except Exception as e:
                    return index, None, e

        tasks = []
        for i, (query, result, ground_truth) in enumerate(
            zip(queries, results, ground_truths, strict=True)
        ):
            coro = self.aevaluate_single(query, result, ground_truth, **kwargs)
            task = asyncio.create_task(_run_with_semaphore(coro, i))
            tasks.append(task)

        final_reports: list[EvaluationReport | None] = [None] * len(queries)
        progress_bar = tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Async Evaluating with '{self.evaluator_type.value}'",
            disable=not self.show_progress,
        )

        for future in progress_bar:
            original_index, report, error = await future
            query_id = queries[original_index].query_id

            if error:
                logger.error("Failed to evaluate query '%s': %s", query_id, error)
                final_reports[original_index] = self._create_empty_report(query_id)
            else:
                final_reports[original_index] = report

        return [report for report in final_reports if report is not None]

    def _create_empty_report(self, query_id: str) -> EvaluationReport:
        return EvaluationReport(
            query_id=query_id,
            evaluator_type=self.evaluator_type,
            metrics=[],
            overall_score=0.0,
            evaluation_time=datetime.now(),
            metadata={"evaluation_failed": True},
        )

    def validate_config(self) -> bool:
        return True


class BaseGraphRAGEvaluator(BaseEvaluator):
    def __init__(
        self,
        config: Config,
        evaluator_type: EvaluatorType,
        rag_chain: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.rag_chain = rag_chain
        super().__init__(config, evaluator_type, **kwargs)

    @abstractmethod
    def _initialize_evaluator(self, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def evaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        pass

    @staticmethod
    def _extract_search_metadata(result: EvaluationResult) -> dict[str, Any]:
        metadata: dict[str, Any] = {}

        if result.search_strategy:
            metadata["search_strategy"] = result.search_strategy
        if result.search_type:
            metadata["search_type"] = result.search_type
        if result.top_k:
            metadata["top_k"] = str(result.top_k)
        if result.retrieval_multiplier:
            metadata["retrieval_multiplier"] = str(result.retrieval_multiplier)
        if result.response_time:
            metadata["response_time"] = str(result.response_time)

        if result.retrieved_contexts:
            num_contexts = len(result.retrieved_contexts)
            metadata["num_contexts"] = int(num_contexts)
            if num_contexts > 0:
                avg_length = (
                    sum(len(ctx) for ctx in result.retrieved_contexts) / num_contexts
                )
                metadata["avg_context_length"] = float(avg_length)

        return metadata
