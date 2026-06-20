# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
import asyncio
from datetime import datetime
from typing import Any, ClassVar

import boto3
import pandas as pd
from botocore.config import Config as BotoConfig
from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from aws_graphrag.adapters.aws import (
    BedrockEmbeddingModelFactory,
    BedrockLanguageModelFactory,
)
from aws_graphrag.adapters.aws.bedrock import get_assumed_role_boto_session
from aws_graphrag.adapters.aws.token_counter import BedrockTokenCounter
from aws_graphrag.core import EvaluationException, get_logger
from aws_graphrag.evaluation.base import BaseGraphRAGEvaluator
from aws_graphrag.models import (
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
)

logger = get_logger(__name__)


class RagasEvaluator(BaseGraphRAGEvaluator):
    BUFFER_TOKENS: ClassVar[int] = 128
    METRIC_MAPPING = {
        EvaluationMetricType.ANSWER_CORRECTNESS: "answer_correctness",
        EvaluationMetricType.ANSWER_RELEVANCY: "answer_relevancy",
        EvaluationMetricType.CONTEXT_PRECISION: "context_precision",
        EvaluationMetricType.CONTEXT_RECALL: "context_recall",
        EvaluationMetricType.FAITHFULNESS: "faithfulness",
    }

    RAGAS_METRICS = {
        EvaluationMetricType.ANSWER_CORRECTNESS: answer_correctness,
        EvaluationMetricType.ANSWER_RELEVANCY: answer_relevancy,
        EvaluationMetricType.CONTEXT_PRECISION: context_precision,
        EvaluationMetricType.CONTEXT_RECALL: context_recall,
        EvaluationMetricType.FAITHFULNESS: faithfulness,
    }

    def __init__(
        self,
        config: Config,
        rag_chain: Any | None = None,
        boto_session: boto3.Session | None = None,
        **kwargs: Any,
    ) -> None:
        self.embeddings: Embeddings | None = None
        self.llm: BaseLanguageModel | None = None
        self.boto_session = boto_session or boto3.Session(
            profile_name=config.aws.profile_name
        )
        assumed_session = get_assumed_role_boto_session(
            self.boto_session, assumed_role_arn=config.aws.bedrock.assumed_role_arn
        )
        bedrock_client = assumed_session.client(
            "bedrock-runtime",
            region_name=config.aws.bedrock.region_name,
            config=BotoConfig(retries={"max_attempts": 3}),
        )
        self._token_counter = BedrockTokenCounter(
            model_id=config.evaluation.evaluation_model_id.value,
            client=bedrock_client,
        )
        self.ignore_errors = config.processing.ignore_errors
        super().__init__(
            config=config,
            evaluator_type=EvaluatorType.RAGAS,
            rag_chain=rag_chain,
            **kwargs,
        )

    def _initialize_evaluator(self, **kwargs: Any) -> None:
        try:
            self._initialize_models()
            logger.info(
                f"Ragas evaluator initialized with {len(self.config.evaluation.ragas_metrics)} metrics"
            )
        except Exception as e:
            raise EvaluationException(
                f"Failed to initialize Ragas evaluator: {e}"
            ) from e

    def _initialize_models(self) -> None:
        embedding_factory = BedrockEmbeddingModelFactory(
            config=self.config,
            boto_session=self.boto_session,
            region_name=self.config.aws.bedrock.region_name,
        )
        self.embeddings = embedding_factory.get_model(
            model_id=self.config.evaluation.embedding_model_id
        )

        llm_factory = BedrockLanguageModelFactory(
            config=self.config,
            boto_session=self.boto_session,
            region_name=self.config.aws.bedrock.region_name,
        )
        self.llm = llm_factory.get_model(
            model_id=self.config.evaluation.evaluation_model_id
        )

    def _truncate_contexts(self, results: list[EvaluationResult]) -> list[list[str]]:
        max_tokens = self.config.evaluation.max_context_tokens

        processed_contexts = []
        for result in results:
            safe_contexts = []
            current_tokens = 0
            for context in result.retrieved_contexts:
                context_token_count = self._token_counter.count_tokens(context)

                if (
                    current_tokens + context_token_count
                    > max_tokens - self.BUFFER_TOKENS
                ):
                    remaining_tokens = max_tokens - current_tokens
                    if remaining_tokens > self.BUFFER_TOKENS:
                        truncated_context, _ = (
                            self._token_counter.truncate_to_token_limit(
                                context, remaining_tokens
                            )
                        )
                        safe_contexts.append(truncated_context + "...")
                    break

                safe_contexts.append(context)
                current_tokens += context_token_count
            processed_contexts.append(safe_contexts)
        return processed_contexts

    def _parse_ragas_reports(
        self,
        ragas_df: pd.DataFrame,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
    ) -> list[EvaluationReport]:
        reports = []
        for i, query in enumerate(queries):
            row = ragas_df.iloc[i]
            metrics = []
            for metric_type, metric_name in self.METRIC_MAPPING.items():
                if (
                    metric_type in self.config.evaluation.ragas_metrics
                    and metric_name in row
                ):
                    raw_value = row[metric_name]
                    value = 0.0 if pd.isna(raw_value) else float(raw_value)
                    metrics.append(
                        EvaluationMetric(metric_type=metric_type, value=value)
                    )

            overall_score = (
                sum(m.value for m in metrics) / len(metrics) if metrics else 0.0
            )
            reports.append(
                EvaluationReport(
                    query_id=query.query_id,
                    evaluator_type=self.evaluator_type,
                    metrics=metrics,
                    overall_score=overall_score,
                    evaluation_time=datetime.now(),
                    metadata=self._extract_search_metadata(results[i]),
                )
            )
        return reports

    async def aevaluate_batch(
        self,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
        ground_truths: list[str],
        **kwargs: Any,
    ) -> list[EvaluationReport]:
        metrics_to_use = [
            self.RAGAS_METRICS[m]
            for m in self.config.evaluation.ragas_metrics
            if m in self.RAGAS_METRICS
        ]
        if not metrics_to_use:
            logger.warning("No valid Ragas metrics found in configuration")
            return []

        processed_contexts = self._truncate_contexts(results)
        dataset_dict = {
            "question": [q.question for q in queries],
            "answer": [r.generated_answer for r in results],
            "contexts": processed_contexts,
            "ground_truth": ground_truths,
        }

        try:
            eval_dataset = Dataset.from_dict(dataset_dict)
            ragas_result_df = await asyncio.to_thread(
                evaluate,
                dataset=eval_dataset,
                metrics=metrics_to_use,
                llm=self.llm,
                embeddings=self.embeddings,
                raise_exceptions=False,
                show_progress=self.show_progress,
                batch_size=self.config.processing.batch_size,
            )
            return self._parse_ragas_reports(
                ragas_result_df.to_pandas(), queries, results
            )
        except Exception as e:
            if self.ignore_errors:
                raise

            logger.error(f"Ragas async evaluation failed for batch: {e}")
            return [self._create_empty_report(q.query_id) for q in queries]

    async def aevaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        reports = await self.aevaluate_batch(
            [query], [result], [ground_truth], **kwargs
        )
        return reports[0] if reports else self._create_empty_report(query.query_id)

    def evaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        try:
            loop = asyncio.get_running_loop()
            return loop.run_until_complete(
                self.aevaluate_single(query, result, ground_truth, **kwargs)
            )
        except RuntimeError:
            return asyncio.run(
                self.aevaluate_single(query, result, ground_truth, **kwargs)
            )

    def validate_config(self) -> bool:
        unsupported_metrics = set(self.config.evaluation.ragas_metrics) - set(
            self.METRIC_MAPPING
        )
        if unsupported_metrics:
            logger.error(f"Unsupported Ragas metrics: '{unsupported_metrics}'")
            return False
        return True
