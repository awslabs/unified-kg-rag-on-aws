# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, ClassVar, cast

import boto3
import pandas as pd
from botocore.config import Config as BotoConfig
from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from ragas import evaluate
from ragas.dataset_schema import EvaluationResult as RagasEvaluationResult
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from unified_kg_rag.adapters.aws import (
    BedrockEmbeddingModelFactory,
    BedrockLanguageModelFactory,
)
from unified_kg_rag.adapters.aws.bedrock import (
    get_assumed_role_boto_session,
    get_language_model_info,
)
from unified_kg_rag.adapters.aws.token_counter import BedrockTokenCounter
from unified_kg_rag.domain.models import (
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
    ModelPurpose,
)
from unified_kg_rag.evaluation.base import (
    FAILED_METRICS_KEY,
    SKIP_REASON_EMPTY_REFERENCE,
    SKIPPED_METRICS_KEY,
    BaseGraphRAGEvaluator,
    judge_model_kwargs,
)
from unified_kg_rag.ports.model_factory import EmbeddingFactoryPort
from unified_kg_rag.shared import EvaluationException, get_logger

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

    # Metrics whose RAGAS implementation reads the `reference` column. Scoring
    # them against an empty reference produces an artificial value, so they are
    # skipped (not zeroed) for rows without a ground-truth answer.
    REFERENCE_METRICS: ClassVar[frozenset[EvaluationMetricType]] = frozenset(
        {
            EvaluationMetricType.ANSWER_CORRECTNESS,
            EvaluationMetricType.CONTEXT_PRECISION,
            EvaluationMetricType.CONTEXT_RECALL,
        }
    )

    def __init__(
        self,
        config: Config,
        rag_chain: Any | None = None,
        boto_session: boto3.Session | None = None,
        embedding_factory: EmbeddingFactoryPort | None = None,
        **kwargs: Any,
    ) -> None:
        self.embeddings: Embeddings | None = None
        self.llm: BaseLanguageModel | None = None
        self.ragas_llm: LangchainLLMWrapper | None = None
        self._embedding_factory = embedding_factory
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
        eval_model_info = get_language_model_info(config.evaluation.evaluation_model_id)
        self._token_counter = BedrockTokenCounter(
            model_id=config.evaluation.evaluation_model_id.value,
            client=bedrock_client,
            api_supported=(
                eval_model_info.supports_count_tokens
                if eval_model_info is not None
                else True
            ),
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
                "Ragas evaluator initialized with %s metrics",
                len(self.config.evaluation.ragas_metrics),
            )
        except Exception as e:
            raise EvaluationException(
                f"Failed to initialize Ragas evaluator: {e}"
            ) from e

    def _initialize_models(self) -> None:
        embedding_factory = self._embedding_factory or BedrockEmbeddingModelFactory(
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
        model_id = self.config.evaluation.evaluation_model_id
        self.llm = llm_factory.get_model(
            model_id=model_id,
            model_purpose=ModelPurpose.EVALUATION,
            **judge_model_kwargs(self.config),
        )
        # ragas sets ``llm.temperature`` on every judge call (and never resets
        # it when the original was None). Claude 4.7+/5 reject sampling params,
        # so langchain-aws drops it with a warning per call — bypass it instead.
        model_info = get_language_model_info(model_id)
        supports_sampling = model_info is None or model_info.supports_sampling_params
        self.ragas_llm = LangchainLLMWrapper(
            self.llm, bypass_temperature=not supports_sampling
        )

    def _build_run_config(self) -> RunConfig:
        evaluation = self.config.evaluation
        return RunConfig(
            timeout=evaluation.ragas_timeout,
            max_workers=evaluation.ragas_max_workers,
            max_retries=evaluation.ragas_max_retries,
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

    def metric_types(self) -> list[EvaluationMetricType]:
        return [
            m for m in self.config.evaluation.ragas_metrics if m in self.RAGAS_METRICS
        ]

    def _parse_ragas_reports(
        self,
        ragas_df: pd.DataFrame,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
        ground_truths: list[str] | None = None,
    ) -> list[EvaluationReport]:
        reports = []
        for i, query in enumerate(queries):
            row = ragas_df.iloc[i]
            has_reference = ground_truths is None or bool(ground_truths[i].strip())
            metrics = []
            failed: dict[str, str] = {}
            skipped: dict[str, str] = {}
            for metric_type, metric_name in self.METRIC_MAPPING.items():
                if metric_type not in self.config.evaluation.ragas_metrics:
                    continue
                if metric_type in self.REFERENCE_METRICS and not has_reference:
                    skipped[metric_type.value] = SKIP_REASON_EMPTY_REFERENCE
                    continue
                if metric_name not in row:
                    failed[metric_type.value] = "metric missing from RAGAS output"
                    continue
                raw_value = row[metric_name]
                # RAGAS returns NaN when a metric is uncomputable (empty
                # answer, no contexts, parse failure). Record it as failed
                # rather than coercing to 0.0, which would conflate "failed to
                # measure" with a genuine zero score and bias the aggregate down.
                if pd.isna(raw_value):
                    failed[metric_type.value] = "uncomputable (NaN)"
                    continue
                metrics.append(
                    EvaluationMetric(metric_type=metric_type, value=float(raw_value))
                )

            overall_score = (
                sum(m.value for m in metrics) / len(metrics) if metrics else 0.0
            )
            metadata = self._extract_search_metadata(results[i])
            if failed:
                metadata[FAILED_METRICS_KEY] = failed
            if skipped:
                metadata[SKIPPED_METRICS_KEY] = skipped
            reports.append(
                EvaluationReport(
                    query_id=query.query_id,
                    evaluator_type=self.evaluator_type,
                    metrics=metrics,
                    overall_score=overall_score,
                    evaluation_time=datetime.now(),
                    metadata=metadata,
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
        metric_types = self.metric_types()
        if not metric_types:
            logger.warning("No valid Ragas metrics found in configuration")
            return []
        # If no row has a reference, reference-based metrics would be skipped
        # for every row anyway — don't pay for the judge calls.
        if not any(gt.strip() for gt in ground_truths):
            metric_types = [m for m in metric_types if m not in self.REFERENCE_METRICS]
        metrics_to_use = [self.RAGAS_METRICS[m] for m in metric_types]
        if not metrics_to_use:
            logger.warning(
                "Only reference-based Ragas metrics are configured and no row has "
                "a ground-truth answer; all Ragas metrics are skipped"
            )
            return [
                self._create_skipped_report(q.query_id, r)
                for q, r in zip(queries, results, strict=True)
            ]

        processed_contexts = self._truncate_contexts(results)
        dataset_dict = {
            "question": [q.question for q in queries],
            "answer": [r.generated_answer for r in results],
            "contexts": processed_contexts,
            "ground_truth": ground_truths,
        }

        try:
            eval_dataset = Dataset.from_dict(dataset_dict)
            ragas_result = await asyncio.to_thread(
                evaluate,
                dataset=eval_dataset,
                metrics=metrics_to_use,
                llm=self.ragas_llm or self.llm,
                embeddings=self.embeddings,
                raise_exceptions=False,
                run_config=self._build_run_config(),
                show_progress=self.show_progress,
                batch_size=self.config.processing.batch_size,
            )
            # ragas types `evaluate` as returning `RagasEvaluationResult |
            # Executor`; the Executor branch is only taken when the caller passes
            # return_executor=True, which this call never does.
            ragas_result = cast(RagasEvaluationResult, ragas_result)
            return self._parse_ragas_reports(
                ragas_result.to_pandas(), queries, results, ground_truths
            )
        except Exception as e:
            if not self.ignore_errors:
                raise

            logger.error("Ragas async evaluation failed for batch: %s", e)
            return [
                self._create_empty_report(
                    q.query_id, reason=f"Ragas batch evaluation failed: {e}"
                )
                for q in queries
            ]

    def _create_skipped_report(
        self, query_id: str, result: EvaluationResult
    ) -> EvaluationReport:
        metadata = self._extract_search_metadata(result)
        metadata[SKIPPED_METRICS_KEY] = {
            m.value: SKIP_REASON_EMPTY_REFERENCE for m in self.metric_types()
        }
        return EvaluationReport(
            query_id=query_id,
            evaluator_type=self.evaluator_type,
            metrics=[],
            overall_score=0.0,
            evaluation_time=datetime.now(),
            metadata=metadata,
        )

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
        coro = self.aevaluate_single(query, result, ground_truth, **kwargs)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No running loop — safe to drive one ourselves.
            return asyncio.run(coro)
        # A loop is ALREADY running on this thread; run_until_complete would
        # raise "This event loop is already running". Run the coroutine on a
        # separate thread with its own loop and block for the result.
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()

    def validate_config(self) -> bool:
        unsupported_metrics = set(self.config.evaluation.ragas_metrics) - set(
            self.METRIC_MAPPING
        )
        if unsupported_metrics:
            logger.error("Unsupported Ragas metrics: '%s'", unsupported_metrics)
            return False
        return True
