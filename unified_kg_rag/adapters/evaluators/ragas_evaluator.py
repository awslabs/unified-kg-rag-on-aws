# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from datetime import datetime
from typing import Any, ClassVar, cast

import pandas as pd
from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from ragas import evaluate
from ragas.executor import Executor
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from unified_kg_rag.adapters.evaluators.base import (
    FAILED_METRICS_KEY,
    SKIP_REASON_EMPTY_REFERENCE,
    SKIP_REASON_GUARDRAIL_BLOCKED,
    SKIP_REASON_NO_CONTEXTS,
    SKIPPED_METRICS_KEY,
    BaseGraphRAGEvaluator,
    judge_model_kwargs,
)
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.domain.models import (
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
    ModelPurpose,
    ModelRole,
)
from unified_kg_rag.ports.model_factory import EmbeddingFactoryPort
from unified_kg_rag.shared import EvaluationException, get_logger

logger = get_logger(__name__)


class RagasEvaluator(BaseGraphRAGEvaluator):
    BUFFER_TOKENS: ClassVar[int] = 128
    # Each metric's score column in the RAGAS frame is its ``name``.
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

    # Metrics judged against the contexts: an answer blocked on output dropped
    # its sources, so they are skipped for it (the answer metrics still run).
    # A query blocked on input retrieved nothing: like the retrieval
    # evaluator, context precision/recall score that as a miss, while
    # faithfulness, undefined without a context (RAGAS returns NaN), is
    # skipped, as it is for an answer given without any context.
    CONTEXT_METRICS: ClassVar[frozenset[EvaluationMetricType]] = frozenset(
        {
            EvaluationMetricType.CONTEXT_PRECISION,
            EvaluationMetricType.CONTEXT_RECALL,
            EvaluationMetricType.FAITHFULNESS,
        }
    )

    # Metrics scored on at most ``evaluation.ragas_max_contexts`` contexts.
    COUNT_CAPPED_METRICS: ClassVar[frozenset[EvaluationMetricType]] = frozenset(
        {EvaluationMetricType.CONTEXT_PRECISION}
    )

    def __init__(
        self,
        config: Config,
        embedding_factory: EmbeddingFactoryPort | None = None,
        *,
        providers: Providers | None = None,
        show_progress: bool = True,
    ) -> None:
        self.embeddings: Embeddings | None = None
        self.llm: BaseLanguageModel | None = None
        self.ragas_llm: LangchainLLMWrapper | None = None
        self._embedding_factory = embedding_factory
        # Judge LLM, embeddings and token counter come from the shared
        # provider bundle (EvaluationManager passes the chain's).
        self.providers = Providers.resolve(config, providers)
        eval_model_info = self.providers.llm_model_info(
            config.model_id(ModelRole.EVALUATION)
        )
        self._token_counter = self.providers.token_counter(
            config.model_id(ModelRole.EVALUATION),
            api_supported=eval_model_info.supports_count_tokens,
        )
        super().__init__(
            config=config,
            evaluator_type=EvaluatorType.RAGAS,
            show_progress=show_progress,
        )

    def _initialize_evaluator(self, **kwargs: Any) -> None:
        self._judge_context_budget = self._resolve_judge_context_budget()
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
        embedding_factory = self._embedding_factory or self.providers.embedding_factory
        self.embeddings = embedding_factory.get_model(
            model_id=self.config.embedding_model_id
        )

        llm_factory = self.providers.llm_factory
        model_id = self.config.model_id(ModelRole.EVALUATION)
        self.llm = llm_factory.get_model(
            model_id=model_id,
            model_purpose=ModelPurpose.EVALUATION,
            **judge_model_kwargs(self.config),
        )
        # ragas sets ``llm.temperature`` on every judge call (and never resets
        # it when the original was None). Claude 4.7+/5 reject sampling params,
        # so langchain-aws drops it with a warning per call — bypass it instead.
        model_info = self.providers.llm_model_info(model_id)
        self.ragas_llm = LangchainLLMWrapper(
            self.llm, bypass_temperature=not model_info.supports_sampling_params
        )

    def _resolve_judge_context_budget(self) -> int | None:
        """The judges' context cap: explicit, else the answer budget (or none).

        The reported contexts are what the answer model was given, already
        within its budget; a smaller judge cap would grade faithfulness and
        context recall against context the answer was not limited to.
        """
        judge = self.config.evaluation.max_context_tokens
        answer = self.config.search.token_manager.max_context_tokens
        if judge is None:
            return answer
        if answer is not None and judge < answer:
            logger.warning(
                "evaluation.max_context_tokens (%d) is below the answer model's "
                "context budget (search.token_manager.max_context_tokens=%d): "
                "RAGAS judges will see less context than the answer model did. "
                "Unset it to judge the full context.",
                judge,
                answer,
            )
        return judge

    def _build_run_config(self) -> RunConfig:
        evaluation = self.config.evaluation
        return RunConfig(
            timeout=evaluation.ragas_timeout,
            max_workers=evaluation.ragas_max_workers,
            max_retries=evaluation.ragas_max_attempts,
        )

    def _budgeted_contexts(
        self, results: list[EvaluationResult]
    ) -> list[list[tuple[str, int]]]:
        """Per result: the top-ranked (context, tokens) within the token budget.

        Reported sources are in retrieval-rank order, so slicing keeps the
        top-ranked ones. The walk only depends on the contexts before each
        step, so a prefix of the result (``_select`` with a count cap) is the
        result for that cap: each context is counted once.
        """
        max_tokens = self._judge_context_budget
        processed: list[list[tuple[str, int]]] = []
        for result in results:
            kept: list[tuple[str, int]] = []
            current_tokens = 0
            for context in result.retrieved_contexts:
                context_token_count = self._token_counter.count_tokens(context)
                if max_tokens is None:
                    kept.append((context, context_token_count))
                    continue

                remaining_tokens = max_tokens - self.BUFFER_TOKENS - current_tokens
                if context_token_count > remaining_tokens:
                    if remaining_tokens > 0:
                        truncated_context, used = (
                            self._token_counter.truncate_to_token_limit(
                                context, remaining_tokens
                            )
                        )
                        kept.append((truncated_context + "...", used))
                    break

                kept.append((context, context_token_count))
                current_tokens += context_token_count
            processed.append(kept)
        return processed

    @staticmethod
    def _select(
        budgeted: list[list[tuple[str, int]]], max_contexts: int | None = None
    ) -> list[tuple[list[str], int]]:
        """(contexts, their tokens) of the first ``max_contexts`` per result."""
        return [
            (
                [c for c, _ in kept[:max_contexts]],
                sum(t for _, t in kept[:max_contexts]),
            )
            for kept in budgeted
        ]

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
        judge_inputs: list[dict[str, int]] | None = None,
        failed_runs: dict[EvaluationMetricType, str] | None = None,
    ) -> list[EvaluationReport]:
        """One report per query from the RAGAS frame.

        A metric of a run that raised (``failed_runs``: metric -> reason) is
        failed with that reason; a reference metric on a row without a
        reference is skipped, whether or not its run failed.
        """
        failed_runs = failed_runs or {}
        reports = []
        for i, query in enumerate(queries):
            row = ragas_df.iloc[i]
            has_reference = ground_truths is None or bool(ground_truths[i].strip())
            metrics = []
            failed: dict[str, str] = {}
            skipped: dict[str, str] = {}
            for metric_type, metric in self.RAGAS_METRICS.items():
                metric_name = metric.name
                if metric_type not in self.config.evaluation.ragas_metrics:
                    continue
                if reason := self._skip_reason(metric_type, has_reference, results[i]):
                    skipped[metric_type.value] = reason
                    continue
                if metric_type in failed_runs:
                    failed[metric_type.value] = failed_runs[metric_type]
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

            metadata = self._extract_search_metadata(results[i])
            if judge_inputs:
                metadata.update(judge_inputs[i])
            if failed:
                metadata[FAILED_METRICS_KEY] = failed
            if skipped:
                metadata[SKIPPED_METRICS_KEY] = skipped
            reports.append(
                EvaluationReport(
                    query_id=query.query_id,
                    evaluator_type=self.evaluator_type,
                    metrics=metrics,
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
        """Score each row on the metrics that apply to it: none of the
        reference-based ones without a reference, none of the context-based
        ones for a guardrail-blocked answer (no judge calls for metrics that
        would be skipped). Rows that run the same metrics share a RAGAS run.

        A RAGAS failure marks that run's metrics as failed for its rows rather
        than aborting the evaluation; ``--max-failure-rate`` decides the exit.
        """
        metric_types = self.metric_types()
        if not metric_types:
            logger.warning("No valid Ragas metrics found in configuration")
            return []
        partitions: dict[tuple[EvaluationMetricType, ...], list[int]] = {}
        for i, (result, gt) in enumerate(zip(results, ground_truths, strict=True)):
            run = tuple(
                m
                for m in metric_types
                if self._skip_reason(m, bool(gt.strip()), result) is None
            )
            partitions.setdefault(run, []).append(i)
        reports: dict[int, EvaluationReport] = {}
        for partition_metrics, rows in partitions.items():
            row_queries = [queries[i] for i in rows]
            row_results = [results[i] for i in rows]
            if partition_metrics:
                scored = await self._score_rows(
                    row_queries,
                    row_results,
                    [ground_truths[i] for i in rows],
                    list(partition_metrics),
                )
            else:
                scored = [
                    self._create_skipped_report(
                        q.query_id, r, bool(ground_truths[i].strip())
                    )
                    for i, q, r in zip(rows, row_queries, row_results, strict=True)
                ]
            reports.update(zip(rows, scored, strict=True))
        return [reports[i] for i in range(len(queries))]

    async def _score_rows(
        self,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
        ground_truths: list[str],
        metric_types: list[EvaluationMetricType],
    ) -> list[EvaluationReport]:
        # Context precision makes one judge call per context, so only it gets
        # the count cap (``context_precision@N``). Faithfulness and context
        # recall judge whether the answer / reference is supported by the
        # context, so they see the full token-budgeted set: capping them would
        # mark claims supported by a lower-ranked source as unsupported.
        # Counted once: the capped selection is a prefix of the full one (a
        # second walk recounted every context past the counter's LRU cache).
        budgeted = self._budgeted_contexts(results)
        full = self._select(budgeted)
        precision = [m for m in metric_types if m in self.COUNT_CAPPED_METRICS]
        capped = (
            self._select(budgeted, self.config.evaluation.ragas_max_contexts)
            if precision
            else full
        )
        judge_inputs = [
            {"judge_contexts": len(ctx), "judge_context_tokens": tokens}
            for ctx, tokens in full
        ]
        if precision:
            for inputs, (ctx, tokens) in zip(judge_inputs, capped, strict=True):
                inputs["context_precision_contexts"] = len(ctx)
                inputs["context_precision_context_tokens"] = tokens
        # One RAGAS run when the cap changes nothing; otherwise split the
        # capped metric into its own run (same judge-call count overall).
        groups = (
            [(metric_types, full)]
            if capped == full
            else [
                (m_types, contexts)
                for m_types, contexts in (
                    ([m for m in metric_types if m not in precision], full),
                    (precision, capped),
                )
                if m_types
            ]
        )

        # Each run fails alone: a failed run marks only its own metrics
        # failed and keeps the scores the other run produced.
        frames: list[pd.DataFrame] = []
        failed_runs: dict[EvaluationMetricType, str] = {}
        for m_types, contexts in groups:
            try:
                frames.append(
                    await self._run_ragas(
                        queries,
                        results,
                        [ctx for ctx, _ in contexts],
                        ground_truths,
                        [self.RAGAS_METRICS[m] for m in m_types],
                    )
                )
            except Exception as e:
                logger.error("Ragas evaluation failed for batch: %s", e)
                failed_runs.update(
                    dict.fromkeys(m_types, f"Ragas batch evaluation failed: {e}")
                )
        return self._parse_ragas_reports(
            (
                pd.concat(frames, axis=1)
                if frames
                else pd.DataFrame(index=range(len(queries)))
            ),
            queries,
            results,
            ground_truths,
            judge_inputs,
            failed_runs=failed_runs,
        )

    async def _run_ragas(
        self,
        queries: list[EvaluationQuery],
        results: list[EvaluationResult],
        contexts: list[list[str]],
        ground_truths: list[str],
        metrics: list[Any],
    ) -> pd.DataFrame:
        eval_dataset = Dataset.from_dict(
            {
                "question": [q.question for q in queries],
                "answer": [r.generated_answer for r in results],
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        # evaluate() binds the judge to the (module-level) metrics it is given
        # and, with return_executor, never unbinds it: restore them after.
        unbound = [
            (metric, attr)
            for metric in metrics
            for attr in ("llm", "embeddings", "answer_similarity")
            if hasattr(metric, attr) and getattr(metric, attr) is None
        ]
        try:
            # The executor, not the result: a thread running the judge calls
            # cannot be cancelled, but the executor can, so a cancelled or
            # timed-out evaluation stops judging instead of finishing the run.
            executor = cast(
                Executor,
                await asyncio.to_thread(
                    evaluate,
                    dataset=eval_dataset,
                    metrics=metrics,
                    llm=self.ragas_llm or self.llm,
                    embeddings=self.embeddings,
                    raise_exceptions=False,
                    run_config=self._build_run_config(),
                    show_progress=self.show_progress,
                    return_executor=True,
                ),
            )
            try:
                scores = await asyncio.to_thread(executor.results)
            except BaseException:
                executor.cancel()
                raise
        finally:
            for metric, attr in unbound:
                setattr(metric, attr, None)
        if not scores:
            raise EvaluationException("RAGAS returned no scores")
        # One row per query, one score column per metric, in submission
        # order (as ragas' own evaluate builds its result).
        return pd.DataFrame(
            [
                {m.name: scores[len(metrics) * i + j] for j, m in enumerate(metrics)}
                for i in range(len(queries))
            ]
        )

    def _skip_reason(
        self,
        metric_type: EvaluationMetricType,
        has_reference: bool,
        result: EvaluationResult,
    ) -> str | None:
        """Why ``metric_type`` does not apply to ``result``'s row, if it does not."""
        if result.guardrail_blocked == "output" and metric_type in self.CONTEXT_METRICS:
            return SKIP_REASON_GUARDRAIL_BLOCKED
        if (
            metric_type is EvaluationMetricType.FAITHFULNESS
            and not result.retrieved_contexts
        ):
            return (
                SKIP_REASON_GUARDRAIL_BLOCKED
                if result.guardrail_blocked
                else SKIP_REASON_NO_CONTEXTS
            )
        if metric_type in self.REFERENCE_METRICS and not has_reference:
            return SKIP_REASON_EMPTY_REFERENCE
        return None

    def _create_skipped_report(
        self, query_id: str, result: EvaluationResult, has_reference: bool
    ) -> EvaluationReport:
        metadata = self._extract_search_metadata(result)
        metadata[SKIPPED_METRICS_KEY] = {
            m.value: reason
            for m in self.metric_types()
            if (reason := self._skip_reason(m, has_reference, result))
        }
        return EvaluationReport(
            query_id=query_id,
            evaluator_type=self.evaluator_type,
            metrics=[],
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

    def validate_config(self) -> bool:
        unsupported_metrics = set(self.config.evaluation.ragas_metrics) - set(
            self.RAGAS_METRICS
        )
        if unsupported_metrics:
            logger.error("Unsupported Ragas metrics: '%s'", unsupported_metrics)
            return False
        return True
