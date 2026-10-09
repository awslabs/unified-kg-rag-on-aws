# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections.abc import Coroutine
from datetime import datetime
from typing import Any

import boto3
from langchain_classic.evaluation import load_evaluator
from langchain_classic.evaluation.schema import EvaluatorType as LCEvaluatorType
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

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
)
from unified_kg_rag.evaluation.base import (
    FAILED_METRICS_KEY,
    SKIP_REASON_EMPTY_REFERENCE,
    SKIPPED_METRICS_KEY,
    BaseGraphRAGEvaluator,
    judge_model_kwargs,
)
from unified_kg_rag.shared import EvaluationException, get_logger
from unified_kg_rag.shared.utils import parse_llm_json, text_digest

logger = get_logger(__name__)


class UnscoredJudgeOutputError(EvaluationException):
    """The judge answered, but its output carries no usable score.

    Like a RAGAS NaN, this is a measurement outcome rather than an
    infrastructure error: the metric is recorded under ``failed_metrics``
    whatever ``processing.ignore_errors`` says.
    """


PARTIAL_CORRECTNESS_PROMPT_TEMPLATE = """You are an expert evaluator tasked with assessing the correctness of a
submitted answer against a reference answer.

**TASK**: Compare the submitted answer with the reference answer and assign a correctness score.

**SCORING CRITERIA**:
- **1.0**: Perfect match - The submitted answer is completely accurate and contains all key information from the
reference
- **0.8-0.9**: Excellent - Minor omissions or slight rephrasing, but all core facts are correct
- **0.6-0.7**: Good - Most information is correct with some minor inaccuracies or missing details
- **0.4-0.5**: Fair - Partially correct with significant gaps or some incorrect information
- **0.2-0.3**: Poor - Contains some relevant information but mostly incorrect or incomplete
- **0.0-0.1**: Completely incorrect or contradicts the reference answer

**EVALUATION GUIDELINES**:
1. Focus on factual accuracy and completeness
2. Consider semantic equivalence (different wording expressing the same meaning)
3. Penalize contradictions more than omissions
4. Reward comprehensive coverage of key points

**OUTPUT FORMAT**:
YOU MUST RESPOND WITH ONLY A SINGLE VALID JSON OBJECT. Do not include any other text, explanations, or preamble before
or after the JSON. The JSON object must conform to this structure:
{{
    "score": <decimal_between_0_and_1>,
    "reasoning": "<A concise explanation for the score, focusing only on the core rationale.>"
}}

**QUESTION**: {query}

**REFERENCE ANSWER**: {answer}

**SUBMITTED ANSWER**: {result}

**EVALUATION (JSON ONLY)**:"""


class LangChainEvaluator(BaseGraphRAGEvaluator):
    METRIC_MAPPING = {
        EvaluationMetricType.CORRECTNESS: {
            "type": LCEvaluatorType.LABELED_CRITERIA,
            "criteria": "correctness",
            "requires_reference": True,
        },
        EvaluationMetricType.PARTIAL_CORRECTNESS: {
            "type": LCEvaluatorType.QA,
            "prompt_template": PARTIAL_CORRECTNESS_PROMPT_TEMPLATE,
            "requires_reference": True,
        },
    }

    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        *,
        providers: Providers | None = None,
        **kwargs: Any,
    ) -> None:
        self.llm: BaseLanguageModel | None = None
        self.evaluators: dict[EvaluationMetricType, Any] = {}
        # The judge LLM comes from the shared provider bundle
        # (EvaluationManager passes the chain's).
        self.providers = Providers.resolve(config, providers, boto_session)
        self.boto_session = self.providers.boto_session
        super().__init__(
            config=config,
            evaluator_type=EvaluatorType.LANGCHAIN,
            **kwargs,
        )
        self.ignore_errors = config.processing.ignore_errors

    def _initialize_evaluator(self, **kwargs: Any) -> None:
        try:
            self.llm = self.providers.llm_factory.get_model(
                model_id=self.config.evaluation.evaluation_model_id,
                model_purpose=ModelPurpose.EVALUATION,
                **judge_model_kwargs(self.config),
            )
            self._initialize_metric_evaluators()
            logger.info(
                "LangChain evaluator initialized with %s metrics", len(self.evaluators)
            )
        except Exception as e:
            logger.error(
                "Failed to initialize LangChain evaluator: %s", e, exc_info=True
            )
            raise EvaluationException(
                f"Failed to initialize LangChain evaluator: {e}"
            ) from e

    def _initialize_metric_evaluators(self) -> None:
        for metric_type in self.config.evaluation.langchain_metrics:
            if metric_type not in self.METRIC_MAPPING:
                logger.warning("Unsupported metric '%s' will be skipped", metric_type)
                continue

            metric_config = self.METRIC_MAPPING[metric_type]
            eval_kwargs: dict[str, Any] = {"llm": self.llm}
            if "criteria" in metric_config:
                eval_kwargs["criteria"] = metric_config["criteria"]
            if "prompt_template" in metric_config:
                eval_kwargs["prompt"] = PromptTemplate.from_template(
                    str(metric_config["prompt_template"])
                )

            evaluator_type = metric_config["type"]
            evaluator_type_value = LCEvaluatorType(evaluator_type)

            self.evaluators[metric_type] = load_evaluator(
                evaluator_type_value, **eval_kwargs
            )

    @staticmethod
    def _judge_score(
        metric_type: EvaluationMetricType, eval_result: dict[str, Any]
    ) -> tuple[float, str]:
        """Return ``(score, explanation)`` from one judge result.

        ``correctness`` uses LangChain's criteria grader, whose ``score`` is the
        parsed Y/N verdict (1/0, ``None`` when no verdict was found).
        ``partial_correctness`` uses a custom prompt that asks for a JSON
        ``{"score", "reasoning"}`` object; LangChain's QA chain still runs its
        CORRECT/INCORRECT word heuristic over that text (``"... is correct"}``
        becomes score 1), so its ``score`` is ignored and only the JSON score
        in the raw output (``reasoning``) counts.

        Raises ``UnscoredJudgeOutputError`` when no valid score in [0, 1] is
        present: the metric is then recorded as failed, never as 0.0.
        """
        reasoning = eval_result.get("reasoning")
        raw = reasoning if isinstance(reasoning, str) else ""
        if metric_type == EvaluationMetricType.PARTIAL_CORRECTNESS:
            data = parse_llm_json(raw)
            score = data.get("score")
            explanation = data.get("reasoning")
            if not isinstance(explanation, str):
                explanation = raw
        else:
            score = eval_result.get("score")
            explanation = raw
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not 0.0 <= score <= 1.0  # also rejects NaN
        ):
            # The judge's text quotes the answer and reference; keep it out of
            # the message (DEBUG only), like other model output.
            logger.debug("Judge output without a valid score: %r", raw)
            raise UnscoredJudgeOutputError(
                f"No valid score in judge output ({text_digest(raw)})"
            )
        return float(score), explanation

    def metric_types(self) -> list[EvaluationMetricType]:
        return [
            m for m in self.config.evaluation.langchain_metrics if m in self.evaluators
        ]

    def _requires_reference(self, metric_type: EvaluationMetricType) -> bool:
        return bool(self.METRIC_MAPPING[metric_type].get("requires_reference"))

    def _prepare_eval_args(
        self,
        metric_type: EvaluationMetricType,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> dict[str, str]:
        eval_args = {"input": question, "prediction": answer}
        if self._requires_reference(metric_type):
            eval_args["reference"] = ground_truth
        return eval_args

    @staticmethod
    def _handle_evaluation_error(
        metric_type: EvaluationMetricType, query_id: str, error: Exception
    ) -> str:
        """Log a judge failure and return the reason recorded in the report.

        A failed judge call yields NO metric value (it used to emit 0.0, which
        dragged the mean down and conflated "could not measure" with "wrong").
        The failure is recorded under the report's ``failed_metrics`` instead,
        matching RAGAS's NaN-skip behaviour.
        """
        logger.error(
            "Evaluation failed for '%s' on query '%s': %s", metric_type, query_id, error
        )
        return f"Evaluation failed: {error}"

    def _create_report(
        self,
        query_id: str,
        metrics: list[EvaluationMetric],
        result: EvaluationResult,
        failed: dict[str, str] | None = None,
        skipped: dict[str, str] | None = None,
    ) -> EvaluationReport:
        metadata = self._extract_search_metadata(result)
        if failed:
            metadata[FAILED_METRICS_KEY] = failed
        if skipped:
            metadata[SKIPPED_METRICS_KEY] = skipped
        return EvaluationReport(
            query_id=query_id,
            evaluator_type=self.evaluator_type,
            metrics=metrics,
            evaluation_time=datetime.now(),
            metadata=metadata,
        )

    def _should_skip(
        self, metric_type: EvaluationMetricType, ground_truth: str
    ) -> bool:
        """Reference-based metrics are not applicable without a reference.

        Datasets may legitimately omit ``answer`` (graph-aware-only rows), in
        which case ground_truth is "". Judging correctness against an empty
        reference yields an artificial zero, so the metric is skipped instead.
        """
        return self._requires_reference(metric_type) and not ground_truth.strip()

    def evaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        metrics = []
        failed: dict[str, str] = {}
        skipped: dict[str, str] = {}
        for metric_type in self.config.evaluation.langchain_metrics:
            if metric_type not in self.evaluators:
                continue
            if self._should_skip(metric_type, ground_truth):
                skipped[metric_type.value] = SKIP_REASON_EMPTY_REFERENCE
                continue
            try:
                metric = self._evaluate_with_metric(
                    self.evaluators[metric_type],
                    metric_type,
                    query.question,
                    result.generated_answer,
                    ground_truth,
                )
                metrics.append(metric)
            except UnscoredJudgeOutputError as e:
                failed[metric_type.value] = self._handle_evaluation_error(
                    metric_type, query.query_id, e
                )
            except Exception as e:
                if not self.ignore_errors:
                    raise
                failed[metric_type.value] = self._handle_evaluation_error(
                    metric_type, query.query_id, e
                )
        return self._create_report(query.query_id, metrics, result, failed, skipped)

    def _evaluate_with_metric(
        self,
        evaluator: Any,
        metric_type: EvaluationMetricType,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> EvaluationMetric:
        eval_args = self._prepare_eval_args(metric_type, question, answer, ground_truth)
        eval_result = evaluator.evaluate_strings(**eval_args)
        return self._to_metric(metric_type, eval_result)

    def _to_metric(
        self, metric_type: EvaluationMetricType, eval_result: dict[str, Any]
    ) -> EvaluationMetric:
        score, explanation = self._judge_score(metric_type, eval_result)
        return EvaluationMetric(
            metric_type=metric_type, value=score, explanation=explanation
        )

    async def aevaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        tasks: list[Coroutine] = []
        metric_types_to_run: list[EvaluationMetricType] = []
        failed: dict[str, str] = {}
        skipped: dict[str, str] = {}

        for metric_type in self.config.evaluation.langchain_metrics:
            if metric_type in self.evaluators:
                if self._should_skip(metric_type, ground_truth):
                    skipped[metric_type.value] = SKIP_REASON_EMPTY_REFERENCE
                    continue
                tasks.append(
                    self._aevaluate_with_metric(
                        self.evaluators[metric_type],
                        metric_type,
                        query.question,
                        result.generated_answer,
                        ground_truth,
                    )
                )
                metric_types_to_run.append(metric_type)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        metrics: list[EvaluationMetric] = []
        for i, res in enumerate(results):
            metric_type = metric_types_to_run[i]
            if isinstance(res, Exception):
                if not self.ignore_errors and not isinstance(
                    res, UnscoredJudgeOutputError
                ):
                    raise res
                failed[metric_type.value] = self._handle_evaluation_error(
                    metric_type, query.query_id, res
                )
            elif isinstance(res, EvaluationMetric):
                metrics.append(res)

        return self._create_report(query.query_id, metrics, result, failed, skipped)

    async def _aevaluate_with_metric(
        self,
        evaluator: Any,
        metric_type: EvaluationMetricType,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> EvaluationMetric:
        eval_args = self._prepare_eval_args(metric_type, question, answer, ground_truth)
        eval_result = await evaluator.aevaluate_strings(**eval_args)
        return self._to_metric(metric_type, eval_result)

    def validate_config(self) -> bool:
        unsupported = set(self.config.evaluation.langchain_metrics) - set(
            self.METRIC_MAPPING
        )
        if unsupported:
            logger.error(
                "Unsupported LangChain metrics: %s", [m.value for m in unsupported]
            )
            return False
        return True
