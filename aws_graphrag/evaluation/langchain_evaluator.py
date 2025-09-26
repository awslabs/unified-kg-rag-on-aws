import asyncio
import json
import re
from collections.abc import Coroutine
from datetime import datetime
from typing import Any

import boto3
from langchain.evaluation import load_evaluator
from langchain.evaluation.schema import EvaluatorType as LCEvaluatorType
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

from aws_graphrag.aws import BedrockLanguageModelFactory
from aws_graphrag.core import EvaluationException, get_logger
from aws_graphrag.models import (
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
)

from .base import BaseGraphRAGEvaluator

logger = get_logger(__name__)
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
        rag_chain: Any | None = None,
        boto_session: boto3.Session | None = None,
        **kwargs: Any,
    ) -> None:
        self.llm: BaseLanguageModel | None = None
        self.evaluators: dict[EvaluationMetricType, Any] = {}
        self.boto_session = boto_session or boto3.Session(
            profile_name=config.aws.profile_name
        )
        super().__init__(
            config=config,
            evaluator_type=EvaluatorType.LANGCHAIN,
            rag_chain=rag_chain,
            **kwargs,
        )
        self.ignore_errors = config.processing.ignore_errors

    def _initialize_evaluator(self, **kwargs: Any) -> None:
        try:
            llm_factory = BedrockLanguageModelFactory(
                config=self.config,
                boto_session=self.boto_session,
                region_name=self.config.aws.bedrock.region_name,
            )
            self.llm = llm_factory.get_model(
                model_id=self.config.evaluation.evaluation_model_id,
            )
            self._initialize_metric_evaluators()
            logger.info(
                f"LangChain evaluator initialized with {len(self.evaluators)} metrics"
            )
        except Exception as e:
            logger.error(
                f"Failed to initialize LangChain evaluator: {e}", exc_info=True
            )
            raise EvaluationException(
                f"Failed to initialize LangChain evaluator: {e}"
            ) from e

    def _initialize_metric_evaluators(self) -> None:
        for metric_type in self.config.evaluation.langchain_metrics:
            if metric_type not in self.METRIC_MAPPING:
                logger.warning(f"Unsupported metric '{metric_type}' will be skipped")
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
    def _parse_score(eval_result: dict[str, Any]) -> float:
        score = eval_result.get("score") or eval_result.get("value")
        if isinstance(score, (int | float)):
            return float(score)

        reasoning = eval_result.get("reasoning", "")
        if isinstance(reasoning, str) and reasoning.strip():
            try:
                data = json.loads(reasoning)
                if isinstance(data.get("score"), (int | float)):
                    return float(data["score"])
            except json.JSONDecodeError:
                json_match = re.search(r"{.*}", reasoning, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                        if isinstance(data.get("score"), (int | float)):
                            return float(data["score"])
                    except json.JSONDecodeError:
                        pass

            match = re.search(r"(\d+\.?\d*)", reasoning)
            if match:
                return float(match.group(1))

        logger.warning(f"Could not parse score from evaluation result: '{eval_result}'")
        return 0.0

    def _prepare_eval_args(
        self,
        metric_type: EvaluationMetricType,
        question: str,
        answer: str,
        ground_truth: str,
    ) -> dict[str, str]:
        eval_args = {"input": question, "prediction": answer}
        if self.METRIC_MAPPING[metric_type].get("requires_reference"):
            eval_args["reference"] = ground_truth
        return eval_args

    @staticmethod
    def _handle_evaluation_error(
        metric_type: EvaluationMetricType, query_id: str, error: Exception
    ) -> EvaluationMetric:
        logger.error(
            f"Evaluation failed for '{metric_type}' on query '{query_id}': {error}"
        )
        return EvaluationMetric(
            metric_type=metric_type,
            value=0.0,
            explanation=f"Evaluation failed: {error}",
            metadata={"evaluation_error": True},
        )

    def _create_report(
        self, query_id: str, metrics: list[EvaluationMetric], result: EvaluationResult
    ) -> EvaluationReport:
        overall_score = sum(m.value for m in metrics) / len(metrics) if metrics else 0.0
        return EvaluationReport(
            query_id=query_id,
            evaluator_type=self.evaluator_type,
            metrics=metrics,
            overall_score=overall_score,
            evaluation_time=datetime.now(),
            metadata=self._extract_search_metadata(result),
        )

    def evaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        metrics = []
        for metric_type in self.config.evaluation.langchain_metrics:
            if metric_type not in self.evaluators:
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
            except Exception as e:
                if not self.ignore_errors:
                    raise
                metrics.append(
                    self._handle_evaluation_error(metric_type, query.query_id, e)
                )
        return self._create_report(query.query_id, metrics, result)

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

        score: float
        if metric_type == EvaluationMetricType.CORRECTNESS:
            score = float(eval_result.get("score", 0.0))
        else:
            score = self._parse_score(eval_result)

        raw_explanation = eval_result.get("reasoning", "")
        explanation = raw_explanation

        if metric_type == EvaluationMetricType.PARTIAL_CORRECTNESS:
            try:
                data = json.loads(raw_explanation)
                if isinstance(data, dict) and "reasoning" in data:
                    explanation = data["reasoning"]
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"Could not parse reasoning as JSON: '{raw_explanation}'"
                )

        return EvaluationMetric(
            metric_type=metric_type,
            value=score,
            explanation=explanation,
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

        for metric_type in self.config.evaluation.langchain_metrics:
            if metric_type in self.evaluators:
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
                if not self.ignore_errors:
                    raise res
                error_metric = self._handle_evaluation_error(
                    metric_type, query.query_id, res
                )
                metrics.append(error_metric)
            else:
                if isinstance(res, EvaluationMetric):
                    metrics.append(res)

        return self._create_report(query.query_id, metrics, result)

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

        score: float
        if metric_type == EvaluationMetricType.CORRECTNESS:
            score = float(eval_result.get("score", 0.0))
        else:
            score = self._parse_score(eval_result)

        raw_explanation = eval_result.get("reasoning", "")
        explanation = raw_explanation

        if metric_type == EvaluationMetricType.PARTIAL_CORRECTNESS:
            try:
                data = json.loads(raw_explanation)
                if isinstance(data, dict) and "reasoning" in data:
                    explanation = data["reasoning"]
            except (json.JSONDecodeError, TypeError):
                logger.warning(
                    f"Could not parse reasoning as JSON: '{raw_explanation}'"
                )

        return EvaluationMetric(
            metric_type=metric_type,
            value=score,
            explanation=explanation,
        )

    def get_supported_metrics(self) -> list[str]:
        return [metric.value for metric in self.METRIC_MAPPING]

    def validate_config(self) -> bool:
        unsupported = set(self.config.evaluation.langchain_metrics) - set(
            self.METRIC_MAPPING
        )
        if unsupported:
            logger.error(
                f"Unsupported LangChain metrics: {[m.value for m in unsupported]}"
            )
            return False
        return True
