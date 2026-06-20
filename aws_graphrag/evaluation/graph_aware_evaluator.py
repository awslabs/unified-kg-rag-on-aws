# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Graph-aware evaluator: entity/relationship coverage of generated answers.

Consumes the previously-unused ``EvaluationGroundTruth.expected_entities`` and
``expected_relationships`` fields. For each query it measures how many of the
expected graph artifacts the generated answer actually surfaces (case-insensitive
substring match), reporting precision/recall/F1 — a deterministic,
LLM-free signal complementing the LangChain/RAGAS text-similarity scores.

Expected artifacts are threaded onto ``EvaluationResult.metadata`` by the
manager (keys ``expected_entities`` / ``expected_relationships``), so this
evaluator needs no signature change to the abstract base.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from aws_graphrag.core import get_logger
from aws_graphrag.domain.models import (
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


class GraphAwareEvaluator(BaseGraphRAGEvaluator):
    """Scores entity/relationship coverage of the generated answer."""

    def __init__(self, config: Config, rag_chain: Any | None = None, **kwargs: Any):
        super().__init__(
            config, EvaluatorType.GRAPH_AWARE, rag_chain=rag_chain, **kwargs
        )

    def _initialize_evaluator(self, **kwargs: Any) -> None:
        # Pure, deterministic evaluator — no model to initialize.
        pass

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase word tokens, stripping surrounding punctuation.

        Deterministic and regex-free: split on whitespace, then strip
        non-alphanumeric characters from each token's edges. This gives
        word-boundary matching so ``"AI"`` does not match inside ``"airport"``.
        """
        tokens = []
        for raw in text.lower().split():
            token = raw.strip("\"'.,;:!?()[]{}<>/\\|`")
            if token:
                tokens.append(token)
        return tokens

    @classmethod
    def _phrase_in_tokens(cls, phrase: str, answer_tokens: list[str]) -> bool:
        """True if the phrase's word sequence appears contiguously in the answer."""
        phrase_tokens = cls._tokenize(phrase)
        if not phrase_tokens:
            return False
        window = len(phrase_tokens)
        for start in range(len(answer_tokens) - window + 1):
            if answer_tokens[start : start + window] == phrase_tokens:
                return True
        return False

    @classmethod
    def _coverage(
        cls, expected: list[str], answer: str
    ) -> tuple[float | None, float | None, float | None, int]:
        """Return (precision, recall, f1, num_matched) for expected-in-answer.

        Matching is word-boundary aware (contiguous token-sequence match), so an
        expected entity is credited only when its full word sequence appears as
        words in the answer — avoiding false substring hits (e.g. "AI" inside
        "airport"). Recall = matched / expected; precision is set equal (we
        cannot enumerate retrieved graph artifacts from free text), so F1
        collapses to the same value; all three are emitted for comparability.

        Returns ``None`` scores when nothing is expected for the dimension, so a
        query with (say) only expected entities is not penalized for having no
        expected relationships when the overall score is averaged.
        """
        if not expected:
            return None, None, None, 0
        answer_tokens = cls._tokenize(answer)
        matched = sum(
            1
            for item in expected
            if item and cls._phrase_in_tokens(item, answer_tokens)
        )
        recall = matched / len(expected)
        precision = recall
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1, matched

    @staticmethod
    def _metrics_for(
        precision: float | None,
        recall: float | None,
        f1: float | None,
        precision_type: EvaluationMetricType,
        recall_type: EvaluationMetricType,
        f1_type: EvaluationMetricType,
    ) -> list[EvaluationMetric]:
        return [
            EvaluationMetric(metric_type=precision_type, value=precision or 0.0),
            EvaluationMetric(metric_type=recall_type, value=recall or 0.0),
            EvaluationMetric(metric_type=f1_type, value=f1 or 0.0),
        ]

    def _build_metrics(
        self, result: EvaluationResult
    ) -> tuple[list[EvaluationMetric], dict[str, Any]]:
        answer = result.generated_answer or ""
        expected_entities = result.metadata.get("expected_entities", []) or []
        expected_relationships = result.metadata.get("expected_relationships", []) or []

        e_p, e_r, e_f1, e_matched = self._coverage(expected_entities, answer)
        r_p, r_r, r_f1, r_matched = self._coverage(expected_relationships, answer)

        # Only emit metrics for dimensions that actually have expectations, so a
        # missing dimension does not dilute the averaged overall score.
        metrics: list[EvaluationMetric] = []
        if expected_entities and e_p is not None:
            metrics += self._metrics_for(
                e_p,
                e_r,
                e_f1,
                EvaluationMetricType.ENTITY_PRECISION,
                EvaluationMetricType.ENTITY_RECALL,
                EvaluationMetricType.ENTITY_F1,
            )
        if expected_relationships and r_p is not None:
            metrics += self._metrics_for(
                r_p,
                r_r,
                r_f1,
                EvaluationMetricType.RELATIONSHIP_PRECISION,
                EvaluationMetricType.RELATIONSHIP_RECALL,
                EvaluationMetricType.RELATIONSHIP_F1,
            )
        metadata = {
            **self._extract_search_metadata(result),
            "expected_entity_count": len(expected_entities),
            "matched_entity_count": e_matched,
            "expected_relationship_count": len(expected_relationships),
            "matched_relationship_count": r_matched,
        }
        return metrics, metadata

    def evaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        metrics, metadata = self._build_metrics(result)
        scored = [m.value for m in metrics if m.value is not None]
        overall = sum(scored) / len(scored) if scored else 0.0
        return EvaluationReport(
            query_id=query.query_id,
            evaluator_type=self.evaluator_type,
            metrics=metrics,
            overall_score=overall,
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
        # Deterministic and CPU-only; reuse the sync path.
        return self.evaluate_single(query, result, ground_truth, **kwargs)
