# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph-aware evaluator: entity/relationship coverage of generated answers.

Consumes the previously-unused ``EvaluationGroundTruth.expected_entities`` and
``expected_relationships`` fields. For each query it measures how many of the
expected graph artifacts the generated answer actually surfaces (whole-word
match after SQuAD normalization, shared with the answer-match evaluator via
``text_matching.phrase_in_text``; Korean particles tolerated; substring match
for single-word CJK phrases), reporting
coverage (= recall) — a deterministic, LLM-free
signal complementing the LangChain/RAGAS text-similarity scores. Precision/F1 are
deliberately NOT reported: they would require enumerating every entity in a
free-text answer (not reliably possible), so emitting them would only duplicate
the recall signal under another name.

An expected relationship given as a ``{"source": A, "target": B}`` pair or an
``"A -> B"`` string counts as covered when the answer mentions both endpoints
(answers rarely restate a relation verbatim); any other string must appear as
a phrase.

Expected artifacts are threaded onto ``EvaluationResult.metadata`` by the
manager (keys ``expected_entities`` / ``expected_relationships``), so this
evaluator needs no signature change to the abstract base.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from unified_kg_rag.domain.models import (
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
)
from unified_kg_rag.shared import get_logger

from .base import SKIPPED_METRICS_KEY, BaseGraphRAGEvaluator
from .text_matching import phrase_in_text

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

    def metric_types(self) -> list[EvaluationMetricType]:
        return [
            EvaluationMetricType.ENTITY_COVERAGE,
            EvaluationMetricType.RELATIONSHIP_COVERAGE,
        ]

    @staticmethod
    def _relationship_endpoints(item: Any) -> tuple[str, str] | None:
        """(source, target) for a pair dict or an ``"A -> B"`` string, else None."""
        if isinstance(item, dict):
            return str(item.get("source", "")), str(item.get("target", ""))
        if isinstance(item, str) and item.count("->") == 1:
            source, target = (part.strip() for part in item.split("->"))
            if source and target:
                return source, target
        return None

    @classmethod
    def _is_covered(cls, item: Any, answer: str, relationships: bool) -> bool:
        if not item:
            return False
        endpoints = cls._relationship_endpoints(item) if relationships else None
        if endpoints is not None:
            return all(phrase_in_text(end, answer) for end in endpoints)
        return isinstance(item, str) and phrase_in_text(item, answer)

    @classmethod
    def _coverage(
        cls, expected: list[Any], answer: str, relationships: bool = False
    ) -> tuple[float | None, int]:
        """Return (coverage, num_matched) for expected-artifacts-in-answer.

        Coverage = matched / expected (i.e. recall): the fraction of expected
        graph artifacts whose full word sequence appears, word-boundary aware,
        in the answer (so "AI" does not match inside "airport"). We intentionally
        report ONLY coverage — not precision/F1 — because precision would require
        enumerating the answer's own entities/relationships, which we cannot do
        from free text; emitting precision as a copy of recall (the previous
        behaviour) overstated the signal.

        Returns ``None`` when nothing is expected for the dimension, so a query
        with (say) only expected entities is not penalized for having no expected
        relationships when the overall score is averaged. With
        ``relationships=True``, pair/arrow items count when both endpoints match.
        """
        if not expected:
            return None, 0
        matched = sum(
            1 for item in expected if cls._is_covered(item, answer, relationships)
        )
        return matched / len(expected), matched

    def _build_metrics(
        self, result: EvaluationResult
    ) -> tuple[list[EvaluationMetric], dict[str, Any]]:
        answer = result.generated_answer or ""
        expected_entities = result.metadata.get("expected_entities", []) or []
        expected_relationships = result.metadata.get("expected_relationships", []) or []

        e_cov, e_matched = self._coverage(expected_entities, answer)
        r_cov, r_matched = self._coverage(
            expected_relationships, answer, relationships=True
        )

        # Only emit metrics for dimensions that actually have expectations, so a
        # missing dimension does not dilute the averaged overall score.
        metrics: list[EvaluationMetric] = []
        if expected_entities and e_cov is not None:
            metrics.append(
                EvaluationMetric(
                    metric_type=EvaluationMetricType.ENTITY_COVERAGE, value=e_cov
                )
            )
        if expected_relationships and r_cov is not None:
            metrics.append(
                EvaluationMetric(
                    metric_type=EvaluationMetricType.RELATIONSHIP_COVERAGE, value=r_cov
                )
            )
        metadata = {
            **self._extract_search_metadata(result),
            "expected_entity_count": len(expected_entities),
            "matched_entity_count": e_matched,
            "expected_relationship_count": len(expected_relationships),
            "matched_relationship_count": r_matched,
        }
        skipped: dict[str, str] = {}
        if not expected_entities:
            skipped[EvaluationMetricType.ENTITY_COVERAGE.value] = "no_expected_entities"
        if not expected_relationships:
            skipped[EvaluationMetricType.RELATIONSHIP_COVERAGE.value] = (
                "no_expected_relationships"
            )
        if skipped:
            metadata[SKIPPED_METRICS_KEY] = skipped
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
