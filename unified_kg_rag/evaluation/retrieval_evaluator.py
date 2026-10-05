# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retrieval evaluator: did the sources the answer model saw include the gold docs?

Deterministic and LLM-free. Scores the rank-ordered sources reported for a
query (``EvaluationResult.retrieved_source_ids``: per source, the document ids
and file names it carries) against the dataset's ``reference_sources``:

- ``hit_at_k``: 1.0 if any reference is matched within the top ``k`` sources.
- ``recall_at_k``: fraction of distinct references matched within the top ``k``.
- ``mrr``: reciprocal rank of the first source matching any reference, over all
  reported sources (0.0 if none matches).

``k`` is ``evaluation.retrieval_k``. Matching rule: a reference matches a
source identifier when their file-name stems are equal, case-insensitively —
directories and extensions are ignored, so ``"docs/Report-A.pdf"``,
``"report-a.pdf"`` and ``"report-a"`` all match a source parsed from
``Report-A.pdf``. A document id (no extension) therefore matches only itself.

A query is skipped when it has no ``reference_sources``, or when none of its
sources carries provenance (no document id or file name) — scoring 0 there
would report missing metadata as a retrieval miss.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
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

from .base import SKIPPED_METRICS_KEY, BaseGraphRAGEvaluator


def source_key(identifier: str) -> str:
    """Normalize a document id / file name / path for matching (see module doc)."""
    name = PurePosixPath(identifier.strip().replace("\\", "/")).name
    return PurePosixPath(name).stem.lower() if name else ""


class RetrievalEvaluator(BaseGraphRAGEvaluator):
    """Scores hit@k / recall@k / MRR of reported sources vs reference_sources."""

    def __init__(self, config: Config, rag_chain: Any | None = None, **kwargs: Any):
        super().__init__(config, EvaluatorType.RETRIEVAL, rag_chain=rag_chain, **kwargs)

    def _initialize_evaluator(self, **kwargs: Any) -> None:
        # Pure, deterministic evaluator — no model to initialize.
        self.k = self.config.evaluation.retrieval_k

    def metric_types(self) -> list[EvaluationMetricType]:
        return [
            EvaluationMetricType.HIT_AT_K,
            EvaluationMetricType.RECALL_AT_K,
            EvaluationMetricType.MRR,
        ]

    def _skip(self, query_id: str, reason: str, **metadata: Any) -> EvaluationReport:
        return EvaluationReport(
            query_id=query_id,
            evaluator_type=self.evaluator_type,
            metrics=[],
            overall_score=None,
            metadata={
                **metadata,
                SKIPPED_METRICS_KEY: {m.value: reason for m in self.metric_types()},
            },
        )

    def evaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        references = {
            key
            for ref in result.metadata.get("reference_sources") or []
            if isinstance(ref, str) and (key := source_key(ref))
        }
        if not references:
            return self._skip(query.query_id, "no_reference_sources")
        ranked = [
            {key for ident in idents if (key := source_key(ident))}
            for idents in result.retrieved_source_ids
        ]
        if not any(ranked):
            return self._skip(
                query.query_id,
                "no_source_provenance",
                num_sources=len(ranked),
            )

        top = set().union(*ranked[: self.k])
        matched_at_k = references & top
        first_rank = next(
            (rank for rank, keys in enumerate(ranked, 1) if keys & references), None
        )
        values = {
            EvaluationMetricType.HIT_AT_K: 1.0 if matched_at_k else 0.0,
            EvaluationMetricType.RECALL_AT_K: len(matched_at_k) / len(references),
            EvaluationMetricType.MRR: 1.0 / first_rank if first_rank else 0.0,
        }
        metrics = [EvaluationMetric(metric_type=m, value=v) for m, v in values.items()]
        return EvaluationReport(
            query_id=query.query_id,
            evaluator_type=self.evaluator_type,
            metrics=metrics,
            overall_score=sum(values.values()) / len(values),
            evaluation_time=datetime.now(),
            metadata={
                **self._extract_search_metadata(result),
                "k": self.k,
                "num_sources": len(ranked),
                "num_references": len(references),
                "matched_references_at_k": sorted(matched_at_k),
                "first_relevant_rank": first_rank,
            },
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
