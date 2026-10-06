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

``k`` is ``evaluation.retrieval_k``. Matching rule (``source_keys``): each
identifier is reduced, case-insensitively, to its full name and — when it ends
in a file extension (``.`` + 1-5 ASCII letters/digits, at least one a letter) —
its stem; a reference matches a source when the two key sets intersect. So
``"docs/Report-A.pdf"``, ``"report-a.pdf"`` and ``"report-a"`` all match a
source parsed from ``Report-A.pdf``. A ``/`` or ``\\`` is a directory separator
only in a path-like identifier (one whose last segment has a file extension, or
that has a URI scheme or a leading ``/``, ``./``, ``../``, ``~/``); otherwise it
is part of the name, so titles such as ``"St. Louis Cardinals"``,
``"U.S. Route 66"`` and ``"AC/DC"`` are kept whole.

A query is skipped when it has no ``reference_sources``, or when none of its
sources carries provenance (no document id or file name) — scoring 0 there
would report missing metadata as a retrieval miss.
"""

from __future__ import annotations

import re
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

from .base import SKIPPED_METRICS_KEY, BaseGraphRAGEvaluator

# A file extension: "." + 1-5 ASCII alphanumerics with at least one letter, so
# "Version 2.0" or "Route 66" keep their numeric tail.
_EXTENSION = re.compile(r"\.(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{1,5}$")
_PATH_PREFIXES = ("/", "./", "../", "~/")


def _is_path_like(identifier: str) -> bool:
    last_segment = re.split(r"[/\\]", identifier)[-1]
    return (
        "://" in identifier
        or identifier.startswith(_PATH_PREFIXES)
        or bool(_EXTENSION.search(last_segment))
    )


def source_keys(identifier: str) -> frozenset[str]:
    """Match keys of a file name / path / title: full name and stem (module doc)."""
    name = identifier.strip()
    if _is_path_like(name):
        name = re.split(r"[/\\]", name)[-1]
    name = name.strip().lower()
    if not name:
        return frozenset()
    stem = _EXTENSION.sub("", name).strip()
    return frozenset(k for k in (name, stem) if k)


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
        # reference display name -> its match keys (deduplicated by keys).
        references: dict[frozenset[str], str] = {}
        for ref in result.metadata.get("reference_sources") or []:
            if isinstance(ref, str) and (keys := source_keys(ref)):
                references.setdefault(keys, ref.strip())
        if not references:
            return self._skip(query.query_id, "no_reference_sources")
        ranked: list[frozenset[str]] = [
            frozenset().union(*(source_keys(ident) for ident in idents))
            for idents in result.retrieved_source_ids
        ]
        if not any(ranked):
            return self._skip(
                query.query_id,
                "no_source_provenance",
                num_sources=len(ranked),
            )

        def _matched(sources: list[frozenset[str]]) -> set[str]:
            return {
                name
                for ref_keys, name in references.items()
                if any(ref_keys & keys for keys in sources)
            }

        matched_at_k = _matched(ranked[: self.k])
        first_rank = next(
            (rank for rank, keys in enumerate(ranked, 1) if _matched([keys])), None
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
