# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retrieval evaluator: did the sources the answer model saw include the gold docs?

Deterministic and LLM-free. Scores the rank-ordered sources reported for a
query (``EvaluationResult.retrieved_source_ids``: per source, the file names
it is attributed to) against the dataset's ``reference_sources``. A text unit
names its file directly; an entity, relationship or community report is
attributed to the files of the text units in its lineage (``text_unit_ids``,
resolved by the manager). A community report's lineage is its whole
community, so for ``global``/``drift`` the scores are an upper bound on what
the answer model actually read. A source that maps to no file is
*unattributable*; only attributable sources are ranked:

- ``hit_at_k``: 1.0 if any reference is matched within the top ``k``
  attributable sources.
- ``recall_at_k``: fraction of distinct references matched within the top ``k``.
- ``mrr``: reciprocal rank of the first attributable source matching any
  reference (0.0 if none matches).
- ``attributable_fraction``: attributable / reported sources — how much of the
  context the rank metrics can see. Emitted whenever the query has sources,
  with or without ``reference_sources``.

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

The rank metrics are skipped when the query has no ``reference_sources``, or
when it has sources but none is attributable — scoring 0 there would report
missing metadata as a retrieval miss. A query that retrieved no sources at
all is a miss and scores 0.
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
    # Underscores stand for spaces in file names derived from titles
    # ("Miquette_Giraudy.txt" for the title "Miquette Giraudy"), so both sides
    # compare with underscores folded to spaces.
    name = " ".join(name.replace("_", " ").split()).lower()
    if not name:
        return frozenset()
    stem = _EXTENSION.sub("", name).strip()
    return frozenset(k for k in (name, stem) if k)


class RetrievalEvaluator(BaseGraphRAGEvaluator):
    """Scores hit@k / recall@k / MRR of reported sources vs reference_sources."""

    def __init__(self, config: Config, **kwargs: Any):
        super().__init__(config, EvaluatorType.RETRIEVAL, **kwargs)

    def _initialize_evaluator(self, **kwargs: Any) -> None:
        # Pure, deterministic evaluator — no model to initialize.
        self.k = self.config.evaluation.retrieval_k

    def metric_types(self) -> list[EvaluationMetricType]:
        return [
            EvaluationMetricType.HIT_AT_K,
            EvaluationMetricType.RECALL_AT_K,
            EvaluationMetricType.MRR,
            EvaluationMetricType.ATTRIBUTABLE_FRACTION,
        ]

    def evaluate_single(
        self,
        query: EvaluationQuery,
        result: EvaluationResult,
        ground_truth: str,
        **kwargs: Any,
    ) -> EvaluationReport:
        reported = [
            frozenset().union(*(source_keys(ident) for ident in idents))
            for idents in result.retrieved_source_ids
        ]
        # Rank only attributable sources: an unattributable one (no file name,
        # no resolvable lineage) can be neither a hit nor a miss, so it must not
        # push attributable sources out of the top k.
        ranked = [keys for keys in reported if keys]
        values: dict[EvaluationMetricType, float] = {}
        skipped: dict[str, str] = {}
        if reported:
            values[EvaluationMetricType.ATTRIBUTABLE_FRACTION] = len(ranked) / len(
                reported
            )
        else:
            skipped[EvaluationMetricType.ATTRIBUTABLE_FRACTION.value] = "no_sources"

        # reference display name -> its match keys (deduplicated by keys).
        references: dict[frozenset[str], str] = {}
        for ref in result.metadata.get("reference_sources") or []:
            if isinstance(ref, str) and (keys := source_keys(ref)):
                references.setdefault(keys, ref.strip())

        def _matched(sources: list[frozenset[str]]) -> set[str]:
            return {
                name
                for ref_keys, name in references.items()
                if any(ref_keys & keys for keys in sources)
            }

        metadata: dict[str, Any] = {
            **self._extract_search_metadata(result),
            "k": self.k,
            "num_sources": len(reported),
            "num_attributable_sources": len(ranked),
            "num_references": len(references),
        }
        rank_metrics = (
            EvaluationMetricType.HIT_AT_K,
            EvaluationMetricType.RECALL_AT_K,
            EvaluationMetricType.MRR,
        )
        # No sources at all is a retrieval miss (scored 0); sources that exist
        # but none attributable cannot be judged, so they are skipped.
        skip_reason = (
            "no_reference_sources"
            if not references
            else "no_source_provenance" if reported and not ranked else None
        )
        if skip_reason:
            skipped.update({m.value: skip_reason for m in rank_metrics})
        else:
            matched_at_k = _matched(ranked[: self.k])
            first_rank = next(
                (rank for rank, keys in enumerate(ranked, 1) if _matched([keys])),
                None,
            )
            values[EvaluationMetricType.HIT_AT_K] = 1.0 if matched_at_k else 0.0
            values[EvaluationMetricType.RECALL_AT_K] = len(matched_at_k) / len(
                references
            )
            values[EvaluationMetricType.MRR] = 1.0 / first_rank if first_rank else 0.0
            metadata["matched_references_at_k"] = sorted(matched_at_k)
            metadata["first_relevant_rank"] = first_rank
        if skipped:
            metadata[SKIPPED_METRICS_KEY] = skipped
        return EvaluationReport(
            query_id=query.query_id,
            evaluator_type=self.evaluator_type,
            metrics=[
                EvaluationMetric(metric_type=m, value=v) for m, v in values.items()
            ],
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
