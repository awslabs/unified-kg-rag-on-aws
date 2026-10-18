# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retrieval evaluator: did the files the answer model saw include the gold docs?

Deterministic and LLM-free. Scores the rank-ordered sources reported for a
query (``EvaluationResult.retrieved_source_ids``: per source, the file names
it is attributed to) against the dataset's ``reference_sources``. A text unit
names its file directly; an entity, relationship or community report is
attributed to the files of the text units in its lineage (``text_unit_ids``,
resolved by the manager, most-supported file first). The sources are
flattened into one list of distinct files in rank order (a file ranks where
the first source naming it ranks), and the metrics count files, not sources:
a community report is attributed to many files, so counting it as one source
let five random reports cover most of a small corpus and score near-perfect
hit@5. A source that maps to no file is *unattributable* and contributes no
file:

- ``hit_at_k``: 1.0 if any reference is among the top ``k`` files.
- ``recall_at_k``: fraction of distinct references among the top ``k`` files.
- ``mrr``: reciprocal rank of the first file matching any reference (0.0 if
  none matches).
- ``attributable_fraction``: attributable / reported sources — how much of the
  context the rank metrics can see. Emitted whenever the query has sources,
  with or without ``reference_sources``.

The report metadata also carries ``by_source_type``: the same three metrics
over each section type's sources alone (``text``, ``entity``,
``relationship``, ``community``, ...), which the manager aggregates into
``summary.grouped_statistics["source_type"]``. The ``text`` row scores only the
text units the answer model read verbatim (reported sources are exactly the
context it was given).

``k`` is ``evaluation.retrieval_k``. Matching rule: each identifier is
reduced, case-insensitively, to its name (``reference_key``); a source also
keeps its stem when it ends in a file extension (``.`` + 1-5 ASCII
letters/digits, at least one a letter, not a version such as ``.v2``;
``source_keys``). A reference matches a source when its name is one of the
source's keys, so a reference without an extension matches by stem and one
with an extension needs the same extension: ``"docs/Report-A.pdf"``,
``"report-a.pdf"`` and ``"report-a"`` all match a source parsed from
``Report-A.pdf``, but ``"Report-A.docx"`` does not. Sources carry only a file
name, so a reference's directories are ignored. A ``/`` or ``\\`` is a directory separator
only in a path-like identifier (one whose last segment has a file extension, or
that has a URI scheme or a leading ``/``, ``./``, ``../``, ``~/``); otherwise it
is part of the name, so titles such as ``"St. Louis Cardinals"``,
``"U.S. Route 66"`` and ``"AC/DC"`` are kept whole.

The rank metrics are skipped when the query has no ``reference_sources``, or
when it has sources but none is attributable — scoring 0 there would report
missing metadata as a retrieval miss. A query that retrieved no sources at
all is a miss and scores 0. Every metric is skipped (``guardrail_blocked``)
when the guardrail blocked the answer, since that drops its sources.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any

from unified_kg_rag.adapters.evaluators.base import (
    SKIP_REASON_GUARDRAIL_BLOCKED,
    SKIPPED_METRICS_KEY,
    BaseGraphRAGEvaluator,
)
from unified_kg_rag.domain.models import (
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
)

# Report-metadata key of the per-section-type scores, and the type of a source
# whose section type is not reported.
BY_SOURCE_TYPE_KEY = "by_source_type"
UNKNOWN_SOURCE_TYPE = "unknown"

# A file extension: "." + 1-5 ASCII alphanumerics with at least one letter, so
# "Version 2.0" or "Route 66" keep their numeric tail, and not a version
# (".v2"), so "notes.v1" and "notes.v2" are different names.
_EXTENSION = re.compile(r"\.(?![vV]\d+$)(?=[A-Za-z0-9]*[A-Za-z])[A-Za-z0-9]{1,5}$")
_PATH_PREFIXES = ("/", "./", "../", "~/")


def _is_path_like(identifier: str) -> bool:
    last_segment = re.split(r"[/\\]", identifier)[-1]
    return (
        "://" in identifier
        or identifier.startswith(_PATH_PREFIXES)
        or bool(_EXTENSION.search(last_segment))
    )


def reference_key(identifier: str) -> str:
    """Normalized name of a file name / path / title ("" if blank).

    NFKC + casefold first: macOS and many PDF tools write decomposed (NFD)
    Hangul file names, and full-width letters ("ＡＢＣ") should equal ASCII.
    """
    name = unicodedata.normalize("NFKC", identifier).casefold().strip()
    if _is_path_like(name):
        name = re.split(r"[/\\]", name)[-1]
    # Underscores stand for spaces in file names derived from titles
    # ("Miquette_Giraudy.txt" for the title "Miquette Giraudy"), so both sides
    # compare with underscores folded to spaces.
    return " ".join(name.replace("_", " ").split())


def source_keys(identifier: str) -> frozenset[str]:
    """Match keys of a source: its name and, with a file extension, its stem."""
    name = reference_key(identifier)
    if not name:
        return frozenset()
    stem = _EXTENSION.sub("", name).strip()
    return frozenset(k for k in (name, stem) if k)


def ranked_files(sources: list[list[str]]) -> list[frozenset[str]]:
    """Match keys of the distinct files of ``sources``, in rank order.

    A file ranks where the first source naming it ranks; within one source,
    files keep the source's order. Files that reduce to the same name count
    once.
    """
    seen: set[str] = set()
    files: list[frozenset[str]] = []
    for idents in sources:
        for ident in idents:
            keys = source_keys(ident)
            name = reference_key(ident)
            if keys and name not in seen:
                seen.add(name)
                files.append(keys)
    return files


class RetrievalEvaluator(BaseGraphRAGEvaluator):
    """Scores hit@k / recall@k / MRR of reported sources vs reference_sources."""

    def __init__(self, config: Config, **kwargs: Any):
        super().__init__(config, EvaluatorType.RETRIEVAL, **kwargs)

    def _initialize_evaluator(self, **kwargs: Any) -> None:
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
        if result.guardrail_blocked == "output":
            return EvaluationReport(
                query_id=query.query_id,
                evaluator_type=self.evaluator_type,
                metrics=[],
                evaluation_time=datetime.now(),
                metadata={
                    **self._extract_search_metadata(result),
                    SKIPPED_METRICS_KEY: {
                        m.value: SKIP_REASON_GUARDRAIL_BLOCKED
                        for m in self.metric_types()
                    },
                },
            )
        sources = result.retrieved_source_ids
        types = result.retrieved_source_types
        source_types = [
            (types[i] if i < len(types) else "") or UNKNOWN_SOURCE_TYPE
            for i in range(len(sources))
        ]
        attributable = sum(1 for idents in sources if any(map(source_keys, idents)))
        values: dict[EvaluationMetricType, float] = {}
        skipped: dict[str, str] = {}
        if sources:
            values[EvaluationMetricType.ATTRIBUTABLE_FRACTION] = attributable / len(
                sources
            )
        else:
            skipped[EvaluationMetricType.ATTRIBUTABLE_FRACTION.value] = "no_sources"

        # reference match key -> its display name (deduplicated by key).
        references: dict[str, str] = {}
        for ref in result.metadata.get("reference_sources") or []:
            if isinstance(ref, str) and (key := reference_key(ref)):
                references.setdefault(key, ref.strip())

        files = ranked_files(sources)
        metadata: dict[str, Any] = {
            **self._extract_search_metadata(result),
            "k": self.k,
            "num_sources": len(sources),
            "num_attributable_sources": attributable,
            "num_ranked_files": len(files),
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
            else "no_source_provenance" if sources and not files else None
        )
        if skip_reason:
            skipped.update({m.value: skip_reason for m in rank_metrics})
        else:
            scores, matched_at_k, first_rank = self._rank_scores(files, references)
            values.update(scores)
            metadata["matched_references_at_k"] = matched_at_k
            metadata["first_relevant_rank"] = first_rank
            by_type: dict[str, dict[str, float]] = {}
            for source_type in dict.fromkeys(source_types):
                type_files = ranked_files(
                    [
                        s
                        for s, t in zip(sources, source_types, strict=True)
                        if t == source_type
                    ]
                )
                if type_files:
                    type_scores, _, _ = self._rank_scores(type_files, references)
                    by_type[source_type] = {m.value: v for m, v in type_scores.items()}
            metadata[BY_SOURCE_TYPE_KEY] = by_type
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

    def _rank_scores(
        self, files: list[frozenset[str]], references: dict[str, str]
    ) -> tuple[dict[EvaluationMetricType, float], list[str], int | None]:
        """(hit@k / recall@k / MRR, matched references at k, first match rank)."""

        def _matched(keys: list[frozenset[str]]) -> set[str]:
            return {
                name
                for ref_key, name in references.items()
                if any(ref_key in file_keys for file_keys in keys)
            }

        matched_at_k = _matched(files[: self.k])
        first_rank = next(
            (rank for rank, keys in enumerate(files, 1) if _matched([keys])), None
        )
        scores = {
            EvaluationMetricType.HIT_AT_K: 1.0 if matched_at_k else 0.0,
            EvaluationMetricType.RECALL_AT_K: len(matched_at_k) / len(references),
            EvaluationMetricType.MRR: 1.0 / first_rank if first_rank else 0.0,
        }
        return scores, sorted(matched_at_k), first_rank
