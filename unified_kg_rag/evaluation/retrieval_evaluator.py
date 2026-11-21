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
hit@5. The passages come first: the files of the sources that name their own
file (text units) in their rank order, then the lineage files of the derived
sources (the section types ``SectionType.lineage_attributed``: summaries,
community reports, entities, relationships, claims, graph query results) in
theirs. A derived source stands for every file of
its lineage, so at its own rank one high-ranked summary or report of twenty
files filled the top ``k`` and crowded out the passages ranked after it
(``tree`` scored below its own passages' ``text`` row). A strategy that
reports only derived sources (``global``) ranks as before. A source that
maps to no file is *unattributable* and contributes no file:

- ``hit_at_k``: 1.0 if any reference is among the top ``k`` files.
- ``recall_at_k``: fraction of distinct references among the top ``k`` files.
- ``ndcg_at_k``: binary-relevance nDCG over the top ``k`` files (Järvelin
  and Kekäläinen 2002, the log2 discount of ``trec_eval``'s ``ndcg_cut``):
  a file at rank ``i`` gains ``1 / log2(i + 1)`` when it names a reference no
  higher-ranked file named, over the gain of the references ranked first
  (``min(k, references)`` of them); so it is 1.0 exactly when the top files
  are the references, and ranks the passages first as recall does.
- ``mrr``: reciprocal rank of the first file matching any reference (0.0 if
  none matches).
- ``attributable_fraction``: attributable / reported sources — how much of the
  context the rank metrics can see. Emitted whenever the query has sources,
  with or without ``reference_sources``.
- ``citation_precision`` / ``citation_recall``: the same matching over what
  the answer *cites* (``evaluation.citation_scoring``).

The report metadata also carries ``by_source_type``: the same three metrics
over each section type's sources alone (``text``, ``entity``,
``relationship``, ``community``, ...), which the manager aggregates into
``summary.grouped_statistics["source_type"]``. The ``text`` row scores only the
text units the answer model read verbatim (reported sources are exactly the
context it was given).

``k`` is ``evaluation.retrieval_k``: its first cutoff scores ``hit_at_k``,
``recall_at_k`` and ``ndcg_at_k``, each further one ``hit_at_<k>``,
``recall_at_<k>`` and ``ndcg_at_<k>`` (``EvaluationMetric.k``). Matching rule: each identifier is
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
all is a miss and scores 0, unless the config makes it report none
(``no_sources_by_config``: a global query with
``search.global_search.evidence_tokens`` 0 answers from the key points
only). Every metric is skipped (``guardrail_blocked``)
when the guardrail blocked the answer, since that drops its sources.
"""

from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime
from typing import Any

from unified_kg_rag.adapters.evaluators.base import (
    SKIP_REASON_GUARDRAIL_BLOCKED,
    SKIPPED_METRICS_KEY,
    BaseGraphRAGEvaluator,
)
from unified_kg_rag.application.retrieval.citations import citations_enabled
from unified_kg_rag.domain.models import (
    STRATEGY_TRAITS,
    Config,
    EvaluationMetric,
    EvaluationMetricType,
    EvaluationQuery,
    EvaluationReport,
    EvaluationResult,
    EvaluatorType,
    SearchStrategy,
)
from unified_kg_rag.domain.models.evaluation import AT_K_METRICS, metric_name
from unified_kg_rag.domain.retrieval.section_types import LINEAGE_ATTRIBUTED_TYPES
from unified_kg_rag.evaluation.citation_scoring import (
    CITATION_METRICS,
    citation_scores,
)

# Why the rank metrics of a query the config makes report no sources are
# skipped (``_reports_no_sources``).
SKIP_REASON_NO_SOURCES_BY_CONFIG = "no_sources_by_config"

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


def distinct_references(
    refs: Any, files: list[frozenset[str]] | None = None
) -> dict[str, str]:
    """``{match key: display name}`` of the distinct references of ``refs``
    (``reference_sources``), against the ranked ``files`` (``ranked_files``).

    References with the same key are one. A reference spelled as others'
    stem (``report-a`` beside ``Report-A.pdf``, whose keys hold
    ``report-a``) is one with them when every file of ``files`` it matches
    one of them matches too: one file then never counts as two relevant
    documents (nDCG's ideal ranked two where it gained once, and perfect
    retrieval scored 0.61), and the more specific spelling stays. A file
    the stem alone matches is another document (``Amazon.txt`` beside the
    reference ``Amazon.com``, ``a.txt`` beside ``a.pdf``), so the stem
    stays a reference of its own, as it does when it matches no file."""
    references: dict[str, str] = {}
    for ref in refs or []:
        if isinstance(ref, str) and (key := reference_key(ref)):
            references.setdefault(key, ref.strip())

    def matched(key: str) -> set[int]:
        return {i for i, file_keys in enumerate(files or []) if key in file_keys}

    def merged(key: str) -> bool:
        longer = [
            other for other in references if other != key and key in source_keys(other)
        ]
        if not longer or not (own := matched(key)):
            return False
        return own <= set().union(*(matched(other) for other in longer))

    return {key: name for key, name in references.items() if not merged(key)}


def ranked_files(
    sources: list[list[str]], types: list[str] | None = None
) -> list[frozenset[str]]:
    """Match keys of the distinct files of ``sources``, in rank order.

    A file ranks where the first source naming it ranks; within one source,
    files keep the source's order. Files that reduce to the same name count
    once. With ``types`` (parallel to ``sources``), the sources of a
    lineage-attributed type (``LINEAGE_ATTRIBUTED_TYPES``) come after the
    others, each part in rank order.
    """
    if types is not None:
        derived = [
            i
            for i, source_type in enumerate(types[: len(sources)])
            if source_type in LINEAGE_ATTRIBUTED_TYPES
        ]
        derived_set = set(derived)
        order = [i for i in range(len(sources)) if i not in derived_set] + derived
        sources = [sources[i] for i in order]
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
    """Scores hit@k / recall@k / nDCG@k / MRR of reported sources vs
    reference_sources."""

    def __init__(self, config: Config, **kwargs: Any):
        super().__init__(config, EvaluatorType.RETRIEVAL, **kwargs)

    def _initialize_evaluator(self, **kwargs: Any) -> None:
        self.k, *self.further_k = self.config.evaluation.retrieval_k

    def metric_types(self) -> list[EvaluationMetricType]:
        return [
            EvaluationMetricType.HIT_AT_K,
            EvaluationMetricType.RECALL_AT_K,
            EvaluationMetricType.NDCG_AT_K,
            EvaluationMetricType.MRR,
            EvaluationMetricType.ATTRIBUTABLE_FRACTION,
            *CITATION_METRICS,
        ]

    def metric_names(self) -> list[str]:
        return [*super().metric_names(), *self._further_names()]

    def _further_names(self) -> list[str]:
        """The names of the rank metrics at the further cutoffs."""
        return [metric_name(m, k) for k in self.further_k for m in AT_K_METRICS]

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
                    SKIPPED_METRICS_KEY: dict.fromkeys(
                        self.metric_names(), SKIP_REASON_GUARDRAIL_BLOCKED
                    ),
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
        # The rank metrics at the further cutoffs, by (type, k).
        further: dict[tuple[EvaluationMetricType, int], float] = {}
        skipped: dict[str, str] = {}
        if sources:
            values[EvaluationMetricType.ATTRIBUTABLE_FRACTION] = attributable / len(
                sources
            )
        else:
            skipped[EvaluationMetricType.ATTRIBUTABLE_FRACTION.value] = "no_sources"

        files = ranked_files(sources, source_types)
        references = distinct_references(
            result.metadata.get("reference_sources"), files
        )
        metadata: dict[str, Any] = {
            **self._extract_search_metadata(result),
            "k": self.k,
            "num_sources": len(sources),
            "num_attributable_sources": attributable,
            "num_ranked_files": len(files),
            "num_references": len(references),
        }
        rank_metrics = (*AT_K_METRICS, EvaluationMetricType.MRR)
        # No sources at all is a retrieval miss (scored 0), unless the config
        # asks for none; sources that exist but none attributable cannot be
        # judged, so they are skipped.
        skip_reason = (
            "no_reference_sources"
            if not references
            else (
                "no_source_provenance"
                if sources and not files
                else (
                    SKIP_REASON_NO_SOURCES_BY_CONFIG
                    if not sources and self._reports_no_sources(result)
                    else None
                )
            )
        )
        citation_values, citation_skips = citation_scores(
            result,
            references,
            source_keys,
            enabled=citations_enabled(self.config, result.search_strategy),
        )
        values.update(citation_values)
        skipped.update(citation_skips)
        if skip_reason:
            skipped.update({m.value: skip_reason for m in rank_metrics})
            skipped.update(dict.fromkeys(self._further_names(), skip_reason))
        else:
            scores, matched_at_k, first_rank = self._rank_scores(files, references)
            values.update(scores)
            for k in self.further_k:
                at_k, _, _ = self._rank_scores(files, references, k)
                further.update({(m, k): at_k[m] for m in AT_K_METRICS})
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
                *(EvaluationMetric(metric_type=m, value=v) for m, v in values.items()),
                *(
                    EvaluationMetric(metric_type=m, k=k, value=v)
                    for (m, k), v in further.items()
                ),
            ],
            evaluation_time=datetime.now(),
            metadata=metadata,
        )

    def _reports_no_sources(self, result: EvaluationResult) -> bool:
        """Whether the config makes the query's strategy report no sources
        (``StrategyTraits.reports_sources``: global search with
        ``search.global_search.evidence_tokens`` 0 answers from the key
        points alone, model output, not sources)."""
        try:
            strategy = SearchStrategy(result.search_strategy)
        except ValueError:
            return False
        return not STRATEGY_TRAITS[strategy].reports_sources(self.config)

    def _rank_scores(
        self,
        files: list[frozenset[str]],
        references: dict[str, str],
        k: int | None = None,
    ) -> tuple[dict[EvaluationMetricType, float], list[str], int | None]:
        """(hit@k / recall@k / nDCG@k / MRR, matched references at k, first
        match rank), at ``k`` (the first cutoff by default)."""
        k = self.k if k is None else k

        def _matched(keys: list[frozenset[str]]) -> set[str]:
            return {
                name
                for ref_key, name in references.items()
                if any(ref_key in file_keys for file_keys in keys)
            }

        matched_at_k = _matched(files[:k])
        first_rank = next(
            (rank for rank, keys in enumerate(files, 1) if _matched([keys])), None
        )
        # A file gains at its rank when it names a reference not named above
        # it, so a reference counts once, as in recall.
        gain, credited = 0.0, set[str]()
        for rank, keys in enumerate(files[:k], 1):
            if new := _matched([keys]) - credited:
                credited |= new
                gain += 1.0 / math.log2(rank + 1)
        ideal = sum(
            1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(references)) + 1)
        )
        scores = {
            EvaluationMetricType.HIT_AT_K: 1.0 if matched_at_k else 0.0,
            EvaluationMetricType.RECALL_AT_K: len(matched_at_k) / len(references),
            EvaluationMetricType.NDCG_AT_K: gain / ideal,
            EvaluationMetricType.MRR: 1.0 / first_rank if first_rank else 0.0,
        }
        return scores, sorted(matched_at_k), first_rank
