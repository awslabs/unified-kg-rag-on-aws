# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Citation precision and recall against ``reference_sources``.

The retrieval evaluator scores what the answer model *saw*; these score what
the answer *cites* (``RAGOutput.citations`` whose marker the answer uses,
docs/design.md §7.1). Each cited citation stands for the files of its
documents (a passage's one file; a graph section's resolved lineage, its
main document first), kept on the result as
``EvaluationResult.cited_source_ids`` (with ``cited_section_types``):

- ``citation_precision``: over the cited citations that name any file, the
  mean share of each one's files that are references. A passage names one
  file, so it counts 1 or 0; a graph citation of ``n`` files counts the
  share of them that are references, so a hub entity whose lineage spans
  ten files, one a reference, counts 0.1, not 1 (any of ten matching
  credited it whole). A citation naming no file (a seed-only fact, an
  unresolved lineage) is left out.
- ``citation_recall``: per distinct reference, the largest weight a cited
  citation naming it gives it (``1 / n`` for a citation of ``n`` files),
  averaged over the references: a passage of the reference recalls it
  fully, a ten-file hub one tenth.
- ``citation_precision_passages``: the share of cited passages (``text``
  sections) whose file is a reference, without any lineage weighting.

Files match references as in the retrieval evaluator (``source_keys``).
All are skipped without ``reference_sources``, with
``search.citations`` off for the result's strategy (``citations_disabled``) and for a result
saved before citations existed (``no_citation_data``); precision is also
skipped for an answer that cites no file (``no_citations``), where recall
is 0, and passage precision for one that cites no passage
(``no_cited_passages``, or ``no_citation_data`` when the section types were
not recorded).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from unified_kg_rag.domain.models import (
    EvaluationMetricType,
    EvaluationResult,
    SectionType,
)

__all__ = [
    "CITATION_METRICS",
    "CITATION_SCORING",
    "cited_section_types",
    "cited_source_ids",
    "citation_scores",
]

CITATION_METRICS = (
    EvaluationMetricType.CITATION_PRECISION,
    EvaluationMetricType.CITATION_RECALL,
    EvaluationMetricType.CITATION_PRECISION_PASSAGES,
)
# How citation_precision and citation_recall weigh a citation's files (the
# run manifest's ``citations.scoring``): ``file_weighted``, each file 1 / n
# of a citation of n files. Runs scored before credited a citation whole
# when any of its files matched.
CITATION_SCORING = "file_weighted"
# The section type of a passage citation.
_PASSAGE = SectionType.TEXT


def _file_names(citation: dict[str, Any]) -> list[str]:
    names = [
        str(d.get("file_name"))
        for d in citation.get("documents") or []
        if isinstance(d, dict) and d.get("file_name")
    ]
    if not names and citation.get("file_name"):
        names = [str(citation["file_name"])]
    return list(dict.fromkeys(names))


def cited_source_ids(raw_result: Any) -> list[list[str]] | None:
    """Per cited citation of an answer (``RAGOutput`` or its dict), the file
    names it stands for; ``None`` for a result without a citation list."""
    if hasattr(raw_result, "model_dump"):
        raw_result = raw_result.model_dump()
    if not isinstance(raw_result, dict):
        return None
    cited = raw_result.get("citations")
    if not isinstance(cited, list):
        return None
    return [_file_names(c) for c in cited if isinstance(c, dict) and c.get("cited")]


def cited_section_types(raw_result: Any) -> list[str] | None:
    """The section type of each citation ``cited_source_ids`` lists."""
    if hasattr(raw_result, "model_dump"):
        raw_result = raw_result.model_dump()
    if not isinstance(raw_result, dict):
        return None
    cited = raw_result.get("citations")
    if not isinstance(cited, list):
        return None
    return [
        str(c.get("section_type") or "")
        for c in cited
        if isinstance(c, dict) and c.get("cited")
    ]


def citation_scores(
    result: EvaluationResult,
    references: dict[str, str],
    source_keys: Callable[[str], frozenset[str]],
    *,
    enabled: bool = True,
) -> tuple[dict[EvaluationMetricType, float], dict[str, str]]:
    """(values, skip reasons) of :data:`CITATION_METRICS` for ``result``.

    ``references`` maps each reference's match key to its name, and
    ``source_keys`` is the retrieval evaluator's matching function, so
    citations and sources match references alike; ``enabled`` is whether
    the result's strategy cites (``citations_enabled``: ``search.citations``
    for that strategy)."""
    cited = result.cited_source_ids
    reason = (
        "no_reference_sources"
        if not references
        else (
            "citations_disabled"
            if not enabled
            else "no_citation_data" if cited is None else None
        )
    )
    if reason is not None:
        return {}, {m.value: reason for m in CITATION_METRICS}
    assert cited is not None
    types = result.cited_section_types
    if types is not None and len(types) != len(cited):
        types = None
    attributable = [
        (files, types[i] if types is not None else None)
        for i, files in enumerate(cited)
        if files
    ]
    recalled: dict[str, float] = {}
    precision: list[float] = []
    passages: list[float] = []
    for files, section_type in attributable:
        hits = [{ref for ref in references if ref in source_keys(f)} for f in files]
        weight = 1.0 / len(files)
        precision.append(sum(bool(h) for h in hits) * weight)
        for ref in set().union(*hits):
            recalled[ref] = max(recalled.get(ref, 0.0), weight)
        if section_type == _PASSAGE:
            passages.append(float(any(hits)))
    values = {
        EvaluationMetricType.CITATION_RECALL: sum(recalled.values()) / len(references)
    }
    skipped: dict[str, str] = {}
    if precision:
        values[EvaluationMetricType.CITATION_PRECISION] = sum(precision) / len(
            precision
        )
    else:
        skipped[EvaluationMetricType.CITATION_PRECISION.value] = "no_citations"
    if passages:
        values[EvaluationMetricType.CITATION_PRECISION_PASSAGES] = sum(passages) / len(
            passages
        )
    else:
        skipped[EvaluationMetricType.CITATION_PRECISION_PASSAGES.value] = (
            "no_citation_data" if types is None else "no_cited_passages"
        )
    return values, skipped
