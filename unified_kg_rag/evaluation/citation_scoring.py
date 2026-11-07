# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Citation precision and recall against ``reference_sources``.

The retrieval evaluator scores what the answer model *saw*; these score what
the answer *cites* (``RAGOutput.citations`` whose marker the answer uses,
docs/design.md §7.1). Each cited citation stands for the files of its
documents (a passage's one file; a graph section's resolved lineage), kept
on the result as ``EvaluationResult.cited_source_ids``:

- ``citation_precision``: cited citations that name a reference file /
  cited citations that name any file. A citation naming no file (a
  seed-only fact, an unresolved lineage) is left out of both counts.
- ``citation_recall``: distinct references named by a cited citation /
  distinct references.

Files match references as in the retrieval evaluator (``source_keys``).
Both are skipped without ``reference_sources``, with
``search.citations.enabled`` off (``citations_disabled``) and for a result
saved before citations existed (``no_citation_data``); precision is also
skipped for an answer that cites no file (``no_citations``), where recall
is 0.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from unified_kg_rag.domain.models import EvaluationMetricType, EvaluationResult

__all__ = [
    "CITATION_METRICS",
    "cited_source_ids",
    "citation_scores",
]

CITATION_METRICS = (
    EvaluationMetricType.CITATION_PRECISION,
    EvaluationMetricType.CITATION_RECALL,
)


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
    citations and sources match references alike; ``enabled`` is
    ``search.citations.enabled``."""
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
    attributable = [files for files in cited if files]
    matched: set[str] = set()
    correct = 0
    for files in attributable:
        keys = [source_keys(f) for f in files]
        hits = {ref for ref in references if any(ref in k for k in keys)}
        matched |= hits
        correct += bool(hits)
    values = {EvaluationMetricType.CITATION_RECALL: len(matched) / len(references)}
    skipped: dict[str, str] = {}
    if attributable:
        values[EvaluationMetricType.CITATION_PRECISION] = correct / len(attributable)
    else:
        skipped[EvaluationMetricType.CITATION_PRECISION.value] = "no_citations"
    return values, skipped
