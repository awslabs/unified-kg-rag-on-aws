# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``RAGOutput.citations``: what each numbered context section rests on.

Built by :class:`~unified_kg_rag.application.retrieval.rag_chain.GraphRAGChain`
from the reported source entry of each numbered section
(``GraphRAGChain._source_entry``: the section's metadata with its document
ids, chunk id and seed provenance). A passage names its own text unit,
document and pages. A graph section (entity, relationship, claim, report,
summary) names its ``text_unit_ids`` lineage and, through
:func:`aresolve_documents`, the documents of up to
``search.citations.lineage_units`` of those units: the ones the context
holds as passages first, the rest read in batched fetches by id, in pages
until each citation has that many (at most :data:`MAX_LINEAGE_PAGES`). A
unit the fetch does not return (excluded by the caller's filters, or
removed) is never named. Whether a query's answer cites at all is
:func:`citations_enabled` (``search.citations``, per strategy). See
docs/design.md §7.1.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from unified_kg_rag.adapters.retrieval.strategy_registry import citations_enabled
from unified_kg_rag.domain.ingestion.report_references import (
    FINDINGS_TEXT_UNITS_FIELD,
)
from unified_kg_rag.domain.ingestion.seed_graph import SEED_UNIT_PREFIX
from unified_kg_rag.domain.models import (
    Citation,
    CitedDocument,
    SectionType,
    Source,
)
from unified_kg_rag.domain.retrieval import source_fields
from unified_kg_rag.domain.retrieval.citations import cited_markers

__all__ = [
    "MAX_LINEAGE_PAGES",
    "aresolve_documents",
    "citation_from_entry",
    "citations_enabled",
    "mark_cited",
]

_TEXT = SectionType.TEXT

# Reads of a query's lineage units at most: each asks of a citation still
# short twice the units the one before did, so a citation reads at most
# ``lineage_units * (2**MAX_LINEAGE_PAGES - 1)`` of its units (150 with the
# default 10). Uncapped, a filter excluding most of a hub's units read its
# whole lineage.
MAX_LINEAGE_PAGES = 4


@dataclass(frozen=True)
class _Unit:
    """What one text unit says about its document."""

    unit_id: str
    document_id: str | None
    file_name: str | None
    file_path: str | None
    page: int | None
    page_end: int | None


def _first_text(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, list) and value:
            value = value[0]
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _page(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and value >= 1:
        return int(value)
    return None


def unit_from_metadata(metadata: dict[str, Any], fallback_id: str | None) -> _Unit:
    """What a stored text unit (a hit's or source's metadata) says about its
    document."""
    attributes = metadata.get("attributes")
    attributes = attributes if isinstance(attributes, dict) else {}
    unit_id = _first_text(metadata.get("id"), fallback_id) or ""
    page = _page(attributes.get("page_start"))
    page_end = _page(attributes.get("page_end"))
    documents = source_fields.document_ids(metadata)
    return _Unit(
        unit_id=unit_id,
        document_id=documents[0] if documents else None,
        file_name=source_fields.file_name(metadata),
        file_path=source_fields.relative_path(metadata),
        page=page,
        page_end=page_end if page is not None and page_end is not None else page,
    )


def citation_from_entry(marker: int, entry: Source | Mapping[str, Any]) -> Citation:
    """The citation of a numbered section from its reported source entry.

    A passage's citation is complete; a graph section's names its lineage,
    its documents filled in by :func:`aresolve_documents`."""
    metadata = entry.get("metadata") or {}
    section_type = SectionType.of(metadata.get("section_type"))
    provenance = metadata.get("provenance")
    provenance = dict(provenance) if isinstance(provenance, dict) else None
    iri = _first_text((provenance or {}).get("iri"), metadata.get("iri"))
    citation = Citation(
        marker=marker,
        section_type=section_type,
        source_id=_first_text(entry.get("source"), metadata.get("source_id")),
        title=(
            None
            if section_type == _TEXT
            else source_fields.title(section_type, metadata)
        ),
        provenance=provenance,
        iri=iri,
    )
    if section_type == _TEXT:
        unit = unit_from_metadata(metadata, _first_text(metadata.get("chunk_id")))
        citation.chunk_id = unit.unit_id or None
        citation.text_unit_ids = [unit.unit_id] if unit.unit_id else []
        citation.title = unit.file_name
        _attach(citation, [unit])
        return citation
    # A report whose findings cite stored records rests on their units
    # (the strategy's ``findings_text_unit_ids``), not on its community's.
    findings = source_fields.listed(metadata, FINDINGS_TEXT_UNITS_FIELD)
    citation.text_unit_ids = findings or source_fields.listed(metadata, "text_unit_ids")
    if not citation.text_unit_ids and (
        documents := source_fields.listed(metadata, "document_ids")
    ):
        # A report holds its documents' ids beside its lineage; with no
        # unit to name their files, the ids are still what it rests on.
        citation.documents = [CitedDocument(document_id=d) for d in documents]
        citation.document_id = documents[0]
    return citation


def _attach(citation: Citation, units: Iterable[_Unit]) -> None:
    """Set ``citation``'s documents from ``units`` (its lineage, in order):
    one per document, the one most units belong to first (then the first
    named), its file and pages from those units."""
    # A unit indexed before its document id was recorded with it is grouped
    # by its file.
    by_document: dict[str, list[_Unit]] = {}
    for unit in units:
        if key := unit.document_id or unit.file_path or unit.file_name:
            by_document.setdefault(key, []).append(unit)
    if not by_document:
        return
    ranked = sorted(by_document.values(), key=lambda held: -len(held))
    documents: list[CitedDocument] = []
    for held in ranked:
        pages = [u.page for u in held if u.page is not None]
        ends = [u.page_end for u in held if u.page_end is not None]
        documents.append(
            CitedDocument(
                document_id=held[0].document_id,
                file_name=next((u.file_name for u in held if u.file_name), None),
                file_path=next((u.file_path for u in held if u.file_path), None),
                page=min(pages) if pages else None,
                page_end=max(ends) if ends else None,
                text_unit_ids=[u.unit_id for u in held if u.unit_id],
            )
        )
    citation.documents = documents
    main = documents[0]
    citation.document_id = main.document_id
    citation.file_name = main.file_name
    citation.file_path = main.file_path
    citation.page = main.page
    citation.page_end = main.page_end


def _lineage(citation: Citation, known: dict[str, _Unit]) -> list[str]:
    """The document units of a graph citation's lineage, the ones the
    context holds first, then in lineage order. Seed units are virtual
    documents: their provenance is the seed graph's."""
    units = [u for u in citation.text_unit_ids if not u.startswith(SEED_UNIT_PREFIX)]
    held = [u for u in units if u in known]
    rest = [u for u in units if u not in known]
    return list(dict.fromkeys(held + rest))


async def aresolve_documents(
    citations: list[Citation],
    fetch: Callable[[list[str]], Awaitable[list[dict[str, Any]]]],
    limit: int,
) -> None:
    """Fill in the documents of the graph citations from their lineage: up
    to ``limit`` units of each that the caller may see.

    The passages among ``citations`` say what their units hold; every other
    unit wanted is read once with ``fetch`` (unit ids -> the stored units'
    metadata, as the caller's filters and suffix allow). A unit the fetch
    does not return is never named, so the units are read in pages until
    each citation has ``limit`` of them or its lineage is read: a filter
    that excluded the first ``limit`` units of a lineage left it naming no
    document though later ones pass. A page asks of each citation still
    short the units it lacks, times two to the number of pages before, so
    a lineage of ``n`` units takes at most about ``log2(n)`` reads, and no
    more than :data:`MAX_LINEAGE_PAGES`: a citation whose units read by then
    pass none names no document. Raises what ``fetch`` raises, leaving the
    citations as they were."""
    known: dict[str, _Unit] = {}
    for citation in citations:
        if citation.section_type == _TEXT and citation.chunk_id:
            known[citation.chunk_id] = _Unit(
                unit_id=citation.chunk_id,
                document_id=citation.document_id,
                file_name=citation.file_name,
                file_path=citation.file_path,
                page=citation.page,
                page_end=citation.page_end,
            )
    graph = [c for c in citations if c.section_type != _TEXT and c.text_unit_ids]
    lineages = {c.marker: _lineage(c, known) for c in graph}
    absent: set[str] = set()

    def chosen(marker: int) -> list[str]:
        return [u for u in lineages[marker] if u in known][:limit]

    scale = 1
    for _ in range(MAX_LINEAGE_PAGES):
        wanted: list[str] = []
        for citation in graph:
            short = limit - len(chosen(citation.marker))
            if short <= 0:
                continue
            unread = [
                u
                for u in lineages[citation.marker]
                if u not in known and u not in absent
            ]
            wanted.extend(unread[: short * scale])
        missing = list(dict.fromkeys(wanted))
        if not missing:
            break
        asked = set(missing)
        for metadata in await fetch(missing):
            unit = unit_from_metadata(metadata, None)
            if unit.unit_id in asked:
                known[unit.unit_id] = unit
        absent.update(u for u in missing if u not in known)
        scale *= 2
    for citation in graph:
        _attach(citation, [known[u] for u in chosen(citation.marker)])


def mark_cited(
    citations: list[Citation], answer: str
) -> tuple[list[Citation], list[int]]:
    """``citations`` with ``cited`` set from the markers ``answer`` uses, and
    the numbers it uses that are no citation's (never made citations)."""
    markers = {c.marker for c in citations}
    used = cited_markers(answer)
    cited = {n for n in used if n in markers}
    marked = [
        (
            c
            if c.cited == (c.marker in cited)
            else c.model_copy(update={"cited": not c.cited})
        )
        for c in citations
    ]
    return marked, [n for n in used if n not in markers]
