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
holds as passages first, the rest read in one batched fetch by id. A unit
the fetch does not return (excluded by the caller's filters, or removed) is
never named. See docs/design.md §7.1.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from unified_kg_rag.domain.ingestion.seed_graph import SEED_UNIT_PREFIX
from unified_kg_rag.domain.models import Citation, CitedDocument
from unified_kg_rag.domain.retrieval.citations import cited_markers

__all__ = [
    "aresolve_documents",
    "citation_from_entry",
    "mark_cited",
]

# Section type of a passage (``SectionType.TEXT``).
_TEXT = "text"


@dataclass(frozen=True)
class _Unit:
    """What one text unit says about its document."""

    unit_id: str
    document_id: str | None
    file_name: str | None
    file_path: str | None
    page: int | None
    page_end: int | None


@dataclass
class _Draft:
    """A citation's documents while they are gathered, in unit order."""

    units: list[_Unit] = field(default_factory=list)


def _ids(value: Any) -> list[str]:
    """A stored id list as strings: a graph store may return a list as its
    JSON string, or one id as the value."""
    if isinstance(value, str) and value.startswith("["):
        try:
            value = json.loads(value)
        except ValueError:
            return [value]
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list | tuple | set):
        return [str(v) for v in value if v is not None and v != ""]
    return []


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


def safe_relative_path(value: Any) -> str | None:
    """``value`` as a path relative to the corpus root, or ``None`` when it
    is not one (absolute, a drive or URL, or climbing out with ``..``): a
    citation never names where the corpus lived on the indexing host."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = value.strip().replace("\\", "/")
    if path.startswith("/") or ":" in path:
        return None
    parts = [p for p in path.split("/") if p not in ("", ".")]
    if not parts or ".." in parts:
        return None
    return "/".join(parts)


def unit_from_metadata(metadata: dict[str, Any], fallback_id: str | None) -> _Unit:
    """What a stored text unit (a hit's or source's metadata) says about its
    document."""
    attributes = metadata.get("attributes")
    attributes = attributes if isinstance(attributes, dict) else {}
    unit_id = _first_text(metadata.get("id"), fallback_id) or ""
    file_name = _first_text(attributes.get("file_name"), metadata.get("file_name"))
    if file_name is not None:
        file_name = file_name.replace("\\", "/").rsplit("/", 1)[-1]
    page = _page(attributes.get("page_start"))
    page_end = _page(attributes.get("page_end"))
    return _Unit(
        unit_id=unit_id,
        document_id=_first_text(
            metadata.get("document_ids"),
            metadata.get("document_id"),
            attributes.get("document_id"),
        ),
        file_name=file_name,
        file_path=safe_relative_path(attributes.get("relative_path")),
        page=page,
        page_end=page_end if page is not None and page_end is not None else page,
    )


def _title(section_type: str, metadata: dict[str, Any]) -> str | None:
    if section_type == _TEXT:
        return None
    if section_type == "relationship":
        source = _first_text(metadata.get("source_name"), metadata.get("source"))
        target = _first_text(metadata.get("target_name"), metadata.get("target"))
        if source and target:
            return f"{source} -> {target}"
    return _first_text(metadata.get("title"), metadata.get("name"))


def citation_from_entry(marker: int, entry: dict[str, Any]) -> Citation:
    """The citation of a numbered section from its reported source entry.

    A passage's citation is complete; a graph section's names its lineage,
    its documents filled in by :func:`aresolve_documents`."""
    metadata = entry.get("metadata") or {}
    section_type = str(metadata.get("section_type") or "general")
    provenance = metadata.get("provenance")
    provenance = dict(provenance) if isinstance(provenance, dict) else None
    iri = _first_text((provenance or {}).get("iri"), metadata.get("iri"))
    citation = Citation(
        marker=marker,
        section_type=section_type,
        source_id=_first_text(entry.get("source"), metadata.get("source_id")),
        title=_title(section_type, metadata),
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
    citation.text_unit_ids = _ids(metadata.get("text_unit_ids"))
    if not citation.text_unit_ids and (documents := _ids(metadata.get("document_ids"))):
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


def _lineage(citation: Citation, known: dict[str, _Unit], limit: int) -> list[str]:
    """Up to ``limit`` document units of a graph citation's lineage, the
    ones the context holds first, then in lineage order. Seed units are
    virtual documents: their provenance is the seed graph's."""
    units = [u for u in citation.text_unit_ids if not u.startswith(SEED_UNIT_PREFIX)]
    held = [u for u in units if u in known]
    rest = [u for u in units if u not in known]
    return list(dict.fromkeys(held + rest))[:limit]


async def aresolve_documents(
    citations: list[Citation],
    fetch: Callable[[list[str]], Awaitable[list[dict[str, Any]]]],
    limit: int,
) -> None:
    """Fill in the documents of the graph citations from their lineage.

    The passages among ``citations`` say what their units hold; every other
    unit wanted is read once with ``fetch`` (unit ids -> the stored units'
    metadata, as the caller's filters and suffix allow). Raises what
    ``fetch`` raises, leaving the citations as they were."""
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
    wanted = {c.marker: _lineage(c, known, limit) for c in graph}
    missing = list(
        dict.fromkeys(u for ids in wanted.values() for u in ids if u not in known)
    )
    if missing:
        for metadata in await fetch(missing):
            unit = unit_from_metadata(metadata, None)
            if unit.unit_id in missing:
                known[unit.unit_id] = unit
    for citation in graph:
        _attach(citation, [known[u] for u in wanted[citation.marker] if u in known])


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
