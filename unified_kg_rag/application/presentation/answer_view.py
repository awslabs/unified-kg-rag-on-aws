# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One view of an answer for every surface.

:meth:`AnswerView.from_output` reads a ``RAGOutput`` once: its outcome
(strategy, abstention, guardrail, error), each source as a
:class:`SourceView` (what it is, its title, file, documents, lineage, the
entities it is about, its seed provenance, the ``[n]`` marker the answer
context gave it and whether the answer cites it) and the model-written
sections the answer may cite that are no source (``written``: global
search's key points, DRIFT's intermediate answers) and what the answer was given without
(:class:`QueryOutcome`: the degraded steps, the unresolved citation reads, the
class of an absorbed error). The console, the MCP ``kg_answer`` tool and
``run-rag`` map it to their own shapes, cutting the text to their own lengths;
none of them reads ``RAGOutput.sources`` or the outcome's metadata keys itself.
A search without an answer reports its :class:`QueryOutcome` too.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from unified_kg_rag.application.retrieval.rag_chain import (
    ROUTING_KEY,
    ProcessedQuery,
    RAGOutput,
    SearchOutput,
)
from unified_kg_rag.domain.models import Citation, RetrievalResult, Source
from unified_kg_rag.domain.retrieval import source_fields
from unified_kg_rag.domain.retrieval.degradation import (
    Degradation,
    DegradationReport,
)
from unified_kg_rag.domain.retrieval.source_fields import EntityRef
from unified_kg_rag.domain.retrieval.source_fields import text_value as _text
from unified_kg_rag.shared.exceptions import ERROR_CLASS_KEY, error_class_of

__all__ = ["AnswerView", "EntityRef", "QueryOutcome", "SourceView"]


class QueryOutcome(BaseModel):
    """What a query's answer (or search) was given without, and the class
    of the error it absorbed, as every surface reports them."""

    degraded: list[Degradation] = Field(
        default_factory=list,
        description="The steps it answered without (a partial answer): each "
        "step, reason and error type",
    )
    degraded_error: str | None = Field(
        default=None, description="The error type of the first that names one"
    )
    citations_unresolved: list[Degradation] = Field(
        default_factory=list,
        description="The lineage reads that failed: its sources name fewer "
        "documents; the answer is not degraded",
    )
    error_class: str | None = Field(
        default=None,
        description="The class of the error answer's failure (ErrorClass: "
        "invalid_filter, timeout, ...), with processing.ignore_errors",
    )
    error_type: str | None = Field(
        default=None, description="The class name of the error it absorbed"
    )
    error_message: str | None = Field(
        default=None, description="The error answer's message"
    )

    @classmethod
    def of(
        cls,
        metadata: dict[str, Any] | None,
        search_metadata: dict[str, Any] | None = None,
    ) -> QueryOutcome:
        """The outcome a result's ``metadata`` reports (and, for the error
        answer's message, its ``search_results.metadata``)."""
        metadata = metadata or {}
        report = DegradationReport.of(metadata)
        error = bool(metadata.get("error"))
        message = (search_metadata or {}).get("error")
        return cls(
            degraded=report.degraded,
            degraded_error=report.degraded_error,
            citations_unresolved=report.citations_unresolved,
            error_class=(
                error_class_of(metadata.get(ERROR_CLASS_KEY)).value if error else None
            ),
            error_type=_text(metadata.get("error_type")),
            error_message=str(message) if error and message else None,
        )

    @classmethod
    def of_output(cls, output: RAGOutput | SearchOutput) -> QueryOutcome:
        """The outcome of an answer or a search without one."""
        return cls.of(output.metadata, output.search_results.metadata)

    @property
    def partial(self) -> bool:
        """It answered without a step (and is no error answer)."""
        return bool(self.degraded) and self.error_class is None

    def outcome_fields(self) -> dict[str, Any]:
        """The outcome's fields, as a surface's JSON document holds them (its
        error's type and message are the document's own failure fields)."""
        return self.model_dump(
            mode="json",
            include=set(QueryOutcome.model_fields) - {"error_type", "error_message"},
        )


class SourceView(BaseModel):
    """One source (or retrieved result) as every surface reads it."""

    rank: int = Field(description="1 for the first source")
    kind: str = Field(description="Its section type: text, entity, relationship, ...")
    id: str | None = Field(default=None, description="The stored item's id")
    title: str | None = Field(
        default=None,
        description="An entity's or report's name, a relationship's 'A -> B', "
        "a graph query's plan description, a passage's file name",
    )
    text: str = Field(default="", description="The text the answer model saw, whole")
    score: float | None = Field(default=None, description="The relevance score")
    truncated: bool = Field(
        default=False, description="The context budget cut the text"
    )
    chunk_id: str | None = Field(default=None, description="A passage's chunk id")
    type: str | None = Field(
        default=None, description="An entity's or relationship's type"
    )
    document_ids: list[str] = Field(default_factory=list)
    file_name: str | None = Field(
        default=None, description="A passage's file name (never a path)"
    )
    documents: list[str] = Field(
        default_factory=list, description="The file names of its documents"
    )
    text_unit_ids: list[str] = Field(default_factory=list)
    source_kinds: list[str] = Field(default_factory=list)
    iri: str | None = None
    provenance: dict[str, Any] | None = Field(
        default=None, description="Seed-graph provenance (kind, seed_graphs, iri)"
    )
    synthesized: bool = Field(
        default=False, description="Model-written, not retrieved from the corpus"
    )
    entities: list[EntityRef] = Field(
        default_factory=list,
        description="The entity itself, a relationship's or claim's endpoints, "
        "a graph query result's entities",
    )
    marker: int | None = Field(
        default=None, description="The [n] the answer context gave it"
    )
    cited: bool | None = Field(
        default=None, description="Whether the answer cites [marker]"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Its stored fields, for a surface's kind-specific extras",
    )

    @classmethod
    def of(
        cls,
        rank: int,
        kind: str,
        content: str,
        *,
        source: str | None,
        score: float | None,
        metadata: dict[str, Any],
        chunk_id: str | None = None,
        truncated: bool = False,
    ) -> SourceView:
        """The view of one item from its content and stored metadata."""
        item_id = _text(metadata.get("id")) or source
        if kind == "text" and chunk_id is None:
            chunk_id = _text(metadata.get("chunk_id")) or item_id
        name = source_fields.file_name(metadata) if kind == "text" else None
        iri = metadata.get("iri")
        return cls(
            rank=rank,
            kind=kind,
            id=item_id,
            title=source_fields.title(kind, metadata),
            text=content or "",
            score=score,
            truncated=truncated or bool(metadata.get("truncated")),
            chunk_id=chunk_id,
            type=(
                _text(metadata.get("type"))
                if kind in ("entity", "relationship")
                else None
            ),
            document_ids=source_fields.document_ids(metadata),
            file_name=name,
            documents=[name] if name else [],
            text_unit_ids=source_fields.listed(metadata, "text_unit_ids"),
            source_kinds=source_fields.listed(metadata, "source_kinds"),
            iri=iri if isinstance(iri, str) and iri else None,
            # A reported source carries what the chain made of it.
            provenance=(
                provenance
                if isinstance(provenance := metadata.get("provenance"), dict)
                else source_fields.seed_provenance(metadata)
            ),
            synthesized=bool(metadata.get("synthesized")),
            entities=source_fields.entity_refs(kind, metadata, item_id),
            metadata=metadata,
        )

    @classmethod
    def of_source(cls, rank: int, source: Source) -> SourceView:
        """The view of an answer's source."""
        metadata = source.get("metadata") or {}
        return cls.of(
            rank,
            str(metadata.get("section_type") or "text"),
            str(source.get("content") or ""),
            source=_text(source.get("source")),
            score=source.get("score"),
            metadata=metadata,
            chunk_id=_text(metadata.get("chunk_id")),
        )

    @classmethod
    def of_result(cls, rank: int, result: RetrievalResult) -> SourceView:
        """The view of a retrieved result (a search without an answer)."""
        return cls.of(
            rank,
            result.retriever_type,
            result.content,
            source=result.source,
            score=result.score,
            metadata=result.metadata or {},
            chunk_id=result.chunk_id,
        )


def _mark(sources: list[SourceView], citations: list[Citation]) -> dict[int, int]:
    """Give each source the marker of the citation that names it (by its id,
    else its chunk id; a source shown twice takes the next citation of the
    same id) and whether the answer cites it; each matched marker's source
    rank."""
    unclaimed: dict[str, list[SourceView]] = {}
    for view in sources:
        for key in dict.fromkeys(k for k in (view.id, view.chunk_id) if k):
            unclaimed.setdefault(key, []).append(view)
    ranks: dict[int, int] = {}
    for citation in citations:
        for wanted in (citation.source_id, citation.chunk_id):
            candidates = [
                v for v in unclaimed.get(wanted or "", []) if v.marker is None
            ]
            if candidates:
                shown = candidates[0]
                shown.marker, shown.cited = citation.marker, citation.cited
                ranks[citation.marker] = shown.rank
                break
    return ranks


def _written(output: RAGOutput, matched: set[int]) -> list[SourceView]:
    """A view of each citation no source is: a model-written section the
    answer model was given and may cite (global search's key point, DRIFT's
    intermediate answer), with its screened text, the documents its
    citation names (resolved with the caller's filters) and their text
    units only, not its whole lineage."""
    written = {
        str(r.source): r
        for r in output.search_results.results
        if (r.metadata or {}).get("synthesized") and r.source is not None
    }
    views: list[SourceView] = []
    for citation in output.citations:
        result = written.get(citation.source_id or "")
        if citation.marker in matched or result is None:
            continue
        metadata = result.metadata or {}
        kind = (
            "key_point"
            if "key_point_rank" in metadata
            else (
                "intermediate_answer"
                if "intermediate_answer_rank" in metadata
                else citation.section_type
            )
        )
        views.append(
            SourceView(
                rank=len(views) + 1,
                kind=kind,
                id=citation.source_id,
                text=result.content,
                score=result.score,
                synthesized=True,
                document_ids=[citation.document_id] if citation.document_id else [],
                file_name=citation.file_name,
                documents=list(
                    dict.fromkeys(
                        d.file_name for d in citation.documents if d.file_name
                    )
                ),
                text_unit_ids=[
                    unit for d in citation.documents for unit in d.text_unit_ids
                ],
                marker=citation.marker,
                cited=citation.cited,
            )
        )
    return views


class AnswerView(QueryOutcome):
    """An answer as every surface shows it (:meth:`from_output`)."""

    answer: str
    strategy: str | None = Field(
        default=None, description="The strategy that ran (auto resolved)"
    )
    routing: dict[str, Any] | None = Field(
        default=None, description="How auto chose it"
    )
    abstained: bool = False
    abstention_reason: str | None = None
    blocked: bool = Field(default=False, description="The guardrail blocked the query")
    guardrail: list[dict[str, Any]] = Field(default_factory=list)
    error: bool = Field(
        default=False, description="The chain's error answer (ignore_errors)"
    )
    rerank: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    processing_seconds: float | None = None
    total_results: int | None = None
    context_sections_included: int | None = None
    context_sections_excluded: int | None = None
    searched_query: str = Field(
        description="The query searched (rewritten or translated)"
    )
    processed_query: ProcessedQuery
    citations: list[Citation] = Field(default_factory=list)
    citation_ranks: dict[int, int] = Field(
        default_factory=dict,
        description="Each matched citation marker's source rank",
    )
    unknown_citation_markers: list[int] = Field(default_factory=list)
    sources: list[SourceView] = Field(default_factory=list)
    written: list[SourceView] = Field(
        default_factory=list,
        description="The model-written sections the answer may cite, no source",
    )

    @classmethod
    def from_output(cls, output: RAGOutput) -> AnswerView:
        metadata = output.metadata or {}
        sources = [SourceView.of_source(i, s) for i, s in enumerate(output.sources, 1)]
        ranks = _mark(sources, output.citations)
        rerank, usage, routing = (
            metadata.get("rerank"),
            metadata.get("usage"),
            metadata.get(ROUTING_KEY),
        )
        return cls(
            **QueryOutcome.of_output(output).model_dump(),
            answer=output.answer,
            strategy=_text(metadata.get("search_strategy")),
            routing=routing if isinstance(routing, dict) else None,
            abstained=bool(metadata.get("abstained")),
            abstention_reason=_text(metadata.get("abstention_reason")),
            blocked=bool(metadata.get("blocked")),
            guardrail=list(metadata.get("guardrail") or []),
            error=bool(metadata.get("error")),
            rerank=rerank if isinstance(rerank, dict) else None,
            usage=usage if isinstance(usage, dict) else None,
            processing_seconds=metadata.get("processing_time"),
            total_results=metadata.get("total_results"),
            context_sections_included=metadata.get("context_sections_included"),
            context_sections_excluded=metadata.get("context_sections_excluded"),
            searched_query=output.processed_query.final_query,
            processed_query=output.processed_query,
            citations=list(output.citations),
            citation_ranks=ranks,
            unknown_citation_markers=list(
                metadata.get("unknown_citation_markers") or []
            ),
            sources=sources,
            written=_written(output, set(ranks)),
        )
