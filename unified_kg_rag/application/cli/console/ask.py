# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Ask page's server side: one ``GraphRAGChain`` per loaded config, a
question answered as a stream of events, and its record for the dashboard.

Events (``kind`` plus fields): ``start`` (the query id, the trace id when
tracing is on), ``chunk`` (answer text as the answer model streams it,
``GraphRAGChain.ainvoke(on_answer_chunk=...)``, with the citation markers
it completes, read by ``CitationMarkerScanner`` so a marker split across
chunks counts once it closes; none for a strategy whose citations are off),
then ``result`` (the answer, its citations and ``cites``, whether its
strategy cites at all (``search.citations``), its sources with their
provenance and context markers, how ``auto`` routed, rerank status, usage,
cost and latency) or ``error``.

A follow-up carries the conversation itself (``RAGInput.history``): the
page keeps its turns, so a chain rebuilt after a settings save or an
ingestion, or a config without conversation memory, still continues it.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from unified_kg_rag.adapters.retrieval.strategy_registry import strategy_methodology
from unified_kg_rag.application.cli.console.jobs import namespace_of
from unified_kg_rag.application.cli.console.store import ConsoleStore
from unified_kg_rag.application.retrieval.citations import citations_enabled
from unified_kg_rag.application.retrieval.rag_chain import (
    ROUTING_KEY,
    ConversationTurn,
    GraphRAGChain,
    RAGInput,
    RAGOutput,
)
from unified_kg_rag.domain.ingestion.methodologies import stored_artifacts
from unified_kg_rag.domain.models import (
    Config,
    SearchStrategy,
    SearchType,
    validate_path_segment,
)
from unified_kg_rag.domain.models.methodology import methodologies_holding
from unified_kg_rag.domain.models.usage import LLMUsage
from unified_kg_rag.domain.retrieval.citations import CitationMarkerScanner
from unified_kg_rag.evaluation.comparison import answer_cost_usd
from unified_kg_rag.shared import get_logger, observability
from unified_kg_rag.shared.exceptions import (
    MethodologyNotIndexedError,
    classify,
    error_class_of,
)

logger = get_logger(__name__)

# Builds the chain the Ask page answers with (a test injects a fake).
ChainFactory = Callable[[Config], GraphRAGChain]

EXCERPT_CHARACTERS = 600
# A graph query result is its table: shown whole up to this.
TABLE_CHARACTERS = 6000
MAX_QUESTION_CHARACTERS = 4000
# A follow-up's history as the page sends it; the chain bounds it further
# (memory.max_messages_per_conversation, memory.max_history_tokens).
MAX_HISTORY_TURNS = 100
MAX_TURN_CHARACTERS = 100_000


class AskTurn(BaseModel):
    """One earlier message of the page's conversation."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=MAX_TURN_CHARACTERS)


class AskRequest(BaseModel):
    """One question from the Ask page."""

    query: str = Field(min_length=1, max_length=MAX_QUESTION_CHARACTERS)
    search_strategy: SearchStrategy = SearchStrategy.AUTO
    search_type: SearchType = SearchType.HYBRID
    top_k: int = Field(default=10, ge=1, le=100)
    history: list[AskTurn] | None = Field(default=None, max_length=MAX_HISTORY_TURNS)
    filters: dict[str, Any] | None = None
    suffix: str | None = Field(default=None, max_length=128)

    @field_validator("suffix")
    @classmethod
    def _safe_suffix(cls, value: str | None) -> str | None:
        # An index namespace names store indexes and cache paths.
        return None if value is None else validate_path_segment(value, "suffix")


def trace_url(trace_id: str | None, template: str | None, config: Config) -> str | None:
    """Where the query's trace opens: ``template`` with ``{trace_id}``, else
    Langfuse's trace page when the Langfuse backend is on."""
    if not trace_id:
        return None
    if template:
        return template.replace("{trace_id}", trace_id)
    if (
        config.observability.enabled
        and config.observability.backend.value == "langfuse"
    ):
        host = os.environ.get("LANGFUSE_HOST") or os.environ.get("LANGFUSE_BASE_URL")
        if host:
            return f"{host.rstrip('/')}/trace/{trace_id}"
    return None


def _listed(value: Any) -> list[str]:
    if isinstance(value, list | tuple | set):
        return [str(v) for v in value if v not in (None, "")]
    if isinstance(value, str) and value:
        return [value]
    return []


def source_view(rank: int, source: dict[str, Any]) -> dict[str, Any]:
    """One source as the Ask page shows it: what it is, where it comes
    from (file, chunk, text units, seed graph, IRI) and the graph ids it
    links to."""
    metadata: dict[str, Any] = source.get("metadata") or {}
    raw_attributes = metadata.get("attributes")
    attributes: dict[str, Any] = (
        raw_attributes if isinstance(raw_attributes, dict) else {}
    )
    kind = str(metadata.get("section_type") or "text")
    view: dict[str, Any] = {
        "rank": rank,
        "kind": kind,
        "id": source.get("source"),
        "score": source.get("score"),
        "excerpt": str(source.get("content") or "")[
            : TABLE_CHARACTERS if kind == "query_result" else EXCERPT_CHARACTERS
        ],
        "truncated": bool(metadata.get("truncated")),
        "document_ids": _listed(metadata.get("document_ids")),
        "file_name": attributes.get("file_name") or metadata.get("file_name"),
        "chunk_id": metadata.get("chunk_id"),
        "text_unit_ids": _listed(metadata.get("text_unit_ids")),
        "source_kinds": _listed(metadata.get("source_kinds")),
        "iri": metadata.get("iri") if isinstance(metadata.get("iri"), str) else None,
        "provenance": metadata.get("provenance"),
    }
    if kind == "entity":
        view.update(
            title=metadata.get("name"),
            entity_type=metadata.get("type"),
            node_ids=[source.get("source")],
        )
    elif kind == "relationship":
        view.update(
            title=f"{metadata.get('source_name')} → {metadata.get('target_name')}",
            relation=metadata.get("type"),
            node_ids=[metadata.get("source_id"), metadata.get("target_id")],
            edge_id=source.get("source"),
        )
    elif kind == "query_result":
        # The structured strategy's graph query: what it computed, how many
        # rows, whether the table is whole, and the entities it returned.
        rows = metadata.get("row_count")
        view.update(
            title=metadata.get("plan_description") or metadata.get("operation"),
            operation=metadata.get("operation"),
            row_count=rows if isinstance(rows, int) else None,
            exact=metadata.get("exact") is not False,
            backend=metadata.get("backend"),
            node_ids=_listed(metadata.get("entity_ids")),
        )
    elif kind == "community":
        view.update(
            title=metadata.get("name"), community_id=metadata.get("community_id")
        )
    else:
        view.update(title=view["file_name"] or source.get("source"))
    return view


def link_sources(sources: list[dict[str, Any]]) -> None:
    """Join the graph sources to the passages of the same answer: each
    entity, relationship or graph query result lists the ranks of the text sources it cites
    (``supported_by``) and how many cited passages the answer did not read
    (``unread_passages``)."""
    by_unit = {s["id"]: s["rank"] for s in sources if s["kind"] == "text" and s["id"]}
    for source in sources:
        units = source.get("text_unit_ids") or []
        if source["kind"] in ("entity", "relationship", "query_result") and units:
            source["supported_by"] = [by_unit[u] for u in units if u in by_unit]
            source["unread_passages"] = sum(1 for u in units if u not in by_unit)


def mark_sources(
    sources: list[dict[str, Any]], citations: list[dict[str, Any]]
) -> None:
    """Give each source the context marker it was shown under (``marker``,
    ``None`` for none) and each citation the rank of its source
    (``source_rank``, ``None`` for a section reported as no source, a
    synthesized summary): the answer's ``[n]`` links to the trail through
    them. A citation names its source by the source's id, in context order."""
    unclaimed: dict[Any, list[dict[str, Any]]] = {}
    for source in sources:
        source["marker"] = None
        unclaimed.setdefault(source["id"], []).append(source)
    for citation in citations:
        candidates = unclaimed.get(citation.get("source_id")) or []
        shown = candidates.pop(0) if candidates else None
        citation["source_rank"] = shown["rank"] if shown else None
        if shown is not None:
            shown["marker"] = citation["marker"]


class AskService:
    """The chain of the current config and the questions asked through it."""

    def __init__(
        self,
        load_config: Callable[[], Config],
        store: ConsoleStore,
        *,
        trace_url_template: str | None = None,
        chain_factory: ChainFactory | None = None,
    ) -> None:
        self.load_config = load_config
        self.store = store
        self.trace_url_template = trace_url_template
        self._chain_factory = chain_factory or (
            lambda config: GraphRAGChain(config=config)
        )
        self._chain: GraphRAGChain | None = None
        self._config: Config | None = None
        self._lock = asyncio.Lock()
        # Moved on by each reset: a chain whose build began before one was
        # built from the config the reset replaced, and is not kept.
        self._generation = 0
        # Chains a reset replaced, closed once their questions finish, and
        # the questions each chain is answering (by id()).
        self._retired: list[GraphRAGChain] = []
        self._in_flight: dict[int, int] = {}
        self._closing: set[asyncio.Task[None]] = set()

    async def chain(self) -> tuple[GraphRAGChain, Config]:
        async with self._lock:
            while self._chain is None:
                generation = self._generation
                config = self.load_config()
                # Built off the loop: it builds the model and store clients.
                built = await asyncio.to_thread(self._chain_factory, config)
                if generation != self._generation:
                    # Reset while it was built: from a replaced config.
                    self._retired.append(built)
                    self._close_idle()
                    continue
                self._chain, self._config = built, config
            assert self._config is not None
            return self._chain, self._config

    def reset(self) -> None:
        """Build the chain again at the next question (the config was
        saved, or an ingestion changed the index); the replaced chain is
        closed once the questions it is answering finish."""
        self._generation += 1
        if self._chain is not None:
            self._retired.append(self._chain)
            self._close_idle()
        self._chain = None
        self._config = None

    @contextlib.asynccontextmanager
    async def _using(self) -> AsyncIterator[tuple[GraphRAGChain, Config]]:
        """The current chain, counted as in use until the block ends."""
        chain, config = await self.chain()
        key = id(chain)
        self._in_flight[key] = self._in_flight.get(key, 0) + 1
        try:
            yield chain, config
        finally:
            self._in_flight[key] -= 1
            if not self._in_flight[key]:
                del self._in_flight[key]
            self._close_idle()

    def _close_idle(self) -> None:
        """Close the retired chains no question uses any more."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # closed by close()
        for chain in [c for c in self._retired if id(c) not in self._in_flight]:
            self._retired.remove(chain)
            task = loop.create_task(self._aclose(chain))
            self._closing.add(task)
            task.add_done_callback(self._closing.discard)

    @staticmethod
    async def _aclose(chain: GraphRAGChain) -> None:
        try:
            await chain.aclose()
        except Exception as e:  # noqa: BLE001 - a close must not fail anything
            logger.warning("Closing a replaced chain failed: %s", e)

    async def close(self) -> None:
        chains = [*self._retired, *([self._chain] if self._chain else [])]
        self._retired.clear()
        self._chain = None
        for chain in chains:
            await self._aclose(chain)
        if self._closing:
            await asyncio.gather(*self._closing, return_exceptions=True)

    async def strategies(self, suffix: str | None = None) -> list[dict[str, Any]]:
        """Every strategy, its methodology, whether ``auto`` routes to it and
        whether the namespace holds what it reads."""
        async with self._using() as (chain, config):
            return await self._strategies(chain, config, suffix)

    async def _strategies(
        self, chain: GraphRAGChain, config: Config, suffix: str | None
    ) -> list[dict[str, Any]]:
        routable = {s.value for s in config.search.auto_routable_strategies}
        views = []
        for strategy in SearchStrategy:
            available, reason = True, None
            try:
                await chain.acheck_methodology(strategy, suffix)
            except MethodologyNotIndexedError as e:
                available, reason = False, str(e)
            except Exception as e:  # noqa: BLE001 - shown, not fatal
                available, reason = False, str(e)
            methodology = None
            if strategy is not SearchStrategy.AUTO:
                with contextlib.suppress(Exception):
                    found = strategy_methodology(strategy, config)
                    methodology = found.value if found else None
            views.append(
                {
                    "name": strategy.value,
                    "methodology": methodology,
                    "routable": strategy.value in routable,
                    "available": available,
                    "reason": reason,
                }
            )
        return views

    async def held_methodologies(self, suffix: str | None = None) -> dict[str, Any]:
        """What the index namespace of ``suffix`` holds, from the build
        record ingestion wrote on it (``GraphRAGChain.aread_build_record``):
        ``recorded`` false (and nothing ``held``) when it has none that
        names its artifacts (never built, or built before methodology
        profiles). A store error raises."""
        async with self._using() as (chain, config):
            record = await chain.aread_build_record(suffix)
            namespace = namespace_of(config, suffix)
        artifacts = stored_artifacts(record)
        return {
            "namespace": namespace,
            "recorded": artifacts is not None,
            "held": (
                [m.value for m in methodologies_holding(artifacts)]
                if artifacts is not None
                else []
            ),
        }

    async def ask(self, request: AskRequest) -> AsyncIterator[dict[str, Any]]:
        """Answer ``request`` as events (see the module docstring); the
        question is recorded whatever the outcome."""
        query_id = uuid.uuid4().hex[:12]
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        started = time.perf_counter()
        record: dict[str, Any] = {
            "id": query_id,
            "created": time.time(),
            "question": request.query,
            "requested_strategy": request.search_strategy.value,
            "suffix": request.suffix,
            "outcome": "failed",
        }

        async def run() -> None:
            try:
                async with self._using() as (chain, config):
                    output, trace_id = await self._invoke(
                        chain, config, request, query_id, queue
                    )
                record["trace_id"] = trace_id
                result = self._result(
                    output, config, time.perf_counter() - started, request.suffix
                )
                result["trace_id"] = trace_id
                result["trace_url"] = trace_url(
                    record.get("trace_id"), self.trace_url_template, config
                )
                record.update(
                    strategy=result["strategy"],
                    latency_seconds=result["latency_seconds"],
                    outcome=(
                        "error"
                        if result["error"]
                        else (
                            "blocked"
                            if result["blocked"]
                            else ("abstained" if result["abstained"] else "answered")
                        )
                    ),
                    abstained=result["abstained"],
                    rerank_status=(result.get("rerank") or {}).get("status"),
                    usage=result.get("usage"),
                    cost_usd=result.get("cost_usd"),
                    error=result.get("error_message"),
                )
                queue.put_nowait({"kind": "result", **result})
            except asyncio.CancelledError:
                record.update(
                    outcome="cancelled", latency_seconds=time.perf_counter() - started
                )
                raise
            except Exception as e:  # noqa: BLE001 - reported to the page
                kind = _error_kind(e)
                if kind == "failure":
                    logger.exception("Console question %s failed", query_id)
                record.update(
                    error=str(e), latency_seconds=time.perf_counter() - started
                )
                queue.put_nowait(
                    {"kind": "error", "error_kind": kind, "message": str(e)}
                )
            finally:
                with contextlib.suppress(Exception):
                    self.store.record_query(record)
                queue.put_nowait(None)

        task = asyncio.create_task(run(), name=f"ask-{query_id}")
        try:
            while (event := await queue.get()) is not None:
                yield event
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task

    async def _invoke(
        self,
        chain: GraphRAGChain,
        config: Config,
        request: AskRequest,
        query_id: str,
        queue: asyncio.Queue[dict[str, Any] | None],
    ) -> tuple[RAGOutput, str | None]:
        """Answer ``request`` under the console's trace span, the start event
        and the answer's chunks put on ``queue``; the output and trace id."""
        with observability.span(
            "kg_rag.console.ask",
            {"kg_rag.search_strategy": request.search_strategy.value},
        ):
            ids = observability.current_trace_ids()
            trace_id = ids[0] if ids else None
            queue.put_nowait(
                {
                    "kind": "start",
                    "query_id": query_id,
                    "trace_id": trace_id,
                    "trace_url": trace_url(trace_id, self.trace_url_template, config),
                }
            )
            scanner = CitationMarkerScanner()
            # An auto query's route is known only with its result.
            scans = citations_enabled(config, request.search_strategy)
            output = await chain.ainvoke(
                RAGInput(
                    query=request.query,
                    search_strategy=request.search_strategy,
                    search_type=request.search_type,
                    top_k=request.top_k,
                    history=(
                        None
                        if request.history is None
                        else [
                            ConversationTurn(role=t.role, content=t.content)
                            for t in request.history
                        ]
                    ),
                    filters=request.filters,
                    suffix=request.suffix,
                ),
                on_answer_chunk=lambda text: queue.put_nowait(
                    {
                        "kind": "chunk",
                        "text": text,
                        "markers": scanner.feed(text) if scans else [],
                    }
                ),
            )
            if scans and (tail := scanner.finish()):
                queue.put_nowait({"kind": "chunk", "text": "", "markers": tail})
        return output, trace_id

    @staticmethod
    def _result(
        output: RAGOutput, config: Config, latency: float, suffix: str | None = None
    ) -> dict[str, Any]:
        metadata = output.metadata or {}
        sources = [source_view(i, s) for i, s in enumerate(output.sources, start=1)]
        link_sources(sources)
        citations = [c.model_dump(mode="json") for c in output.citations]
        mark_sources(sources, citations)
        usage_data = metadata.get("usage")
        cost: float | None = None
        unpriced: list[str] = []
        if isinstance(usage_data, dict):
            try:
                usd, missing = answer_cost_usd(
                    LLMUsage.model_validate(usage_data), config.evaluation.prices
                )
                cost, unpriced = usd, sorted(missing)
            except ValidationError:
                usage_data = None
        detail = (output.search_results.metadata or {}) if output.search_results else {}
        error = bool(metadata.get("error"))
        return {
            "answer": output.answer,
            "citations": citations,
            # Whether the answer's strategy cites (search.citations): the
            # page shows a [n] of an answer that does not as plain text.
            "cites": citations_enabled(config, metadata.get("search_strategy")),
            "unknown_citation_markers": metadata.get("unknown_citation_markers") or [],
            "strategy": metadata.get("search_strategy"),
            "routing": metadata.get(ROUTING_KEY),
            "abstained": bool(metadata.get("abstained")),
            "abstention_reason": metadata.get("abstention_reason"),
            "blocked": bool(metadata.get("blocked")),
            "guardrail": metadata.get("guardrail"),
            "error": error,
            # The class of a failure the chain absorbed (ignore_errors), as
            # an error event names a raised one (_error_kind).
            "error_kind": (
                error_class_of(metadata.get("error_class")).value if error else None
            ),
            "error_message": (
                str(detail.get("error")) if error and detail.get("error") else None
            ),
            "rerank": metadata.get("rerank"),
            "usage": usage_data,
            "cost_usd": cost,
            "unpriced_models": unpriced,
            "prices_configured": bool(config.evaluation.prices),
            "latency_seconds": latency,
            "processing_seconds": metadata.get("processing_time"),
            "total_results": metadata.get("total_results"),
            "context_sections_included": metadata.get("context_sections_included"),
            "context_sections_excluded": metadata.get("context_sections_excluded"),
            "processed_query": output.processed_query.model_dump(mode="json"),
            "sources": sources,
            "namespace": namespace_of(config, suffix),
        }


def _error_kind(error: Exception) -> str:
    """The page's error kind of ``error``: its ``ErrorClass`` value."""
    return classify(error).value
