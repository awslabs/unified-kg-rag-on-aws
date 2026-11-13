# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured graph queries as a strategy: ``structured`` (opt-in;
docs/design.md §6.6, guide docs/guides/structured-queries.md).

One planner call (``models.roles.graph_query``) writes the question as a
``GraphQueryPlan`` over the queried graph's schema profile; code validates
it (one repair call with the refusal), links the names it mentions through
the hybrid entity search, and the graph store (``GraphQueryPort``) runs it
read-only within the query's suffix. The result table, the entities and
paths it names and the passages they cite are fused through the hybrid
scorer and answered with an instruction to use the result as computed.
Without a structured answer the query runs ``search.structured.
fallback_strategy`` (``local``) and records why; ``mode: augment`` runs the
fallback beside the graph query and fuses both.

Fusion buckets (search.fusion.fusion_weights): ``structured_query_result``,
``structured_entities``, ``structured_relationships``,
``structured_text_units`` (the units the returned entities cite, by
citations), ``structured_text_hits`` (a hybrid text search of the question
among them) and, under augment, ``structured_fallback``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, cast

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig
from pydantic import ValidationError

from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
    entity_content,
    record_degradation,
    units_by_support,
)
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    QueryInput,
    RunRecord,
    get_strategy_spec,
    query_inputs,
    register_strategy,
    serves,
)
from unified_kg_rag.adapters.store_errors import is_fatal_retrieval_error
from unified_kg_rag.domain.ingestion.cache_keys import index_language
from unified_kg_rag.domain.ingestion.seed_graph import (
    DOCUMENT_SOURCE_KIND,
    SEED_SOURCE_KIND,
    is_seed_unit,
)
from unified_kg_rag.domain.models import (
    GENERAL_FALLBACKS,
    Config,
    Constants,
    IndexArtifact,
    Methodology,
    ModelRole,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
    SectionType,
)
from unified_kg_rag.domain.models.config import language_name
from unified_kg_rag.domain.models.graph_query import (
    GraphProfile,
    GraphQueryPlan,
    GraphQueryRequest,
    GraphQueryResult,
    PathRow,
    QueryAnswer,
    ValidatedPlan,
)
from unified_kg_rag.domain.models.schema import ExtractionSchema
from unified_kg_rag.domain.prompts import GraphQueryPlanPrompt
from unified_kg_rag.domain.prompts.graph_query import REPAIR_FEEDBACK, schema_text
from unified_kg_rag.domain.retrieval.abstention import (
    CONTEXT_SUFFICIENT_KEY,
    INSUFFICIENCY_REASON_KEY,
)
from unified_kg_rag.domain.retrieval.degradation import DegradationStep
from unified_kg_rag.domain.retrieval.filters import document_filters
from unified_kg_rag.domain.retrieval.graph_query import (
    FILTERS_KEY,
    Operation,
    aggregate_bindings,
    answer_from_result,
    answer_holders,
    answer_vertex_ids,
    build_profile,
    describe_plan,
    evidence_vertex_ids,
    hydrate_keys,
    literal_conditions,
    matching_bindings,
    merge_declared,
    plan_operation,
    render_answer,
    validate_plan,
    value_check_plan,
    with_condition,
)
from unified_kg_rag.domain.retrieval.index_prefixes import held_artifacts_of
from unified_kg_rag.domain.retrieval.linking import choose_links, hit_name
from unified_kg_rag.domain.retrieval.source_fields import LIST_FIELDS, seed_provenance
from unified_kg_rag.ports.graph_query import GraphQueryPort
from unified_kg_rag.shared import get_logger, observability
from unified_kg_rag.shared.exceptions import (
    ConfigurationError,
    GraphQueryCompileError,
    GraphQueryExecutionError,
    InvalidGraphQueryError,
    ModelOutputError,
)
from unified_kg_rag.shared.logging import log_failure
from unified_kg_rag.shared.usage import usage_scope
from unified_kg_rag.shared.utils import parse_model_object, text_digest
from unified_kg_rag.shared.utils.concurrency import gather_or_cancel

logger = get_logger(__name__)

# The fusion buckets this strategy emits (search.fusion.fusion_weights).
QUERY_RESULT_BUCKET = "structured_query_result"
ENTITIES_BUCKET = "structured_entities"
RELATIONSHIPS_BUCKET = "structured_relationships"
TEXT_UNITS_BUCKET = "structured_text_units"
TEXT_HITS_BUCKET = "structured_text_hits"
FALLBACK_BUCKET = "structured_fallback"

# metadata.structured.status
ANSWERED, FALLBACK, FAILED = "answered", "fallback", "failed"
# The fallback reasons the query's degradation ledger records: a failed
# planner or store call, a planner answer no repair made readable JSON, and
# a compiler refusal. A designed fallback (a plan the checks still refuse
# after the repair, an absent name, an unknown value, an unanswerable
# question) is no failure: metadata.structured.reason alone keeps it.
_DEGRADING = frozenset(
    {"planner_error", "unreadable_plan", "execution_error", "timeout", "compile_error"}
)
# What _parsed says of an answer that holds no JSON object.
_UNREADABLE = "plan: the answer is not one JSON object"
# What a fallback gives way to when the namespace does not serve it.
# The JSON keys of a plan, to tell it from an example the model echoes.
_PLAN_KEYS = ("nodes", "returns", "unanswerable")
# Suffixes whose profile a strategy keeps at most.
_MAX_PROFILES = 256


@dataclass
class _Run:
    """What one query's structured path did (``metadata.structured``)."""

    mode: str
    status: str = ANSWERED
    reason: str | None = None
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    plan: dict[str, Any] | None = None
    description: str | None = None
    operation: str | None = None
    linked: dict[str, list[str]] = field(default_factory=dict)
    linked_names: dict[str, list[str]] = field(default_factory=dict)
    repairs: int = 0
    row_count: int = 0
    truncated: bool = False
    exact: bool = False
    backend: str | None = None
    compiled_digest: str | None = None
    fallback_strategy: str | None = None
    schema_cut: bool = False
    elapsed_ms: dict[str, float] = field(default_factory=dict)

    def fall_back(self, reason: str, error: BaseException | None = None) -> None:
        """Give way to the fallback for ``reason``; one the query answers
        without (``_DEGRADING``) is also a degradation of the query."""
        self.status, self.reason = FALLBACK, reason
        if reason in _DEGRADING:
            record_degradation(DegradationStep.STRUCTURED_QUERY, reason, error)

    def record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "mode": self.mode,
            "fallback_strategy": self.fallback_strategy,
            "plan": self.plan,
            "plan_description": self.description,
            "operation": self.operation,
            "row_count": self.row_count,
            "truncated": self.truncated,
            "exact": self.exact,
            "linked": self.linked,
            "linked_names": self.linked_names,
            "repairs": self.repairs,
            "backend": self.backend,
            "compiled_digest": self.compiled_digest,
            "schema_cut": self.schema_cut,
            "elapsed_ms": {k: round(v, 1) for k, v in self.elapsed_ms.items()},
        }


@dataclass
class _Answered:
    """A structured answer: its fusion buckets."""

    buckets: dict[str, list[RetrievalResult]]


def _answers_graph_queries(retriever: object) -> bool:
    """Whether ``retriever`` implements ``GraphQueryPort``."""
    return isinstance(retriever, GraphQueryPort)


def _validation_message(error: ValidationError) -> str:
    """A plan's grammar errors as ``field: reason`` lines."""
    lines = []
    for item in error.errors()[:10]:
        where = ".".join(str(part) for part in item.get("loc", ()))
        lines.append(f"{where or 'plan'}: {item.get('msg', 'invalid')}")
    return "\n".join(lines)


@register_strategy(
    SearchStrategy.STRUCTURED,
    required_artifacts=frozenset(
        {IndexArtifact.TEXT_UNITS, IndexArtifact.ENTITIES, IndexArtifact.GRAPH}
    ),
    # The plan (the model's names, values, entities and keywords), its
    # description and the names of the entities its mentions linked to,
    # which the description and the query result render.
    run_record=RunRecord(
        "structured", model_text=("plan", "plan_description", "linked_names")
    ),
)
class StructuredSearchStrategy(BaseSearchStrategy):
    """A graph query planned as JSON and run by the graph store (see the
    module docstring). Served by every namespace that holds the graph."""

    # The graph core every graph methodology builds (required_artifacts
    # decides what serves it: graphrag, lightrag and hipporag namespaces).
    methodology = Methodology.GRAPHRAG

    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        **kwargs: Any,
    ) -> None:
        super().__init__(config, retrievers, **kwargs)
        self.settings = config.search.structured
        graph = self.graph_retriever
        # Checked when a query runs (``_port``), not here: the chain builds
        # every strategy it checks filter keys for.
        self._graph_query: GraphQueryPort | None = (
            cast(GraphQueryPort, graph) if _answers_graph_queries(graph) else None
        )
        self.planner: Runnable[Any, Any] = setup_chain(
            factory=self.providers.llm_factory,
            model_id=config.model_id(ModelRole.GRAPH_QUERY),
            prompt_class=GraphQueryPlanPrompt,
            parser=StrOutputParser(),
            custom_prompts=config.custom_prompts,
            output_ceiling=config.search.output_ceilings.graph_query,
        )
        self._lock = threading.Lock()
        # suffix -> (read at, profile): a bounded read per suffix and TTL,
        # the least recently used dropped past _MAX_PROFILES suffixes.
        self._profiles: OrderedDict[str, tuple[float, GraphProfile]] = OrderedDict()
        self._fallbacks: dict[SearchStrategy, BaseSearchStrategy] = {}
        self._declared = self._declared_schema(config)

    @staticmethod
    def _declared_schema(config: Config) -> ExtractionSchema | None:
        try:
            schema = config.processing.graph_extraction.extraction_schema()
        except ConfigurationError:
            # A schema_source not loaded into the inline keys: the planner
            # sees what extraction stored (the observed profile).
            return None
        return schema if schema.entity_types or schema.relationship_types else None

    @property
    def graph_query(self) -> GraphQueryPort:
        """The GRAPH retriever as a ``GraphQueryPort``.

        Raises:
            ConfigurationError: The graph store answers no structured query
                (a custom retriever without the port).
        """
        if self._graph_query is None:
            raise ConfigurationError(
                "search strategy 'structured' needs a graph store that answers "
                "structured queries (GraphQueryPort); providers.graph.kind "
                f"'{self.config.providers.graph.kind}' does not"
            )
        return self._graph_query

    # --- the filter contract ---------------------------------------------------------

    def filtered_index_prefixes(self, strategy: SearchStrategy) -> list[str]:
        # Linking and the document-filter checks (entities), the path edges'
        # check (relationships) and the passages (text units): indexes local
        # and mix read too (the graph's keys come with the graph retriever).
        o = self.config.indexing.opensearch
        return self._built_prefixes(
            o.entities_index_prefix,
            o.relationships_index_prefix,
            o.text_units_index_prefix,
        )

    def filter_delegates(
        self, strategy: SearchStrategy, artifacts: Collection[IndexArtifact] | None
    ) -> tuple[SearchStrategy, ...]:
        # The fallback that would run on that namespace (fallback and
        # augment alike); without the namespace's artifacts, every one that
        # may (the configured one, local, mix).
        configured = self.settings.fallback_strategy
        if configured is None:
            return ()
        if artifacts is None:
            return tuple(dict.fromkeys((configured, *GENERAL_FALLBACKS)))
        chosen = self._fallback_for(frozenset(artifacts))
        return () if chosen is None else (chosen,)

    # --- the query ---------------------------------------------------------------------

    async def asearch(
        self, query: SearchQuery, config: RunnableConfig | None = None
    ) -> SearchResult:
        self.graph_query  # noqa: B018 - before any model call, fail on a store without the port
        start = time.time()
        run = _Run(mode=self.settings.mode)
        rerank = self._new_rerank_trace()
        logger.info(
            "Structured search started - query: %s (mode %s)",
            text_digest(query.query),
            run.mode,
        )
        fallback_result: SearchResult | None = None
        # The planner's calls (the fallback's count in the query's usage).
        with usage_scope() as meter:
            plan, linked = await self._planned(query, config, run)
            usage = meter.snapshot()
        if self.settings.mode == "augment":
            answering = asyncio.ensure_future(self._answered(query, plan, linked, run))
            falling_back = asyncio.ensure_future(self._run_fallback(query, config, run))
            await gather_or_cancel([answering, falling_back])
            answered, fallback_result = answering.result(), falling_back.result()
        else:
            answered = await self._answered(query, plan, linked, run)
            if answered is None:
                fallback_result = await self._run_fallback(query, config, run)

        metadata: dict[str, Any] = {}
        if answered is None:
            if fallback_result is None:
                # No structured answer and no fallback: the answer abstains.
                run.status = FAILED
                results: list[RetrievalResult] = []
            else:
                results = list(fallback_result.results)
                metadata.update(fallback_result.metadata)
        else:
            buckets = dict(answered.buckets)
            if fallback_result is not None and fallback_result.results:
                buckets[FALLBACK_BUCKET] = list(fallback_result.results)
                # The fallback's own verdict on its context does not judge
                # one that holds the graph query result.
                metadata.update(
                    {
                        k: v
                        for k, v in fallback_result.metadata.items()
                        if k
                        not in (
                            "rerank",
                            CONTEXT_SUFFICIENT_KEY,
                            INSUFFICIENCY_REASON_KEY,
                        )
                    }
                )
            results = await self._fused(query, buckets, rerank)
            metadata["rerank"] = rerank.summary()
        processing_time = time.time() - start
        record = run.record()
        record["usage"] = usage
        metadata["structured"] = record
        metadata["methodology"] = self.methodology.value
        logger.info(
            "Structured search completed - %s (%s), %s results in %.3fs",
            run.status,
            run.reason or run.operation,
            len(results),
            processing_time,
        )
        return SearchResult(
            query=query,
            results=results,
            total_results=len(results),
            search_strategy="structured_search",
            processing_time=processing_time,
            metadata=metadata,
        )

    # --- planning ----------------------------------------------------------------------

    async def _planned(
        self, query: SearchQuery, config: RunnableConfig | None, run: _Run
    ) -> tuple[ValidatedPlan | None, dict[str, tuple[str, ...]]]:
        """The validated plan and its linked names, or ``(None, {})`` with
        ``run``'s fallback reason."""
        started = time.monotonic()
        with observability.span("kg_rag.graph_query.plan") as span:
            profile = await self._profile(query, run)
            plan = None
            if profile is not None:
                plan = await self._plan(query, profile, config, run)
            linked: dict[str, tuple[str, ...]] = {}
            if plan is not None:
                linked = await self._link(query, plan, run)
                if run.reason is not None:
                    plan = None
            span.set_attributes(
                {
                    "kg_rag.graph_query.status": "planned" if plan else FALLBACK,
                    "kg_rag.graph_query.reason": run.reason or "",
                    "kg_rag.graph_query.repairs": run.repairs,
                }
            )
        run.elapsed_ms["plan"] = (time.monotonic() - started) * 1000
        return plan, linked

    async def _profile(self, query: SearchQuery, run: _Run) -> GraphProfile | None:
        """The schema the planner sees: the observed profile of the
        suffix's graph (cached ``profile_ttl_seconds``) and the declared
        extraction schema (``schema_source``)."""
        settings = self.settings
        key = query.suffix or ""
        # Without a store read: the core properties alone.
        observed: GraphProfile | None = build_profile([], [], sampled=False)
        if settings.schema_source != "declared":
            now = time.monotonic()
            with self._lock:
                cached = self._profiles.get(key)
                if cached is not None:
                    self._profiles.move_to_end(key)
            if cached is not None and now - cached[0] < settings.profile_ttl_seconds:
                observed = cached[1]
            else:
                try:
                    observed = await self.graph_query.adescribe(
                        query.suffix,
                        sample_vertices=settings.profile_sample_vertices,
                        edges_per_vertex=settings.profile_edges_per_vertex,
                        max_examples=settings.max_example_values,
                    )
                except Exception as e:
                    if is_fatal_retrieval_error(e):
                        raise
                    log_failure(
                        logger,
                        logging.WARNING,
                        "Structured search: reading the graph profile failed: %s",
                        error=e,
                    )
                    observed = None
                else:
                    with self._lock:
                        self._profiles[key] = (now, observed)
                        self._profiles.move_to_end(key)
                        while len(self._profiles) > _MAX_PROFILES:
                            self._profiles.popitem(last=False)
        declared = None if settings.schema_source == "observed" else self._declared
        if observed is None and declared is None:
            run.fall_back("no_schema")
            return None
        profile = merge_declared(
            observed or build_profile([], [], sampled=False), declared
        )
        if not profile.entity_types:
            run.fall_back("no_schema")
            return None
        return profile

    async def _plan(
        self,
        query: SearchQuery,
        profile: GraphProfile,
        config: RunnableConfig | None,
        run: _Run,
    ) -> ValidatedPlan | None:
        settings = self.settings
        schema, run.schema_cut = schema_text(
            profile,
            max_tokens=settings.max_schema_tokens,
            count_tokens=self.token_manager.count_tokens,
            # The profile is read without the caller's filters: under them
            # no stored value is shown (it may be an excluded document's).
            with_examples=not query.filters,
        )
        if run.schema_cut:
            logger.debug("Structured search: the schema was cut to its budget")
        feedback = ""
        error: ValidatedPlan | str | None = None
        for attempt in range(settings.repair_attempts + 1):
            if attempt:
                run.repairs += 1
            try:
                raw = await self.planner.ainvoke(
                    {
                        "query": query.query,
                        "schema": schema,
                        "data_language": language_name(index_language(self.config)),
                        "max_hops": str(settings.max_hops),
                        "feedback": feedback,
                    },
                    config,
                )
            except Exception as e:
                if is_fatal_retrieval_error(e):
                    raise
                log_failure(
                    logger,
                    logging.WARNING,
                    "Structured search: the planner call failed: %s",
                    error=e,
                )
                run.fall_back("planner_error", e)
                return None
            error = self._parsed(str(raw), profile, run)
            if isinstance(error, ValidatedPlan):
                return error
            if error is None:
                return None
            # The checker's reason may quote names the model chose.
            logger.info("Structured search: plan refused (%s)", text_digest(error))
            logger.debug("Structured search: plan refused (%s)", error)
            feedback = REPAIR_FEEDBACK.format(error=error[:1500])
        # The last answer decides: still no JSON is a degraded step, a plan
        # the checks refuse a designed fallback.
        if error == _UNREADABLE:
            # An unreadable answer names its reader's error, as every
            # degraded step of unreadable model output does.
            run.fall_back("unreadable_plan", ModelOutputError(_UNREADABLE))
        else:
            run.fall_back("refused_plan")
        return None

    def _parsed(
        self, raw: str, profile: GraphProfile, run: _Run
    ) -> ValidatedPlan | str | None:
        """The validated plan of the planner's answer, the refusal to
        repair (a string), or ``None`` for an unanswerable question."""
        # The planner's repair loop (REPAIR_FEEDBACK) is this prompt's fixer.
        try:
            parsed = parse_model_object(raw, answer_keys=_PLAN_KEYS)
        except ModelOutputError:
            parsed = {}
        if not parsed:
            return _UNREADABLE
        try:
            plan = GraphQueryPlan.model_validate(parsed)
        except ValidationError as e:
            # The names a broken plan lists still help the fallback.
            entities = parsed.get("entities")
            if isinstance(entities, list):
                run.entities = [str(e)[:256] for e in entities[:10] if str(e).strip()]
            return _validation_message(e)
        run.entities, run.keywords = list(plan.entities), list(plan.keywords)
        run.plan = plan.model_dump(mode="json", exclude_defaults=True)
        if plan.unanswerable:
            run.fall_back("unanswerable")
            return None
        try:
            validated = validate_plan(plan, profile, max_hops=self.settings.max_hops)
        except InvalidGraphQueryError as e:
            return str(e)
        return validated

    async def _link(
        self, query: SearchQuery, plan: ValidatedPlan, run: _Run
    ) -> dict[str, tuple[str, ...]]:
        """Each mention node's entity ids (``domain.retrieval.linking``): the
        hits of a hybrid entity search (under the caller's filters and the
        node's types) whose name has the mention's ``linking.link_key``;
        else the best ``link_candidates`` hits of a lexical (BM25) search
        above ``link_min_score`` whose names cover every term of the
        mention (or one ``processing.similarity_threshold`` alike), so an
        absent name that shares a word with a stored one ("Zorblax
        Supplies", "Acme Supplies") links nothing. A mention that links
        nothing sets the fallback reason ``unlinked_mention``."""
        nodes = [node for node in plan.nodes if node.mention is not None]
        if not nodes:
            return {}
        documents = self.document_retriever
        if documents is None:
            run.fall_back("unlinked_mention")
            return {}
        settings = self.settings
        prefix = self.config.indexing.opensearch.entities_index_prefix

        async def search(
            mention: str, types: tuple[str, ...], search_type: SearchType, top_k: int
        ) -> list[RetrievalResult]:
            scope: dict[str, Any] = {"type": list(types)} if types else {}
            return await self._safe_aretrieve(
                documents,
                SearchQuery(
                    query=mention,
                    search_type=search_type,
                    top_k=top_k,
                    retrieval_multiplier=1,
                    index_prefixes=[prefix],
                    suffix=query.suffix,
                    filters=self._scoped_filters(query, **scope),
                ),
                "Structured query entity link",
            )

        async def link(node: Any) -> list[RetrievalResult]:
            mention = str(node.mention)
            hits = await search(
                mention,
                node.types,
                query.search_type,
                max(settings.link_candidates * 3, 10),
            )
            exact = choose_links(mention, hits, policy="cover", limit=0)
            if exact:
                return exact
            lexical = await search(
                mention, node.types, SearchType.LEXICAL, settings.link_candidates
            )
            return choose_links(
                mention,
                lexical,
                policy="cover",
                limit=settings.link_candidates,
                floor=settings.link_min_score,
                similarity=self.config.processing.similarity_threshold,
            )

        found = await gather_or_cancel(link(node) for node in nodes)
        linked: dict[str, tuple[str, ...]] = {}
        names: dict[str, list[str]] = {}
        for node, hits in zip(nodes, found, strict=True):
            chosen = list({str(h.source): h for h in hits}.values())
            if not chosen:
                run.fall_back("unlinked_mention")
                return {}
            linked[node.var] = tuple(str(h.source) for h in chosen)
            names[node.var] = [hit_name(h) or str(h.source) for h in chosen]
        run.linked = {var: list(ids) for var, ids in linked.items()}
        run.linked_names = names
        return linked

    # --- execution ---------------------------------------------------------------------

    async def _answered(
        self,
        query: SearchQuery,
        plan: ValidatedPlan | None,
        linked: dict[str, tuple[str, ...]],
        run: _Run,
    ) -> _Answered | None:
        """The structured answer's buckets, or ``None`` with ``run``'s
        fallback reason."""
        if plan is None:
            return None
        run.description = describe_plan(plan, run.linked_names)
        started = time.monotonic()
        with observability.span("kg_rag.graph_query.execute") as span:
            outcome = await self._executed(query, plan, linked, run)
            span.set_attributes(
                {
                    "kg_rag.graph_query.status": run.status,
                    "kg_rag.graph_query.reason": run.reason or "",
                    "kg_rag.graph_query.operation": run.operation or "",
                    "kg_rag.graph_query.rows": run.row_count,
                    "kg_rag.graph_query.truncated": run.truncated,
                    "kg_rag.graph_query.backend": run.backend or "",
                }
            )
        run.elapsed_ms["execute"] = (time.monotonic() - started) * 1000
        if outcome is None:
            return None
        answer, result, evidence = outcome
        started = time.monotonic()
        buckets = await self._buckets(
            query, plan, answer, result, evidence, linked, run
        )
        run.elapsed_ms["evidence"] = (time.monotonic() - started) * 1000
        return _Answered(buckets)

    async def _executed(
        self,
        query: SearchQuery,
        plan: ValidatedPlan,
        linked: dict[str, tuple[str, ...]],
        run: _Run,
    ) -> tuple[QueryAnswer, GraphQueryResult, list[str]] | None:
        """The answer, the store result and the vertices that evidence it
        (``_evidence``), or ``None`` with ``run``'s fallback reason."""
        settings = self.settings
        unapplied = self._graph_unapplied_filters(query)
        op = plan_operation(plan, scan=bool(unapplied), cap=settings.max_scan_rows)
        run.operation = op.operation if not op.scan else f"{op.operation} (scan)"
        request = self._request(query, plan, op, linked)
        try:
            result = await self.graph_query.aexecute(request)
        except GraphQueryExecutionError as e:
            log_failure(
                logger,
                logging.WARNING,
                "Structured search: the graph query failed: %s",
                error=e,
            )
            run.fall_back("timeout" if e.timed_out else "execution_error", e)
            return None
        except GraphQueryCompileError as e:
            run.fall_back("compile_error", e)
            return None
        except Exception as e:
            if is_fatal_retrieval_error(e):
                raise
            log_failure(
                logger,
                logging.WARNING,
                "Structured search: the graph query failed: %s",
                error=e,
            )
            run.fall_back("execution_error", e)
            return None
        run.backend, run.compiled_digest = result.backend, result.compiled_digest
        kept: list[dict[str, str]] | None = None
        if op.scan:
            answer, kept = await self._scanned(query, plan, result, unapplied)
        else:
            answer = answer_from_result(plan, result)
        run.truncated = answer.truncated
        run.exact = not answer.truncated
        run.row_count = (
            answer.count
            if answer.count is not None
            else len(answer.rows) or len(answer.groups) or len(answer.paths)
        ) or 0
        if answer.empty:
            run.fall_back("empty_result")
            return None
        if (answer.count == 0 or answer.exists is False) and not await self._known(
            query, plan, unapplied
        ):
            # A zero or a no over a value the data does not hold ("Supplier"
            # for "supplier", a translated value) is no answer.
            run.fall_back("unknown_value")
            return None
        evidence, result = await self._evidence(plan, answer, result, request, kept)
        return answer, result, evidence

    def _request(
        self,
        query: SearchQuery,
        plan: ValidatedPlan,
        op: Operation,
        linked: dict[str, tuple[str, ...]],
    ) -> GraphQueryRequest:
        """The store request of ``plan``'s operation ``op`` within the
        query's suffix and graph-applied filters."""
        filters = dict(query.filters or {})
        allowed = filters.pop("id", None)
        return GraphQueryRequest(
            plan=plan,
            suffix=query.suffix,
            operation=op.operation,
            vars=op.vars,
            group_var=op.group_var,
            group_key=op.group_key,
            order=op.order,
            limit=op.limit,
            filters=filters or None,
            allowed_ids=(
                None
                if allowed is None
                else tuple(
                    str(i)
                    for i in (allowed if isinstance(allowed, list) else [allowed])
                )
            ),
            linked=linked,
            hydrate=hydrate_keys(plan),
            timeout_seconds=self.settings.store_timeout_seconds,
        )

    async def _known(
        self, query: SearchQuery, plan: ValidatedPlan, unapplied: dict[str, Any]
    ) -> bool:
        """Whether some vertex the caller may read holds each string value
        ``plan`` compares by equality (``literal_conditions``, one bounded
        ``exists`` request each, concurrently); a failed check counts as
        held, so it never turns an answer into a fallback on its own."""
        checks = [value_check_plan(plan, c) for c in literal_conditions(plan)]
        if not checks:
            return True

        async def holds(check: ValidatedPlan) -> bool:
            op = plan_operation(
                check, scan=bool(unapplied), cap=self.settings.max_scan_rows
            )
            try:
                result = await self.graph_query.aexecute(
                    self._request(query, check, op, {})
                )
            except Exception as e:
                if is_fatal_retrieval_error(e):
                    raise
                log_failure(
                    logger,
                    logging.WARNING,
                    "Structured search: checking a value failed: %s",
                    error=e,
                )
                return True
            if op.scan:
                answer, _ = await self._scanned(query, check, result, unapplied)
            else:
                answer = answer_from_result(check, result)
            return bool(answer.exists)

        return all(await gather_or_cancel(holds(check) for check in checks))

    async def _evidence(
        self,
        plan: ValidatedPlan,
        answer: QueryAnswer,
        result: GraphQueryResult,
        request: GraphQueryRequest,
        kept: list[dict[str, str]] | None,
    ) -> tuple[list[str], GraphQueryResult]:
        """The vertices the answer cites: those it names (rows, groups of
        vertices, paths), else, for a count, an aggregate, a yes or groups by
        a property, the first ``rows_in_context`` matches
        (``evidence_vertex_ids``: those holding the answer's value first, a
        ``min``/``max``'s value or the listed groups' keys, then by rank),
        from the scan, the store's rows or a second bounded store request
        (of the holders, when the answer has any), so the answer still has
        entities and passages to cite, and a maximum cites the vertex that
        holds it."""
        cap = self.settings.rows_in_context
        named = answer_vertex_ids(plan, answer)
        if named or answer.exists is False or answer.count == 0:
            return named[:cap], result
        if kept is not None:
            rows = matching_bindings(plan, kept, result.vertices)
            return (
                evidence_vertex_ids(plan, answer, rows, result.vertices)[:cap],
                result,
            )
        if result.rows and request.vars == plan.vars:
            ids = evidence_vertex_ids(plan, answer, result.rows, result.vertices)
            return ids[:cap], result
        holders = answer_holders(plan, answer)
        sampled = with_condition(plan, *holders) if holders is not None else plan
        sample = request.model_copy(
            update={
                "plan": sampled,
                "operation": "tuples",
                "vars": plan.vars,
                "limit": cap,
            }
        )
        try:
            found = await self.graph_query.aexecute(sample)
            if holders is not None and not found.rows:
                # No vertex holds it as compared (a store's number
                # semantics): the first matches instead.
                found = await self.graph_query.aexecute(
                    sample.model_copy(update={"plan": plan})
                )
        except Exception as e:
            if is_fatal_retrieval_error(e):
                raise
            log_failure(
                logger,
                logging.WARNING,
                "Structured search: reading the evidence failed: %s",
                error=e,
            )
            return [], result
        vertices = {**found.vertices, **result.vertices}
        ids = evidence_vertex_ids(plan, answer, found.rows, vertices)
        merged = result.model_copy(update={"vertices": vertices})
        return ids[:cap], merged

    async def _scanned(
        self,
        query: SearchQuery,
        plan: ValidatedPlan,
        result: GraphQueryResult,
        unapplied: dict[str, Any],
    ) -> tuple[QueryAnswer, list[dict[str, str]]]:
        """The answer of a scan: the bindings whose every vertex (and, of a
        path, every edge) passes the filters the graph cannot apply,
        aggregated in code with the seed conditions."""
        paths = list(result.paths)
        bindings = list(result.rows) or [
            dict(zip(plan.vars, (p.vertex_ids[0], p.vertex_ids[-1]), strict=True))
            for p in paths
        ]
        if unapplied:
            ids = {v for row in bindings for v in row.values()}
            ids |= {v for p in paths for v in p.vertex_ids}
            passing = await self._passing_vertices(query, ids, result, unapplied)
            bindings = [b for b in bindings if set(b.values()) <= passing]
            if paths:
                passing_edges = await self._passing_edges(query, paths, unapplied)
                paths = [
                    p
                    for p in paths
                    if set(p.vertex_ids) <= passing
                    and all(e.id in passing_edges for e in p.edges)
                ]
                ends = {(p.vertex_ids[0], p.vertex_ids[-1]) for p in paths}
                bindings = [
                    b for b in bindings if (b[plan.vars[0]], b[plan.vars[-1]]) in ends
                ]
        answer = aggregate_bindings(
            plan, bindings, result.vertices, truncated=result.truncated, paths=paths
        )
        return answer, bindings

    async def _passing_vertices(
        self,
        query: SearchQuery,
        ids: set[str],
        result: GraphQueryResult,
        unapplied: dict[str, Any],
    ) -> set[str]:
        """Of ``ids``, the entities that pass ``unapplied`` on the entities
        index; under two or more document keys, with a text unit that
        passes them all (as ``_vouch_by_entities_index`` checks)."""
        o = self.config.indexing.opensearch
        passed = await self._ids_passing(
            query, o.entities_index_prefix, sorted(ids), unapplied
        )
        documents = document_filters(unapplied)
        if len(documents) > 1:
            units = {
                vertex: [
                    str(u)
                    for u in (
                        result.vertices.get(vertex, {}).get("text_unit_ids") or ()
                    )
                ]
                for vertex in passed
            }
            passed = await self._one_text_unit_passes(query, units, documents)
        return passed

    async def _passing_edges(
        self, query: SearchQuery, paths: list[PathRow], unapplied: dict[str, Any]
    ) -> set[str]:
        """The path edges that pass the document filters on the
        relationships index; every edge without that index (its endpoints
        were checked)."""
        edges = {e.id for p in paths for e in p.edges}
        documents = document_filters(unapplied)
        if not documents or not self._reads_artifact(
            query, IndexArtifact.RELATIONSHIP_VECTORS
        ):
            return edges
        return await self._ids_passing(
            query,
            self.config.indexing.opensearch.relationships_index_prefix,
            sorted(edges),
            documents,
        )

    # --- evidence ----------------------------------------------------------------------

    async def _buckets(
        self,
        query: SearchQuery,
        plan: ValidatedPlan,
        answer: QueryAnswer,
        result: GraphQueryResult,
        vertex_ids: list[str],
        linked: dict[str, tuple[str, ...]],
        run: _Run,
    ) -> dict[str, list[RetrievalResult]]:
        settings = self.settings
        entities = [
            self._entity_result(vertex_id, result.vertices[vertex_id], rank)
            for rank, vertex_id in enumerate(vertex_ids)
            if vertex_id in result.vertices
        ]
        provenance = self._provenance(entities)
        # By support, off the event loop (``units_by_support``): the query
        # result's lineage, which its citation resolves to documents under
        # the caller's filters.
        lineage = await units_by_support(entities)
        names = {
            str(e.source): str(name)
            for e in entities
            if (name := (e.metadata or {}).get("name"))
        }
        table = RetrievalResult(
            content=render_answer(
                plan,
                answer,
                result.vertices,
                linked=run.linked_names,
                max_lines=settings.rows_in_context,
            ),
            score=1.0,
            source="graph-query",
            retriever_type=SectionType.QUERY_RESULT,
            metadata={
                "plan": run.plan,
                "plan_description": run.description,
                "operation": run.operation,
                "row_count": run.row_count,
                "truncated": answer.truncated,
                "exact": not answer.truncated,
                "entity_ids": vertex_ids,
                # Each returned entity's name, in entity_ids order (None for
                # a vertex the result holds no properties of).
                "entity_names": [names.get(vertex_id) for vertex_id in vertex_ids],
                "backend": result.backend,
                # The result's citation names the documents its evidence
                # rows rest on (application/retrieval/citations.py).
                **({"text_unit_ids": lineage} if lineage else {}),
                **({"provenance": provenance} if provenance else {}),
            },
        )
        buckets: dict[str, list[RetrievalResult]] = {QUERY_RESULT_BUCKET: [table]}
        if entities:
            buckets[ENTITIES_BUCKET] = entities
        relationships_task = asyncio.ensure_future(
            self._path_relationships(query, answer, result)
        )
        passages_task = asyncio.ensure_future(self._passages(query, entities, lineage))
        await gather_or_cancel([relationships_task, passages_task])
        relationships, passages = relationships_task.result(), passages_task.result()
        if relationships:
            buckets[RELATIONSHIPS_BUCKET] = relationships
        buckets.update(passages)
        return buckets

    def _entity_result(
        self, vertex_id: str, vertex: dict[str, list[Any]], rank: int
    ) -> RetrievalResult:
        """A returned vertex as graph expansion projects one, with its
        document filters (seed provenance)."""
        node: dict[str, Any] = {
            key: (values if key in LIST_FIELDS or len(values) != 1 else values[0])
            for key, values in vertex.items()
            if key != FILTERS_KEY
        }
        node["id"] = vertex_id
        raw = next(iter(vertex.get(FILTERS_KEY) or ()), None)
        try:
            filters = json.loads(raw) if isinstance(raw, str) else raw
        except ValueError:
            filters = None
        if isinstance(filters, dict):
            node["attributes"] = {Constants.FILTERS.value: filters}
        return RetrievalResult(
            content=entity_content(node, []),
            score=1.0 / (1 + rank),
            source=vertex_id,
            retriever_type=SectionType.ENTITY,
            metadata=node,
        )

    @staticmethod
    def _provenance(entities: list[RetrievalResult]) -> dict[str, Any] | None:
        """The seed graphs the returned entities come from, when any does:
        the union of their ``seed_provenance``."""
        stated = [
            found
            for entity in entities
            if (found := seed_provenance(entity.metadata or {})) is not None
        ]
        if not stated:
            return None
        both = any(p["kind"] != SEED_SOURCE_KIND for p in stated)
        return {
            "kind": (
                f"{DOCUMENT_SOURCE_KIND}+{SEED_SOURCE_KIND}"
                if both
                else SEED_SOURCE_KIND
            ),
            "seed_graphs": sorted({g for p in stated for g in p["seed_graphs"]}),
        }

    async def _path_relationships(
        self, query: SearchQuery, answer: QueryAnswer, result: GraphQueryResult
    ) -> list[RetrievalResult]:
        """A path answer's edges: their relationships-index documents when
        the namespace holds the index, else drawn from the path."""
        edges = {
            e.id: e
            for p in answer.paths[: self.settings.rows_in_context]
            for e in p.edges
        }
        if not edges:
            return []
        if self.document_retriever is not None and self._reads_artifact(
            query, IndexArtifact.RELATIONSHIP_VECTORS
        ):
            found = await self._fetch_by_ids(
                self.document_retriever,
                SearchQuery(
                    query="",
                    search_type=SearchType.LEXICAL,
                    top_k=len(edges),
                    retrieval_multiplier=1,
                    index_prefixes=[
                        self.config.indexing.opensearch.relationships_index_prefix
                    ],
                    suffix=query.suffix,
                    filters=self._scoped_filters(query, id=list(edges)),
                ),
                "Structured query path relationships",
            )
            if found:
                return self._restore_rank(found, list(edges))

        def name(vertex_id: str) -> str:
            return str(
                next(
                    iter(result.vertices.get(vertex_id, {}).get("name") or ()),
                    vertex_id,
                )
            )

        return [
            RetrievalResult(
                content=f"{name(e.source_id)} -[{e.type}]-> {name(e.target_id)}",
                score=1.0 / (1 + rank),
                source=e.id,
                retriever_type=SectionType.RELATIONSHIP,
                metadata={
                    "id": e.id,
                    "source_id": e.source_id,
                    "target_id": e.target_id,
                    "type": e.type,
                },
            )
            for rank, e in enumerate(edges.values())
        ]

    async def _passages(
        self, query: SearchQuery, entities: list[RetrievalResult], lineage: list[str]
    ) -> dict[str, list[RetrievalResult]]:
        """The supporting passages: the document text units the returned
        entities cite, by how many cite each (under the caller's filters),
        and a hybrid search of the question among them."""
        count = self.settings.supporting_chunks
        documents = self.document_retriever
        if count == 0 or documents is None or not entities:
            return {}
        ranked = [unit for unit in lineage if not is_seed_unit(unit)]
        if not ranked:
            return {}
        prefix = self.config.indexing.opensearch.text_units_index_prefix
        chosen = await self._passing_ranked_units(query, ranked, count * 2)

        async def fetched() -> list[RetrievalResult]:
            if not chosen:
                return []
            hits = await self._fetch_by_ids(
                documents,
                SearchQuery(
                    query="",
                    search_type=SearchType.LEXICAL,
                    top_k=len(chosen),
                    retrieval_multiplier=1,
                    index_prefixes=[prefix],
                    suffix=query.suffix,
                    filters=self._scoped_filters(query, id=chosen),
                ),
                "Structured query cited passages",
            )
            return self._restore_rank(hits, chosen)

        async def searched() -> list[RetrievalResult]:
            if not query.query:
                return []
            # The best-cited units one terms clause takes: a text search is
            # not split in batches (each would rank, and normalize, its own
            # hits), so the list is cut as tree and global search cut theirs.
            limits = await documents.aindex_limits(prefix, query.suffix)
            among = ranked[: min(count * 10, limits.max_terms_count)]
            return await self._safe_aretrieve(
                documents,
                SearchQuery(
                    query=query.query,
                    search_type=query.search_type,
                    top_k=count,
                    retrieval_multiplier=1,
                    index_prefixes=[prefix],
                    suffix=query.suffix,
                    filters=self._scoped_filters(query, id=among),
                ),
                "Structured query passage search",
            )

        units, hits = await gather_or_cancel([fetched(), searched()])
        buckets: dict[str, list[RetrievalResult]] = {}
        if units:
            buckets[TEXT_UNITS_BUCKET] = units
        if hits:
            buckets[TEXT_HITS_BUCKET] = hits
        return buckets

    async def _fused(
        self,
        query: SearchQuery,
        buckets: dict[str, list[RetrievalResult]],
        rerank: Any,
    ) -> list[RetrievalResult]:
        settings = self.settings
        quota = {
            SectionType.QUERY_RESULT.value: 1,
            SectionType.ENTITY.value: settings.rows_in_context,
            SectionType.RELATIONSHIP.value: settings.rows_in_context,
        }
        if settings.supporting_chunks:
            quota[SectionType.TEXT.value] = settings.supporting_chunks
        if FALLBACK_BUCKET in buckets:
            # The fallback's own sections keep their own quotas' width
            # (search.local_search.type_quota).
            for type_, slots in self._per_type_quota(query.top_k).items():
                quota[type_] = max(quota.get(type_, 0), slots)
        return await self._fuse_and_rerank(
            buckets,
            rerank_trace=rerank,
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
            per_type_quota=quota,
            rerank_only_types=(
                {SectionType.TEXT.value} if FALLBACK_BUCKET not in buckets else set()
            ),
        )

    # --- the fallback ------------------------------------------------------------------

    def _fallback(self, strategy: SearchStrategy) -> BaseSearchStrategy:
        with self._lock:
            found = self._fallbacks.get(strategy)
            if found is None:
                spec = get_strategy_spec(strategy)
                found = spec.strategy_class(
                    self.config, self.retrievers, providers=self.providers
                )
                found.hybrid_scorer = self.hybrid_scorer
                found.token_manager = self.token_manager
                self._fallbacks[strategy] = found
            return found

    def _served_fallback(self, query: SearchQuery) -> SearchStrategy | None:
        """The configured fallback, or, where the namespace does not serve
        it, local, then mix (``None``: none is configured or served)."""
        held = held_artifacts_of(query.metadata)
        if held is None:
            return self.settings.fallback_strategy
        return self._fallback_for(held)

    def _fallback_for(self, held: frozenset[IndexArtifact]) -> SearchStrategy | None:
        configured = self.settings.fallback_strategy
        if configured is None:
            return None
        for candidate in (configured, *GENERAL_FALLBACKS):
            if serves(candidate, held, self.config):
                return candidate
        return None

    async def _run_fallback(
        self, query: SearchQuery, config: RunnableConfig | None, run: _Run
    ) -> SearchResult | None:
        strategy = self._served_fallback(query)
        if strategy is None:
            return None
        run.fallback_strategy = strategy.value
        inputs = query_inputs(strategy, self.config)
        update: dict[str, Any] = {
            "metadata": {**query.metadata, "search_strategy": strategy.value}
        }
        if QueryInput.ENTITIES in inputs:
            update["entity_focus"] = list(run.entities)
        if QueryInput.DUAL_KEYWORDS in inputs:
            update["ll_keywords"] = list(run.entities)
            update["hl_keywords"] = list(run.keywords)
        logger.info(
            "Structured search: running %s (%s)",
            strategy.value,
            run.reason or self.settings.mode,
        )
        with observability.span("kg_rag.graph_query.fallback", strategy=strategy.value):
            return await self._fallback(strategy).asearch(
                query.model_copy(update=update), config
            )
