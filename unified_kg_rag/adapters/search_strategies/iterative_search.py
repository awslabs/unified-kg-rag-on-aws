# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Iterative bridge follow-up search (``iterative``; ``auto`` routes multi-hop
questions to it, no default selects it).

A multi-hop question often fails on its last hop: round 0 finds the passage
that names the bridge entity, but not the passage about it. This strategy
wraps a base strategy (``search.iterative_search.base``: ``local`` or
``mix``) by composition:

1. ROUND 0 — the base strategy's own search, unchanged.
2. CONTROL — a fast-model controller (``models.roles.query_controller``)
   reads the question and the context packed from the candidates and says
   whether it answers the question; if not, it names the bridge entities and
   at most ``max_next_queries`` follow-up queries (IRCoT-style).
3. FOLLOW-UP ROUND — the base search again, seeded with the bridge entities
   (local's query entities, mix's low-level keywords), and a text-unit
   search for each follow-up query, both under the request's ``search_type``
   and the caller's filters. The new candidates are fused with every earlier
   round's (RRF, one bucket per round), the text reranked against the
   question and the follow-up queries, and each type kept to round 0's
   share, so a round-0 item leaves only when outranked. Back to 2, for at
   most ``max_rounds`` follow-up rounds; the controller judges the last one
   too.

The answer model answers once, from the final candidates, which the chain packs
into the usual token budget. A follow-up round starts only while the query
deadline leaves ``search.model_timeout_seconds`` for the answer, and is
cancelled when it would not.

``SearchResult.metadata["context_sufficient"]`` is ``False`` (with the
stop reason under ``insufficiency_reason``) when the last
controller verdict said the context does not answer the question and no
further round could run (the round cap, nothing new to search, or the
deadline cancelling the follow-up round that verdict started: the returned
candidates are the ones it judged). It is never set on a degraded run (a failed or
unparsable model call), nor when the deadline ends a run before the
controller judged what it returns, whose verdict is unknown. ``search.abstention.on_insufficient`` acts on it
(hedge or abstain); this strategy only reports it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig

from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.llm.retry import current_deadline
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
)
from unified_kg_rag.adapters.retrieval.hybrid_scorer import RerankTrace
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    get_strategy_spec,
    register_strategy,
)
from unified_kg_rag.adapters.search_strategies.lightrag_search import (
    LightRAGSearchStrategy,
)
from unified_kg_rag.adapters.search_strategies.local_search import (
    LocalSearchStrategy,
)
from unified_kg_rag.adapters.store_errors import is_fatal_retrieval_error
from unified_kg_rag.domain.ingestion.base_resolver import compact_name_key
from unified_kg_rag.domain.models import (
    Config,
    ModelRole,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from unified_kg_rag.domain.prompts import IterativeControllerPrompt
from unified_kg_rag.domain.retrieval.abstention import (
    CONTEXT_SUFFICIENT_KEY,
    INSUFFICIENCY_REASON_KEY,
)
from unified_kg_rag.shared import get_logger, observability
from unified_kg_rag.shared.usage import usage_scope
from unified_kg_rag.shared.utils import interleave, parse_llm_json, text_digest
from unified_kg_rag.shared.utils.concurrency import gather_or_cancel

logger = get_logger(__name__)

# The fusion bucket of round n's base search (round 0: the first search) and
# of its follow-up queries' text-unit search (search.fusion.fusion_weights).
ROUND_BUCKET = "iterative_round_{}"
ROUND_TEXT_BUCKET = "iterative_round_{}_text"

# What the controller's JSON answer holds.
_CONTROLLER_KEYS = ("sufficient", "next_queries", "bridge_entities")


def _base_strategy(config: Config) -> SearchStrategy:
    return SearchStrategy(config.search.iterative_search.base)


@dataclass
class _Verdict:
    """One controller answer."""

    sufficient: bool
    known_facts: list[str] = field(default_factory=list)
    next_queries: list[str] = field(default_factory=list)
    bridge_entities: list[str] = field(default_factory=list)


@dataclass
class _Run:
    """What one query's rounds did (``SearchResult.metadata["iterative"]``)."""

    rounds: int = 0
    llm_calls: int = 0
    stop_reason: str | None = None
    # Why the run stopped early, when a model call failed or could not be
    # read (the answer then reads what the rounds before it found).
    degraded: str | None = None
    # The non-fatal error of a failed call, for ``degraded_error``.
    error: str | None = None
    insufficient: bool = False
    searched: list[str] = field(default_factory=list)
    bridge_keys: set[str] = field(default_factory=set)
    verdicts: list[dict[str, Any]] = field(default_factory=list)


def _strings(value: Any) -> list[str]:
    """A JSON list of strings, stripped, blank ones dropped."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _flag(value: Any) -> bool | None:
    """A JSON boolean (or its string form); ``None`` when it is neither."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return None


def _query_key(text: str) -> str:
    return " ".join(text.casefold().split())


@register_strategy(SearchStrategy.ITERATIVE, wraps=_base_strategy)
class IterativeSearchStrategy(BaseSearchStrategy):
    """Bounded bridge follow-up rounds over a base strategy (see the module
    docstring). Reads what the base reads: its query extraction, its indexes
    and its context packing (``StrategySpec.wraps``), so it needs no
    artifact the base does not."""

    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        **kwargs: Any,
    ) -> None:
        super().__init__(config, retrievers, **kwargs)
        self.settings = config.search.iterative_search
        self.base_strategy = _base_strategy(config)
        base_class = get_strategy_spec(self.base_strategy).strategy_class
        # Round 0 is the base search as it runs on its own; the follow-up
        # rounds rerun it without its rerank, since their union is reranked
        # once against the question and the follow-up queries. Both share
        # this strategy's providers, scorer and token counter.
        self.base = base_class(config, retrievers, providers=self.providers)
        self.follow_up = base_class(config, retrievers, providers=self.providers)
        for strategy in (self.base, self.follow_up):
            strategy.hybrid_scorer = self.hybrid_scorer
            strategy.token_manager = self.token_manager
        if isinstance(self.follow_up, LocalSearchStrategy | LightRAGSearchStrategy):
            self.follow_up.rerank_types = frozenset()
        factory = self.providers.llm_factory
        model_id = config.model_id(ModelRole.QUERY_CONTROLLER)
        self.controller: Runnable[Any, Any] = setup_chain(
            factory=factory,
            model_id=model_id,
            prompt_class=IterativeControllerPrompt,
            parser=StrOutputParser(),
            custom_prompts=config.custom_prompts,
        )

    # The filters a query may carry are the base's, on the indexes it reads
    # (the follow-up text search reads the text units, which both bases do).
    def filtered_index_prefixes(self, strategy: SearchStrategy) -> list[str] | None:
        return self.base.filtered_index_prefixes(self.base_strategy)

    def reserved_filter_keys(self, strategy: SearchStrategy) -> frozenset[str]:
        return self.base.reserved_filter_keys(self.base_strategy)

    def filtered_retrievers(
        self, strategy: SearchStrategy
    ) -> list[BaseGraphRAGRetriever]:
        return self.base.filtered_retrievers(self.base_strategy)

    async def asearch(
        self, query: SearchQuery, config: RunnableConfig | None = None
    ) -> SearchResult:
        start_time = time.time()
        logger.info(
            "Iterative search started - base: '%s', query: %s",
            self.base_strategy.value,
            text_digest(query.query),
        )
        base_query = query.model_copy(
            update={
                "metadata": {
                    **query.metadata,
                    "search_strategy": self.base_strategy.value,
                }
            }
        )
        run = _Run(searched=[query.query])
        rerank = self._new_rerank_trace()
        # The rounds' own model calls (the controller): the query's usage
        # counts them too, by model.
        with usage_scope() as meter:
            with observability.span(
                "kg_rag.iterative.round", round=0, base=self.base_strategy.value
            ):
                first = await self.base.asearch(base_query, config)
            pool: dict[str, list[RetrievalResult]] = {
                ROUND_BUCKET.format(0): first.results
            }
            results = await self._control(
                query, base_query, first, pool, rerank, run, config
            )
            usage = meter.snapshot()

        processing_time = time.time() - start_time
        logger.info(
            "Iterative search completed: %s follow-up rounds (%s), %s model "
            "calls, %s results in %.3fs",
            run.rounds,
            run.stop_reason,
            run.llm_calls,
            len(results),
            processing_time,
        )
        # The rerank of the last fusion orders the results; round 0's own is
        # kept beside it.
        metadata: dict[str, Any] = {
            **first.metadata,
            "rerank": (
                rerank.summary() if rerank.stages else first.metadata.get("rerank")
            ),
            "iterative": {
                "round_0_rerank": first.metadata.get("rerank"),
                "base": self.base_strategy.value,
                "rounds": run.rounds,
                "stop_reason": run.stop_reason,
                "degraded": run.degraded,
                "llm_calls": run.llm_calls,
                "usage": usage,
                "searched": run.searched[1:],
                "verdicts": run.verdicts,
                "added_results": self._added(first.results, results),
            },
        }
        if run.insufficient:
            metadata[CONTEXT_SUFFICIENT_KEY] = False
            metadata[INSUFFICIENCY_REASON_KEY] = run.stop_reason
        if run.error is not None:
            metadata["degraded_error"] = run.error
        return SearchResult(
            query=query,
            results=results,
            total_results=len(results),
            search_strategy="iterative_search",
            processing_time=processing_time,
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # The controller loop
    # ------------------------------------------------------------------ #

    async def _control(
        self,
        query: SearchQuery,
        base_query: SearchQuery,
        first: SearchResult,
        pool: dict[str, list[RetrievalResult]],
        rerank: RerankTrace,
        run: _Run,
        config: RunnableConfig | None,
    ) -> list[RetrievalResult]:
        """Judge, follow up, fuse; until sufficient or out of rounds. The
        candidates of the last round that finished are returned."""
        current = first.results
        budget = self._time_budget()
        if budget is not None and budget <= 0:
            run.stop_reason = "deadline"
            return current
        scope = asyncio.timeout(budget)
        # Whether the controller found ``current`` insufficient: a deadline
        # that ends the follow-up round after that verdict returns it as
        # judged.
        judged_insufficient = False
        try:
            async with scope:
                while True:
                    verdict = await self._judge(query, current, run, config)
                    if verdict is None:
                        return current
                    judged_insufficient = not verdict.sufficient
                    if verdict.sufficient:
                        run.stop_reason = "sufficient"
                        return current
                    if run.rounds >= self.settings.max_rounds:
                        run.stop_reason = "max_rounds"
                        run.insufficient = True
                        return current
                    queries, bridges = self._new_leads(
                        verdict.next_queries, verdict.bridge_entities, run
                    )
                    if not queries and not bridges:
                        run.stop_reason = "no_follow_up"
                        run.insufficient = True
                        return current
                    round_no = run.rounds + 1
                    # Without a new bridge a mix rerun would read the
                    # question's keywords again, which round 0 searched.
                    streams = await self._round_search(
                        query,
                        base_query,
                        round_no,
                        queries,
                        bridges,
                        config,
                        rerun_base=bool(bridges) or not self._keyword_seeded(),
                    )
                    pool.update(streams)
                    run.rounds = round_no
                    current = await self._fuse(query, first, pool, rerank, run)
                    judged_insufficient = False
        except TimeoutError:
            if not scope.expired():
                raise
            logger.warning(
                "Iterative search: the query deadline leaves no time for "
                "follow-up round %s; answering from the earlier rounds",
                run.rounds + 1,
            )
            run.stop_reason = "deadline"
            run.insufficient = judged_insufficient
            return current

    async def _judge(
        self,
        query: SearchQuery,
        results: list[RetrievalResult],
        run: _Run,
        config: RunnableConfig | None,
    ) -> _Verdict | None:
        """The controller's verdict on ``results``; ``None`` when its call
        failed or its answer could not be read (``run.degraded``)."""
        context = await self._pack(results, query.query, query)
        run.llm_calls += 1
        try:
            raw = await self.controller.ainvoke(
                {
                    "query": query.query,
                    "context": context,
                    "searched": "\n".join(f"- {q}" for q in run.searched),
                    "max_next_queries": self.settings.max_next_queries,
                    "target_language": self.query_language(query),
                },
                config,
            )
        except Exception as e:  # noqa: BLE001 - classified by _degrade
            self._degrade(run, "controller_failed", e)
            return None
        payload = parse_llm_json(str(raw), answer_keys=_CONTROLLER_KEYS)
        sufficient = _flag(payload.get("sufficient"))
        if sufficient is None:
            logger.warning(
                "Iterative search: the controller's answer holds no "
                "'sufficient' verdict; answering from the earlier rounds"
            )
            run.degraded = run.stop_reason = "controller_unparsed"
            return None
        verdict = _Verdict(
            sufficient=sufficient,
            known_facts=_strings(payload.get("known_facts")),
            next_queries=_strings(payload.get("next_queries")),
            bridge_entities=_strings(payload.get("bridge_entities")),
        )
        run.verdicts.append(
            {
                "round": run.rounds,
                "sufficient": verdict.sufficient,
                "known_facts": len(verdict.known_facts),
                "bridge_entities": verdict.bridge_entities,
                "next_queries": verdict.next_queries,
            }
        )
        return verdict

    def _new_leads(
        self, queries: list[str], bridges: list[str], run: _Run
    ) -> tuple[list[str], list[str]]:
        """The follow-up queries not searched yet (at most
        ``max_next_queries``) and the bridge entities not followed yet (at
        most ``max_bridge_entities``), recorded as searched and followed."""
        asked = {_query_key(q) for q in run.searched}
        fresh: list[str] = []
        for text in queries:
            if len(fresh) >= self.settings.max_next_queries:
                break
            if (key := _query_key(text)) not in asked:
                asked.add(key)
                fresh.append(text)
        names: list[str] = []
        for name in bridges:
            if len(names) >= self.settings.max_bridge_entities:
                break
            key = compact_name_key(name)
            if key and key not in run.bridge_keys:
                run.bridge_keys.add(key)
                names.append(name)
        run.searched.extend(fresh)
        return fresh, names

    # ------------------------------------------------------------------ #
    # Rounds, fusion and packing
    # ------------------------------------------------------------------ #

    async def _round_search(
        self,
        query: SearchQuery,
        base_query: SearchQuery,
        round_no: int,
        queries: list[str],
        bridges: list[str],
        config: RunnableConfig | None,
        *,
        rerun_base: bool = True,
    ) -> dict[str, list[RetrievalResult]]:
        """Round ``round_no``'s candidates: the base search seeded with
        ``bridges`` (local's query entities, mix's low-level keywords; the
        first query is its text) and a text-unit search per query, all
        under the caller's filters and the request's ``search_type``.

        Without ``bridges`` mix keeps the question's own keywords (with
        none it would fall back to the search text as a keyword, or search
        no entity at all), and local looks its entities up by the search
        text. ``rerun_base`` false runs the text searches alone."""
        seeds: dict[str, Any] = {"entity_focus": bridges}
        if bridges:
            seeds |= {"ll_keywords": bridges, "hl_keywords": []}
        seeded = base_query.model_copy(
            update={
                "query": queries[0] if queries else " ".join(bridges),
                **seeds,
                "optional_keywords": [],
                "seed_weights": None,
            }
        )
        with observability.span(
            "kg_rag.iterative.round",
            round=round_no,
            queries=len(queries),
            bridge_entities=len(bridges),
            base_rerun=rerun_base,
        ) as span:
            base_task = (
                asyncio.ensure_future(self.follow_up.asearch(seeded, config))
                if rerun_base
                else None
            )
            text_tasks = [
                asyncio.ensure_future(self._text_search(query, text))
                for text in queries
            ]
            await gather_or_cancel(
                task for task in (base_task, *text_tasks) if task is not None
            )
            streams: dict[str, list[RetrievalResult]] = {}
            if base_task is not None:
                streams[ROUND_BUCKET.format(round_no)] = base_task.result().results
            if text_tasks:
                streams[ROUND_TEXT_BUCKET.format(round_no)] = interleave(
                    task.result() for task in text_tasks
                )
            if observability.is_enabled():
                span.set_attributes(
                    {f"kg_rag.hits.{name}": len(v) for name, v in streams.items()}
                )
            return streams

    async def _text_search(
        self, query: SearchQuery, text: str
    ) -> list[RetrievalResult]:
        """The ``round_top_k`` text units ``text`` finds under the request's
        ``search_type`` (BM25 + vector by default) and the caller's filters."""
        if self.document_retriever is None:
            return []
        return await self._safe_aretrieve(
            self.document_retriever,
            SearchQuery(
                query=text,
                search_type=query.search_type,
                top_k=self.settings.round_top_k,
                index_prefixes=[
                    self.config.indexing.opensearch.text_units_index_prefix
                ],
                suffix=query.suffix,
                filters=self._scoped_filters(query),
            ),
            "Iterative follow-up text search",
        )

    async def _fuse(
        self,
        query: SearchQuery,
        first: SearchResult,
        pool: dict[str, list[RetrievalResult]],
        rerank: RerankTrace,
        run: _Run,
    ) -> list[RetrievalResult]:
        """Every round's candidates fused (RRF over the round buckets), the
        base's rerank types reranked against the question and the follow-up
        queries, each type kept to round 0's share of it (at least the
        strategies' own quota)."""
        quota = self._per_type_quota(query.top_k)
        for result in first.results:
            quota.setdefault(result.retriever_type, 0)
        shares: dict[str, int] = {}
        for result in first.results:
            shares[result.retriever_type] = shares.get(result.retriever_type, 0) + 1
        per_type = {
            kind: max(slots, shares.get(kind, 0)) for kind, slots in quota.items()
        }
        return await self._fuse_and_rerank(
            pool,
            rerank_trace=rerank,
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query="\n".join(run.searched),
            per_type_quota=per_type,
            rerank_only_types=set(self._rerank_types()),
        )

    def _keyword_seeded(self) -> bool:
        """Whether the base reads its graph seeds from the query's keywords
        (mix) rather than looking entities up by the search text (local)."""
        return isinstance(self.follow_up, LightRAGSearchStrategy)

    def _rerank_types(self) -> frozenset[str]:
        if isinstance(self.base, LocalSearchStrategy | LightRAGSearchStrategy):
            return self.base.rerank_types
        return frozenset()

    async def _pack(
        self, results: list[RetrievalResult], text: str, query: SearchQuery
    ) -> str:
        """``results`` packed as the base's answer context is, within
        ``controller_context_tokens`` (and the query's own budget); without
        either, the answer's budget, so the controller judges what the
        answer will read."""
        budgets = [
            b
            for b in (self.settings.controller_context_tokens, query.max_tokens)
            if b is not None
        ]
        # Counting can be one CountTokens request per section: off the loop.
        optimized = await asyncio.to_thread(
            self.token_manager.optimize_context,
            results,
            text,
            min(budgets) if budgets else None,
            type(self.base).spare_context_budget_first,
        )
        return self.token_manager.build_context_string(optimized)

    def _time_budget(self) -> float | None:
        """Seconds the follow-up rounds may take: what the query deadline
        leaves after ``search.model_timeout_seconds`` for the answer;
        ``None`` without a deadline."""
        deadline = current_deadline()
        if deadline is None:
            return None
        return deadline - time.monotonic() - self.config.search.model_timeout_seconds

    def _degrade(self, run: _Run, reason: str, error: Exception) -> None:
        """Record a failed model call, or raise ``error`` when it is fatal
        (no model access, an unusable store): the rounds stop and the
        answer reads what the earlier ones found."""
        if is_fatal_retrieval_error(error):
            raise error
        logger.error("Iterative search: %s: %s", reason, error)
        run.degraded = run.stop_reason = reason
        run.error = type(error).__name__

    @staticmethod
    def _added(before: list[RetrievalResult], after: list[RetrievalResult]) -> int:
        """Results of ``after`` that round 0 did not return."""
        seen = {(r.retriever_type, r.source or r.content) for r in before}
        return sum(
            1 for r in after if (r.retriever_type, r.source or r.content) not in seen
        )
