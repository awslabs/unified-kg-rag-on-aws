# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""MS GraphRAG DRIFT search: a community-report primer, then local search.

1. PRIMER — the query is expanded into a hypothetical answer shaped like a
   community report (HyDE); the ``k_followups`` reports (of any depth of the
   hierarchy) most similar to it are split into ``primer_folds`` and each
   fold drafts an intermediate answer, a 0-100 score and follow-up queries.
2. ACTIONS — for ``n_depth`` rounds, the ``k_followups`` open follow-up queries
   whose parent answer scored best run as local searches (not reranked);
   each local context is answered with its own score and follow-up queries.
3. REDUCE — the ranked intermediate answers are synthesized context
   sections, each citable by the evidence it was written from, and the
   local evidence behind them, reranked once, follows them;
   the answer model reduces them into the answer.

``search.max_concurrency`` primer calls and actions run at a time.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig

from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
    evidence_lineage,
    prompt_call_tokens,
    record_degradation,
)
from unified_kg_rag.adapters.retrieval.strategy_registry import register_strategy
from unified_kg_rag.adapters.search_strategies.local_search import (
    LocalSearchStrategy,
)
from unified_kg_rag.adapters.store_errors import is_fatal_retrieval_error
from unified_kg_rag.domain.ingestion.cache_keys import index_language
from unified_kg_rag.domain.models import (
    Config,
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
from unified_kg_rag.domain.models.methodology import REPORT_READER_ARTIFACTS
from unified_kg_rag.domain.prompts import (
    DriftActionPrompt,
    DriftHydePrompt,
    DriftPrimerPrompt,
)
from unified_kg_rag.domain.retrieval.degradation import DegradationStep
from unified_kg_rag.shared import get_logger, observability, usage
from unified_kg_rag.shared.logging import log_failure
from unified_kg_rag.shared.utils import (
    parse_llm_score,
    text_digest,
)
from unified_kg_rag.shared.utils.concurrency import gather_or_cancel

logger = get_logger(__name__)

# Follow-up queries one primer or action call is asked for (MS GraphRAG's
# primer asks for "at least five").
_FOLLOW_UPS_PER_ANSWER = 5


@dataclass
class _Answer:
    """One intermediate answer: of the primer (to the question) or an action."""

    query: str
    text: str
    score: int
    follow_ups: list[str] = field(default_factory=list)
    evidence: list[RetrievalResult] = field(default_factory=list)
    # What the answer was written from: the primer fold's reports, or the
    # sections of the action's local context (its lineage).
    read: list[RetrievalResult] = field(default_factory=list)
    # The non-fatal error that left the answer without text, if any.
    error: Exception | None = None
    # Whether the query's limits refused the action's call (not sent).
    limited: bool = False


# The keys of the primer's and of an action's JSON answer.
_PRIMER_KEYS = ("intermediate_answer", "follow_up_queries")
_ACTION_KEYS = ("response", "follow_up_queries")


def _follow_ups(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("follow_up_queries")
    if not isinstance(raw, list):
        return []
    return [str(q).strip() for q in raw if str(q).strip()]


@register_strategy(SearchStrategy.DRIFT, required_artifacts=REPORT_READER_ARTIFACTS)
class DriftSearchStrategy(BaseSearchStrategy):
    methodology = Methodology.GRAPHRAG

    @classmethod
    def fewest_model_calls(cls, config: Config) -> int:
        # The query expansion, a primer call per fold and one action.
        return 1 + config.search.drift_search.primer_folds + 1

    @classmethod
    def answering_calls(cls, config: Config, question: str) -> list[int]:
        # One action: the expansion and the primer are optional, and the
        # question itself is followed up without them, with the least
        # context.
        return [
            prompt_call_tokens(
                config,
                DriftActionPrompt,
                question,
                estimate_token_count(question)
                + config.search.token_manager.min_truncated_section_tokens,
            )
        ]

    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        **kwargs: Any,
    ):
        super().__init__(config, retrievers, **kwargs)
        self.drift_config = self.config.search.drift_search
        self.ignore_errors = config.processing.ignore_errors
        self.max_concurrency = config.search.max_concurrency

        factory = self.providers.llm_factory
        parser = StrOutputParser()
        self.hyde: Runnable[Any, Any] = setup_chain(
            factory=factory,
            model_id=self.config.model_id(ModelRole.DRIFT_PRIMER),
            prompt_class=DriftHydePrompt,
            parser=parser,
            custom_prompts=self.config.custom_prompts,
        )
        self.primer: Runnable[Any, Any] = setup_chain(
            factory=factory,
            model_id=self.config.model_id(ModelRole.DRIFT_PRIMER),
            prompt_class=DriftPrimerPrompt,
            parser=parser,
            custom_prompts=self.config.custom_prompts,
        )
        self.action: Runnable[Any, Any] = setup_chain(
            factory=factory,
            model_id=self.config.model_id(ModelRole.DRIFT_ACTION),
            prompt_class=DriftActionPrompt,
            parser=parser,
            custom_prompts=self.config.custom_prompts,
        )

        # Each follow-up query is a local search, as in MS GraphRAG. It shares
        # this strategy's providers, scorer and token counter, and does not
        # rerank: the evidence of every follow-up is reranked once, at the
        # final fusion (one rerank call per action cost a model call each,
        # and their outcomes were never reported).
        self.local_search = LocalSearchStrategy(
            config, retrievers, providers=self.providers
        )
        self.local_search.hybrid_scorer = self.hybrid_scorer
        self.local_search.token_manager = self.token_manager
        self.local_search.rerank_types = frozenset()
        # Its own text-unit ranking (support by default: local's ppr default
        # is unmeasured under DRIFT).
        self.local_search.text_unit_ranking = self.drift_config.text_unit_ranking

    def filtered_index_prefixes(self, strategy: SearchStrategy) -> list[str]:
        # The primer's reports and the follow-ups' local searches.
        return self.local_search.filtered_index_prefixes(SearchStrategy.LOCAL)

    async def asearch(
        self, query: SearchQuery, config: RunnableConfig | None = None
    ) -> SearchResult:
        start_time = time.time()
        rerank = self._new_rerank_trace()
        logger.info(
            "Drift search started - query: %s ('%s')",
            text_digest(query.query),
            query.search_type.value,
        )
        # report_selection: how many follow-up local searches read their
        # reports by each selection (local's metadata of the same name).
        stats: dict[str, Any] = {"drift_llm_calls": 0, "report_selection": {}}
        semaphore = asyncio.Semaphore(self.max_concurrency)

        with observability.span("kg_rag.drift.primer") as span:
            reports = await self._primer_reports(query, stats, config)
            answers = await self._prime(query, reports, semaphore, stats, config)
            span.set_attributes(
                {"kg_rag.reports": len(reports), "kg_rag.answers": len(answers)}
            )

        # Open follow-up queries: (parent score, arrival order, query). MS
        # GraphRAG shuffles them; the parent's score orders them here, so the
        # follow-ups of the most useful answers run first.
        queue: list[tuple[int, int, str]] = []
        asked = {query.query.strip().lower()}

        def enqueue(parent: _Answer) -> None:
            for follow_up in parent.follow_ups:
                key = follow_up.lower()
                if key not in asked:
                    asked.add(key)
                    queue.append((parent.score, len(asked), follow_up))

        for answer in answers:
            enqueue(answer)
        if not queue:
            # Nothing to follow up (no reports, or the primer gave none): one
            # local search on the question itself still answers it.
            queue.append((0, 0, query.query))

        actions = 0
        depth = 0
        # Whether any action ran without an error, and whether the query's
        # limits cut a round (the last one).
        acted = False
        limited = False
        while queue and depth < self.drift_config.n_depth:
            queue.sort(key=lambda item: (-item[0], item[1]))
            batch = [q for _, _, q in queue[: self.drift_config.k_followups]]
            # Under the query's limits: the best follow-ups whose actions
            # (each with at least the least context) fit beside the answer,
            # which reads the answers so far; the rounds end with them, and
            # each reads an even share of what is left.
            answer_tokens = self._answer_tokens(answers)
            # Each action reads its follow-up besides the question.
            follow_ups = [estimate_token_count(q) for q in batch]
            fit = self._calls_within_limits(
                DriftActionPrompt,
                query,
                [tokens + self._min_context_tokens for tokens in follow_ups],
                answer_context=answer_tokens,
            )
            if fit < len(batch):
                self._record_limit_stop(DegradationStep.DRIFT_ACTION, fit, len(batch))
                batch = batch[:fit]
                limited = True
            if not batch:
                break
            del queue[: len(batch)]
            with observability.span(
                "kg_rag.drift.actions", depth=depth, actions=len(batch)
            ):
                context_tokens = self._context_share(
                    len(batch),
                    DriftActionPrompt,
                    query,
                    ceiling=self.drift_config.local_context_tokens,
                    answer_context=answer_tokens,
                    call_tokens=max(follow_ups[: len(batch)]),
                )
                done = await gather_or_cancel(
                    self._act(sub, query, semaphore, stats, config, context_tokens)
                    for sub in batch
                )
            actions += len(batch)
            depth += 1
            failed = [a.error for a in done if a.error is not None]
            acted = acted or len(failed) < len(done)
            # An action the limits refused when built ends the rounds as a
            # planned one that did not fit.
            limited = limited or any(a.limited for a in done)
            for answer in done:
                answers.append(answer)
                enqueue(answer)
            if limited:
                break
            if failed and len(failed) == len(done):
                # Every action of the round failed: the model is not answering
                # (throttled past its retries, timing out), so another round
                # would fail the same way. One failed action among others only
                # leaves its own answer out.
                if not acted and not self.ignore_errors:
                    raise failed[0]
                logger.warning(
                    "All %s DRIFT actions of round %s failed; stopping the rounds",
                    len(done),
                    depth,
                )
                break

        evidence = [r for answer in answers for r in answer.evidence]
        fused = (
            await self._fuse_and_rerank(
                {"results": self._rank_within_type(self._dedupe(evidence))},
                rerank_trace=rerank,
                top_k=query.top_k,
                retrieval_multiplier=query.retrieval_multiplier,
                query=query.query,
                per_type_quota=self._per_type_quota(query.top_k),
                rerank_only_types={SectionType.TEXT.value},
            )
            if evidence
            else []
        )
        # A report cites what its findings cite, as an answer read from it
        # does: one read for both (the follow-ups' reports carry it already).
        lineages = await self._findings_lineages(
            query, [*fused, *(r for answer in answers for r in answer.read)]
        )
        fused = self._with_findings_lineages(fused, lineages)
        results = await self._answers_items(answers, lineages) + fused

        processing_time = time.time() - start_time
        logger.info(
            "Drift search completed: %s primer reports, %s follow-up queries in %s "
            "rounds, %s LLM calls, %s results in %.3fs",
            len(reports),
            actions,
            depth,
            stats["drift_llm_calls"],
            len(results),
            processing_time,
        )
        return SearchResult(
            query=query,
            results=results,
            total_results=len(results),
            search_strategy="drift_search",
            processing_time=processing_time,
            metadata={
                **stats,
                "drift_primer_reports": len(reports),
                "drift_follow_up_queries": actions,
                "drift_depth": depth,
                "drift_answers": sum(1 for a in answers if a.text),
                "drift_open_follow_ups": len(queue),
                "rerank": rerank.summary(),
            },
        )

    # ------------------------------------------------------------------ #
    # Primer
    # ------------------------------------------------------------------ #

    def _reports_query(self, base: SearchQuery, **update: Any) -> SearchQuery:
        """A search of the community reports, of any depth.

        MS GraphRAG reads the reports of one community_level. Across every
        depth, similarity picks the granularity that fits the hypothetical
        answer: on a synthetic six-hop corpus the depth-2 cut (its leaves)
        answered 16 of 21 sub-questions where every depth answered 21.
        """
        return base.model_copy(
            update={
                "index_prefixes": [
                    self.config.indexing.opensearch.community_reports_index_prefix
                ],
                "retrieval_multiplier": 1,
                **update,
            }
        )

    async def _primer_reports(
        self,
        query: SearchQuery,
        stats: dict[str, Any],
        config: RunnableConfig | None,
    ) -> list[RetrievalResult]:
        """The ``k_followups`` reports most similar to the expanded query."""
        retriever = self.document_retriever
        if retriever is None:
            return []
        if self._source_kinds_skip_reports(query):
            # No primer: the question itself is the one follow-up, a local
            # search, which reads no report under the filter either.
            logger.info(
                "DRIFT primer skipped: the source_kinds filter keeps one "
                "source kind and a community report carries none"
            )
            return []
        # The template the hypothetical answer takes its shape from: MS
        # GraphRAG picks a random report; the most important one is stable.
        template = await self._safe_aretrieve(
            retriever,
            self._reports_query(
                query, query="", search_type=SearchType.LEXICAL, top_k=1
            ),
            "DRIFT report template",
        )
        if not template:
            return []
        expanded = query.query
        # Under the query's limits: the expansion only beside the least
        # primer call, action and answer (else the question finds the
        # reports).
        if not self._calls_within_limits(
            DriftHydePrompt,
            query,
            [template[0].content],
            reserve_calls=3,
            reserve_tokens=self._least_call_tokens(DriftPrimerPrompt, query)
            + self._least_call_tokens(DriftActionPrompt, query),
        ):
            self._record_limit_stop(DegradationStep.DRIFT_EXPANSION, 0, 1)
        else:
            expanded = await self._expand(query, template[0].content, stats, config)
        return await self._safe_aretrieve(
            retriever,
            self._reports_query(
                query, query=expanded, top_k=self.drift_config.k_followups
            ),
            "DRIFT primer reports",
        )

    async def _expand(
        self,
        query: SearchQuery,
        template: str,
        stats: dict[str, Any],
        config: RunnableConfig | None,
    ) -> str:
        """The question expanded into a hypothetical report shaped like
        ``template`` (HyDE); the question itself when the call failed."""
        expanded = query.query
        try:
            stats["drift_llm_calls"] += 1
            hypothetical = await self.hyde.ainvoke(
                {
                    "query": query.query,
                    "template": template,
                    # The reports' language (the index's), not a requested
                    # answer language: the hypothetical report only finds
                    # reports.
                    "target_language": language_name(index_language(self.config)),
                },
                config,
            )
            expanded = str(hypothetical).strip() or query.query
        except Exception as e:
            if not self.ignore_errors:
                raise
            self._degrade(DegradationStep.DRIFT_EXPANSION, "DRIFT query expansion", e)
        return expanded

    async def _prime(
        self,
        query: SearchQuery,
        reports: list[RetrievalResult],
        semaphore: asyncio.Semaphore,
        stats: dict[str, Any],
        config: RunnableConfig | None,
    ) -> list[_Answer]:
        """One intermediate answer per fold of the primer reports, at most
        ``search.max_concurrency`` folds at a time (as the actions), each
        citing its fold's reports."""
        folds = self._folds(reports, self.drift_config.primer_folds)
        # Under the query's limits: the folds that fit beside the least
        # action and the answer.
        fit = self._calls_within_limits(
            DriftPrimerPrompt,
            query,
            ["\n\n".join(r.content for r in fold) for fold in folds],
            reserve_calls=2,
            reserve_tokens=self._least_call_tokens(DriftActionPrompt, query),
        )
        if fit < len(folds):
            self._record_limit_stop(DegradationStep.DRIFT_PRIMER, fit, len(folds))
            folds = folds[:fit]

        async def prime(fold: list[RetrievalResult]) -> _Answer | Exception | None:
            stats["drift_llm_calls"] += 1
            try:
                async with semaphore:
                    raw = await self.primer.ainvoke(
                        {
                            "query": query.query,
                            "community_reports": "\n\n".join(r.content for r in fold),
                            "num_follow_ups": _FOLLOW_UPS_PER_ANSWER,
                            "target_language": self.query_language(query),
                        },
                        config,
                    )
            except Exception as e:  # noqa: BLE001 - classified by _degrade
                self._degrade(DegradationStep.DRIFT_PRIMER, "DRIFT primer", e)
                return e
            try:
                payload = await self.json_answers(answer_keys=_PRIMER_KEYS).aparse(
                    str(raw)
                )
            except OutputParserException as e:
                logger.warning("DRIFT primer returned no JSON; fold skipped")
                record_degradation(
                    DegradationStep.DRIFT_PRIMER, "primer answer unreadable", e
                )
                return None
            # The fold's reports are the answer's evidence, as an action's
            # local search results are: the primer answer reaches the answer
            # model, so the reports it was written from are its sources.
            return _Answer(
                query=query.query,
                text=str(payload.get("intermediate_answer", "")).strip(),
                score=parse_llm_score(payload.get("score", 0)),
                follow_ups=_follow_ups(payload),
                evidence=list(fold),
                read=list(fold),
            )

        primed = await gather_or_cancel(prime(fold) for fold in folds)
        failed = [p for p in primed if isinstance(p, Exception)]
        if failed and len(failed) == len(primed) and not self.ignore_errors:
            raise failed[0]
        return [answer for answer in primed if isinstance(answer, _Answer)]

    @staticmethod
    def _folds(items: list[RetrievalResult], folds: int) -> list[list[RetrievalResult]]:
        """``items`` in ``folds`` contiguous near-equal parts (none empty)."""
        count = min(folds, len(items))
        if count == 0:
            return []
        size, extra = divmod(len(items), count)
        parts, start = [], 0
        for i in range(count):
            end = start + size + (1 if i < extra else 0)
            parts.append(items[start:end])
            start = end
        return parts

    # ------------------------------------------------------------------ #
    # Actions
    # ------------------------------------------------------------------ #

    async def _act(
        self,
        follow_up: str,
        query: SearchQuery,
        semaphore: asyncio.Semaphore,
        stats: dict[str, Any],
        config: RunnableConfig | None,
        context_tokens: int | None = None,
    ) -> _Answer:
        """A local search for ``follow_up`` and its answer from that context
        (of ``context_tokens``, by default ``local_context_tokens``).

        A failure that is not fatal leaves the answer without text, keeps
        what the local search found, and records the error on the answer;
        the round decides whether the query can go on without it.
        """
        async with semaphore:
            sub_query = query.model_copy(
                update={"query": follow_up, "entity_focus": [], "optional_keywords": []}
            )
            try:
                local = await self.local_search.asearch(sub_query, config)
            except Exception as e:  # noqa: BLE001 - classified by _degrade
                self._degrade(DegradationStep.DRIFT_ACTION, "DRIFT action search", e)
                return _Answer(query=follow_up, text="", score=0, error=e)
            selection = (local.metadata or {}).get("report_selection")
            if isinstance(selection, str):
                counts = stats["report_selection"]
                counts[selection] = counts.get(selection, 0) + 1
            # Under an input-token limit the share is planned at the meter's
            # estimate: the built context is packed down to it.
            within = (
                context_tokens if usage.remaining().input_tokens is not None else None
            )
            # Counting can be one CountTokens request per section: off the loop.
            optimized = await asyncio.to_thread(
                self.token_manager.optimize_context,
                local.results,
                follow_up,
                self.drift_config.local_context_tokens,
                estimated_within=within,
            )
            context = self.token_manager.build_context_string(optimized)
            answer = _Answer(
                query=follow_up,
                text="",
                score=0,
                evidence=local.results,
                read=[
                    local.results[s.result_index]
                    for s in optimized.sections
                    if s.result_index is not None
                    and 0 <= s.result_index < len(local.results)
                ],
            )
            if not local.results:
                return answer
            if within is not None and not optimized.sections:
                # No section of its evidence fits the share the limits leave
                # it: an action that did not fit, not one asked to answer
                # from nothing.
                self._record_limit_stop(DegradationStep.DRIFT_ACTION, 0, 1)
                answer.limited = True
                return answer
            try:
                raw = await self.action.ainvoke(
                    {
                        "query": follow_up,
                        "global_query": query.query,
                        "context": context,
                        "num_follow_ups": _FOLLOW_UPS_PER_ANSWER,
                        "target_language": self.query_language(query),
                    },
                    config,
                )
            except Exception as e:  # noqa: BLE001 - classified by _degrade
                if self._stopped_by_limits(DegradationStep.DRIFT_ACTION, e):
                    # Its evidence stays: the follow-up's local search found it.
                    answer.limited = True
                    return answer
                stats["drift_llm_calls"] += 1
                self._degrade(DegradationStep.DRIFT_ACTION, "DRIFT action", e)
                answer.error = e
                return answer
            stats["drift_llm_calls"] += 1
        try:
            payload = await self.json_answers(answer_keys=_ACTION_KEYS).aparse(str(raw))
        except OutputParserException as e:
            logger.warning(
                "DRIFT action returned no JSON for %s", text_digest(follow_up)
            )
            record_degradation(
                DegradationStep.DRIFT_ACTION, "action answer unreadable", e
            )
            return answer
        answer.text = str(payload.get("response", "") or "").strip()
        answer.score = parse_llm_score(payload.get("score", 0))
        answer.follow_ups = _follow_ups(payload)
        return answer

    def _answer_tokens(self, answers: list[_Answer]) -> int:
        """What the answer reads of DRIFT's work, kept for it under the
        query's limits: the intermediate answers so far, at most
        ``max_answer_tokens`` and at least the least context."""
        written = sum(estimate_token_count(a.text) for a in answers if a.text)
        return max(
            self._min_context_tokens,
            min(self.drift_config.max_answer_tokens, written),
        )

    def _degrade(self, step: str, label: str, error: Exception) -> None:
        """Record ``step``'s failure (``label``) in the query's degradation
        ledger, or raise ``error`` when it is fatal.

        A fatal error (no model access, an unusable store) fails the query.
        Any other leaves its own answer out: the caller raises it only when
        nothing of that step succeeded and ``ignore_errors`` is off.
        """
        if is_fatal_retrieval_error(error):
            raise error
        log_failure(logger, logging.ERROR, "%s failed: %s", label, error=error)
        record_degradation(step, label, error)

    # ------------------------------------------------------------------ #
    # Reduce
    # ------------------------------------------------------------------ #

    async def _answers_items(
        self, answers: list[_Answer], lineages: dict[str, dict[str, list[str]]]
    ) -> list[RetrievalResult]:
        """The intermediate answers, best first, within ``max_answer_tokens``,
        one context section each.

        LLM output rather than retrieved evidence: the answer model reads
        them, but they are never reported as sources. Each carries the
        lineage of what it was written from (its fold's reports, or its
        local context: their text units and documents; a report's are those
        of the records its findings cite, ``lineages`` from
        ``_findings_lineages``, or its community's where they cite none), so
        the answer cites
        it ``[n]`` as MS GraphRAG's DRIFT reduce cites ``[Data: Sources
        (...)]``.
        """
        ranked = sorted(
            (a for a in answers if a.text), key=lambda a: a.score, reverse=True
        )
        if not ranked:
            return []
        entries = [
            f"Intermediate answer (score {a.score}/100) to: {a.query}\n{a.text}"
            for a in ranked
        ]
        costs = await asyncio.to_thread(self.token_manager.count_tokens_many, entries)
        packed: list[tuple[_Answer, str]] = []
        used = 0
        for answer, entry, cost in zip(ranked, entries, costs, strict=True):
            if packed and used + cost > self.drift_config.max_answer_tokens:
                break
            packed.append((answer, entry))
            used += cost
        items: list[RetrievalResult] = []
        for rank, (answer, entry) in enumerate(packed, start=1):
            lineage = evidence_lineage(answer.read, lineages)
            items.append(
                RetrievalResult(
                    content=entry,
                    score=answer.score / 100.0,
                    source=f"synthesized_drift_answers_{rank}",
                    retriever_type=SectionType.GENERAL,
                    metadata={
                        "synthesized": True,
                        "citable": bool(
                            lineage["text_unit_ids"] or lineage["document_ids"]
                        ),
                        "intermediate_answer_rank": rank,
                        "intermediate_answers": len(packed),
                        **lineage,
                    },
                )
            )
        return items

    @staticmethod
    def _dedupe(results: list[RetrievalResult]) -> list[RetrievalResult]:
        """One result per stored item; the best-scored copy is kept."""
        best: dict[tuple[str, str], RetrievalResult] = {}
        for result in results:
            key = (str(result.retriever_type), str(result.source or result.content))
            kept = best.get(key)
            if kept is None or result.score > kept.score:
                best[key] = result
        return list(best.values())

    @staticmethod
    def _rank_within_type(results: list[RetrievalResult]) -> list[RetrievalResult]:
        """Order each section type of each store by native score, in place.

        Results arrive per follow-up local search, so within a type the
        arrival order ranked a later search's best item below an earlier
        one's worst. The slots each group occupies are kept, so the mix of
        types that reaches the diversity cut is unchanged; only the order
        within a group changes.
        A group is one type from one store: entities come from the graph
        (distance bands in (0, 1]) and from the entity index (BM25, or the sum
        of two kNN scores), whose scores do not compare, so sorting them
        together let one store's hits displace all of the other's.
        """

        def group(result: RetrievalResult) -> tuple[str, str | None]:
            index = (result.metadata or {}).get("_search_index")
            return result.retriever_type, None if index is None else str(index)

        by_group: dict[tuple[str, str | None], list[RetrievalResult]] = {}
        for result in results:
            by_group.setdefault(group(result), []).append(result)
        ranked = {
            key: iter(sorted(members, key=lambda r: r.score, reverse=True))
            for key, members in by_group.items()
        }
        return [next(ranked[group(result)]) for result in results]
