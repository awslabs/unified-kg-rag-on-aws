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
3. REDUCE — the ranked intermediate answers are one synthesized context
   section and the local evidence behind them, reranked once, follows it;
   the answer model reduces them into the answer.

``search.max_concurrency`` primer calls and actions run at a time.
"""

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig

from unified_kg_rag.adapters.llm.chain_factory import setup_chain
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
    is_fatal_retrieval_error,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.adapters.search_strategies.local_search import (
    LocalSearchStrategy,
)
from unified_kg_rag.domain.models import (
    Config,
    ModelRole,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
    language_name,
)
from unified_kg_rag.domain.prompts import (
    DriftActionPrompt,
    DriftHydePrompt,
    DriftPrimerPrompt,
)
from unified_kg_rag.domain.retrieval.strategy_registry import register_strategy
from unified_kg_rag.shared import get_logger, observability
from unified_kg_rag.shared.utils import (
    parse_llm_json,
    safe_float_parse,
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
    # The non-fatal error that left the answer without text, if any.
    error: Exception | None = None


def _score(value: Any) -> int:
    """A 0-100 integer score from untrusted LLM output (non-finite -> 0)."""
    number = safe_float_parse(str(value), default_value=0.0)
    if number is None or not math.isfinite(number):
        return 0
    return max(0, min(100, int(number)))


def _follow_ups(payload: dict[str, Any]) -> list[str]:
    raw = payload.get("follow_up_queries")
    if not isinstance(raw, list):
        return []
    return [str(q).strip() for q in raw if str(q).strip()]


@register_strategy(SearchStrategy.DRIFT, needs_community_reports=True)
class DriftSearchStrategy(BaseSearchStrategy):
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
        # Follow-up queries are written in the corpus language so they hit the
        # language-analyzed index.
        self.target_language = language_name(
            config.processing.translation.target_language
        )

        factory = self.providers.llm_factory
        parser = StrOutputParser()
        self.hyde: Runnable = setup_chain(
            factory=factory,
            model_id=self.config.model_id(ModelRole.DRIFT_PRIMER),
            prompt_class=DriftHydePrompt,
            parser=parser,
            custom_prompts=self.config.custom_prompts,
        )
        self.primer: Runnable = setup_chain(
            factory=factory,
            model_id=self.config.model_id(ModelRole.DRIFT_PRIMER),
            prompt_class=DriftPrimerPrompt,
            parser=parser,
            custom_prompts=self.config.custom_prompts,
        )
        self.action: Runnable = setup_chain(
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
        stats: dict[str, Any] = {"drift_llm_calls": 0}
        degraded: list[str] = []
        semaphore = asyncio.Semaphore(self.max_concurrency)

        with observability.span("kg_rag.drift.primer") as span:
            reports = await self._primer_reports(query, stats, config, degraded)
            answers = await self._prime(
                query, reports, semaphore, stats, config, degraded
            )
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
        # Whether any action ran without an error.
        acted = False
        while queue and depth < self.drift_config.n_depth:
            queue.sort(key=lambda item: (-item[0], item[1]))
            batch = [q for _, _, q in queue[: self.drift_config.k_followups]]
            del queue[: self.drift_config.k_followups]
            with observability.span(
                "kg_rag.drift.actions", depth=depth, actions=len(batch)
            ):
                done = await gather_or_cancel(
                    self._act(sub, query, semaphore, stats, config, degraded)
                    for sub in batch
                )
            actions += len(batch)
            depth += 1
            failed = [a.error for a in done if a.error is not None]
            acted = acted or len(failed) < len(done)
            for answer in done:
                answers.append(answer)
                enqueue(answer)
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
        synthesized = await self._answers_item(answers)
        results = ([synthesized] if synthesized else []) + fused

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
                **({"degraded_error": degraded[0]} if degraded else {}),
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
        degraded: list[str],
    ) -> list[RetrievalResult]:
        """The ``k_followups`` reports most similar to the expanded query."""
        retriever = self.document_retriever
        if retriever is None:
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
        try:
            stats["drift_llm_calls"] += 1
            hypothetical = await self.hyde.ainvoke(
                {"query": query.query, "template": template[0].content}, config
            )
            expanded = str(hypothetical).strip() or query.query
        except Exception as e:
            if not self.ignore_errors:
                raise
            self._degrade("DRIFT query expansion", e, degraded)
        return await self._safe_aretrieve(
            retriever,
            self._reports_query(
                query, query=expanded, top_k=self.drift_config.k_followups
            ),
            "DRIFT primer reports",
        )

    async def _prime(
        self,
        query: SearchQuery,
        reports: list[RetrievalResult],
        semaphore: asyncio.Semaphore,
        stats: dict[str, Any],
        config: RunnableConfig | None,
        degraded: list[str],
    ) -> list[_Answer]:
        """One intermediate answer per fold of the primer reports, at most
        ``search.max_concurrency`` folds at a time (as the actions)."""
        folds = self._folds(reports, self.drift_config.primer_folds)

        async def prime(fold: list[RetrievalResult]) -> _Answer | Exception | None:
            stats["drift_llm_calls"] += 1
            try:
                async with semaphore:
                    raw = await self.primer.ainvoke(
                        {
                            "query": query.query,
                            "community_reports": "\n\n".join(r.content for r in fold),
                            "num_follow_ups": _FOLLOW_UPS_PER_ANSWER,
                            "target_language": self.target_language,
                        },
                        config,
                    )
            except Exception as e:
                self._degrade("DRIFT primer", e, degraded)
                return e
            payload = parse_llm_json(str(raw))
            if not payload:
                logger.warning("DRIFT primer returned no JSON; fold skipped")
                return None
            return _Answer(
                query=query.query,
                text=str(payload.get("intermediate_answer", "")).strip(),
                score=_score(payload.get("score", 0)),
                follow_ups=_follow_ups(payload),
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
        degraded: list[str],
    ) -> _Answer:
        """A local search for ``follow_up`` and its answer from that context.

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
            except Exception as e:
                self._degrade("DRIFT action search", e, degraded)
                return _Answer(query=follow_up, text="", score=0, error=e)
            # Counting can be one CountTokens request per section: off the loop.
            optimized = await asyncio.to_thread(
                self.token_manager.optimize_context,
                local.results,
                follow_up,
                self.drift_config.local_context_tokens,
            )
            context = self.token_manager.build_context_string(optimized)
            answer = _Answer(query=follow_up, text="", score=0, evidence=local.results)
            if not local.results:
                return answer
            stats["drift_llm_calls"] += 1
            try:
                raw = await self.action.ainvoke(
                    {
                        "query": follow_up,
                        "global_query": query.query,
                        "context": context,
                        "num_follow_ups": _FOLLOW_UPS_PER_ANSWER,
                        "target_language": self.target_language,
                    },
                    config,
                )
            except Exception as e:
                self._degrade("DRIFT action", e, degraded)
                answer.error = e
                return answer
        payload = parse_llm_json(str(raw))
        if not payload:
            logger.warning(
                "DRIFT action returned no JSON for %s", text_digest(follow_up)
            )
            return answer
        answer.text = str(payload.get("response", "") or "").strip()
        answer.score = _score(payload.get("score", 0))
        answer.follow_ups = _follow_ups(payload)
        return answer

    def _degrade(self, step: str, error: Exception, degraded: list[str]) -> None:
        """Record ``step``'s failure, or raise ``error`` when it is fatal.

        A fatal error (no model access, an unusable store) fails the query.
        Any other leaves its own answer out: the caller raises it only when
        nothing of that step succeeded and ``ignore_errors`` is off.
        """
        if is_fatal_retrieval_error(error):
            raise error
        logger.error("%s failed: %s", step, error)
        degraded.append(type(error).__name__)

    # ------------------------------------------------------------------ #
    # Reduce
    # ------------------------------------------------------------------ #

    async def _answers_item(self, answers: list[_Answer]) -> RetrievalResult | None:
        """The intermediate answers, best first, within ``max_answer_tokens``.

        LLM output rather than retrieved evidence: the answer model reads it,
        but it is never reported as a source.
        """
        ranked = sorted(
            (a for a in answers if a.text), key=lambda a: a.score, reverse=True
        )
        if not ranked:
            return None
        entries = [f"- (score {a.score}) Q: {a.query}\n  A: {a.text}" for a in ranked]
        costs = await asyncio.to_thread(self.token_manager.count_tokens_many, entries)
        packed: list[str] = []
        used = 0
        for entry, cost in zip(entries, costs, strict=True):
            if packed and used + cost > self.drift_config.max_answer_tokens:
                break
            packed.append(entry)
            used += cost
        return RetrievalResult(
            content=(
                "Intermediate answers to the question and its follow-up "
                "questions, ranked by how well each addresses the question "
                "(0-100):\n\n" + "\n\n".join(packed)
            ),
            score=1.0,
            source="synthesized_drift_answers",
            retriever_type=SectionType.GENERAL.value,
            metadata={"synthesized": True, "intermediate_answers": len(packed)},
        )

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
