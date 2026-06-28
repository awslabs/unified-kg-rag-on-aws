# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from typing import Any

import boto3
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    StrOutputParser,
)

from unified_kg_rag.adapters.aws import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
)
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from unified_kg_rag.domain.prompts import (
    ConvergenceAssessmentPrompt,
    DriftPrimerPrompt,
    KeywordExpansionPrompt,
    QueryRefinementPrompt,
)
from unified_kg_rag.domain.retrieval.strategy_registry import register_strategy
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import (
    compute_hash,
    parse_llm_json,
    safe_float_parse,
)

logger = get_logger(__name__)


@register_strategy(SearchStrategy.DRIFT)
class DriftSearchStrategy(BaseSearchStrategy):
    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        boto_session: boto3.Session | None = None,
        entity_focus_multiplier: int = 2,
        **kwargs: Any,
    ):
        super().__init__(config, retrievers, boto_session, **kwargs)
        self.drift_config = self.config.search.drift_search
        self.entity_focus_multiplier = entity_focus_multiplier
        self.ignore_errors = config.processing.ignore_errors
        # Keyword expansion / query refinement must produce terms in the corpus
        # language so they hit the language-analyzed index (not English-biased).
        self.target_language = config.processing.translation.target_language.value

        factory = BedrockLanguageModelFactory(
            config=config,
            boto_session=boto_session,
            region_name=self.config.aws.bedrock.region_name,
        )

        str_output_parser = StrOutputParser()
        self.query_refiner = setup_chain(
            factory=factory,
            model_id=self.drift_config.query_refinement_model_id,
            prompt_class=QueryRefinementPrompt,
            parser=str_output_parser,
        )
        self.keyword_expander = setup_chain(
            factory=factory,
            model_id=self.drift_config.keyword_expansion_model_id,
            prompt_class=KeywordExpansionPrompt,
            parser=CommaSeparatedListOutputParser(),
        )
        self.convergence_assessor = setup_chain(
            factory=factory,
            model_id=self.drift_config.convergence_assessment_model_id,
            prompt_class=ConvergenceAssessmentPrompt,
            parser=str_output_parser,
        )
        # The HyDE primer chain is only built when the primer path is enabled,
        # so the default DRIFT flow constructs no extra chain.
        if self.drift_config.enable_primer:
            self.primer = setup_chain(
                factory=factory,
                model_id=self.drift_config.primer_model_id,
                prompt_class=DriftPrimerPrompt,
                parser=str_output_parser,
            )

    async def asearch(self, query: SearchQuery) -> SearchResult:
        start_time = time.time()
        logger.info(
            "Drift search started - query: '%s...' ('%s')",
            query.query[:50],
            query.search_type.value,
        )

        candidate_communities = await self._find_candidate_communities(query)
        if not candidate_communities:
            logger.warning("No candidate communities found, proceeding with empty seed")

        community_ids = self._get_ids(candidate_communities, "community_id")
        logger.debug(
            "Found %s candidate communities: '%s%s'",
            len(community_ids),
            ", ".join(community_ids[:5]),
            "..." if len(community_ids) > 5 else "",
        )

        all_results: list[RetrievalResult] = list(candidate_communities)
        seen_hashes: set[str] = set()
        metrics: list[dict[str, Any]] = []
        self._update_seen_content(candidate_communities, seen_hashes)

        if self.drift_config.enable_primer:
            await self._primer_search(
                query, candidate_communities, all_results, seen_hashes, metrics
            )
        else:
            await self._iterative_search(query, all_results, seen_hashes, metrics)

        final_results = self.hybrid_scorer.fuse_and_rerank_results(
            {"results": all_results},
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
        )
        processing_time = time.time() - start_time
        self._record_search_metrics(processing_time, len(all_results), len(metrics))

        logger.info(
            "Search completed: %s iterations, %s results in %.3fs",
            len(metrics),
            len(final_results),
            processing_time,
        )

        return SearchResult(
            query=query,
            results=final_results,
            total_results=len(final_results),
            search_strategy="drift_search",
            processing_time=processing_time,
            metadata={
                "iterations_completed": len(metrics),
                "iteration_metrics": metrics,
            },
        )

    async def _iterative_search(
        self,
        query: SearchQuery,
        all_results: list[RetrievalResult],
        seen_hashes: set[str],
        metrics: list[dict[str, Any]],
    ) -> None:
        """Original DRIFT loop: carry one mutating query forward each iteration."""
        current_query = query.model_copy(deep=True)
        for iteration in range(self.drift_config.max_iterations):
            if await self._should_stop(iteration, metrics, query.query):
                logger.info("Convergence achieved at iteration %s", iteration)
                break

            current_query = await self._evolve_query(
                current_query, query.query, all_results, iteration
            )
            logger.info(
                "Iteration %s: evolved query='%s', optional keywords='%s'",
                iteration,
                current_query.query,
                ", ".join(current_query.optional_keywords),
            )

            iteration_results = await self._execute_search_iteration(current_query)
            unique_new = self._filter_unique_results(iteration_results, seen_hashes)

            self._update_seen_content(unique_new, seen_hashes)
            all_results.extend(unique_new)

            metrics.append(
                {
                    "iteration": iteration,
                    "query": current_query.query,
                    "retrieved": len(iteration_results),
                    "unique_new": len(unique_new),
                }
            )

            improvement_ratio = len(unique_new) / max(len(iteration_results), 1)
            if (
                iteration > 0
                and improvement_ratio < self.drift_config.improvement_threshold
            ):
                logger.info(
                    "Early stop at iteration %s: "
                    "improvement ratio %.2f below threshold",
                    iteration,
                    improvement_ratio,
                )
                break

    async def _primer_search(
        self,
        query: SearchQuery,
        candidate_communities: list[RetrievalResult],
        all_results: list[RetrievalResult],
        seen_hashes: set[str],
        metrics: list[dict[str, Any]],
    ) -> None:
        """MS GraphRAG primer flow: HyDE primer -> per-follow-up local searches.

        The primer drafts a hypothetical answer from the seed community reports
        and decomposes the query into specific follow-up sub-queries; each
        follow-up is then run as its own search iteration (capped by
        ``max_iterations``). Falls back to the iterative loop if the primer
        yields no follow-up queries, so the strategy never returns seed-only.
        """
        follow_ups, intermediate_answer = await self._run_primer(
            query, candidate_communities
        )

        # Seed the primer's hypothetical (HyDE) answer into the result set so it
        # informs fusion + final synthesis (MS GraphRAG carries this forward
        # rather than discarding it). Deduped like any other result.
        if intermediate_answer:
            seed = RetrievalResult(
                content=intermediate_answer,
                score=1.0,
                source="drift_primer",
                retriever_type="drift_primer",
                metadata={"source": "primer_intermediate_answer"},
            )
            if (
                seed.content
                and compute_hash(seed.content, length=16) not in seen_hashes
            ):
                all_results.append(seed)
                self._update_seen_content([seed], seen_hashes)

        if not follow_ups:
            logger.info("Primer produced no follow-ups; using iterative loop")
            await self._iterative_search(query, all_results, seen_hashes, metrics)
            return

        for iteration, follow_up in enumerate(
            follow_ups[: self.drift_config.max_iterations]
        ):
            follow_up_query = query.model_copy(deep=True)
            follow_up_query.query = follow_up
            logger.info("Primer follow-up %s: '%s'", iteration, follow_up)

            iteration_results = await self._execute_search_iteration(follow_up_query)
            unique_new = self._filter_unique_results(iteration_results, seen_hashes)
            self._update_seen_content(unique_new, seen_hashes)
            all_results.extend(unique_new)

            metrics.append(
                {
                    "iteration": iteration,
                    "query": follow_up,
                    "retrieved": len(iteration_results),
                    "unique_new": len(unique_new),
                    "source": "primer_follow_up",
                }
            )

    async def _run_primer(
        self, query: SearchQuery, candidate_communities: list[RetrievalResult]
    ) -> tuple[list[str], str]:
        """Run the HyDE primer; return (follow-up sub-queries, intermediate answer).

        The ``intermediate_answer`` is the primer's hypothetical answer drafted
        from the seed community summaries (HyDE). The caller seeds it into the
        result set so it informs fusion/answer synthesis, matching MS GraphRAG's
        DRIFT, which carries the primer answer forward rather than discarding it.
        """
        reports = "\n".join(
            f"- {r.content[: self.drift_config.summary_char_limit]}"
            for r in candidate_communities[: self.drift_config.initial_top_k]
        )
        try:
            raw = await self.primer.ainvoke(
                {
                    "query": query.query,
                    "community_reports": reports or "(no community summaries found)",
                    "num_follow_ups": self.drift_config.primer_follow_ups,
                }
            )
            payload = parse_llm_json(raw)
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.error("DRIFT primer failed: %s", e)
            return [], ""

        raw_follow_ups = payload.get("follow_up_queries")
        follow_ups = (
            [str(f).strip() for f in raw_follow_ups if str(f).strip()]
            if isinstance(raw_follow_ups, list)
            else []
        )
        intermediate_answer = str(payload.get("intermediate_answer", "")).strip()
        return follow_ups, intermediate_answer

    async def _find_candidate_communities(
        self, query: SearchQuery
    ) -> list[RetrievalResult]:
        if not self.document_retriever:
            return []

        search_query = query.model_copy(deep=True)
        search_query.index_prefixes = [
            self.config.indexing.opensearch.community_reports_index_prefix
        ]
        search_query.top_k = self.config.search.drift_search.initial_top_k

        try:
            return await self.document_retriever.aretrieve(search_query)
        except Exception as e:
            logger.error("Failed to find candidate communities: %s", e)
            return []

    @staticmethod
    def _update_seen_content(
        results: list[RetrievalResult], seen_hashes: set[str]
    ) -> None:
        for result in results:
            seen_hashes.add(compute_hash(result.content, length=16))

    async def _should_stop(
        self, iteration: int, metrics: list[dict[str, Any]], original_query: str
    ) -> bool:
        # The caller loops over range(max_iterations), so the hard cap is already
        # enforced there; this method only decides EARLY convergence.
        if iteration > 1:
            recent_gains = [m["unique_new"] for m in metrics[-2:]]
            if all(gain < 2 for gain in recent_gains):
                return True

        if iteration > 2 and await self._assess_convergence_with_llm(
            original_query, iteration, metrics
        ):
            return True

        return False

    async def _assess_convergence_with_llm(
        self, original_query: str, iteration: int, metrics: list[dict[str, Any]]
    ) -> bool:
        if not metrics:
            return False

        try:
            llm_output = await self.convergence_assessor.ainvoke(
                {
                    "original_query": original_query,
                    "iterations": iteration,
                    "total_results": sum(m["unique_new"] for m in metrics),
                    "new_results": metrics[-1]["unique_new"],
                }
            )
            parsed_score = safe_float_parse(llm_output, default_value=0.5) or 0.0
            return parsed_score >= self.drift_config.convergence_threshold

        except Exception as e:
            if not self.ignore_errors:
                raise

            logger.error("Convergence assessment failed: %s", e)
            return False

    async def _evolve_query(
        self,
        query: SearchQuery,
        original_query: str,
        results: list[RetrievalResult],
        iteration: int,
        max_keywords: int = 20,
    ) -> SearchQuery:
        evolved_query = query.model_copy(deep=True)
        tasks = {}

        if self.drift_config.enable_query_refinement:
            summary = self._summarize_results(results)
            tasks["refinement"] = self.query_refiner.ainvoke(
                {
                    "original_query": original_query,
                    "results_summary": summary,
                    "iteration": iteration,
                    "target_language": self.target_language,
                }
            )

        if self.drift_config.enable_keyword_extraction:
            entities = [
                r.metadata.get("name")
                for r in results[: self.drift_config.n_entities]
                if r.metadata
            ]
            tasks["expansion"] = self.keyword_expander.ainvoke(
                {
                    "query": original_query,
                    "entities": entities,
                    "topics": [],
                    "max_keywords": max_keywords,
                    "target_language": self.target_language,
                }
            )

        if not tasks:
            return evolved_query

        try:
            task_results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            results_map = dict(zip(tasks.keys(), task_results, strict=True))

            refinement = results_map.get("refinement")
            if (
                refinement is not None
                and not isinstance(refinement, Exception)
                and isinstance(refinement, str)
            ):
                refinement = refinement.strip()
                if refinement:
                    evolved_query.query = refinement

            expansion = results_map.get("expansion")
            if (
                expansion is not None
                and not isinstance(expansion, Exception)
                and isinstance(expansion, list)
            ):
                if expansion:
                    evolved_query.optional_keywords = expansion

        except Exception as e:
            if not self.ignore_errors:
                raise

            logger.error("Query evolution failed: %s", e)

        return evolved_query

    def _summarize_results(self, results: list[RetrievalResult]) -> str:
        if not results:
            return "No information gathered yet. Start by exploring broad topics related to the original query."

        sorted_results = sorted(results, key=lambda x: x.score or 0.0, reverse=True)
        summaries = []
        for result in sorted_results:
            if result.metadata and "community_reports" in result.metadata.get(
                "_search_index", ""
            ):
                summaries.append(
                    f"Community: {result.content[: self.drift_config.summary_char_limit]}..."
                )
            else:
                summaries.append(
                    f"Item: {result.content[: self.drift_config.summary_char_limit]}..."
                )

        return "\n".join(summaries[: self.drift_config.summary_length])

    async def _execute_search_iteration(
        self, query: SearchQuery
    ) -> list[RetrievalResult]:
        tasks = []
        candidate_entity_ids = await self._find_candidate_entities_for_iteration(query)

        if self.graph_retriever and candidate_entity_ids:
            graph_query = query.model_copy(deep=True)
            graph_query.query = ""
            graph_query.entity_focus = []
            graph_query.filters = (graph_query.filters or {}).copy()
            graph_query.filters["id"] = candidate_entity_ids
            tasks.append(self.graph_retriever.aretrieve(graph_query))

        if self.document_retriever:
            document_query = query.model_copy(deep=True)
            document_query.top_k = query.top_k
            tasks.append(self.document_retriever.aretrieve(document_query))

        if not tasks:
            return []

        results_lists = await asyncio.gather(*tasks, return_exceptions=True)
        return [
            item
            for result_list in results_lists
            if isinstance(result_list, list)
            for item in result_list
        ]

    async def _find_candidate_entities_for_iteration(
        self, query: SearchQuery
    ) -> list[str]:
        if not self.document_retriever:
            return []

        n_candidates = len(query.entity_focus) * self.entity_focus_multiplier
        entity_search_query = query.model_copy(deep=True)
        entity_search_query.index_prefixes = [
            self.config.indexing.opensearch.entities_index_prefix
        ]
        entity_search_query.top_k = n_candidates
        entity_search_query.retrieval_multiplier = 1

        try:
            results = await self.document_retriever.aretrieve(entity_search_query)
            return [
                str(result.metadata.get("id") or result.source)
                for result in results
                if result.metadata or result.source
            ]
        except Exception as e:
            logger.error("Failed to find candidate entities: %s", e)
            return []

    @staticmethod
    def _filter_unique_results(
        results: list[RetrievalResult], seen_hashes: set[str]
    ) -> list[RetrievalResult]:
        return [
            result
            for result in results
            if compute_hash(result.content, length=16) not in seen_hashes
        ]

    def _record_search_metrics(
        self, processing_time: float, results_count: int, iterations: int
    ) -> None:
        self._record_timing("processing_time", processing_time)
        self._record_metric("retrieved_count", results_count)
        self._record_metric("iterations_completed", iterations)
