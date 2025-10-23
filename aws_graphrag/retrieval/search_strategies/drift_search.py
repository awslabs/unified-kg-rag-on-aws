import asyncio
import time
from typing import Any

import boto3
from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    StrOutputParser,
)

from aws_graphrag.aws import BedrockLanguageModelFactory
from aws_graphrag.core import get_logger
from aws_graphrag.models import (
    Config,
    RetrievalResult,
    RetrieverType,
    SearchQuery,
    SearchResult,
)
from aws_graphrag.prompts import (
    ConvergenceAssessmentPrompt,
    KeywordExpansionPrompt,
    QueryRefinementPrompt,
)
from aws_graphrag.retrieval.base import (
    BaseContextBuilder,
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
)
from aws_graphrag.utils import compute_hash, safe_float_parse, setup_chain

logger = get_logger(__name__)


class DriftSearchStrategy(BaseSearchStrategy):
    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        context_builder: BaseContextBuilder | None = None,
        boto_session: boto3.Session | None = None,
        entity_focus_multiplier: int = 2,
        **kwargs: Any,
    ):
        super().__init__(config, retrievers, context_builder, boto_session, **kwargs)
        self.drift_config = self.config.search.drift_search
        self.neptune_retriever = retrievers.get(RetrieverType.NEPTUNE.value)
        self.opensearch_retriever = retrievers.get(RetrieverType.OPENSEARCH.value)
        self.entity_focus_multiplier = entity_focus_multiplier
        self.ignore_errors = config.processing.ignore_errors

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

    async def asearch(self, query: SearchQuery) -> SearchResult:
        start_time = time.time()
        logger.info(
            f"Drift search started - query: '{query.query[:50]}...' ('{query.search_type.value}')"
        )

        candidate_communities = await self._find_candidate_communities(query)
        if not candidate_communities:
            logger.warning("No candidate communities found, proceeding with empty seed")

        community_ids = self._get_ids(candidate_communities, "community_id")
        logger.debug(
            f"Found {len(community_ids)} candidate communities: '{', '.join(community_ids[:5])}"
            f"{'...' if len(community_ids) > 5 else ''}'"
        )

        current_query = query.model_copy(deep=True)
        all_results = []
        seen_hashes: set[str] = set()
        metrics: list[dict[str, Any]] = []

        all_results.extend(candidate_communities)
        self._update_seen_content(candidate_communities, seen_hashes)

        for iteration in range(self.drift_config.max_iterations):
            if await self._should_stop(iteration, metrics, query.query):
                logger.info(f"Convergence achieved at iteration {iteration}")
                break

            current_query = await self._evolve_query(
                current_query, query.query, all_results, iteration
            )
            logger.info(
                f"Iteration {iteration}: evolved query='{current_query.query}', "
                f"optional keywords='{', '.join(current_query.optional_keywords)}'"
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
                    f"Early stop at iteration {iteration}: "
                    f"improvement ratio {improvement_ratio:.2f} below threshold"
                )
                break

        final_results = self.hybrid_scorer.fuse_and_rerank_results(
            {"results": all_results},
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
        )
        processing_time = time.time() - start_time
        self._record_search_metrics(processing_time, len(all_results), len(metrics))

        logger.info(
            f"Search completed: {len(metrics)} iterations, {len(final_results)} results in {processing_time:.3f}s"
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

    async def _find_candidate_communities(
        self, query: SearchQuery
    ) -> list[RetrievalResult]:
        if not self.opensearch_retriever:
            return []

        search_query = query.model_copy(deep=True)
        search_query.index_prefixes = [
            self.config.indexing.opensearch.community_reports_index_prefix
        ]
        search_query.top_k = self.config.search.drift_search.initial_top_k

        try:
            return await self.opensearch_retriever.aretrieve(search_query)
        except Exception as e:
            logger.error(f"Failed to find candidate communities: {e}")
            return []

    @staticmethod
    def _update_seen_content(
        results: list[RetrievalResult], seen_hashes: set[str]
    ) -> None:
        for result in results:
            seen_hashes.add(compute_hash(result.content, algorithm="md5", length=16))

    async def _should_stop(
        self, iteration: int, metrics: list[dict[str, Any]], original_query: str
    ) -> bool:
        if iteration >= self.drift_config.max_iterations:
            return True

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

            logger.error(f"Convergence assessment failed: {e}")
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

            logger.error(f"Query evolution failed: {e}")

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
                summaries.append(f"Community: {result.content[:200]}...")
            else:
                summaries.append(f"Item: {result.content[:150]}...")

        return "\n".join(summaries[: self.drift_config.summary_length])

    async def _execute_search_iteration(
        self, query: SearchQuery
    ) -> list[RetrievalResult]:
        tasks = []
        candidate_entity_ids = await self._find_candidate_entities_for_iteration(query)

        if self.neptune_retriever and candidate_entity_ids:
            neptune_query = query.model_copy(deep=True)
            neptune_query.query = ""
            neptune_query.entity_focus = []
            neptune_query.filters = (neptune_query.filters or {}).copy()
            neptune_query.filters["id"] = candidate_entity_ids
            tasks.append(self.neptune_retriever.aretrieve(neptune_query))

        if self.opensearch_retriever:
            opensearch_query = query.model_copy(deep=True)
            opensearch_query.top_k = query.top_k
            tasks.append(self.opensearch_retriever.aretrieve(opensearch_query))

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
        if not self.opensearch_retriever:
            return []

        n_candidates = len(query.entity_focus) * self.entity_focus_multiplier
        entity_search_query = query.model_copy(deep=True)
        entity_search_query.index_prefixes = [
            self.config.indexing.opensearch.entities_index_prefix
        ]
        entity_search_query.top_k = n_candidates
        entity_search_query.retrieval_multiplier = 1

        try:
            results = await self.opensearch_retriever.aretrieve(entity_search_query)
            return [
                result.metadata.get("id", result.source)
                for result in results
                if result.metadata or result.source
            ]
        except Exception as e:
            logger.error(f"Failed to find candidate entities: {e}")
            return []

    @staticmethod
    def _filter_unique_results(
        results: list[RetrievalResult], seen_hashes: set[str]
    ) -> list[RetrievalResult]:
        return [
            result
            for result in results
            if compute_hash(result.content, algorithm="md5", length=16)
            not in seen_hashes
        ]

    def _record_search_metrics(
        self, processing_time: float, results_count: int, iterations: int
    ) -> None:
        self._record_timing("processing_time", processing_time)
        self._record_metric("retrieved_count", results_count)
        self._record_metric("iterations_completed", iterations)
