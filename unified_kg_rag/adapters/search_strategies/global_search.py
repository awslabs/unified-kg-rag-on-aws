# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from typing import Any

import boto3
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableConfig

from unified_kg_rag.adapters.aws import BedrockLanguageModelFactory
from unified_kg_rag.adapters.aws.chain_factory import setup_chain
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
)
from unified_kg_rag.domain.prompts import (
    CommunityRelevancePrompt,
    GlobalMapPrompt,
    MapReduceSummaryPrompt,
)
from unified_kg_rag.domain.retrieval.strategy_registry import register_strategy
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import (
    BatchProcessor,
    parse_llm_json,
    safe_float_parse,
)

logger = get_logger(__name__)


class _MapPoint:
    """A scored key point produced by the global-search map step.

    Lightweight value object (not a Pydantic model) so the map/filter/rank/pack
    plumbing stays internal to this adapter.
    """

    __slots__ = ("description", "score")

    def __init__(self, description: str, score: int) -> None:
        self.description = description
        self.score = score


@register_strategy(SearchStrategy.GLOBAL)
class GlobalSearchStrategy(BaseSearchStrategy):
    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        boto_session: boto3.Session | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, retrievers, boto_session, **kwargs)
        self.global_search_config = config.search.global_search
        self.ignore_errors = config.processing.ignore_errors
        self.target_language = config.processing.translation.target_language.value

        factory = BedrockLanguageModelFactory(
            config=config,
            boto_session=boto_session,
            region_name=config.aws.bedrock.region_name,
        )

        str_output_parser = StrOutputParser()
        self.community_relevance_scorer: Runnable = setup_chain(
            factory=factory,
            model_id=self.global_search_config.community_relevance_model_id,
            prompt_class=CommunityRelevancePrompt,
            parser=str_output_parser,
        )
        # MAP step: rate community-report key points (0-100) for the query. Cheap
        # model by default since rating is cheap. StrOutputParser + a robust JSON
        # parse below keeps map output handling fault-tolerant.
        self.map_rater: Runnable = setup_chain(
            factory=factory,
            model_id=self.global_search_config.map_model_id,
            prompt_class=GlobalMapPrompt,
            parser=str_output_parser,
            custom_prompts=config.custom_prompts,
        )
        # REDUCE step: synthesize the final answer from the ranked key points.
        self.map_reducer: Runnable = setup_chain(
            factory=factory,
            model_id=self.global_search_config.map_reduce_model_id,
            prompt_class=MapReduceSummaryPrompt,
            parser=str_output_parser,
        )
        # One prepared input per map LLM call (each input already packs
        # ``map_batch_size`` reports), so BatchProcessor's own batch_size is 1;
        # max_concurrency fans the map calls out over the report batches.
        self.batch_processor = BatchProcessor(
            batch_size=1, max_concurrency=config.processing.max_concurrency
        )

    async def asearch(self, query: SearchQuery) -> SearchResult:
        start_time = time.time()
        logger.info(
            "Global search started - query: '%s...' ('%s')",
            query.query[:50],
            query.search_type.value,
        )

        retrieved_communities = await self._retrieve_and_fuse_communities(query)
        if not retrieved_communities:
            logger.warning("No results found for query: '%s...'", query.query[:50])
            return SearchResult(
                query=query,
                results=[],
                total_results=0,
                search_strategy="global_search",
                processing_time=time.time() - start_time,
                metadata={},
            )

        community_ids = self._get_ids(retrieved_communities, "community_id")
        logger.debug(
            "Found %s communities: '%s%s'",
            len(community_ids),
            ", ".join(community_ids[:5]),
            "..." if len(community_ids) > 5 else "",
        )

        selected_communities = await self._select_relevant_communities(
            retrieved_communities, query
        )
        community_ids = self._get_ids(selected_communities, "id")
        logger.debug(
            "Selected %s communities: '%s%s'",
            len(community_ids),
            ", ".join(community_ids[:5]),
            "..." if len(community_ids) > 5 else "",
        )

        final_results = await self._augment_and_rerank_communities(
            selected_communities, retrieved_communities, query
        )
        logger.debug(
            "Augmented %s items: '%s%s'",
            len(final_results),
            ", ".join(
                str(item.metadata.get("community_id") or item.metadata.get("id"))
                for item in final_results[:5]
                if item.metadata
            ),
            "..." if len(final_results) > 5 else "",
        )

        if self.global_search_config.enable_map_reduce:
            final_results = await self._apply_map_reduce(final_results, query)

        final_results = final_results[: query.top_k]
        processing_time = time.time() - start_time

        self._record_search_metrics(
            processing_time,
            len(retrieved_communities),
            len(selected_communities),
            len(final_results),
        )

        logger.info(
            "Search completed - retrieved: %s results (%.3fs)",
            len(final_results),
            processing_time,
        )

        return SearchResult(
            query=query,
            results=final_results,
            total_results=len(final_results),
            search_strategy="global_search",
            processing_time=processing_time,
            metadata={
                "retrieved_community_count": len(retrieved_communities),
                "selected_community_count": len(selected_communities),
                "map_reduce_applied": self._was_map_reduce_applied(final_results),
            },
        )

    async def _retrieve_and_fuse_communities(
        self, query: SearchQuery
    ) -> list[RetrievalResult]:
        candidate_community_reports = await self._retrieve_community_reports(query)
        if not candidate_community_reports:
            return []

        candidate_community_ids = self._get_ids(
            candidate_community_reports, "community_id"
        )
        if not candidate_community_ids:
            return candidate_community_reports

        logger.debug(
            "Found %s candidate communities: '%s%s'",
            len(candidate_community_ids),
            ", ".join(str(cid) for cid in candidate_community_ids[:5]),
            "..." if len(candidate_community_ids) > 5 else "",
        )

        expanded_community_nodes = await self._retrieve_community_nodes(
            query, candidate_community_ids
        )
        if not expanded_community_nodes:
            return candidate_community_reports

        expanded_community_ids = self._get_ids(expanded_community_nodes, "id")
        if not expanded_community_ids:
            return candidate_community_reports

        expanded_community_reports = await self._retrieve_reports_by_ids(
            expanded_community_ids, query
        )
        logger.debug(
            "Expanded to %s communities: '%s%s'",
            len(expanded_community_reports),
            ", ".join(str(cid) for cid in expanded_community_ids[:5]),
            "..." if len(expanded_community_ids) > 5 else "",
        )

        return self.hybrid_scorer.fuse_and_rerank_results(
            {
                "opensearch_candidate_community_reports": candidate_community_reports,
                "opensearch_expanded_community_reports": expanded_community_reports,
            },
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
        )

    async def _retrieve_community_reports(
        self, query: SearchQuery
    ) -> list[RetrievalResult]:
        index_prefixes = [
            self.config.indexing.opensearch.community_reports_index_prefix
        ]
        return await self._retrieve_documents(query, index_prefixes)

    async def _retrieve_reports_by_ids(
        self, community_ids: list[str], query: SearchQuery
    ) -> list[RetrievalResult]:
        if not self.document_retriever or not community_ids:
            return []

        try:
            search_query = query.model_copy(deep=True)
            search_query.query = ""
            search_query.filters = (search_query.filters or {}).copy()
            search_query.filters["community_id"] = community_ids
            search_query.top_k = len(community_ids)

            index_prefixes = [
                self.config.indexing.opensearch.community_reports_index_prefix
            ]
            return await self._retrieve_documents(search_query, index_prefixes)
        except Exception as e:
            logger.error("OpenSearch retrieval failed: %s", e)
            return []

    async def _retrieve_documents(
        self, query: SearchQuery, index_prefixes: list[str]
    ) -> list[RetrievalResult]:
        if not self.document_retriever:
            return []

        try:
            search_query = query.model_copy(deep=True)
            search_query.index_prefixes = index_prefixes
            return await self.document_retriever.aretrieve(search_query)
        except Exception as e:
            logger.error("OpenSearch retrieval failed: %s", e)
            return []

    async def _retrieve_community_nodes(
        self, query: SearchQuery, community_ids: list[str]
    ) -> list[RetrievalResult]:
        if not self.graph_retriever:
            return []

        try:
            search_query = query.model_copy(deep=True)
            search_query.query = ""
            search_query.filters = (search_query.filters or {}).copy()
            search_query.filters["id"] = community_ids
            search_query.label_prefixes = [
                self.config.indexing.neptune.community_label_prefix
            ]

            return await asyncio.wait_for(
                self.graph_retriever.aretrieve(search_query),
                timeout=self.global_search_config.graph_timeout_seconds,
            )
        except Exception as e:
            logger.error("Neptune community retrieval failed: %s", e)
            return []

    async def _select_relevant_communities(
        self, all_communities: list[RetrievalResult], query: SearchQuery
    ) -> list[RetrievalResult]:
        max_communities = (
            self.global_search_config.max_communities * query.retrieval_multiplier
        )
        if not self.global_search_config.use_dynamic_selection:
            # Without per-query LLM scoring, fall back to the community-report
            # `rank` (graph-importance, set at indexing time), tie-breaking on the
            # LLM-assigned `rating` (importance/impact severity, MS GraphRAG
            # parity) so we keep the most central + most important communities
            # rather than truncating in arbitrary retrieval order. Both missing
            # sort last.
            return sorted(
                all_communities,
                key=lambda c: (
                    float(c.metadata.get("rank", 0.0) or 0.0),
                    float(c.metadata.get("rating", 0.0) or 0.0),
                ),
                reverse=True,
            )[:max_communities]

        async def score_item(item: RetrievalResult) -> tuple[RetrievalResult, float]:
            # Return the relevance on a 0-1 scale so it is comparable to
            # relevance_threshold (config-constrained to [0.0, 1.0]). The LLM
            # emits a 0-10 score; normalizing here was previously only done for
            # the blended item.score, leaving the threshold filter (line below)
            # comparing a 0-1 threshold against a 0-10 score -> a near no-op.
            #
            # Work on a COPY: the input RetrievalResult objects are shared with
            # the fallback list and the downstream fuse/rerank, so mutating
            # item.score in place would corrupt the scores fed to fusion (and,
            # under concurrent queries sharing a strategy, race across queries).
            llm_output = await self.community_relevance_scorer.ainvoke(
                {"community_summary": item.content, "query": query.query}
            )
            parsed_score = safe_float_parse(llm_output, default_value=5.0)
            relevance_score = (parsed_score / 10.0) if parsed_score is not None else 0.0
            scored = item.model_copy()
            if parsed_score is not None:
                scored.score = ((item.score or 0.5) * 0.4) + (relevance_score * 0.6)
            return scored, relevance_score

        # return_exceptions=True so a single throttled/failed LLM scoring call
        # degrades that one community to score 0 rather than aborting the whole
        # query (matching the drift/map-reduce paths' graceful degradation).
        tasks = [score_item(item) for item in all_communities]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)

        relevance_threshold = self.global_search_config.relevance_threshold
        evaluated_items: list[tuple[RetrievalResult, float]] = []
        for original, outcome in zip(all_communities, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                if not self.ignore_errors:
                    raise outcome
                logger.warning("Community scoring failed: %s", outcome)
                # Admit the community ABOVE the threshold so a transient scoring
                # failure cannot silently drop a strong hit. Its retrieval score
                # is a raw RRF value (~1/(k+rank), e.g. 0.016) on a different
                # scale than the 0-1 relevance the threshold filters on, so
                # comparing it directly would drop it; assign the threshold value
                # to keep it as a candidate (ranked by its retrieval score below).
                evaluated_items.append((original, relevance_threshold))
            else:
                evaluated_items.append(outcome)

        relevant_items = [
            item for item, score in evaluated_items if score >= relevance_threshold
        ]
        relevant_items.sort(key=lambda x: x.score or 0.0, reverse=True)
        logger.debug(
            "Filtered %s communities based on relevance threshold %s",
            len(evaluated_items) - len(relevant_items),
            relevance_threshold,
        )

        return relevant_items[:max_communities]

    async def _augment_and_rerank_communities(
        self,
        selected: list[RetrievalResult],
        fallback: list[RetrievalResult],
        query: SearchQuery,
    ) -> list[RetrievalResult]:
        if not selected:
            return fallback

        context = await self._retrieve_community_context(selected, query)
        return self.hybrid_scorer.fuse_and_rerank_results(
            {
                "opensearch_community_reports": selected,
                "text_units": context,
            },
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
        )

    async def _retrieve_community_context(
        self, communities: list[RetrievalResult], query: SearchQuery
    ) -> list[RetrievalResult]:
        community_ids = self._get_ids(communities, "community_id")
        if not community_ids:
            return []

        search_query = query.model_copy(deep=True)
        search_query.filters = {
            **(query.filters or {}),
            "community_ids": list(community_ids),
        }
        search_query.top_k = min(query.top_k, self.global_search_config.max_text_units)
        index_prefixes = [self.config.indexing.opensearch.text_units_index_prefix]

        return await self._retrieve_documents(search_query, index_prefixes)

    async def _apply_map_reduce(
        self, results: list[RetrievalResult], query: SearchQuery
    ) -> list[RetrievalResult]:
        """MS GraphRAG global-search map-reduce.

        MAP — rate the key points in each batch of community reports (0-100).
        FILTER+RANK — drop low-scored points, sort by score descending.
        PACK — take ranked points up to ``max_map_reduce_tokens``.
        REDUCE — synthesize the answer from the ranked, packed points.

        Below ``map_reduce_min_results`` the results pass through unchanged (the
        existing direct path). If the map phase yields no usable scored points
        (e.g. every map call failed to parse), it degrades to the legacy
        concat-and-reduce path so global search never hard-fails.
        """
        if len(results) < self.global_search_config.map_reduce_min_results:
            return results

        try:
            map_points = await self._run_map_phase(results, query)
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.error("Map phase failed: %s", e)
            map_points = []

        if not map_points:
            logger.warning(
                "Map phase produced no usable key points; degrading to "
                "concat-and-reduce synthesis."
            )
            return await self._concat_reduce(results, query)

        ranked_points = self._filter_and_rank_points(map_points)
        if not ranked_points:
            logger.warning(
                "All %s map key points were filtered out by threshold %s; "
                "degrading to concat-and-reduce synthesis.",
                len(map_points),
                self.global_search_config.map_relevance_threshold,
            )
            return await self._concat_reduce(results, query)

        packed_points = self._pack_points_within_budget(ranked_points)
        return await self._reduce_from_points(packed_points, results, query)

    async def _run_map_phase(
        self, results: list[RetrievalResult], query: SearchQuery
    ) -> list[_MapPoint]:
        """Fan map calls over batches of reports and parse the scored points.

        Each batch of ``map_batch_size`` reports becomes one map LLM call;
        BatchProcessor runs them concurrently with graceful per-item fallback.
        """
        batch_size = self.global_search_config.map_batch_size
        report_batches = [
            results[i : i + batch_size] for i in range(0, len(results), batch_size)
        ]

        def prepare_inputs(
            batches: list[list[RetrievalResult]],
        ) -> list[dict[str, Any]]:
            return [
                {
                    "query": query.query,
                    "reports": self._format_reports(batch),
                    "target_language": self.target_language,
                }
                for batch in batches
            ]

        def batch_func(
            inputs: list[dict[str, Any]], config: RunnableConfig | None = None
        ) -> list[str]:
            return list(self.map_rater.batch(inputs, config=config))

        def sequential_func(single_input: dict[str, Any]) -> str:
            return str(self.map_rater.invoke(single_input))

        raw_outputs = await asyncio.to_thread(
            self.batch_processor.execute_with_fallback,
            items_to_process=report_batches,
            prepare_inputs_func=prepare_inputs,
            batch_func=batch_func,
            sequential_func=sequential_func,
            task_name="global_search_map",
            show_progress=False,
        )

        points: list[_MapPoint] = []
        for raw in raw_outputs:
            if not isinstance(raw, str) or not raw:
                # Sequential fallback inserts {} for a failed item.
                continue
            points.extend(self._parse_map_points(raw))
        return points

    @staticmethod
    def _format_reports(reports: list[RetrievalResult]) -> str:
        parts = []
        for report in reports:
            rid = (report.metadata or {}).get("community_id") or report.source or "?"
            parts.append(f"--- Report (id: {rid}) ---\n{report.content}")
        return "\n\n".join(parts)

    @staticmethod
    def _parse_map_points(raw: str) -> list[_MapPoint]:
        """Robustly parse a map JSON payload into scored key points.

        Tolerates the LLM wrapping the JSON in prose or code fences. Returns an
        empty list on any parse failure (the caller degrades gracefully) so a
        single bad map response never aborts global search.
        """
        payload = parse_llm_json(raw)
        if not payload:
            return []

        points: list[_MapPoint] = []
        for item in payload.get("points", []) or []:
            if not isinstance(item, dict):
                continue
            description = str(item.get("description", "")).strip()
            if not description:
                continue
            score = safe_float_parse(str(item.get("score", 0)), default_value=0.0)
            score_int = int(score) if score is not None else 0
            score_int = max(0, min(100, score_int))
            points.append(_MapPoint(description=description, score=score_int))
        return points

    def _filter_and_rank_points(self, points: list[_MapPoint]) -> list[_MapPoint]:
        threshold = self.global_search_config.map_relevance_threshold
        kept = [p for p in points if p.score > threshold]
        kept.sort(key=lambda p: p.score, reverse=True)
        return kept

    def _pack_points_within_budget(self, points: list[_MapPoint]) -> list[_MapPoint]:
        budget = self.global_search_config.max_map_reduce_tokens
        packed: list[_MapPoint] = []
        used = 0
        for point in points:
            cost = self.token_manager.count_tokens(point.description)
            if packed and used + cost > budget:
                break
            packed.append(point)
            used += cost
        logger.debug(
            "Packed %s/%s ranked key points into the %s-token reduce budget",
            len(packed),
            len(points),
            budget,
        )
        return packed

    async def _reduce_from_points(
        self,
        points: list[_MapPoint],
        results: list[RetrievalResult],
        query: SearchQuery,
    ) -> list[RetrievalResult]:
        synthesis_input = "\n\n".join(
            f"- (relevance {p.score}) {p.description}" for p in points
        )
        try:
            summary = await self.map_reducer.ainvoke(
                {"summaries": synthesis_input, "query": query.query}
            )
        except Exception as e:
            if not self.ignore_errors:
                raise
            logger.error("Map-reduce synthesis failed: %s", e)
            return results

        summary_result = RetrievalResult(
            content=summary,
            score=1.0,
            source="synthesized_summary",
            retriever_type=SectionType.GENERAL.value,
            metadata={
                "source_results_count": len(results),
                "ranked_key_points": len(points),
            },
        )
        return [summary_result] + results

    async def _concat_reduce(
        self, results: list[RetrievalResult], query: SearchQuery
    ) -> list[RetrievalResult]:
        """Legacy direct concat-and-reduce path (map-reduce degradation target)."""
        try:
            context = "\n\n---\n\n".join([r.content for r in results])
            summary = await self.map_reducer.ainvoke(
                {"summaries": context, "query": query.query}
            )
            summary_result = RetrievalResult(
                content=summary,
                score=1.0,
                source="synthesized_summary",
                retriever_type=SectionType.GENERAL.value,
                metadata={"source_results_count": len(results)},
            )
            return [summary_result] + results
        except Exception as e:
            if not self.ignore_errors:
                raise

            logger.error("Map-reduce synthesis failed: %s", e)
            return results

    def _record_search_metrics(
        self,
        processing_time: float,
        retrieved_count: int,
        selected_count: int,
        final_count: int,
    ) -> None:
        self._record_timing("processing_time", processing_time)
        self._record_metric("retrieved_count", retrieved_count)
        self._record_metric("selected_count", selected_count)
        self._record_metric("final_count", final_count)

    def _was_map_reduce_applied(self, final_results: list[RetrievalResult]) -> bool:
        if not self.global_search_config.enable_map_reduce:
            return False
        return any("synthesized" in (r.source or "") for r in final_results)
