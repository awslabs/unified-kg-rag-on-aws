# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from typing import Any

import boto3

from aws_graphrag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
)
from aws_graphrag.domain.models import (
    Config,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
)
from aws_graphrag.domain.retrieval.strategy_registry import register_strategy
from aws_graphrag.shared import get_logger

logger = get_logger(__name__)


@register_strategy(SearchStrategy.LOCAL)
class LocalSearchStrategy(BaseSearchStrategy):
    def __init__(
        self,
        config: Config,
        retrievers: dict[str, BaseGraphRAGRetriever],
        boto_session: boto3.Session | None = None,
        entity_focus_multiplier: int = 2,
        **kwargs: Any,
    ):
        super().__init__(config, retrievers, boto_session, **kwargs)
        self.entity_focus_multiplier = entity_focus_multiplier

    async def asearch(self, query: SearchQuery) -> SearchResult:
        start_time = time.time()
        logger.info(
            "Local search started - query: '%s...' ('%s') with entities: '%s'",
            query.query[:50],
            query.search_type.value,
            ", ".join(query.entity_focus),
        )

        candidate_entity_ids = await self._find_candidate_entities(query)
        logger.debug(
            "Found %s candidate entities: '%s%s'",
            len(candidate_entity_ids),
            ", ".join(candidate_entity_ids[:5]),
            "..." if len(candidate_entity_ids) > 5 else "",
        )

        expanded_entity_nodes = await self._expand_via_graph(
            query, candidate_entity_ids
        )
        filtered_entity_nodes = self._filter_entities(
            expanded_entity_nodes,
            frequency_threshold=self.config.search.local_search.entity_frequency_threshold,
        )

        expanded_entity_ids = self._get_ids(filtered_entity_nodes, "id")
        logger.debug(
            "Expanded to %s entities: '%s%s'",
            len(expanded_entity_ids),
            ", ".join(expanded_entity_ids[:5]),
            "..." if len(expanded_entity_ids) > 5 else "",
        )

        text_unit_ids = self._get_ids(filtered_entity_nodes, "text_unit_ids")
        logger.debug(
            "Found %s text units: '%s%s'",
            len(text_unit_ids),
            ", ".join(text_unit_ids[:5]),
            "..." if len(text_unit_ids) > 5 else "",
        )

        text_units = await self._retrieve_documents(text_unit_ids, query.suffix)
        all_results = {"graph_entities": expanded_entity_nodes, **text_units}

        final_results = self.hybrid_scorer.fuse_and_rerank_results(
            all_results,
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
        )

        processing_time = time.time() - start_time
        self._record_search_metrics(
            processing_time,
            len(final_results),
            len(set(candidate_entity_ids + expanded_entity_ids)),
            len(text_unit_ids),
        )

        logger.info(
            "Search completed - retrieved: %s results in %.3fs",
            len(final_results),
            processing_time,
        )

        return SearchResult(
            query=query,
            results=final_results,
            total_results=len(final_results),
            search_strategy="local_search",
            processing_time=processing_time,
            metadata={
                "candidate_entity_count": len(candidate_entity_ids),
                "expanded_entity_count": len(expanded_entity_ids),
                "text_unit_count": len(text_unit_ids),
            },
        )

    async def _find_candidate_entities(self, query: SearchQuery) -> list[str]:
        if not self.document_retriever or not query.entity_focus:
            return []

        n_candidates = len(query.entity_focus) * self.entity_focus_multiplier
        search_query = SearchQuery(
            query=" ".join(query.entity_focus),
            search_type=query.search_type,
            top_k=n_candidates,
            index_prefixes=[self.config.indexing.opensearch.entities_index_prefix],
            suffix=query.suffix,
        )

        try:
            results = await self.document_retriever.aretrieve(search_query)
            return [res.source for res in results if res.source]
        except Exception as e:
            logger.error("Failed to find candidate entities: %s", e)
            return []

    async def _expand_via_graph(
        self, query: SearchQuery, seed_entity_ids: list[str]
    ) -> list[RetrievalResult]:
        if not self.graph_retriever or not seed_entity_ids:
            return []

        search_query = query.model_copy(deep=True)
        search_query.label_prefixes = [self.config.indexing.neptune.entity_label_prefix]
        search_query.entity_focus = []
        search_query.filters = (search_query.filters or {}).copy()
        search_query.filters["id"] = seed_entity_ids

        try:
            return await self.graph_retriever.aretrieve(search_query)
        except Exception as e:
            logger.error("Neptune retrieval failed: %s", e)
            return []

    @staticmethod
    def _filter_entities(
        expanded_entity_nodes: list[RetrievalResult],
        frequency_threshold: int,
    ) -> list[RetrievalResult]:
        filtered_nodes = []
        for node in expanded_entity_nodes:
            text_unit_count = len(node.metadata.get("text_unit_ids", []))
            if 0 < text_unit_count <= frequency_threshold or text_unit_count == 0:
                filtered_nodes.append(node)

        original_count = len(expanded_entity_nodes)
        filtered_count = len(filtered_nodes)
        if original_count != filtered_count:
            logger.debug(
                "Filtered %s entities based on frequency threshold %s",
                original_count - filtered_count,
                frequency_threshold,
            )

        return filtered_nodes

    async def _retrieve_documents(
        self, text_unit_ids: list[str], suffix: str | None
    ) -> dict[str, list[RetrievalResult]]:
        if not self.document_retriever or not text_unit_ids:
            return {}

        search_query = SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=len(text_unit_ids),
            index_prefixes=[self.config.indexing.opensearch.text_units_index_prefix],
            suffix=suffix,
            filters={"id": text_unit_ids},
        )

        try:
            results = await self.document_retriever.aretrieve(search_query)
            return {"text_units": results}
        except Exception as e:
            logger.error("OpenSearch retrieval failed: %s", e)
            return {}

    def _record_search_metrics(
        self,
        processing_time: float,
        retrieved_count: int,
        entity_count: int,
        text_unit_count: int,
    ) -> None:
        self._record_timing("processing_time", processing_time)
        self._record_metric("retrieved_count", retrieved_count)
        self._record_metric("entity_count", entity_count)
        self._record_metric("text_unit_count", text_unit_count)
