# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import time
from typing import Any

import boto3

from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    BaseSearchStrategy,
    is_fatal_retrieval_error,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.domain.models import (
    Config,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.domain.retrieval.strategy_registry import register_strategy
from unified_kg_rag.shared import get_logger

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

    def _per_type_quota(self, top_k: int) -> dict[str, int]:
        """Reserved fusion slots per section type, from configured shares of top_k."""
        quota_config = self.config.search.local_search.type_quota
        shares: list[tuple[SectionType, float, int]] = [
            (SectionType.TEXT, quota_config.text_multiplier, quota_config.text_floor),
            (
                SectionType.ENTITY,
                quota_config.entity_multiplier,
                quota_config.entity_floor,
            ),
            (
                SectionType.RELATIONSHIP,
                quota_config.relationship_multiplier,
                quota_config.relationship_floor,
            ),
            (
                SectionType.COMMUNITY,
                quota_config.community_multiplier,
                quota_config.community_floor,
            ),
            (
                SectionType.CLAIM,
                quota_config.claim_multiplier,
                quota_config.claim_floor,
            ),
        ]
        return {
            section_type.value: max(int(multiplier * top_k), floor)
            for section_type, multiplier, floor in shares
        }

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

        text_unit_ids = self._rank_text_unit_ids(filtered_entity_nodes)
        logger.debug(
            "Found %s text units: '%s%s'",
            len(text_unit_ids),
            ", ".join(text_unit_ids[:5]),
            "..." if len(text_unit_ids) > 5 else "",
        )

        text_units = await self._retrieve_documents(text_unit_ids, query.suffix)
        all_results = {"graph_entities": expanded_entity_nodes, **text_units}

        # MS GraphRAG local search builds context from entities + the community
        # reports those entities belong to + in-network relationships + text
        # units (+ claims). We mirror that: enrich the entity/text-unit core with
        # a community-report section and a relationship section so a local query
        # also sees the higher-level community synthesis and the relationship
        # descriptions, not just raw entities and chunks.
        community_reports = await self._retrieve_community_reports(query)
        if community_reports:
            all_results["community_reports"] = community_reports

        relationships = await self._retrieve_relationships(query)
        if relationships:
            all_results["relationships"] = relationships

        # MS GraphRAG injects covariates (claims) into local-search context.
        # Gated strictly on claim extraction being enabled so the default path
        # (claims off) is unchanged: no extra retrieval is issued.
        claims = await self._retrieve_claims(query)
        if claims:
            all_results["claims"] = claims

        # The same divergences as mix apply to local (shared fuse path).
        # Give each section type its own quota so the diversity filter + fusion don't
        # collapse the candidate set to top_k before assembly (was dropping gold KG
        # items), and rerank ONLY text chunks so content-vs-query reranking doesn't bury
        # multi-hop bridge entities/relations. Mirrors MS local's proportional,
        # per-section context assembly.
        final_results = self.hybrid_scorer.fuse_and_rerank_results(
            all_results,
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
            per_type_quota=self._per_type_quota(query.top_k),
            rerank_only_types={SectionType.TEXT.value},
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
            if is_fatal_retrieval_error(e):
                raise
            logger.error("Failed to find candidate entities: %s", e)
            return []

    async def _retrieve_claims(self, query: SearchQuery) -> list[RetrievalResult]:
        # Only consume the claims (covariate) index when extraction is enabled;
        # otherwise the index is empty/absent and querying it is pure overhead.
        if (
            not self.document_retriever
            or not self.config.processing.claim_extraction.enabled
        ):
            return []

        # Mirror _find_candidate_entities: search the claims index with the
        # entity focus when present, falling back to the raw query text.
        claim_query = (
            " ".join(query.entity_focus) if query.entity_focus else query.query
        )
        if not claim_query:
            return []

        search_query = SearchQuery(
            query=claim_query,
            search_type=query.search_type,
            top_k=query.top_k,
            index_prefixes=[self.config.indexing.opensearch.claims_index_prefix],
            suffix=query.suffix,
        )

        try:
            return await self.document_retriever.aretrieve(search_query)
        except Exception as e:
            if is_fatal_retrieval_error(e):
                raise
            logger.error("Failed to retrieve claims: %s", e)
            return []

    async def _retrieve_community_reports(
        self, query: SearchQuery
    ) -> list[RetrievalResult]:
        # Pull the community reports most relevant to the query so local context
        # carries the community-level synthesis (mirrors MS GraphRAG local
        # search). The community-reports index always exists on the GraphRAG
        # path; degrade to no section on any retrieval error.
        if not self.document_retriever:
            return []

        report_query = (
            " ".join(query.entity_focus) if query.entity_focus else query.query
        )
        if not report_query:
            return []

        search_query = SearchQuery(
            query=report_query,
            search_type=query.search_type,
            top_k=query.top_k,
            index_prefixes=[
                self.config.indexing.opensearch.community_reports_index_prefix
            ],
            suffix=query.suffix,
        )

        try:
            return await self.document_retriever.aretrieve(search_query)
        except Exception as e:
            if is_fatal_retrieval_error(e):
                raise
            logger.error("Failed to retrieve community reports: %s", e)
            return []

    async def _retrieve_relationships(
        self, query: SearchQuery
    ) -> list[RetrievalResult]:
        # Add a relationship section (relationship descriptions for the query).
        # Gated on the relationship VECTOR index being built — for a GraphRAG-only
        # deployment with build_relationship_vector_index=False the index is
        # absent, so querying it is pure overhead.
        if (
            not self.document_retriever
            or not self.config.indexing.opensearch.build_relationship_vector_index
        ):
            return []

        rel_query = " ".join(query.entity_focus) if query.entity_focus else query.query
        if not rel_query:
            return []

        search_query = SearchQuery(
            query=rel_query,
            search_type=query.search_type,
            top_k=query.top_k,
            index_prefixes=[self.config.indexing.opensearch.relationships_index_prefix],
            suffix=query.suffix,
        )

        try:
            return await self.document_retriever.aretrieve(search_query)
        except Exception as e:
            if is_fatal_retrieval_error(e):
                raise
            logger.error("Failed to retrieve relationships: %s", e)
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
            if is_fatal_retrieval_error(e):
                raise
            logger.error("Neptune retrieval failed: %s", e)
            return []

    @classmethod
    def _rank_text_unit_ids(cls, entity_nodes: list[RetrievalResult]) -> list[str]:
        # Chunk candidates used to come from
        # `_get_ids(nodes, "text_unit_ids")`, which unions them into a SET — so
        # all ordering was lost, and `_retrieve_documents` then fetches them by
        # id filter with `query=""` (an ID batch fetch, no relevance score). The
        # chunk stream therefore reached fusion in arbitrary order with score 0,
        # and whatever the per-type quota sliced off was an arbitrary subset.
        # That is invisible while expansion is narrow and every chunk is
        # on-topic, but it makes widening the expansion actively harmful: a
        # measured 5x more chunks came with 4x LESS gold in the context.
        #
        # MS GraphRAG local ranks candidate text units by how many distinct
        # query-relevant entities reference them, with the entity's own rank as
        # the tiebreak, before applying its text-unit budget. Mirror that here:
        # score each chunk by (number of referencing entities, best referencing
        # entity score) and return ids in descending order, so downstream
        # truncation keeps the chunks with the most graph support.
        hit_count: dict[str, int] = {}
        best_score: dict[str, float] = {}
        for node in entity_nodes:
            ids = node.metadata.get("text_unit_ids") or []
            if isinstance(ids, str):
                ids = [ids]
            if not isinstance(ids, list):
                continue
            score = node.score or 0.0
            for raw_id in ids:
                unit_id = str(raw_id)
                hit_count[unit_id] = hit_count.get(unit_id, 0) + 1
                if score > best_score.get(unit_id, float("-inf")):
                    best_score[unit_id] = score
        return sorted(
            hit_count,
            key=lambda unit_id: (hit_count[unit_id], best_score.get(unit_id, 0.0)),
            reverse=True,
        )

    @staticmethod
    def _text_unit_count(node: RetrievalResult) -> int:
        # Neptune's `_clean_property_map` unwraps any
        # single-element value_map list into a bare scalar, so an entity that
        # appears in EXACTLY ONE text unit arrives with `text_unit_ids` as a str
        # (a 36-char UUID), not a list. `len()` on that counted CHARACTERS (36),
        # which exceeds every sane frequency threshold, so the most specific
        # entities in the graph — the multi-hop bridge nodes that occur in a
        # single document — were silently dropped by the frequency filter, which
        # also starved the text-unit fan-out those entities feed.
        ids = node.metadata.get("text_unit_ids") or []
        if isinstance(ids, str):
            return 1
        return len(ids)

    @classmethod
    def _filter_entities(
        cls,
        expanded_entity_nodes: list[RetrievalResult],
        frequency_threshold: int,
    ) -> list[RetrievalResult]:
        filtered_nodes = []
        for node in expanded_entity_nodes:
            text_unit_count = cls._text_unit_count(node)
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
            return {"text_units": self._restore_rank(results, text_unit_ids)}
        except Exception as e:
            if is_fatal_retrieval_error(e):
                raise
            logger.error("OpenSearch retrieval failed: %s", e)
            return {}

    @staticmethod
    def _restore_rank(
        results: list[RetrievalResult], ranked_ids: list[str]
    ) -> list[RetrievalResult]:
        # This lookup is an ID-batch FETCH (`query=""`,
        # LEXICAL, pure id filter), so OpenSearch returns the batch in index
        # order with no meaningful relevance score — 1915 of 2355 emitted
        # contexts carried score 0.0. That threw away the graph-support ranking
        # computed in `_rank_text_unit_ids`. Re-impose it here, and project it
        # into `score` as a normalized descending value so the shared fusion /
        # per-type-quota path (which sorts by score) preserves it instead of
        # tie-breaking arbitrarily.
        if not results or not ranked_ids:
            return results
        rank_of = {unit_id: i for i, unit_id in enumerate(ranked_ids)}
        fallback = len(ranked_ids)
        ordered = sorted(results, key=lambda r: rank_of.get(str(r.source), fallback))
        total = len(ordered)
        rescored: list[RetrievalResult] = []
        for position, result in enumerate(ordered):
            scored = result.model_copy()
            scored.score = (total - position) / total
            rescored.append(scored)
        return rescored

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
