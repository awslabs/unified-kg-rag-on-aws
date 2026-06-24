# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from collections.abc import Coroutine
from typing import Any

import boto3

from aws_graphrag.adapters.aws import BedrockEmbeddingModelFactory, OpenSearchClient
from aws_graphrag.adapters.retrieval.base import BaseGraphRAGRetriever
from aws_graphrag.adapters.retrieval.token_manager import SectionType
from aws_graphrag.domain.models import Config, RetrievalResult, SearchQuery, SearchType
from aws_graphrag.shared import get_logger

logger = get_logger(__name__)


class OpenSearchRetriever(BaseGraphRAGRetriever):
    def __init__(
        self,
        config: Config,
        opensearch_client: OpenSearchClient,
        boto_session: boto3.Session | None = None,
        **kwargs: Any,
    ):
        super().__init__(config, boto_session, **kwargs)
        self._opensearch_client = opensearch_client
        self._opensearch_config = config.indexing.opensearch
        # Clause-budget knobs are config-driven so they can track the cluster's
        # indices.query.bool.max_clause_count without code changes. Stored as
        # private attributes (this is a pydantic model, which rejects undeclared
        # public attributes).
        self._max_size = self._opensearch_config.max_query_size
        self._terms_batch_size = self._opensearch_config.terms_batch_size
        self._max_total_clauses = self._opensearch_config.max_total_clauses
        self._reserved_clauses = self._opensearch_config.reserved_clauses
        self._embedding_factory = BedrockEmbeddingModelFactory(
            config=config,
            boto_session=boto_session,
            region_name=config.aws.bedrock.region_name,
        )
        self._embedding_model = self._embedding_factory.get_model(
            self._opensearch_config.embedding_model_id
        )
        self._field_mappings = self._initialize_field_mappings()

    def _initialize_field_mappings(self) -> dict[str, dict[str, list[str]]]:
        target_language = self._config.processing.translation.target_language.value
        return {
            self._opensearch_config.text_units_index_prefix: {
                "lexical": ["text", f"translated_text_{target_language}"],
                "vector": ["text_embedding"],
            },
            self._opensearch_config.entities_index_prefix: {
                "lexical": ["name", "description"],
                "vector": ["name_embedding", "description_embedding"],
            },
            self._opensearch_config.relationships_index_prefix: {
                "lexical": ["description", "source_name", "target_name"],
                "vector": ["description_embedding"],
            },
            self._opensearch_config.claims_index_prefix: {
                "lexical": [
                    "description",
                    "subject_name",
                    "object_name",
                    "source_text",
                ],
                "vector": ["description_embedding"],
            },
            self._opensearch_config.community_reports_index_prefix: {
                "lexical": ["name", "summary", "full_content"],
                "vector": [
                    "name_embedding",
                    "summary_embedding",
                    "full_content_embedding",
                ],
            },
        }

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        start_time = time.time()
        query_preview = query.query[:50] if query.query else "(empty)"
        logger.info(
            "OpenSearch retrieval started - query: '%s...' ('%s')",
            query_preview,
            query.search_type.value,
        )

        search_type = query.search_type or SearchType.HYBRID
        index_prefixes = self._normalize_index_prefixes(query.index_prefixes)

        try:
            query_vector = await self._get_query_vector(
                query.query, search_type, ["any"]
            )

            safe_batch_size = self._calculate_safe_batch_size(query.filters)
            large_filters = self._find_all_large_filter_lists(
                query.filters, safe_batch_size
            )
            if large_filters:
                all_results = await self._execute_multi_batched_retrieval(
                    query,
                    large_filters,
                    safe_batch_size,
                    search_type,
                    index_prefixes,
                    query_vector,
                )
            else:
                search_tasks = self._create_search_tasks(
                    query, search_type, index_prefixes, query_vector
                )

                if not search_tasks:
                    logger.warning(
                        "No searchable indices available for query: %s", query_preview
                    )
                    return []

                all_results = []
                for results in await asyncio.gather(*search_tasks):
                    all_results.extend(results)

            all_results.sort(key=lambda x: x.score or 0.0, reverse=True)
            final_results = all_results[: query.top_k * query.retrieval_multiplier]

            processing_time = time.time() - start_time
            self._record_timing("retrieval_time", processing_time)

            logger.info(
                "OpenSearch retrieval completed - retrieved: %s results (%.2fs)",
                len(final_results),
                processing_time,
            )
            return final_results

        except Exception as e:
            logger.error("Retrieval failed: %s", e)
            return []

    def _create_search_tasks(
        self,
        query: SearchQuery,
        search_type: SearchType,
        index_prefixes: list[str],
        query_vector: list[float] | None,
    ) -> list[Coroutine[Any, Any, list[RetrievalResult]]]:
        search_tasks = []
        for prefix in index_prefixes:
            mapping = self._field_mappings.get(prefix)
            if not mapping:
                continue

            lexical_fields = mapping.get("lexical", [])
            vector_fields = mapping.get("vector", [])

            if not self._is_search_type_supported(
                search_type, lexical_fields, vector_fields
            ):
                continue

            body, params = self._build_search_request(
                query, search_type, lexical_fields, vector_fields, query_vector
            )
            target_alias = self._get_name(prefix, query.suffix)
            search_tasks.append(self._execute_search([target_alias], body, params))

        return search_tasks

    def _normalize_index_prefixes(self, prefixes: str | list[str] | None) -> list[str]:
        if isinstance(prefixes, str):
            return [prefixes]
        return prefixes or list(self._field_mappings.keys())

    async def _get_query_vector(
        self, query_text: str, search_type: SearchType, vector_fields: list[str]
    ) -> list[float] | None:
        if (
            query_text
            and search_type in [SearchType.VECTOR, SearchType.HYBRID]
            and vector_fields
        ):
            return await self._embedding_model.aembed_query(query_text)
        return None

    @staticmethod
    def _is_search_type_supported(
        search_type: SearchType,
        lexical_fields: list[str],
        vector_fields: list[str],
    ) -> bool:
        if search_type == SearchType.LEXICAL:
            return bool(lexical_fields)
        if search_type == SearchType.VECTOR:
            return bool(vector_fields)
        return bool(lexical_fields or vector_fields)

    def _build_search_request(
        self,
        query: SearchQuery,
        search_type: SearchType,
        lexical_fields: list[str],
        vector_fields: list[str],
        query_vector: list[float] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        size = min(query.top_k * query.retrieval_multiplier, self._max_size)
        filters = self._build_filter_clauses(query.filters)

        main_query = self._build_main_query(
            query,
            search_type,
            lexical_fields,
            vector_fields,
            query_vector,
            size,
            filters,
        )

        search_body = {"size": size, "query": main_query}
        params = {}

        if search_type == SearchType.HYBRID:
            params["search_pipeline"] = (
                self._opensearch_config.hybrid_search_pipeline_name
            )

        return search_body, params

    def _build_main_query(
        self,
        query: SearchQuery,
        search_type: SearchType,
        lexical_fields: list[str],
        vector_fields: list[str],
        query_vector: list[float] | None,
        size: int,
        filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if search_type == SearchType.LEXICAL:
            lexical_query = self._build_lexical_query(query, lexical_fields)
            return (
                {"bool": {"must": [lexical_query], "filter": filters}}
                if filters
                else lexical_query
            )

        if filters and not query.query:
            return {"bool": {"filter": filters}}

        if search_type == SearchType.VECTOR and query_vector:
            return self._build_vector_query(query_vector, size, vector_fields)

        if search_type == SearchType.HYBRID:
            return self._build_hybrid_query(
                query, query_vector, lexical_fields, vector_fields, size, filters
            )

        return {"match_all": {}}

    @staticmethod
    def _build_lexical_query(query: SearchQuery, fields: list[str]) -> dict[str, Any]:
        if not query.query or query.query == "*":
            return {"match_all": {}}

        main_query = {
            "multi_match": {"query": query.query, "fields": fields, "fuzziness": "AUTO"}
        }

        if not query.optional_keywords:
            return main_query

        optional_keywords_query = " ".join(query.optional_keywords)
        should_clause = {
            "multi_match": {"query": optional_keywords_query, "fields": fields}
        }

        return {"bool": {"must": [main_query], "should": [should_clause]}}

    @staticmethod
    def _build_vector_query(
        vector: list[float], k: int, fields: list[str]
    ) -> dict[str, Any]:
        if len(fields) == 1:
            return {"knn": {fields[0]: {"vector": vector, "k": k}}}

        return {
            "bool": {
                "should": [
                    {"knn": {field: {"vector": vector, "k": k}}} for field in fields
                ]
            }
        }

    def _build_hybrid_query(
        self,
        query: SearchQuery,
        vector: list[float] | None,
        lexical_fields: list[str],
        vector_fields: list[str],
        size: int,
        filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        queries = []

        lexical_query = self._build_lexical_query(query, lexical_fields)
        if lexical_query.get("match_all") and filters:
            queries.append({"bool": {"must": lexical_query, "filter": filters}})
        elif lexical_fields:
            if filters:
                queries.append({"bool": {"must": [lexical_query], "filter": filters}})
            else:
                queries.append(lexical_query)

        if vector_fields and vector:
            queries.append(self._build_vector_query(vector, size, vector_fields))

        return {"hybrid": {"queries": queries}} if queries else {"match_none": {}}

    @staticmethod
    def _build_filter_clauses(filters: dict[str, Any] | None) -> list[dict[str, Any]]:
        if not filters:
            return []

        clauses: list[dict[str, Any]] = []
        for key, value in filters.items():
            if isinstance(value, dict):
                clauses.append({"range": {key: value}})
            elif isinstance(value, list):
                clauses.append({"terms": {key: value}})
            else:
                clauses.append({"term": {key: value}})

        return clauses

    def _calculate_safe_batch_size(self, filters: dict[str, Any] | None) -> int:
        if not filters:
            return self._terms_batch_size

        list_filters = [
            (k, v) for k, v in filters.items() if isinstance(v, list) and len(v) > 0
        ]
        if not list_filters:
            return self._terms_batch_size

        list_filter_count = len(list_filters)
        total_terms = sum(len(v) for _, v in list_filters)

        available_clauses = self._max_total_clauses - self._reserved_clauses
        safe_size = available_clauses // list_filter_count

        min_batch_size = max(1, available_clauses // max(list_filter_count, 1) // 2)

        final_size = max(min_batch_size, min(safe_size, self._terms_batch_size))

        logger.debug(
            "Batch size calculation: %s list filters, total_terms=%s, available_clauses=%s, safe_size=%s, min_batch_size=%s, final_size=%s",
            list_filter_count,
            total_terms,
            available_clauses,
            safe_size,
            min_batch_size,
            final_size,
        )

        return final_size

    @staticmethod
    def _find_all_large_filter_lists(
        filters: dict[str, Any] | None, batch_size: int
    ) -> dict[str, list[Any]]:
        if not filters:
            return {}

        large_filters = {}
        for key, value in filters.items():
            if isinstance(value, list) and len(value) > batch_size:
                large_filters[key] = value
        return large_filters

    async def _execute_multi_batched_retrieval(
        self,
        query: SearchQuery,
        large_filters: dict[str, list[Any]],
        batch_size: int,
        search_type: SearchType,
        index_prefixes: list[str],
        query_vector: list[float] | None,
    ) -> list[RetrievalResult]:
        all_results: list[RetrievalResult] = []
        seen_ids: set[str] = set()

        filter_batches: dict[str, list[list[Any]]] = {}
        for key, values in large_filters.items():
            filter_batches[key] = [
                values[i : i + batch_size] for i in range(0, len(values), batch_size)
            ]

        max_batches = max(len(batches) for batches in filter_batches.values())

        logger.info(
            "Executing multi-batched retrieval: %s large filters, %s batches, batch_size=%s (filter sizes: %s)",
            len(large_filters),
            max_batches,
            batch_size,
            ", ".join(f"{k}={len(v)}" for k, v in large_filters.items()),
        )

        for batch_idx in range(max_batches):
            batch_filters = {**(query.filters or {})}
            for key, batches in filter_batches.items():
                actual_batch_idx = min(batch_idx, len(batches) - 1)
                batch_filters[key] = batches[actual_batch_idx]

            batch_query = query.model_copy(update={"filters": batch_filters})

            search_tasks = self._create_search_tasks(
                batch_query, search_type, index_prefixes, query_vector
            )

            for results in await asyncio.gather(*search_tasks):
                for result in results:
                    if result.source is not None:
                        if result.source not in seen_ids:
                            seen_ids.add(result.source)
                            all_results.append(result)

        logger.debug(
            "Multi-batched retrieval completed: %s unique results from %s batches",
            len(all_results),
            max_batches,
        )
        return all_results

    async def _execute_search(
        self, aliases: list[str], body: dict[str, Any], params: dict[str, Any]
    ) -> list[RetrievalResult]:
        if not aliases:
            return []

        try:
            response = await self._opensearch_client.asearch(
                index=",".join(aliases), body=body, **params
            )
            hits = response.get("hits", {}).get("hits", [])
            return [self._parse_hit(hit) for hit in hits]
        except Exception as e:
            logger.error("Search failed on indices %s: %s", aliases, e)
            return []

    def _parse_hit(self, hit: dict[str, Any]) -> RetrievalResult:
        source = hit.get("_source", {})
        index_name = hit.get("_index", "")
        source_id = str(source.get("id", hit.get("_id", "")))
        section_type = self._determine_section_type(index_name)
        content = self._extract_content(source)

        return RetrievalResult(
            content=content,
            score=hit.get("_score", 0.0),
            source=source_id,
            retriever_type=str(section_type.value),
            metadata={**source, "_search_index": index_name},
        )

    def _extract_content(self, source: dict[str, Any]) -> str:
        content_parts = []
        target_language = self._config.processing.translation.target_language.value
        translated_key = f"translated_text_{target_language}"

        if translated_text := source.get(translated_key):
            content_parts.append(translated_text)

        if description := source.get("description"):
            content_parts.append(f"Description: {description}")

        if full_content := source.get("full_content"):
            content_parts.append(full_content)

        if name := source.get("name"):
            content_parts.append(f"Title: {name}")

        if summary := source.get("summary"):
            content_parts.append(f"Summary: {summary}")

        if (text := source.get("text")) and not source.get(translated_key):
            content_parts.append(text)

        unique_parts = list(dict.fromkeys(content_parts))
        return "\n\n".join(unique_parts).strip()

    def _determine_section_type(self, index_name: str) -> SectionType:
        opensearch_config = self._opensearch_config

        if index_name.startswith(opensearch_config.text_units_index_prefix):
            return SectionType.TEXT
        if index_name.startswith(opensearch_config.entities_index_prefix):
            return SectionType.ENTITY
        if index_name.startswith(opensearch_config.relationships_index_prefix):
            return SectionType.RELATIONSHIP
        if index_name.startswith(opensearch_config.claims_index_prefix):
            return SectionType.CLAIM
        if index_name.startswith(opensearch_config.community_reports_index_prefix):
            return SectionType.COMMUNITY

        return SectionType.GENERAL
