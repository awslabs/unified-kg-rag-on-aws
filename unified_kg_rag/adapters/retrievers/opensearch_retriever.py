# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from collections import OrderedDict
from collections.abc import Coroutine
from typing import Any, ClassVar

import boto3
from opensearchpy.exceptions import NotFoundError

from unified_kg_rag.adapters.aws import BedrockEmbeddingModelFactory, OpenSearchClient
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    is_fatal_retrieval_error,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.adapters.storage.filter_schema import (
    FilterFields,
    opensearch_filter_fields,
    union_filter_fields,
)
from unified_kg_rag.domain.models import (
    Config,
    Constants,
    RetrievalResult,
    SearchQuery,
    SearchType,
)
from unified_kg_rag.domain.retrieval.index_prefixes import configured_index_prefixes
from unified_kg_rag.ports.model_factory import EmbeddingFactoryPort
from unified_kg_rag.shared import IndexNotFoundError, get_logger
from unified_kg_rag.shared.utils import (
    EMBEDDING_FIELD_SUFFIX,
    strip_embedding_fields,
    text_digest,
)
from unified_kg_rag.shared.utils.scripts import has_dense_script

logger = get_logger(__name__)


def _is_index_not_found(exc: BaseException) -> bool:
    """True when ``exc`` is OpenSearch's 404 ``index_not_found_exception``.

    ``OpenSearchClient`` re-raises ``NotFoundError`` unwrapped.
    """
    return isinstance(exc, NotFoundError) and exc.error == "index_not_found_exception"


class OpenSearchRetriever(BaseGraphRAGRetriever):
    # Query embeddings are reused across the sub-queries of one search (local
    # search sends the same entity-focus text to several indices) and across
    # repeated queries. Bounded LRU: entries are only reused for identical
    # text; 256 x a 1024-dim vector is ~2 MB.
    QUERY_EMBEDDING_CACHE_SIZE: ClassVar[int] = 256

    def __init__(
        self,
        config: Config,
        opensearch_client: OpenSearchClient,
        boto_session: boto3.Session | None = None,
        embedding_factory: EmbeddingFactoryPort | None = None,
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
        # Hard ceiling on hits per query (OpenSearch index.max_result_window
        # default). Fetch-by-id size may be floored up to the terms-filter size
        # but never beyond this, since the index is created with the default.
        self._max_result_window = self._opensearch_config.index_settings.get(
            "max_result_window", 10000
        )
        self._embedding_factory = embedding_factory or BedrockEmbeddingModelFactory(
            config=config,
            boto_session=boto_session,
            region_name=config.aws.bedrock_region,
        )
        self._embedding_model = self._embedding_factory.get_model(
            self._opensearch_config.embedding_model_id
        )
        self._query_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._field_mappings = self._initialize_field_mappings()
        self._index_filter_fields = self._initialize_index_filter_fields()

    def close(self) -> None:
        """Release the underlying OpenSearch client's connections (best-effort)."""
        self._opensearch_client.close()

    async def aclose(self) -> None:
        """Async teardown: await the underlying OpenSearch client's close."""
        await self._opensearch_client.aclose()

    def _initialize_field_mappings(self) -> dict[str, dict[str, list[str]]]:
        languages = self._config.processing.translation.translated_languages
        return {
            self._opensearch_config.text_units_index_prefix: {
                "lexical": ["text", *(f"translated_text_{lang}" for lang in languages)],
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

    def _initialize_index_filter_fields(self) -> dict[str, FilterFields]:
        """Index prefix -> its declared filterable fields (``filter_schema``)."""
        o = self._opensearch_config
        by_kind = opensearch_filter_fields(
            *self._config.processing.translation.translated_languages
        )
        return {
            o.text_units_index_prefix: by_kind["text_units"],
            o.entities_index_prefix: by_kind["entities"],
            o.relationships_index_prefix: by_kind["relationships"],
            o.claims_index_prefix: by_kind["claims"],
            o.community_reports_index_prefix: by_kind["community_reports"],
        }

    def filter_fields(self) -> FilterFields:
        return union_filter_fields(self._index_filter_fields.values())

    def _filters_for_index(
        self, prefix: str, filters: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """``filters`` restricted to the keys the ``prefix`` index declares.

        A ``term``/``terms``/``range`` clause on a field an index does not map
        matches no document, so one caller filter shared across sub-queries
        (e.g. ``type`` on entities, ``attr_category`` on text units) would empty
        every other index. See ``filter_schema`` for the declared fields.
        """
        declared = self._index_filter_fields.get(prefix)
        if not filters or declared is None:
            return filters
        kept = {k: v for k, v in filters.items() if declared.declares(k)}
        if len(kept) != len(filters):
            logger.debug(
                "Dropped filter keys absent from index '%s': %s",
                prefix,
                ", ".join(sorted(set(filters) - set(kept))),
            )
        return kept or None

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        start_time = time.time()
        query_preview = text_digest(query.query)
        logger.info(
            "OpenSearch retrieval started - query: %s ('%s')",
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
            # Log loudly with the traceback. Re-raise clearly-fatal errors
            # (auth/credentials/endpoint/connection) so a broken configuration
            # surfaces instead of masquerading as "0 results"; degrade to an
            # empty list only on genuinely-transient failures.
            if is_fatal_retrieval_error(e):
                logger.error(
                    "OpenSearch retrieval failed (fatal): %s", e, exc_info=True
                )
                raise
            logger.error(
                "OpenSearch retrieval failed (transient, degrading to empty "
                "results): %s",
                e,
                exc_info=True,
            )
            return []

    def _create_search_tasks(
        self,
        query: SearchQuery,
        search_type: SearchType,
        index_prefixes: list[str],
        query_vector: list[float] | None,
    ) -> list[Coroutine[Any, Any, list[RetrievalResult]]]:
        """One search per index, each with the filters that index declares."""
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

            index_query = query
            if query.filters:
                index_query = query.model_copy(
                    update={"filters": self._filters_for_index(prefix, query.filters)}
                )

            body, params = self._build_search_request(
                index_query, search_type, lexical_fields, vector_fields, query_vector
            )
            target_alias = self._get_name(prefix, query.suffix)
            search_tasks.append(
                self._execute_search([target_alias], body, params, query.suffix)
            )

        return search_tasks

    def _normalize_index_prefixes(self, prefixes: str | list[str] | None) -> list[str]:
        if isinstance(prefixes, str):
            return [prefixes]
        return prefixes or self._default_index_prefixes()

    def _default_index_prefixes(self) -> list[str]:
        return [
            prefix
            for prefix in configured_index_prefixes(self._config)
            if prefix in self._field_mappings
        ]

    async def _get_query_vector(
        self, query_text: str, search_type: SearchType, vector_fields: list[str]
    ) -> list[float] | None:
        if (
            query_text
            and search_type in [SearchType.VECTOR, SearchType.HYBRID]
            and vector_fields
        ):
            return await self._embed_query_cached(query_text)
        return None

    async def _embed_query_cached(self, query_text: str) -> list[float]:
        cache = self._query_embedding_cache
        cached = cache.get(query_text)
        if cached is not None:
            cache.move_to_end(query_text)
            return cached
        vector = await self._embedding_model.aembed_query(query_text)
        cache[query_text] = vector
        if len(cache) > self.QUERY_EMBEDDING_CACHE_SIZE:
            cache.popitem(last=False)
        return vector

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
        # Fetch-by-id callers (local/global/mix expansion) set top_k=len(ids) and
        # pass those ids as a terms filter to retrieve every one. Capping size at
        # max_query_size (default 100) would silently drop ids beyond 100 --
        # worse when terms_batch_size (150) > max_query_size, since each batch
        # then returns only 100 of its 150 ids. When a terms filter is present,
        # floor size at the largest terms-list length so no filtered id is lost,
        # bounded by max_result_window (the cluster's hard hits ceiling).
        largest_terms = self._largest_terms_filter_size(query.filters)
        if largest_terms > size:
            size = min(largest_terms, self._max_result_window)
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

        # Embedding vectors are only needed server-side for kNN scoring; never
        # ship them back (community reports alone carry three per hit).
        search_body = {
            "size": size,
            "query": main_query,
            "_source": {"excludes": [f"*{EMBEDDING_FIELD_SUFFIX}"]},
        }
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
            return self._build_vector_query(query_vector, size, vector_fields, filters)

        if search_type == SearchType.HYBRID:
            return self._build_hybrid_query(
                query, query_vector, lexical_fields, vector_fields, size, filters
            )

        return {"match_all": {}}

    @staticmethod
    def _build_lexical_query(query: SearchQuery, fields: list[str]) -> dict[str, Any]:
        if not query.query or query.query == "*":
            return {"match_all": {}}

        main_query: dict[str, Any] = {
            "multi_match": {"query": query.query, "fields": fields}
        }
        # AUTO allows one edit on a 3-5 character term. On Latin words that
        # absorbs a typo; on Hangul/Han/Kana, where one character is a whole
        # syllable or morpheme, it matches a different word ("가나라상사" for
        # "가나다상사"). A query containing such text is matched exactly.
        if not has_dense_script(query.query):
            main_query["multi_match"]["fuzziness"] = "AUTO"

        if not query.optional_keywords:
            return main_query

        optional_keywords_query = " ".join(query.optional_keywords)
        should_clause = {
            "multi_match": {"query": optional_keywords_query, "fields": fields}
        }

        return {"bool": {"must": [main_query], "should": [should_clause]}}

    @staticmethod
    def _build_vector_query(
        vector: list[float],
        k: int,
        fields: list[str],
        filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # Attach filters to the kNN clause itself (efficient/pre-filtering on the
        # Lucene engine). Without this, a VECTOR query that also carries filters
        # (e.g. a non-empty query scoped to a community_id set) returns the
        # global top-k nearest vectors, leaking documents outside the filter.
        def _knn(field: str) -> dict[str, Any]:
            clause: dict[str, Any] = {"vector": vector, "k": k}
            if filters:
                clause["filter"] = {"bool": {"filter": filters}}
            return {"knn": {field: clause}}

        if len(fields) == 1:
            return _knn(fields[0])

        return {"bool": {"should": [_knn(field) for field in fields]}}

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
            # Propagate filters to the kNN sub-query too: otherwise the lexical
            # sub-query respects the filter but the vector sub-query returns the
            # global nearest neighbours, polluting the fused result with
            # documents outside the filter.
            queries.append(
                self._build_vector_query(vector, size, vector_fields, filters)
            )

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

    @staticmethod
    def _largest_terms_filter_size(filters: dict[str, Any] | None) -> int:
        """Length of the largest list-valued (terms) filter, else 0.

        Used to floor the query size so a fetch-by-id (ids passed as a terms
        filter with top_k=len(ids)) is never truncated below the number of ids
        requested.
        """
        if not filters:
            return 0
        list_lengths = [len(v) for v in filters.values() if isinstance(v, list)]
        return max(list_lengths, default=0)

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
        self,
        aliases: list[str],
        body: dict[str, Any],
        params: dict[str, Any],
        suffix: str | None = None,
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
            # A fatal error (auth/config/connection) must propagate so the
            # caller surfaces it, rather than masquerading as "0 results".
            # Transient/per-index errors degrade to an empty list.
            if is_fatal_retrieval_error(e):
                logger.error("Fatal search error on indices %s: %s", aliases, e)
                raise
            if _is_index_not_found(e):
                await self._require_ingested_suffix(suffix, e)
                # A missing alias is a config/index mismatch (the index was
                # never built), not a transient failure.
                logger.warning(
                    "Index not found for %s; skipping it (was it built by the "
                    "ingestion config?): %s",
                    aliases,
                    e,
                )
                return []
            logger.error("Search failed on indices %s: %s", aliases, e)
            return []

    async def _require_ingested_suffix(
        self, suffix: str | None, cause: Exception
    ) -> None:
        """Raise when nothing was ingested under ``suffix``.

        One missing index is normal: an index is created only when ingestion
        has items for it (a corpus without claims has no claims index). The
        text-units index exists for every ingested corpus, so its absence means
        the suffix was never ingested and every search would come back empty.
        """
        alias = self._get_name(self._opensearch_config.text_units_index_prefix, suffix)
        try:
            if await self._opensearch_client.aindex_exists(alias):
                return
        except Exception:
            return  # Cannot tell: keep skipping just the missing index.
        shown = suffix or Constants.DEFAULT_SUFFIX.value
        raise IndexNotFoundError(
            f"No indices found for suffix '{shown}' (OpenSearch alias '{alias}' "
            "does not exist). Did you run ingestion with "
            f"processing.document_parsing.index_value '{shown}'? Query with the "
            "suffix the corpus was ingested under (run-rag/run-eval --suffix)."
        ) from cause

    def _parse_hit(self, hit: dict[str, Any]) -> RetrievalResult:
        source = hit.get("_source", {})
        index_name = hit.get("_index", "")
        source_id = str(source.get("id", hit.get("_id", "")))
        section_type = self._determine_section_type(index_name)
        content = self._extract_content(source)

        # Defensive twin of the `_source` excludes in the request: a custom
        # client or a cluster that ignores excludes must still not leak vectors
        # into result metadata (and from there into reported sources).
        return RetrievalResult(
            content=content,
            score=hit.get("_score", 0.0),
            source=source_id,
            retriever_type=str(section_type.value),
            metadata={**strip_embedding_fields(source), "_search_index": index_name},
        )

    def _extract_content(self, source: dict[str, Any]) -> str:
        content_parts = []
        target_language = self._config.processing.translation.target_language.value
        translated_key = f"translated_text_{target_language}"

        if translated_text := source.get(translated_key):
            content_parts.append(translated_text)

        # A relationship hit MUST name its endpoints.
        # Upstream LightRAG renders every relation as
        # `{"entity1": src, "entity2": tgt, "description": ...}` (operate.py
        # `relations_context`), so the reader can always tell what is related to
        # what. We indexed `source_name`/`target_name` (opensearch_indexer.py:371)
        # but never rendered them, emitting bare fragments like
        # "Description: Date of birth relationship" — text naming neither endpoint,
        # which the reader cannot resolve. That is why widening the relationship
        # stream made scores WORSE instead of better: it multiplied unusable lines.
        src_name = source.get("source_name")
        tgt_name = source.get("target_name")
        if src_name or tgt_name:
            rel_type = source.get("type")
            arrow = f" -[{rel_type}]-> " if rel_type else " -> "
            content_parts.append(
                f"Relationship: {src_name or 'unknown'}{arrow}{tgt_name or 'unknown'}"
            )

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
