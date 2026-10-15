# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from collections import OrderedDict
from collections.abc import Coroutine
from typing import Any, ClassVar

from unified_kg_rag.adapters.aws.opensearch import is_index_not_found
from unified_kg_rag.adapters.retrieval.base import BaseGraphRAGRetriever
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.adapters.storage.embedding_settings import (
    configured_embedding_model,
    resolve_embedding_dimension,
    stored_embedding_settings,
)
from unified_kg_rag.adapters.storage.filter_schema import (
    FilterFields,
    opensearch_filter_fields,
    union_filter_fields,
)
from unified_kg_rag.adapters.storage.search_engine import SearchEngineClient
from unified_kg_rag.adapters.store_errors import is_fatal_retrieval_error
from unified_kg_rag.domain.models import (
    Config,
    Constants,
    RetrievalResult,
    SearchQuery,
    SearchType,
)
from unified_kg_rag.domain.retrieval.index_prefixes import configured_index_prefixes
from unified_kg_rag.ports.model_factory import EmbeddingFactoryPort
from unified_kg_rag.shared import (
    ConfigurationError,
    EmbeddingModelError,
    IndexNotFoundError,
    InvalidFilterError,
    get_logger,
)
from unified_kg_rag.shared.exceptions import (
    StoreAccessDeniedError,
    StoreRequestRejectedError,
    StoreTimeoutError,
)
from unified_kg_rag.shared.utils import (
    EMBEDDING_FIELD_SUFFIX,
    strip_embedding_fields,
    text_digest,
)
from unified_kg_rag.shared.utils.scripts import (
    has_dense_script,
    is_dense_script_char,
)

logger = get_logger(__name__)

# OpenSearch's kNN rejection of a query vector of another dimension than the
# index's (the reason of a 400).
_INVALID_DIMENSION_MARKER = "query vector has invalid dimension"
# OpenSearch's default index.max_terms_count: the most values one terms query
# may hold.
_DEFAULT_MAX_TERMS_COUNT = 65536
# How much of a query the lexical (multi_match) side searches for. Lucene
# allows 1,024 clauses per query (indices.query.bool.max_clause_count), and a
# multi_match makes one per analyzed term per field, so a long question failed
# the search (400 "maxClauseCount", or 500 too_many_nested_clauses). A term is
# counted per word, and per character of Hangul, Han or Kana text, which
# analyzes to about a term per character. The terms get 1,000 clauses spread
# over the fields, at most 200 terms (five fields or fewer; text units search
# one more field per translated language): the question up to 150 of them and
# the optional keywords what is left. Fuzziness rewrites each term into up to
# max_expansions terms, so it is kept to short queries (entity names, keyword
# lists) and a longer question is matched exactly. The vector side still
# embeds the whole query.
_MAX_LEXICAL_QUERY_TERMS = 150
_LEXICAL_TERM_BUDGET = 200
_LEXICAL_CLAUSE_BUDGET = 1000
_MAX_FUZZY_QUERY_TERMS = 10
_FUZZY_MAX_EXPANSIONS = 10


def _invalid_dimension_reason(reason: str | None) -> str | None:
    """OpenSearch's invalid-dimension rejection in a 400's ``reason``, if any."""
    start = (reason or "").lower().find(_INVALID_DIMENSION_MARKER)
    if start < 0:
        return None
    return (reason or "")[start:].rstrip(" .,")


def _lexical_query_text(text: str, max_terms: int) -> tuple[str, int]:
    """``text`` cut to ``max_terms`` terms (see the budget), and its term count."""
    terms = 0
    in_word = False
    for i, ch in enumerate(text):
        if is_dense_script_char(ch):
            terms += 1
            in_word = False
        elif not ch.isalnum():
            in_word = False
        elif not in_word:
            terms += 1
            in_word = True
        if terms > max_terms:
            return text[:i].rstrip(), max_terms
    return text, terms


def _index_setting(settings: dict[str, Any], name: str, default: int) -> int:
    """``settings[name]``, also when written ``index.<name>``.

    OpenSearch takes either spelling of an index setting.
    """
    return int(settings.get(name, settings.get(f"index.{name}", default)))


class OpenSearchRetriever(BaseGraphRAGRetriever):
    # Query embeddings are reused across the sub-queries of one search (local
    # search sends the same entity-focus text to several indices) and across
    # repeated queries. Bounded LRU: entries are only reused for identical
    # text; 256 x a 1024-dim vector is ~2 MB.
    QUERY_EMBEDDING_CACHE_SIZE: ClassVar[int] = 256

    def __init__(
        self,
        config: Config,
        opensearch_client: SearchEngineClient,
        *,
        embedding_factory: EmbeddingFactoryPort,
        **kwargs: Any,
    ):
        super().__init__(config, **kwargs)
        self._opensearch_client = opensearch_client
        self._opensearch_config = config.indexing.opensearch
        # Private attributes: this is a pydantic model, which rejects
        # undeclared public attributes.
        self._max_size = self._opensearch_config.max_query_size
        # Hard ceiling on hits per query (OpenSearch index.max_result_window
        # default). Fetch-by-id size may be floored up to the terms-filter size
        # but never beyond this, since the index is created with the default.
        self._max_result_window = _index_setting(
            self._opensearch_config.index_settings, "max_result_window", 10000
        )
        self._embedding_factory = embedding_factory
        # Queries embed with the model and dimension the indexer embeds with
        # (OpenSearchIndexer), so their vectors are comparable.
        self._embedding_model_id = configured_embedding_model(self._opensearch_config)
        self._embedding_dimension = resolve_embedding_dimension(
            embedding_factory, self._opensearch_config
        )
        self._embedding_model = self._embedding_factory.get_model(
            self._opensearch_config.embedding_model_id,
            dimensions=self._embedding_dimension,
        )
        # Aliases whose indices were checked to hold this model's embeddings.
        self._embedding_checked_aliases: set[str] = set()
        # Whether skipping the check for want of permission was logged.
        self._embedding_check_denied_logged = False
        self._query_embedding_cache: OrderedDict[str, list[float]] = OrderedDict()
        # One embedding call per text in flight: concurrent sub-queries of one
        # search share it instead of each missing the cache.
        self._pending_embeddings: dict[str, asyncio.Future[list[float]]] = {}
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
            if query_vector is not None:
                await self._check_embedding_settings(index_prefixes, query.suffix)

            self._check_terms_filter_sizes(query.filters)
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

            logger.info(
                "OpenSearch retrieval completed - retrieved: %s results (%.2fs)",
                len(final_results),
                processing_time,
            )
            return final_results

        except Exception as e:
            # Log loudly with the traceback. Re-raise fatal errors (an
            # unusable store, a refused configuration, a query embedding model
            # call that cannot succeed) so they surface instead of
            # masquerading as "0 results"; degrade only on transient ones.
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
                index_query,
                search_type,
                lexical_fields,
                vector_fields,
                query_vector,
                sort=self._importance_sort(prefix, index_query, search_type),
            )
            target_alias = self._get_name(prefix, query.suffix)
            search_tasks.append(
                self._execute_search([target_alias], body, params, query.suffix)
            )

        return search_tasks

    def _importance_sort(
        self, prefix: str, query: SearchQuery, search_type: SearchType
    ) -> list[dict[str, Any]] | None:
        """The order of a search without query text, on an index with ``rank``.

        Such a search (a filter browse, e.g. the edges incident to some
        entities) has no relevance to rank by, so OpenSearch returns its hits
        in index order and a capped page kept arbitrary ones. Rank them by the
        artifact's importance instead: ``rank`` (degree), then ``weight``,
        then ``id`` for a stable order. A fetch by id is left alone: it asks
        for every id it names, so no cut drops one.
        """
        if (
            search_type != SearchType.LEXICAL
            or (query.query and query.query != "*")
            or "id" in (query.filters or {})
        ):
            return None
        declared = self._index_filter_fields.get(prefix)
        if declared is None or "rank" not in declared.fields:
            return None
        return [
            {name: {"order": order}}
            for name, order in (("rank", "desc"), ("weight", "desc"), ("id", "asc"))
            if name in declared.fields
        ]

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
        pending = self._pending_embeddings.get(query_text)
        if pending is None:
            pending = asyncio.ensure_future(self._store_embedding(query_text))
            self._pending_embeddings[query_text] = pending
            pending.add_done_callback(
                lambda done: self._settle_embedding(query_text, done)
            )
        # Shielded: one waiter's cancellation must not cancel the others' call.
        return await asyncio.shield(pending)

    def _settle_embedding(self, query_text: str, done: asyncio.Future) -> None:
        self._pending_embeddings.pop(query_text, None)
        # Mark a failure as retrieved: when every waiter was cancelled nobody
        # awaits the shared call, and asyncio would log "Task exception was
        # never retrieved". Waiters still awaiting it get the error as usual.
        if not done.cancelled():
            done.exception()

    async def _store_embedding(self, query_text: str) -> list[float]:
        try:
            vector: list[float] = await self._embedding_model.aembed_query(query_text)
        except Exception as e:
            # Raised after the client's own retries. Every vector section of
            # the query shares this one embedding: degrading it to "no
            # results" read as "could not find relevant information".
            raise EmbeddingModelError(
                f"Query embedding failed: {type(e).__name__}: {e}"
            ) from e
        cache = self._query_embedding_cache
        cache[query_text] = vector
        if len(cache) > self.QUERY_EMBEDDING_CACHE_SIZE:
            cache.popitem(last=False)
        return vector

    async def _check_embedding_settings(
        self, index_prefixes: list[str], suffix: str | None
    ) -> None:
        """Raise when an index to search holds another model's embeddings.

        A kNN search of a query vector of another dimension fails on every
        index, and one of another model of the same dimension ranks by
        unrelated vectors; either way the answer would come back empty or
        wrong while the query reports success. Each alias is checked once per
        retriever, against the model and dimension its indices record in
        ``_meta`` (indexes built before they did are checked by their
        ``knn_vector`` dimension alone). A missing or unreadable index is left
        to the search itself.

        Raises:
            ConfigurationError: An index was built with another embedding
                model or dimension than the query embeds with.
        """
        aliases = [
            alias
            for prefix in index_prefixes
            if prefix in self._field_mappings
            and (alias := self._get_name(prefix, suffix))
            not in self._embedding_checked_aliases
        ]
        if not aliases:
            return
        for alias, mappings in zip(
            aliases,
            await asyncio.gather(*(self._index_mappings(alias) for alias in aliases)),
            strict=True,
        ):
            if mappings is None:
                continue
            for index_name, index_mappings in mappings.items():
                self._raise_on_embedding_mismatch(index_name, index_mappings)
            self._embedding_checked_aliases.add(alias)

    async def _index_mappings(self, alias: str) -> dict[str, dict[str, Any]] | None:
        try:
            mappings: dict[str, dict[str, Any]] = (
                await self._opensearch_client.aget_mappings(alias)
            )
        except Exception as e:  # noqa: BLE001 - the search reports real failures
            if isinstance(e, StoreAccessDeniedError):
                # A search-only principal may not read mappings; the search
                # itself still runs, and reports its own errors.
                self._embedding_checked_aliases.add(alias)
                if not self._embedding_check_denied_logged:
                    self._embedding_check_denied_logged = True
                    logger.warning(
                        "Not allowed to read the mappings of %s (%s), so queries "
                        "are not checked against the embedding model the indices "
                        "were built with. Grant indices:admin/mappings/get to "
                        "enable the check.",
                        alias,
                        e,
                    )
                return None
            if isinstance(e, StoreTimeoutError):
                raise self._unanswered_error(alias) from e
            if is_fatal_retrieval_error(e):
                raise
            logger.debug("Could not read the mappings of %s: %s", alias, e)
            return None
        return mappings

    def _unanswered_error(self, target: str) -> StoreTimeoutError:
        """The error a request OpenSearch never answered fails the query with."""
        search = self._config.search
        return StoreTimeoutError(
            f"OpenSearch did not answer a request on {target} within "
            f"{search.store_timeout_seconds} s in {search.store_max_attempts} "
            "attempt(s) (search.store_timeout_seconds, search.store_max_attempts)"
        )

    def _raise_on_embedding_mismatch(
        self, index_name: str, mappings: dict[str, Any]
    ) -> None:
        stored_model, stored_dimensions = stored_embedding_settings(mappings)
        model, dimension = self._embedding_model_id, self._embedding_dimension
        if stored_dimensions <= {dimension} and stored_model in (None, model):
            return
        stored = f"model '{stored_model}'" if stored_model else "an unrecorded model"
        shown_dimensions = ", ".join(str(d) for d in sorted(stored_dimensions))
        raise ConfigurationError(
            f"The OpenSearch index '{index_name}' holds embeddings of {stored} "
            f"(dimension {shown_dimensions or 'unknown'}), but queries embed with "
            f"'{model}' (dimension {dimension}), so its vector search would fail "
            "or rank by unrelated vectors. Query with the embedding model and "
            "dimension the corpus was ingested with "
            "(indexing.opensearch.embedding_model_id / embedding_dimension), or "
            "re-ingest the corpus with this model (indexing.reset: true)."
        )

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
        *,
        sort: list[dict[str, Any]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        size = min(query.top_k * query.retrieval_multiplier, self._max_size)
        # Fetch-by-id callers (local/mix expansion, source resolution) set
        # top_k=len(ids) and pass those ids as an ``id`` filter to retrieve
        # every one. Capping size at max_query_size (default 100) would
        # silently drop ids beyond 100, so size is floored at the id count,
        # bounded by max_result_window (the cluster's hard hits ceiling). Only
        # the id list counts: a long list in another filter narrows the hits
        # rather than naming them, and must not raise size (and kNN k).
        ids = (query.filters or {}).get("id")
        wanted = len(ids) if isinstance(ids, list) else 0
        if wanted > size:
            size = min(wanted, self._max_result_window)
            if wanted > size:
                logger.warning(
                    "A filter lists %s ids, more than the %s hits one search "
                    "returns (max_result_window); the rest are not fetched",
                    wanted,
                    size,
                )
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
        search_body: dict[str, Any] = {
            "size": size,
            "query": main_query,
            "_source": {"excludes": [f"*{EMBEDDING_FIELD_SUFFIX}"]},
        }
        if sort:
            # Sorted hits carry no _score unless asked to track it.
            search_body["sort"] = sort
            search_body["track_scores"] = True
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

        budget = min(
            _LEXICAL_TERM_BUDGET, _LEXICAL_CLAUSE_BUDGET // max(len(fields), 1)
        )
        text, terms = _lexical_query_text(
            query.query, min(_MAX_LEXICAL_QUERY_TERMS, budget)
        )
        main_query: dict[str, Any] = {"multi_match": {"query": text, "fields": fields}}
        # AUTO allows one edit on a 3-5 character term. On Latin words that
        # absorbs a typo; on Hangul/Han/Kana, where one character is a whole
        # syllable or morpheme, it matches a different word ("가나라상사" for
        # "가나다상사"). A query containing such text is matched exactly, as
        # is a long one (see _MAX_FUZZY_QUERY_TERMS).
        if (
            not has_dense_script(text)
            and terms <= _MAX_FUZZY_QUERY_TERMS
            and terms * _FUZZY_MAX_EXPANSIONS <= budget
        ):
            main_query["multi_match"]["fuzziness"] = "AUTO"
            main_query["multi_match"]["max_expansions"] = _FUZZY_MAX_EXPANSIONS
            terms *= _FUZZY_MAX_EXPANSIONS

        optional_text, _ = _lexical_query_text(
            " ".join(query.optional_keywords), budget - terms
        )
        if not optional_text:
            return main_query

        should_clause = {"multi_match": {"query": optional_text, "fields": fields}}

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
        if lexical_fields:
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

    def _check_terms_filter_sizes(self, filters: dict[str, Any] | None) -> None:
        """Reject a list filter longer than one ``terms`` query may be.

        A ``terms`` query is one clause however many values it holds, so a
        long id list needs no splitting; the limit is the index's
        ``max_terms_count`` (65,536 by default), which the cluster would
        otherwise answer with a 400.

        Raises:
            InvalidFilterError: A list filter has more values than that.
        """
        limit = _index_setting(
            self._opensearch_config.index_settings,
            "max_terms_count",
            _DEFAULT_MAX_TERMS_COUNT,
        )
        for key, value in (filters or {}).items():
            if isinstance(value, list) and len(value) > limit:
                raise InvalidFilterError(
                    f"Filter '{key}' lists {len(value)} values; an OpenSearch "
                    f"terms query takes at most {limit} (index.max_terms_count). "
                    "Narrow the filter."
                )

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
            shown = ", ".join(aliases)
            if isinstance(e, StoreAccessDeniedError):
                raise ConfigurationError(
                    f"OpenSearch denied the search on {shown}: {e}. Grant the "
                    "query role indices:data/read/search on these indices, or "
                    "check its credentials."
                ) from e
            if isinstance(e, StoreTimeoutError):
                raise self._unanswered_error(shown) from e
            if reason := _invalid_dimension_reason(
                e.reason if isinstance(e, StoreRequestRejectedError) else None
            ):
                raise ConfigurationError(
                    f"OpenSearch rejected the query vector on {shown}: "
                    f"{reason}. The index was built with another embedding model "
                    f"or dimension than queries embed with ('{self._embedding_model_id}', "
                    f"dimension {self._embedding_dimension}). Query with the "
                    "embedding settings the corpus was ingested with, or re-ingest "
                    "it with this model (indexing.reset: true)."
                ) from e
            if isinstance(e, StoreRequestRejectedError):
                # A 400 (or a clause-limit 500) is the same for every retry and
                # every query: a filter value the field cannot hold, a setting
                # naming something the cluster lacks, a query over the cluster's
                # clause limit. Degrading it answered from no (or unfiltered)
                # context.
                raise ConfigurationError(
                    f"OpenSearch rejected the search on {shown}: "
                    f"{e.reason}. Check the query "
                    "filters (a value of another type than the field's) and the "
                    "OpenSearch settings the search uses, such as "
                    "indexing.opensearch.hybrid_search_pipeline_name and "
                    "indices.query.bool.max_clause_count."
                ) from e
            # Any other fatal error (an unreachable or overloaded cluster)
            # propagates; a failure of this one search degrades to no hits.
            if is_fatal_retrieval_error(e):
                logger.error("Fatal search error on indices %s: %s", aliases, e)
                raise
            if is_index_not_found(e):
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
            # A community report's full_content opens with its title and
            # summary (CommunityReport.render_full_content): repeating them
            # spent the context budget twice, and a character cut kept only
            # the repeats.
            content_parts.append(full_content)
        else:
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
