# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import time
from collections import OrderedDict
from collections.abc import Coroutine
from itertools import zip_longest
from typing import Any, ClassVar

from unified_kg_rag.adapters.aws.opensearch import is_index_not_found
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    current_query_scope,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.adapters.storage.embedding_settings import (
    configured_embedding_model,
    resolve_embedding_dimension,
    stored_build_record,
    stored_embedding_settings,
)
from unified_kg_rag.adapters.storage.filter_schema import (
    FilterFields,
    IndexLimits,
    check_filter_values,
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
from unified_kg_rag.domain.retrieval.hybrid_normalization import combine_min_max
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
from unified_kg_rag.shared.utils.concurrency import discard, gather_bounded
from unified_kg_rag.shared.utils.scripts import (
    has_dense_script,
    is_dense_script_char,
)

logger = get_logger(__name__)

# OpenSearch's kNN rejection of a query vector of another dimension than the
# index's (the reason of a 400).
_INVALID_DIMENSION_MARKER = "query vector has invalid dimension"
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


def _beyond_index_limits(error: Exception) -> bool:
    """Whether OpenSearch rejected a search for asking more hits than the
    index's ``max_result_window`` or more terms than its ``max_terms_count``."""
    return any(
        marker in str(error)
        for marker in ("Result window is too large", "index.max_terms_count")
    )


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


def _scope(
    filters: list[dict[str, Any]], must_not: list[dict[str, Any]]
) -> dict[str, Any]:
    """The bool clauses that scope a search: its filters and exclusions."""
    scope: dict[str, Any] = {"filter": filters} if filters else {}
    if must_not:
        scope["must_not"] = must_not
    return scope


class OpenSearchRetriever(BaseGraphRAGRetriever):
    # Query embeddings are reused across the sub-queries of one search (local
    # search sends the same entity-focus text to several indices) and across
    # repeated queries. Bounded LRU: entries are only reused for identical
    # text; 256 x a 1024-dim vector is ~2 MB.
    QUERY_EMBEDDING_CACHE_SIZE: ClassVar[int] = 256
    # Hits each search of a hybrid query fetches (at least) for the min-max
    # normalization ``_fuse`` applies before it cuts to ``size``. Normalized
    # over only the ``size`` hits returned, a hit's fused score, and so the
    # ranking, depended on how many hits were asked for: the top 4 of a
    # 4-hit and a 40-hit query shared 75% of their hits.
    HYBRID_NORMALIZATION_WINDOW: ClassVar[int] = 50

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
        # The configured limits; each index's own may be lower (_limits).
        self._configured_limits = IndexLimits.of(self._opensearch_config.index_settings)
        # The limits in force on each alias read, or the read in flight
        # (see _limits).
        self._index_limits: dict[str, IndexLimits | asyncio.Future[Any]] = {}
        # Whether a denied settings read was logged.
        self._settings_denied_logged = False
        self._embedding_factory = embedding_factory
        # Queries embed with the model and dimension the indexer embeds with
        # (OpenSearchIndexer), so their vectors are comparable.
        self._embedding_model_id = configured_embedding_model(config)
        self._embedding_dimension = resolve_embedding_dimension(
            embedding_factory, config
        )
        self._embedding_model = self._embedding_factory.get_model(
            config.embedding_model_id,
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

            found: list[list[RetrievalResult]] = []
            for attempt in range(2):
                limits = await self._limits_by_prefix(index_prefixes, query.suffix)
                terms_limit = min(
                    (lim.max_terms_count for lim in limits.values()),
                    default=self._configured_limits.max_terms_count,
                )
                if batches := self._batches(query, limits, terms_limit):
                    parts = await gather_bounded(
                        self._config.search.max_concurrency,
                        (self.aretrieve(b) for b in batches),
                    )
                    return [hit for part in parts for hit in part]
                asked = self._within_terms(query, terms_limit)
                self._check_filter_values(asked.filters, terms_limit)
                self._check_filter_values(asked.exclude, terms_limit)
                search_tasks = self._create_search_tasks(
                    asked, search_type, index_prefixes, query_vector, limits
                )

                if not search_tasks:
                    logger.warning(
                        "No searchable indices available for query: %s",
                        query_preview,
                    )
                    return []
                try:
                    found = await asyncio.gather(*search_tasks)
                except ConfigurationError as e:
                    if attempt or not _beyond_index_limits(e):
                        raise
                    # An index replaced since its limits were read (a reset
                    # or reindex under other index_settings): read again, once.
                    # A read still in flight is left to finish: it is newer.
                    for prefix in limits:
                        name = self._get_name(prefix, query.suffix)
                        if isinstance(self._index_limits.get(name), IndexLimits):
                            del self._index_limits[name]
                    logger.info("Index limits changed; searching again: %s", e)
                    continue
                break

            final_results = self._interleave(found)[
                : query.top_k * query.retrieval_multiplier
            ]

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

    @staticmethod
    def _batches(
        query: SearchQuery, limits: dict[str, IndexLimits], terms_limit: int
    ) -> list[SearchQuery] | None:
        """A text-less search split by a list filter longer than one search
        takes, or ``None`` when it fits.

        The strategies size every list by ``aindex_limits`` themselves
        (``BaseSearchStrategy._fetch_by_ids``, ``_ask_incident``, global
        search's browse, the source resolver), so theirs fit unless the
        store rejected the limits they were sized by: asked again after an
        index was rebuilt smaller, a search is split by the ``limits`` just
        read, a fetch by id by ``IndexLimits.id_batch_size``, any other
        list by ``max_terms_count``, each batch with the search's ``top_k``.
        A direct caller's oversized list is split the same way.
        """
        if query.query:
            return None
        id_step = min(
            (lim.id_batch_size for lim in limits.values()), default=terms_limit
        )
        filters = query.filters or {}
        for key, values in filters.items():
            step = id_step if key == "id" else terms_limit
            if not isinstance(values, list) or len(values) <= step:
                continue
            logger.debug(
                "Searching %s %s values in batches of %s", len(values), key, step
            )
            # Each batch keeps the search's top_k: a fetch by id asks for
            # all its ids (top_k=len(ids)), more than any batch names.
            return [
                query.model_copy(
                    update={"filters": {**filters, key: values[i : i + step]}}
                )
                for i in range(0, len(values), step)
            ]
        return None

    @staticmethod
    def _within_terms(query: SearchQuery, terms_limit: int) -> SearchQuery:
        """``query`` with each exclusion list cut to the ``terms_limit``
        values listed last. An exclusion only keeps hits out of the page (the
        walk leaves out the vertices it reached, in reach order: the last
        are the frontier's neighbourhood, whose edges would fill it)."""
        exclude = query.exclude or {}
        if not any(
            isinstance(v, list) and len(v) > terms_limit for v in exclude.values()
        ):
            return query
        cut = {
            k: v[-terms_limit:] if isinstance(v, list) else v
            for k, v in exclude.items()
        }
        return query.model_copy(update={"exclude": cut})

    def _create_search_tasks(
        self,
        query: SearchQuery,
        search_type: SearchType,
        index_prefixes: list[str],
        query_vector: list[float] | None,
        limits: dict[str, IndexLimits],
    ) -> list[Coroutine[Any, Any, list[RetrievalResult]]]:
        """One search per index, each with the filters that index declares,
        sized by the index's ``limits``."""
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

            window = limits[prefix].max_result_window
            size = self._hit_count(index_query, window)
            bodies = self._build_search_bodies(
                index_query,
                search_type,
                lexical_fields,
                vector_fields,
                query_vector,
                size=size,
                sort=self._importance_sort(prefix, index_query, search_type),
                window=window,
            )
            if query.source_fields is not None:
                # A check that needs the ids only ships no document back.
                for body in bodies:
                    body["_source"] = (
                        {"includes": query.source_fields}
                        if query.source_fields
                        else False
                    )
            probe = (
                self._attribute_probe(prefix, index_query.filters)
                if query_vector is not None and search_type != SearchType.LEXICAL
                else None
            )
            target_alias = self._get_name(prefix, query.suffix)
            search_tasks.append(
                self._search_index(target_alias, bodies, size, query.suffix, probe)
            )

        return search_tasks

    def _attribute_probe(
        self, prefix: str, filters: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """A search for one document matching the filters' attribute keys.

        ``attr_<key>`` and ``attributes.<path>`` are mapped only once a
        document carries them, and OpenSearch 2.13 rejects a kNN filter on a
        field the index does not map (a 400, "Rewrite first"), so an
        attribute filter failed every vector and hybrid search of an index
        whose documents lack the attribute, community reports above all. When
        no document matches the attribute clauses the kNN search could find
        nothing anyway, so the index is skipped instead (fail-closed, as
        ``filter_schema`` documents). None when the filters have no such key.
        """
        declared = self._index_filter_fields.get(prefix) if filters else None
        if declared is None:
            return None
        attributes = {
            key: value
            for key, value in (filters or {}).items()
            if key not in declared.fields
        }
        if not attributes:
            return None
        return {
            "size": 1,
            "query": {"bool": {"filter": self._build_filter_clauses(attributes)}},
            "_source": False,
        }

    @staticmethod
    def _interleave(
        per_index: list[list[RetrievalResult]],
    ) -> list[RetrievalResult]:
        """The hits of several indices, each index's next best in turn.

        Scores of different indices are not comparable: a hybrid search
        normalizes each index's best hit to about 1.0, a vector search sums
        the scores of the index's two or three embedding fields, and BM25
        depends on each index's term statistics. Cut by raw score, one index
        crowded the others out before fusion and reranking saw them. Taking
        the indices' hits rank by rank keeps what an equal-weight rank fusion
        of the indices would.
        """
        return [
            hit
            for same_rank in zip_longest(*per_index)
            for hit in same_rank
            if hit is not None
        ]

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

    async def aread_build_record(self, suffix: str | None) -> dict[str, Any] | None:
        """The build record of ``suffix``'s text-unit index (see
        ``VectorIndexer.record_build``), or ``None`` when it records none or
        cannot be read."""
        alias = self._get_name(self._opensearch_config.text_units_index_prefix, suffix)
        mappings = await self._index_mappings(alias)
        for index_mappings in (mappings or {}).values():
            if (record := stored_build_record(index_mappings)) is not None:
                return record
        return None

    async def aindex_limits(self, prefix: str, suffix: str | None) -> IndexLimits:
        """The limits in force on ``prefix``'s index (``_limits``)."""
        return await self._limits(self._get_name(prefix, suffix))

    async def _limits_by_prefix(
        self, index_prefixes: list[str], suffix: str | None
    ) -> dict[str, IndexLimits]:
        """The limits in force on the index of each of ``index_prefixes``."""
        prefixes = [p for p in index_prefixes if p in self._field_mappings]
        found = await asyncio.gather(
            *(self._limits(self._get_name(p, suffix)) for p in prefixes)
        )
        return dict(zip(prefixes, found, strict=True))

    async def _limits(self, alias: str) -> IndexLimits:
        """The limits in force on ``alias``'s indices: the smallest of the
        configured ``index_settings`` and the settings the indices were
        created with, read once per alias. An index keeps its settings when
        the config changes (until a reindex), so a configured window above
        the live one would ask for more hits than a search returns (a 400;
        ``aretrieve`` then reads them again). A concurrent caller waits for
        the read in flight. Settings that cannot be read leave the
        configured limits: kept when the read was denied or found no index,
        read again next time on any other failure."""
        cached = self._index_limits.get(alias)
        if isinstance(cached, IndexLimits):
            return cached
        if cached is not None:
            shared: IndexLimits | None = await asyncio.shield(cached)
            return shared if shared is not None else (await self._read_limits(alias))[0]
        pending: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._index_limits[alias] = pending
        limits: IndexLimits | None = None
        keep = False
        try:
            limits, keep = await self._read_limits(alias)
        finally:
            # Resolved whatever happens (None: the waiters read for
            # themselves). The entry is still this read's: a rejected search
            # drops only limits already read, never a read in flight.
            if keep and limits is not None:
                self._index_limits[alias] = limits
            else:
                self._index_limits.pop(alias, None)
            pending.set_result(limits)
        return limits

    async def _read_limits(self, alias: str) -> tuple[IndexLimits, bool]:
        """``_limits``' read of ``alias``'s settings, and whether to keep it."""
        try:
            live = await self._opensearch_client.aget_index_settings(alias)
        except Exception as e:  # noqa: BLE001 - the search reports real failures
            if isinstance(e, StoreTimeoutError):
                raise self._unanswered_error(alias) from e
            if isinstance(e, StoreAccessDeniedError):
                if not self._settings_denied_logged:
                    self._settings_denied_logged = True
                    logger.warning(
                        "Not allowed to read the settings of %s (%s), so "
                        "searches are sized by the configured index_settings, "
                        "not the live index's max_result_window / "
                        "max_terms_count. Grant indices:monitor/settings/get "
                        "to size them by the live index.",
                        alias,
                        e,
                    )
                return self._configured_limits, True
            logger.debug("Could not read the settings of %s: %s", alias, e)
            return self._configured_limits, is_index_not_found(e)
        return (
            IndexLimits.of(self._opensearch_config.index_settings, *live.values()),
            True,
        )

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
            "(models.embedding.model_id / dimension), or "
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

    def _hit_count(self, query: SearchQuery, window: int) -> int:
        """How many hits of one index a search returns: at most ``window``
        (the index's ``max_result_window``)."""
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
        if wanted > window:
            logger.warning(
                "A filter lists %s ids, more than the %s hits one search "
                "returns (max_result_window); the rest are not fetched",
                wanted,
                window,
            )
        elif size > window:
            logger.warning("Search size %s cut to max_result_window %s", size, window)
        return min(max(size, wanted), window)

    def _build_search_bodies(
        self,
        query: SearchQuery,
        search_type: SearchType,
        lexical_fields: list[str],
        vector_fields: list[str],
        query_vector: list[float] | None,
        *,
        size: int,
        window: int,
        sort: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """The searches of one index: one, or for a hybrid search two.

        A hybrid search is a lexical and a kNN search whose hits
        ``_fuse`` combines, rather than one ``hybrid`` query fused by a search
        pipeline: OpenSearch 2.13's normalization processor min-max normalized
        the lexical sub-query over a document outside its returned hits (the
        last-indexed match), which skewed every fused score of the index.
        Fused here, each search is normalized over exactly the hits it
        returned, on OpenSearch and the embedded engine alike. Both fetch
        ``HYBRID_NORMALIZATION_WINDOW`` hits at least, so the normalization
        range does not depend on ``size``; ``_fuse`` cuts to ``size``.
        """
        filters = self._build_filter_clauses(query.filters)
        # The exclusions sit beside the filters, in their bool: a bool of
        # must_not alone matches every other document, which the embedded
        # engine builds in full before the filters narrow it.
        must_not = self._build_filter_clauses(query.exclude)
        scope = _scope(filters, must_not)
        lexical = self._build_lexical_query(query, lexical_fields)
        if scope:
            lexical = {"bool": {"must": [lexical], **scope}}
        if query_vector is not None and search_type != SearchType.LEXICAL:
            if search_type != SearchType.VECTOR:
                size = min(max(size, self.HYBRID_NORMALIZATION_WINDOW), window)
            knn = self._search_body(
                self._build_vector_query(
                    query_vector, self._knn_k(size), vector_fields, filters, must_not
                ),
                size,
            )
            if search_type == SearchType.VECTOR:
                return [knn]
            return [self._search_body(lexical, size), knn]
        if search_type != SearchType.LEXICAL and not query.query:
            # A filter browse: no text to score by.
            lexical = {"bool": scope} if scope else {"match_all": {}}
        return [self._search_body(lexical, size, sort)]

    def _knn_k(self, size: int) -> int:
        """The ``k`` of a kNN search returning ``size`` hits.

        The lucene engine ignores ``index.knn.algo_param.ef_search`` and
        explores ``k`` candidates per field, so a ``k`` of ``size`` (10 to 40
        for a query) cut recall@10 to about 0.8; it gets
        ``vector_search.ef_search`` at least. faiss and nmslib read the index
        setting, and ``size`` hits are kept either way.
        """
        vector_search = self._opensearch_config.vector_search
        if vector_search.get("engine", "lucene") != "lucene":
            return size
        return max(size, int(vector_search.get("ef_search", 100)))

    @staticmethod
    def _search_body(
        query: dict[str, Any], size: int, sort: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        # Embedding vectors are only needed server-side for kNN scoring; never
        # ship them back (community reports alone carry three per hit).
        body: dict[str, Any] = {
            "size": size,
            "query": query,
            "_source": {"excludes": [f"*{EMBEDDING_FIELD_SUFFIX}"]},
        }
        if sort:
            # Sorted hits carry no _score unless asked to track it.
            body["sort"] = sort
            body["track_scores"] = True
        return body

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
        must_not: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # Attach filters (and exclusions) to the kNN clause itself
        # (efficient/pre-filtering on the Lucene engine). Without this, a
        # VECTOR query that also carries filters (e.g. a non-empty query
        # scoped to a community_id set) returns the global top-k nearest
        # vectors, leaking documents outside the filter.
        scope = _scope(filters or [], must_not or [])

        def _knn(field: str) -> dict[str, Any]:
            clause: dict[str, Any] = {"vector": vector, "k": k}
            if scope:
                clause["filter"] = {"bool": scope}
            return {"knn": {field: clause}}

        if len(fields) == 1:
            return _knn(fields[0])

        return {"bool": {"should": [_knn(field) for field in fields]}}

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
    def _check_filter_values(filters: dict[str, Any] | None, limit: int) -> None:
        """Reject a filter value OpenSearch would answer with a 400.

        A ``terms`` query takes at most the index's ``max_terms_count``
        values (65,536 by default), which the cluster would otherwise answer
        with a 400 (a text-less search is split under it first: ``_batches``).

        A value OpenSearch would refuse is rejected too (``check_filter_values``),
        so both engines refuse the same filters.

        Raises:
            InvalidFilterError: A list filter has more values than that, or a
                value is malformed.
        """
        check_filter_values(filters)
        for key, value in (filters or {}).items():
            if isinstance(value, list) and len(value) > limit:
                raise InvalidFilterError(
                    f"Filter '{key}' lists {len(value)} values; an OpenSearch "
                    f"terms query takes at most {limit} (index.max_terms_count). "
                    "Narrow the filter."
                )

    async def _search_index(
        self,
        alias: str,
        bodies: list[dict[str, Any]],
        size: int,
        suffix: str | None,
        probe: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """The hits of ``bodies`` on ``alias``; two (hybrid) are fused.

        With a ``probe`` (``_attribute_probe``) that finds nothing, none. The
        searches start alongside the probe, so a filtered query waits for one
        round trip instead of two; when the probe finds nothing they are
        cancelled and their outcome (the 400 a kNN filter on an unmapped
        field draws, say) ignored. A search that failed transiently or found
        no index degrades the index to no hits (logged once), as one hybrid
        search did.
        """
        searches = [asyncio.ensure_future(self._search(alias, b)) for b in bodies]
        if probe is not None:
            try:
                found = await self._probe(alias, probe)
            except BaseException:
                await discard(searches)
                raise
            if isinstance(found, Exception) or not found:
                await discard(searches)
            if isinstance(found, Exception):
                return await self._degraded_search(alias, suffix, found)
            if not found:
                logger.debug(
                    "No document of %s matches the attribute filters; "
                    "skipping its search",
                    alias,
                )
                return []
        outcomes = await asyncio.gather(*searches)
        responses: list[list[dict[str, Any]]] = []
        for outcome in outcomes:
            if isinstance(outcome, Exception):
                return await self._degraded_search(alias, suffix, outcome)
            responses.append(outcome)
        if len(responses) == 1:
            return [self._parse_hit(hit) for hit in responses[0]]
        return self._fuse(responses, size)

    async def _probe(
        self, alias: str, probe: dict[str, Any]
    ) -> list[dict[str, Any]] | Exception | bool:
        """``probe``'s hits on ``alias``, or its outcome earlier in this query.

        The sub-searches of one query (DRIFT's follow-ups above all) probe
        the same index with the same filters; the outcome is kept in the
        query's memo (``query_scope``) only, so the next query sees what an
        ingestion or reset in between changed. A concurrent sub-search waits
        for the probe in flight and gets its outcome, a failure included (it
        degrades as the owner does). A failed probe is not kept, and when
        the owner is cancelled or raises, a waiter probes for itself.
        """
        memo = current_query_scope()
        key = f"probe\n{alias}\n{json.dumps(probe, sort_keys=True, default=str)}"
        if memo is None:
            return await self._search(alias, probe)
        pending = memo.get(key)
        if pending is not None:
            shared: list[dict[str, Any]] | Exception | bool | None = (
                await asyncio.shield(pending)
            )
            if shared is not None:
                return shared
            return await self._search(alias, probe)
        outcome: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        memo[key] = outcome
        try:
            found = await self._search(alias, probe)
        except BaseException:
            del memo[key]
            outcome.set_result(None)  # the waiters probe for themselves
            raise
        if isinstance(found, Exception):
            del memo[key]  # not kept: the next sub-search probes again
            outcome.set_result(found)
        else:
            outcome.set_result(bool(found))
        return found

    def _fuse(
        self, sub_hits: list[list[dict[str, Any]]], size: int
    ) -> list[RetrievalResult]:
        """The lexical and kNN hits of one index fused, best ``size`` first.

        Min-max normalization of each search over its own hits and their
        weighted arithmetic mean (``search.hybrid``), as the neural-search
        normalization processor defines it (``combine_min_max``).
        """
        weights = self._config.search.hybrid
        hits: dict[str, dict[str, Any]] = {}
        sub_results: list[list[tuple[str, float]]] = []
        for found in sub_hits:
            scored = []
            for hit in found:
                key = f"{hit.get('_index', '')}\x00{hit.get('_id', '')}"
                hits.setdefault(key, hit)
                scored.append((key, float(hit.get("_score") or 0.0)))
            sub_results.append(scored)
        fused = combine_min_max(
            sub_results, [weights.lexical_weight, weights.vector_weight]
        )
        return [
            self._parse_hit({**hits[key], "_score": score})
            for key, score in fused[:size]
        ]

    async def _search(
        self, alias: str, body: dict[str, Any]
    ) -> list[dict[str, Any]] | Exception:
        """The hits of one search, or the error when it may degrade to none.

        Raises:
            ConfigurationError: OpenSearch denied or rejected the search.
            StoreTimeoutError: OpenSearch did not answer.
        """
        try:
            response = await self._opensearch_client.asearch(index=alias, body=body)
            hits: list[dict[str, Any]] = response.get("hits", {}).get("hits", [])
            return hits
        except Exception as e:
            if isinstance(e, StoreAccessDeniedError):
                raise ConfigurationError(
                    f"OpenSearch denied the search on {alias}: {e}. Grant the "
                    "query role indices:data/read/search on these indices, or "
                    "check its credentials."
                ) from e
            if isinstance(e, StoreTimeoutError):
                raise self._unanswered_error(alias) from e
            if reason := _invalid_dimension_reason(
                e.reason if isinstance(e, StoreRequestRejectedError) else None
            ):
                raise ConfigurationError(
                    f"OpenSearch rejected the query vector on {alias}: "
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
                    f"OpenSearch rejected the search on {alias}: "
                    f"{e.reason}. Check the query "
                    "filters (a value of another type than the field's) and the "
                    "OpenSearch settings the search uses, such as "
                    "indices.query.bool.max_clause_count."
                ) from e
            # Any other fatal error (an unreachable or overloaded cluster)
            # propagates (aretrieve logs it); a failure of this one search
            # degrades to no hits. A search discarded after the attribute probe
            # logs neither.
            if is_fatal_retrieval_error(e):
                raise
            return e

    async def _degraded_search(
        self, alias: str, suffix: str | None, error: Exception
    ) -> list[RetrievalResult]:
        if is_index_not_found(error):
            await self._require_ingested_suffix(suffix, error)
            # A missing alias is a config/index mismatch (the index was
            # never built), not a transient failure.
            logger.warning(
                "Index not found for %s; skipping it (was it built by the "
                "ingestion config?): %s",
                alias,
                error,
            )
            return []
        logger.error("Search failed on %s: %s", alias, error)
        return []

    async def ahas_index(self, prefix: str, suffix: str | None) -> bool | None:
        try:
            return bool(
                await self._opensearch_client.aindex_exists(
                    self._get_name(prefix, suffix)
                )
            )
        except Exception as e:
            if is_fatal_retrieval_error(e):
                raise
            return None  # cannot tell

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
