# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from typing import Any, ClassVar

from langchain_core.runnables import RunnableConfig

from unified_kg_rag.adapters.retrieval.base import BaseSearchStrategy, scope_filters
from unified_kg_rag.adapters.retrieval.strategy_registry import (
    QueryInput,
    register_strategy,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.domain.ingestion.base_resolver import compact_name_key
from unified_kg_rag.domain.ingestion.seed_graph import SEED_SOURCE_KIND, is_seed_unit
from unified_kg_rag.domain.models import (
    Constants,
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.domain.retrieval.index_prefixes import configured_index_prefixes
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import text_digest
from unified_kg_rag.shared.utils.concurrency import gather_bounded, gather_or_cancel

logger = get_logger(__name__)


@register_strategy(SearchStrategy.LOCAL, query_inputs=frozenset({QueryInput.ENTITIES}))
class LocalSearchStrategy(BaseSearchStrategy):
    # Entities-index hits fetched per focus entity, among which its exact
    # name match is looked for (``_find_candidate_entities``).
    ENTITY_CANDIDATES_PER_FOCUS: ClassVar[int] = 2
    # The section types reranked against the query. DRIFT, whose follow-up
    # queries are local searches, clears it and reranks their evidence once.
    rerank_types: frozenset[str] = frozenset({SectionType.TEXT.value})

    def filtered_index_prefixes(self, strategy: SearchStrategy) -> list[str]:
        # Entities, their reports, relationships and claims, and the text
        # units they cite: each index the pipeline builds.
        return configured_index_prefixes(self.config)

    async def asearch(
        self, query: SearchQuery, config: RunnableConfig | None = None
    ) -> SearchResult:
        start_time = time.time()
        rerank = self._new_rerank_trace()
        logger.info(
            "Local search started - query: %s ('%s') with %s focus entities",
            text_digest(query.query),
            query.search_type.value,
            len(query.entity_focus),
        )
        logger.debug("Local search focus entities: '%s'", ", ".join(query.entity_focus))

        # MS GraphRAG local search builds context from entities + the community
        # reports those entities belong to + in-network relationships + text
        # units (+ claims). The community-report, relationship and claim
        # sections are queried from the entity focus alone, so they run
        # concurrently with the entity -> graph -> text-unit chain below
        # instead of waiting for it; the first error cancels the others.
        reports_task, relationships_task, claims_task = (
            asyncio.ensure_future(section)
            for section in (
                self._retrieve_community_reports(query),
                self._retrieve_relationships(query),
                self._retrieve_claims(query),
            )
        )
        chain_task = asyncio.ensure_future(self._retrieve_entity_chain(query))
        tasks = [reports_task, relationships_task, claims_task, chain_task]
        await gather_or_cancel(tasks)
        all_results, chain_stats = chain_task.result()
        community_reports = reports_task.result()
        relationships = relationships_task.result()
        claims = claims_task.result()

        # The community-report and relationship sections give a local query the
        # higher-level community synthesis and the relationship descriptions,
        # not just raw entities and chunks.
        if community_reports:
            all_results["community_reports"] = community_reports
        if relationships:
            all_results["relationships"] = relationships
        # MS GraphRAG injects covariates (claims) into local-search context.
        # Gated strictly on claim extraction being enabled so the default path
        # (claims off) is unchanged: no extra retrieval is issued.
        if claims:
            all_results["claims"] = claims

        # The same divergences as mix apply to local (shared fuse path).
        # Give each section type its own quota so the diversity filter + fusion don't
        # collapse the candidate set to top_k before assembly (was dropping gold KG
        # items), and rerank ONLY text chunks so content-vs-query reranking doesn't bury
        # multi-hop bridge entities/relations. Mirrors MS local's proportional,
        # per-section context assembly.
        final_results = await self._fuse_and_rerank(
            all_results,
            rerank_trace=rerank,
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
            per_type_quota=self._per_type_quota(query.top_k),
            rerank_only_types=set(self.rerank_types),
        )

        processing_time = time.time() - start_time

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
                "candidate_entity_count": chain_stats["candidate_entity_count"],
                "expanded_entity_count": chain_stats["expanded_entity_count"],
                "text_unit_count": chain_stats["text_unit_count"],
                "rerank": rerank.summary(),
            },
        )

    async def _retrieve_entity_chain(
        self, query: SearchQuery
    ) -> tuple[dict[str, list[RetrievalResult]], dict[str, int]]:
        """Entities -> Neptune expansion -> ranked text units (the dependent chain)."""
        candidate_hits = await self._find_candidate_entities(query)
        candidate_entity_ids = [str(hit.source) for hit in candidate_hits]
        logger.debug(
            "Found %s candidate entities: '%s%s'",
            len(candidate_entity_ids),
            ", ".join(candidate_entity_ids[:5]),
            "..." if len(candidate_entity_ids) > 5 else "",
        )

        # The candidates were found under the caller's filters.
        expanded_entity_nodes = await self._expand_via_graph(
            query,
            candidate_entity_ids,
            seeds_pass=True,
            seed_weights=self.seed_weights(query, candidate_hits),
        )
        if not expanded_entity_nodes and candidate_hits:
            # A successful expansion returns at least the seeds, so none means
            # it failed (and degraded): keep the entities the index found, and
            # the text units they name, rather than answer without either.
            logger.warning(
                "Graph expansion returned no entities; using the %s entities "
                "the entity index matched",
                len(candidate_hits),
            )
            expanded_entity_nodes = candidate_hits
        expanded_entity_nodes = self._with_hit_filters(
            expanded_entity_nodes, candidate_hits
        )
        filtered_entity_nodes = self._filter_entities(
            expanded_entity_nodes,
            frequency_threshold=self.config.search.local_search.entity_frequency_threshold,
            seed_ids=set(candidate_entity_ids),
        )

        expanded_entity_ids = self._get_ids(filtered_entity_nodes, "id")
        logger.debug(
            "Expanded to %s entities: '%s%s'",
            len(expanded_entity_ids),
            ", ".join(expanded_entity_ids[:5]),
            "..." if len(expanded_entity_ids) > 5 else "",
        )

        # Fusion keeps at most the text quota of them: fetch no more (the
        # first that pass, under a filter on text units).
        text_quota = self._per_type_quota(query.top_k)[SectionType.TEXT.value]
        ranked_unit_ids = self._rank_text_unit_ids(filtered_entity_nodes)
        logger.debug(
            "Found %s text units: '%s%s'",
            len(ranked_unit_ids),
            ", ".join(ranked_unit_ids[:5]),
            "..." if len(ranked_unit_ids) > 5 else "",
        )

        async def text_units_of_quota() -> dict[str, list[RetrievalResult]]:
            chosen = await self._passing_ranked_units(
                query, ranked_unit_ids, text_quota
            )
            return await self._retrieve_documents(
                chosen, query.suffix, filters=query.filters
            )

        # Both lookups depend only on the expansion, so they run together.
        text_units_task = asyncio.ensure_future(text_units_of_quota())
        bridges_task = asyncio.ensure_future(
            self._retrieve_bridge_relationships(
                query, list(dict.fromkeys(candidate_entity_ids + expanded_entity_ids))
            )
        )
        await gather_or_cancel([text_units_task, bridges_task])
        text_units, bridge_relationships = (
            text_units_task.result(),
            bridges_task.result(),
        )
        all_results = {"graph_entities": filtered_entity_nodes, **text_units}
        if bridge_relationships:
            all_results["bridge_relationships"] = bridge_relationships
        stats = {
            "candidate_entity_count": len(candidate_entity_ids),
            "expanded_entity_count": len(expanded_entity_ids),
            "text_unit_count": len(text_units.get("text_units", [])),
        }
        return all_results, stats

    @staticmethod
    def _with_hit_filters(
        nodes: list[RetrievalResult], hits: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """``nodes`` with a candidate's document filters on its graph copy.

        A graph vertex projects no document filters, so the provenance of a
        seed fact local search cites (``rag_chain.seed_provenance``) named
        no seed graph. Each node a seed graph states (``source_kinds``)
        that is a candidate takes the entity hit's ``attributes.filters``,
        which name them; any other node is returned as it came.
        """
        key = Constants.FILTERS.value
        filters_of: dict[str, Any] = {}
        for hit in hits:
            attributes = (hit.metadata or {}).get("attributes")
            if isinstance(attributes, dict) and isinstance(attributes.get(key), dict):
                filters_of.setdefault(str(hit.source), attributes[key])
        if not filters_of:
            return nodes
        kept: list[RetrievalResult] = []
        for node in nodes:
            metadata = node.metadata or {}
            attributes = metadata.get("attributes")
            filters = filters_of.get(str(node.source))
            if (
                filters is None
                or SEED_SOURCE_KIND not in (metadata.get("source_kinds") or ())
                or (isinstance(attributes, dict) and key in attributes)
            ):
                kept.append(node)
                continue
            attributes = {**(attributes if isinstance(attributes, dict) else {})}
            attributes[key] = filters
            kept.append(
                node.model_copy(
                    update={"metadata": {**metadata, "attributes": attributes}}
                )
            )
        return kept

    async def _find_candidate_entities(
        self, query: SearchQuery
    ) -> list[RetrievalResult]:
        """The entities-index hits that seed graph expansion (each with an id).

        Each focus entity is looked up on its own, concurrently (at most
        ``search.max_concurrency`` at once), and seeds the expansion with
        the hits that are its name (``compact_name_key``), else with its top
        hit when that scores at least
        ``search.local_search.seed_score_ratio`` of the best hit of any
        focus entity. One query of the joined names, two hits per name and
        no floor seeded a wrong neighbour beside the right entity on every
        query of a synthetic 12-query multi-hop corpus, and a generic focus
        word ("province") seeded unrelated entities. Without an
        entity focus the raw query text is looked up instead (as the
        claims/community-report/relationship sections do), its ``top_k``
        hits the seeds, so local search still seeds graph expansion.
        """
        if not self.document_retriever:
            return []
        focus = self._focus_names(query)
        if not focus:
            if not query.query:
                return []
            return await self._entity_hits(query, query.query, query.top_k)
        per_focus = await gather_bounded(
            self.config.search.max_concurrency,
            (
                self._entity_hits(query, name, self.ENTITY_CANDIDATES_PER_FOCUS)
                for name in focus
            ),
        )
        hybrid = self.config.search.hybrid
        return self.focus_seeds(
            focus,
            per_focus,
            self.seed_floor(
                self.config.search.local_search.seed_score_ratio,
                query.search_type,
                hybrid.vector_weight / (hybrid.lexical_weight + hybrid.vector_weight),
            ),
        )

    # The vector weight the seed_score_ratio scale is set at (the default
    # equal weights), where a hit only the vector search found scores at most
    # this fraction of one found by both.
    _SEED_RATIO_VECTOR_WEIGHT: ClassVar[float] = 0.5

    @classmethod
    def seed_floor(
        cls, ratio: float, search_type: SearchType, vector_weight: float
    ) -> float:
        """The fraction of the best hit a focus entity's top hit must score
        to seed: ``ratio`` (``search.local_search.seed_score_ratio``) read
        on the scale of equal hybrid weights.

        A hybrid search's fused score of a hit only the kNN search found is
        at most ``vector_weight`` (the weights sum to 1, each side min-max
        normalized to at most 1), against up to 1 for a hit both found. At
        equal weights that cap is 0.5, so a ratio above 0.5 drops such a
        hit when another focus entity matched both ways. The floor maps
        ``[0, 0.5]`` linearly onto ``[0, vector_weight]`` and ``[0.5, 1]``
        onto ``[vector_weight, 1]``: equal weights keep ``ratio`` as is, and
        any ratio above 0.5 stays above the cap at any weights (0.6 is 0.84
        at a 0.8 vector weight, where 0.6 let every vector-only hit seed).
        A vector or lexical search compares raw scores, so the floor is
        ``ratio`` there.
        """
        if search_type is not SearchType.HYBRID:
            return ratio
        half = cls._SEED_RATIO_VECTOR_WEIGHT
        if ratio <= half:
            return ratio / half * vector_weight
        return vector_weight + (ratio - half) / half * (1.0 - vector_weight)

    async def _entity_hits(
        self, query: SearchQuery, text: str, top_k: int
    ) -> list[RetrievalResult]:
        """The ``top_k`` entities-index hits of ``text`` that have an id."""
        if not self.document_retriever:
            return []
        search_query = SearchQuery(
            query=text,
            search_type=query.search_type,
            top_k=top_k,
            index_prefixes=[self.config.indexing.opensearch.entities_index_prefix],
            suffix=query.suffix,
            filters=self._scoped_filters(query),
        )
        results = await self._safe_aretrieve(
            self.document_retriever, search_query, "Candidate entity lookup"
        )
        return [res for res in results if res.source]

    @staticmethod
    def _focus_names(query: SearchQuery) -> list[str]:
        """The query's focus entities, each once, without blank ones."""
        return [name for name in dict.fromkeys(query.entity_focus) if name.strip()]

    @staticmethod
    def focus_seeds(
        focus: list[str], per_focus: list[list[RetrievalResult]], ratio: float
    ) -> list[RetrievalResult]:
        """The seeds of the focus entities ``focus`` from their own hits
        ``per_focus`` (best first): a focus entity's exact name matches
        (every entity of that ``compact_name_key``), else its top hit if it
        scores at least ``ratio`` times the best hit of all; each seed once,
        in focus order."""
        best = max(
            (hit.score or 0.0 for hits in per_focus for hit in hits), default=0.0
        )
        seeds: dict[str, RetrievalResult] = {}
        for name, hits in zip(focus, per_focus, strict=True):
            key = compact_name_key(name)
            chosen = [
                hit
                for hit in hits
                if key
                and compact_name_key(str((hit.metadata or {}).get("name") or "")) == key
            ]
            if not chosen and hits and (hits[0].score or 0.0) >= ratio * best:
                chosen = hits[:1]
            for hit in chosen:
                seeds.setdefault(str(hit.source), hit)
        return list(seeds.values())

    @classmethod
    def seed_weights(
        cls, query: SearchQuery, seeds: list[RetrievalResult]
    ) -> dict[str, float] | None:
        """The graph walk's weight of each seed of a focused query: its score,
        so a seed reaches as many entities as its share of the seeds' total
        score (``seed_shares``); ``None`` (an even split) without an entity
        focus, whose ``top_k`` seeds are the raw query's hits."""
        if not cls._focus_names(query):
            return None
        return {str(seed.source): float(seed.score or 0.0) for seed in seeds}

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
            filters=self._scoped_filters(query),
        )

        return await self._safe_aretrieve(
            self.document_retriever, search_query, "Claims retrieval"
        )

    async def _retrieve_community_reports(
        self, query: SearchQuery
    ) -> list[RetrievalResult]:
        # Pull the community reports most relevant to the query so local context
        # carries the community-level synthesis (mirrors MS GraphRAG local
        # search). The community-reports index always exists on the GraphRAG
        # path; degrade to no section on any retrieval error.
        if not self.document_retriever:
            return []
        if self._source_kinds_skip_reports(query):
            logger.info(
                "Community reports skipped: the source_kinds filter keeps one "
                "source kind and a report carries none"
            )
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
            filters=self._scoped_filters(query),
        )

        return await self._safe_aretrieve(
            self.document_retriever, search_query, "Community reports retrieval"
        )

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
            filters=self._scoped_filters(query),
        )

        return await self._safe_aretrieve(
            self.document_retriever, search_query, "Relationships retrieval"
        )

    async def _retrieve_bridge_relationships(
        self, query: SearchQuery, entity_ids: list[str]
    ) -> list[RetrievalResult]:
        # MS GraphRAG local search adds the relationships BETWEEN the selected
        # entities ("in-network" first, then out-of-network). The vector query
        # above only finds relationships whose description resembles the query
        # text, which a multi-hop bridge edge usually does not. Fetch the edges
        # incident to the expanded entities, the in-network (both endpoints
        # retrieved) ones first.
        if (
            not self.config.search.local_search.include_bridge_relationships
            or not self.config.indexing.opensearch.build_relationship_vector_index
        ):
            return []
        relationships = await self._fetch_incident_relationships(
            query, [eid for eid in entity_ids if eid], bridge_first=True
        )
        # Fusion keeps at most the relationship quota anyway, and every extra
        # candidate competes in the pre-quota diversity cut (sized to the sum of
        # the quotas) with the text units the answer usually lives in. Offer
        # only as many edges as the quota can seat, best (in-network) first.
        relationships = relationships[
            : self._per_type_quota(query.top_k)[SectionType.RELATIONSHIP.value]
        ]
        if relationships:
            logger.debug(
                "Bridge expansion: %s entities -> %s incident relationships",
                len(entity_ids),
                len(relationships),
            )
        return relationships

    @classmethod
    def _rank_text_unit_ids(cls, entity_nodes: list[RetrievalResult]) -> list[str]:
        # Chunk candidates used to come from
        # `_get_ids(nodes, "text_unit_ids")`, which unions them into a SET — so
        # all ordering was lost, and `_retrieve_documents` then fetches them by
        # id filter with `query=""` (an ID batch fetch, no relevance score). The
        # chunk stream therefore reached fusion in arbitrary order with score 0,
        # and whatever the per-type quota sliced off was an arbitrary subset.
        # That is invisible while expansion is narrow and every chunk is
        # on-topic, but it makes widening the expansion actively harmful: more
        # candidate chunks then means fewer relevant ones survive the quota.
        #
        # MS GraphRAG local (mixed_context._build_text_unit_context) orders
        # the candidate text units by the rank of the first selected entity
        # that cites them, then by how many of that entity's relationships
        # the unit states, before applying its text-unit budget. This ranks
        # by graph support instead: each chunk by (number of expanded
        # entities citing it, best citing entity score), in descending order,
        # so downstream truncation keeps the chunks most entities point to.
        #
        # A seed graph's units are no text chunks (the retriever leaves them
        # out of every text search too): they take no slot of the quota.
        hit_count: dict[str, int] = {}
        best_score: dict[str, float] = {}
        for node in entity_nodes:
            ids = node.metadata.get("text_unit_ids") or []
            if isinstance(ids, str):
                ids = [ids]
            if not isinstance(ids, list):
                continue
            score = node.score
            for raw_id in ids:
                unit_id = str(raw_id)
                if is_seed_unit(unit_id):
                    continue
                hit_count[unit_id] = hit_count.get(unit_id, 0) + 1
                if score > best_score.get(unit_id, float("-inf")):
                    best_score[unit_id] = score
        return sorted(
            hit_count,
            key=lambda unit_id: (hit_count[unit_id], best_score.get(unit_id, 0.0)),
            reverse=True,
        )

    @staticmethod
    def _entity_id(node: RetrievalResult) -> str:
        return str((node.metadata or {}).get("id") or node.source or "")

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
        #
        # Only document units count: an entity a seed graph states is not
        # more generic for it (one seed citation never pushes an entity over
        # the threshold).
        ids = node.metadata.get("text_unit_ids") or []
        if isinstance(ids, str):
            ids = [ids]
        return sum(1 for unit_id in ids if not is_seed_unit(str(unit_id)))

    @classmethod
    def _filter_entities(
        cls,
        expanded_entity_nodes: list[RetrievalResult],
        frequency_threshold: int,
        seed_ids: set[str] | frozenset[str] = frozenset(),
    ) -> list[RetrievalResult]:
        """Drop expanded entities in more than ``frequency_threshold`` text units.

        Too generic to be discriminative: they leave the context as well as
        the text-unit fan-out. The seeds (``seed_ids``, the entities the query
        matched) are exempt -- the query asked about them, however common --
        so a hub-heavy graph cannot filter the context down to no entities.
        """
        filtered_nodes = [
            node
            for node in expanded_entity_nodes
            if cls._entity_id(node) in seed_ids
            or cls._text_unit_count(node) <= frequency_threshold
        ]

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
        self,
        text_unit_ids: list[str],
        suffix: str | None,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, list[RetrievalResult]]:
        if not self.document_retriever or not text_unit_ids:
            return {}
        prefix = self.config.indexing.opensearch.text_units_index_prefix

        search_query = SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=len(text_unit_ids),
            index_prefixes=[prefix],
            suffix=suffix,
            filters=scope_filters(filters, id=text_unit_ids),
        )

        results = await self._fetch_by_ids(
            self.document_retriever, search_query, "Text unit fetch"
        )
        if not results:
            return {}
        return {"text_units": self._restore_rank(results, text_unit_ids)}
