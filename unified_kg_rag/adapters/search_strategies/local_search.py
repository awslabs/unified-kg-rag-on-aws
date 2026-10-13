# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from typing import Any, ClassVar

from langchain_core.runnables import RunnableConfig

from unified_kg_rag.adapters.retrieval.base import (
    BaseSearchStrategy,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.domain.models import (
    RetrievalResult,
    SearchQuery,
    SearchResult,
    SearchStrategy,
    SearchType,
)
from unified_kg_rag.domain.retrieval.strategy_registry import (
    QueryInput,
    register_strategy,
)
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import text_digest

logger = get_logger(__name__)


def _consume_outcome(future: asyncio.Future[Any]) -> None:
    """Mark an abandoned future's exception as retrieved."""
    if not future.cancelled():
        future.exception()


@register_strategy(SearchStrategy.LOCAL, query_inputs=frozenset({QueryInput.ENTITIES}))
class LocalSearchStrategy(BaseSearchStrategy):
    # Entities-index candidates fetched per focus entity.
    ENTITY_CANDIDATES_PER_FOCUS: ClassVar[int] = 2

    async def asearch(
        self, query: SearchQuery, config: RunnableConfig | None = None
    ) -> SearchResult:
        start_time = time.time()
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
        # instead of waiting for it.
        side_sections = asyncio.gather(
            self._retrieve_community_reports(query),
            self._retrieve_relationships(query),
            self._retrieve_claims(query),
        )
        try:
            all_results, chain_stats = await self._retrieve_entity_chain(query)
        except BaseException:
            side_sections.cancel()
            # A side section may already have failed (a missing index fails
            # them all): consume its outcome so it is not logged later as
            # "exception was never retrieved"; the chain's error is raised.
            side_sections.add_done_callback(_consume_outcome)
            raise
        community_reports, relationships, claims = await side_sections

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
            top_k=query.top_k,
            retrieval_multiplier=query.retrieval_multiplier,
            query=query.query,
            per_type_quota=self._per_type_quota(query.top_k),
            rerank_only_types={SectionType.TEXT.value},
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

        expanded_entity_nodes = await self._expand_via_graph(
            query, candidate_entity_ids
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

        # Fusion keeps at most the text quota of them: fetch no more.
        text_unit_ids = self._rank_text_unit_ids(filtered_entity_nodes)[
            : self._per_type_quota(query.top_k)[SectionType.TEXT.value]
        ]
        logger.debug(
            "Found %s text units: '%s%s'",
            len(text_unit_ids),
            ", ".join(text_unit_ids[:5]),
            "..." if len(text_unit_ids) > 5 else "",
        )

        # Both lookups depend only on the expansion, so they run together.
        text_units, bridge_relationships = await asyncio.gather(
            self._retrieve_documents(
                text_unit_ids, query.suffix, filters=query.filters
            ),
            self._retrieve_bridge_relationships(
                query, list(dict.fromkeys(candidate_entity_ids + expanded_entity_ids))
            ),
        )
        all_results = {"graph_entities": filtered_entity_nodes, **text_units}
        if bridge_relationships:
            all_results["bridge_relationships"] = bridge_relationships
        stats = {
            "candidate_entity_count": len(candidate_entity_ids),
            "expanded_entity_count": len(expanded_entity_ids),
            "text_unit_count": len(text_unit_ids),
        }
        return all_results, stats

    async def _find_candidate_entities(
        self, query: SearchQuery
    ) -> list[RetrievalResult]:
        """The entities-index hits that seed graph expansion (each with an id)."""
        if not self.document_retriever:
            return []

        # Map the extracted entity focus onto the entities index; when no
        # entities were extracted, fall back to the raw query text (as the
        # claims/community-report/relationship sections do) so local search
        # still seeds graph expansion instead of returning no chunks at all.
        if query.entity_focus:
            entity_query = " ".join(query.entity_focus)
            n_candidates = len(query.entity_focus) * self.ENTITY_CANDIDATES_PER_FOCUS
        else:
            entity_query = query.query
            n_candidates = query.top_k
        if not entity_query:
            return []

        search_query = SearchQuery(
            query=entity_query,
            search_type=query.search_type,
            top_k=n_candidates,
            index_prefixes=[self.config.indexing.opensearch.entities_index_prefix],
            suffix=query.suffix,
            filters=self._scoped_filters(query),
        )

        results = await self._safe_aretrieve(
            self.document_retriever, search_query, "Candidate entity lookup"
        )
        return [res for res in results if res.source]

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
            score = node.score
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
        ids = node.metadata.get("text_unit_ids") or []
        if isinstance(ids, str):
            return 1
        return len(ids)

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

        search_query = SearchQuery(
            query="",
            search_type=SearchType.LEXICAL,
            top_k=len(text_unit_ids),
            index_prefixes=[self.config.indexing.opensearch.text_units_index_prefix],
            suffix=suffix,
            filters={**(filters or {}), "id": text_unit_ids},
        )

        results = await self._safe_aretrieve(
            self.document_retriever, search_query, "Text unit fetch"
        )
        if not results:
            return {}
        return {"text_units": self._restore_rank(results, text_unit_ids)}
