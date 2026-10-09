# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import math
import re
import time
from collections.abc import Coroutine
from typing import Any, ClassVar

import boto3
from gremlin_python.process.graph_traversal import (
    GraphTraversal,
    GraphTraversalSource,
    __,
)
from gremlin_python.process.traversal import Order, P, TextP, Traversal

from unified_kg_rag.adapters.aws import NeptuneClient
from unified_kg_rag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    is_fatal_retrieval_error,
)
from unified_kg_rag.adapters.retrieval.token_manager import SectionType
from unified_kg_rag.adapters.storage.filter_schema import (
    ATTRIBUTE_KEY_PREFIX,
    NEPTUNE_FILTER_FIELDS,
    FilterFields,
    union_filter_fields,
)
from unified_kg_rag.domain.models import Config, RetrievalResult, SearchQuery
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import text_digest

logger = get_logger(__name__)


class NeptuneRetriever(BaseGraphRAGRetriever):
    SEED_NODE_LIMIT: ClassVar[int] = 10
    DEFAULT_MAX_HOPS: ClassVar[int] = 3

    def __init__(
        self,
        config: Config,
        neptune_client: NeptuneClient,
        boto_session: boto3.Session | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, boto_session, **kwargs)
        self._neptune_client = neptune_client
        self._neptune_config = config.indexing.neptune
        self._max_hops = self._neptune_config.max_hops
        self._max_results_per_hop = self._neptune_config.max_results_per_hop

    def close(self) -> None:
        """Close the underlying Neptune websocket + thread pool (best-effort)."""
        self._neptune_client.close()

    async def aclose(self) -> None:
        """Async teardown: the blocking Neptune close runs off the loop thread."""
        await asyncio.to_thread(self._neptune_client.close)

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        start_time = time.time()
        logger.info(
            "Neptune retrieval started - query: %s ('%s')",
            text_digest(query.query),
            query.search_type.value,
        )

        try:
            # Opening the connection is blocking: gremlinpython drives its
            # websocket handshake with run_until_complete on a private loop,
            # which fails on a thread that already runs one (the old code
            # only worked because the CLIs apply nest_asyncio).
            g = await asyncio.to_thread(lambda: self._neptune_client.g)
            seed_entities, seed_communities = await self._get_seed_nodes(g, query)

            if not seed_entities and not seed_communities:
                logger.warning(
                    "No seed nodes found for query: %s", text_digest(query.query)
                )
                return []

            traversal_results = await self._traverse_from_seeds(
                g, seed_entities, seed_communities, query
            )
            results = self._process_traversal_results(traversal_results, query)

            self._record_metrics(
                len(results), len(seed_entities), len(seed_communities)
            )

            processing_time = time.time() - start_time
            logger.info(
                "Neptune retrieval completed - retrieved: %s results (%.2fs)",
                len(results),
                processing_time,
            )

            return results

        except Exception as e:
            self._record_metric("error_count", 1)
            # Re-raise clearly-fatal errors (auth/credentials/endpoint/
            # connection) so a broken configuration is not silently reported as
            # "no results"; degrade to an empty list only on transient failures.
            if is_fatal_retrieval_error(e):
                logger.error("Neptune retrieval failed (fatal): %s", e, exc_info=True)
                raise
            logger.error(
                "Neptune retrieval failed (transient, degrading to empty "
                "results): %s",
                e,
                exc_info=True,
            )
            return []
        finally:
            self._record_timing("total_retrieval", time.time() - start_time)

    async def _get_seed_nodes(
        self, g: GraphTraversalSource, query: SearchQuery
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        label_prefixes = self._normalize_label_prefixes(query.label_prefixes)
        seed_ids_from_filter = query.filters.get("id") if query.filters else None

        if isinstance(seed_ids_from_filter, list):
            seeds = [{"id": _id} for _id in seed_ids_from_filter]
            is_community_search = (
                self._neptune_config.community_label_prefix in label_prefixes
            )
            is_entity_search = (
                self._neptune_config.entity_label_prefix in label_prefixes
            )

            if is_community_search and not is_entity_search:
                return [], seeds
            if is_entity_search and not is_community_search:
                return seeds, []

            return seeds, []

        tasks = {}
        if self._neptune_config.entity_label_prefix in label_prefixes:
            tasks["entity"] = self._find_seeds_by_type(g, query, is_community=False)
        if self._neptune_config.community_label_prefix in label_prefixes:
            tasks["community"] = self._find_seeds_by_type(g, query, is_community=True)

        if not tasks:
            return [], []

        results = await asyncio.gather(*tasks.values())
        results_map = dict(zip(tasks.keys(), results, strict=True))
        return results_map.get("entity", []), results_map.get("community", [])

    def _normalize_label_prefixes(
        self, label_prefixes: str | list[str] | None
    ) -> list[str]:
        if isinstance(label_prefixes, str):
            return [label_prefixes]
        return label_prefixes or [
            self._neptune_config.entity_label_prefix,
            self._neptune_config.community_label_prefix,
        ]

    async def _find_seeds_by_type(
        self, g: GraphTraversalSource, query: SearchQuery, is_community: bool
    ) -> list[dict[str, Any]]:
        label_prefix = (
            self._neptune_config.community_label_prefix
            if is_community
            else self._neptune_config.entity_label_prefix
        )
        # Entity vertices store "rank" (the indexed entity rank) and community
        # vertices "size"; seeds are the highest-valued matches.
        order_by_prop = "size" if is_community else "rank"

        label = self._get_label(label_prefix, query.suffix)
        traversal = g.V().hasLabel(label)

        if query.entity_focus:
            traversal = traversal.has("name", P.within(query.entity_focus))
        else:
            logger.debug(
                "No entity focus provided; matching names against query terms: '%s'",
                query.query,
            )
            query_terms = re.findall(r"\b\w+\b", query.query.lower())
            if not query_terms:
                return []
            text_search_filter = __.or_(
                *[__.has("name", TextP.containing(term)) for term in query_terms]
            )
            traversal = traversal.where(text_search_filter)

        kind = "community" if is_community else "entity"
        filters, exempt = self._scope_filters_to_labels({label: kind}, query.filters)
        traversal = self._apply_filters(traversal, filters, exempt)

        traversal = (
            traversal.order()
            .by(order_by_prop, Order.desc)
            .limit(self.SEED_NODE_LIMIT)
            .valueMap(True)
        )

        raw_results = await self._execute_traversal(traversal)
        return [self._clean_property_map(r) for r in raw_results]

    def filter_fields(self) -> FilterFields:
        return union_filter_fields(NEPTUNE_FILTER_FIELDS.values())

    @staticmethod
    def _scope_filters_to_labels(
        label_kinds: dict[str, str], filters: dict[str, Any] | None
    ) -> tuple[dict[str, Any] | None, dict[str, list[str]]]:
        """Restrict filters to the vertex labels that declare each key.

        ``label_kinds`` maps each target label to its ``NEPTUNE_FILTER_FIELDS``
        kind. Mirrors the OpenSearch per-index scoping: ``has(key, ...)`` drops
        every vertex lacking ``key``, so keys no target label declares are
        dropped, and for a key only some labels declare the returned ``exempt``
        map lists the labels the filter must not touch.
        """
        if not filters:
            return filters, {}
        kept: dict[str, Any] = {}
        exempt: dict[str, list[str]] = {}
        for key, value in filters.items():
            lacking = [
                label
                for label, kind in label_kinds.items()
                if not NEPTUNE_FILTER_FIELDS[kind].declares(key)
            ]
            if len(lacking) == len(label_kinds):
                logger.debug(
                    "Dropped filter key '%s': no %s vertex declares it",
                    key,
                    "/".join(label_kinds),
                )
                continue
            kept[key] = value
            if lacking:
                exempt[key] = lacking
        return kept or None, exempt

    @staticmethod
    def _filter_steps(key: str, value: Any) -> list[tuple[str, Any]]:
        """``(key, predicate)`` pairs for one filter entry (``has`` arguments)."""
        if isinstance(value, list):
            return [(key, P.within(value))]
        if isinstance(value, dict):
            return [
                (key, getattr(P, op)(val))
                for op, val in value.items()
                if op in {"gte", "lte", "gt", "lt", "eq", "neq"}
            ]
        return [(key, value)]

    @staticmethod
    def _filter_predicate(key: str, steps: list[tuple[str, Any]]) -> GraphTraversal:
        """Anonymous traversal a vertex must match for one filter entry.

        ``attr_*`` keys apply where present: a vertex without the property
        passes. Entity vertices carry their own extracted attributes (e.g.
        ``attr_role``) but not document attribute filters (``attr_category``
        lives in OpenSearch), so a strict match would empty graph expansion for
        every document-attribute filter. Every other key is strict.
        """
        matched = __.has(*steps[0])
        for step in steps[1:]:
            matched = matched.has(*step)
        if key.startswith(ATTRIBUTE_KEY_PREFIX):
            return __.or_(matched, __.hasNot(key))
        return matched

    @classmethod
    def _apply_filters(
        cls,
        traversal: GraphTraversal,
        filters: dict[str, Any] | None,
        exempt_labels: dict[str, list[str]] | None = None,
    ) -> GraphTraversal:
        if not filters:
            return traversal

        for key, value in filters.items():
            if key == "id":
                continue
            steps = cls._filter_steps(key, value)
            if not steps:
                continue
            predicate = cls._filter_predicate(key, steps)
            exempt = (exempt_labels or {}).get(key)
            if exempt:
                # Only the labels that carry ``key`` are filtered; vertices of
                # the exempt labels pass through untouched.
                traversal = traversal.or_(__.hasLabel(*exempt), predicate)
            elif key.startswith(ATTRIBUTE_KEY_PREFIX):
                traversal = traversal.where(predicate)
            else:
                for step in steps:
                    traversal = traversal.has(*step)
        return traversal

    async def _traverse_from_seeds(
        self,
        g: GraphTraversalSource,
        seed_entities: list[dict[str, Any]],
        seed_communities: list[dict[str, Any]],
        query: SearchQuery,
    ) -> list[dict[str, Any]]:
        tasks: list[Coroutine] = []
        if seed_entities:
            tasks.append(self._traverse_from_entities(g, seed_entities, query))
        if seed_communities:
            tasks.append(self._traverse_from_communities(g, seed_communities, query))

        if not tasks:
            return []

        results_list = await asyncio.gather(*tasks)
        return [item for sublist in results_list for item in sublist]

    async def _traverse_from_entities(
        self, g: GraphTraversalSource, seeds: list[dict[str, Any]], query: SearchQuery
    ) -> list[dict[str, Any]]:
        seed_ids = [s["id"] for s in seeds if "id" in s][: self.SEED_NODE_LIMIT]
        if not seed_ids:
            return []

        # Use the configured max_hops directly (it is already validated/bounded
        # by NeptuneIndexingConfig). DEFAULT_MAX_HOPS is only a fallback for an
        # unset value — clamping UP to it silently ignored a user lowering hops
        # to bound entity-expansion cost.
        hops = self._max_hops or self.DEFAULT_MAX_HOPS
        entity_label = self._get_label(
            self._neptune_config.entity_label_prefix, query.suffix
        )
        logger.info(
            "Traversing from entities with label: '%s', seed_count: %s, max_hops: %s",
            entity_label,
            len(seed_ids),
            hops,
        )

        # Over-fetch, then let `_process_traversal_results` rank and cut to
        # top_k * retrieval_multiplier. The fetch budget is split evenly across
        # the seeds and spent inside a per-seed local(), because a limit()
        # inside repeat() counts every traverser of the whole traversal: one
        # global cap let the first seeds' neighbourhoods use it up and left the
        # other seeds with none. Within a seed, local(both().limit(n)) caps how
        # many neighbours any one node contributes per hop (max_results_per_hop)
        # so a hub cannot fan out unbounded. The seeds come back through the
        # identity() branch (path length 1, proximity 1.0); repeat().emit()
        # emits from the first hop on, never the seed itself, and the expansion
        # excludes the seeds so it spends its budget on new entities. Both
        # shapes stay Neptune-friendly: repeat().times().emit() and
        # union(identity(), ...). An emit() before repeat() is not optimized by
        # Neptune and times out.
        fetch_limit = (
            query.top_k
            * query.retrieval_multiplier
            * self._neptune_config.traversal_fetch_multiplier
        )
        per_seed_limit = math.ceil(fetch_limit / len(seed_ids))
        traversal = (
            g.V()
            .hasLabel(entity_label)
            .has("id", P.within(seed_ids))
            .union(
                __.identity(),
                __.local(
                    __.repeat(
                        __.local(
                            __.both()
                            .hasLabel(entity_label)
                            .limit(self._max_results_per_hop)
                        )
                        .has("id", P.without(seed_ids))
                        .dedup()
                        .limit(per_seed_limit)
                    )
                    .times(hops)
                    .emit()
                ),
            )
        )
        filters, exempt = self._scope_filters_to_labels(
            {entity_label: "entity"}, query.filters
        )
        traversal = self._apply_filters(traversal, filters, exempt)
        traversal = self._with_projection(
            traversal,
            with_degree=self._neptune_config.entity_importance_source == "degree",
        )
        return await self._execute_traversal(traversal)

    async def _traverse_from_communities(
        self, g: GraphTraversalSource, seeds: list[dict[str, Any]], query: SearchQuery
    ) -> list[dict[str, Any]]:
        seed_ids = [c["id"] for c in seeds if "id" in c]
        if not seed_ids:
            return []

        # Mirror entity traversal: honor the configured max_hops (validated by
        # NeptuneIndexingConfig) rather than capping at DEFAULT_MAX_HOPS, so the
        # config is authoritative in both directions.
        hops = self._max_hops or self.DEFAULT_MAX_HOPS
        community_label = self._get_label(
            self._neptune_config.community_label_prefix, query.suffix
        )
        entity_label = self._get_label(
            self._neptune_config.entity_label_prefix, query.suffix
        )
        logger.info(
            "Traversing from communities with label: '%s', seed_count: %s, max_hops: %s",
            community_label,
            len(seed_ids),
            hops,
        )

        # Same budget shape as _traverse_from_entities: a limit() inside
        # repeat() is global, so the fetch width is split across the seed
        # communities and spent inside a per-seed local(), once on members and
        # once on their neighbourhood. _process_traversal_results ranks the
        # result and cuts it to top_k * retrieval_multiplier.
        fetch_limit = (
            query.top_k
            * query.retrieval_multiplier
            * self._neptune_config.traversal_fetch_multiplier
        )
        per_seed_limit = math.ceil(fetch_limit / len(seed_ids))
        traversal = (
            g.V()
            .hasLabel(community_label)
            .has("id", P.within(seed_ids))
            .union(
                __.identity(),
                __.local(
                    __.in_("MemberOf").hasLabel(entity_label).limit(per_seed_limit)
                ),
                __.local(
                    __.in_("MemberOf")
                    .hasLabel(entity_label)
                    .repeat(
                        __.local(
                            __.both()
                            .hasLabel(entity_label)
                            .limit(self._max_results_per_hop)
                        )
                        .dedup()
                        .limit(per_seed_limit)
                    )
                    .times(hops)
                    .emit()
                ),
            )
        )
        filters, exempt = self._scope_filters_to_labels(
            {community_label: "community", entity_label: "entity"}, query.filters
        )
        traversal = self._apply_filters(traversal, filters, exempt)
        traversal = self._with_projection(traversal)
        return await self._execute_traversal(traversal)

    @staticmethod
    def _with_projection(
        traversal: GraphTraversal, with_degree: bool = False
    ) -> GraphTraversal:
        # `rank` is what the indexer writes on entity vertices (see
        # NeptuneIndexer.index_entities); `size` is the community counterpart.
        keys = ["node", "path", "node_type"] + (["degree"] if with_degree else [])
        projected = (
            traversal.project(*keys)
            .by(
                __.value_map(
                    "id", "name", "description", "rank", "text_unit_ids", "size"
                )
            )
            .by(__.path().by(__.value_map("name")))
            .by(__.label())
        )
        if with_degree:
            projected = projected.by(__.bothE().count())
        return projected

    @staticmethod
    async def _execute_traversal(traversal: Traversal) -> list[Any]:
        try:
            # to_list() is a BLOCKING Gremlin round trip; running it directly in
            # this coroutine would block the event loop and serialize the seed
            # lookups that callers fan out via asyncio.gather. Offload it to a
            # worker thread so the gather actually overlaps.
            result: list[Any] = await asyncio.to_thread(traversal.to_list)
            return result
        except Exception as e:
            # Fatal errors (auth/credentials/endpoint/connection) must reach
            # `aretrieve`, which re-raises them; swallowing them here turned a
            # broken Neptune configuration into a silent "no seed nodes found".
            if is_fatal_retrieval_error(e):
                raise
            logger.error("Gremlin traversal execution failed: %s", e)
            return []

    @staticmethod
    def _clean_property_map(prop_map: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v[0] if isinstance(v, list) and len(v) == 1 else v
            for k, v in prop_map.items()
        }

    def _process_traversal_results(
        self, traversal_results: list[dict[str, Any]], query: SearchQuery
    ) -> list[RetrievalResult]:
        community_sizes = [
            float(self._clean_property_map(item.get("node", {})).get("size") or 0)
            for item in traversal_results
            if self._neptune_config.community_label_prefix
            in str(item.get("node_type", ""))
        ]
        max_community_size = max(community_sizes, default=0.0)
        importance_key = self._entity_importance_key()
        max_importance = 0.0
        if importance_key is not None:
            max_importance = max(
                (
                    self._as_float(self._node_data(item).get(importance_key)) or 0.0
                    for item in traversal_results
                    if self._neptune_config.community_label_prefix
                    not in str(item.get("node_type", ""))
                ),
                default=0.0,
            )

        # Entity expansion runs per seed, so a node two seeds reach arrives once
        # per seed; keep the copy with the best score (its shortest path).
        best: dict[Any, RetrievalResult] = {}
        for item in traversal_results:
            node_data = self._node_data(item)
            node_id = node_data.get("id")
            if not node_id:
                continue

            result = self._create_retrieval_result(
                item,
                node_data,
                query,
                max_community_size=max_community_size,
                importance_key=importance_key,
                max_importance=max_importance,
            )
            kept = best.get(node_id)
            if kept is None or (result.score or 0.0) > (kept.score or 0.0):
                best[node_id] = result

        results = sorted(best.values(), key=lambda x: x.score or 0.0, reverse=True)
        if self._neptune_config.traversal_fetch_multiplier > 1:
            results = self._cut_entities(
                results, query.top_k * query.retrieval_multiplier
            )
        return results

    def _entity_importance_key(self) -> str | None:
        """The node property graph expansion reads as entity importance."""
        source = self._neptune_config.entity_importance_source
        return None if source == "none" else source

    def _node_data(self, item: dict[str, Any]) -> dict[str, Any]:
        """The cleaned property map, with the projected degree folded in."""
        node_data = self._clean_property_map(item.get("node", {}))
        if "degree" in item:
            node_data["degree"] = item["degree"]
        return node_data

    @staticmethod
    def _as_float(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _cut_entities(
        ranked: list[RetrievalResult], limit: int
    ) -> list[RetrievalResult]:
        """Keep the ``limit`` best-ranked entities; community results pass through.

        The entity traversal over-fetches (``traversal_fetch_multiplier``) so the
        cut happens here, after relevance ranking, instead of in emit order.
        """
        kept: list[RetrievalResult] = []
        entities = 0
        for result in ranked:
            if result.retriever_type == SectionType.ENTITY.value:
                if entities >= limit:
                    continue
                entities += 1
            kept.append(result)
        return kept

    def _create_retrieval_result(
        self,
        item: dict[str, Any],
        node_data: dict[str, Any],
        query: SearchQuery,
        max_community_size: float = 0.0,
        importance_key: str | None = None,
        max_importance: float = 0.0,
    ) -> RetrievalResult:
        node_id = str(node_data.get("id"))
        node_type_str = item.get("node_type", "unknown")
        is_community = self._neptune_config.community_label_prefix in node_type_str

        section_type = SectionType.COMMUNITY if is_community else SectionType.ENTITY

        path_data = item.get("path", [])
        content = self._build_content(node_data, path_data, is_community)
        score = self._calculate_relevance(
            node_data,
            path_data,
            is_community,
            max_community_size,
            importance_key=importance_key,
            max_importance=max_importance,
        )

        return RetrievalResult(
            content=content,
            score=score,
            source=node_id,
            retriever_type=str(section_type.value),
            metadata={**node_data, "_node_type": node_type_str},
        )

    @staticmethod
    def _build_content(
        node_data: dict[str, Any], path_data: list[dict[str, Any]], is_community: bool
    ) -> str:
        if is_community:
            content_parts = [
                f"Community: {node_data.get('name', 'Unknown')}",
                f"Size: {node_data.get('size', 'N/A')}",
            ]
        else:
            content_parts = [
                f"Entity: {node_data.get('name', 'Unknown')}",
                f"Description: {node_data.get('description', 'N/A')}",
            ]

        path_names = [p.get("name", [""])[0] for p in path_data if isinstance(p, dict)]
        if path_names:
            content_parts.append(f"Path: {' -> '.join(filter(None, path_names))}")

        return "\n".join(content_parts)

    @staticmethod
    def _calculate_relevance(
        node: dict[str, Any],
        path: list[dict[str, Any]],
        is_community: bool,
        max_community_size: float = 0.0,
        importance_key: str | None = None,
        max_importance: float = 0.0,
    ) -> float:
        """Mean of the node's importance and its proximity to the seeds, in [0, 1].

        ``proximity = 1 / path length`` (1.0 for a seed, 0.5 one hop out), so
        nearer nodes rank first and importance breaks ties among equally near
        ones. An entity's importance is its ``importance_key`` value (indexed
        ``rank`` or projected ``degree``) relative to the largest one in the
        same result, neutral 0.5 when absent; a community's is its size
        relative to the largest community in the same result, so no
        corpus-size constant is needed. Without an ``importance_key`` the
        legacy ``importance`` property is read (entity vertices never store it,
        so this is the neutral 0.5). The two terms are weighted equally because
        neither has a measured reason to dominate. (A query-text term was
        dropped: graph expansion passes the whole question, which never occurs
        inside a node name.)
        """
        proximity = 1.0 / (len(path) or 1)
        if is_community:
            size = float(node.get("size") or 0)
            importance = size / max_community_size if max_community_size > 0 else 0.0
        elif importance_key is not None:
            raw = NeptuneRetriever._as_float(node.get(importance_key))
            importance = (
                raw / max_importance if raw is not None and max_importance > 0 else 0.5
            )
        else:
            importance = float(node.get("importance", 0.5))
        importance = min(max(importance, 0.0), 1.0)
        return (importance + proximity) / 2.0

    def _record_metrics(
        self, result_count: int, entity_count: int, community_count: int
    ) -> None:
        self._record_metric("retrieved_count", result_count)
        self._record_metric("seed_entity_count", entity_count)
        self._record_metric("seed_community_count", community_count)
