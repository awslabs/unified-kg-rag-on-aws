# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
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

from aws_graphrag.adapters.aws import NeptuneClient
from aws_graphrag.adapters.retrieval.base import (
    BaseGraphRAGRetriever,
    is_fatal_retrieval_error,
)
from aws_graphrag.adapters.retrieval.token_manager import SectionType
from aws_graphrag.domain.models import Config, RetrievalResult, SearchQuery
from aws_graphrag.shared import get_logger

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
        self._min_entity_importance = self._neptune_config.min_entity_importance

    def close(self) -> None:
        """Close the underlying Neptune websocket + thread pool (best-effort)."""
        self._neptune_client.close()

    async def aclose(self) -> None:
        """Async-symmetric teardown; the Neptune client close is synchronous."""
        self._neptune_client.close()

    async def aretrieve(self, query: SearchQuery) -> list[RetrievalResult]:
        start_time = time.time()
        logger.info(
            "Neptune retrieval started - query: '%s...' ('%s')",
            query.query[:50],
            query.search_type.value,
        )

        try:
            g = self._neptune_client.g
            seed_entities, seed_communities = await self._get_seed_nodes(g, query)

            if not seed_entities and not seed_communities:
                logger.warning("No seed nodes found for query: '%s'", query.query)
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
        order_by_prop = "size" if is_community else "importance"
        min_prop_value = None if is_community else self._min_entity_importance

        label = self._get_name(label_prefix.capitalize(), query.suffix)
        traversal = g.V().hasLabel(label)

        if query.entity_focus:
            traversal = traversal.has("name", P.within(query.entity_focus))
        else:
            logger.warning("No entity focus provided. Using query: '%s'", query.query)
            query_terms = re.findall(r"\b\w+\b", query.query.lower())
            if not query_terms:
                return []
            text_search_filter = __.or_(
                *[__.has("name", TextP.containing(term)) for term in query_terms]
            )
            traversal = traversal.where(text_search_filter)

        traversal = self._apply_filters(traversal, query.filters)

        if min_prop_value is not None:
            traversal = traversal.has(order_by_prop, P.gte(min_prop_value))

        traversal = (
            traversal.order()
            .by(order_by_prop, Order.desc)
            .limit(self.SEED_NODE_LIMIT)
            .valueMap(True)
        )

        raw_results = await self._execute_traversal(traversal)
        return [self._clean_property_map(r) for r in raw_results]

    @staticmethod
    def _apply_filters(
        traversal: GraphTraversal, filters: dict[str, Any] | None
    ) -> GraphTraversal:
        if not filters:
            return traversal

        for key, value in filters.items():
            if key == "id":
                continue
            if isinstance(value, list):
                traversal = traversal.has(key, P.within(value))
            elif isinstance(value, dict):
                for op, val in value.items():
                    if op in {"gte", "lte", "gt", "lt", "eq", "neq"}:
                        traversal = traversal.has(key, getattr(P, op)(val))
            else:
                traversal = traversal.has(key, value)
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
        entity_label = self._get_name(
            self._neptune_config.entity_label_prefix.capitalize(), query.suffix
        )
        logger.info(
            "Traversing from entities with label: '%s', seed_count: %s, max_hops: %s",
            entity_label,
            len(seed_ids),
            hops,
        )

        traversal = (
            g.V()
            .hasLabel(entity_label)
            .has("id", P.within(seed_ids))
            .repeat(__.both().dedup().limit(self._max_results_per_hop))
            .times(hops)
            .emit()
            .dedup()
            .hasLabel(entity_label)
            .limit(query.top_k * query.retrieval_multiplier)
        )
        traversal = self._apply_filters(traversal, query.filters)
        traversal = self._with_projection(traversal)
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
        community_label = self._get_name(
            self._neptune_config.community_label_prefix.capitalize(), query.suffix
        )
        entity_label = self._get_name(
            self._neptune_config.entity_label_prefix.capitalize(), query.suffix
        )
        logger.info(
            "Traversing from communities with label: '%s', seed_count: %s, max_hops: %s",
            community_label,
            len(seed_ids),
            hops,
        )

        traversal = (
            g.V()
            .hasLabel(community_label)
            .has("id", P.within(seed_ids))
            .union(
                __.identity(),
                __.in_("MemberOf").hasLabel(entity_label),
                __.in_("MemberOf")
                .hasLabel(entity_label)
                .repeat(__.both().dedup())
                .times(hops)
                .emit(),
            )
            .dedup()
            .limit(query.top_k * query.retrieval_multiplier)
        )
        traversal = self._apply_filters(traversal, query.filters)
        traversal = self._with_projection(traversal)
        return await self._execute_traversal(traversal)

    @staticmethod
    def _with_projection(traversal: GraphTraversal) -> GraphTraversal:
        return (
            traversal.project("node", "path", "node_type")
            .by(
                __.value_map(
                    "id", "name", "description", "importance", "text_unit_ids", "size"
                )
            )
            .by(__.path().by(__.value_map("name")))
            .by(__.label())
        )

    @staticmethod
    async def _execute_traversal(traversal: Traversal) -> list[Any]:
        try:
            result: list[Any] = traversal.to_list()
            return result
        except Exception as e:
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
        results, seen_ids = [], set()

        for item in traversal_results:
            node_data = self._clean_property_map(item.get("node", {}))
            node_id = node_data.get("id")
            if not node_id or node_id in seen_ids:
                continue

            result = self._create_retrieval_result(item, node_data, query)
            results.append(result)
            seen_ids.add(node_id)

        results.sort(key=lambda x: x.score or 0.0, reverse=True)
        return results

    def _create_retrieval_result(
        self, item: dict[str, Any], node_data: dict[str, Any], query: SearchQuery
    ) -> RetrievalResult:
        node_id = str(node_data.get("id"))
        node_type_str = item.get("node_type", "unknown")
        is_community = self._neptune_config.community_label_prefix in node_type_str

        section_type = SectionType.COMMUNITY if is_community else SectionType.ENTITY

        path_data = item.get("path", [])
        content = self._build_content(node_data, path_data, is_community)
        score = self._calculate_relevance(node_data, path_data, query, is_community)

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
        query: SearchQuery,
        is_community: bool,
    ) -> float:
        score_weights = {"importance": 0.4, "path": 0.3, "text": 0.2, "type": 0.1}
        path_len_penalty = 1.0 / (len(path) or 1)
        query_lower = query.query.lower()

        if is_community:
            importance = min(node.get("size", 1) / 100.0, 1.0)
            type_boost = 0.1
            text_match_score = (
                0.2 if query_lower in node.get("name", "").lower() else 0.0
            )
        else:
            importance = float(node.get("importance", 0.5))
            type_boost = 0.0
            text_match_score = 0.0
            if query_lower in node.get("name", "").lower():
                text_match_score += 0.2
            if query_lower in node.get("description", "").lower():
                text_match_score += 0.1

        score = (
            (importance * score_weights["importance"])
            + (path_len_penalty * score_weights["path"])
            + (text_match_score * score_weights["text"])
            + (type_boost * score_weights["type"])
        )
        return float(min(score, 1.0))

    def _record_metrics(
        self, result_count: int, entity_count: int, community_count: int
    ) -> None:
        self._record_metric("retrieved_count", result_count)
        self._record_metric("seed_entity_count", entity_count)
        self._record_metric("seed_community_count", community_count)
