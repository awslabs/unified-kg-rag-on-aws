import json
import time
from collections.abc import Callable, Iterator
from typing import Any, cast

from gremlin_python.process.graph_traversal import (
    GraphTraversal,
    GraphTraversalSource,
    __,
)
from gremlin_python.process.traversal import P

from aws_graphrag.aws import NeptuneClient
from aws_graphrag.core import get_logger
from aws_graphrag.models import Community, Config, Constants, Entity, Relationship
from aws_graphrag.storage import GraphIndexer, IndexingStats

logger = get_logger(__name__)


class NeptuneIndexer(GraphIndexer):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.neptune_config = self.config.indexing.neptune
        self.neptune_client = NeptuneClient(config=self.config)

    def clear(self, suffixes: list[str]) -> bool:
        if not suffixes:
            return True

        entity_prefix = self.neptune_config.entity_label_prefix.capitalize()
        community_prefix = self.neptune_config.community_label_prefix.capitalize()

        labels_to_delete = {self._get_name(entity_prefix, s) for s in suffixes} | {
            self._get_name(community_prefix, s) for s in suffixes
        }

        try:
            if labels_to_delete:
                logger.info(f"Clearing Neptune data for labels: {labels_to_delete}")
                for label in labels_to_delete:
                    self.neptune_client.delete_vertices_in_batches(label)
            return True
        except Exception as e:
            logger.error(f"Failed to clear Neptune data for suffixes '{suffixes}': {e}")
            return False

    def get_entity_count(self, suffixes: list[str]) -> int:
        if not suffixes:
            return 0

        entity_prefix = self.neptune_config.entity_label_prefix.capitalize()
        entity_labels = [self._get_name(entity_prefix, s) for s in suffixes]

        try:
            g = self.neptune_client.g
            result = g.V().hasLabel(*entity_labels).count().next()
            return int(result) if isinstance(result, (int | float)) else 0
        except Exception as e:
            logger.error(f"Failed to get entity count for '{entity_labels}': {e}")
            return 0

    def get_stats(self) -> dict[str, Any]:
        stats = self.neptune_client.get_graph_stats()
        if not isinstance(stats, dict):
            return {}
        return stats

    def initialize(self) -> bool:
        return True

    def index_entities(self, entities: list[Entity]) -> IndexingStats:
        def get_traversal_builder(label: str) -> Callable:
            def builder(g: GraphTraversalSource, batch: list[Entity]) -> GraphTraversal:
                t = g
                for entity in batch:
                    props = self._build_vertex_properties(
                        entity,
                        {
                            "name": entity.name,
                            "type": entity.type,
                            "description": entity.description,
                            "rank": entity.rank,
                            "text_unit_ids": entity.text_unit_ids,
                            "community_ids": entity.community_ids,
                        },
                    )
                    v_traversal = t.add_v(label).property("id", entity.id)
                    self._add_properties_to_traversal(v_traversal, props)
                    t = v_traversal
                return cast(GraphTraversal, t)

            return builder

        return self._index_generic(
            entities,
            "Entity",
            self.neptune_config.entity_label_prefix.capitalize(),
            self.neptune_config.entity_label_prefix.capitalize(),
            get_traversal_builder,
        )

    def index_relationships(self, relationships: list[Relationship]) -> IndexingStats:
        def get_traversal_builder(label: str, entity_label: str) -> Callable:
            def builder(
                g: GraphTraversalSource, batch: list[Relationship]
            ) -> GraphTraversal:
                t = g
                for rel in batch:
                    props = self._build_vertex_properties(
                        rel,
                        {
                            "source_name": rel.source_name,
                            "target_name": rel.target_name,
                            "weight": rel.weight,
                            "description": rel.description,
                            "rank": rel.rank,
                            "text_unit_ids": rel.text_unit_ids,
                        },
                    )
                    e_traversal = (
                        t.V()
                        .hasLabel(entity_label)
                        .has("id", rel.source_id)
                        .addE(rel.type or "RELATED_TO")
                        .to(__.V().hasLabel(entity_label).has("id", rel.target_id))
                        .property("id", rel.id)
                    )
                    self._add_properties_to_traversal(e_traversal, props)
                    t = e_traversal
                return cast(GraphTraversal, t)

            return builder

        return self._index_generic(
            relationships,
            "Relationship",
            self.neptune_config.entity_label_prefix.capitalize(),
            "",
            get_traversal_builder,
            entity_label=self.neptune_config.entity_label_prefix.capitalize(),
        )

    def index_communities(self, communities: list[Community]) -> IndexingStats:
        if not communities:
            return IndexingStats()

        total_stats = IndexingStats()
        grouped_items = self._group_items_by_suffix(communities)

        for suffix, comms in grouped_items.items():
            stats = IndexingStats(total_items=len(comms))
            start_time = time.time()
            community_label = self._get_name(
                self.neptune_config.community_label_prefix.capitalize(), suffix
            )
            entity_label = self._get_name(
                self.neptune_config.entity_label_prefix.capitalize(), suffix
            )

            self._clear_existing_data_by_label(community_label)

            try:
                logger.info(
                    f"Indexing {len(comms)} communities for '{community_label}'..."
                )
                for batch in self._batch_iterator(comms):
                    t = self.neptune_client.g
                    for comm in batch:
                        props = self._build_vertex_properties(
                            comm,
                            {
                                "name": comm.name,
                                "level": comm.level,
                                "parent": comm.parent,
                                "size": comm.size,
                                "period": comm.period,
                                "children": comm.children,
                            },
                        )
                        v_traversal = t.add_v(community_label).property("id", comm.id)
                        self._add_properties_to_traversal(v_traversal, props)
                        t = v_traversal
                    self._execute_with_retries(
                        cast(GraphTraversal, t), "Community vertex indexing"
                    )

                logger.info(
                    f"Indexing 'MemberOf' edges for {len(comms)} communities..."
                )
                for comm in comms:
                    if not comm.entity_ids:
                        continue
                    for entity_id_batch in self._batch_iterator(comm.entity_ids):
                        edge_traversal = (
                            self.neptune_client.g.V()
                            .hasLabel(entity_label)
                            .has("id", P.within(entity_id_batch))
                            .addE("MemberOf")
                            .to(__.V().hasLabel(community_label).has("id", comm.id))
                        )
                        self._execute_with_retries(
                            edge_traversal, "Community edge indexing"
                        )
                stats.add_success(len(comms))
            except Exception as e:
                logger.error(
                    f"Failed to index communities for '{community_label}': {e}"
                )
                stats.add_error(str(e), len(comms))

            stats.processing_time = time.time() - start_time
            total_stats.merge(stats)

        self._log_indexing_summary("communities", total_stats)
        return total_stats

    def _add_properties_to_traversal(
        self, traversal: GraphTraversal, props: dict[str, Any]
    ) -> None:
        for key, value in props.items():
            if isinstance(value, list):
                for item in value:
                    traversal.property(key, item)
            else:
                traversal.property(key, value)

    def _build_vertex_properties(
        self, item: Any, base_props: dict[str, Any]
    ) -> dict[str, Any]:
        properties = {
            k: self._safe_property_value(v)
            for k, v in base_props.items()
            if v is not None
        }
        if hasattr(item, "attributes") and item.attributes:
            for key, value in item.attributes.items():
                if value is not None:
                    properties[f"{Constants.ATTRIBUTE_PREFIX.value}_{key}"] = (
                        self._safe_property_value(value)
                    )
        return properties

    def _safe_property_value(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            return self._truncate(json.dumps(value))
        return self._truncate(value)

    def _truncate(self, value: Any) -> Any:
        max_len = self.neptune_config.property_max_length
        return (
            value[:max_len]
            if isinstance(value, str) and len(value) > max_len
            else value
        )

    def _index_generic(
        self,
        items: list[Any],
        item_type_name: str,
        label_prefix: str,
        clear_label_prefix: str,
        traversal_builder_func: Callable,
        **kwargs: Any,
    ) -> IndexingStats:
        if not items:
            return IndexingStats()

        grouped_items = self._group_items_by_suffix(items)
        total_stats = IndexingStats(total_items=len(items))

        for suffix, chunk in grouped_items.items():
            label = self._get_name(label_prefix, suffix)
            if clear_label_prefix:
                clear_label = self._get_name(clear_label_prefix, suffix)
                self._clear_existing_data_by_label(clear_label)

            start_time = time.time()
            stats = IndexingStats(total_items=len(chunk))
            try:
                logger.info(
                    f"Indexing {len(chunk)} {item_type_name.lower()}s for '{label}'..."
                )
                final_kwargs = kwargs.copy()
                if "entity_label" in final_kwargs:
                    final_kwargs["entity_label"] = self._get_name(
                        final_kwargs["entity_label"], suffix
                    )

                traversal_builder = traversal_builder_func(label=label, **final_kwargs)
                self._execute_batch_traversal(
                    chunk, traversal_builder, f"{item_type_name} indexing"
                )
                stats.add_success(len(chunk))
            except Exception as e:
                logger.error(
                    f"Failed to index {item_type_name.lower()}s for '{label}': {e}"
                )
                stats.add_error(str(e), len(chunk))

            stats.processing_time = time.time() - start_time
            total_stats.merge(stats)

        self._log_indexing_summary(f"{item_type_name.lower()}s", total_stats)
        return total_stats

    def _clear_existing_data_by_label(self, label: str) -> None:
        try:
            count_result = (
                self.neptune_client.g.V().hasLabel(label).limit(1).count().next()
            )
            count = count_result[0] if isinstance(count_result, list) else count_result
            if count > 0:
                self.neptune_client.delete_vertices_in_batches(label)
        except Exception as e:
            logger.error(f"Failed to clear data for label '{label}': {e}")
            raise

    def _execute_batch_traversal(
        self,
        items: list[Any],
        traversal_builder: Callable[[GraphTraversalSource, list[Any]], GraphTraversal],
        operation_name: str,
    ) -> None:
        if not items:
            return
        for batch in self._batch_iterator(items):
            g = self.neptune_client.g
            traversal = traversal_builder(g, batch)
            self._execute_with_retries(traversal, operation_name)

    def _execute_with_retries(
        self, traversal: GraphTraversal, operation_name: str
    ) -> None:
        max_retries = self.neptune_config.max_retries
        delay = self.neptune_config.retry_delay_seconds
        for attempt in range(max_retries + 1):
            try:
                traversal.iterate()
                return
            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        f"Failed {operation_name} after {attempt + 1} attempts: {e}"
                    )
                    raise
                logger.warning(
                    f"{operation_name} attempt {attempt + 1} failed, retrying in {delay}s: {e}"
                )
                time.sleep(delay)

    def _batch_iterator(self, items: list[Any]) -> Iterator[list[Any]]:
        batch_size = self.neptune_config.batch_size
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    def _log_indexing_summary(self, item_type_name: str, stats: IndexingStats) -> None:
        if stats.failed_items > 0:
            logger.warning(
                f"Indexed {stats.successful_items}/{stats.total_items} {item_type_name} "
                f"({stats.failed_items} failed) in {stats.processing_time:.2f}s."
            )
        else:
            logger.info(
                f"Successfully indexed {stats.successful_items} {item_type_name} "
                f"in {stats.processing_time:.2f}s."
            )
