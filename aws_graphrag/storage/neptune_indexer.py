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

        labels_to_delete = [self._get_name(entity_prefix, s) for s in suffixes] + [
            self._get_name(community_prefix, s) for s in suffixes
        ]

        try:
            if labels_to_delete:
                g = self.neptune_client.g
                g.V().hasLabel(*labels_to_delete).drop().iterate()
                logger.info(f"Cleared Neptune data for '{labels_to_delete}'")

            return True
        except Exception as e:
            logger.error(f"Failed to clear Neptune data for '{labels_to_delete}': {e}")
            return False

    def get_entity_count(self, suffixes: list[str]) -> int:
        if not suffixes:
            return 0

        entity_prefix = self.neptune_config.entity_label_prefix.capitalize()
        entity_labels = [self._get_name(entity_prefix, s) for s in suffixes]

        try:
            g = self.neptune_client.g
            result = g.V().hasLabel(*entity_labels).count().next()
            return (
                int(result)
                if isinstance(result, (int | float)) and result is not None
                else 0
            )
        except Exception as e:
            logger.error(f"Failed to get entity count for '{entity_labels}': {e}")
            return 0

    def get_stats(self) -> dict[str, Any]:
        return self.neptune_client.get_graph_stats()

    def initialize(self) -> bool:
        return True

    def index_entities(self, entities: list[Entity]) -> IndexingStats:
        def get_traversal_builder(label: str) -> Callable:
            def builder(g: GraphTraversalSource, batch: list[Entity]) -> GraphTraversal:
                traversal = g
                for entity in batch:
                    props = self._build_vertex_properties(
                        entity,
                        {
                            "name": entity.name or "",
                            "type": entity.type or "",
                            "description": entity.description or "",
                            "rank": entity.rank or 1,
                            "text_unit_ids": entity.text_unit_ids or [],
                            "community_ids": entity.community_ids or [],
                        },
                    )
                    v_traversal = traversal.add_v(label).property("id", entity.id)
                    for key, value in props.items():
                        if isinstance(value, list):
                            for item in value:
                                v_traversal = v_traversal.property(key, item)
                        else:
                            v_traversal = v_traversal.property(key, value)
                    traversal = v_traversal
                return cast(GraphTraversal, traversal)

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
                traversal = g
                for rel in batch:
                    props = self._build_vertex_properties(
                        rel,
                        {
                            "source_name": rel.source_name or "",
                            "target_name": rel.target_name or "",
                            "weight": rel.weight or 1.0,
                            "description": rel.description or "",
                            "rank": rel.rank or 1,
                            "text_unit_ids": rel.text_unit_ids or [],
                        },
                    )
                    edge_traversal = (
                        traversal.V()
                        .hasLabel(entity_label)
                        .has("id", rel.source_id)
                        .addE(rel.type or "RELATED_TO")
                        .to(__.V().hasLabel(entity_label).has("id", rel.target_id))
                        .property("id", rel.id)
                    )
                    for key, value in props.items():
                        if isinstance(value, list):
                            for item in value:
                                edge_traversal = edge_traversal.property(key, item)
                        else:
                            edge_traversal = edge_traversal.property(key, value)
                    traversal = edge_traversal
                return cast(GraphTraversal, traversal)

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
        edge_batch_size = self.neptune_config.batch_size

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
                    traversal = self.neptune_client.g
                    for comm in batch:
                        props = self._build_vertex_properties(
                            comm,
                            {
                                "name": comm.name or "",
                                "level": comm.level or 0,
                                "parent": comm.parent or "",
                                "size": comm.size or 0,
                                "period": comm.period or "",
                                "children": comm.children or [],
                            },
                        )
                        v_traversal = traversal.add_v(community_label).property(
                            "id", comm.id
                        )
                        for key, value in props.items():
                            if isinstance(value, list):
                                for item in value:
                                    v_traversal = v_traversal.property(key, item)
                            else:
                                v_traversal = v_traversal.property(key, value)
                        traversal = v_traversal
                    self._execute_with_retries(
                        cast(GraphTraversal, traversal), "Community vertex indexing"
                    )

                logger.info(
                    f"Indexing 'MemberOf' edges for {len(comms)} communities..."
                )
                for comm in comms:
                    if not comm.entity_ids:
                        continue

                    for i in range(0, len(comm.entity_ids), edge_batch_size):
                        entity_id_batch = comm.entity_ids[i : i + edge_batch_size]
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

        if total_stats.failed_items > 0:
            logger.warning(
                f"Indexed {total_stats.successful_items}/{total_stats.total_items} "
                f"communities ({total_stats.failed_items} failed)"
            )
        else:
            logger.info(
                f"Successfully indexed {total_stats.successful_items} communities"
            )

        return total_stats

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
        **kwargs,
    ) -> IndexingStats:
        if not items:
            return IndexingStats()

        grouped_items = self._group_items_by_suffix(items)
        total_stats = IndexingStats()

        for suffix, chunk in grouped_items.items():
            label = self._get_name(label_prefix, suffix)
            if clear_label_prefix:
                self._clear_existing_data_by_label(
                    self._get_name(clear_label_prefix, suffix)
                )

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

        if total_stats.failed_items > 0:
            logger.warning(
                f"Indexed {total_stats.successful_items}/{len(items)} {item_type_name.lower()}s "
                f"({total_stats.failed_items} failed)"
            )
        else:
            logger.info(
                f"Successfully indexed {total_stats.successful_items} {item_type_name.lower()}s"
            )

        return total_stats

    def _clear_existing_data_by_label(self, label: str) -> None:
        try:
            g = self.neptune_client.g
            if g.V().hasLabel(label).limit(1).count().next() == 0:
                return
            g.V().hasLabel(label).drop().iterate()
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
