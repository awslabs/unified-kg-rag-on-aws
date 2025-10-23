import re
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

from aws_graphrag.core import get_logger
from aws_graphrag.models import Config, Entity, Relationship, TextUnit
from aws_graphrag.utils import generate_stable_id, normalize_name

T = TypeVar("T")
logger = get_logger(__name__)


class BaseProcessor:
    def __init__(self, config: Config, show_progress: bool = True) -> None:
        self.config = config
        self.extraction_config = self.config.processing.graph_extraction
        self.show_progress = show_progress

    def get_text_for_processing(self, text_unit: TextUnit) -> str:
        if text_unit.translated_texts:
            target_language = self.config.processing.translation.target_language.value
            return text_unit.translated_texts.get(target_language, text_unit.text)
        return text_unit.text

    def parse_entity_data(
        self, entity_data: dict[str, Any], text_unit: TextUnit
    ) -> Entity | None:
        try:
            name = entity_data.get("name", "").strip()
            if not name:
                logger.warning(
                    f"Skipping entity with missing name in text unit '{text_unit.short_id}'"
                )
                return None

            normalized_name = normalize_name(name)
            if not normalized_name:
                logger.warning(
                    f"Skipping entity with empty name after normalization (from '{name}') in text unit '{text_unit.short_id}'"
                )
                return None

            entity_id = self._generate_entity_id(normalized_name)
            attributes = self._parse_attributes(
                entity_data.get("attributes"), text_unit
            )

            logger.debug(
                f"Successfully parsed entity: '{normalized_name}' (from: '{name}', id: {entity_id[:8]})"
            )
            return Entity(
                id=entity_id,
                short_id=entity_id[:8],
                name=normalized_name,
                name_embedding=None,
                type=entity_data.get("type", "").strip(),
                description=entity_data.get("description", "").strip(),
                description_embedding=None,
                text_unit_ids=[text_unit.id],
                community_ids=None,
                rank=entity_data.get("rank", 1),
                attributes=attributes,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        except Exception as e:
            logger.warning(
                f"Failed to parse entity '{entity_data.get('name', 'unknown')}' in text unit '{text_unit.short_id}': {e}"
            )
            return None

    @staticmethod
    def _generate_entity_id(name: str) -> str:
        entity_id_content = f"entity:{name}".lower()
        return generate_stable_id(entity_id_content)

    @staticmethod
    def _parse_attributes(
        attributes: list[dict[str, Any]] | None,
        text_unit: TextUnit | None = None,
    ) -> dict[str, Any]:
        parsed = {}
        if isinstance(attributes, list):
            parsed.update(
                {
                    d["key"]: d["value"]
                    for d in attributes
                    if isinstance(d, dict) and "key" in d and "value" in d
                }
            )

        if text_unit and text_unit.attributes:
            for key in ["index", "filters"]:
                if key in text_unit.attributes:
                    parsed[key] = text_unit.attributes[key]
        return parsed

    def parse_relationship_data(
        self,
        rel_data: dict[str, Any],
        text_unit: TextUnit,
        entity_name_to_id: dict[str, str],
    ) -> Relationship | None:
        raw_source_name, raw_target_name = "", ""
        try:
            raw_source_name = rel_data.get("source", "").strip()
            raw_target_name = rel_data.get("target", "").strip()
            rel_type = rel_data.get("type", "").strip()

            source_name = normalize_name(raw_source_name)
            target_name = normalize_name(raw_target_name)

            if not all((source_name, target_name, rel_type)):
                logger.warning(
                    f"Skipping relationship with missing data in text unit '{text_unit.short_id}': "
                    f"source='{source_name}' (from '{raw_source_name}'), "
                    f"target='{target_name}' (from '{raw_target_name}'), type='{rel_type}'"
                )
                return None

            source_id = entity_name_to_id.get(source_name)
            target_id = entity_name_to_id.get(target_name)

            if not (source_id and target_id):
                logger.warning(
                    f"Entity not found for relationship '{source_name}' -> '{target_name}' in text unit '{text_unit.short_id}'"
                )
                return None

            rel_id = self._generate_relationship_id(source_name, target_name, rel_type)
            attributes = self._parse_attributes(rel_data.get("attributes"), text_unit)
            weight = self._parse_weight(rel_data)

            logger.debug(
                f"Successfully parsed relationship: '{source_name}' -> '{target_name}' "
                f"(type: '{rel_type}', id: '{rel_id[:8]}')"
            )
            return Relationship(
                id=rel_id,
                short_id=rel_id[:8],
                source_id=source_id,
                source_name=source_name,
                target_id=target_id,
                target_name=target_name,
                type=rel_type,
                weight=weight,
                description=rel_data.get("description", "").strip(),
                description_embedding=None,
                rank=rel_data.get("rank", 1),
                text_unit_ids=[text_unit.id],
                attributes=attributes,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        except Exception as e:
            logger.warning(
                f"Failed to parse relationship '{raw_source_name}' -> '{raw_target_name}' in text unit '{text_unit.short_id}': {e}"
            )
            return None

    @staticmethod
    def _generate_relationship_id(
        source_name: str, target_name: str, rel_type: str
    ) -> str:
        rel_id_content = f"relationship:{source_name}:{target_name}:{rel_type}".lower()
        return generate_stable_id(rel_id_content)

    @staticmethod
    def _parse_weight(data: dict[str, Any], default: float = 1.0) -> float:
        value = data.get("strength", data.get("weight"))
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _merge_description(current: str | None, new: str | None) -> str | None:
        if current is None or new is None:
            return current or new

        if current and current.strip():
            return f"{current}; {new}"
        return new

    @staticmethod
    def _merge_items(
        items: list[T],
        item_name: str,
        field_mergers: dict[str, Callable[[Any, Any], Any]],
        frequency_fields: list[str] | None = None,
        log_message_formatter: Callable[[T], str] | None = None,
    ) -> list[T]:
        if not items:
            return []

        if frequency_fields is None:
            frequency_fields = []

        merged_items_map = {}
        merge_counts: dict[str, int] = defaultdict(int)
        frequency_counts: dict[str, dict[str, dict[str, int]]] = {
            field: defaultdict(lambda: defaultdict(int)) for field in frequency_fields
        }

        for item in items:
            item_id = getattr(item, "id", None)
            if item_id is None:
                continue
            merge_counts[item_id] += 1

            if item_id not in merged_items_map:
                merged_items_map[item_id] = item
            else:
                existing_item = merged_items_map[item_id]
                for field, merger in field_mergers.items():
                    new_value = getattr(item, field, None)
                    if new_value is not None:
                        current_value = getattr(existing_item, field, None)
                        setattr(existing_item, field, merger(current_value, new_value))

            for field in frequency_fields:
                value = getattr(item, field, "").lower()
                frequency_counts[field][item_id][value] += 1

        for item_id, item in merged_items_map.items():
            for field in frequency_fields:
                counts = frequency_counts.get(field, {}).get(item_id, {})
                if isinstance(counts, dict) and counts:
                    most_frequent_value = max(counts, key=lambda k: counts[k])
                    setattr(item, field, most_frequent_value)

        total_merges = 0
        for item_id, count in merge_counts.items():
            if count > 1:
                total_merges += count - 1
                if log_message_formatter:
                    item = merged_items_map[item_id]
                    logger.debug(log_message_formatter(item).format(count=count))

        if total_merges > 0:
            logger.info(
                f"Merged {total_merges} duplicate {item_name.lower()}(s) from {len(items)} to {len(merged_items_map)}"
            )

        return list(merged_items_map.values())


def check_entity_relevance_task(
    entity: Entity, unit_text: str, unit_tokens: set[str], threshold: float
) -> tuple[Entity, bool]:
    entity_tokens = set(re.findall(r"\b\w+\b", entity.name.lower()))
    if entity_tokens & unit_tokens:
        return entity, True
    similarity = calculate_similarity_task(entity.name.lower(), unit_text)
    return entity, similarity > threshold


def calculate_similarity_task(text1: str, text2: str) -> float:
    tokens1 = set(re.findall(r"\b\w+\b", text1))
    tokens2 = set(re.findall(r"\b\w+\b", text2))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def check_relationship_relevance_task(
    rel: Relationship, relevant_entity_names: set[str]
) -> tuple[Relationship, bool]:
    is_relevant = (
        rel.source_name is not None and rel.source_name.lower() in relevant_entity_names
    ) or (
        rel.target_name is not None and rel.target_name.lower() in relevant_entity_names
    )
    return rel, is_relevant
