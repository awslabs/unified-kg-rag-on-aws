# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

from unified_kg_rag.domain.models import Config, Entity, Relationship, TextUnit
from unified_kg_rag.shared import get_logger
from unified_kg_rag.shared.utils import (
    clean_display_name,
    entity_key,
    generate_stable_id,
)

T = TypeVar("T")
logger = get_logger(__name__)


def coerce_llm_text(value: Any, *, join: bool = False) -> str:
    """Coerce an LLM-parsed field to one stripped string.

    The XML output parser returns a list when the model repeats a tag
    (``<type>A</type><type>B</type>``). Identity fields (name, type, source,
    target) take the first non-empty element; description-like fields pass
    ``join=True`` to keep every distinct non-empty element, newline-joined.
    ``None`` and other non-string scalars become ``""`` / ``str(value)``.
    """
    if value is None:
        return ""
    if not isinstance(value, list):
        return str(value).strip()
    parts = [p for p in (coerce_llm_text(v) for v in value) if p]
    if not parts:
        return ""
    return "\n".join(dict.fromkeys(parts)) if join else parts[0]


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
            name = clean_display_name(coerce_llm_text(entity_data.get("name")))
            if not name:
                logger.warning(
                    "Skipping entity with missing name in text unit '%s'",
                    text_unit.short_id,
                )
                return None

            # ``name`` keeps the display form ("$1,000 penalty", "C++") so
            # answers can quote it verbatim; the id is derived from the
            # identity key so case/whitespace variants still converge.
            entity_id = self._generate_entity_id(name)
            attributes = self._parse_attributes(
                entity_data.get("attributes"), text_unit
            )
            # Stash the verbatim evidence span (when the model emits one) so the
            # extractor can verify the entity is grounded in the source chunk.
            # Reserved key; not persisted as a public attribute downstream.
            source_text = entity_data.get("source_text")
            if isinstance(source_text, str) and source_text.strip():
                attributes["_source_text"] = source_text.strip()
            confidence = self._parse_confidence(entity_data)

            logger.debug(
                "Successfully parsed entity: '%s' (id: %s, confidence: %.2f)",
                name,
                entity_id[:8],
                confidence,
            )
            return Entity(
                id=entity_id,
                short_id=entity_id[:8],
                name=name,
                name_embedding=None,
                type=coerce_llm_text(entity_data.get("type")),
                description=coerce_llm_text(entity_data.get("description"), join=True),
                description_embedding=None,
                text_unit_ids=[text_unit.id],
                community_ids=None,
                rank=entity_data.get("rank", 1),
                frequency=None,
                confidence=confidence,
                attributes=attributes,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        except Exception as e:
            logger.warning(
                "Failed to parse entity in text unit '%s': %s", text_unit.short_id, e
            )
            logger.debug("Unparsed entity name: '%s'", entity_data.get("name"))
            return None

    @staticmethod
    def _generate_entity_id(name: str) -> str:
        """Stable entity id from the name's identity key (see ``entity_key``)."""
        return generate_stable_id(f"entity:{entity_key(name)}")

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
            raw_source_name = coerce_llm_text(rel_data.get("source"))
            raw_target_name = coerce_llm_text(rel_data.get("target"))
            rel_type = coerce_llm_text(rel_data.get("type"))

            source_name = clean_display_name(raw_source_name)
            target_name = clean_display_name(raw_target_name)

            if not all((source_name, target_name, rel_type)):
                logger.warning(
                    "Skipping relationship with missing source, target or type "
                    "in text unit '%s'",
                    text_unit.short_id,
                )
                logger.debug(
                    "Skipped relationship: source='%s', target='%s', type='%s'",
                    raw_source_name,
                    raw_target_name,
                    rel_type,
                )
                return None
            source_key = entity_key(source_name)
            target_key = entity_key(target_name)

            # ``entity_name_to_id`` is keyed by ``entity_key`` (callers build it
            # with :meth:`build_entity_key_index`). Entity ids are deterministic
            # from that key, so an endpoint id is recoverable even when the entity was not listed in
            # THIS chunk's entity block (the LLM commonly omits it, or it was
            # extracted in another chunk). Prefer the locally-extracted id, but
            # fall back to deriving it from the name rather than dropping a valid
            # relationship (the global merge + orphan filter still guard
            # integrity; missing endpoints are materialized as stub entities).
            source_id = entity_name_to_id.get(source_key) or self._generate_entity_id(
                source_name
            )
            target_id = entity_name_to_id.get(target_key) or self._generate_entity_id(
                target_name
            )

            rel_id = self._generate_relationship_id(source_key, target_key, rel_type)
            attributes = self._parse_attributes(rel_data.get("attributes"), text_unit)
            # Stash the verbatim evidence span (when emitted) for grounding; a
            # reserved key, stripped before persistence.
            source_text = rel_data.get("source_text")
            if isinstance(source_text, str) and source_text.strip():
                attributes["_source_text"] = source_text.strip()
            weight = self._parse_weight(rel_data)

            logger.debug(
                "Successfully parsed relationship: '%s' -> '%s' (type: '%s', id: '%s')",
                source_name,
                target_name,
                rel_type,
                rel_id[:8],
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
                description=coerce_llm_text(rel_data.get("description"), join=True),
                description_embedding=None,
                rank=rel_data.get("rank", 1),
                text_unit_ids=[text_unit.id],
                attributes=attributes,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
        except Exception as e:
            logger.warning(
                "Failed to parse relationship in text unit '%s': %s",
                text_unit.short_id,
                e,
            )
            logger.debug(
                "Unparsed relationship: '%s' -> '%s'", raw_source_name, raw_target_name
            )
            return None

    @staticmethod
    def build_entity_key_index(entities: list[Entity]) -> dict[str, str]:
        """Map each entity's identity key to its id, for endpoint resolution.

        Keyed by ``entity_key`` (not the display name) so a relationship
        endpoint spelled "ACME Corp." resolves to the entity named "Acme Corp",
        including one renamed in place by a gleaner correction (whose id is no
        longer derivable from its new name).
        """
        return {entity_key(entity.name): entity.id for entity in entities}

    @staticmethod
    def _generate_relationship_id(
        source_name: str, target_name: str, rel_type: str
    ) -> str:
        rel_id_content = (
            f"relationship:{entity_key(source_name)}:{entity_key(target_name)}:"
            f"{rel_type}".lower()
        )
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
    def _parse_confidence(data: dict[str, Any], default: float = 1.0) -> float:
        """Normalize an extracted confidence to ``[0, 1]``.

        The extraction prompt asks for an integer on a 1-10 scale, but models
        (or custom prompts) sometimes answer with a 0-1 fraction. Rule:

        * ``> 1``: 1-10 scale, divided by 10 (``8`` -> 0.8, ``8.5`` -> 0.85).
        * exactly ``1`` written as an integer (``1`` / ``"1"``): the lowest
          point of the 1-10 scale -> 0.1. Reading it as the fraction 1.0 would
          turn the least confident answer into the most confident one.
        * any other value in ``[0, 1]``, including ``1.0`` written with a
          decimal point: already a fraction, kept as-is.

        The result is clamped to ``[0, 1]``; unparseable values use ``default``.
        """
        value = data.get("confidence")
        if value is None:
            return default
        try:
            raw_confidence = float(value)
            written_as_integer = isinstance(value, int) or (
                isinstance(value, str) and value.strip().lstrip("+").isdigit()
            )
            if raw_confidence > 1.0 or (raw_confidence == 1.0 and written_as_integer):
                normalized = raw_confidence / 10.0
            else:
                normalized = raw_confidence
            return max(0.0, min(1.0, normalized))
        except (ValueError, TypeError):
            logger.warning("Invalid confidence value, using default %s", default)
            logger.debug("Invalid confidence value: '%s'", value)
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
                # `or ""` guards an explicit None field value (e.g. model-default
                # Relationship.type / Claim.status): getattr's "" default only
                # applies when the attribute is missing, not when it is None.
                value = (getattr(item, field, "") or "").lower()
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
                "Merged %s duplicate %s(s) from %s to %s",
                total_merges,
                item_name.lower(),
                len(items),
                len(merged_items_map),
            )

        return list(merged_items_map.values())


def check_entity_relevance_task(entity: Entity, unit_id: str) -> tuple[Entity, bool]:
    """An entity is relevant to a text unit iff it was extracted from it.

    Uses the authoritative ``text_unit_ids`` lineage recorded during extraction
    rather than re-deriving relevance from token overlap — which is exact,
    language-agnostic, and free of brittle substring/Jaccard heuristics.
    """
    return entity, unit_id in (entity.text_unit_ids or [])


def check_relationship_relevance_task(
    rel: Relationship, unit_id: str
) -> tuple[Relationship, bool]:
    """A relationship is relevant to a text unit iff it was extracted from it."""
    return rel, unit_id in (rel.text_unit_ids or [])
