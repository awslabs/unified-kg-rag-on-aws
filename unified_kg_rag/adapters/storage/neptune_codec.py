# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Property encoding shared by the Neptune write path and its read-back.

Cross-run merge and shared-artifact pruning read entities and relationships
back from Neptune, merge them and write them again, so the read must invert
exactly what the write produced. Both directions live here:

- vertex lists are multi-valued (set-cardinality) properties;
- edges cannot hold multi-valued properties, so edge lists are JSON strings;
- dict values are JSON strings on vertices and edges;
- string values are truncated to ``property_max_length``; JSON strings are not,
  because a cut JSON string no longer parses;
- item attributes are ``attr_<key>`` properties;
- a relationship's type is the edge label.

Read-back limits: a set-cardinality vertex list keeps no order or duplicates,
and a one-element list attribute reads back as its single value (Gremlin
cannot tell them apart). A string attribute whose text is a JSON object or
array reads back as the decoded value.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from unified_kg_rag.domain.models import Constants, Entity, Relationship
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

ATTRIBUTE_KEY_PREFIX = f"{Constants.ATTRIBUTE_PREFIX.value}_"


def property_value(value: Any, max_length: int) -> Any:
    """Encode one property value (see the module docstring)."""
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, str) and len(value) > max_length:
        return value[:max_length]
    return value


def item_properties(
    item: Any, base_props: dict[str, Any], max_length: int
) -> dict[str, Any]:
    """``base_props`` plus the item's ``attr_*`` attributes, encoded, no Nones."""
    properties = {
        key: property_value(value, max_length)
        for key, value in base_props.items()
        if value is not None
    }
    for key, value in (getattr(item, "attributes", None) or {}).items():
        if value is not None:
            properties[f"{ATTRIBUTE_KEY_PREFIX}{key}"] = property_value(
                value, max_length
            )
    return properties


def entity_properties(entity: Entity, max_length: int) -> dict[str, Any]:
    """Vertex properties of an entity (lists stay lists: multi-valued)."""
    return item_properties(
        entity,
        {
            "name": entity.name,
            "type": entity.type,
            "description": entity.description,
            "rank": entity.rank,
            "confidence": entity.confidence,
            "text_unit_ids": entity.text_unit_ids,
            "community_ids": entity.community_ids,
        },
        max_length,
    )


def relationship_properties(rel: Relationship, max_length: int) -> dict[str, Any]:
    """Edge properties of a relationship (lists become JSON strings)."""
    properties = item_properties(
        rel,
        {
            "source_name": rel.source_name,
            "target_name": rel.target_name,
            "weight": rel.weight,
            "description": rel.description,
            "rank": rel.rank,
            "text_unit_ids": rel.text_unit_ids,
        },
        max_length,
    )
    return {
        key: json.dumps(value) if isinstance(value, list) else value
        for key, value in properties.items()
    }


def relationship_label(rel: Relationship) -> str:
    """The edge label a relationship is written under."""
    return rel.type or Constants.DEFAULT_RELATIONSHIP_TYPE.value


def _values(value: Any) -> list[Any]:
    """Gremlin returns vertex properties as lists and edge properties bare."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _single(value: Any) -> Any:
    values = _values(value)
    return values[0] if values else None


def _vertex_list(value: Any) -> list[str] | None:
    values = _values(value)
    return [str(v) for v in values] if values else None


def _edge_list(key: str, value: Any) -> list[str] | None:
    if value is None:
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        logger.warning("Edge property '%s' is not a JSON list; ignoring it", key)
        return None
    if not isinstance(decoded, list):
        logger.warning("Edge property '%s' is not a JSON list; ignoring it", key)
        return None
    return [str(v) for v in decoded]


def _attribute_value(value: Any) -> Any:
    values = _values(value)
    single = values[0] if len(values) == 1 else values
    if isinstance(single, str) and single[:1] in ("{", "["):
        try:
            decoded = json.loads(single)
        except ValueError:
            return single
        if isinstance(decoded, (dict, list)):
            return decoded
    return single


def _attributes(value_map: Mapping[Any, Any]) -> dict[str, Any] | None:
    attributes = {
        key[len(ATTRIBUTE_KEY_PREFIX) :]: _attribute_value(value)
        for key, value in value_map.items()
        if isinstance(key, str) and key.startswith(ATTRIBUTE_KEY_PREFIX)
    }
    return attributes or None


def entity_from_vertex(value_map: Mapping[Any, Any]) -> Entity | None:
    """Rebuild an entity from a vertex ``valueMap()``; None without id/name."""
    entity_id, name = _single(value_map.get("id")), _single(value_map.get("name"))
    if entity_id is None or name is None:
        return None
    fields: dict[str, Any] = {
        "id": str(entity_id),
        "name": str(name),
        "type": _single(value_map.get("type")),
        "description": _single(value_map.get("description")),
        "text_unit_ids": _vertex_list(value_map.get("text_unit_ids")),
        "community_ids": _vertex_list(value_map.get("community_ids")),
        "attributes": _attributes(value_map),
    }
    # Absent rank/confidence keep the model defaults, as on the write side.
    for key in ("rank", "confidence"):
        value = _single(value_map.get(key))
        if value is not None:
            fields[key] = value
    return Entity.model_validate(fields)


def relationship_from_edge(
    value_map: Mapping[Any, Any], label: str, source_id: Any, target_id: Any
) -> Relationship | None:
    """Rebuild a relationship from an edge ``valueMap()``, label and endpoints."""
    rel_id = _single(value_map.get("id"))
    if rel_id is None or source_id is None or target_id is None:
        return None
    fields: dict[str, Any] = {
        "id": str(rel_id),
        "source_id": str(source_id),
        "target_id": str(target_id),
        "type": label,
        "source_name": _single(value_map.get("source_name")),
        "target_name": _single(value_map.get("target_name")),
        "description": _single(value_map.get("description")),
        "text_unit_ids": _edge_list(
            "text_unit_ids", _single(value_map.get("text_unit_ids"))
        ),
        "attributes": _attributes(value_map),
    }
    for key in ("weight", "rank"):
        value = _single(value_map.get(key))
        if value is not None:
            fields[key] = value
    return Relationship.model_validate(fields)
