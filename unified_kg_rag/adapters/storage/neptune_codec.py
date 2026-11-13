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
- a relationship's type is the edge label;
- the schema annotations (``in_schema``, a boolean, and ``type_iri`` /
  ``predicate_iri``) are properties of their own, absent while ``None``;
- so are the seed-graph fields: an entity's ``iri`` (not truncated) and
  ``source_kinds``, a vertex list, or on an edge the JSON string of the
  sorted kinds (``["seed_graph"]`` exactly for an edge only a seed graph
  states).

Read-back limits: a set-cardinality vertex list keeps no order or duplicates,
and a one-element list attribute reads back as its single value (Gremlin
cannot tell them apart). A string attribute whose text is a JSON object or
array reads back as the decoded value.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Collection, Mapping
from typing import Any

from unified_kg_rag.domain.ingestion.names import compact_name_key
from unified_kg_rag.domain.ingestion.seed_graph import (
    SOURCE_KINDS_FIELD,
    encoded_source_kinds,
)
from unified_kg_rag.domain.models import Community, Constants, Entity, Relationship
from unified_kg_rag.domain.retrieval.filters import ATTRIBUTE_KEY_PREFIX
from unified_kg_rag.ports.indexer import (
    AnnotatedKind,
    GraphCensusRow,
    StoredAnnotation,
)
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)


def property_value(value: Any, max_length: int) -> Any:
    """Encode one property value (see the module docstring)."""
    if isinstance(value, dict):
        return json.dumps(value)
    if isinstance(value, str) and len(value) > max_length:
        return value[:max_length]
    return value


def item_properties(
    item: Any,
    base_props: dict[str, Any],
    max_length: int,
    *,
    whole: Collection[str] = (),
) -> dict[str, Any]:
    """``base_props`` plus the item's ``attr_*`` attributes, encoded, no Nones;
    the ``whole`` properties are not truncated."""
    properties = {
        key: value if key in whole else property_value(value, max_length)
        for key, value in base_props.items()
        if value is not None
    }
    for key, value in (getattr(item, "attributes", None) or {}).items():
        if value is not None:
            properties[f"{ATTRIBUTE_KEY_PREFIX}{key}"] = property_value(
                value, max_length
            )
    return properties


# The tag a full build replacing a suffix in place writes on every vertex and
# edge, so the build can then drop what it did not write
# (GraphIndexer.begin_rebuild). Not read back.
BUILD_TAG_PROPERTY = "build_tag"

# The compact name key of an entity's name, stored with it so the cross-run
# merge finds the stored entities a delta name links to without reading every
# name (GraphIndexer.find_entity_ids_by_name_key). Not read back.
NAME_KEY_PROPERTY = "name_key"


def _kinds(source_kinds: list[str] | None) -> list[str] | None:
    """``source_kinds`` as written: sorted, ``None`` when empty."""
    return sorted(source_kinds) if source_kinds else None


def entity_properties(entity: Entity, max_length: int) -> dict[str, Any]:
    """Vertex properties of an entity (lists stay lists: multi-valued)."""
    return item_properties(
        entity,
        {
            NAME_KEY_PROPERTY: compact_name_key(entity.name),
            "name": entity.name,
            "type": entity.type,
            "description": entity.description,
            "rank": entity.rank,
            "confidence": entity.confidence,
            "text_unit_ids": entity.text_unit_ids,
            "community_ids": entity.community_ids,
            "in_schema": entity.in_schema,
            # Not truncated: a cut IRI names another resource.
            "type_iri": entity.type_iri,
            "iri": entity.iri,
            SOURCE_KINDS_FIELD: _kinds(entity.source_kinds),
        },
        max_length,
        whole=("type_iri", "iri"),
    )


def community_properties(comm: Community, max_length: int) -> dict[str, Any]:
    """Vertex properties of a community (lists stay lists: multi-valued)."""
    return item_properties(
        comm,
        {
            "name": comm.name,
            "level": comm.level,
            "parent": comm.parent,
            "size": comm.size,
            "period": comm.period,
            "children": comm.children,
            # Found by them when its documents are removed.
            "text_unit_ids": comm.text_unit_ids,
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
            "in_schema": rel.in_schema,
            "predicate_iri": rel.predicate_iri,
            # The one edge encoding (seed_graph.SEED_ONLY_SOURCE_KINDS is the
            # value a seed-only edge holds, which the neighbour reads skip).
            SOURCE_KINDS_FIELD: (
                encoded_source_kinds(rel.source_kinds) if rel.source_kinds else None
            ),
        },
        max_length,
        whole=("predicate_iri", SOURCE_KINDS_FIELD),
    )
    return {
        key: json.dumps(value) if isinstance(value, list) else value
        for key, value in properties.items()
    }


def relationship_label(rel: Relationship) -> str:
    """The edge label a relationship is written under."""
    return rel.type or Constants.DEFAULT_RELATIONSHIP_TYPE.value


def property_values(value: Any) -> list[Any]:
    """A property as Gremlin's ``valueMap`` returns it, as a list: a
    vertex's properties come as lists, an edge's bare."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _single(value: Any) -> Any:
    values = property_values(value)
    return values[0] if values else None


def _vertex_list(value: Any) -> list[str] | None:
    values = property_values(value)
    return [str(v) for v in values] if values else None


def decode_edge_list(value: Any, key: str | None = None) -> list[str] | None:
    """An edge's list property as written (one JSON list string, the
    inverse of the write's ``json.dumps``), read bare or as Gremlin's
    one-element list; ``None`` when absent or unreadable, WARNed naming
    ``key`` when given (a read that skips a malformed value passes none)."""
    value = _single(value)
    if value is None:
        return None
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        decoded = None
    if not isinstance(decoded, list):
        if key is not None:
            logger.warning("Edge property '%s' is not a JSON list; ignoring it", key)
        return None
    return [str(v) for v in decoded]


def edge_text_unit_ids(value: Any) -> list[str]:
    """The text units of an edge's ``text_unit_ids`` property (a JSON list)."""
    return decode_edge_list(value, "text_unit_ids") or []


def _attribute_value(value: Any) -> Any:
    values = property_values(value)
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


def _annotations(value_map: Mapping[Any, Any], iri_key: str) -> dict[str, Any]:
    """The schema annotations a vertex or edge holds (``in_schema`` and
    ``iri_key``); a value of another type than written is left out."""
    fields: dict[str, Any] = {}
    in_schema = _single(value_map.get("in_schema"))
    if isinstance(in_schema, bool):
        fields["in_schema"] = in_schema
    iri = _single(value_map.get(iri_key))
    if isinstance(iri, str):
        fields[iri_key] = iri
    return fields


# The IRI property of each annotated kind.
IRI_PROPERTY: dict[AnnotatedKind, str] = {
    "entities": "type_iri",
    "relationships": "predicate_iri",
}


def stored_annotation(
    value_map: Mapping[Any, Any], kind: AnnotatedKind, item_type: Any
) -> StoredAnnotation:
    """The schema annotations a vertex or edge ``valueMap()`` holds, with
    the item's type (``read_schema_annotations``)."""
    iri_key = IRI_PROPERTY[kind]
    fields = _annotations(value_map, iri_key)
    return StoredAnnotation(
        type=None if item_type is None else str(item_type),
        in_schema=fields.get("in_schema"),
        iri=fields.get(iri_key),
    )


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
    fields.update(_annotations(value_map, "type_iri"))
    iri = _single(value_map.get("iri"))
    if isinstance(iri, str):
        fields["iri"] = iri
    kinds = _vertex_list(value_map.get(SOURCE_KINDS_FIELD))
    if kinds:
        fields[SOURCE_KINDS_FIELD] = sorted(kinds)
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
        "text_unit_ids": decode_edge_list(
            value_map.get("text_unit_ids"), "text_unit_ids"
        ),
        "attributes": _attributes(value_map),
    }
    for key in ("weight", "rank"):
        value = _single(value_map.get(key))
        if value is not None:
            fields[key] = value
    fields.update(_annotations(value_map, "predicate_iri"))
    kinds = decode_edge_list(value_map.get(SOURCE_KINDS_FIELD), SOURCE_KINDS_FIELD)
    if kinds:
        fields[SOURCE_KINDS_FIELD] = sorted(kinds)
    return Relationship.model_validate(fields)


# What an absent rank or weight reads as (the full reads keep the model's
# default): a census row decodes them alike.
_ENTITY_RANK = Entity.model_fields["rank"].default
_EDGE_RANK = Relationship.model_fields["rank"].default
_EDGE_WEIGHT = Relationship.model_fields["weight"].default
# The properties a census row reads (``GraphIndexer.scan_census_rows``).
CENSUS_PROPERTIES: tuple[str, ...] = (
    "id",
    "type",
    "rank",
    "description",
    "text_unit_ids",
    "weight",
)


def _or_default(value: Any, default: Any, cast: Callable[[Any], Any]) -> Any:
    value = _single(value)
    return default if value is None else cast(value)


def _text(value: Any) -> str | None:
    value = _single(value)
    return None if value is None else str(value)


def census_row_from_vertex(value_map: Mapping[Any, Any]) -> GraphCensusRow | None:
    """The census row of an entity vertex's ``valueMap()``; None without id."""
    entity_id = _single(value_map.get("id"))
    if entity_id is None:
        return None
    return GraphCensusRow(
        id=str(entity_id),
        type=_text(value_map.get("type")),
        rank=_or_default(value_map.get("rank"), _ENTITY_RANK, int),
        description=_text(value_map.get("description")),
        text_unit_ids=sorted(_vertex_list(value_map.get("text_unit_ids")) or ()),
    )


def census_row_from_edge(
    value_map: Mapping[Any, Any], label: Any, source_id: Any, target_id: Any
) -> GraphCensusRow | None:
    """The census row of a relationship edge's ``valueMap()``, label and
    endpoints; None without id."""
    rel_id = _single(value_map.get("id"))
    if rel_id is None:
        return None
    return GraphCensusRow(
        id=str(rel_id),
        type=None if label is None else str(label),
        rank=_or_default(value_map.get("rank"), _EDGE_RANK, int),
        description=_text(value_map.get("description")),
        text_unit_ids=sorted(edge_text_unit_ids(value_map.get("text_unit_ids"))),
        source_id=_text(source_id),
        target_id=_text(target_id),
        weight=_or_default(value_map.get("weight"), _EDGE_WEIGHT, float),
    )
