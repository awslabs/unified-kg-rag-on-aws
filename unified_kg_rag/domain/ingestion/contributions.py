# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The graph as the merge of every text unit's extraction record.

Extraction and gleaning each write one :class:`TextUnitExtraction` per text
unit; nothing edits an item another unit stated. The graph that graph
resolution starts from is :func:`merge_contributions` over all records, so
the same records give the same graph whichever run extracted them.

A record is also stored, unit-independently (:func:`record_payload`), in the
extraction store under its text's hash, and re-stamped onto any text unit
with that text (:func:`record_from_payload`): an unchanged chunk is never
extracted twice.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from unified_kg_rag.domain.ingestion.base_processor import (
    BaseProcessor,
    union_all_document_filters,
)
from unified_kg_rag.domain.ingestion.descriptions import (
    merge_descriptions,
)
from unified_kg_rag.domain.ingestion.relationship_endpoints import (
    cite_relationship_endpoints,
)
from unified_kg_rag.domain.ingestion.relationship_weights import (
    TEXT_UNIT_WEIGHTS_ATTRIBUTE,
    apply_text_unit_weights,
    sum_weights,
    text_unit_weights,
)
from unified_kg_rag.domain.models import (
    Claim,
    Constants,
    Entity,
    Relationship,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.shared.utils import compute_hash

# Attributes a record's items take from their text unit (the index suffix and
# the document filters), not from what the unit says: stored records drop
# them and take the reusing unit's.
_PLACEMENT = (Constants.INDEX.value, Constants.FILTERS.value)


def _union_ids(values: list[Any]) -> list[str]:
    return list(
        dict.fromkeys(i for ids in values if isinstance(ids, list) for i in ids)
    )


def merge_entity_contributions(entities: list[Entity]) -> list[Entity]:
    """Merge entity statements by id (mutates the first of each id).

    Descriptions are unioned line by line, text units and document filters
    unioned, the confidence is the max (monotonic in evidence), and the type
    and the name (the ids' names differ in case only) the most frequent of
    the statements (the smallest on a tie, whatever the order of the
    statements).
    """
    field_mergers: dict[str, Callable[[list[Any]], Any]] = {
        "description": merge_descriptions,
        "text_unit_ids": _union_ids,
        "confidence": lambda values: max(
            value if value is not None else 0.0 for value in values
        ),
        "attributes": union_all_document_filters,
    }
    spellings: dict[str, Counter[str]] = defaultdict(Counter)
    for entity in entities:
        if entity.id is not None and entity.name:
            spellings[entity.id][entity.name] += 1
    merged = BaseProcessor._merge_items(
        items=entities,
        item_name="Entity",
        field_mergers=field_mergers,
        frequency_fields=["type"],
        describe_item=lambda e: f"Entity '{e.name}'",
    )
    for entity in merged:
        if names := spellings.get(entity.id or ""):
            entity.name = min(names, key=lambda n: (-names[n], n))
    return merged


def merge_relationship_contributions(
    relationships: list[Relationship],
) -> list[Relationship]:
    """Merge relationship statements by id (mutates the first of each id).

    The weight is the sum of the statements' strengths, kept per text unit
    (see ``relationship_weights``); descriptions and filters are unioned.
    """
    weight_maps: dict[str, list[dict[str, float]]] = defaultdict(list)
    for rel in relationships:
        weight_maps[rel.id].append(text_unit_weights(rel))
    field_mergers: dict[str, Callable[[list[Any]], Any]] = {
        "description": merge_descriptions,
        "attributes": union_all_document_filters,
    }
    merged = BaseProcessor._merge_items(
        items=relationships,
        item_name="Relationship",
        field_mergers=field_mergers,
        describe_item=lambda r: (
            f"Relationship '{r.source_name}' -> '{r.target_name}' (type: '{r.type}')"
        ),
    )
    for rel in merged:
        apply_text_unit_weights(rel, sum_weights(weight_maps[rel.id]))
    return merged


def merge_contributions(
    records: Iterable[TextUnitExtraction],
) -> tuple[list[Entity], list[Relationship]]:
    """The graph the records state: their items merged by id, endpoints cited.

    Inputs are not mutated. Every relationship endpoint becomes an entity
    citing the relationship's text units (a stub when no unit lists it, see
    ``cite_relationship_endpoints``). The records are merged in text-unit id
    order, so where a merge keeps the first statement the result does not
    depend on which records were extracted together: an incremental run
    that merges a subset of them makes the same choice.
    """
    records = sorted(records, key=lambda record: record.text_unit_id)
    entities = merge_entity_contributions(
        [e.model_copy(deep=True) for r in records for e in r.entities]
    )
    relationships = merge_relationship_contributions(
        [rel.model_copy(deep=True) for r in records for rel in r.relationships]
    )
    entities, _ = cite_relationship_endpoints(entities, relationships)
    return entities, relationships


def unit_entities(record: TextUnitExtraction) -> list[Entity]:
    """The entities citing the record's unit: its own and its endpoints'."""
    entities, _ = cite_relationship_endpoints(record.entities, record.relationships)
    return entities


# Text-unit attribute naming the extraction-store keys of the unit's records:
# {"graph": key, "claims": key}. Written with the text unit, so a later run
# finds the records of a stored unit by its id.
RECORD_KEYS_ATTRIBUTE = "extraction_records"


def record_keys(unit: TextUnit) -> dict[str, str]:
    """The unit's extraction-store keys by record kind ("graph", "claims")."""
    return attribute_record_keys(unit.attributes)


def attribute_record_keys(attributes: dict[str, Any] | None) -> dict[str, str]:
    """:func:`record_keys` of a text unit's stored ``attributes``."""
    keys = (attributes or {}).get(RECORD_KEYS_ATTRIBUTE)
    if not isinstance(keys, dict):
        return {}
    return {kind: key for kind, key in keys.items() if isinstance(key, str)}


def text_hash(text: str) -> str:
    """Key of a text unit's text in the extraction store (SHA-256)."""
    return compute_hash(text, length=64)


def record_key(fingerprint: str, text: str) -> str:
    """Extraction-store key of the record of ``text`` under ``fingerprint``."""
    return f"{fingerprint}:{text_hash(text)}"


def _portable(item: Entity | Relationship | Claim) -> dict[str, Any]:
    attributes = {
        key: value
        for key, value in (item.attributes or {}).items()
        if key not in _PLACEMENT and key != TEXT_UNIT_WEIGHTS_ATTRIBUTE
    }
    # The schema annotations are the indexing's, from the merged type: a
    # record holds what extraction stated.
    excluded = {
        "text_unit_ids",
        "attributes",
        "name_embedding",
        "description_embedding",
        "in_schema",
        "type_iri",
        "predicate_iri",
    }
    return item.model_dump(
        mode="json", exclude=excluded & set(type(item).model_fields)
    ) | {"attributes": attributes or None}


def record_payload(record: TextUnitExtraction) -> dict[str, Any]:
    """The record without what ties it to its text unit, for the store."""
    return {
        "entities": [_portable(e) for e in record.entities],
        "relationships": [_portable(r) for r in record.relationships],
        "rejected": dict(record.rejected),
        "gleaned": record.gleaned,
    }


def _placement(unit: TextUnit) -> dict[str, Any]:
    attributes = unit.attributes or {}
    return {key: attributes[key] for key in _PLACEMENT if key in attributes}


def _stamp(item: dict[str, Any], unit: TextUnit) -> dict[str, Any]:
    attributes = {**(item.get("attributes") or {}), **_placement(unit)}
    return {**item, "text_unit_ids": [unit.id], "attributes": attributes or None}


def record_from_payload(payload: dict[str, Any], unit: TextUnit) -> TextUnitExtraction:
    """A stored record re-stamped onto ``unit`` (its id and placement)."""

    def stamp(item: dict[str, Any]) -> dict[str, Any]:
        return _stamp(item, unit)

    relationships = [
        Relationship.model_validate(stamp(item))
        for item in payload.get("relationships") or []
    ]
    for rel in relationships:
        apply_text_unit_weights(
            rel, {unit.id: rel.weight if rel.weight is not None else 1.0}
        )
    return TextUnitExtraction(
        text_unit_id=unit.id,
        entities=[
            Entity.model_validate(stamp(item)) for item in payload.get("entities") or []
        ],
        relationships=relationships,
        rejected=dict(payload.get("rejected") or {}),
        gleaned=bool(payload.get("gleaned")),
        stored=True,
    )


def claims_payload(claims: list[Claim]) -> dict[str, Any]:
    """A text unit's claims without what ties them to the unit, for the store."""
    return {"claims": [_portable(claim) for claim in claims]}


def claims_from_payload(payload: dict[str, Any], unit: TextUnit) -> list[Claim]:
    """Stored claims re-stamped onto ``unit``."""
    return [
        Claim.model_validate(_stamp(item, unit)) for item in payload.get("claims") or []
    ]
