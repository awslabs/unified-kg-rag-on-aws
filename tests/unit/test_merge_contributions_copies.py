# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``merge_contributions`` leaves its records as they were and returns a
graph that shares no mutable object with them, while copying only the
statement each merge keeps. Synthetic data only."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from unified_kg_rag.domain.ingestion import contributions
from unified_kg_rag.domain.ingestion.base_processor import (
    SEED_PROPERTIES_ATTRIBUTE,
    BaseProcessor,
)
from unified_kg_rag.domain.ingestion.contributions import merge_contributions
from unified_kg_rag.domain.models import (
    Constants,
    Entity,
    Relationship,
    TextUnitExtraction,
)

pytestmark = pytest.mark.unit


def _attributes(unit: int) -> dict[str, Any]:
    return {
        Constants.FILTERS.value: {
            "region": f"r{unit}",
            # Unhashable filter values take _union_filters' list path.
            "tags": [{"k": f"t{unit}"}],
        },
        SEED_PROPERTIES_ATTRIBUTE: {
            "aliases": [f"alias {unit}"],
            "address": {"city": f"city {unit}"},
        },
        "nested": {"list": [unit]},
    }


def _records() -> list[TextUnitExtraction]:
    records = []
    for unit in range(3):
        unit_id = f"unit-{unit}"
        entities = [
            Entity(
                id=BaseProcessor._generate_entity_id(name),
                name=name,
                type="ORGANIZATION",
                description=f"{name} in unit {unit}",
                text_unit_ids=[unit_id],
                attributes=_attributes(unit),
            )
            for name in ("Vendor", "Buyer")
        ]
        relationship = Relationship(
            id=BaseProcessor._generate_relationship_id("Vendor", "Buyer", "SUPPLIES"),
            source_id=entities[0].id,
            source_name="Vendor",
            target_id=entities[1].id,
            target_name="Buyer",
            type="SUPPLIES",
            description=f"supplies in unit {unit}",
            text_unit_ids=[unit_id],
            attributes=_attributes(unit),
        )
        records.append(
            TextUnitExtraction(
                text_unit_id=unit_id, entities=entities, relationships=[relationship]
            )
        )
    return records


def _mutables(value: Any, found: dict[int, Any]) -> dict[int, Any]:
    """Every mutable container reachable from ``value``, by id."""
    if isinstance(value, BaseModel):
        found[id(value)] = value
        for name in type(value).model_fields:
            _mutables(getattr(value, name), found)
    elif isinstance(value, dict):
        found[id(value)] = value
        for item in value.values():
            _mutables(item, found)
    elif isinstance(value, list | set):
        found[id(value)] = value
        for item in value:
            _mutables(item, found)
    return found


def test_the_merged_graph_shares_no_mutable_object_with_the_records() -> None:
    records = _records()
    before = [record.model_dump() for record in records]
    entities, relationships = merge_contributions(records)

    assert [record.model_dump() for record in records] == before
    inputs = _mutables(records, {})
    outputs = _mutables([entities, relationships], {})
    shared = [type(outputs[i]).__name__ for i in outputs.keys() & inputs.keys()]
    assert not shared
    # The merge kept every unit's seed properties and filters.
    vendor = next(e for e in entities if e.name == "Vendor")
    assert vendor.attributes is not None
    assert vendor.attributes[SEED_PROPERTIES_ATTRIBUTE]["address"] == {"city": "city 2"}
    assert vendor.attributes[Constants.FILTERS.value]["region"] == ["r0", "r1", "r2"]


def test_only_the_statement_each_merge_keeps_is_copied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copies: list[str] = []
    for model in (Entity, Relationship):
        original = model.model_copy

        def counted(
            self: Any, *args: Any, __original: Any = original, **kw: Any
        ) -> Any:
            copies.append(type(self).__name__)
            return __original(self, *args, **kw)

        monkeypatch.setattr(model, "model_copy", counted)
    monkeypatch.setattr(
        contributions, "cite_relationship_endpoints", lambda e, r: (e, [])
    )
    merge_contributions(_records())
    # Two entity ids and one relationship id over three units each.
    assert sorted(copies) == ["Entity", "Entity", "Relationship"]
