# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-parsed fields that arrive as lists (repeated XML tags) are coerced."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.base_processor import (
    BaseProcessor,
    coerce_llm_text,
)
from unified_kg_rag.domain.models import Config, TextUnit

pytestmark = pytest.mark.unit


def _text_unit() -> TextUnit:
    return TextUnit(id="tu1", short_id="tu1", text="Vendor supplies Buyer.")


@pytest.mark.parametrize(
    ("value", "join", "expected"),
    [
        (None, False, ""),
        ("  Vendor ", False, "Vendor"),
        (["", " Vendor ", "Supplier"], False, "Vendor"),
        (["", "  "], False, ""),
        (["First.", "Second.", "First."], True, "First.\nSecond."),
        (3, False, "3"),
    ],
)
def test_coerce_llm_text(value, join: bool, expected: str) -> None:
    assert coerce_llm_text(value, join=join) == expected


def test_relationship_with_repeated_tags_is_parsed() -> None:
    proc = BaseProcessor(config=Config())
    rel = proc.parse_relationship_data(
        {
            "source": ["Vendor", "Vendor"],
            "target": ["Buyer"],
            "type": ["SUPPLIES", "SELLS_TO"],
            "description": ["Vendor supplies parts.", "Paid monthly."],
        },
        _text_unit(),
        entity_name_to_id={},
    )
    assert rel is not None
    assert (rel.source_name, rel.target_name, rel.type) == (
        "Vendor",
        "Buyer",
        "SUPPLIES",
    )
    assert rel.description == "Vendor supplies parts.\nPaid monthly."


def test_entity_with_repeated_tags_is_parsed() -> None:
    proc = BaseProcessor(config=Config())
    entity = proc.parse_entity_data(
        {
            "name": ["Vendor", "Vendor Inc"],
            "type": ["ORGANIZATION", "COMPANY"],
            "description": ["A supplier.", "Ships parts."],
        },
        _text_unit(),
    )
    assert entity is not None
    assert (entity.name, entity.type) == ("Vendor", "ORGANIZATION")
    assert entity.description == "A supplier.\nShips parts."
