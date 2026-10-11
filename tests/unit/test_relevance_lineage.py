# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for lineage-based entity/relationship relevance.

Relevance uses the authoritative ``text_unit_ids`` recorded during
extraction instead of token-overlap heuristics — exact and language-agnostic.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.base_processor import group_by_text_unit
from unified_kg_rag.domain.models import Entity, Relationship

pytestmark = pytest.mark.unit


def test_entities_grouped_by_the_units_they_cite() -> None:
    alice = Entity(id="e1", name="Alice", text_unit_ids=["t1", "t2"])
    bob = Entity(id="e2", name="Bob", text_unit_ids=["t2"])
    by_unit = group_by_text_unit([alice, bob])
    assert by_unit == {"t1": [alice], "t2": [alice, bob]}


def test_no_lineage_is_relevant_to_no_unit() -> None:
    assert group_by_text_unit([Entity(id="e1", name="Alice")]) == {}


def test_language_agnostic() -> None:
    # A non-Latin name that token-overlap heuristics would mishandle is
    # still matched purely by lineage.
    entity = Entity(id="e1", name="앨리스", text_unit_ids=["t1"])
    assert group_by_text_unit([entity])["t1"] == [entity]


def test_relationships_and_repeated_unit_ids() -> None:
    rel = Relationship(
        id="r1", source_id="e1", target_id="e2", text_unit_ids=["t1", "t1"]
    )
    assert group_by_text_unit([rel]) == {"t1": [rel]}
