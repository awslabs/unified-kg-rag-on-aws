# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for cite_relationship_endpoints (pure domain step)."""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.ingestion.relationship_endpoints import (
    cite_relationship_endpoints,
)
from unified_kg_rag.domain.models import Entity, Relationship

pytestmark = pytest.mark.unit


def _rel(source: str, target: str, units: list[str]) -> Relationship:
    return Relationship(
        id=f"{source}-{target}",
        source_id=source,
        target_id=target,
        source_name=source.title(),
        target_name=target.title(),
        type="X",
        text_unit_ids=units,
    )


def test_duplicate_ids_each_keep_their_own_units() -> None:
    # Before a gleaning round's duplicate merge the same id appears twice;
    # each copy gains the edge's units and keeps its own.
    entities = [
        Entity(id="a", name="A", text_unit_ids=["t1"]),
        Entity(id="a", name="A", text_unit_ids=["t2"]),
        Entity(id="b", name="B", text_unit_ids=["t3"]),
    ]
    out, stubs = cite_relationship_endpoints(entities, [_rel("a", "b", ["t3"])])
    assert stubs == 0
    assert [e.text_unit_ids for e in out] == [["t1", "t3"], ["t2", "t3"], ["t3"]]


def test_pure_and_idempotent() -> None:
    entities = [Entity(id="a", name="A", text_unit_ids=["t1"])]
    rels = [_rel("a", "c", ["t2"])]
    once, stubs = cite_relationship_endpoints(entities, rels)
    twice, again = cite_relationship_endpoints(once, rels)
    assert entities[0].text_unit_ids == ["t1"]
    assert (stubs, again) == (1, 0)
    assert [(e.id, e.text_unit_ids, e.frequency) for e in once] == [
        (e.id, e.text_unit_ids, e.frequency) for e in twice
    ]
    assert [(e.id, e.text_unit_ids) for e in once] == [
        ("a", ["t1", "t2"]),
        ("c", ["t2"]),
    ]
