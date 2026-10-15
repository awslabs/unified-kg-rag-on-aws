# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph resolution and the cross-run merge merge attributes with one rule.

A full build resolves the N instances of an item at once; an incremental
build folds them into the stored item two at a time. Both must give the same
attributes, document ``filters`` included (a multi-valued filter is a sorted
list): the incremental two-way merge kept a delta's list unsorted when the
stored item had no filters, while a full build sorted it.
"""

from __future__ import annotations

from functools import reduce
from itertools import product
from typing import Any

import pytest

from unified_kg_rag.domain.ingestion.claim_resolver import ClaimResolver
from unified_kg_rag.domain.ingestion.graph_resolver import (
    EntityResolver,
    RelationshipResolver,
)
from unified_kg_rag.domain.ingestion.merge import (
    merge_claims,
    merge_entities,
    merge_relationships,
)
from unified_kg_rag.domain.models import Claim, Config, Entity, Relationship

pytestmark = pytest.mark.unit

ATTRIBUTES: list[dict[str, Any] | None] = [
    None,
    {},
    {"filters": {}},
    {"filters": {"tenant": "b"}},
    {"filters": {"tenant": ["b", "a"]}},
    {"filters": {"tenant": ["a"]}, "suffix": "s1"},
    {"filters": {"tenant": "a", "region": "x"}},
    {"suffix": "s2"},
]

COMBINATIONS = [
    *product(ATTRIBUTES, repeat=2),
    *product(ATTRIBUTES[3:], repeat=3),
]


def _entity(index: int, attributes: dict[str, Any] | None) -> Entity:
    return Entity(
        id="e1", name="Vendor", text_unit_ids=[f"t{index}"], attributes=attributes
    )


def _relationship(index: int, attributes: dict[str, Any] | None) -> Relationship:
    return Relationship(
        id="r1",
        source_id="e1",
        target_id="e2",
        type="SUPPLIES",
        text_unit_ids=[f"t{index}"],
        attributes=attributes,
    )


def _claim(index: int, attributes: dict[str, Any] | None) -> Claim:
    return Claim(
        id="c1",
        subject_id="e1",
        subject_name="Vendor",
        object_id="e2",
        object_name="Buyer",
        type="OBLIGATION",
        text_unit_ids=[f"t{index}"],
        attributes=attributes,
    )


def _fold(merge, items):
    """Index the items one run at a time, each merged into the stored one."""
    return reduce(lambda stored, item: merge([stored], [item]), items[1:], items[0])


@pytest.mark.parametrize("maps", COMBINATIONS)
def test_full_and_incremental_builds_merge_attributes_alike(maps) -> None:
    config = Config()
    entities = [_entity(i, a) for i, a in enumerate(maps)]
    relationships = [_relationship(i, a) for i, a in enumerate(maps)]
    claims = [_claim(i, a) for i, a in enumerate(maps)]

    full = (
        EntityResolver(config, use_process_pool=False)._merge_entities(entities),
        RelationshipResolver(config, use_process_pool=False)._merge_relationships(
            relationships
        ),
        ClaimResolver(config)._merge_claims(claims),
    )
    incremental = (
        _fold(lambda old, new: merge_entities(old, new)[0][0], entities),
        _fold(lambda old, new: merge_relationships(old, new)[0], relationships),
        _fold(lambda old, new: merge_claims(old, new)[0], claims),
    )

    assert [item.attributes for item in incremental] == [
        item.attributes for item in full
    ]
