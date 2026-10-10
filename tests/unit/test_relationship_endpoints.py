# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for cite_relationship_endpoints (pure domain step)."""

from __future__ import annotations

import random
import time

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


def _reference_units(
    entities: list[Entity], relationships: list[Relationship]
) -> list[tuple[str, list[str]]]:
    """The citation rule spelled out with lists, in first-seen order."""
    known = [e.id for e in entities]
    stubs: list[str] = []
    cited: dict[str, list[str]] = {}
    for rel in relationships:
        for ent_id in (rel.source_id, rel.target_id):
            if ent_id not in known and ent_id not in stubs:
                stubs.append(ent_id)
            units = cited.setdefault(ent_id, [])
            units.extend(t for t in rel.text_unit_ids or [] if t not in units)
    out = []
    for ent_id, own in [(e.id, list(e.text_unit_ids or [])) for e in entities] + [
        (s, []) for s in stubs
    ]:
        out.append((ent_id, own + [t for t in cited.get(ent_id, []) if t not in own]))
    return out


@pytest.mark.parametrize("seed", range(20))
def test_order_matches_the_list_based_rule(seed: int) -> None:
    rng = random.Random(seed)
    ids = [f"e{i}" for i in range(6)]
    units = [f"t{i}" for i in range(8)]
    entities = [
        Entity(id=ent_id, name=ent_id.upper(), text_unit_ids=rng.sample(units, 2))
        for ent_id in ids[:3]
    ]
    rels = [
        _rel(
            rng.choice(ids),
            rng.choice(ids),
            [rng.choice(units) for _ in range(rng.randint(0, 3))],
        )
        for _ in range(rng.randint(0, 12))
    ]
    out, _ = cite_relationship_endpoints(entities, rels)
    assert [(e.id, e.text_unit_ids) for e in out] == _reference_units(entities, rels)


def test_high_degree_entity_is_linear() -> None:
    # One entity on 20k edges, each from its own unit: the list-based
    # membership checks took ~5 s here; the bound only catches that regression.
    hub = Entity(id="hub", name="Hub", text_unit_ids=["t-own"])
    rels = [_rel("hub", f"leaf{i}", [f"t{i}"]) for i in range(20_000)]
    start = time.perf_counter()
    out, stubs = cite_relationship_endpoints([hub], rels)
    elapsed = time.perf_counter() - start
    assert stubs == 20_000
    assert out[0].text_unit_ids == ["t-own"] + [f"t{i}" for i in range(20_000)]
    assert elapsed < 2.0
