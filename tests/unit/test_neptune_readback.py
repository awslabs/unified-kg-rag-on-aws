# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Neptune read-back inverts the Neptune write encoding (AWS-free).

Cross-run merge reads existing entities/relationships back from Neptune and
merges the delta into them. These tests build the REAL write traversals, turn
their property steps into the rows Gremlin would return (vertex lists are
multi-valued set-cardinality properties, edge lists are JSON strings, the
relationship type is the edge label), and feed those rows to the REAL read
methods, so a lossy encoding/decoding pair fails here instead of in a live
graph.
"""

from __future__ import annotations

from typing import Any

import pytest
from gremlin_python.process.anonymous_traversal import traversal
from gremlin_python.process.traversal import Cardinality
from gremlin_python.structure.graph import Graph

from unified_kg_rag.domain.ingestion.merge import merge_relationships
from unified_kg_rag.domain.models import Config, Entity, Relationship
from unified_kg_rag.ports.indexer import BaseIndexer, IndexingStats

pytestmark = pytest.mark.unit

_ENTITY_LABEL = "Entity-default"


class _ReadChain:
    """Chainable stand-in for ``neptune_client.g`` whose ``toList`` yields rows.

    Records the step names so tests can assert the read is label-scoped.
    """

    def __init__(self, rows: list[Any], steps: list[tuple[str, tuple]]) -> None:
        self._rows = rows
        self.steps = steps

    def __getattr__(self, name: str):
        def _step(*args: Any, **kwargs: Any) -> Any:
            if name == "toList":
                return list(self._rows)
            self.steps.append((name, args))
            return self

        return _step


@pytest.fixture
def indexer(mocker):
    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer.NeptuneClient")
    from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer

    return NeptuneIndexer(config=Config())


def _g():
    return traversal().withGraph(Graph())


def _edge_row(indexer, rel: Relationship) -> dict[str, Any]:
    """The projected row Gremlin returns for the edge the indexer writes."""
    written = indexer._build_add_edge_traversal(_g(), rel, _ENTITY_LABEL)
    steps = written.bytecode.step_instructions
    add_e = next(i for i, step in enumerate(steps) if step[0] == "addE")
    props: dict[str, Any] = {}
    for step in steps[add_e + 1 :]:
        if step[0] != "property":
            continue
        # Edge properties never take a cardinality and hold one value each.
        assert len(step) == 3, step
        props[step[1]] = step[2]
    return {
        "props": props,
        "label": steps[add_e][1],
        "source_id": rel.source_id,
        "target_id": rel.target_id,
    }


def _serve(indexer, rows: list[Any]) -> list[tuple[str, tuple]]:
    steps: list[tuple[str, tuple]] = []
    indexer.neptune_client.g = _ReadChain(rows, steps)
    return steps


def _supplies(**overrides: Any) -> Relationship:
    fields: dict[str, Any] = {
        "id": "r-supplies",
        "source_id": "e-vendor",
        "source_name": "Vendor",
        "target_id": "e-buyer",
        "target_name": "Buyer",
        "type": "SUPPLIES",
        "weight": 2.0,
        "rank": 3,
        "description": "Vendor supplies Buyer.",
        "text_unit_ids": ["t1", "t2"],
        "attributes": {"index": "default", "filters": {"region": "north"}},
    }
    fields.update(overrides)
    return Relationship(**fields)


def test_relationship_read_back_inverts_the_edge_encoding(indexer) -> None:
    written = _supplies()
    _serve(indexer, [_edge_row(indexer, written)])

    (read,) = indexer.read_relationships([written.id])

    assert read.model_dump(exclude={"created_at", "updated_at"}) == written.model_dump(
        exclude={"created_at", "updated_at"}
    )


def test_relationship_read_back_is_scoped_to_the_suffix_label(indexer) -> None:
    steps = _serve(indexer, [])

    indexer.read_relationships(["r-supplies"], suffix="tenant-a")

    assert ("hasLabel", ("Entity-tenant-a",)) in [
        (name, args) for name, args in _flatten_steps(steps)
    ]


def test_cross_run_merge_of_a_read_back_edge_keeps_one_edge(indexer) -> None:
    stored = _supplies(text_unit_ids=["t1"], weight=1.0)
    _serve(indexer, [_edge_row(indexer, stored)])
    old = indexer.read_relationships([stored.id])

    delta = _supplies(text_unit_ids=["t2"], description="Vendor ships to Buyer.")
    merged = merge_relationships(old, [delta])

    assert len(merged) == 1
    (edge,) = merged
    assert edge.id == stored.id
    assert edge.type == "SUPPLIES"
    assert edge.text_unit_ids == ["t1", "t2"]
    assert edge.weight == 2.0


def test_an_untyped_edge_merges_with_its_untyped_delta(indexer) -> None:
    # Written under the default label, it reads back with that type; the merge
    # key must treat both forms as one type.
    stored = _supplies(type=None, text_unit_ids=["t1"])
    _serve(indexer, [_edge_row(indexer, stored)])
    old = indexer.read_relationships([stored.id])
    assert old[0].type == "RELATED_TO"

    merged = merge_relationships(old, [_supplies(type=None, text_unit_ids=["t2"])])

    assert [(r.id, r.text_unit_ids) for r in merged] == [(stored.id, ["t1", "t2"])]


def test_text_unit_ids_survive_repeated_round_trips(indexer) -> None:
    edge = _supplies(text_unit_ids=["t1"])
    for unit in ("t2", "t3"):
        _serve(indexer, [_edge_row(indexer, edge)])
        (edge,) = merge_relationships(
            indexer.read_relationships([edge.id]),
            [_supplies(text_unit_ids=[unit])],
        )

    assert edge.text_unit_ids == ["t1", "t2", "t3"]
    assert edge.weight == 3.0


def test_long_text_unit_lists_are_not_truncated_into_invalid_json(indexer) -> None:
    max_length = indexer.neptune_config.property_max_length
    units = [f"text-unit-{i:06d}" for i in range(max_length // 10)]
    written = _supplies(text_unit_ids=units)
    _serve(indexer, [_edge_row(indexer, written)])

    (read,) = indexer.read_relationships([written.id])

    assert read.text_unit_ids == units


def _flatten_steps(steps: list[tuple[str, tuple]]) -> list[tuple[str, tuple]]:
    """Top-level steps plus the steps of anonymous traversals passed as args."""
    flat: list[tuple[str, tuple]] = []
    for name, args in steps:
        flat.append((name, args))
        for arg in args:
            bytecode = getattr(arg, "bytecode", None)
            if bytecode is not None:
                flat.extend(
                    (step[0], tuple(step[1:])) for step in bytecode.step_instructions
                )
    return flat


# --- entities ----------------------------------------------------------------


def _vertex_value_map(
    indexer, entity: Entity, stored: dict[str, list[Any]] | None = None
) -> dict[str, list[Any]]:
    """The ``valueMap()`` of the vertex after the indexer's upsert of ``entity``.

    Applies the upsert's property steps with Neptune's semantics: ``single``
    replaces, ``set`` (and no cardinality, Neptune's default) adds a distinct
    value, and a ``sideEffect(properties(k).drop())`` clears ``k``.
    """
    captured: dict[str, Any] = {}

    def capture(items, builder, operation_name):
        captured["traversal"] = builder(_g(), items)
        return IndexingStats(total_items=len(items), successful_items=len(items))

    indexer._execute_batch_traversal = capture
    indexer.upsert_entities([entity])
    value_map = {key: list(values) for key, values in (stored or {}).items()}
    value_map.setdefault("id", [entity.id])
    for step in captured["traversal"].bytecode.step_instructions:
        if step[0] == "sideEffect":
            dropped = step[1].step_instructions[0]
            assert dropped[0] == "properties"
            value_map.pop(dropped[1], None)
        elif step[0] == "property":
            args = step[1:]
            cardinality = args[0] if isinstance(args[0], Cardinality) else None
            key, value = args[-2], args[-1]
            if cardinality is Cardinality.single:
                value_map[key] = [value]
            elif value not in value_map.setdefault(key, []):
                value_map[key].append(value)
    return value_map


def _vendor(**overrides: Any) -> Entity:
    fields: dict[str, Any] = {
        "id": "e-vendor",
        "name": "Vendor",
        "type": "organization",
        "description": "Vendor supplies parts.",
        "text_unit_ids": ["t1", "t2"],
        "community_ids": ["c1"],
        "rank": 4,
        "confidence": 0.75,
        "attributes": {"index": "tenant-a", "filters": {"region": "north"}},
    }
    fields.update(overrides)
    return Entity(**fields)


def test_entity_read_back_inverts_the_vertex_encoding(indexer) -> None:
    written = _vendor()
    _serve(indexer, [_vertex_value_map(indexer, written)])

    (read,) = indexer.read_entities([written.id], suffix="tenant-a")

    # frequency is derived (len(text_unit_ids)) and not stored.
    exclude = {"created_at", "updated_at", "frequency"}
    assert read.model_dump(exclude=exclude) == written.model_dump(exclude=exclude)
    assert BaseIndexer.get_suffix(read) == "tenant-a"


def test_single_text_unit_reads_back_as_a_list(indexer) -> None:
    written = _vendor(text_unit_ids=["t1"], community_ids=None, attributes=None)
    _serve(indexer, [_vertex_value_map(indexer, written)])

    (read,) = indexer.read_entities([written.id])

    assert read.text_unit_ids == ["t1"]
    assert read.community_ids is None


def test_entity_read_back_is_scoped_to_the_suffix_label(indexer) -> None:
    steps = _serve(indexer, [])

    indexer.read_entities(["e-vendor"], suffix="tenant-a")

    assert ("hasLabel", ("Entity-tenant-a",)) in _flatten_steps(steps)


def test_upsert_of_a_merged_entity_replaces_its_lists(indexer) -> None:
    stored = _vertex_value_map(indexer, _vendor(text_unit_ids=["t1", "t2"]))
    rewritten = _vendor(text_unit_ids=["t2", "t3"], community_ids=["c2"])
    _serve(indexer, [_vertex_value_map(indexer, rewritten, stored)])

    (read,) = indexer.read_entities(["e-vendor"], suffix="tenant-a")

    assert read.text_unit_ids == ["t2", "t3"]
    assert read.community_ids == ["c2"]
