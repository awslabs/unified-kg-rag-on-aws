# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Neptune idempotent upsert traversal construction (M2).

The headline M2 guarantee is that an incremental upsert does NOT create
duplicate vertices/edges or accumulate duplicate property values on re-run.
These tests assert the *traversal shape* the builder emits — fold/coalesce for
vertices, drop-edge-by-id before addE, and multi-valued list properties via
Cardinality.set (not a JSON string) — using a recording fake traversal so no
Neptune connection is needed.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import Config, Entity, Relationship

pytestmark = pytest.mark.unit


class RecordingTraversal:
    """Chainable fake that records every step name it receives."""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def __getattr__(self, name: str):
        def _step(*args, **kwargs):
            self._calls.append(name)
            return self

        return _step


@pytest.fixture
def indexer(mocker):
    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer.NeptuneClient")
    from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer

    return NeptuneIndexer(config=Config())


def _run_entity_builder(indexer, entities: list[Entity]) -> list[str]:
    calls: list[str] = []
    # Reach the inner builder the same way _index_generic does.
    builder_factory = None

    def fake_index_generic(items, name, prefix, factory, *, clear_first):
        nonlocal builder_factory
        builder_factory = factory("Entity")
        return None

    import unified_kg_rag.adapters.storage.neptune_indexer as mod  # noqa: F401

    orig = indexer._index_generic
    indexer._index_generic = fake_index_generic  # type: ignore[assignment]
    try:
        indexer.upsert_entities(entities)
    finally:
        indexer._index_generic = orig  # type: ignore[assignment]

    builder_factory(RecordingTraversal(calls), entities)
    return calls


def test_upsert_entity_uses_fold_coalesce(indexer) -> None:
    calls = _run_entity_builder(indexer, [Entity(id="e1", name="Alice")])
    # Idempotent create-or-match: fold().coalesce(unfold(), addV(...)).
    assert "fold" in calls
    assert "coalesce" in calls


def test_upsert_entity_sets_list_props_without_json_string(indexer, mocker) -> None:
    # Spy on _set_properties_on_traversal to capture the props dict passed.
    captured: dict = {}
    orig = indexer._set_properties_on_traversal

    def spy(traversal, props):
        captured.update(props)
        return orig(traversal, props)

    mocker.patch.object(indexer, "_set_properties_on_traversal", side_effect=spy)
    _run_entity_builder(
        indexer, [Entity(id="e1", name="Alice", text_unit_ids=["t1", "t2"])]
    )
    # text_unit_ids must remain a real list (multi-valued), not a JSON string.
    assert isinstance(captured.get("text_unit_ids"), list)
    assert captured["text_unit_ids"] == ["t1", "t2"]


def test_set_properties_list_uses_set_cardinality_and_drop(indexer) -> None:
    calls: list[str] = []
    indexer._set_properties_on_traversal(
        RecordingTraversal(calls), {"text_unit_ids": ["t1", "t2"], "name": "Alice"}
    )
    # List property is cleared (sideEffect/drop) then re-added per element.
    assert "sideEffect" in calls
    # property() called for each list element + the scalar.
    assert calls.count("property") == 3


def test_upsert_relationship_drops_edge_by_id_before_readd(indexer, mocker) -> None:
    # Per-edge write path: existing edges are dropped by id (batched) up front,
    # then each edge is added as its OWN small traversal (not a sideEffect fan-out
    # off one root — that shape made Neptune ~240x slower per edge). We record the
    # steps issued on the shared g and the add-edge traversals built per edge.
    drop_calls: list[str] = []
    added_edges: list[str] = []

    indexer.neptune_client.g = RecordingTraversal(drop_calls)
    # _execute_with_retries just iterates the traversal; stub it so no real
    # traversal .iterate() is needed and count the per-edge add builds instead.
    mocker.patch.object(indexer, "_execute_with_retries")
    orig_build = indexer._build_add_edge_traversal

    def spy_build(g, rel, entity_label):
        added_edges.append(rel.id)
        return orig_build(g, rel, entity_label)

    mocker.patch.object(indexer, "_build_add_edge_traversal", side_effect=spy_build)

    indexer.upsert_relationships(
        [
            Relationship(id="r1", source_id="e1", target_id="e2"),
            Relationship(id="r2", source_id="e2", target_id="e3"),
        ]
    )
    # Existing edges dropped by id before re-add (idempotency).
    assert "drop" in drop_calls
    # Each relationship is written as its own add-edge traversal.
    assert added_edges == ["r1", "r2"]


def test_write_relationships_isolates_a_failing_edge(indexer, mocker) -> None:
    # Per-edge failure isolation (the behaviour IndexingConfig.max_failure_rate
    # depends on): one failing edge is recorded as an error and skipped, the
    # others still succeed, and stats.total_items counts the whole batch.
    indexer.neptune_client.g = RecordingTraversal([])

    calls = {"n": 0}

    def flaky_execute(traversal, op, *, results=False):
        calls["n"] += 1
        if calls["n"] == 2:  # second edge fails
            raise RuntimeError("neptune write rejected")
        return ["edge-id"]

    mocker.patch.object(indexer, "_execute_with_retries", side_effect=flaky_execute)

    stats = indexer.upsert_relationships(
        [
            Relationship(id="r1", source_id="e1", target_id="e2"),
            Relationship(id="r2", source_id="e2", target_id="e3"),
            Relationship(id="r3", source_id="e3", target_id="e4"),
        ]
    )
    assert stats.total_items == 3
    assert stats.successful_items == 2
    assert stats.failed_items == 1


def test_write_relationships_counts_an_edge_without_source_as_failed(
    indexer, mocker
) -> None:
    # addE off V().has(id, source) yields no traverser and no error when the
    # source vertex is missing; the edge must not be counted as written.
    indexer.neptune_client.g = RecordingTraversal([])
    written = {"r1": ["r1"], "r2": [], "r3": ["r3"]}
    rels = [
        Relationship(id="r1", source_id="e1", target_id="e2"),
        Relationship(id="r2", source_id="missing", target_id="e3"),
        Relationship(id="r3", source_id="e3", target_id="e4"),
    ]
    # Edges are written in input order, so key the fake result on call order.
    order = iter(r.id for r in rels)
    mocker.patch.object(
        indexer,
        "_execute_with_retries",
        side_effect=lambda traversal, op, *, results=False: written[next(order)],
    )

    stats = indexer.upsert_relationships(rels)
    assert stats.total_items == 3
    assert stats.successful_items == 2
    assert stats.failed_items == 1


def test_add_edge_traversal_returns_the_edge_id(indexer) -> None:
    calls: list[str] = []
    indexer._build_add_edge_traversal(
        RecordingTraversal(calls),
        Relationship(id="r1", source_id="e1", target_id="e2"),
        "Entity-default",
    )
    assert calls[-1] == "id_"


def test_write_relationships_empty_input_returns_empty_stats(indexer) -> None:
    stats = indexer.upsert_relationships([])
    assert stats.total_items == 0
    assert stats.successful_items == 0
    assert stats.failed_items == 0


def _edge_property_steps(indexer, rel: Relationship) -> list[list]:
    from gremlin_python.process.anonymous_traversal import traversal
    from gremlin_python.structure.graph import Graph

    g = traversal().withGraph(Graph())
    steps = indexer._build_add_edge_traversal(
        g, rel, "Entity-default"
    ).bytecode.step_instructions
    add_e = next(i for i, step in enumerate(steps) if step[0] == "addE")
    return [step for step in steps[add_e + 1 :] if step[0] == "property"]


def test_edge_properties_never_use_cardinality(indexer) -> None:
    # Regression: Neptune raises "Cardinality specification may not be used with
    # Edge properties". The edge write must emit plain property(key, value)
    # steps with NO Cardinality positional argument.
    rel = Relationship(
        id="r1",
        source_id="a",
        target_id="b",
        weight=0.5,
        description="x",
        text_unit_ids=["t1", "t2"],
    )
    steps = _edge_property_steps(indexer, rel)
    assert steps, "expected property() steps"
    for step in steps:
        # Each step is (key, value): a Cardinality arg would make the first
        # positional a Cardinality enum.
        assert len(step) == 3, f"edge property got cardinality arg: {step}"
        assert isinstance(step[1], str)


def test_edge_list_property_serialized_to_json_string(indexer) -> None:
    # Edges cannot hold multi-valued properties, so a list must become a single
    # JSON string (one property step), not repeated property() steps.
    rel = Relationship(
        id="r1", source_id="a", target_id="b", text_unit_ids=["t1", "t2"]
    )
    steps = [s for s in _edge_property_steps(indexer, rel) if s[1] == "text_unit_ids"]
    assert steps == [["property", "text_unit_ids", '["t1", "t2"]']]


def test_delete_by_id_scopes_by_label_when_suffix_given(indexer) -> None:
    # Cross-tenant safety: with a suffix, the VERTEX drop scopes by hasLabel and
    # the EDGE drop scopes by endpoint label via where(outV().hasLabel(...)) —
    # NOT by edge hasLabel (edges are labeled by relationship type, so an
    # edge hasLabel(entity_label) filter would match nothing and leak edges).
    label_calls: list[tuple] = []
    step_names: list[str] = []

    class Recorder:
        def hasLabel(self, *args):  # noqa: N802
            label_calls.append(args)
            return self

        def __getattr__(self, name):
            step_names.append(name)
            return lambda *a, **k: self

    indexer.neptune_client.g = Recorder()
    indexer.delete_by_id(["id1", "id2"], suffix="default")
    # Vertex drop scopes to this suffix's entity + community labels.
    assert any("Entity-default" in a for a in label_calls)
    assert any("Community-default" in a for a in label_calls)
    # Edge drop is scoped by an endpoint-vertex label filter, not edge hasLabel.
    assert "where" in step_names


def test_delete_by_id_unscoped_without_suffix(indexer) -> None:
    # Legacy single-tenant path: no suffix -> no hasLabel scoping.
    labels: list[tuple] = []

    class LabelRecorder:
        def hasLabel(self, *args):  # noqa: N802
            labels.append(args)
            return self

        def __getattr__(self, name):
            return lambda *a, **k: self

    indexer.neptune_client.g = LabelRecorder()
    indexer.delete_by_id(["id1"])
    assert labels == [], "unscoped delete must not call hasLabel"


def test_relationship_pre_drop_is_scoped_to_the_suffix(indexer, mocker) -> None:
    # Relationship ids are suffix-independent, so the idempotency pre-drop must
    # only match edges whose source vertex carries THIS suffix's entity label;
    # an unscoped g.E().has("id", ...) would drop tenant B's identical edge when
    # tenant A is indexed. The scope mirrors delete_by_id: where(outV().hasLabel).
    from unified_kg_rag.domain.models import Constants

    steps: list[tuple[str, tuple]] = []

    class Recorder:
        def __getattr__(self, name):
            def _step(*args, **kwargs):
                steps.append((name, args))
                return self

            return _step

    indexer.neptune_client.g = Recorder()
    mocker.patch.object(indexer, "_execute_with_retries")
    mocker.patch.object(indexer, "_build_add_edge_traversal")

    indexer.upsert_relationships(
        [
            Relationship(
                id="r1",
                source_id="e1",
                target_id="e2",
                attributes={Constants.INDEX.value: "tenant-a"},
            )
        ]
    )

    names = [name for name, _ in steps]
    # The edge drop is filtered by an endpoint-vertex label before drop().
    assert names.index("where") < names.index("drop")
    where_args = next(args for name, args in steps if name == "where")
    # The anonymous where() traversal is outV().hasLabel(<this suffix's label>).
    rendered = repr(where_args[0].bytecode)
    assert "outV" in rendered
    assert "Entity-tenant-a" in rendered
