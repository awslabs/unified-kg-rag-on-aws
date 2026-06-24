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

from aws_graphrag.domain.models import Config, Entity, Relationship

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
    mocker.patch("aws_graphrag.adapters.storage.neptune_indexer.NeptuneClient")
    from aws_graphrag.adapters.storage.neptune_indexer import NeptuneIndexer

    return NeptuneIndexer(config=Config())


def _run_entity_builder(indexer, entities: list[Entity]) -> list[str]:
    calls: list[str] = []
    # Reach the inner builder the same way _index_generic does.
    builder_factory = None

    def fake_index_generic(items, name, prefix, clear, factory, **kw):
        nonlocal builder_factory
        builder_factory = factory("Entity")
        return None

    import aws_graphrag.adapters.storage.neptune_indexer as mod  # noqa: F401

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


def test_upsert_relationship_drops_edge_by_id_before_readd(indexer) -> None:
    calls: list[str] = []

    def fake_index_generic(items, name, prefix, clear, factory, **kw):
        builder = factory("Entity", "Entity")
        builder(RecordingTraversal(calls), items)
        return None

    indexer._index_generic = fake_index_generic  # type: ignore[assignment]
    indexer.upsert_relationships(
        [Relationship(id="r1", source_id="e1", target_id="e2")]
    )
    # Existing edges are dropped first, then each addE is fanned out from a
    # single root via sideEffect (the addE lives inside the anonymous __ spawn
    # passed to sideEffect, so it is not a call on the root recorder).
    assert "drop" in calls
    assert "sideEffect" in calls
    # The root chain must NOT re-enter V()/E() between edges (the old chaining
    # bug): after the initial drop, only inject + sideEffect drive the root.
    assert calls.count("sideEffect") == 1


def test_edge_properties_never_use_cardinality(indexer) -> None:
    # Regression: Neptune raises "Cardinality specification may not be used with
    # Edge properties". _set_edge_properties_on_traversal must emit plain
    # property(key, value) calls with NO Cardinality positional argument.
    recorded: list[tuple] = []

    class ArgRecordingTraversal:
        def property(self, *args, **kwargs):
            recorded.append(args)
            return self

        def __getattr__(self, name):
            return lambda *a, **k: self

    indexer._set_edge_properties_on_traversal(
        ArgRecordingTraversal(),
        {"weight": 0.5, "text_unit_ids": ["t1", "t2"], "description": "x"},
    )
    assert recorded, "expected property() calls"
    for args in recorded:
        # Each call is (key, value): exactly two positional args, key is a str
        # (a Cardinality arg would make the first positional a Cardinality enum).
        assert len(args) == 2, f"edge property got cardinality arg: {args}"
        assert isinstance(args[0], str)


def test_edge_list_property_serialized_to_json_string(indexer) -> None:
    # Edges cannot hold multi-valued properties, so a list must become a single
    # JSON string (one property call), not repeated property() calls.
    recorded: list[tuple] = []

    class ArgRecordingTraversal:
        def property(self, *args, **kwargs):
            recorded.append(args)
            return self

    indexer._set_edge_properties_on_traversal(
        ArgRecordingTraversal(), {"text_unit_ids": ["t1", "t2"]}
    )
    assert len(recorded) == 1
    key, value = recorded[0]
    assert key == "text_unit_ids"
    assert value == '["t1", "t2"]'


def test_delete_by_id_scopes_by_label_when_suffix_given(indexer) -> None:
    # Cross-tenant safety: with a suffix, the drop must scope to that suffix's
    # entity/community labels (hasLabel), not match raw id across all tenants.
    labels: list[tuple] = []

    class LabelRecorder:
        def hasLabel(self, *args):  # noqa: N802
            labels.append(args)
            return self

        def __getattr__(self, name):
            return lambda *a, **k: self

    indexer.neptune_client.g = LabelRecorder()
    indexer.delete_by_id(["id1", "id2"], suffix="default")
    # Both the edge drop and vertex drop carried a hasLabel scope.
    assert labels, "expected hasLabel scoping when suffix is provided"
    assert any("Entity-default" in a for a in labels)
    assert any("Community-default" in a for a in labels)


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
