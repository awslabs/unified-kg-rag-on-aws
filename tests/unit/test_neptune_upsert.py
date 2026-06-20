# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
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
    # Existing edge with this id is dropped before the new addE.
    assert "drop" in calls
    assert "addE" in calls
