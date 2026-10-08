# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for NeptuneIndexer write-side logic.

Complements ``test_neptune_upsert.py`` (which covers upsert fold/coalesce,
edge JSON serialization, and delete_by_id label scoping). Here we cover the
non-upsert paths and pure helpers:

* ``index_entities`` / ``index_communities`` full-index traversal SHAPE via a
  recording fake traversal (addV per item, label scoping, no fold/coalesce).
* ``_group_items_by_suffix`` partitioning by item ``attributes["index"]``.
* ``_build_vertex_properties`` (None drop, attribute_ prefix, truncation).
* ``neptune_codec`` truncation and JSON serialization for edge list values.

NeptuneClient is patched so no real connection is made.
"""

from __future__ import annotations

import json

import pytest

from unified_kg_rag.adapters.storage.neptune_codec import (
    property_value,
    relationship_properties,
)
from unified_kg_rag.domain.models import Community, Config, Entity, Relationship

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


def _run_full_entity_builder(indexer, entities: list[Entity]) -> list[str]:
    """Drive index_entities' inner builder against a recording traversal."""
    calls: list[str] = []
    captured_factory = None

    def fake_index_generic(items, name, prefix, clear, factory, **kw):
        nonlocal captured_factory
        captured_factory = factory("Entity")
        return None

    orig = indexer._index_generic
    indexer._index_generic = fake_index_generic  # type: ignore[assignment]
    try:
        indexer.index_entities(entities)
    finally:
        indexer._index_generic = orig  # type: ignore[assignment]

    captured_factory(RecordingTraversal(calls), entities)
    return calls


# --------------------------------------------------------------------------- #
# index_entities (full index path: addV, NO fold/coalesce)
# --------------------------------------------------------------------------- #


def test_index_entities_uses_add_v_not_coalesce(indexer) -> None:
    calls = _run_full_entity_builder(indexer, [Entity(id="e1", name="Alice")])
    # Full (clear-and-rebuild) index appends fresh vertices: add_v, never the
    # idempotent fold/coalesce that the upsert path uses.
    assert "add_v" in calls
    assert "coalesce" not in calls
    assert "fold" not in calls


def test_index_entities_one_add_v_per_entity(indexer) -> None:
    calls = _run_full_entity_builder(
        indexer,
        [Entity(id="e1", name="Alice"), Entity(id="e2", name="Bob")],
    )
    assert calls.count("add_v") == 2


def test_index_entities_passes_full_label_to_factory(indexer, mocker) -> None:
    # _index_generic must be called with item type, the capitalized entity
    # prefix as both label and clear prefix (full index clears its own label).
    captured: dict = {}

    def fake_index_generic(items, name, prefix, clear, factory, **kw):
        captured.update({"name": name, "prefix": prefix, "clear": clear, "kw": kw})

    indexer._index_generic = fake_index_generic  # type: ignore[assignment]
    indexer.index_entities([Entity(id="e1", name="Alice")])
    assert captured["name"] == "Entity"
    assert captured["prefix"] == "Entity"
    # Full index clears its own label before rebuild.
    assert captured["clear"] == "Entity"


def test_index_entities_writes_list_props_multi_valued(indexer) -> None:
    # Full index uses _add_properties_to_traversal: list -> repeated property()
    # calls (multi-valued vertex property), scalar -> single property() call.
    calls: list[str] = []
    captured_factory = None

    def fake_index_generic(items, name, prefix, clear, factory, **kw):
        nonlocal captured_factory
        captured_factory = factory("Entity")

    indexer._index_generic = fake_index_generic  # type: ignore[assignment]
    indexer.index_entities([Entity(id="e1", name="Alice", text_unit_ids=["t1", "t2"])])
    captured_factory(
        RecordingTraversal(calls),
        [Entity(id="e1", name="Alice", text_unit_ids=["t1", "t2"])],
    )
    # A 2-element list property fans out to two property() calls (multi-valued),
    # on top of the scalar properties (id, name, and any non-None defaults).
    # So the total exceeds the count of non-list properties by the list length.
    assert calls.count("property") >= 4
    # No sideEffect/drop: the full-index path never clears existing values
    # (unlike the upsert path's _set_properties_on_traversal).
    assert "sideEffect" not in calls


# --------------------------------------------------------------------------- #
# index_communities (full path: clear + addV, MemberOf edges, NO drop-by-target)
# --------------------------------------------------------------------------- #


def test_index_communities_full_clears_label_and_builds_vertices(
    indexer, mocker
) -> None:
    cleared: list[str] = []
    mocker.patch.object(
        indexer,
        "_clear_existing_data_by_label",
        side_effect=lambda label: cleared.append(label),
    )

    vertex_calls: list[str] = []

    def fake_execute_batch(comms, builder, op_name):
        builder(RecordingTraversal(vertex_calls), comms)
        from unified_kg_rag.ports.indexer import IndexingStats

        return IndexingStats(total_items=len(comms), successful_items=len(comms))

    mocker.patch.object(
        indexer, "_execute_batch_traversal", side_effect=fake_execute_batch
    )
    # Stub out the edge creation traversal calls (they touch neptune_client.g).
    mocker.patch.object(indexer, "_execute_with_retries")

    comm = Community(
        id="c1",
        name="Cluster",
        level="0",
        parent="",
        children=[],
        entity_ids=["e1", "e2"],
    )
    indexer.index_communities([comm])

    # Full index clears the community label before rebuilding.
    assert cleared == ["Community-default"]
    # Vertex builder uses add_v (full path), not fold/coalesce.
    assert "add_v" in vertex_calls
    assert "coalesce" not in vertex_calls


def test_upsert_communities_does_not_clear_label(indexer, mocker) -> None:
    cleared: list[str] = []
    mocker.patch.object(
        indexer,
        "_clear_existing_data_by_label",
        side_effect=lambda label: cleared.append(label),
    )

    vertex_calls: list[str] = []

    def fake_execute_batch(comms, builder, op_name):
        builder(RecordingTraversal(vertex_calls), comms)
        from unified_kg_rag.ports.indexer import IndexingStats

        return IndexingStats()

    mocker.patch.object(
        indexer, "_execute_batch_traversal", side_effect=fake_execute_batch
    )
    mocker.patch.object(indexer, "_execute_with_retries")
    # upsert drops existing MemberOf edges through neptune_client.g — that is a
    # MagicMock (NeptuneClient was patched), so the chained calls are inert.

    comm = Community(
        id="c1",
        name="Cluster",
        level="0",
        parent="",
        children=[],
        entity_ids=["e1"],
    )
    indexer.upsert_communities([comm])

    # Incremental upsert must NOT clear the label (would wipe out-of-delta data).
    assert cleared == []
    # Upsert vertex builder uses fold/coalesce for create-or-match by id.
    assert "fold" in vertex_calls
    assert "coalesce" in vertex_calls


def test_index_communities_empty_returns_empty_stats(indexer) -> None:
    stats = indexer.index_communities([])
    assert stats.total_items == 0
    assert stats.successful_items == 0


def test_index_communities_skips_member_edges_when_no_entities(indexer, mocker) -> None:
    # A community with no entity_ids must not attempt any edge creation.
    mocker.patch.object(indexer, "_clear_existing_data_by_label")

    def fake_execute_batch(comms, builder, op_name):
        from unified_kg_rag.ports.indexer import IndexingStats

        return IndexingStats()

    mocker.patch.object(
        indexer, "_execute_batch_traversal", side_effect=fake_execute_batch
    )
    edge_spy = mocker.patch.object(indexer, "_execute_with_retries")

    comm = Community(
        id="c1", name="Cluster", level="0", parent="", children=[], entity_ids=None
    )
    indexer.index_communities([comm])
    edge_spy.assert_not_called()


# --------------------------------------------------------------------------- #
# _group_items_by_suffix
# --------------------------------------------------------------------------- #


def test_group_items_by_suffix_default(indexer) -> None:
    items = [Entity(id="e1", name="A"), Entity(id="e2", name="B")]
    grouped = indexer._group_items_by_suffix(items)
    assert set(grouped.keys()) == {"default"}
    assert len(grouped["default"]) == 2


def test_group_items_by_suffix_partitions_by_index_attribute(indexer) -> None:
    items = [
        Entity(id="e1", name="A", attributes={"index": "tenant-a"}),
        Entity(id="e2", name="B", attributes={"index": "tenant-b"}),
        Entity(id="e3", name="C", attributes={"index": "tenant-a"}),
    ]
    grouped = indexer._group_items_by_suffix(items)
    assert set(grouped.keys()) == {"tenant-a", "tenant-b"}
    assert len(grouped["tenant-a"]) == 2
    assert len(grouped["tenant-b"]) == 1


def test_group_items_by_suffix_index_list_uses_first(indexer) -> None:
    items = [Entity(id="e1", name="A", attributes={"index": ["sfx", "other"]})]
    grouped = indexer._group_items_by_suffix(items)
    assert list(grouped.keys()) == ["sfx"]


# --------------------------------------------------------------------------- #
# _build_vertex_properties
# --------------------------------------------------------------------------- #


def test_build_vertex_properties_drops_none(indexer) -> None:
    entity = Entity(id="e1", name="Alice")
    props = indexer._build_vertex_properties(
        entity, {"name": "Alice", "type": None, "description": None}
    )
    assert props == {"name": "Alice"}


def test_build_vertex_properties_prefixes_attributes(indexer) -> None:
    entity = Entity(id="e1", name="Alice", attributes={"sector": "tech", "x": None})
    props = indexer._build_vertex_properties(entity, {"name": "Alice"})
    # attribute keys are prefixed; None-valued attribute is skipped.
    assert props["attr_sector"] == "tech"
    assert "attr_x" not in props


def test_build_vertex_properties_truncates_long_strings(indexer) -> None:
    max_len = indexer.neptune_config.property_max_length
    long_value = "x" * (max_len + 50)
    entity = Entity(id="e1", name="Alice")
    props = indexer._build_vertex_properties(entity, {"description": long_value})
    assert len(props["description"]) == max_len


def test_build_vertex_properties_serializes_dict_value(indexer) -> None:
    entity = Entity(id="e1", name="Alice", attributes={"meta": {"k": "v"}})
    props = indexer._build_vertex_properties(entity, {"name": "Alice"})
    # dict attribute values are JSON-serialized to a single string.
    assert props["attr_meta"] == json.dumps({"k": "v"})


# --------------------------------------------------------------------------- #
# neptune_codec.property_value (truncation)
# --------------------------------------------------------------------------- #


def test_property_value_clamps_overlong_string() -> None:
    assert property_value("y" * 20, 10) == "y" * 10


def test_property_value_passes_short_string_and_non_strings_through() -> None:
    assert property_value("short", 10) == "short"
    assert property_value(42, 10) == 42
    assert property_value(["a", "b"], 1) == ["a", "b"]


def test_property_value_never_truncates_serialized_json() -> None:
    # A cut JSON string no longer parses, so the read-back would lose it.
    value = {"region": "north", "tier": "gold"}
    assert json.loads(property_value(value, 5)) == value


# --------------------------------------------------------------------------- #
# neptune_codec.relationship_properties (JSON serialization for lists)
# --------------------------------------------------------------------------- #


def test_relationship_properties_skip_none() -> None:
    rel = Relationship(id="r1", source_id="a", target_id="b", weight=0.5, rank=None)
    assert relationship_properties(rel, 100) == {"weight": 0.5}


def test_relationship_properties_never_truncate_serialized_list() -> None:
    rel = Relationship(
        id="r1",
        source_id="a",
        target_id="b",
        rank=None,
        text_unit_ids=["aaaa", "bbbb", "cccc"],
    )
    props = relationship_properties(rel, 5)
    assert json.loads(props["text_unit_ids"]) == ["aaaa", "bbbb", "cccc"]


# --------------------------------------------------------------------------- #
# _execute_batch_traversal — sequential vs concurrent fan-out
# --------------------------------------------------------------------------- #


def _no_op_builder(g, batch):
    """Builder whose product is irrelevant; success is decided by the patched
    _execute_with_retries, so we just return a sentinel traversal."""
    return ("traversal", tuple(batch))


def test_execute_batch_empty_short_circuits(indexer) -> None:
    stats = indexer._execute_batch_traversal([], _no_op_builder, "op")
    assert stats.total_items == 0
    assert stats.successful_items == 0


def test_execute_batch_sequential_counts_all_successes(indexer, mocker) -> None:
    mocker.patch.object(indexer.neptune_config, "batch_size", 2)
    mocker.patch.object(indexer.neptune_config, "index_concurrency", 1)
    mocker.patch.object(indexer, "_execute_with_retries")  # never raises
    items = list(range(5))  # 3 batches: [0,1],[2,3],[4]
    stats = indexer._execute_batch_traversal(items, _no_op_builder, "op")
    assert stats.total_items == 5
    assert stats.successful_items == 5
    assert stats.failed_items == 0


def test_execute_batch_concurrent_matches_sequential_totals(indexer, mocker) -> None:
    # With per-batch isolated stats merged on the main thread, the concurrent
    # path must produce identical totals to the sequential path (no lost
    # updates from shared mutation).
    mocker.patch.object(indexer.neptune_config, "batch_size", 1)
    mocker.patch.object(indexer.neptune_config, "index_concurrency", 4)
    mocker.patch.object(indexer, "_execute_with_retries")
    items = list(range(50))  # 50 single-item batches across 4 workers
    stats = indexer._execute_batch_traversal(items, _no_op_builder, "op")
    assert stats.total_items == 50
    assert stats.successful_items == 50
    assert stats.failed_items == 0


def test_execute_batch_concurrent_multi_item_batches_total_and_rate(
    indexer, mocker
) -> None:
    # Regression: the concurrent path seeded each per-batch IndexingStats with
    # total_items=0, so the merged total_items was 0 and success_rate was 0.0
    # even on full success. With multi-item batches (batch_size>1) the total
    # must equal the item count and success_rate must be 1.0 — and must NOT
    # double-count the outer seed.
    mocker.patch.object(indexer.neptune_config, "batch_size", 10)
    mocker.patch.object(indexer.neptune_config, "index_concurrency", 4)
    mocker.patch.object(indexer, "_execute_with_retries")
    items = list(range(50))  # 5 ten-item batches across 4 workers
    stats = indexer._execute_batch_traversal(items, _no_op_builder, "op")
    assert stats.total_items == 50
    assert stats.successful_items == 50
    assert stats.success_rate == 1.0


def test_execute_batch_concurrency_clamped_to_batch_count(indexer, mocker) -> None:
    # index_concurrency > number of batches must not spawn idle workers nor
    # change correctness; one batch -> sequential-equivalent.
    mocker.patch.object(indexer.neptune_config, "batch_size", 100)
    mocker.patch.object(indexer.neptune_config, "index_concurrency", 8)
    mocker.patch.object(indexer, "_execute_with_retries")
    stats = indexer._execute_batch_traversal([1, 2, 3], _no_op_builder, "op")
    assert stats.total_items == 3
    assert stats.successful_items == 3


def test_execute_batch_falls_back_to_per_item_on_batch_failure(indexer, mocker) -> None:
    # First (batch) call raises; the per-item retries then succeed. The batch
    # had 3 items, so 1 failed batch attempt -> 3 individual successes.
    mocker.patch.object(indexer.neptune_config, "batch_size", 3)
    mocker.patch.object(indexer.neptune_config, "index_concurrency", 1)
    calls = {"n": 0}

    def flaky(traversal, op):
        calls["n"] += 1
        # The very first call is the whole-batch attempt -> fail it.
        if calls["n"] == 1:
            raise RuntimeError("batch boom")

    mocker.patch.object(indexer, "_execute_with_retries", side_effect=flaky)
    stats = indexer._execute_batch_traversal([1, 2, 3], _no_op_builder, "op")
    assert stats.total_items == 3
    assert stats.successful_items == 3  # all recovered individually
    assert stats.failed_items == 0


def test_execute_batch_records_individual_failures(indexer, mocker) -> None:
    mocker.patch.object(indexer.neptune_config, "batch_size", 2)
    mocker.patch.object(indexer.neptune_config, "index_concurrency", 1)

    def always_fail(traversal, op):
        raise RuntimeError("nope")

    mocker.patch.object(indexer, "_execute_with_retries", side_effect=always_fail)
    stats = indexer._execute_batch_traversal([1, 2], _no_op_builder, "op")
    # Batch fails, both items fail individually too.
    assert stats.failed_items == 2
    assert stats.successful_items == 0
    assert stats.errors  # error messages captured


# --------------------------------------------------------------------------- #
# find_incident_relationship_ids (orphan-edge cleanup support)
# --------------------------------------------------------------------------- #


def test_find_incident_relationship_ids_empty_short_circuits(indexer) -> None:
    assert indexer.find_incident_relationship_ids([]) == []


def test_find_incident_relationship_ids_scopes_to_entity_label(indexer, mocker) -> None:
    g = mocker.MagicMock()
    # g.V().hasLabel(label).has(...).bothE().values("id").dedup().toList() -> ids
    (
        g.V.return_value.hasLabel.return_value.has.return_value.bothE.return_value.values.return_value.dedup.return_value.toList.return_value
    ) = ["r1", "r2"]
    indexer.neptune_client.g = g

    out = indexer.find_incident_relationship_ids(["e1", "e2"], suffix="default")
    assert out == ["r1", "r2"]
    # Suffix path scopes by the capitalized entity label.
    g.V.return_value.hasLabel.assert_called_once_with("Entity-default")


def test_find_incident_relationship_ids_coerces_and_drops_none(indexer, mocker) -> None:
    g = mocker.MagicMock()
    (
        g.V.return_value.has.return_value.bothE.return_value.values.return_value.dedup.return_value.toList.return_value
    ) = [1, None, "r3"]
    indexer.neptune_client.g = g
    # No suffix -> unscoped V().has(...) path; results coerced to str, None dropped.
    out = indexer.find_incident_relationship_ids(["e1"])
    assert out == ["1", "r3"]


def test_find_incident_relationship_ids_swallows_errors(indexer, mocker) -> None:
    g = mocker.MagicMock()
    g.V.side_effect = RuntimeError("neptune down")
    indexer.neptune_client.g = g
    assert indexer.find_incident_relationship_ids(["e1"]) == []
