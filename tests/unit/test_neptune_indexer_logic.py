# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""AWS-free unit tests for NeptuneIndexer write-side logic.

Complements ``test_neptune_upsert.py`` (which covers upsert fold/coalesce,
edge JSON serialization, and delete_by_id label scoping). Here we cover the
non-upsert paths and pure helpers:

* ``index_entities`` / ``index_communities`` full-index traversal SHAPE via a
  recording fake traversal (addV per item, label scoping, no fold/coalesce).
* ``get_entity_count`` label construction + result coercion.
* ``_group_items_by_suffix`` partitioning by item ``attributes["index"]``.
* ``_build_vertex_properties`` (None drop, attribute_ prefix, truncation).
* ``_truncate`` length clamping.
* ``_set_edge_properties_on_traversal`` JSON serialization for list values.

NeptuneClient is patched so no real connection is made.
"""

from __future__ import annotations

import json

import pytest

from aws_graphrag.domain.models import Community, Config, Entity

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
        from aws_graphrag.ports.indexer import IndexingStats

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
        from aws_graphrag.ports.indexer import IndexingStats

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
        from aws_graphrag.ports.indexer import IndexingStats

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
# get_entity_count
# --------------------------------------------------------------------------- #


def test_get_entity_count_empty_suffixes_short_circuits(indexer) -> None:
    assert indexer.get_entity_count([]) == 0


def test_get_entity_count_builds_labels_and_coerces_int(indexer, mocker) -> None:
    g = mocker.MagicMock()
    # g.V().hasLabel(*labels).count().next() -> 7
    g.V.return_value.hasLabel.return_value.count.return_value.next.return_value = 7
    indexer.neptune_client.g = g

    assert indexer.get_entity_count(["default", "v2"]) == 7
    # Labels built from capitalized entity prefix + suffix.
    g.V.return_value.hasLabel.assert_called_once_with("Entity-default", "Entity-v2")


def test_get_entity_count_non_numeric_result_yields_zero(indexer, mocker) -> None:
    g = mocker.MagicMock()
    g.V.return_value.hasLabel.return_value.count.return_value.next.return_value = (
        "not-a-number"
    )
    indexer.neptune_client.g = g
    assert indexer.get_entity_count(["default"]) == 0


def test_get_entity_count_swallows_errors(indexer, mocker) -> None:
    g = mocker.MagicMock()
    g.V.side_effect = RuntimeError("boom")
    indexer.neptune_client.g = g
    assert indexer.get_entity_count(["default"]) == 0


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
# _truncate
# --------------------------------------------------------------------------- #


def test_truncate_clamps_overlong_string(indexer) -> None:
    max_len = indexer.neptune_config.property_max_length
    assert indexer._truncate("y" * (max_len * 2)) == "y" * max_len


def test_truncate_passes_short_string_through(indexer) -> None:
    assert indexer._truncate("short") == "short"


def test_truncate_non_string_unchanged(indexer) -> None:
    assert indexer._truncate(42) == 42
    assert indexer._truncate(None) is None


# --------------------------------------------------------------------------- #
# _set_edge_properties_on_traversal (JSON serialization for lists)
# --------------------------------------------------------------------------- #


def test_set_edge_properties_skips_none(indexer) -> None:
    recorded: list[tuple] = []

    class ArgRecorder:
        def property(self, *args, **kwargs):
            recorded.append(args)
            return self

        def __getattr__(self, name):
            return lambda *a, **k: self

    indexer._set_edge_properties_on_traversal(
        ArgRecorder(), {"weight": 0.5, "description": None}
    )
    # None value is skipped; only weight is set.
    assert recorded == [("weight", 0.5)]


def test_set_edge_properties_truncates_serialized_list(indexer, mocker) -> None:
    # The JSON-serialized list passes through _truncate; force a small cap.
    mocker.patch.object(indexer.neptune_config, "property_max_length", 5)
    recorded: list[tuple] = []

    class ArgRecorder:
        def property(self, *args, **kwargs):
            recorded.append(args)
            return self

    indexer._set_edge_properties_on_traversal(
        ArgRecorder(), {"text_unit_ids": ["aaaa", "bbbb", "cccc"]}
    )
    assert len(recorded) == 1
    key, value = recorded[0]
    assert key == "text_unit_ids"
    assert len(value) == 5  # truncated
