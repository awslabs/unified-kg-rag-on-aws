# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free unit tests for NeptuneIndexer write-side logic.

Complements ``test_neptune_upsert.py`` (the upsert path: per-edge failure
isolation, write concurrency, label scoping and property serialization) and
``tests/integration/test_local_stores.py`` (the traversal shapes against a
real Gremlin Server). Here we cover the non-upsert paths and pure helpers:

* ``index_entities`` / ``index_communities`` full-index traversal SHAPE via a
  recording fake traversal (create-or-match per item, label scoping), and the
  label clear.
* Retries of reads and drops, and writes that a retry cannot duplicate.
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
    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer._sleep")
    from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer

    return NeptuneIndexer(config=Config())


def _run_full_entity_builder(indexer, entities: list[Entity]) -> list[str]:
    """Drive index_entities' inner builder against a recording traversal."""
    calls: list[str] = []
    captured_factory = None

    def fake_index_generic(items, name, prefix, factory, *, clear_first, suffixes):
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
# index_entities (full index path: clear, then create-or-match per entity)
# --------------------------------------------------------------------------- #


def test_index_entities_matches_an_existing_vertex_instead_of_adding_one(
    indexer,
) -> None:
    # Regression: the full build appended vertices (addV). A retry reaches
    # Neptune over a new connection, so a batch whose response was lost was
    # written twice: 100 duplicate entity vertices with "0 failed".
    calls = _run_full_entity_builder(indexer, [Entity(id="e1", name="Alice")])
    assert "add_v" not in calls  # only inside the coalesce
    assert calls[:4] == ["V", "has", "fold", "coalesce"]


def test_index_entities_one_create_or_match_per_entity(indexer) -> None:
    calls = _run_full_entity_builder(
        indexer,
        [Entity(id="e1", name="Alice"), Entity(id="e2", name="Bob")],
    )
    assert calls.count("coalesce") == 2


def test_index_entities_passes_full_label_to_factory(indexer, mocker) -> None:
    # _index_generic must be called with the item type and the entity label
    # prefix, clearing that label first (full index rebuilds its own label).
    captured: dict = {}

    def fake_index_generic(items, name, prefix, factory, *, clear_first, suffixes):
        captured.update({"name": name, "prefix": prefix, "clear": clear_first})

    indexer._index_generic = fake_index_generic  # type: ignore[assignment]
    indexer.index_entities([Entity(id="e1", name="Alice")])
    assert captured["name"] == "Entity"
    assert captured["prefix"] == "Entity"
    # Full index clears its own label before rebuild.
    assert captured["clear"] is True


def test_index_entities_writes_list_props_multi_valued(indexer) -> None:
    # list -> its values replaced (drop, then one property() per value: a
    # multi-valued vertex property), scalar -> single property() call.
    calls: list[str] = []
    captured_factory = None

    def fake_index_generic(items, name, prefix, factory, *, clear_first, suffixes):
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
    # The values are replaced, not appended to, so a retry adds none.
    assert "sideEffect" in calls


# --------------------------------------------------------------------------- #
# index_communities (full path: clear + create-or-match, MemberOf edges, NO
# drop-by-target)
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
    # A retry after a lost response matches the vertex instead of adding one.
    assert "add_v" not in vertex_calls
    assert "coalesce" in vertex_calls


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


@pytest.mark.parametrize("method", ["index_communities", "upsert_communities"])
def test_communities_are_written_without_membership_edges(
    indexer, mocker, method
) -> None:
    # No query read the MemberOf edges, a re-keyed entity lost them and they
    # counted toward its degree: a community is its vertex, an entity names
    # its communities (community_ids).
    from unified_kg_rag.ports.indexer import IndexingStats

    mocker.patch.object(indexer, "_clear_existing_data_by_label")
    batch = mocker.patch.object(
        indexer, "_execute_batch_traversal", return_value=IndexingStats()
    )
    execute = mocker.patch.object(indexer, "_execute_with_retries")
    comm = Community(
        id="c1", name="C", level="0", parent="", children=[], entity_ids=["e1"]
    )

    getattr(indexer, method)([comm])

    assert batch.call_count == 1
    execute.assert_not_called()


def test_index_communities_empty_returns_empty_stats(indexer) -> None:
    stats = indexer.index_communities([])
    assert stats.total_items == 0
    assert stats.successful_items == 0


@pytest.mark.parametrize("method", ["index_communities", "index_entities"])
def test_full_index_without_items_clears_the_suffix_label(
    indexer, mocker, method
) -> None:
    # A rebuild with nothing of this type for the suffix empties its label
    # instead of leaving the previous build's vertices in place.
    clear = mocker.patch.object(indexer, "_clear_existing_data_by_label")
    batch = mocker.patch.object(indexer, "_execute_batch_traversal")

    getattr(indexer, method)([], suffixes=["tenant-a"])

    (label,) = [call.args[0] for call in clear.call_args_list]
    assert label.endswith("tenant-a")
    batch.assert_not_called()


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

    def flaky(traversal, op, **kwargs):
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


def test_execute_batch_reports_the_ids_of_failed_items(indexer, mocker) -> None:
    from unified_kg_rag.domain.models import Entity

    mocker.patch.object(indexer.neptune_config, "batch_size", 2)
    mocker.patch.object(indexer.neptune_config, "index_concurrency", 1)
    entities = [Entity(id="e1", name="Vendor"), Entity(id="e2", name="Buyer")]

    def fail_e2(build, op):
        traversal = build(None)
        if traversal == ["e2"] or len(traversal) == 2:
            raise RuntimeError("rejected")

    mocker.patch.object(indexer, "_execute_with_retries", side_effect=fail_e2)
    builder = lambda g, batch: [entity.id for entity in batch]  # noqa: E731
    stats = indexer._execute_batch_traversal(entities, builder, "op")

    assert (stats.successful_items, stats.failed_items) == (1, 1)
    assert stats.failed_ids == ["e2"]


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
        g.V.return_value.hasLabel.return_value.has.return_value.bothE.return_value.values.return_value.dedup.return_value.toList.return_value
    ) = [1, None, "r3"]
    indexer.neptune_client.g = g
    # Results coerced to str, None dropped.
    out = indexer.find_incident_relationship_ids(["e1"])
    assert out == ["1", "r3"]


def test_find_incident_relationship_ids_raises_on_a_failed_read(
    indexer, mocker
) -> None:
    # [] would read as "no incident edges": their vector documents would be
    # orphaned for good.
    from unified_kg_rag.shared import AWSServiceError

    g = mocker.MagicMock()
    g.V.side_effect = RuntimeError("neptune down")
    indexer.neptune_client.g = g
    with pytest.raises(AWSServiceError, match="neptune down"):
        indexer.find_incident_relationship_ids(["e1"])


def test_find_incident_relationship_ids_batches_the_ids(indexer, mocker) -> None:
    indexer.neptune_config.batch_size = 2
    g = mocker.MagicMock()
    has = g.V.return_value.hasLabel.return_value.has
    has.return_value.bothE.return_value.values.return_value.dedup.return_value.toList.side_effect = [
        ["r1"],
        ["r1", "r2"],
    ]
    indexer.neptune_client.g = g

    assert indexer.find_incident_relationship_ids(["e1", "e2", "e3"]) == ["r1", "r2"]
    assert [c.args[1].value for c in has.call_args_list] == [["e1", "e2"], ["e3"]]


# --------------------------------------------------------------------------- #
# _clear_existing_data_by_label (bounded, retried drops)
# --------------------------------------------------------------------------- #


class _ClearTraversal:
    """A fake traversal source: the label count returns ``initial`` and each
    drop request the next scripted count of dropped vertices; labels, limit(n),
    counts and drops are recorded."""

    def __init__(self, dropped: list[int], initial: int | None = None) -> None:
        self._dropped = list(dropped)
        self._initial = sum(dropped) if initial is None else initial
        self._limited = False
        self.labels: list[str] = []
        self.limits: list[int] = []
        self.counts = 0
        self.drops = 0

    def V(self):  # noqa: N802
        self._limited = False
        return self

    def hasLabel(self, label):  # noqa: N802
        self.labels.append(label)
        return self

    def limit(self, n):
        self._limited = True
        self.limits.append(n)
        return self

    def sideEffect(self, traversal):  # noqa: N802
        self.drops += 1
        return self

    def count(self):
        return self

    def toList(self):  # noqa: N802
        if not self._limited:
            self.counts += 1
            return [self._initial]
        return [self._dropped.pop(0)]


def _clearing(indexer, dropped: list[int], initial: int | None = None):
    fake = _ClearTraversal(dropped, initial)
    indexer.neptune_client.g = fake
    return fake


def test_clearing_an_empty_label_sends_one_drop(indexer) -> None:
    fake = _clearing(indexer, dropped=[0])
    indexer._clear_existing_data_by_label("MyLabel")
    assert (fake.counts, fake.drops) == (1, 1)


def test_clearing_stops_after_a_short_batch(indexer) -> None:
    fake = _clearing(indexer, dropped=[5, 5, 2])
    indexer._clear_existing_data_by_label("MyLabel", batch_size=5)
    # One count up front, then one bounded drop per request, each scoped to
    # the label, no recount.
    assert fake.counts == 1
    assert fake.drops == 3
    assert fake.limits == [5, 5, 5]
    assert set(fake.labels) == {"MyLabel"}


def test_clearing_stops_after_an_empty_batch(indexer) -> None:
    fake = _clearing(indexer, dropped=[5, 0])
    indexer._clear_existing_data_by_label("MyLabel", batch_size=5)
    assert fake.drops == 2


def test_clearing_allows_vertices_added_meanwhile(indexer) -> None:
    # 10 counted, 15 dropped (writers added some): within the slack.
    fake = _clearing(indexer, dropped=[5, 5, 5, 0], initial=10)
    indexer._clear_existing_data_by_label("MyLabel", batch_size=5)
    assert fake.drops == 4


def test_clearing_fails_instead_of_looping_forever(indexer) -> None:
    from unified_kg_rag.adapters.storage import neptune_indexer
    from unified_kg_rag.shared import AWSServiceError

    # Every request reports a full batch: the passes stop at the bound.
    fake = _clearing(indexer, dropped=[5] * 100, initial=10)
    with pytest.raises(AWSServiceError, match="did not finish"):
        indexer._clear_existing_data_by_label("MyLabel", batch_size=5)
    assert fake.drops == 2 + neptune_indexer._EXTRA_DELETE_PASSES


def test_clear_raises_the_failure(indexer, monkeypatch) -> None:
    # Raised, not reported as False: the caller needs the reason.
    from unified_kg_rag.adapters.storage import neptune_indexer
    from unified_kg_rag.shared import AWSServiceError

    monkeypatch.setattr(neptune_indexer, "_sleep", lambda _s: None)

    class _Boom:
        def V(self):  # noqa: N802
            raise RuntimeError("gremlin exploded")

    indexer.neptune_client.g = _Boom()
    with pytest.raises(AWSServiceError, match="gremlin exploded"):
        indexer.clear(["default"])


# --------------------------------------------------------------------------- #
# Reads and drops survive a lost connection (retried like writes)
# --------------------------------------------------------------------------- #


class _LostOnce:
    """A traversal source whose first request fails as a lost connection.

    Every later request answers ``rows``; a drop answers nothing.
    """

    def __init__(self, rows: list) -> None:
        self.rows = rows
        self.requests = 0

    def __getattr__(self, name: str):
        def step(*args, **kwargs):
            if name in ("toList", "iterate"):
                self.requests += 1
                if self.requests == 1:
                    raise ConnectionResetError("connection reset by peer")
                return list(self.rows) if name == "toList" else None
            return self

        return step


_READS = {
    "read_entities": ([{"id": ["e1"], "name": ["Vendor"]}], 1),
    "read_relationships": (
        [
            {
                "props": {"id": "r1"},
                "label": "RELATED",
                "source_id": "a",
                "target_id": "b",
            }
        ],
        1,
    ),
    "read_neighbor_ids": ([{"id": "e1", "neighbors": ["e2"]}], 1),
    "find_incident_relationship_ids": (["r1"], 1),
}


@pytest.mark.parametrize("read", sorted(_READS))
def test_a_read_retries_after_a_lost_connection(indexer, read) -> None:
    # Regression: reads used client.g directly with no retry and no reset, so
    # a sub-second failover failed the run, and in a long-lived process every
    # later incremental run failed its first read (check_graph_holds_registry).
    rows, expected = _READS[read]
    g = _LostOnce(rows)
    indexer.neptune_client.g = g

    result = getattr(indexer, read)(["e1"], suffix="default")

    assert len(result) == expected
    assert g.requests == 2
    indexer.neptune_client.reset.assert_called_once_with(g)


def test_read_entity_names_retries_after_a_lost_connection(indexer) -> None:
    indexer.neptune_client.g = _LostOnce([{"id": "e1", "name": "Vendor"}])
    assert indexer.read_entity_names("default") == [("e1", "Vendor")]


def test_delete_by_id_retries_after_a_lost_connection(indexer) -> None:
    g = _LostOnce([])
    indexer.neptune_client.g = g

    stats = indexer.delete_by_id(["e1"], suffix="default")

    assert (stats.successful_items, stats.failed_items) == (1, 0)
    assert g.requests == 3  # edges (lost, retried), then vertices


def test_relationship_drop_retries_after_a_lost_connection(indexer, mocker) -> None:
    # Regression: the drop-by-id before the edge writes had one attempt, so a
    # lost connection failed the whole batch of edges. Every request answers
    # the endpoints' ids (the endpoint read, then the edge write).
    g = _LostOnce(["a", "b"])
    indexer.neptune_client.g = g
    rel = Relationship(id="r1", source_id="a", target_id="b")

    stats = indexer.upsert_relationships([rel])

    assert (stats.successful_items, stats.failed_items) == (1, 0)


def test_an_edge_whose_endpoint_is_missing_keeps_its_stored_copy(
    indexer, mocker
) -> None:
    # Regression: the edges were dropped by id before the write found the
    # endpoint missing, so an edge re-pointed at a re-keyed entity whose write
    # failed was lost, though an unchanged document still states it.
    dropped: list[str] = []

    class _Query:
        def __init__(self) -> None:
            self.steps: list[tuple[str, tuple]] = []

        def __getattr__(self, name: str):
            def step(*args, **kwargs):
                self.steps.append((name, args))
                return self

            return step

    def execute(build, operation, *, results=False, **kwargs):
        query = build(_Query())
        if operation == "Reading edge endpoints":
            return ["a", "b"]
        if operation == "Dropping existing edges":
            within = next(args for name, args in query.steps if name == "has")
            dropped.extend(within[1].value)
            return []
        return ["written"]

    mocker.patch.object(indexer, "_execute_with_retries", side_effect=execute)
    kept = Relationship(id="r1", source_id="a", target_id="b")
    repointed = Relationship(id="r2", source_id="a", target_id="gone")

    stats = indexer.upsert_relationships([kept, repointed])

    assert dropped == ["r1"]
    assert (stats.successful_items, stats.failed_items) == (1, 1)
    assert stats.failed_ids == ["r2"]


def test_member_drop_retries_after_a_lost_connection(indexer, mocker) -> None:
    from unified_kg_rag.ports.indexer import IndexingStats

    mocker.patch.object(
        indexer, "_execute_batch_traversal", return_value=IndexingStats()
    )
    indexer.neptune_client.g = _LostOnce([])
    comm = Community(
        id="c1", name="C", level="0", parent="", children=[], entity_ids=["e1"]
    )

    assert indexer.upsert_communities([comm]).failed_ids == []


def test_a_rebuild_in_place_writes_tagged_and_clears_no_label(indexer, mocker) -> None:
    clear = mocker.patch.object(indexer, "_clear_existing_data_by_label")
    properties: list[tuple] = []

    class Recorder(RecordingTraversal):
        def property(self, *args):
            properties.append(args)
            return self

    def fake_execute_batch(items, builder, op_name):
        builder(Recorder([]), items)
        from unified_kg_rag.ports.indexer import IndexingStats

        return IndexingStats(total_items=len(items), successful_items=len(items))

    mocker.patch.object(
        indexer, "_execute_batch_traversal", side_effect=fake_execute_batch
    )
    drops = mocker.patch.object(indexer, "_drop_in_batches", return_value=0)

    assert indexer.begin_rebuild(["default"])
    indexer.index_entities([Entity(id="e1", name="A")], suffixes=["default"])
    clear.assert_not_called()
    # Every vertex the build writes carries its tag.
    assert any(args[1:] == ("build_tag", indexer._rebuild_tag) for args in properties)
    indexer.end_rebuild(["default"])

    # Edges, then entity and community vertices the build did not write.
    assert [c.args[1] for c in drops.call_args_list] == [
        "stale edges of 'Entity-default'",
        "stale vertices of 'Entity-default'",
        "stale vertices of 'Community-default'",
    ]
    # The rebuild is over: a full build clears the label again.
    indexer.index_entities([Entity(id="e1", name="A")], suffixes=["default"])
    clear.assert_called_once_with("Entity-default")


def test_a_failed_rebuild_drops_nothing(indexer, mocker) -> None:
    drops = mocker.patch.object(indexer, "_drop_in_batches")
    indexer.begin_rebuild(["default"])
    indexer.end_rebuild(["default"], sweep=False)
    drops.assert_not_called()
