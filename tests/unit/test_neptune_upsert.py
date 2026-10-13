# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Neptune indexer's write path, with no Neptune connection.

Traversal shapes (fold/coalesce upserts, set-cardinality list properties,
drop-before-re-add edges) are checked against a real Gremlin Server by
``tests/integration/test_local_stores.py`` (the CI local-stores job). These
tests cover what a round trip cannot easily force: per-edge failure isolation,
write concurrency, label scoping and property serialization, using a recording
fake traversal.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.domain.models import Config, Relationship

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
    # The failed edge's id lets an incremental commit retry its documents.
    assert stats.failed_ids == ["r2"]


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
    assert stats.failed_ids == ["r2"]


def test_an_edge_whose_drop_failed_is_not_re_added(indexer, mocker) -> None:
    # Re-adding next to the old edge would duplicate it: the batch fails.
    indexer.neptune_config.batch_size = 1
    g = mocker.MagicMock()
    drop = g.E.return_value.has.return_value.where.return_value.drop.return_value
    drop.iterate.side_effect = [RuntimeError("drop timed out"), None]
    indexer.neptune_client.g = g
    built = mocker.patch.object(indexer, "_build_add_edge_traversal")
    mocker.patch.object(indexer, "_execute_with_retries", return_value=["edge-id"])

    stats = indexer.upsert_relationships(
        [
            Relationship(id="r1", source_id="e1", target_id="e2"),
            Relationship(id="r2", source_id="e2", target_id="e3"),
        ]
    )

    assert [c.args[1].id for c in built.call_args_list] == ["r2"]
    assert (stats.successful_items, stats.failed_ids) == (1, ["r1"])


def test_edges_are_written_index_concurrency_at_a_time(indexer, mocker) -> None:
    import threading

    indexer.neptune_config.index_concurrency = 4
    indexer.neptune_client.g = RecordingTraversal([])
    threads: set[int] = set()
    barrier = threading.Barrier(4, timeout=5)

    def execute(traversal, op, *, results=False):
        threads.add(threading.get_ident())
        barrier.wait()
        return ["edge-id"]

    mocker.patch.object(indexer, "_execute_with_retries", side_effect=execute)
    rels = [
        Relationship(id=f"r{i}", source_id=f"e{i}", target_id="e9") for i in range(8)
    ]

    stats = indexer.upsert_relationships(rels)

    assert stats.successful_items == 8
    assert len(threads) == 4


def test_add_edge_traversal_returns_the_edge_id(indexer) -> None:
    calls: list[str] = []
    indexer._build_add_edge_traversal(
        RecordingTraversal(calls),
        Relationship(id="r1", source_id="e1", target_id="e2"),
        "Entity-default",
    )
    assert calls[-1] == "id_"


def test_add_edge_traversal_returns_the_stored_edge_on_a_retry(indexer) -> None:
    # A retry after a write whose response was lost must not add a second
    # edge with the same id: the traversal looks the edge up first.
    from gremlin_python.process.anonymous_traversal import traversal
    from gremlin_python.structure.graph import Graph

    written = indexer._build_add_edge_traversal(
        traversal().withGraph(Graph()),
        Relationship(id="r1", source_id="e1", target_id="e2"),
        "Entity-default",
    )
    steps = written.bytecode.step_instructions
    assert [step[0] for step in steps] == [
        "E",
        "has",
        "where",
        "fold",
        "coalesce",
        "id",
    ]
    assert steps[1][1:] == ["id", "r1"]
    unfold, add_edge = steps[4][1:]
    assert [step[0] for step in unfold.step_instructions] == ["unfold"]
    assert "addE" in [step[0] for step in add_edge.step_instructions]


def test_write_relationships_empty_input_returns_empty_stats(indexer) -> None:
    stats = indexer.upsert_relationships([])
    assert stats.total_items == 0
    assert stats.successful_items == 0
    assert stats.failed_items == 0


def _edge_property_steps(indexer, rel: Relationship) -> list[list]:
    from gremlin_python.process.anonymous_traversal import traversal
    from gremlin_python.structure.graph import Graph

    g = traversal().withGraph(Graph())
    written = indexer._build_add_edge_traversal(g, rel, "Entity-default")
    # The addE branch of the fold().coalesce(unfold(), addE) upsert.
    coalesce = next(
        step for step in written.bytecode.step_instructions if step[0] == "coalesce"
    )
    steps = coalesce[-1].step_instructions
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


def test_a_failed_delete_batch_counts_each_of_its_ids(indexer, mocker) -> None:
    indexer.neptune_config.batch_size = 2
    g = mocker.MagicMock()
    drop = g.E.return_value.has.return_value.where.return_value.drop.return_value
    drop.iterate.side_effect = [RuntimeError("timed out"), None]
    indexer.neptune_client.g = g

    stats = indexer.delete_by_id(["a", "b", "c"], suffix="default")

    assert (stats.total_items, stats.successful_items) == (3, 1)
    assert (stats.failed_items, stats.failed_ids) == (2, ["a", "b"])
