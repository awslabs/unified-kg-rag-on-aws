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

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("skip_vertex_id_check")]


class RecordingTraversal:
    """Chainable fake that records every step name it receives."""

    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def __getattr__(self, name: str):
        def _step(*args, **kwargs):
            self._calls.append(name)
            return self

        return _step


def _stored_ends(op: str, rels: list[Relationship]) -> list[str] | None:
    """The endpoint read before the writes: every endpoint is stored."""
    if op == "Reading edge endpoints":
        return [end for rel in rels for end in (rel.source_id, rel.target_id)]
    return None


@pytest.fixture
def indexer(mocker):
    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer.NeptuneClient")
    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer._sleep")
    from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer

    return NeptuneIndexer(config=Config())


def test_write_relationships_isolates_a_failing_edge(indexer, mocker) -> None:
    # Per-edge failure isolation (the behaviour IndexingConfig.max_failure_rate
    # depends on): one failing edge is recorded as an error and skipped, the
    # others still succeed, and stats.total_items counts the whole batch.
    indexer.neptune_client.g = RecordingTraversal([])

    calls = {"n": 0}
    rels = [
        Relationship(id="r1", source_id="e1", target_id="e2"),
        Relationship(id="r2", source_id="e2", target_id="e3"),
        Relationship(id="r3", source_id="e3", target_id="e4"),
    ]

    def flaky_execute(traversal, op, *, results=False):
        if op != "Relationship indexing":  # the reads and drops before
            return _stored_ends(op, rels) or []
        calls["n"] += 1
        if calls["n"] == 2:  # second edge fails
            raise RuntimeError("neptune write rejected")
        return ["edge-id"]

    mocker.patch.object(indexer, "_execute_with_retries", side_effect=flaky_execute)

    stats = indexer.upsert_relationships(rels)
    assert stats.total_items == 3
    assert stats.successful_items == 2
    assert stats.failed_items == 1
    # The failed edge's id lets an incremental commit retry its documents.
    assert stats.failed_ids == ["r2"]


def test_write_relationships_counts_an_edge_without_source_as_failed(
    indexer, mocker
) -> None:
    # addE off V().has(id, source) yields no traverser and no error when the
    # source vertex is missing (here: deleted after the endpoint read); the
    # edge must not be counted as written.
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
        side_effect=lambda traversal, op, *, results=False: (
            written[next(order)]
            if op == "Relationship indexing"
            else _stored_ends(op, rels) or []
        ),
    )

    stats = indexer.upsert_relationships(rels)
    assert stats.total_items == 3
    assert stats.successful_items == 2
    assert stats.failed_items == 1
    assert stats.failed_ids == ["r2"]
    # Either endpoint may be the one gone: the error names both.
    assert any("'missing'" in e and "'e3'" in e for e in stats.errors)


def test_edges_are_written_index_concurrency_at_a_time(indexer, mocker) -> None:
    import threading

    indexer.neptune_config.index_concurrency = 4
    indexer.neptune_client.g = RecordingTraversal([])
    threads: set[int] = set()
    barrier = threading.Barrier(4, timeout=5)

    rels = [
        Relationship(id=f"r{i}", source_id=f"e{i}", target_id="e9") for i in range(8)
    ]

    def execute(traversal, op, *, results=False):
        if op != "Relationship indexing":  # the reads and drops before
            return _stored_ends(op, rels) or []
        threads.add(threading.get_ident())
        barrier.wait()
        return ["edge-id"]

    mocker.patch.object(indexer, "_execute_with_retries", side_effect=execute)

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


def test_the_edge_rewrite_drops_and_adds_in_one_traversal(indexer) -> None:
    # Regression: the stored copy was dropped by a request of its own, so an
    # interruption before the edge's write (a kill, a stop, a lost lease, a
    # failing write) lost the edge for good. The drop is now a side effect of
    # the request that adds the edge, after both endpoints were found.
    from gremlin_python.process.anonymous_traversal import traversal
    from gremlin_python.structure.graph import Graph

    written = indexer._build_add_edge_traversal(
        traversal().withGraph(Graph()),
        Relationship(id="r1", source_id="e1", target_id="e2"),
        "Entity-default",
    )
    steps = written.bytecode.step_instructions
    names = [step[0] for step in steps]
    assert names[:14] == [
        "E",
        "has",
        "where",
        "fold",
        "as",
        "V",
        "hasLabel",
        "has",
        "as",
        "V",
        "hasLabel",
        "has",
        "as",
        "sideEffect",
    ]
    assert steps[1][1:] == ["id", "r1"]
    assert steps[7][1:] == ["id", "e1"] and steps[11][1:] == ["id", "e2"]
    drop = steps[13][1]
    assert [step[0] for step in drop.step_instructions] == ["select", "unfold", "drop"]
    assert names.index("addE") > names.index("sideEffect")
    # Each endpoint is looked up once: the edge goes between the two found.
    assert names.count("V") == 2
    assert [s[1:] for s in steps if s[0] in ("from", "to")] == [["source"], ["target"]]
    assert names[-1] == "id"


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
    steps = written.bytecode.step_instructions
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


def test_delete_by_id_without_a_suffix_is_scoped_to_the_default(indexer) -> None:
    # None used to drop the ids under every label: another tenant's vertex
    # with the same content-derived id went too.
    labels: list[tuple] = []

    class LabelRecorder:
        def hasLabel(self, *args):  # noqa: N802
            labels.append(args)
            return self

        def __getattr__(self, name):
            return lambda *a, **k: self

    indexer.neptune_client.g = LabelRecorder()
    indexer.delete_by_id(["id1"])
    assert labels and all(
        label.endswith("-default") for args in labels for label in args
    ), labels


def test_relationship_pre_drop_is_scoped_to_the_suffix(indexer) -> None:
    # Relationship ids are suffix-independent, so the rewrite's drop must
    # only match edges whose source vertex carries THIS suffix's entity
    # label; an unscoped g.E().has("id", ...) would drop tenant B's identical
    # edge when tenant A is indexed. The scope mirrors delete_by_id.
    from gremlin_python.process.anonymous_traversal import traversal
    from gremlin_python.structure.graph import Graph

    written = indexer._build_add_edge_traversal(
        traversal().withGraph(Graph()),
        Relationship(id="r1", source_id="e1", target_id="e2"),
        "Entity-tenant-a",
    )
    steps = written.bytecode.step_instructions
    where = next(step for step in steps if step[0] == "where")
    rendered = repr(where[1])
    assert "outV" in rendered and "Entity-tenant-a" in rendered
    assert [step[0] for step in steps].index("where") < [
        step[0] for step in steps
    ].index("sideEffect")


def test_a_failed_delete_batch_counts_each_of_its_ids(indexer, mocker) -> None:
    indexer.neptune_config.batch_size = 2
    indexer.neptune_config.max_attempts = 1  # the first batch's drop fails
    g = mocker.MagicMock()
    drop = g.E.return_value.has.return_value.where.return_value.drop.return_value
    drop.iterate.side_effect = [RuntimeError("timed out"), None]
    indexer.neptune_client.g = g

    stats = indexer.delete_by_id(["a", "b", "c"], suffix="default")

    assert (stats.total_items, stats.successful_items) == (3, 1)
    assert (stats.failed_items, stats.failed_ids) == (2, ["a", "b"])
