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

from tests.fixtures.fakes.gremlin import RecordingTraversal
from unified_kg_rag.domain.models import Config, Relationship

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("skip_vertex_id_check")]


def _stored_ends(op: str, rels: list[Relationship]) -> list[dict] | None:
    """The endpoint read before the writes: every endpoint is stored (its
    vertex id ``v-<id>``)."""
    if op == "Reading edge endpoints":
        return [
            {"id": end, "vertex": f"v-{end}"}
            for rel in rels
            for end in (rel.source_id, rel.target_id)
        ]
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

    def flaky_execute(traversal, op, *, results=False, fenced=True):
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
        side_effect=lambda traversal, op, *, results=False, fenced=True: (
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

    def execute(traversal, op, *, results=False, fenced=True):
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
        "v-e1",
        "v-e2",
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
        "v-e1",
        "v-e2",
    )
    steps = written.bytecode.step_instructions
    names = [step[0] for step in steps]
    assert names[:10] == [
        "E",
        "has",
        "where",
        "fold",
        "as",
        "V",
        "as",
        "V",
        "as",
        "sideEffect",
    ]
    assert steps[1][1:] == ["id", "r1"]
    assert steps[5][1:] == ["v-e1"] and steps[7][1:] == ["v-e2"]
    drop = steps[9][1]
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
    written = indexer._build_add_edge_traversal(g, rel, "Entity-default", "v1", "v2")
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
        "v-e1",
        "v-e2",
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
    indexer.config.retry.graph_store_max_attempts = 1  # the first batch's drop fails
    g = mocker.MagicMock()
    drop = g.E.return_value.has.return_value.where.return_value.drop.return_value
    drop.iterate.side_effect = [RuntimeError("timed out"), None]
    indexer.neptune_client.g = g

    stats = indexer.delete_by_id(["a", "b", "c"], suffix="default")

    assert (stats.total_items, stats.successful_items) == (3, 1)
    assert (stats.failed_items, stats.failed_ids) == (2, ["a", "b"])


def _set_value_requests(indexer, mocker, fail_op: str | None = None) -> list[list]:
    """The ``text_unit_ids`` values of each request the writes send (the
    requests recorded as their steps)."""
    from tests.fixtures.fakes.gremlin import RecordingTraversal

    requests: list[list] = []

    def execute(
        build, op, *, results=False, fenced=True, exhausted_level=None
    ):  # noqa: ANN001, ANN202
        steps: list = []
        build(RecordingTraversal(steps, with_args=True))
        values = [
            args[2]
            for name, args in steps
            if name == "property" and len(args) == 3 and args[1] == "text_unit_ids"
        ]
        requests.append(values)
        if op == fail_op:
            raise RuntimeError("query too large")
        return ["v"] if results else []

    mocker.patch.object(indexer, "_execute_with_retries", side_effect=execute)
    return requests


def test_a_hub_vertex_writes_its_set_values_in_bounded_requests(
    indexer, mocker
) -> None:
    # A hub entity's 2,500 text units in one request: Neptune refused 5,000
    # with QueryTooLargeException under load. Each request writes at most
    # _SET_VALUES_PER_REQUEST of them; the rest are added (set cardinality:
    # adding a value again changes nothing).
    from unified_kg_rag.adapters.storage import neptune_indexer as module
    from unified_kg_rag.domain.models import Entity

    units = [f"t{i:05d}" for i in range(2_500)]
    requests = _set_value_requests(indexer, mocker)

    stats = indexer.upsert_entities(
        [Entity(id="hub", name="Hub", text_unit_ids=units), Entity(id="e", name="E")]
    )

    assert (stats.successful_items, stats.failed_items) == (2, 0)
    bound = module._SET_VALUES_PER_REQUEST
    assert all(len(values) <= bound for values in requests)
    assert [v for values in requests for v in values] == units
    assert len([values for values in requests if values]) == 3


def test_a_failed_set_value_request_fails_its_vertex_only(indexer, mocker) -> None:
    from unified_kg_rag.domain.models import Entity

    requests = _set_value_requests(
        indexer, mocker, fail_op="Adding vertex property values"
    )

    stats = indexer.upsert_entities(
        [
            Entity(id="hub", name="Hub", text_unit_ids=[f"t{i}" for i in range(1_500)]),
            Entity(id="e", name="E", text_unit_ids=["t1"]),
        ]
    )

    # The hub is written again whole by a later write (its failure is noted).
    assert (stats.successful_items, stats.failed_items) == (1, 1)
    assert stats.failed_ids == ["hub"]
    assert len(requests) == 2


class _VertexModel:
    """One stored vertex, changed by the requests the write sends as Neptune
    would apply them (each request whole), so a test can stop the write
    after any request and read what a reader, or the next run, would."""

    def __init__(self, stored: dict[str, list]) -> None:
        self.props = {key: list(values) for key, values in stored.items()}

    def apply(self, steps: list, op: str) -> list:
        names = [name for name, _ in steps]
        if op == "Reading vertex property values":
            (key,) = next(args for name, args in steps if name == "values")
            return list(self.props.get(key, []))
        if "hasValue" in names:  # dropping stale values
            (key,) = next(args for name, args in steps if name == "properties")
            (within,) = next(args for name, args in steps if name == "hasValue")
            self.props[key] = [v for v in self.props[key] if v not in within.value]
            return []
        for name, args in steps:
            if name == "sideEffect":  # the write's drop of stored properties
                kept = {
                    str(step[1].value)
                    for step in args[0].bytecode.step_instructions
                    if step[0] == "hasKey"
                }
                self.props = {k: v for k, v in self.props.items() if k in kept}
            elif name == "property" and len(args) == 3:
                values = self.props.setdefault(args[1], [])
                if args[2] not in values:
                    values.append(args[2])
        return ["v"]


@pytest.mark.parametrize("kill_at", [*range(1, 7), None])
def test_an_interrupted_hub_write_never_stores_part_of_a_set(
    indexer, mocker, kill_at: int | None
) -> None:
    # Regression: the first request dropped the stored text_unit_ids and wrote
    # 1,000 of the new ones; a run killed before the follow-ups left the hub
    # with those 1,000 for good (the next run merges with the stored vertex).
    # Now every request until the last only adds, so at any stop the vertex
    # holds the old set or the new one in full, and the finished write the
    # new one exactly.
    from unified_kg_rag.domain.models import Entity

    old = [f"t{i:05d}" for i in range(2_200)]
    new = [f"t{i:05d}" for i in range(1_100, 3_400)] + ["t-new"]
    vertex = _VertexModel({"text_unit_ids": old})

    class _Killed(BaseException):
        pass

    sent = {"n": 0}

    def execute(
        build, op, *, results=False, fenced=True, exhausted_level=None
    ):  # noqa: ANN001, ANN202
        sent["n"] += 1
        if sent["n"] == kill_at:
            raise _Killed
        steps: list = []
        build(RecordingTraversal(steps, with_args=True))
        out = vertex.apply(steps, op)
        return out if results else []

    mocker.patch.object(indexer, "_execute_with_retries", side_effect=execute)
    entity = Entity(id="hub", name="Hub", text_unit_ids=new)
    try:
        stats = indexer.upsert_entities([entity])
    except _Killed:
        stored = set(vertex.props.get("text_unit_ids", []))
        assert stored >= set(old) or stored >= set(new)
        return
    assert kill_at is None
    # Three adds, the read, two drops of the 1,100 stale values.
    assert sent["n"] == 6
    assert (stats.successful_items, stats.failed_items) == (1, 0)
    assert vertex.props["text_unit_ids"] == new
