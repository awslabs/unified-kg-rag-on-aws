# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``indexing.neptune.vertex_ids``: the traversals of each mode and the check
that refuses a graph built in the other one.

Traversals are built on a connection-less source and translated to Groovy,
so their shape is checked without a server; the same writes and reads run
against Gremlin Server in both modes in the graph conformance suites
(``container-custom-ids``) and ``tests/integration/test_neptune_vertex_ids.py``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import pytest
from gremlin_python.process.translator import Translator
from gremlin_python.structure.graph import Graph

from unified_kg_rag.adapters.aws.neptune import (
    VERTEX_ID_SWITCH_LABEL,
    stored_vertex_id_mode,
    vertex_id,
)
from unified_kg_rag.domain.models import (
    Community,
    Config,
    Entity,
    Relationship,
    SearchQuery,
)
from unified_kg_rag.shared.exceptions import StoreConfigurationError

pytestmark = pytest.mark.unit

_LABEL = "Entity-default"


def _config(mode: str) -> Config:
    config = Config()
    config.indexing.neptune.vertex_ids = mode  # type: ignore[assignment]
    return config


def _groovy(traversal: Any) -> str:
    return str(Translator("g").translate(traversal.bytecode))


@pytest.fixture
def make_indexer(mocker) -> Callable[[str], Any]:
    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer.NeptuneClient")
    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer._sleep")
    from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer

    return lambda mode: NeptuneIndexer(config=_config(mode))


class _Graph:
    """Answers the indexer's requests (by operation name) and records them.

    ``stored`` is the mode of the label's sample vertex (``None``: empty),
    ``switch`` the mode a switch vertex records (``None``: none).
    """

    def __init__(self, stored: str | None = None, switch: str | None = None) -> None:
        self.stored = stored
        self.switch = switch
        self.requests: list[tuple[str, str]] = []

    def execute(self, build: Callable, operation: str, **kwargs: Any) -> list[Any]:
        groovy = _groovy(build(Graph().traversal()))
        self.requests.append((operation, groovy))
        if operation.startswith("Reading the vertex-id switch"):
            return [self.switch] if self.switch else []
        if operation.startswith("Reading the vertex ids"):
            label = groovy.split("hasLabel('")[1].split("'")[0]
            if self.stored is None:
                return []
            vid = vertex_id(label, "e1") if self.stored == "custom" else 4096
            return [{"vertex_id": vid, "id": ["e1"]}]
        if operation.startswith("Recording the vertex-id switch"):
            self.switch = groovy.split("property(Cardinality.single,'vertex_ids','")[
                1
            ].split("'")[0]
        if operation.startswith("Dropping the vertex-id switch"):
            self.switch = None
        if operation.startswith(("Counting", "Dropping")):
            return [0]
        return []

    def writes(self) -> list[str]:
        return [op for op, _ in self.requests if not op.startswith("Reading")]


def _on(indexer: Any, graph: _Graph) -> _Graph:
    indexer._execute_with_retries = graph.execute
    return graph


# --- the id format -----------------------------------------------------------


def test_a_custom_vertex_id_is_the_label_and_the_item_id() -> None:
    assert vertex_id("Entity-t1-v2", "abc") == "Entity-t1-v2:abc"
    # One entity id under two suffixes: two vertex ids.
    assert vertex_id("Entity-t1", "abc") != vertex_id("Entity-t2", "abc")


def test_the_sample_vertex_tells_the_mode() -> None:
    assert stored_vertex_id_mode(_LABEL, []) is None
    assert (
        stored_vertex_id_mode(_LABEL, [{"vertex_id": f"{_LABEL}:e1", "id": ["e1"]}])
        == "custom"
    )
    # Neptune's UUIDs, TinkerGraph's numbers, another label's custom id.
    for vid in ("4f1e-uuid", 7, "Entity-other:e1"):
        assert (
            stored_vertex_id_mode(_LABEL, [{"vertex_id": vid, "id": ["e1"]}])
            == "property"
        )


# --- the traversals of each mode ---------------------------------------------


def test_custom_mode_upserts_a_vertex_by_its_vertex_id(make_indexer) -> None:
    entity = Entity(id="e1", name="Vendor")
    custom = _groovy(
        make_indexer("custom")._entity_vertex_builder(label=_LABEL)(
            Graph().traversal(), [entity]
        )
    )
    assert custom.startswith(
        "g.V('Entity-default:e1').fold().coalesce(__.unfold(),"
        "__.addV('Entity-default').property(T.id,'Entity-default:e1')"
        ".property('id','e1'))"
    ), custom
    assert "has('Entity-default','id'" not in custom

    default = _groovy(
        make_indexer("property")._entity_vertex_builder(label=_LABEL)(
            Graph().traversal(), [entity]
        )
    )
    assert default.startswith(
        "g.V().has('Entity-default','id','e1').fold().coalesce(__.unfold(),"
        "__.addV('Entity-default').property('id','e1'))"
    ), default
    assert "T.id" not in default


def test_custom_mode_finds_the_edge_endpoints_by_vertex_id(make_indexer) -> None:
    rel = Relationship(id="r1", source_id="a", target_id="b")
    custom = _groovy(
        make_indexer("custom")._build_add_edge_traversal(
            Graph().traversal(), rel, _LABEL
        )
    )
    # The edge itself keeps its id property (and Neptune's own edge id).
    assert custom.startswith(
        "g.E().has('id','r1').where(__.outV().hasLabel('Entity-default'))"
        ".fold().as('stored').V('Entity-default:a').as('source')"
        ".V('Entity-default:b').as('target')"
    ), custom
    assert ".property('id','r1')" in custom and custom.endswith(".id()")

    default = _groovy(
        make_indexer("property")._build_add_edge_traversal(
            Graph().traversal(), rel, _LABEL
        )
    )
    assert (
        ".V().hasLabel('Entity-default').has('id','a').as('source')"
        ".V().hasLabel('Entity-default').has('id','b').as('target')" in default
    ), default


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "custom",
            "g.V('Entity-default:e1','Entity-default:e2',"
            "'Community-default:e1','Community-default:e2').drop()",
        ),
        (
            "property",
            "g.V().hasLabel('Entity-default','Community-default')"
            ".has('id',within(['e1','e2'])).drop()",
        ),
    ],
)
def test_delete_by_id_drops_the_vertices_of_both_labels(
    make_indexer, mode: str, expected: str
) -> None:
    indexer = make_indexer(mode)
    graph = _on(indexer, _Graph(stored=mode))
    assert indexer.delete_by_id(["e1", "e2"], suffix="default").failed_items == 0
    assert ("Deleting vertices by id", expected) in graph.requests


@pytest.mark.parametrize(
    ("read", "operation"),
    [
        (lambda i: i.read_entities(["e1"]), "read_entities"),
        (lambda i: i.read_neighbor_ids(["e1"]), "read_neighbor_ids"),
        (lambda i: i.count_neighbors(["e1"]), "count_neighbors"),
        (
            lambda i: i.find_incident_relationship_ids(["e1"]),
            "find_incident_relationship_ids",
        ),
    ],
)
def test_reads_by_id_look_vertices_up_by_vertex_id(
    make_indexer, read: Callable, operation: str
) -> None:
    indexer = make_indexer("custom")
    graph = _on(indexer, _Graph(stored="custom"))
    read(indexer)
    (sent,) = [groovy for op, groovy in graph.requests if op == operation]
    assert sent.startswith("g.V('Entity-default:e1').")
    assert "has('id'" not in sent


def _retriever(mode: str) -> Any:
    from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever

    config = _config(mode)
    inst = NeptuneRetriever.__new__(NeptuneRetriever)
    for name, value in {
        "_config": config,
        "_neptune_config": config.indexing.neptune,
        "_max_hops": 2,
        "_max_results_per_hop": 5,
        "_checked_labels": set(),
    }.items():
        object.__setattr__(inst, name, value)
    return inst


def _run_retriever(retriever: Any, answer: Callable[[str], list[Any]]) -> list[str]:
    """Run one expansion from seeds e1, e2; the Groovy of each request."""
    sent: list[str] = []

    async def execute(build: Callable) -> list[Any]:
        groovy = _groovy(build(Graph().traversal()))
        sent.append(groovy)
        return answer(groovy)

    projected: list[dict[str, Any]] = retriever.__dict__.setdefault("projected", [])

    def project(traversal: Any, **kwargs: Any) -> Any:
        projected.append(kwargs)
        return traversal

    object.__setattr__(retriever, "_with_projection", project)
    object.__setattr__(retriever, "_execute_traversal", execute)
    asyncio.run(
        retriever._traverse(["e1", "e2"], SearchQuery(query="", suffix="default"))
    )
    return sent


def _sample(mode: str | None) -> Callable[[str], list[Any]]:
    def answer(groovy: str) -> list[Any]:
        if ".limit(1).project('vertex_id','id')" in groovy and mode is not None:
            vid = vertex_id(_LABEL, "e1") if mode == "custom" else 12
            return [{"vertex_id": vid, "id": ["e1"]}]
        return []

    return answer


def test_custom_mode_seeds_the_expansion_by_vertex_id() -> None:
    sent = _run_retriever(_retriever("custom"), _sample("custom"))
    expansion = sent[-1]
    assert expansion.startswith(
        "g.V('Entity-default:e1','Entity-default:e2').union(__.identity(),"
    ), expansion
    # The walk keeps filtering the seeds out by their id property.
    assert "has('id',without(['e1','e2']))" in expansion

    default = _run_retriever(_retriever("property"), _sample("property"))[-1]
    assert default.startswith(
        "g.V().hasLabel('Entity-default').has('id',within(['e1','e2']))"
    ), default


# --- the mode check: the indexer ---------------------------------------------


@pytest.mark.parametrize(
    ("configured", "stored"), [("custom", "property"), ("property", "custom")]
)
def test_a_label_of_the_other_mode_is_refused_before_any_write(
    make_indexer, configured: str, stored: str
) -> None:
    indexer = make_indexer(configured)
    graph = _on(indexer, _Graph(stored=stored))
    with pytest.raises(StoreConfigurationError) as raised:
        indexer.upsert_entities([Entity(id="e1", name="Vendor")])
    message = str(raised.value)
    assert f"vertex_ids: {stored}" in message
    assert "indexing.reset: true" in message
    assert graph.writes() == []


@pytest.mark.parametrize(
    "call",
    [
        lambda i: i.read_entities(["e1"]),
        lambda i: i.read_relationships(["r1"]),
        lambda i: i.read_entity_names(),
        lambda i: i.find_ids_citing(["t1"]),
        lambda i: i.find_relationship_ids_citing(["t1"]),
        lambda i: i.find_entity_ids_by_name_key(["vendor"]),
        lambda i: i.read_neighbor_ids(["e1"]),
        lambda i: i.count_neighbors(["e1"]),
        lambda i: i.find_incident_relationship_ids(["e1"]),
        lambda i: i.delete_by_id(["e1"]),
        lambda i: i.upsert_relationships(
            [Relationship(id="r1", source_id="a", target_id="b")]
        ),
        lambda i: i.upsert_communities(
            [
                Community(
                    id="c1",
                    name="c1",
                    level="0",
                    parent="-1",
                    children=[],
                    entity_ids=[],
                )
            ]
        ),
        lambda i: i.upsert_entities([Entity(id="e1", name="Vendor")]),
    ],
)
def test_every_read_and_write_checks_the_mode(make_indexer, call: Callable) -> None:
    indexer = make_indexer("custom")
    _on(indexer, _Graph(stored="property"))
    with pytest.raises(StoreConfigurationError):
        call(indexer)


def test_an_empty_or_matching_label_is_checked_once(make_indexer) -> None:
    indexer = make_indexer("custom")
    graph = _on(indexer, _Graph(stored=None))
    indexer.upsert_entities([Entity(id="e1", name="Vendor")])
    indexer.upsert_entities([Entity(id="e2", name="Buyer")])
    checks = [op for op, _ in graph.requests if op.startswith("Reading")]
    assert checks == [
        f"Reading the vertex-id switch of '{_LABEL}'",
        f"Reading the vertex ids of '{_LABEL}'",
    ]


def test_an_unfinished_switch_is_refused_in_either_mode(make_indexer) -> None:
    for mode in ("custom", "property"):
        indexer = make_indexer(mode)
        _on(indexer, _Graph(stored=mode, switch="custom"))
        with pytest.raises(StoreConfigurationError, match="did not finish"):
            indexer.read_entities(["e1"])


def test_a_full_build_switches_the_label_and_its_sweep_ends_the_switch(
    make_indexer, caplog
) -> None:
    caplog.set_level(logging.WARNING)
    indexer = make_indexer("custom")
    graph = _on(indexer, _Graph(stored="property"))
    assert indexer.begin_rebuild(["default"])
    indexer.index_entities([Entity(id="e1", name="Vendor")], suffixes=["default"])
    # Queries go on reading the label whole meanwhile (both copies).
    assert "queries read old and rewritten vertices until it ends" in caplog.text
    # Recorded before the build's first write to the label.
    ops = [op for op, _ in graph.requests]
    assert ops.index(f"Recording the vertex-id switch of '{_LABEL}'") < ops.index(
        "Entity indexing"
    )
    assert graph.switch == "custom"
    graph.stored = "custom"  # what the sweep leaves
    indexer.end_rebuild(["default"])
    assert graph.switch is None
    assert f"Dropping the vertex-id switch of 'Community-default', '{_LABEL}'" in [
        op for op, _ in graph.requests
    ]
    # And the label is usable at once.
    indexer.upsert_entities([Entity(id="e2", name="Buyer")])


def test_a_full_build_without_its_sweep_leaves_the_label_refused(
    make_indexer,
) -> None:
    indexer = make_indexer("custom")
    graph = _on(indexer, _Graph(stored="property"))
    indexer.begin_rebuild(["default"])
    indexer.index_entities([Entity(id="e1", name="Vendor")], suffixes=["default"])
    indexer.end_rebuild(["default"], sweep=False)
    assert graph.switch == "custom"
    # Same process: checked again, and refused.
    with pytest.raises(StoreConfigurationError, match="did not finish"):
        indexer.upsert_entities([Entity(id="e2", name="Buyer")])
    # A later full build finishes it.
    indexer.begin_rebuild(["default"])
    indexer.index_entities([Entity(id="e1", name="Vendor")], suffixes=["default"])
    indexer.end_rebuild(["default"])
    assert graph.switch is None


def test_a_full_build_in_the_labels_own_mode_records_no_switch(make_indexer) -> None:
    indexer = make_indexer("custom")
    graph = _on(indexer, _Graph(stored="custom"))
    indexer.begin_rebuild(["default"])
    indexer.index_entities([Entity(id="e1", name="Vendor")], suffixes=["default"])
    indexer.end_rebuild(["default"])
    assert not [op for op, _ in graph.requests if op.startswith("Recording")]


def test_a_full_build_finishes_a_switch_whose_sample_is_rewritten(
    make_indexer,
) -> None:
    # A rerun of an unswept switch whose sample vertex is one the earlier
    # build already rewrote: the switch is still recorded and swept.
    indexer = make_indexer("custom")
    graph = _on(indexer, _Graph(stored="custom", switch="custom"))
    indexer.begin_rebuild(["default"])
    indexer.index_entities([Entity(id="e1", name="Vendor")], suffixes=["default"])
    assert _LABEL in indexer._switching_labels
    indexer.end_rebuild(["default"])
    assert graph.switch is None


class _Labels(_Graph):
    """``_Graph`` with a switch vertex per label (``switches``)."""

    def __init__(self, stored: str | None, switches: dict[str, str]) -> None:
        super().__init__(stored=stored)
        self.switches = dict(switches)

    def execute(self, build: Callable, operation: str, **kwargs: Any) -> list[Any]:
        groovy = _groovy(build(Graph().traversal()))
        if operation.startswith("Reading the vertex-id switch"):
            self.requests.append((operation, groovy))
            label = groovy.split("has('id','")[1].split("'")[0]
            return [self.switches[label]] if label in self.switches else []
        if operation.startswith("Recording the vertex-id switch"):
            self.requests.append((operation, groovy))
            label = groovy.split("has('id','")[1].split("'")[0]
            self.switches[label] = "custom"
            return []
        if operation.startswith("Dropping the vertex-id switch"):
            self.requests.append((operation, groovy))
            labels = groovy.split("within([")[1].split("])")[0]
            for label in [x.strip("'") for x in labels.split(",")]:
                self.switches.pop(label, None)
            return []
        return super().execute(build, operation, **kwargs)


def test_a_full_build_ends_the_switch_of_a_label_it_never_used(
    make_indexer,
) -> None:
    # An earlier build left the community label switching; this one writes
    # entities only. Its sweep empties the community label too, and the
    # label's switch vertex goes with it: left behind, it refused every
    # later run on the label ("did not finish").
    community = "Community-default"
    indexer = make_indexer("custom")
    graph = _on(indexer, _Labels(stored="custom", switches={community: "custom"}))
    indexer.begin_rebuild(["default"])
    indexer.index_entities([Entity(id="e1", name="Vendor")], suffixes=["default"])
    indexer.end_rebuild(["default"])
    assert graph.switches == {}
    graph.stored = None
    assert indexer.find_ids_citing(["t1"], communities=True) == {}
    # Unswept, it stays.
    graph.switches = {community: "custom"}
    indexer.begin_rebuild(["default"])
    indexer.end_rebuild(["default"], sweep=False)
    assert graph.switches == {community: "custom"}


def test_custom_mode_looks_each_id_up_once(make_indexer) -> None:
    indexer = make_indexer("custom")
    graph = _on(indexer, _Graph(stored="custom"))
    indexer.read_entities(["e1", "e2", "e1"])
    (sent,) = [groovy for op, groovy in graph.requests if op == "read_entities"]
    assert sent.startswith("g.V('Entity-default:e1','Entity-default:e2').")


def test_a_graph_store_without_string_ids_names_the_setting(
    make_indexer, mocker
) -> None:
    # TinkerGraph's default id manager takes Long ids only: every custom-id
    # lookup failed with a raw 500, retried graph_store_max_attempts times.
    from gremlin_python.driver.protocol import GremlinServerError

    from unified_kg_rag.adapters.aws.neptune import neptune_request_error

    refused = GremlinServerError(
        {
            "code": 500,
            "message": "Expected an id that is convertible to class "
            "java.lang.Long but received class java.lang.String - "
            "[Entity-default:e1]",
            "attributes": {},
        }
    )
    error = neptune_request_error(refused, None)
    assert isinstance(error, StoreConfigurationError)
    assert "vertexIdManager=ANY" in str(error)
    assert "vertex_ids: property" in str(error)

    indexer = make_indexer("custom")
    calls: list[str] = []

    def run(client: Any, build: Callable, **kwargs: Any) -> list[Any]:
        calls.append("sent")
        raise error

    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer.run_traversal", run)
    with pytest.raises(StoreConfigurationError):
        indexer._execute_with_retries(lambda g: g.V(), "Entity indexing")
    # Not retried: it fails the same way every time.
    assert calls == ["sent"]


def test_clearing_a_label_drops_its_switch(make_indexer) -> None:
    indexer = make_indexer("custom")
    graph = _on(indexer, _Graph(stored="property", switch="custom"))
    indexer.clear(["default"])
    assert graph.switch is None
    graph.stored = None
    indexer.upsert_entities([Entity(id="e1", name="Vendor")])


def test_the_switch_vertex_is_found_by_its_label_and_id_property(
    make_indexer,
) -> None:
    indexer = make_indexer("custom")
    graph = _on(indexer, _Graph(stored=None))
    indexer.read_entities(["e1"])
    (sent,) = [
        groovy
        for op, groovy in graph.requests
        if op.startswith("Reading the vertex-id")
    ]
    assert sent == (
        f"g.V().hasLabel('{VERTEX_ID_SWITCH_LABEL}').has('id','{_LABEL}')"
        ".values('vertex_ids')"
    )


# --- the mode check: the retriever -------------------------------------------


@pytest.fixture
def warned(monkeypatch) -> None:
    """Every once-per-process warning logged again."""
    from unified_kg_rag.shared import logging as shared_logging

    monkeypatch.setattr(shared_logging, "_warned_once", set())


_BY_PROPERTY = "g.V().hasLabel('Entity-default').has('id',within(['e1','e2']))"


@pytest.mark.parametrize(
    ("configured", "stored"), [("custom", "property"), ("property", "custom")]
)
def test_the_retriever_reads_a_label_of_the_other_mode_by_property(
    caplog, warned, configured: str, stored: str
) -> None:
    # The query service switched before the full build's first write (or
    # the build switched and the query service has not): a refusal failed
    # every graph expansion until then. The id property is in both modes,
    # so the expansion looks the seeds up by it, and warns once.
    caplog.set_level(logging.WARNING)
    retriever = _retriever(configured)
    for _ in range(2):
        sent = _run_retriever(retriever, _sample(stored))
        assert sent[-1].startswith(_BY_PROPERTY), sent[-1]
    warnings = [r for r in caplog.records if "vertex_ids" in r.getMessage()]
    assert len(warnings) == 1, warnings
    assert f"built with vertex_ids '{stored}'" in warnings[0].getMessage()
    # Checked again on each query: once a build rewrites the label, the
    # configured lookups take over.
    sent = _run_retriever(retriever, _sample(configured))
    if configured == "custom":
        assert sent[-1].startswith("g.V('Entity-default:e1','Entity-default:e2')")
    else:
        assert sent[-1].startswith(_BY_PROPERTY)


@pytest.mark.parametrize("configured", ["custom", "property"])
@pytest.mark.parametrize("target", ["custom", "property"])
def test_the_retriever_reads_a_label_being_switched_by_property(
    caplog, warned, configured: str, target: str
) -> None:
    # The sample may hit a vertex of either mode: a label being switched
    # holds both, and only the property lookup finds them all.
    caplog.set_level(logging.WARNING)
    for sampled in ("custom", "property"):

        def answer(groovy: str, sampled: str = sampled) -> list[Any]:
            if f"hasLabel('{VERTEX_ID_SWITCH_LABEL}')" in groovy:
                return [target]
            return _sample(sampled)(groovy)

        retriever = _retriever(configured)
        sent = _run_retriever(retriever, answer)
        assert sent[-1].startswith(_BY_PROPERTY), sent[-1]
        # With each vertex's id, to tell the rewritten copy from the old.
        assert retriever.projected[-1]["with_vertex_id"] is True
        assert "being rewritten" in caplog.text


def _row(vid: Any, item_id: str, path: int, name: str = "") -> dict[str, Any]:
    return {
        "node": {"id": [item_id], "name": [name or item_id]},
        "path": [{"name": [f"p{i}"]} for i in range(path)],
        "node_type": _LABEL,
        "vertex_id": vid,
    }


@pytest.mark.parametrize("target", ["custom", "property"])
def test_a_switch_keeps_the_better_scored_copy_of_a_vertex(target: str) -> None:
    from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever

    def vid(item_id: str, rewritten: bool) -> Any:
        custom = rewritten == (target == "custom")
        return vertex_id(_LABEL, item_id) if custom else f"uuid-{item_id}"

    rows = [
        _row(vid("e1", False), "e1", 1, "old"),
        _row(vid("e1", True), "e1", 1, "rewritten"),
        # Reached over the old copy only: kept.
        _row(vid("e2", False), "e2", 2),
        # Reached over both, nearer over the old copy's edges (not
        # rewritten yet): the old copy.
        _row(vid("e3", True), "e3", 3, "rewritten"),
        _row(vid("e3", False), "e3", 2, "old"),
        # Nearer over the rewritten copy: the rewritten copy.
        _row(vid("e4", False), "e4", 3, "old"),
        _row(vid("e4", True), "e4", 2, "rewritten"),
    ]
    marked = NeptuneRetriever._mark_old_copies(rows, _LABEL, target)
    assert all("vertex_id" not in r for r in marked)
    assert [(r["node"]["name"][0], r.get("superseded", False)) for r in marked] == [
        ("old", True),
        ("rewritten", False),
        ("e2", False),
        ("rewritten", False),
        ("old", True),
        ("old", True),
        ("rewritten", False),
    ]
    config = _config(target)
    retriever = _retriever(target)
    object.__setattr__(retriever, "_neptune_config", config.indexing.neptune)
    results = retriever._process_traversal_results(
        marked, SearchQuery(query="", filters={"id": ["e1"]})
    )
    names = {r.source: r.metadata["name"] for r in results}
    # Two copies scored alike: the rewritten one (not the first by content).
    assert names == {"e1": "rewritten", "e2": "e2", "e3": "old", "e4": "rewritten"}


def test_a_rewritten_copy_beats_a_higher_ranked_old_copy_as_near() -> None:
    from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever

    # The old copy keeps its rank (or degree) until the sweep: as near, it
    # can score higher, but the rewritten copy is the one read.
    def ranked(vid: Any, item_id: str, path: int, name: str, rank: int) -> Any:
        row = _row(vid, item_id, path, name)
        row["node"]["rank"] = [rank]
        return row

    rows = [
        ranked(vertex_id(_LABEL, "e1"), "e1", 1, "seed", 1),
        ranked("uuid-e2", "e2", 2, "old", 9),
        ranked(vertex_id(_LABEL, "e2"), "e2", 2, "new", 1),
        # A nearer old copy still wins over a farther rewritten one.
        ranked("uuid-e3", "e3", 2, "near", 1),
        ranked(vertex_id(_LABEL, "e3"), "e3", 3, "far", 9),
    ]
    marked = NeptuneRetriever._mark_old_copies(rows, _LABEL, "custom")
    for ordered in (marked, marked[::-1]):
        results = _retriever("custom")._process_traversal_results(
            ordered, SearchQuery(query="", filters={"id": ["e1"]})
        )
        names = {r.source: r.metadata["name"] for r in results}
        assert names == {"e1": "seed", "e2": "new", "e3": "near"}, names


def _expanded(sample: Callable[[str], list[Any]]) -> Callable[[str], list[Any]]:
    """``sample``, with a row for the expansion (the seeds were found)."""

    def answer(groovy: str) -> list[Any]:
        if ".union(" in groovy:
            return [_row(None, "e1", 1)]
        return sample(groovy)

    return answer


def _switching(target: str, sample: Callable[[str], list[Any]]) -> Callable:
    def answer(groovy: str) -> list[Any]:
        if f"hasLabel('{VERTEX_ID_SWITCH_LABEL}')" in groovy:
            return [target]
        return sample(groovy)

    return answer


def test_the_retriever_samples_a_matching_label_once() -> None:
    retriever = _retriever("custom")
    first = _run_retriever(retriever, _expanded(_sample("custom")))
    second = _run_retriever(retriever, _expanded(_sample("custom")))
    # The sample and the switch, then the expansion; then the switch alone
    # (one indexed read) before each custom-id expansion.
    assert len(first) == 3
    assert len(second) == 2
    assert f"hasLabel('{VERTEX_ID_SWITCH_LABEL}')" in second[0]
    assert second[1].startswith("g.V('Entity-default:e1','Entity-default:e2')")


@pytest.mark.parametrize("target", ["custom", "property"])
def test_a_settled_custom_label_follows_a_switch_and_its_end(
    warned, target: str
) -> None:
    # One retriever across a full build switching a label it found settled:
    # the build's switch vertex is seen on the next query, and once the
    # sweep has dropped it the label is sampled again.
    retriever = _retriever("custom")
    _run_retriever(retriever, _expanded(_sample("custom")))
    sent = _run_retriever(retriever, _expanded(_switching(target, _sample(target))))
    assert sent[-1].startswith(_BY_PROPERTY), sent[-1]
    assert retriever.projected[-1]["with_vertex_id"] is True
    sent = _run_retriever(retriever, _expanded(_sample(target)))
    assert any(".limit(1).project('vertex_id','id')" in groovy for groovy in sent)
    if target == "custom":
        assert sent[-1].startswith("g.V('Entity-default:e1','Entity-default:e2')")
    else:
        assert sent[-1].startswith(_BY_PROPERTY), sent[-1]


def test_a_settled_property_label_is_not_read_again(warned) -> None:
    # Property lookups find the vertices of both modes, so a label settled
    # in property mode costs the expansion alone on each later query, during
    # a switch to custom ids and after it. Each vertex id is read along: a
    # switch can only be to custom ids, whose copies it tells apart.
    retriever = _retriever("property")
    first = _run_retriever(retriever, _expanded(_sample("property")))
    assert len(first) == 3, first
    for answer in (
        _expanded(_sample("property")),
        _expanded(_switching("custom", _sample("custom"))),
        _expanded(_sample("custom")),
        _sample("property"),
    ):
        sent = _run_retriever(retriever, answer)
        assert len(sent) == 1 and sent[0].startswith(_BY_PROPERTY), sent
        assert retriever.projected[-1]["with_vertex_id"] is True
    rows = [
        _row("uuid-e1", "e1", 1, "seed"),
        _row("uuid-e2", "e2", 2, "old"),
        _row(vertex_id(_LABEL, "e2"), "e2", 2, "rewritten"),
    ]

    async def execute(build: Callable) -> list[Any]:
        return rows if ".union(" in _groovy(build(Graph().traversal())) else []

    object.__setattr__(retriever, "_execute_traversal", execute)
    for ordered in (rows, rows[::-1]):
        rows = ordered
        hits = asyncio.run(
            retriever.aretrieve(SearchQuery(query="", filters={"id": ["e1"]}))
        )
        assert {h.source: h.metadata["name"] for h in hits} == {
            "e1": "seed",
            "e2": "rewritten",
        }


def test_a_settled_custom_label_found_without_its_seeds_is_sampled_again(
    warned,
) -> None:
    # A full build switched the label to property ids and swept between two
    # queries (no switch vertex left to see): the custom-id lookup finds no
    # seed, so the label is sampled again and the expansion re-run by the id
    # property.
    retriever = _retriever("custom")
    _run_retriever(retriever, _expanded(_sample("custom")))

    def swept(groovy: str) -> list[Any]:
        if groovy.startswith("g.V('Entity-default:"):
            return []
        return _expanded(_sample("property"))(groovy)

    sent = _run_retriever(retriever, swept)
    assert len(sent) == 5, sent
    assert sent[1].startswith("g.V('Entity-default:e1','Entity-default:e2')")
    assert sent[-1].startswith(_BY_PROPERTY), sent[-1]
    assert "Entity-default" not in retriever._checked_labels
    # Still custom, the seeds just absent: sampled again, not re-expanded.
    retriever = _retriever("custom")
    _run_retriever(retriever, _expanded(_sample("custom")))
    sent = _run_retriever(retriever, _sample("custom"))
    assert len(sent) == 4, sent
    assert sent[-1].startswith(f"g.V().hasLabel('{VERTEX_ID_SWITCH_LABEL}')")
    assert "Entity-default" in retriever._checked_labels


@pytest.mark.parametrize(
    ("switch", "text"), [(None, "built with"), ("custom", "being rewritten")]
)
def test_each_label_warns_once(caplog, warned, switch: str | None, text: str) -> None:
    # One WARNING per label (and mode or target), however often it is read:
    # two labels warn once each.
    caplog.set_level(logging.WARNING)
    retriever = _retriever("custom")
    sample = _sample("property")
    answer = _switching(switch, sample) if switch else sample

    async def execute(build: Callable) -> list[Any]:
        return answer(_groovy(build(Graph().traversal())))

    object.__setattr__(retriever, "_execute_traversal", execute)
    for label in ("Entity-a", "Entity-b", "Entity-a"):
        asyncio.run(retriever._vertex_lookup(label))
    messages = [r.getMessage() for r in caplog.records if text in r.getMessage()]
    assert len(messages) == 2, messages
    for label in ("Entity-a", "Entity-b"):
        assert sum(label in m for m in messages) == 1, messages


def test_the_retriever_checks_an_empty_label_again() -> None:
    retriever = _retriever("custom")
    _run_retriever(retriever, _sample(None))
    sent = _run_retriever(retriever, _sample("property"))
    assert sent[-1].startswith(_BY_PROPERTY)


def test_the_local_graph_store_takes_the_setting_and_ignores_it(tmp_path) -> None:
    from unified_kg_rag.adapters.storage.sqlite_graph_indexer import (
        SQLiteGraphIndexer,
    )

    config = _config("custom")
    config.providers.graph.kind = "sqlite"
    config.providers.graph.sqlite_path = str(tmp_path / "graph.sqlite3")
    indexer = SQLiteGraphIndexer(config)
    try:
        assert (
            indexer.upsert_entities([Entity(id="e1", name="Vendor")]).failed_items == 0
        )
        assert [e.id for e in indexer.read_entities(["e1"])] == ["e1"]
    finally:
        indexer.close()


def test_a_switch_to_property_ids_rewrites_no_custom_vertex_in_place(
    make_indexer,
) -> None:
    # A custom-id vertex has the id property too: matched by it, it would be
    # updated in place and keep its custom vertex id (the switch would never
    # happen). Only the vertices the build wrote are matched.
    indexer = make_indexer("property")
    graph = _on(indexer, _Graph(stored="custom"))
    indexer.begin_rebuild(["default"])
    tag = indexer._rebuild_tag
    indexer.index_entities([Entity(id="e1", name="Vendor")], suffixes=["default"])
    (written,) = [g for op, g in graph.requests if op == "Entity indexing"]
    assert written.startswith(
        f"g.V().has('{_LABEL}','id','e1').has('build_tag','{tag}').fold()"
    ), written
    edge = _groovy(
        indexer._build_add_edge_traversal(
            Graph().traversal(),
            Relationship(id="r1", source_id="a", target_id="b"),
            _LABEL,
        )
    )
    assert (
        f".has('id','a').has('build_tag','{tag}').as('source')" in edge
        and f".has('id','b').has('build_tag','{tag}').as('target')" in edge
    ), edge
    # Outside a switch the build rewrites the stored vertices in place.
    other = make_indexer("property")
    graph = _on(other, _Graph(stored="property"))
    other.begin_rebuild(["default"])
    other.index_entities([Entity(id="e1", name="Vendor")], suffixes=["default"])
    (written,) = [g for op, g in graph.requests if op == "Entity indexing"]
    assert "has('build_tag'" not in written.split(".fold()")[0]


def test_graph_expansion_seeds_each_id_once() -> None:
    # g.V(a, a) looks a custom vertex id up, and expands it, twice.
    from unified_kg_rag.adapters.retrievers.graph_expansion import (
        GraphExpansionRetriever,
    )

    query = SearchQuery(query="", filters={"id": ["e2", "e1", "e2"]})
    assert GraphExpansionRetriever._seed_ids(query) == ["e2", "e1"]
