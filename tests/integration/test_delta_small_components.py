# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A delta's small components get a community when a full build's would.

An incremental run clusters only its delta. A component of fewer than
``min_community_size`` entities gets no community (#347), so a new entity
whose one relationship is to a stored entity, a pair in the delta graph,
got none, where a full build clusters it with its stored neighbour. The
component is measured in the whole graph a full build clusters: the stored
graph without what the run replaces or removes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.adapters.storage.sqlite_graph_indexer import SQLiteGraphIndexer
from unified_kg_rag.domain.models import PipelineStageStatus

pytestmark = pytest.mark.integration


# Every case runs under the default suffix and another one: the stored graph
# is read under the suffix of the delta's entities.
SUFFIXES = pytest.mark.parametrize("suffix", [None, "tenant-a"])


def _setup(tmp_path: Path, suffix: str | None):
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    config.processing.document_parsing.index_value = suffix
    server = FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM)
    providers = e2e._providers(
        config, server, e2e.FakeCrossEncoder("synthetic-reranker")
    )
    return config, providers


def _ingest_all(tmp_path: Path, runs: list[dict[str, str]], suffix: str | None):
    """Ingest each corpus of ``runs`` in turn into one set of stores."""
    config, providers = _setup(tmp_path, suffix)
    source = tmp_path / "corpus"
    for n, corpus in enumerate(runs):
        source.mkdir(parents=True, exist_ok=True)
        for old in source.iterdir():
            old.unlink()
        for name, text in corpus.items():
            (source / name).write_text(text)
        context = e2e._ingest(config, providers, source, tmp_path / f"run-{n}")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    return providers, context


def _unclustered(providers, suffix: str | None) -> set[str]:
    graph = providers.build_graph_indexer()
    try:
        assert isinstance(graph, SQLiteGraphIndexer)
        ids = [i for i, _ in graph.read_entity_names(suffix=suffix)]
        entities = graph.read_entities(ids, suffix=suffix)
        assert entities
        return {e.name for e in entities if not e.community_ids}
    finally:
        graph.close()


def _communities(providers, name: str, suffix: str | None = None) -> list[str]:
    graph = providers.build_graph_indexer()
    try:
        assert isinstance(graph, SQLiteGraphIndexer)
        ids = [i for i, n in graph.read_entity_names(suffix=suffix) if n == name]
        assert ids
        entities = graph.read_entities(ids, suffix=suffix)
        return [c for e in entities for c in e.community_ids or []]
    finally:
        graph.close()


@SUFFIXES
def test_a_new_entity_linked_only_to_a_stored_one_gets_a_community(
    tmp_path: Path, no_aws, suffix: str | None
) -> None:
    # The delta graph is the pair Carrier - Vendor; Vendor is stored.
    stored = {
        "contract.txt": "Vendor supplies widgets to Buyer.",
        "logistics.txt": "Depot stores widgets for Vendor.",
    }
    providers, context = _ingest_all(
        tmp_path,
        [stored, {**stored, "freight.txt": "Carrier moves widgets for Vendor."}],
        suffix,
    )

    assert len(context.incremental_delta.new) == 1
    assert _communities(providers, "Carrier", suffix)


_PAIR = "Depot and Carrier trade."


@pytest.mark.parametrize(
    "runs",
    [
        # A singleton renamed: the stored entity is only the deleted file's.
        [
            {"a.txt": "Vendor trades.", "x.txt": "Buyer and Depot and Carrier."},
            {"r.txt": "Vendor trades.", "x.txt": "Buyer and Depot and Carrier."},
        ],
        # A stored singleton gains a neighbour: a pair in the whole graph too.
        [
            {"a.txt": "Vendor trades.", "x.txt": _PAIR},
            {"a.txt": "Vendor trades.", "b.txt": "Vendor and Buyer.", "x.txt": _PAIR},
        ],
        # The stored edge to a third entity is only the edited file's.
        [
            {"u.txt": "Vendor trades.", "g.txt": _PAIR, "c.txt": "Vendor and Depot."},
            {"u.txt": "Vendor trades.", "g.txt": _PAIR, "c.txt": "Vendor and Buyer."},
        ],
    ],
    ids=["renamed", "stored-singleton-pair", "edge-of-edited-file"],
)
@SUFFIXES
def test_a_small_delta_component_small_in_the_whole_graph_gets_none(
    tmp_path: Path, no_aws, runs: list[dict[str, str]], suffix: str | None
) -> None:
    # Old behaviour: any entity the stores held from another text unit, even
    # one of a deleted or edited file, clustered the component, and so did a
    # component that is a pair in the whole graph too.
    _, delta = _ingest_all(tmp_path / "incremental", runs, suffix)
    full, _ = _ingest_all(tmp_path / "full", runs[-1:], suffix)

    assert {e.name for e in delta.resolved_entities} <= _unclustered(full, suffix)
    assert not delta.communities


@SUFFIXES
def test_stored_entities_a_delta_links_without_a_community_are_reported(
    tmp_path: Path, no_aws, suffix: str | None, caplog
) -> None:
    # Vendor - Depot is stored, a pair without a community; the delta's
    # Depot - Carrier makes it a component of three, so the delta pair is
    # clustered, but Vendor, outside the delta, stays without one.
    caplog.set_level("INFO")
    providers, context = _ingest_all(
        tmp_path,
        [
            {"a.txt": "Vendor and Depot."},
            {"a.txt": "Vendor and Depot.", "b.txt": _PAIR},
        ],
        suffix,
    )

    assert _unclustered(providers, suffix) == {"Vendor"}
    metrics = {r.stage_name: r.metrics for r in context.stage_results}
    assert metrics["community_detection"]["unclustered_stored_entities"] == 1
    assert metrics["indexing"]["community_drift"]["unclustered_entities"] == 1
    # Reported at INFO: not a rebuild signal (before, a WARNING told the user
    # to rebuild while community_rebuild_recommended stayed 0).
    (record,) = [
        r for r in caplog.records if "without a community are" in str(r.getMessage())
    ]
    assert record.levelname == "INFO" and "Rebuild" not in str(record.getMessage())


def test_a_deleted_community_leaves_no_id_on_its_surviving_members(
    tmp_path: Path, no_aws
) -> None:
    # Old behaviour: Carrier kept the id of the delta community n.txt made,
    # deleted with n.txt, so the next run touching it counted two stale
    # earlier communities (stale_communities, overlap_share) instead of one.
    base = {"b.txt": "Buyer and Depot and Carrier."}
    providers, context = _ingest_all(
        tmp_path,
        [
            base,
            {**base, "n.txt": "Vendor and Carrier."},
            base,
            {**base, "m.txt": "Depot and Carrier."},
        ],
        None,
    )

    drift = {r.stage_name: r.metrics for r in context.stage_results}["indexing"]
    assert drift["community_drift"]["stale_communities"] == 1
    assert len(_communities(providers, "Carrier")) == 2


no_aws = e2e.no_aws


@pytest.mark.parametrize(
    "lone",
    [[], [{"lone-a", "lone-b"}]],
    ids=["hub-only", "with-a-component-that-cannot-reach-min-size"],
)
def test_the_walk_stops_at_min_size_on_a_hub(
    tmp_path: Path, mocker, lone: list[set[str]]
) -> None:
    # A delta component linked to a hub of 500 stored neighbours reaches
    # min_community_size within the hub's first edges: the walk reads a
    # batch of them, not the hub's whole neighbourhood, also when another
    # delta component that can never reach it is walked alongside (the walk
    # used to read on until every component had reached it).
    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.domain.models import Entity, Relationship

    config, providers = _setup(tmp_path, None)
    graph = providers.build_graph_indexer()
    try:
        assert isinstance(graph, SQLiteGraphIndexer)
        graph.initialize()
        hub = Entity(id="hub", name="Hub", text_unit_ids=["t-hub"])
        others = [
            Entity(id=f"o{i}", name=f"Org {i}", text_unit_ids=[f"t{i}"])
            for i in range(500)
        ]
        graph.upsert_entities([hub, *others])
        graph.upsert_relationships(
            [
                Relationship(
                    id=f"r{i}",
                    source_id="hub",
                    target_id=f"o{i}",
                    text_unit_ids=[f"t{i}"],
                )
                for i in range(500)
            ]
        )
        read = mocker.spy(graph, "read_relationships")

        found, _ = ps._components_reaching(graph, [{"hub", "new"}, *lone], {}, set(), 3)

        assert found == {"hub", "new"}
        assert sum(len(call.args[0]) for call in read.call_args_list) <= 64
    finally:
        graph.close()


def test_a_quarantined_documents_edges_do_not_reach_the_whole_graph(
    tmp_path: Path,
) -> None:
    # The only stored edge from the delta's entity to the rest of the graph
    # cites a unit of a quarantined document: a full build leaves that
    # document out, so the component stays small and gets no community.
    from datetime import datetime

    from tests.fixtures.fakes.doc_status import FakeDocStatusStore
    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.domain.models import (
        DocStatus,
        DocStatusRecord,
        DocumentDelta,
        Entity,
        PipelineContext,
        Relationship,
    )

    config, providers = _setup(tmp_path, None)
    graph = providers.build_graph_indexer()
    try:
        graph.initialize()
        graph.upsert_entities(
            [
                Entity(id=i, name=i.upper(), text_unit_ids=["t-ok"])
                for i in ("new", "s1", "s2", "s3")
            ]
        )
        graph.upsert_relationships(
            [
                Relationship(
                    id="r0", source_id="new", target_id="s1", text_unit_ids=["t-q"]
                ),
                Relationship(
                    id="r1", source_id="s1", target_id="s2", text_unit_ids=["t-ok"]
                ),
                Relationship(
                    id="r2", source_id="s2", target_id="s3", text_unit_ids=["t-ok"]
                ),
            ]
        )
    finally:
        graph.close()
    store = FakeDocStatusStore()
    store.put(
        DocStatusRecord(
            doc_id="q",
            content_hash="h",
            status=DocStatus.QUARANTINED,
            suffix="default",
            text_unit_ids=["t-q"],
        )
    )
    stage = ps.CommunityDetectionStage(config, providers=providers)
    stage._doc_status = store
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=str(tmp_path),
        resolved_entities=[Entity(id="new", name="NEW", text_unit_ids=["t-new"])],
        incremental_delta=DocumentDelta(new=["n"], unchanged=["u"], quarantined=["q"]),
    )

    large = stage._whole_graph(context)

    assert large is not None
    assert large([{"new"}]) == set()


@SUFFIXES
def test_a_delta_entity_linked_through_another_delta_component_gets_one(
    tmp_path: Path, no_aws, suffix: str | None
) -> None:
    # Vendor - Depot is stored; the delta states Vendor alone and Depot -
    # Carrier: the whole graph is the path Vendor - Depot - Carrier. The walk
    # from Vendor read only stored edges, found Vendor - Depot and stopped.
    stored = {"s.txt": "Vendor and Depot."}
    runs = [
        stored,
        {**stored, "a.txt": "Vendor trades.", "b.txt": "Depot and Carrier."},
    ]
    providers, _ = _ingest_all(tmp_path / "incremental", runs, suffix)
    full, _ = _ingest_all(tmp_path / "full", runs[-1:], suffix)

    assert not _unclustered(full, suffix)
    assert "Vendor" not in _unclustered(providers, suffix)


def test_the_walk_starts_from_another_spelling_of_a_delta_entity(
    tmp_path: Path,
) -> None:
    # "Acme Inc" - Globex is stored; the delta pair is "Acme Corp" - Buyer. A
    # full build merges the two spellings (same type): Globex, Buyer and the
    # one Acme are a component of three. The walk seeded by the delta ids
    # found nothing stored.
    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.domain.models import Entity, Relationship

    _, providers = _setup(tmp_path, None)
    graph = providers.build_graph_indexer()
    try:
        graph.initialize()
        graph.upsert_entities(
            [
                Entity(
                    id="e-inc",
                    name="Acme Inc",
                    type="ORGANIZATION",
                    text_unit_ids=["t1"],
                ),
                Entity(
                    id="e-globex",
                    name="Globex",
                    type="ORGANIZATION",
                    text_unit_ids=["t1"],
                ),
            ]
        )
        graph.upsert_relationships(
            [
                Relationship(
                    id="r",
                    source_id="e-inc",
                    target_id="e-globex",
                    text_unit_ids=["t1"],
                )
            ]
        )
        delta = [
            Entity(
                id="e-corp", name="Acme Corp", type="ORGANIZATION", text_unit_ids=["t2"]
            ),
            Entity(
                id="e-buyer", name="Buyer", type="ORGANIZATION", text_unit_ids=["t2"]
            ),
        ]
        component = {"e-corp", "e-buyer"}
        neighbors = {"e-corp": {"e-buyer"}, "e-buyer": {"e-corp"}}

        found, _ = ps._components_reaching(
            graph, [component], {}, set(), 3, neighbors, delta
        )
        apart = [e.model_copy(update={"type": "PERSON"}) for e in delta]
        kept_apart, _ = ps._components_reaching(
            graph, [component], {}, set(), 3, neighbors, apart
        )
    finally:
        graph.close()

    assert found == component
    assert kept_apart == set()


def _walk(tmp_path: Path, stored, edges, components, neighbors, delta, min_size):
    """``_components_reaching`` over a graph of ``stored`` and ``edges``."""
    from unified_kg_rag.application.ingestion import pipeline_stages as ps

    _, providers = _setup(tmp_path, None)
    graph = providers.build_graph_indexer()
    try:
        graph.initialize()
        graph.upsert_entities(stored)
        graph.upsert_relationships(edges)
        return ps._components_reaching(
            graph, components, {}, set(), min_size, neighbors, delta
        )
    finally:
        graph.close()


def _entity(entity_id: str, name: str, unit: str = "t1"):
    from unified_kg_rag.domain.models import Entity

    return Entity(id=entity_id, name=name, type="ORGANIZATION", text_unit_ids=[unit])


def _edge(edge_id: str, source: str, target: str):
    from unified_kg_rag.domain.models import Relationship

    return Relationship(
        id=edge_id, source_id=source, target_id=target, text_unit_ids=["t1"]
    )


def test_a_stored_spelling_of_a_delta_entity_counts_once(tmp_path: Path) -> None:
    # "Acme Inc" - Buyer is stored; the delta pair is "Acme Corp" - Buyer. A
    # full build merges the two spellings: the component is the one Acme and
    # Buyer, two entities, short of three (three if Acme counted under both
    # ids).
    found, _ = _walk(
        tmp_path,
        [_entity("e-inc", "Acme Inc"), _entity("e-buyer", "Buyer")],
        [_edge("r1", "e-inc", "e-buyer")],
        [{"e-corp", "e-buyer"}],
        {"e-corp": {"e-buyer"}, "e-buyer": {"e-corp"}},
        [_entity("e-corp", "Acme Corp", "t2"), _entity("e-buyer", "Buyer", "t2")],
        3,
    )

    assert found == set()


def test_the_walk_goes_on_from_a_stored_spelling_of_a_delta_entity(
    tmp_path: Path,
) -> None:
    # The delta has the pairs Xenon - Ypsilon and "Zeta Corp" - Wolf. Stored
    # are Ypsilon - "Zeta Inc" - Quux, and a full build merges the two Zetas:
    # one component of Xenon, Ypsilon, Zeta, Wolf and Quux. Reaching "Zeta
    # Inc" from Ypsilon, the walk goes on from the delta entity it is a
    # spelling of too, or it misses Wolf.
    found, _ = _walk(
        tmp_path,
        [
            _entity("y", "Ypsilon"),
            _entity("zs", "Zeta Inc"),
            _entity("q", "Quux"),
        ],
        [_edge("r1", "y", "zs"), _edge("r2", "zs", "q")],
        [{"x", "y"}, {"z", "w"}],
        {"x": {"y"}, "y": {"x"}, "z": {"w"}, "w": {"z"}},
        [
            _entity("x", "Xenon", "t2"),
            _entity("y", "Ypsilon", "t2"),
            _entity("z", "Zeta Corp", "t2"),
            _entity("w", "Wolf", "t2"),
        ],
        5,
    )

    assert found == {"x", "y", "z", "w"}
