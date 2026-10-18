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


def test_the_walk_stops_at_min_size_on_a_hub(tmp_path: Path, mocker) -> None:
    # A delta component linked to a hub of 500 stored neighbours reaches
    # min_community_size within the hub's first edges: the walk reads a
    # batch of them, not the hub's whole neighbourhood.
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

        found, _ = ps._components_reaching(graph, [{"hub", "new"}], {}, set(), 3)

        assert found == {"hub", "new"}
        assert sum(len(call.args[0]) for call in read.call_args_list) <= 64
    finally:
        graph.close()
