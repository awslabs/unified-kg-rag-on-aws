# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Document-backed ranks and evidenced community reports (seed graphs).

An edge is document-backed unless every one of its text units is a seed
unit. Ranks (entity degree, relationship combined degree, a report's rank
and its entity order) count document-backed edges only; Leiden still sees
every edge. A community no document evidences is stored without a report.
With no seed unit, every result is the one before seeds existed.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import networkx as nx
import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.ingestion.community_detector import CommunityDetector
from unified_kg_rag.adapters.search_strategies.local_search import LocalSearchStrategy
from unified_kg_rag.adapters.storage import neptune_codec
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.ingestion.graph_builder import GraphBuilder
from unified_kg_rag.domain.ingestion.graph_rank import with_graph_ranks
from unified_kg_rag.domain.ingestion.graph_resolver import GraphResolver
from unified_kg_rag.domain.ingestion.seed_graph import (
    SEED_ONLY_SOURCE_KINDS,
    SOURCE_KINDS_FIELD,
    annotate_sources,
    encoded_source_kinds,
    is_document_backed,
    seed_unit_id,
)
from unified_kg_rag.domain.models import (
    Community,
    CommunityReport,
    Config,
    DocumentLineage,
    Entity,
    PipelineContext,
    PipelineStageStatus,
    Relationship,
    RetrievalResult,
    TextUnit,
)
from unified_kg_rag.shared.exceptions import PipelineStageError

pytestmark = pytest.mark.unit

SEED = seed_unit_id("seed-doc", 0)
SEED_2 = seed_unit_id("seed-doc", 1)


def _rel(
    rid: str, source: str, target: str, units: list[str], rel_type: str = "REL"
) -> Relationship:
    return Relationship(
        id=rid, source_id=source, target_id=target, type=rel_type, text_unit_ids=units
    )


# --- the predicate and its stored encoding ----------------------------------


def test_an_edge_is_document_backed_unless_every_unit_is_a_seed_unit() -> None:
    assert is_document_backed(_rel("r", "a", "b", ["u1"]))
    assert is_document_backed(_rel("r", "a", "b", ["u1", SEED]))
    # An edge citing no unit is not seed-only.
    assert is_document_backed(_rel("r", "a", "b", []))
    assert not is_document_backed(_rel("r", "a", "b", [SEED, SEED_2]))


def test_the_seed_only_value_is_the_codec_encoding_of_a_seed_only_edge() -> None:
    seed_only = _rel("r", "a", "b", [SEED])
    kinds = annotate_sources(seed_only)
    assert kinds is not None
    assert SEED_ONLY_SOURCE_KINDS == encoded_source_kinds(kinds) == '["seed_graph"]'
    # The graph codec writes exactly this value on a seed-only edge (what the
    # document-backed neighbour reads skip).
    stored = seed_only.model_copy(update={"source_kinds": kinds})
    properties = neptune_codec.relationship_properties(stored, 10_000)
    assert properties[SOURCE_KINDS_FIELD] == SEED_ONLY_SOURCE_KINDS
    both = _rel("r", "a", "b", ["u1", SEED])
    both.source_kinds = annotate_sources(both)
    written = neptune_codec.relationship_properties(both, 10_000)
    assert written[SOURCE_KINDS_FIELD] == json.dumps(["document", "seed_graph"])
    # Never truncated: a cut value would no longer match the seed-only one.
    cut = neptune_codec.relationship_properties(stored, 5)
    assert cut[SOURCE_KINDS_FIELD] == SEED_ONLY_SOURCE_KINDS
    # A document-only edge stores no property.
    plain = neptune_codec.relationship_properties(_rel("r", "a", "b", ["u1"]), 10_000)
    assert SOURCE_KINDS_FIELD not in plain


# --- with_graph_ranks --------------------------------------------------------


def test_ranks_count_document_backed_edges_only() -> None:
    entities = [Entity(id=i, name=i) for i in ("a", "b", "c", "d")]
    relationships = [
        _rel("r1", "a", "b", ["u1"]),
        _rel("r2", "a", "c", [SEED]),
        _rel("r3", "b", "c", ["u2", SEED]),
        _rel("r4", "c", "d", []),
    ]
    ranked_entities, ranked_relationships = with_graph_ranks(entities, relationships)

    assert {e.id: e.rank for e in ranked_entities} == {"a": 1, "b": 2, "c": 2, "d": 1}
    # The seed-only edge does not count but is ranked by its endpoints.
    assert {r.id: r.rank for r in ranked_relationships} == {
        "r1": 3,
        "r2": 3,
        "r3": 4,
        "r4": 3,
    }
    # counts= decides: every edge counting is the plain degree.
    every, _ = with_graph_ranks(entities, relationships, counts=lambda _: True)
    assert {e.id: e.rank for e in every} == {"a": 2, "b": 2, "c": 3, "d": 1}


def test_a_seed_only_edge_beside_a_document_edge_leaves_the_link_counted() -> None:
    (a, b), _ = with_graph_ranks(
        [Entity(id="a", name="a"), Entity(id="b", name="b")],
        [_rel("r1", "a", "b", [SEED]), _rel("r2", "b", "a", ["u1"], "PAYS")],
    )
    assert (a.rank, b.rank) == (1, 1)


def test_full_build_resolution_ranks_over_document_backed_edges() -> None:
    resolver = GraphResolver(Config(), max_workers=1, use_process_pool=False)
    result, _ = resolver.resolve_graph(
        [
            Entity(id="a", name="Vendor", text_unit_ids=["u1"]),
            Entity(id="b", name="Buyer", text_unit_ids=["u1"]),
            Entity(id="c", name="Catalog", text_unit_ids=[SEED]),
        ],
        [_rel("r1", "a", "b", ["u1"]), _rel("r2", "a", "c", [SEED])],
    )
    assert {e.id: e.rank for e in result["entities"]} == {"a": 1, "b": 1, "c": 0}
    assert {r.id: r.rank for r in result["relationships"]} == {"r1": 2, "r2": 1}


# --- the incremental paths ---------------------------------------------------


def _manager() -> tuple[IndexingManager, FakeGraphStore]:
    config = Config()
    graph = FakeGraphStore()
    manager = IndexingManager(
        config=config,
        vector_indexer=FakeVectorStore(opensearch_config=config.indexing.opensearch),
        graph_indexer=graph,
    )
    return manager, graph


def test_an_incremental_merge_reads_the_stored_document_backed_neighbours() -> None:
    manager, graph = _manager()
    incremental = IncrementalIndexer(FakeDocStatusStore(), manager)
    names = {"a": "Vendor", "b": "Catalog", "c": "Buyer"}

    def commit(doc_id: str, unit: str, relationship: Relationship) -> None:
        ids = [relationship.source_id, relationship.target_id]
        incremental.commit(
            lineages=[DocumentLineage(doc_id=doc_id)],
            fingerprints={doc_id: f"h-{doc_id}"},
            entities=[Entity(id=i, name=names[i], text_unit_ids=[unit]) for i in ids],
            relationships=[relationship],
        )

    commit("seed", SEED, _rel("r1", "a", "b", [SEED]))
    commit("d1", "u1", _rel("r2", "a", "c", ["u1"]))

    # The stored seed edge a-b is not followed: a ranks 1 (c), not 2.
    ranks = {e.id: e.rank for e in graph.read_entities(["a", "b", "c"])}
    assert ranks == {"a": 1, "b": 0, "c": 1}


def _store(graph: FakeGraphStore, relationships: list[Relationship]) -> None:
    """Store ``relationships`` with their ``source_kinds`` set, as indexing
    annotates them (the graph codec writes the field)."""
    graph.upsert_entities([Entity(id=i, name=i.upper()) for i in ("a", "b", "c")])
    graph.upsert_relationships(
        [
            r.model_copy(update={"source_kinds": annotate_sources(r)})
            for r in relationships
        ]
    )


def test_a_stored_edge_written_seed_only_no_longer_ranks_its_endpoints() -> None:
    # A strip removed the document unit of a-b, its seed unit is left: the
    # stored copy is still document-backed until it is written.
    manager, graph = _manager()
    stored = _rel("r-ab", "a", "b", ["u1", SEED])
    _store(graph, [stored, _rel("r-ac", "a", "c", ["u2"])])

    entities, _ = manager._finish(
        "default",
        [Entity(id="a", name="A", text_unit_ids=["u2", SEED])],
        [_rel("r-ab", "a", "b", [SEED])],
        stored_relationships=[stored],
    )
    assert [e.rank for e in entities] == [1]


def test_another_document_edge_of_the_pair_keeps_the_link() -> None:
    manager, graph = _manager()
    stored = _rel("r-ab", "a", "b", ["u1", SEED])
    _store(graph, [stored, _rel("r-ba", "b", "a", ["u3"], "PAYS")])

    entities, _ = manager._finish(
        "default",
        [Entity(id="a", name="A", text_unit_ids=["u3", SEED])],
        [_rel("r-ab", "a", "b", [SEED])],
        stored_relationships=[stored],
    )
    assert [e.rank for e in entities] == [1]


def test_another_seed_only_edge_of_the_pair_does_not_keep_the_link() -> None:
    manager, graph = _manager()
    stored = _rel("r-ab", "a", "b", ["u1", SEED])
    _store(
        graph,
        [
            stored,
            _rel("r-ba", "b", "a", [SEED_2], "PAYS"),
            _rel("r-ac", "a", "c", ["u2"]),
        ],
    )

    entities, _ = manager._finish(
        "default",
        [Entity(id="a", name="A", text_unit_ids=["u2", SEED])],
        [_rel("r-ab", "a", "b", [SEED])],
        stored_relationships=[stored],
    )
    assert [e.rank for e in entities] == [1]


def test_a_stored_edge_under_its_old_id_is_demoted_by_its_rekeyed_write() -> None:
    # A rename rekeys the stored r-old to r-new, written seed-only.
    manager, graph = _manager()
    stored = _rel("r-old", "a", "b", ["u1", SEED])
    _store(graph, [stored])

    links = manager._demoted_links(
        "default", [_rel("r-new", "a", "b", [SEED])], [stored], {"r-old": "r-new"}, ()
    )
    assert links == {frozenset(("a", "b"))}


def test_a_stored_seed_only_edge_written_seed_only_is_not_demoted() -> None:
    # Its stored copy never counted: nothing changes.
    manager, graph = _manager()
    stored = _rel("r-ab", "a", "b", [SEED])
    _store(graph, [stored])

    links = manager._demoted_links(
        "default", [_rel("r-ab", "a", "b", [SEED, SEED_2])], [stored], {}, ()
    )
    assert links == set()


def test_a_dropped_document_edge_of_the_pair_does_not_keep_the_link() -> None:
    manager, graph = _manager()
    stored = _rel("r-ab", "a", "b", ["u1", SEED])
    _store(graph, [stored, _rel("r-ba", "b", "a", ["u3"], "PAYS")])

    def demoted(dropped: tuple[str, ...]) -> set[frozenset[str]]:
        return manager._demoted_links(
            "default", [_rel("r-ab", "a", "b", [SEED])], [stored], {}, dropped
        )

    assert demoted(()) == set()
    assert demoted(("r-ba",)) == {frozenset(("a", "b"))}


def test_without_a_seed_only_edge_nothing_more_is_read(mocker) -> None:
    manager, graph = _manager()
    _store(graph, [_rel("r-ab", "a", "b", ["u1"])])
    incident = mocker.spy(graph, "find_incident_relationship_ids")
    neighbours = mocker.spy(graph, "read_neighbor_ids")

    manager._finish(
        "default",
        [Entity(id="a", name="A", text_unit_ids=["u1"])],
        [_rel("r-ab", "a", "b", ["u1"])],
        stored_relationships=[_rel("r-ab", "a", "b", ["u1", "u9"])],
    )
    # The one incident read is _finish's own (the stored edges to rewrite).
    assert incident.call_count == 1
    assert all(c.kwargs["document_backed"] for c in neighbours.call_args_list)


# --- the fake graph store ----------------------------------------------------


def test_the_fake_reads_document_backed_neighbours() -> None:
    _, graph = _manager()
    _store(
        graph,
        [
            _rel("r-ab", "a", "b", [SEED]),
            _rel("r-ac", "a", "c", ["u1", SEED]),
            _rel("r-bc", "b", "c", ["u2"]),
        ],
    )
    assert graph.read_neighbor_ids(["a", "b"], document_backed=True) == {
        "a": ["c"],
        "b": ["c"],
    }
    assert graph.read_neighbor_ids(["a", "b"]) == {"a": ["b", "c"], "b": ["a", "c"]}


# --- community reports -------------------------------------------------------


def _detector(*, report: bool = False) -> CommunityDetector:
    config = Config()
    config.graph.community_detection.report_generation.enabled = report
    return CommunityDetector(config, show_progress=False)


def _community(
    cid: str, entity_ids: list[str], units: list[str], level: str = "0", **kw: Any
) -> Community:
    return Community(
        id=cid,
        name=cid,
        level=level,
        parent=kw.pop("parent", ""),
        children=kw.pop("children", []),
        entity_ids=entity_ids,
        text_unit_ids=units,
        size=len(entity_ids),
        **kw,
    )


def _seeded_graph() -> nx.Graph:
    """A document pair v-b, and a seed hub ``cat`` linked to v, b and x."""
    entities = [
        Entity(id="v", name="Vendor", type="ORG", text_unit_ids=["u1", SEED]),
        Entity(id="b", name="Buyer", type="ORG", text_unit_ids=["u1", SEED]),
        Entity(id="cat", name="Catalog", type="ORG", text_unit_ids=[SEED]),
        Entity(id="x", name="Item", type="ORG", text_unit_ids=[SEED]),
    ]
    relationships = [
        _rel("r-vb", "v", "b", ["u1"]),
        _rel("r-cv", "cat", "v", [SEED]),
        _rel("r-cb", "cat", "b", [SEED]),
        _rel("r-cx", "cat", "x", [SEED]),
    ]
    return GraphBuilder(entities, relationships).build()


def test_document_degrees_are_graph_degrees_without_seed_units() -> None:
    detector = _detector()
    graph = nx.Graph()
    graph.add_edge("a", "b", text_unit_ids=["u1"])
    graph.add_edge("b", "c")
    graph.add_edge("c", "c", text_unit_ids=["u2"])
    detector.graph = graph
    assert detector._document_degrees() == dict(graph.degree())


def test_report_rank_and_entity_order_count_document_backed_edges() -> None:
    detector = _detector()
    detector.graph = _seeded_graph()
    community = _community("c", ["cat", "v", "b", "x"], ["u1", SEED])

    assert detector._document_degrees() == {"v": 1, "b": 1, "cat": 0, "x": 0}
    # graph.degree would give 3 + 2 + 2 + 1 = 8.
    assert detector._compute_report_rank(community) == 2
    # The seed hub no longer leads: the document entities fill the cap first.
    assert detector._select_report_entities(community) == ["b", "v", "cat", "x"]
    # The document pair's edge leads the relationships (by endpoint degree).
    prepared = detector._prepare_report_input(community)
    first = prepared["relationships"][0]
    assert {first["source"], first["target"]} == {"Vendor", "Buyer"}


def test_an_attributeless_member_reads_no_none_placeholder() -> None:
    # A node with neither a type nor a description (a seed edge's endpoint)
    # reads "unknown" and an empty description, or the stored type it is
    # given (generate_reports(types=)), never "None".
    detector = _detector()
    detector.graph = GraphBuilder(
        [
            Entity(id="v", name="Vendor", type="ORG", text_unit_ids=["u1"]),
            Entity(id="h", name="Harbor Port", text_unit_ids=[SEED]),
        ],
        [_rel("r", "v", "h", [SEED], "SHIPS_FROM")],
        [],
    ).build()
    community = _community("c", ["v", "h"], ["u1", SEED])

    def harbor_line() -> str:
        (line,) = [
            detector._format_entity_line(e)
            for e in detector._prepare_report_input(community)["entities"]
            if e["id"] == "h"
        ]
        return line

    assert harbor_line() == "- Harbor Port (unknown): "
    detector._report_types = {"h": "location"}
    detector._report_descriptions = {"h": "Harbor docks."}
    assert harbor_line() == "- Harbor Port (location): Harbor docks."


def test_an_edge_with_one_document_relationship_counts_in_the_degrees() -> None:
    # One graph edge carries both relationships of the pair: the document
    # one keeps it counted beside a seed-only one of another type.
    detector = _detector()
    detector.graph = GraphBuilder(
        [
            Entity(id="v", name="Vendor", type="ORG", text_unit_ids=["u1", SEED]),
            Entity(id="b", name="Buyer", type="ORG", text_unit_ids=["u1", SEED]),
        ],
        [
            _rel("r-doc", "v", "b", ["u1"], "SUPPLIES"),
            _rel("r-seed", "v", "b", [SEED], "SHIPS_TO"),
        ],
    ).build()

    assert detector._document_degrees() == {"v": 1, "b": 1}


def test_the_degrees_follow_the_graph() -> None:
    detector = _detector()
    detector.graph = _seeded_graph()
    assert detector._document_degrees()["cat"] == 0
    graph = nx.Graph()
    graph.add_edge("cat", "v")
    detector.graph = graph
    assert detector._document_degrees() == {"cat": 1, "v": 1}


def _community_stage(mocker, communities: list[Community], failed: list[str]):
    from unified_kg_rag.application.ingestion import pipeline_stages as ps

    detector = mocker.MagicMock()
    detector.generate_community_objects.return_value = communities
    detector.generate_reports.return_value = ([], failed)
    detector.get_community_metrics.return_value = None
    mocker.patch.object(ps, "CommunityDetector", return_value=detector)
    config = Config()
    config.indexing.max_failure_rate = 0.5
    stage = ps.CommunityDetectionStage(config=config)
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory="/tmp/src",
    )
    context.knowledge_graph = nx.Graph()
    return stage, context, detector


def test_seed_only_communities_get_no_report_and_no_failure(mocker) -> None:
    leaf = _community("leaf", ["v", "b"], ["u1", SEED], parent="root")
    seed_leaf = _community("seed-leaf", ["cat", "x"], [SEED, SEED_2], parent="root")
    # A parent of an evidenced child is evidenced.
    root = _community(
        "root", ["v", "b", "cat", "x"], ["u1", SEED, SEED_2], "1", children=["leaf"]
    )
    seed_root = _community("seed-root", ["cat", "x"], [SEED], "1")
    stage, context, detector = _community_stage(
        mocker, [leaf, seed_leaf, root, seed_root], failed=["leaf"]
    )

    _, _, metrics = stage._execute_core(context)

    (asked,) = detector.generate_reports.call_args.args
    assert [c.id for c in asked] == ["leaf", "root"]
    # Stored all the same.
    assert [c.id for c in context.communities] == [
        "leaf",
        "seed-leaf",
        "root",
        "seed-root",
    ]
    assert metrics is not None
    assert metrics["communities_seed_only"] == 2
    assert metrics["reports_failed"] == 1
    # A failed report never re-imports a seed shard: its seed units are left out.
    assert context.failed_text_unit_ids == {"community_detection": ["u1"]}


def test_the_failure_rate_counts_the_evidenced_communities_only(mocker) -> None:
    # 1 of 2 evidenced is within 50%; over all four it would be 25%, and a
    # seed-only community that "failed" would not count either.
    communities = [
        _community("leaf", ["v"], ["u1"]),
        _community("other", ["b"], ["u2"]),
        _community("s1", ["cat"], [SEED]),
        _community("s2", ["x"], [SEED_2]),
    ]
    stage, context, _ = _community_stage(mocker, communities, failed=["leaf"])
    _, _, metrics = stage._execute_core(context)
    assert metrics is not None and metrics["communities_seed_only"] == 2
    # Both evidenced ones failing is 100%, over the limit; over all four it
    # would be 50%, within it.
    stage, context, _ = _community_stage(mocker, communities, failed=["leaf", "other"])
    with pytest.raises(PipelineStageError, match="for 2 of 2 communities"):
        stage._execute_core(context)


def test_unread_members_count_their_communities_failed(mocker) -> None:
    # The stored members of the seed-only-looking communities could not be
    # read: whether the corpus evidences them is unknown, so they are
    # stored without a report and count as failed, not seed-only.
    from unified_kg_rag.application.ingestion import pipeline_stages as ps

    communities = [
        _community("leaf", ["v"], ["u1"]),
        _community("s1", ["cat"], [SEED]),
        _community("s2", ["x"], [SEED_2]),
    ]
    stage, context, detector = _community_stage(mocker, communities, failed=[])
    mocker.patch.object(
        stage,
        "_evidenced_by_stored_members",
        return_value=ps._StoredMembers(None, {}, {}),
    )
    _, _, metrics = stage._execute_core(context)
    (asked,) = detector.generate_reports.call_args.args
    assert [c.id for c in asked] == ["leaf"]
    assert metrics is not None
    assert (metrics["reports_failed"], metrics["communities_seed_only"]) == (2, 0)
    # Seed units only: no document is recorded failed for them.
    assert context.failed_text_unit_ids == {"community_detection": []}


def test_without_seed_units_every_community_is_reported(mocker) -> None:
    communities = [_community("c0", ["v"], ["u1"]), _community("c1", ["b"], [])]
    stage, context, detector = _community_stage(mocker, communities, failed=[])
    _, _, metrics = stage._execute_core(context)
    (asked,) = detector.generate_reports.call_args.args
    assert asked == communities
    assert metrics is not None and metrics["communities_seed_only"] == 0


def test_a_parent_whose_child_has_no_report_rolls_up_raw_context() -> None:
    detector = _detector()
    detector.community_detection_config.report_generation.max_report_context_tokens = 1
    detector.graph = _seeded_graph()
    # Its child "seed-leaf" (cat, x) is seed-only: it has no report.
    parent = _community(
        "root", ["v", "b", "cat", "x"], ["u1", SEED], "1", children=["seed-leaf"]
    )
    captured: list[dict[str, Any]] = []

    def run_batch(inputs, communities, attributes):
        captured.extend(inputs)
        return [
            CommunityReport(id=f"r-{c.id}", community_id=c.id, name=c.name)
            for c in communities
        ], []

    detector._run_report_batch = run_batch  # type: ignore[method-assign]
    detector.report_generator = object()
    detector.community_detection_config.report_generation.enabled = True
    # Only the evidenced parent is asked for (the stage leaves the child out).
    reports, failed = detector.generate_reports([parent])

    assert [r.community_id for r in reports] == ["root"] and failed == []
    (prepared,) = captured
    assert prepared["sub_community_reports"] == ""
    # Its raw context leads with a document entity.
    assert prepared["entities"][0]["name"] in {"Vendor", "Buyer"}


def test_report_lineage_keeps_seed_documents_and_the_seed_filter() -> None:
    community = _community("c", ["v", "cat"], ["u1", SEED])
    report = CommunityReport(id="r", community_id="c", name="c")
    units = [
        TextUnit(
            id="u1",
            text="...",
            document_ids=["d1"],
            attributes={"filters": {"region": "north"}},
        ),
        TextUnit(
            id=SEED,
            text="...",
            document_ids=["seed_graphs/procure/0000-of-0001.jsonl"],
            attributes={
                "source_kind": "seed_graph",
                "filters": {"seed_graph": "procure"},
            },
        ),
    ]
    CommunityDetector._attach_report_lineage([report], [community], units)
    assert report.document_ids == ["d1", "seed_graphs/procure/0000-of-0001.jsonl"]
    assert (report.attributes or {})["filters"] == {
        "region": "north",
        "seed_graph": "procure",
    }


# --- local search ------------------------------------------------------------


def _node(name: str, units: list[str]) -> RetrievalResult:
    return RetrievalResult(
        content=name,
        source=name,
        score=1.0,
        retriever_type="graph",
        metadata={"id": name, "text_unit_ids": units},
    )


def test_local_frequency_counts_document_units_only() -> None:
    nodes = [
        _node("seeded", ["u1", "u2", SEED]),
        _node("generic", ["u1", "u2", "u3"]),
        _node("single", [SEED]),
    ]
    kept = LocalSearchStrategy._filter_entities(nodes, frequency_threshold=2)
    assert [n.source for n in kept] == ["seeded", "single"]
    assert LocalSearchStrategy._text_unit_count(_node("one", ["u1"])) == 1
