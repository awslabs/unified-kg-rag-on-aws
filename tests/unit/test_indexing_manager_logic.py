# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for IndexingManager orchestration helpers (AWS-free).

Covers the pure-logic helpers that ``test_indexing_manager_delta`` does NOT
exercise: ``_enrich_text_units`` back-linking, ``_run_indexing_phase`` pool
sizing + empty-task skipping, ``merge_with_existing_graph`` cross-run union,
and ``_log_completion_summary``. Indexers are replaced with the in-memory
``FakeGraphStore`` / ``FakeVectorStore`` fakes so no AWS clients are built.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.domain.ingestion.relationship_weights import (
    apply_text_unit_weights,
)
from unified_kg_rag.domain.models import (
    Community,
    Config,
    Entity,
    Relationship,
    TextUnit,
)
from unified_kg_rag.ports.indexer import IndexingStats
from unified_kg_rag.shared.exceptions import StoreAccessDeniedError

pytestmark = pytest.mark.unit


@pytest.fixture
def manager():
    """An IndexingManager wired to in-memory fake stores (no AWS)."""
    graph = FakeGraphStore()
    vector = FakeVectorStore()
    from unified_kg_rag.application.storage.indexing_manager import (
        IndexingManager,
        IndexingTask,
    )

    mgr = IndexingManager(config=Config(), vector_indexer=vector, graph_indexer=graph)
    return mgr, graph, vector, IndexingTask


# --- _enrich_text_units --------------------------------------------------


def test_enrich_text_units_backlinks_community_ids() -> None:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    units = [TextUnit(id="t1", text="a"), TextUnit(id="t2", text="b")]
    communities = [
        Community(
            id="c1",
            name="C1",
            level="0",
            parent="",
            children=[],
            entity_ids=[],
            text_unit_ids=["t1", "t2"],
        ),
        Community(
            id="c2",
            name="C2",
            level="0",
            parent="",
            children=[],
            entity_ids=[],
            text_unit_ids=["t1"],
        ),
    ]

    IndexingManager._enrich_text_units(units, communities)

    assert units[0].community_ids == ["c1", "c2"]
    assert units[1].community_ids == ["c1"]


def test_enrich_text_units_idempotent_no_duplicates() -> None:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    units = [TextUnit(id="t1", text="a", community_ids=["c1"])]
    communities = [
        Community(
            id="c1",
            name="C1",
            level="0",
            parent="",
            children=[],
            entity_ids=[],
            text_unit_ids=["t1"],
        ),
    ]

    IndexingManager._enrich_text_units(units, communities)

    # Already present: not appended twice.
    assert units[0].community_ids == ["c1"]


def test_enrich_text_units_noop_without_communities() -> None:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    units = [TextUnit(id="t1", text="a")]
    IndexingManager._enrich_text_units(units, None)
    IndexingManager._enrich_text_units(units, [])
    assert units[0].community_ids is None


def test_enrich_text_units_ignores_unknown_text_unit_id() -> None:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    units = [TextUnit(id="t1", text="a")]
    communities = [
        Community(
            id="c1",
            name="C1",
            level="0",
            parent="",
            children=[],
            entity_ids=[],
            text_unit_ids=["missing"],
        ),
    ]
    IndexingManager._enrich_text_units(units, communities)
    # The single known unit is untouched; the unknown id is skipped.
    assert units[0].community_ids is None


# --- _run_indexing_phase -------------------------------------------------


def test_run_indexing_phase_skips_empty_tasks(manager) -> None:
    mgr, _graph, _vector, IndexingTask = manager
    calls: list[str] = []

    def fn_ok(items):
        calls.append("ok")
        return IndexingStats(total_items=len(items), successful_items=len(items))

    def fn_empty(items):  # should never run
        calls.append("empty")
        return IndexingStats()

    tasks = [
        IndexingTask(fn_ok, [["x"]], "ok"),
        IndexingTask(fn_empty, [[]], "empty_list"),
        IndexingTask(fn_empty, [None], "none_arg"),
    ]

    results = mgr._run_indexing_phase(tasks)

    assert calls == ["ok"]
    assert set(results.keys()) == {"ok"}
    assert results["ok"].successful_items == 1


def test_run_indexing_phase_no_valid_tasks_returns_empty(manager) -> None:
    mgr, _g, _v, IndexingTask = manager
    results = mgr._run_indexing_phase([IndexingTask(lambda x: None, [[]], "k")])
    assert results == {}


def test_run_indexing_phase_captures_task_exception_as_failed_stats(manager) -> None:
    mgr, _g, _v, IndexingTask = manager

    def boom(items):
        raise RuntimeError("backend down")

    results = mgr._run_indexing_phase([IndexingTask(boom, [["a"]], "boom")])

    assert "boom" in results
    stats = results["boom"]
    assert stats.failed_items == 1
    assert any("backend down" in e for e in stats.errors)


def test_run_indexing_phase_pool_size_capped_at_eight(manager, mocker) -> None:
    mgr, _g, _v, IndexingTask = manager
    captured: dict[str, int] = {}

    real_pool = (
        "unified_kg_rag.application.storage.indexing_manager.ContextThreadPoolExecutor"
    )
    RealExecutor = mocker.patch(real_pool, wraps=None)

    from concurrent.futures import ThreadPoolExecutor as _Real

    def factory(max_workers):
        captured["max_workers"] = max_workers
        return _Real(max_workers=max_workers)

    RealExecutor.side_effect = factory

    # 10 valid tasks -> pool capped at 8.
    tasks = [
        IndexingTask(lambda items: IndexingStats(), [["a"]], f"k{i}") for i in range(10)
    ]
    mgr._run_indexing_phase(tasks)
    assert captured["max_workers"] == 8


def test_run_indexing_phase_pool_size_matches_task_count_when_small(
    manager, mocker
) -> None:
    mgr, _g, _v, IndexingTask = manager
    captured: dict[str, int] = {}
    from concurrent.futures import ThreadPoolExecutor as _Real

    RealExecutor = mocker.patch(
        "unified_kg_rag.application.storage.indexing_manager.ContextThreadPoolExecutor"
    )
    RealExecutor.side_effect = lambda max_workers: (
        captured.update(max_workers=max_workers) or _Real(max_workers=max_workers)
    )

    tasks = [
        IndexingTask(lambda items: IndexingStats(), [["a"]], f"k{i}") for i in range(3)
    ]
    mgr._run_indexing_phase(tasks)
    assert captured["max_workers"] == 3


# --- merge_with_existing_graph ------------------------------------------


def test_merge_with_existing_graph_unions_existing(manager) -> None:
    mgr, graph, _vector, _t = manager
    # Seed the graph store with an existing entity/relationship the delta touches.
    existing_e = Entity(id="e1", name="Alice", description="old", text_unit_ids=["t0"])
    existing_r = Relationship(id="r1", source_id="e1", target_id="e2", weight=1.0)
    graph.index_entities([existing_e])
    graph.index_relationships([existing_r])

    new_e = Entity(id="e1", name="Alice", description="new", text_unit_ids=["t1"])
    new_r = Relationship(id="r1", source_id="e1", target_id="e2", weight=2.0)

    result = mgr.merge_with_existing_graph([new_e], [new_r])
    merged_e, merged_r = result.entities, result.relationships

    assert len(merged_e) == 1
    # Merge unions text_unit_ids from existing + new.
    assert set(merged_e[0].text_unit_ids) >= {"t0", "t1"}
    assert len(merged_r) == 1


def test_merge_with_existing_graph_passthrough_when_nothing_existing(manager) -> None:
    mgr, _graph, _vector, _t = manager
    new_e = [Entity(id="e9", name="Bob")]
    new_r = [Relationship(id="r9", source_id="e9", target_id="e8")]

    result = mgr.merge_with_existing_graph(new_e, new_r)
    merged_e, merged_r = result.entities, result.relationships

    # No existing rows read back -> returns the inputs (overwrite mode), ranked
    # by the delta's relationships.
    assert merged_e == [new_e[0].model_copy(update={"rank": 1})]
    assert merged_r == [new_r[0].model_copy(update={"rank": 2})]


def test_merge_with_existing_graph_handles_empty_inputs(manager) -> None:
    mgr, _graph, _vector, _t = manager
    result = mgr.merge_with_existing_graph(None, None)
    merged_e, merged_r = result.entities, result.relationships
    assert merged_e is None
    assert merged_r is None


# --- _log_completion_summary ---------------------------------------------


def test_log_completion_summary_aggregates_and_warns_on_failures(caplog) -> None:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    results = {
        "opensearch_entities": IndexingStats(
            total_items=10, successful_items=10, failed_items=0
        ),
        "neptune_relationships": IndexingStats(
            total_items=10,
            successful_items=2,
            failed_items=8,
            errors=["timeout"],
        ),
    }
    import logging

    with caplog.at_level(logging.WARNING):
        IndexingManager._log_completion_summary(results, elapsed_time=1.5)

    text = caplog.text
    # High failure rate (>50%) for the relationship task is surfaced.
    assert "neptune_relationships" in text
    assert "Failed items" in text


def test_log_completion_summary_no_warnings_on_full_success(caplog) -> None:
    import logging

    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    results = {
        "opensearch_text_units": IndexingStats(
            total_items=5, successful_items=5, failed_items=0
        ),
    }
    with caplog.at_level(logging.WARNING):
        IndexingManager._log_completion_summary(results, elapsed_time=0.1)

    assert "Failed items" not in caplog.text


def test_log_completion_summary_handles_empty_results(caplog) -> None:
    import logging

    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    # No items -> the success_rate guard avoids dividing by zero.
    with caplog.at_level(logging.INFO):
        IndexingManager._log_completion_summary({}, elapsed_time=0.0)
    assert "0/0 items (0.0%)" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


# --- _discover_suffixes --------------------------------------------------


def test_discover_suffixes_empty_returns_empty() -> None:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    assert IndexingManager._discover_suffixes(None) == []
    assert IndexingManager._discover_suffixes([]) == []


def test_discover_suffixes_defaults_when_no_attributes() -> None:
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    units = [TextUnit(id="t1", text="a"), TextUnit(id="t2", text="b")]
    assert IndexingManager._discover_suffixes(units) == ["default"]


# --- build_relationship_vector_index toggle (GraphRAG-only deployments) ------


def _rel(rid: str = "r1"):
    from unified_kg_rag.domain.models import Relationship

    return Relationship(id=rid, source_id="e1", target_id="e2", type="REL")


def test_edges_of_a_failed_entity_write_are_held_back(manager, mocker) -> None:
    # Regression: the edges were written after their endpoints' failed graph
    # writes; a store drops an edge's stored copy before writing it, so an
    # edge re-pointed at a re-keyed entity whose write failed was lost.
    mgr, graph, _, _ = manager
    entities = [Entity(id=i, name=i.upper()) for i in ("a", "b", "c")]
    rels = [
        Relationship(id="r-ab", source_id="a", target_id="b"),
        Relationship(id="r-bc", source_id="b", target_id="c"),
    ]
    graph.upsert_entities(entities)  # stored by an earlier run
    rejected = IndexingStats(total_items=3, successful_items=2)
    rejected.add_error("rejected", ids=["c"])
    mocker.patch.object(graph, "upsert_entities", return_value=rejected)
    write = mocker.spy(graph, "upsert_relationships")

    results = mgr.index_delta(entities=entities, relationships=rels)

    (written,) = write.call_args.args
    assert [r.id for r in written] == ["r-ab"]
    assert results["neptune_relationships"].failed_ids == ["r-bc"]

    # A failure naming no entity holds back every edge.
    mocker.patch.object(graph, "upsert_entities", side_effect=RuntimeError("down"))
    write.reset_mock()
    results = mgr.index_delta(entities=entities, relationships=rels)
    write.assert_not_called()
    assert sorted(results["neptune_relationships"].failed_ids) == ["r-ab", "r-bc"]


def test_a_deletion_that_dies_between_the_stores_keeps_the_graph_copy(
    manager, mocker
) -> None:
    # Regression: the graph copy went first, so a run killed before the
    # vector delete left a vector copy no retry finds (retries find what to
    # delete, e.g. a re-keyed entity's old id, through the graph).
    mgr, graph, vector, _ = manager
    graph.upsert_entities([Entity(id="old", name="Old")])

    class Killed(BaseException):
        pass

    mocker.patch.object(vector, "delete_document_artifacts", side_effect=Killed)
    with pytest.raises(Killed):
        mgr.delete_documents({"default": ["old"]})

    assert graph.ids("entities") == {"old"}


class _RebuildingGraph(FakeGraphStore):
    """A graph store rebuilt in place that records how each rebuild ended."""

    def __init__(self) -> None:
        super().__init__()
        self.sweeps: list[bool] = []

    def end_rebuild(self, suffixes, sweep: bool = True) -> None:
        self.sweeps.append(sweep)
        super().end_rebuild(suffixes, sweep=sweep)


def _rebuild_manager(graph: FakeGraphStore):
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    return IndexingManager(
        config=Config(), vector_indexer=FakeVectorStore(), graph_indexer=graph
    )


def test_a_rebuild_whose_graph_write_failed_keeps_the_previous_graph(
    mocker,
) -> None:
    graph = _RebuildingGraph()
    graph.upsert_entities([Entity(id="old", name="Old")])
    failed = IndexingStats(total_items=1)
    failed.add_error("rejected", ids=["new"])
    mocker.patch.object(graph, "index_entities", return_value=failed)

    _rebuild_manager(graph).index_all_data(entities=[Entity(id="new", name="New")])

    assert graph.sweeps == [False]
    assert graph.ids("entities") == {"old"}


def test_a_rebuild_that_raised_keeps_the_previous_graph(mocker) -> None:
    graph = _RebuildingGraph()
    graph.upsert_entities([Entity(id="old", name="Old")])
    manager = _rebuild_manager(graph)
    mocker.patch.object(manager, "_run_indexing_phase", side_effect=KeyboardInterrupt)

    with pytest.raises(KeyboardInterrupt):
        manager.index_all_data(entities=[Entity(id="new", name="New")])

    assert graph.sweeps == [False]
    assert graph.ids("entities") == {"old"}


def test_a_failed_sweep_is_reported_as_a_failed_write(mocker) -> None:
    graph = _RebuildingGraph()
    mocker.patch.object(graph, "end_rebuild", side_effect=RuntimeError("timeout"))

    results = _rebuild_manager(graph).index_all_data(
        entities=[Entity(id="new", name="New")]
    )

    assert results["neptune_sweep"].failed_items == 1
    assert "timeout" in results["neptune_sweep"].errors[0]


def test_an_edge_rewritten_for_its_rank_only_is_not_embedded_again(
    manager, mocker
) -> None:
    # Regression: a hub's new neighbour changed every hub edge's rank, and
    # each was upserted (embedded) again although only its rank changed.
    mgr, _, vector, _ = manager
    stored = Relationship(
        id="r-old", source_id="a", target_id="b", rank=1, description="A to B"
    )
    # Stored with no description: its copy is there all the same.
    blank = Relationship(id="r-blank", source_id="a", target_id="d", rank=1)
    vector.upsert_relationships([stored, blank])
    upsert = mocker.spy(vector, "upsert_relationships")
    mgr._rank_only = {"r-old", "r-blank", "r-gone"}
    rels = [
        stored.model_copy(update={"rank": 5}),
        blank.model_copy(update={"rank": 4}),
        # Rewritten for its rank, but its vector copy is missing: upserted.
        Relationship(id="r-gone", source_id="a", target_id="c", rank=2),
        Relationship(id="r-new", source_id="b", target_id="c"),
    ]

    stats = mgr._upsert_vector_relationships(rels)

    (written,) = upsert.call_args.args
    assert sorted(r.id for r in written) == ["r-gone", "r-new"]
    assert (stats.total_items, stats.successful_items, stats.failed_items) == (
        4,
        4,
        0,
    )
    assert vector.data["relationships"]["r-old"].rank == 5
    assert vector.data["relationships"]["r-blank"].rank == 4
    assert mgr._rank_only == set()


def test_items_derivation_cannot_derive_are_counted(manager) -> None:
    # A stored text unit naming no record (indexed before records existed):
    # its items keep their stored content, counted for the stage's metric.
    mgr, _, vector, _ = manager
    vector.upsert_text_units([TextUnit(id="t-legacy", text="legacy")])
    stored = Entity(id="e", name="E", description="stale", text_unit_ids=["t-legacy"])

    derived = mgr.derive("default", [stored], [], [])

    assert derived.entities[0].description == "stale"
    assert mgr.underived_items == 1


def test_the_stored_edges_of_a_re_keyed_entity_are_derived_with_it(
    manager,
) -> None:
    # Regression: an interrupted run had written "Acme" (the delta's
    # spelling) next to the stored "Acme Inc"; the retry merged the two and
    # derived the delta's Acme -> Depot edge, but the stored Acme Inc -> Depot
    # edge was only re-pointed at Acme, leaving two edges between them.
    from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
    from unified_kg_rag.domain.ingestion.contributions import (
        RECORD_KEYS_ATTRIBUTE,
        record_payload,
    )
    from unified_kg_rag.domain.models import TextUnitExtraction

    mgr, graph, vector, _ = manager

    def entity(name: str, units: list[str]) -> Entity:
        return Entity(
            id=BaseProcessor._generate_entity_id(name),
            name=name,
            type="ORG",
            description=f"{name} in {units}",
            text_unit_ids=units,
        )

    def edge(source: str, target: str, unit: str) -> Relationship:
        rel = Relationship(
            id=BaseProcessor._generate_relationship_id(source, target, "SUPPLIES"),
            source_id=BaseProcessor._generate_entity_id(source),
            target_id=BaseProcessor._generate_entity_id(target),
            source_name=source,
            target_name=target,
            type="SUPPLIES",
            description=f"{source} supplies {target} in {unit}",
        )
        apply_text_unit_weights(rel, {unit: 1.0})
        return rel

    # t-a (new) states Acme -> Depot, t-b (stored) Acme Inc -> Depot.
    records = {
        "t-a": TextUnitExtraction(
            text_unit_id="t-a",
            entities=[entity("Acme", ["t-a"]), entity("Depot", ["t-a"])],
            relationships=[edge("Acme", "Depot", "t-a")],
        ),
        "t-b": TextUnitExtraction(
            text_unit_id="t-b",
            entities=[entity("Acme Inc", ["t-b"]), entity("Depot", ["t-b"])],
            relationships=[edge("Acme Inc", "Depot", "t-b")],
        ),
    }
    vector.write_extraction_records(
        {f"g-{u}": record_payload(r) for u, r in records.items()}
    )
    units = {
        u: TextUnit(
            id=u, text=u, attributes={RECORD_KEYS_ATTRIBUTE: {"graph": f"g-{u}"}}
        )
        for u in records
    }
    vector.upsert_text_units([units["t-b"]])
    graph.upsert_entities(
        [
            entity("Acme Inc", ["t-b"]),
            entity("Depot", ["t-b"]),
            # Written by the interrupted run.
            entity("Acme", ["t-a"]),
        ]
    )
    graph.upsert_relationships([edge("Acme Inc", "Depot", "t-b")])

    merged = mgr.merge_with_existing_graph(
        [entity("Acme", ["t-a"]), entity("Depot", ["t-a"])],
        [edge("Acme", "Depot", "t-a")],
        [units["t-a"]],
    )
    assert mgr.index_delta(entities=merged.entities, relationships=merged.relationships)
    mgr.delete_retired_ids()

    stored = graph.read_relationships(sorted(graph.ids("relationships")))
    assert [(r.source_name, r.target_name) for r in stored] == [("Acme", "Depot")]
    assert sorted(stored[0].text_unit_ids or []) == ["t-a", "t-b"]


def test_an_entity_split_off_under_a_folded_delta_id_is_kept(manager) -> None:
    # Regression: the delta's "Acme Inc" folded into the stored "Acme" and
    # the derivation split it off again under its own id, but the folded
    # delta id was then deleted with the retired ids.
    from unified_kg_rag.domain.ingestion.base_processor import BaseProcessor
    from unified_kg_rag.domain.ingestion.contributions import (
        RECORD_KEYS_ATTRIBUTE,
        record_payload,
    )
    from unified_kg_rag.domain.ingestion.graph_resolver import ENTITY_KEYS_ATTRIBUTE
    from unified_kg_rag.domain.models import TextUnitExtraction

    mgr, graph, vector, _ = manager

    def entity(name: str, type_: str, units: list[str], **kw) -> Entity:
        return Entity(
            id=BaseProcessor._generate_entity_id(name),
            name=name,
            type=type_,
            description=f"{name} in {units}",
            text_unit_ids=units,
            **kw,
        )

    # t-b (stored) states both spellings as persons; t-a (new, sorting
    # first) states Acme as a location, so a full build keeps them apart.
    records = {
        "t-a": TextUnitExtraction(
            text_unit_id="t-a",
            entities=[
                entity("Acme", "LOCATION", ["t-a"]),
                entity("Acme Inc", "PERSON", ["t-a"]),
            ],
        ),
        "t-b": TextUnitExtraction(
            text_unit_id="t-b",
            entities=[
                entity("Acme", "PERSON", ["t-b"]),
                entity("Acme Inc", "PERSON", ["t-b"]),
            ],
        ),
    }
    vector.write_extraction_records(
        {f"g-{u}": record_payload(r) for u, r in records.items()}
    )
    units = {
        u: TextUnit(
            id=u, text=u, attributes={RECORD_KEYS_ATTRIBUTE: {"graph": f"g-{u}"}}
        )
        for u in records
    }
    vector.upsert_text_units([units["t-b"]])
    graph.upsert_entities(
        [
            entity(
                "Acme",
                "person",
                ["t-b"],
                attributes={ENTITY_KEYS_ATTRIBUTE: ["acme", "acme inc"]},
            )
        ]
    )

    merged = mgr.merge_with_existing_graph(
        [
            entity("Acme", "location", ["t-a"]),
            entity("Acme Inc", "person", ["t-a"]),
        ],
        [],
        [units["t-a"]],
    )
    mgr.index_delta(entities=merged.entities, relationships=merged.relationships)
    mgr.delete_retired_ids()

    names = sorted(e.name for e in graph.read_entities(sorted(graph.ids("entities"))))
    assert names == ["Acme", "Acme Inc"]


def test_relationship_vector_index_built_by_default(manager) -> None:
    mgr, graph, vector, _ = manager
    mgr.index_all_data(relationships=[_rel()], entities=[])
    # OpenSearch (vector) relationship index built by default...
    assert "relationships" in vector.data
    # ...and the Neptune relationship edges too.
    assert "relationships" in graph.data


def test_relationship_vector_index_skipped_when_disabled(manager) -> None:
    mgr, graph, vector, _ = manager
    mgr.config.indexing.opensearch.build_relationship_vector_index = False
    mgr.index_all_data(relationships=[_rel()], entities=[])
    # The LightRAG-only vector index is skipped...
    assert "relationships" not in vector.data
    # ...but the Neptune relationship edges (graph expansion) are still built.
    assert "relationships" in graph.data


def test_relationship_vector_index_disabled_in_delta(manager) -> None:
    mgr, graph, vector, _ = manager
    mgr.config.indexing.opensearch.build_relationship_vector_index = False
    mgr.index_delta(relationships=[_rel()], entities=[])
    assert "relationships" not in vector.data
    assert "relationships" in graph.data


def test_full_rebuild_empties_an_index_type_the_run_has_none_of(manager) -> None:
    # Run 2 has no claims, reports or communities: it must not keep serving
    # run 1's.
    from unified_kg_rag.domain.models import Claim, CommunityReport

    mgr, graph, vector, _ = manager
    community = Community(
        id="c1", name="C1", level="0", parent="", children=[], entity_ids=["e1"]
    )
    mgr.index_all_data(
        text_units=[TextUnit(id="t1", text="a")],
        entities=[Entity(id="e1", name="Vendor")],
        communities=[community],
        community_reports=[CommunityReport(id="cr1", name="R1", community_id="c1")],
        claims=[
            Claim(
                id="cl1",
                subject_id="e1",
                subject_name="Vendor",
                object_name="100",
                type="OWES",
            )
        ],
    )
    assert vector.ids("claims") == {"cl1"}

    mgr.index_all_data(
        text_units=[TextUnit(id="t2", text="b")],
        entities=[Entity(id="e2", name="Buyer")],
    )

    assert vector.ids("claims") == set()
    assert vector.ids("community_reports") == set()
    assert graph.ids("communities") == set()
    assert vector.ids("text_units") == {"t2"}
    assert graph.ids("entities") == {"e2"}


# --- initialize / clear_all_data surface the store's error -----------------


class _DeniedVectorStore(FakeVectorStore):
    """A vector store whose writes are denied (e.g. a missing IAM action)."""

    def initialize(self) -> None:
        raise StoreAccessDeniedError("403 denied", reason="no es:ESHttpPut")

    def clear(self, suffixes: list[str]) -> None:
        raise StoreAccessDeniedError("403 denied", reason="no es:ESHttpDelete")


def _denied_manager():
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager

    return IndexingManager(
        config=Config(),
        vector_indexer=_DeniedVectorStore(),
        graph_indexer=FakeGraphStore(),
    )


def test_initialize_raises_the_store_error() -> None:
    # Was a bare False, which the stage turned into an untyped RuntimeError
    # without the reason.
    with pytest.raises(StoreAccessDeniedError, match="403 denied"):
        _denied_manager().initialize()


@pytest.mark.parametrize("reset", [False, True])
def test_indexing_stage_reports_the_store_error(mocker, reset: bool) -> None:
    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.domain.models import PipelineContext, PipelineStageStatus

    mocker.patch.object(ps, "IndexingManager", return_value=_denied_manager())
    config = Config()
    config.indexing.reset = reset
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory="/tmp/src",
        text_units=[TextUnit(id="t1", text="a")],
    )
    with pytest.raises(StoreAccessDeniedError):
        ps.IndexingStage(config=config)._execute_core(context)
