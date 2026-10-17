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
