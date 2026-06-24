# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Unit tests for IndexingManager orchestration helpers (AWS-free).

Covers the pure-logic helpers that ``test_indexing_manager_delta`` does NOT
exercise: ``_enrich_text_units`` back-linking, ``_run_indexing_phase`` pool
sizing + empty-task skipping, ``_merge_with_existing_graph`` cross-run union,
and ``_log_completion_summary``. Indexers are replaced with the in-memory
``FakeGraphStore`` / ``FakeVectorStore`` fakes so no AWS clients are built.
"""

from __future__ import annotations

import pytest

from aws_graphrag.domain.models import (
    Community,
    Config,
    Entity,
    Relationship,
    TextUnit,
)
from aws_graphrag.ports.indexer import IndexingStats
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore

pytestmark = pytest.mark.unit


@pytest.fixture
def manager(mocker):
    """An IndexingManager wired to in-memory fake stores (no AWS)."""
    graph = FakeGraphStore()
    vector = FakeVectorStore()
    mocker.patch(
        "aws_graphrag.application.storage.indexing_manager.OpenSearchIndexer",
        return_value=vector,
    )
    mocker.patch(
        "aws_graphrag.application.storage.indexing_manager.NeptuneIndexer",
        return_value=graph,
    )
    from aws_graphrag.application.storage.indexing_manager import (
        IndexingManager,
        IndexingTask,
    )

    mgr = IndexingManager(config=Config())
    return mgr, graph, vector, IndexingTask


# --- _enrich_text_units --------------------------------------------------


def test_enrich_text_units_backlinks_community_ids() -> None:
    from aws_graphrag.application.storage.indexing_manager import IndexingManager

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
    from aws_graphrag.application.storage.indexing_manager import IndexingManager

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
    from aws_graphrag.application.storage.indexing_manager import IndexingManager

    units = [TextUnit(id="t1", text="a")]
    IndexingManager._enrich_text_units(units, None)
    IndexingManager._enrich_text_units(units, [])
    assert units[0].community_ids is None


def test_enrich_text_units_ignores_unknown_text_unit_id() -> None:
    from aws_graphrag.application.storage.indexing_manager import IndexingManager

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

    real_pool = "aws_graphrag.application.storage.indexing_manager.ThreadPoolExecutor"
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
        "aws_graphrag.application.storage.indexing_manager.ThreadPoolExecutor"
    )
    RealExecutor.side_effect = lambda max_workers: (
        captured.update(max_workers=max_workers) or _Real(max_workers=max_workers)
    )

    tasks = [
        IndexingTask(lambda items: IndexingStats(), [["a"]], f"k{i}") for i in range(3)
    ]
    mgr._run_indexing_phase(tasks)
    assert captured["max_workers"] == 3


# --- _merge_with_existing_graph ------------------------------------------


def test_merge_with_existing_graph_unions_existing(manager) -> None:
    mgr, graph, _vector, _t = manager
    # Seed the graph store with an existing entity/relationship the delta touches.
    existing_e = Entity(id="e1", name="Alice", description="old", text_unit_ids=["t0"])
    existing_r = Relationship(id="r1", source_id="e1", target_id="e2", weight=1.0)
    graph.index_entities([existing_e])
    graph.index_relationships([existing_r])

    new_e = Entity(id="e1", name="Alice", description="new", text_unit_ids=["t1"])
    new_r = Relationship(id="r1", source_id="e1", target_id="e2", weight=2.0)

    merged_e, merged_r = mgr._merge_with_existing_graph([new_e], [new_r])

    assert len(merged_e) == 1
    # Merge unions text_unit_ids from existing + new.
    assert set(merged_e[0].text_unit_ids) >= {"t0", "t1"}
    assert len(merged_r) == 1


def test_merge_with_existing_graph_passthrough_when_nothing_existing(manager) -> None:
    mgr, _graph, _vector, _t = manager
    new_e = [Entity(id="e9", name="Bob")]
    new_r = [Relationship(id="r9", source_id="e9", target_id="e8")]

    merged_e, merged_r = mgr._merge_with_existing_graph(new_e, new_r)

    # No existing rows read back -> returns the inputs unchanged (overwrite mode).
    assert merged_e == new_e
    assert merged_r == new_r


def test_merge_with_existing_graph_handles_empty_inputs(manager) -> None:
    mgr, _graph, _vector, _t = manager
    merged_e, merged_r = mgr._merge_with_existing_graph(None, None)
    assert merged_e is None
    assert merged_r is None


# --- _log_completion_summary ---------------------------------------------


def test_log_completion_summary_aggregates_and_warns_on_failures(caplog) -> None:
    from aws_graphrag.application.storage.indexing_manager import IndexingManager

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

    from aws_graphrag.application.storage.indexing_manager import IndexingManager

    results = {
        "opensearch_text_units": IndexingStats(
            total_items=5, successful_items=5, failed_items=0
        ),
    }
    with caplog.at_level(logging.WARNING):
        IndexingManager._log_completion_summary(results, elapsed_time=0.1)

    assert "Failed items" not in caplog.text


def test_log_completion_summary_handles_empty_results() -> None:
    from aws_graphrag.application.storage.indexing_manager import IndexingManager

    # No items -> success_rate guard divides safely (no exception).
    IndexingManager._log_completion_summary({}, elapsed_time=0.0)


# --- _discover_suffixes --------------------------------------------------


def test_discover_suffixes_empty_returns_empty() -> None:
    from aws_graphrag.application.storage.indexing_manager import IndexingManager

    assert IndexingManager._discover_suffixes(None) == []
    assert IndexingManager._discover_suffixes([]) == []


def test_discover_suffixes_defaults_when_no_attributes() -> None:
    from aws_graphrag.application.storage.indexing_manager import IndexingManager

    units = [TextUnit(id="t1", text="a"), TextUnit(id="t2", text="b")]
    assert IndexingManager._discover_suffixes(units) == ["default"]
