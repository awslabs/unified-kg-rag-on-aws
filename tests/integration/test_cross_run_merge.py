# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-run merge: delta upsert unions with existing graph state (AWS-free).

When ``indexing.cross_run_merge`` is on, IndexingManager.index_delta reads the
existing entities/relationships back from the graph store and merges them with
the delta (description / text_unit_ids union, frequency recompute) instead of
overwriting. Validated against the in-memory fakes (which support read-back).
"""

from __future__ import annotations

import pytest

from aws_graphrag.application.storage.indexing_manager import IndexingManager
from aws_graphrag.domain.models import Config, Entity
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore

pytestmark = pytest.mark.integration


def _manager(mocker, *, cross_run_merge: bool):
    config = Config()
    config.indexing.cross_run_merge = cross_run_merge
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    mocker.patch(
        "aws_graphrag.application.storage.indexing_manager.OpenSearchIndexer",
        return_value=vector,
    )
    mocker.patch(
        "aws_graphrag.application.storage.indexing_manager.NeptuneIndexer",
        return_value=graph,
    )
    return IndexingManager(config=config), graph


def test_cross_run_merge_unions_descriptions_and_text_units(mocker) -> None:
    mgr, graph = _manager(mocker, cross_run_merge=True)

    # Run 1: entity 'alice' seen in text unit t1 with description A.
    mgr.index_delta(
        entities=[Entity(id="e1", name="alice", description="A", text_unit_ids=["t1"])]
    )
    # Run 2: same entity (by name) seen in t2 with description B.
    mgr.index_delta(
        entities=[Entity(id="e1", name="alice", description="B", text_unit_ids=["t2"])]
    )

    stored = graph.data["entities"]["e1"]
    # Descriptions unioned (not overwritten) and text units accumulated.
    assert "A" in (stored.description or "") and "B" in (stored.description or "")
    assert set(stored.text_unit_ids or []) == {"t1", "t2"}


def test_without_flag_overwrites(mocker) -> None:
    mgr, graph = _manager(mocker, cross_run_merge=False)

    mgr.index_delta(
        entities=[Entity(id="e1", name="alice", description="A", text_unit_ids=["t1"])]
    )
    mgr.index_delta(
        entities=[Entity(id="e1", name="alice", description="B", text_unit_ids=["t2"])]
    )

    stored = graph.data["entities"]["e1"]
    # Overwrite semantics: only the latest delta's values remain.
    assert stored.description == "B"
    assert set(stored.text_unit_ids or []) == {"t2"}
