# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regressions for the cycle-34 dead-code and duplication cleanup (AWS-free)."""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.local.sqlite_graph import SQLiteGraphIndexer
from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer
from unified_kg_rag.ports import GraphIndexer

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("cls", [SQLiteGraphIndexer, NeptuneIndexer])
def test_graph_indexers_share_the_port_index_relationships(cls: type) -> None:
    """A full run writes edges through the same path as a delta."""
    assert cls.index_relationships is GraphIndexer.index_relationships


def test_graph_index_relationships_delegates_to_upsert(mocker) -> None:
    upsert = mocker.Mock(return_value="stats")
    store = mocker.Mock(upsert_relationships=upsert)
    assert GraphIndexer.index_relationships(store, ["edge"]) == "stats"
    upsert.assert_called_once_with(["edge"])
