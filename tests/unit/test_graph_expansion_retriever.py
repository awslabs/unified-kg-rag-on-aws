# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The backend-independent graph-expansion seam (``GraphExpansionRetriever``).

A subclass only supplies ``_traverse`` rows; seeding, ranking, de-duplication
and error typing come from the base, so a non-Gremlin backend gets the same
results as Neptune for the same rows.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from tests.fixtures.saturated_executor import (
    CPU_THREAD_PREFIX,
    saturated_default_executor,
)
from unified_kg_rag.adapters.retrievers.graph_expansion import GraphExpansionRetriever
from unified_kg_rag.adapters.retrievers.neptune_retriever import NeptuneRetriever
from unified_kg_rag.domain.models import Config, SearchQuery
from unified_kg_rag.shared import StoreConnectionError

pytestmark = pytest.mark.unit


def _row(node_id: str, path: list[str], **extra: Any) -> dict[str, Any]:
    return {
        "node": {"id": [node_id], "name": [node_id], "description": ["d"], **extra},
        "path": [{"name": [name]} for name in path],
        "node_type": "Entity-default",
    }


class _RowsRetriever(GraphExpansionRetriever):
    rows: list[dict[str, Any]] = []
    error: Exception | None = None

    async def _traverse(
        self, seed_ids: list[Any], query: SearchQuery
    ) -> list[dict[str, Any]]:
        if self.error is not None:
            raise self.error
        return list(self.rows)


def test_neptune_retriever_is_a_graph_expansion_retriever() -> None:
    assert issubclass(NeptuneRetriever, GraphExpansionRetriever)


def test_traverse_is_abstract(config: Config) -> None:
    with pytest.raises(TypeError):
        GraphExpansionRetriever(config)  # type: ignore[abstract]


async def test_rows_are_ranked_and_deduplicated(config: Config) -> None:
    retriever = _RowsRetriever(config)
    retriever.rows = [
        _row("seed", ["seed"]),
        _row("far", ["seed", "near", "far"]),
        _row("near", ["seed", "near"]),
        # Reached again from another seed by a longer path: the shorter wins.
        _row("near", ["other", "x", "near"]),
    ]
    query = SearchQuery(query="q", filters={"id": ["seed"]})
    results = await retriever.aretrieve(query)
    assert [r.source for r in results] == ["seed", "near", "far"]
    assert results[1].content.endswith("Path: seed -> near")
    assert all(r.metadata["_node_type"] == "Entity-default" for r in results)


async def test_the_rows_are_processed_on_the_cpu_pool(config: Config) -> None:
    # A hub's thousands of lineage ids are validated as its result is made:
    # on the CPU pool, so with every default-executor thread in a blocking
    # call (``asyncio.to_thread`` would queue it behind them) it still runs.
    threads: list[str] = []
    process = _RowsRetriever._process_traversal_results

    class _Recording(_RowsRetriever):
        def _process_traversal_results(self, *args: Any) -> Any:
            threads.append(threading.current_thread().name)
            return process(self, *args)

    retriever = _Recording(config)
    retriever.rows = [_row("seed", ["seed"]), _row("near", ["seed", "near"])]
    query = SearchQuery(query="q", filters={"id": ["seed"]})
    async with saturated_default_executor():
        results = await asyncio.wait_for(retriever.aretrieve(query), 5)
    assert [r.source for r in results] == ["seed", "near"]
    assert threads and all(name.startswith(CPU_THREAD_PREFIX) for name in threads)


async def test_no_seed_ids_skips_the_traversal(config: Config) -> None:
    retriever = _RowsRetriever(config)
    retriever.error = AssertionError("traversed without seeds")
    assert await retriever.aretrieve(SearchQuery(query="q")) == []


async def test_transient_error_degrades_and_fatal_error_raises(
    config: Config,
) -> None:
    retriever = _RowsRetriever(config)
    query = SearchQuery(query="q", filters={"id": ["seed"]})
    retriever.error = RuntimeError("transient")
    assert await retriever.aretrieve(query) == []
    retriever.error = StoreConnectionError("down")
    with pytest.raises(StoreConnectionError):
        await retriever.aretrieve(query)
