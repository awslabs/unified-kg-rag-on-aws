# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A failed read of a removal or a deletion is a failed item, nothing written.

``IndexingManager.remove_text_units`` and ``delete_documents`` read the
stores before they write; a read that fails must leave every store as it
was and report the items as failed, so the caller keeps the registry rows and
the next run retries. Runs on the port fakes (AWS-free); data is synthetic.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.models import Config, Constants, Entity, Relationship

pytestmark = pytest.mark.unit

DEFAULT = Constants.DEFAULT_SUFFIX.value


def _scoped(item: Any, suffix: str) -> Any:
    item.attributes = {**(item.attributes or {}), Constants.INDEX.value: suffix}
    return item


def _manager(*suffixes: str) -> tuple[IndexingManager, FakeGraphStore, FakeVectorStore]:
    config = Config()
    graph = FakeGraphStore(config)
    vectors = FakeVectorStore(config.indexing.opensearch)
    for suffix in suffixes:
        entities = [
            _scoped(Entity(id="e-vendor", name="Vendor", text_unit_ids=["t1"]), suffix),
            _scoped(
                Entity(id="e-buyer", name="Buyer", text_unit_ids=["t1", "t2"]), suffix
            ),
        ]
        edge = _scoped(
            Relationship(
                id="r1", source_id="e-vendor", target_id="e-buyer", text_unit_ids=["t1"]
            ),
            suffix,
        )
        graph.upsert_entities(entities)
        graph.upsert_relationships([edge])
        vectors.upsert_entities(entities)
        vectors.upsert_relationships([edge])
    manager = IndexingManager(
        config=config, vector_indexer=vectors, graph_indexer=graph
    )
    return manager, graph, vectors


def _snapshot(graph: FakeGraphStore, vectors: FakeVectorStore) -> Any:
    return copy.deepcopy(graph.data), copy.deepcopy(vectors.data)


def test_a_failed_citing_read_reports_every_removed_unit_and_writes_nothing() -> None:
    manager, graph, vectors = _manager(DEFAULT)
    before = _snapshot(graph, vectors)

    def fail(*_: Any, **__: Any) -> None:
        raise RuntimeError("graph read timed out")

    graph.find_ids_citing = fail  # type: ignore[method-assign]

    results = manager.remove_text_units(DEFAULT, ["t1", "t2"])

    assert set(results) == {f"find_removed_{DEFAULT}"}
    stats = results[f"find_removed_{DEFAULT}"]
    assert (stats.total_items, stats.failed_items) == (2, 2)
    # Named by its type, its text (a store's, which may quote a document)
    # at DEBUG only.
    assert "RuntimeError" in " ".join(stats.errors)
    assert "graph read timed out" not in " ".join(stats.errors)
    assert _snapshot(graph, vectors) == before


def test_a_removal_that_reads_its_citing_items_deletes_them() -> None:
    # The baseline of the test above: the same removal, with the read
    # working, does write (so "nothing written" is the failure's doing).
    manager, graph, vectors = _manager(DEFAULT)

    results = manager.remove_text_units(DEFAULT, ["t1"])

    assert all(not stats.failed_items for stats in results.values())
    assert "e-vendor" not in vectors.ids("entities")
    assert "e-buyer" in vectors.ids("entities")


def test_a_failed_incident_edge_read_fails_only_its_suffix() -> None:
    # Deleting without the incident edges would orphan their vector copies
    # for good: the suffix whose read failed deletes nothing and reports its
    # ids as failed; another suffix of the same call is deleted.
    manager, graph, vectors = _manager("tenant-a", "tenant-b")
    find = graph.find_incident_relationship_ids

    def fail_for_a(entity_ids: list[str], suffix: str | None = None) -> list[str]:
        if suffix == "tenant-a":
            raise RuntimeError("edge read timed out")
        return find(entity_ids, suffix=suffix)

    graph.find_incident_relationship_ids = fail_for_a  # type: ignore[method-assign]

    results = manager.delete_documents(
        {"tenant-a": ["e-vendor", "e-buyer"], "tenant-b": ["e-vendor"]}
    )

    failed = results["graph_read_tenant-a"]
    assert (failed.total_items, failed.failed_items) == (2, 2)
    assert "RuntimeError" in " ".join(failed.errors)
    assert "edge read timed out" not in " ".join(failed.errors)
    assert not any(
        key.endswith("tenant-a") for key in set(results) - {"graph_read_tenant-a"}
    )
    graph_entities = graph.data["entities"]
    assert ("tenant-a", "e-vendor") in graph_entities
    assert ("tenant-b", "e-vendor") not in graph_entities
    # The vector copies: tenant-a's kept, tenant-b's (and its incident edge)
    # deleted.
    assert vectors.delete_calls
    assert all(suffix == "tenant-b" for _, suffix in vectors.delete_calls)
    assert results["graph_delete_tenant-b"].failed_items == 0
