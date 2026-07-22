# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS-free tests for OpenSearchIndexer's blue/green index-write + alias swap.

``_index_item_type`` is the core full-index write orchestration — create a new
timestamped index, embed + prepare docs, bulk-index, then atomically swap the
alias onto the new index and reap stale indices; on failure it rolls back by
deleting the half-written index. A recording fake ``opensearch_client`` drives
the real method so the swap/cleanup/rollback ordering is verified without AWS.
"""

from __future__ import annotations

import pytest

from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.domain.models import Config, Entity

pytestmark = pytest.mark.unit


class _RecordingClient:
    """Records create/alias/cleanup calls; lets a test inject existing indices."""

    def __init__(self, existing_for_alias: list[str] | None = None) -> None:
        self.calls: list[tuple] = []
        self._existing = existing_for_alias or []
        self.created: list[str] = []
        self.bulk_should_fail = False

    def create_index(self, index_name, mapping):
        self.calls.append(("create_index", index_name))
        self.created.append(index_name)

    def update_alias(self, alias_name, index_name, remove_pattern=None):
        self.calls.append(("update_alias", alias_name, index_name, remove_pattern))

    def get_indices_by_alias(self, pattern):
        self.calls.append(("get_indices_by_alias", pattern))
        # The just-created index plus any pre-existing (stale) ones.
        return list(self.created) + self._existing

    def delete_indices(self, indices):
        self.calls.append(("delete_indices", tuple(indices)))

    def bulk_index(self, index_name, documents, **kwargs):
        self.calls.append(("bulk_index", index_name, len(documents)))
        if self.bulk_should_fail:
            raise RuntimeError("bulk failed")
        return {"errors": False, "items": []}


def _indexer(config: Config, client: _RecordingClient) -> OpenSearchIndexer:
    inst = OpenSearchIndexer.__new__(OpenSearchIndexer)
    inst.config = config
    inst.opensearch_config = config.indexing.opensearch
    inst.analyzer = "standard"
    inst.target_language = config.processing.translation.target_language.value
    inst._embedding_dimension = 1024
    inst.opensearch_client = client
    # Avoid Bedrock: deterministic embeddings + flush no-op.
    inst._batch_embed = lambda texts, batch_size=50: [[0.1] for _ in texts]
    inst._flush_embedding_cache = lambda: None
    return inst


def _op_names(calls: list[tuple]) -> list[str]:
    return [c[0] for c in calls]


def test_full_index_creates_then_swaps_alias_then_cleans_stale(config) -> None:
    # One stale index pre-exists for the alias; after a successful write the new
    # index is created, alias swapped onto it, and the stale one reaped.
    client = _RecordingClient(existing_for_alias=["graphrag-entities-default-OLD"])
    indexer = _indexer(config, client)

    stats = indexer.index_entities([Entity(id="e1", name="Alice")])

    ops = _op_names(client.calls)
    # Order: create_index -> bulk_index -> update_alias -> get_indices_by_alias
    # -> delete_indices.
    assert ops.index("create_index") < ops.index("bulk_index")
    assert ops.index("bulk_index") < ops.index("update_alias")
    assert ops.index("update_alias") < ops.index("delete_indices")

    # Alias swap targets the newly created index and removes the old pattern.
    swap = next(c for c in client.calls if c[0] == "update_alias")
    assert swap[2] == client.created[0]
    assert swap[3] is not None  # remove_pattern set

    # The stale index (not the new one) is the one deleted.
    delete = next(c for c in client.calls if c[0] == "delete_indices")
    assert "graphrag-entities-default-OLD" in delete[1]
    assert client.created[0] not in delete[1]
    assert stats.successful_items == 1


def test_first_write_has_no_stale_index_to_clean(config) -> None:
    client = _RecordingClient(existing_for_alias=[])
    indexer = _indexer(config, client)
    indexer.index_entities([Entity(id="e1", name="Alice")])
    # get_indices_by_alias returns only the new index, so nothing is deleted.
    assert not any(c[0] == "delete_indices" for c in client.calls)


def test_bulk_failure_does_not_swap_alias(config) -> None:
    # A bulk-index failure is absorbed by _perform_indexing (returns failed
    # stats, does not raise), so the alias is NOT swapped onto an index with
    # zero successful docs — the live alias keeps pointing at the prior index.
    client = _RecordingClient(existing_for_alias=[])
    client.bulk_should_fail = True
    indexer = _indexer(config, client)

    stats = indexer.index_entities([Entity(id="e1", name="Alice")])

    assert "update_alias" not in _op_names(client.calls)
    assert stats.successful_items == 0
    assert stats.failed_items >= 1


def test_create_index_failure_rolls_back(config) -> None:
    # An exception from the write machinery (here: create_index) triggers the
    # except branch, which deletes the half-written index so no orphan is left.
    client = _RecordingClient(existing_for_alias=[])

    def _boom_create(index_name, mapping):
        client.created.append(index_name)  # the name was allocated before failing
        client.calls.append(("create_index", index_name))
        raise RuntimeError("create failed")

    client.create_index = _boom_create
    indexer = _indexer(config, client)

    stats = indexer.index_entities([Entity(id="e1", name="Alice")])

    delete = next(c for c in client.calls if c[0] == "delete_indices")
    assert client.created[0] in delete[1]
    assert "update_alias" not in _op_names(client.calls)
    assert stats.failed_items >= 1
