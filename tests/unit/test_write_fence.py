# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run that lost its write lease stops writing (AWS-free).

The lease trusts itself only from when a renewal was sent, keeps a third of
its term in reserve, and is the fence every write path checks: the indexing
phases, the rebuild's sweep, a Neptune batch or drop and an OpenSearch bulk
request.
"""

from __future__ import annotations

import signal
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

import unified_kg_rag.adapters.aws.opensearch as opensearch_mod
import unified_kg_rag.application.ingestion.namespace_lease as lease_mod
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from unified_kg_rag.adapters.aws.opensearch import OpenSearchClient
from unified_kg_rag.application.ingestion.namespace_lease import (
    NamespaceLease,
    check_lease,
)
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.storage.indexing_manager import IndexingManager
from unified_kg_rag.domain.models import Config, Entity, TextUnit
from unified_kg_rag.shared import NamespaceLeaseError, PipelineInterrupted
from unified_kg_rag.shared.write_fence import (
    install_write_fence,
    remove_write_fence,
)

pytestmark = pytest.mark.unit


class _SlowStore(FakeDocStatusStore):
    """Lease renewals that follow ``plan``: a number takes the lease when
    sent (as DynamoDB does) and answers that many seconds later; ``None``
    hangs until ``unreachable`` is set and then fails, taking nothing."""

    def __init__(self) -> None:
        super().__init__()
        self.plan: list[float | None] = []
        self.unreachable = threading.Event()

    def try_acquire_lease(self, namespace: str, owner: str, seconds: float) -> bool:
        if not self.plan:
            return super().try_acquire_lease(namespace, owner, seconds)
        delay = self.plan[0]
        if delay is None:
            self.unreachable.wait(30)
            raise RuntimeError("registry unreachable")
        held = super().try_acquire_lease(namespace, owner, seconds)
        self.plan.pop(0)
        time.sleep(delay)
        return held


def test_the_run_stops_writing_while_it_still_holds_the_lease() -> None:
    # Old behaviour: the renewal time was taken when a slow renewal answered,
    # and check_lease passed until a full term after that, so it still passed
    # once the store had expired the lease and another run could take it.
    store = _SlowStore()
    with NamespaceLease(store, "ns", 1.2) as lease:
        # The next renewal answers 0.7 s late; the registry is then gone.
        store.plan = [0.7, None]
        while store.plan != [None]:
            time.sleep(0.01)
        expiry = store.leases["ns"][1]
        deadline = time.time() + 5
        while time.time() < deadline:
            try:
                check_lease()
            except NamespaceLeaseError:
                break
            time.sleep(0.01)
        stopped_at = time.time()
        # Stopped while the store still holds the lease: no other run can
        # have taken it.
        assert stopped_at < expiry
        assert not FakeDocStatusStore.try_acquire_lease(store, "ns", "intruder", 60)
        lease._stop.set()
        store.unreachable.set()


def test_ending_a_run_does_not_wait_for_a_hung_renewal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Old behaviour: the run waited for the renewal in flight, as long as the
    # registry client retried it.
    monkeypatch.setattr(lease_mod, "_HEARTBEAT_JOIN_SECONDS", 0.1, raising=False)
    store = _SlowStore()
    lease = NamespaceLease(store, "ns", 0.6)
    with lease:
        store.plan = [2.0]
        while store.plan:
            time.sleep(0.01)  # the slow renewal was sent
        started = time.time()
    assert time.time() - started < 1
    # The renewal in flight took the lease again; once it answers, the
    # heartbeat releases it.
    assert lease._thread is not None
    lease._thread.join(5)
    assert store.lease_holder("ns") is None


class _Fence:
    def __init__(self, allowed: int) -> None:
        self.allowed = allowed
        self.checks = 0

    def check(self) -> None:
        self.checks += 1
        if self.checks > self.allowed:
            raise NamespaceLeaseError("lost")


def _fenced(fence: _Fence):  # type: ignore[no-untyped-def]
    return install_write_fence(fence)


def test_a_lost_lease_stops_a_rebuild_before_its_sweep() -> None:
    # Old behaviour: the sweep ran without a check and dropped what the new
    # holder had written meanwhile.
    config = Config()
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    manager.index_all_data(
        entities=[Entity(id="e-old", name="Old", type="ORG", text_unit_ids=["u0"])],
        text_units=[TextUnit(id="u0", text="old")],
    )
    sweeps: list[bool] = []
    end_rebuild = graph.end_rebuild

    def record(suffixes: Any, sweep: bool = True) -> None:
        sweeps.append(sweep)
        end_rebuild(suffixes, sweep=sweep)

    graph.end_rebuild = record  # type: ignore[method-assign]
    # Before phase 1 and phase 2 pass; before the sweep the lease is lost.
    token = _fenced(_Fence(allowed=2))
    try:
        with pytest.raises(NamespaceLeaseError):
            manager.index_all_data(
                entities=[
                    Entity(id="e-new", name="New", type="ORG", text_unit_ids=["u1"])
                ],
                text_units=[TextUnit(id="u1", text="new")],
            )
    finally:
        remove_write_fence(token)
    assert sweeps == [False]
    assert "e-old" in graph.ids("entities")


def test_a_lost_lease_stops_the_next_neptune_batch(mocker) -> None:
    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer.NeptuneClient")
    from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer

    indexer = NeptuneIndexer(config=Config())
    sent = mocker.patch.object(indexer, "_execute_with_retries")
    token = _fenced(_Fence(allowed=0))
    try:
        with pytest.raises(NamespaceLeaseError):
            indexer._execute_batch_traversal(
                [1, 2, 3], lambda g, batch: MagicMock(), "Indexing entities"
            )
    finally:
        remove_write_fence(token)
    sent.assert_not_called()


def test_a_lost_lease_stops_the_next_opensearch_bulk_request(mocker) -> None:
    client = OpenSearchClient.__new__(OpenSearchClient)
    client._client = MagicMock()
    requests: list[int] = []

    def streaming_bulk(client, actions, chunk_size, **kwargs):  # type: ignore[no-untyped-def]
        chunk: list[Any] = []
        for action in actions:
            chunk.append(action)
            if len(chunk) == chunk_size:
                requests.append(len(chunk))
                yield from ((True, {}) for _ in chunk)
                chunk = []
        if chunk:
            requests.append(len(chunk))
            yield from ((True, {}) for _ in chunk)

    mocker.patch.object(opensearch_mod, "streaming_bulk", side_effect=streaming_bulk)
    mocker.patch.object(OpenSearchClient, "BULK_CHUNK_SIZE", 2)
    token = _fenced(_Fence(allowed=1))
    try:
        with pytest.raises(NamespaceLeaseError):
            client.bulk_index("idx", [{"id": str(i)} for i in range(5)])
    finally:
        remove_write_fence(token)
    assert requests == [2]


def test_a_stopped_run_uploads_its_checkpoint_before_releasing_the_lease(
    tmp_path: Path, mocker
) -> None:
    # Old behaviour: the lease was released first, so the next run could
    # start from the checkpoint the stopping run had not uploaded yet.
    store = FakeDocStatusStore()
    pipe = object.__new__(DataIngestionPipeline)
    pipe.config = Config()
    pipe._doc_status = store
    pipe.source_directory = tmp_path
    pipe.pipeline_config = SimpleNamespace(
        s3_sync_enabled=True,
        pipeline_id=None,
        resume_from_stage=None,
        force_rebuild=False,
    )
    pipe.cache_manager = mocker.MagicMock()
    pipe.cache_manager.get_pipeline_cache_dir.return_value = tmp_path
    pipe.s3_cache_manager = mocker.MagicMock()
    pipe.state_manager = mocker.MagicMock()
    pipe.state_manager.pipeline_exists.return_value = False
    pipe._prepare_new_run = mocker.MagicMock(
        return_value=(SimpleNamespace(incremental_delta=None), None)
    )
    for name in ("_set_corpus_fingerprint", "_validate_retired_scopes"):
        mocker.patch.object(pipe, name)
    mocker.patch.object(pipe, "_use_source_directory")
    held_during_upload: list[bool] = []
    pipe.s3_cache_manager.sync_pipeline_to_s3.side_effect = (
        lambda *_: held_during_upload.append(bool(store.leases))
    )

    def execute(context, start):  # type: ignore[no-untyped-def]
        pipe.request_stop(signal.SIGTERM)

    mocker.patch.object(pipe, "_execute_pipeline", side_effect=execute)
    with pytest.raises(PipelineInterrupted):
        pipe.run(tmp_path, pipeline_id="pid")
    assert held_during_upload == [True]
    assert not store.leases


def test_a_lost_lease_stops_the_deletion_of_retired_ids() -> None:
    # The re-keyed items' old ids, merged-away edges and folded delta ids are
    # deleted after the commit's writes: a run that lost its lease meanwhile
    # must not delete what the new holder wrote under them.
    config = Config()
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    graph.upsert_entities([Entity(id="e-old", name="Old")])
    manager._retire("default", {"e-old": "e-new"})
    token = _fenced(_Fence(allowed=0))
    try:
        with pytest.raises(NamespaceLeaseError):
            manager.delete_retired_ids()
    finally:
        remove_write_fence(token)
    assert graph.ids("entities") == {"e-old"}


def test_a_lost_lease_stops_the_next_neptune_edge_write(mocker) -> None:
    # The relationship writes (endpoint check, drop, per-edge add) and the
    # drop by id do not go through the batch path.
    from unified_kg_rag.domain.models import Relationship

    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer.NeptuneClient")
    from unified_kg_rag.adapters.storage.neptune_indexer import NeptuneIndexer

    indexer = NeptuneIndexer(config=Config())
    sent = mocker.patch.object(indexer, "_execute_with_retries")
    token = _fenced(_Fence(allowed=0))
    try:
        with pytest.raises(NamespaceLeaseError):
            indexer.upsert_relationships(
                [Relationship(id="r", source_id="a", target_id="b")]
            )
        with pytest.raises(NamespaceLeaseError):
            indexer.delete_by_id(["a"], suffix="default")
    finally:
        remove_write_fence(token)
    sent.assert_not_called()
