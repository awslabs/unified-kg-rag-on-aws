# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run that lost its write lease stops writing (AWS-free).

The lease trusts itself only from when a renewal was sent, keeps a third of
its term in reserve, and is the fence every write path checks: the indexing
phases, the rebuild's sweep, a Neptune batch or drop and an OpenSearch bulk
request.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import signal
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from opensearchpy import RequestsHttpConnection
from opensearchpy.exceptions import ConnectionError as OpenSearchConnectionError

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
from unified_kg_rag.domain.models import (
    Config,
    Entity,
    PipelineStageStatus,
    TextUnit,
)
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
        # The next renewal answers 0.5 s late (before the run's own check
        # would trip on the previous one); the registry is then gone.
        store.plan = [0.5, None]
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
    sent = mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer.run_traversal")
    token = _fenced(_Fence(allowed=0))
    try:
        with pytest.raises(NamespaceLeaseError):
            indexer._execute_batch_traversal(
                [1, 2, 3], lambda g, batch: MagicMock(), "Indexing entities"
            )
    finally:
        remove_write_fence(token)
    sent.assert_not_called()


@pytest.mark.parametrize("phase", ["rebuild", "retain"])
def test_the_embedded_engine_swaps_no_alias_after_a_lost_lease(
    tmp_path: Path, phase: str
) -> None:
    # The embedded search engine's requests do not check the fence (only
    # OpenSearch's connection does): the indexer's own check before the swap
    # is what keeps a run that lost its lease from swapping in its index.
    from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
    from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
    from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer

    engine = SQLiteSearchEngine(tmp_path / "vectors.sqlite3")
    indexer = OpenSearchIndexer(
        Config(),
        embedding_factory=HashingEmbeddingFactory(dimensions=8),
        opensearch_client=engine,
    )
    indexer.index_text_units(
        [TextUnit(id="old", text="old text")], suffixes=["default"]
    )
    indexer.write_extraction_records({"k1": {"a": 1}}, suffix="default")
    prefix = (
        indexer.opensearch_config.text_units_index_prefix
        if phase == "rebuild"
        else indexer.opensearch_config.extraction_records_index_prefix
    )
    alias = indexer._get_name(prefix, "default")
    live = engine.get_indices_by_alias(alias)

    token = _fenced(_Fence(allowed=0))
    try:
        with pytest.raises(NamespaceLeaseError):
            if phase == "rebuild":
                indexer.index_text_units(
                    [TextUnit(id="new", text="new text")], suffixes=["default"]
                )
            else:
                indexer.retain_extraction_records([], suffix="default")
    finally:
        remove_write_fence(token)
    assert engine.get_indices_by_alias(alias) == live


@pytest.mark.parametrize("cleanup", ["reap", "drop"])
def test_a_lost_lease_is_not_swallowed_by_the_index_cleanup(
    mocker, cleanup: str
) -> None:
    # Both cleanups are best-effort, but a lost lease must stop the run there
    # rather than be logged as a failed delete.
    from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer

    indexer = object.__new__(OpenSearchIndexer)
    indexer.opensearch_client = MagicMock()
    indexer.opensearch_client.get_aliases_by_index.return_value = {
        "a-20260101000000000": {}
    }
    indexer.opensearch_client.get_indices_by_alias.return_value = []
    indexer.opensearch_client.delete_indices.side_effect = NamespaceLeaseError("lost")
    with pytest.raises(NamespaceLeaseError):
        if cleanup == "reap":
            indexer._reap_stale_indices("a", "a-20260102000000000")
        else:
            indexer._drop_unswapped_index("a", "a-20260102000000000")


def _opensearch_client(mocker) -> OpenSearchClient:  # type: ignore[no-untyped-def]
    config = Config()
    config.aws.opensearch.endpoint = "localhost"
    config.aws.opensearch.port = 1
    config.aws.opensearch.use_iam = False
    config.aws.opensearch.use_ssl = False
    config.aws.opensearch.allow_anonymous = True
    mocker.patch.object(opensearch_mod.OpenSearch, "ping", return_value=True)
    mocker.patch.object(opensearch_mod.OpenSearch, "info", return_value={})
    return OpenSearchClient(config)


def test_a_lost_lease_stops_the_opensearch_transport_retries(mocker) -> None:
    # Old behaviour: the fence was checked once per bulk request, and the
    # transport then made up to 6 attempts of 180 s each without checking.
    client = _opensearch_client(mocker)
    sent: list[str] = []

    def unreachable(self, method, url, *args, **kwargs):  # type: ignore[no-untyped-def]
        sent.append(method)
        raise OpenSearchConnectionError("N/A", "unreachable", Exception())

    mocker.patch.object(RequestsHttpConnection, "perform_request", unreachable)
    token = _fenced(_Fence(allowed=2))
    try:
        with pytest.raises(NamespaceLeaseError):
            client.bulk_index("idx", [{"id": str(i)} for i in range(5)])
        # Reads do not check it.
        with pytest.raises(Exception, match="unreachable"):
            client.get_indices_by_alias("alias")
    finally:
        remove_write_fence(token)
    assert sent[:2] == ["POST", "POST"]
    assert sent.count("POST") == 2


def test_a_lost_lease_stops_a_full_build_before_its_alias_swap(tmp_path: Path) -> None:
    # Old behaviour: the indexer recorded the lost lease as failed items,
    # swapped the alias onto the partial index and deleted the complete one.
    from tests.fixtures.fakes.embeddings import HashingEmbeddingFactory
    from unified_kg_rag.adapters.local.search_engine import SQLiteSearchEngine
    from unified_kg_rag.adapters.storage.opensearch_indexer import (
        _WRITE_SLICE_SIZE,
        OpenSearchIndexer,
    )
    from unified_kg_rag.shared.write_fence import check_write_fence

    class _FencedEngine(SQLiteSearchEngine):
        def bulk_index(self, *args: Any, **kwargs: Any) -> Any:
            check_write_fence()  # as the OpenSearch client's requests do
            return super().bulk_index(*args, **kwargs)

    engine = _FencedEngine(tmp_path / "vectors.sqlite3")
    indexer = OpenSearchIndexer(
        Config(),
        embedding_factory=HashingEmbeddingFactory(dimensions=8),
        opensearch_client=engine,
    )
    count = _WRITE_SLICE_SIZE * 2
    indexer.index_text_units(
        [TextUnit(id=f"old{i}", text=f"old text {i}") for i in range(count)],
        suffixes=["default"],
    )
    alias = indexer._get_name(
        indexer.opensearch_config.text_units_index_prefix, "default"
    )
    live = engine.get_indices_by_alias(alias)
    indexer.write_extraction_records({"k1": {"a": 1}}, suffix="default")
    records = indexer._get_name(
        indexer.opensearch_config.extraction_records_index_prefix, "default"
    )
    live_records = engine.get_indices_by_alias(records)

    token = _fenced(_Fence(allowed=1))  # lost after the first slice
    try:
        with pytest.raises(NamespaceLeaseError):
            indexer.index_text_units(
                [TextUnit(id=f"new{i}", text=f"new text {i}") for i in range(count)],
                suffixes=["default"],
            )
    finally:
        remove_write_fence(token)
    token = _fenced(_Fence(allowed=0))
    try:
        with pytest.raises(NamespaceLeaseError):
            indexer.retain_extraction_records(["k1"], suffix="default")
        with pytest.raises(NamespaceLeaseError):
            indexer.upsert_text_units([TextUnit(id="u", text="text")])
        with pytest.raises(NamespaceLeaseError):
            indexer.write_extraction_records({"k2": {"a": 2}}, suffix="default")
    finally:
        remove_write_fence(token)

    assert engine.get_indices_by_alias(alias) == live
    hits = engine.search(index=alias, body={"query": {"match_all": {}}, "size": 1})
    assert hits["hits"]["total"]["value"] == count
    assert engine.get_indices_by_alias(records) == live_records


def test_a_lost_lease_stops_the_neptune_retries(mocker) -> None:
    # Old behaviour: the fence was checked once before a write, which was
    # then retried (neptune.max_attempts, up to 180 s each) without a check,
    # and a batch's failure fell back to writing its items one by one.
    mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer.NeptuneClient")
    from unified_kg_rag.adapters.storage import neptune_indexer as neptune_mod

    config = Config()
    config.indexing.neptune.retry_delay_seconds = 0
    indexer = neptune_mod.NeptuneIndexer(config=config)
    sent: list[int] = []

    def timed_out(*args: Any, **kwargs: Any) -> Any:
        sent.append(1)
        raise TimeoutError("Neptune did not answer")

    mocker.patch.object(neptune_mod, "run_traversal", side_effect=timed_out)
    # The first attempt's check passes; the lease is lost before the retry.
    token = _fenced(_Fence(allowed=1))
    try:
        with pytest.raises(NamespaceLeaseError):
            indexer._execute_batch_traversal(
                [1, 2, 3], lambda g, batch: MagicMock(), "Indexing entities"
            )
    finally:
        remove_write_fence(token)
    assert len(sent) == 1


def _stopping_pipeline() -> DataIngestionPipeline:
    pipe = object.__new__(DataIngestionPipeline)
    pipe._stop_signal = None
    pipe._checkpointing = False
    return pipe


def test_a_stop_ends_the_indexing_phase_at_the_next_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Old behaviour: the main thread's PipelineInterrupted waited in the
    # pool's exit for every task of the phase to finish (> the task's 120 s
    # stop timeout on a real phase), and the tasks kept writing.
    from unified_kg_rag.application.ingestion.pipeline import _StopFence
    from unified_kg_rag.application.storage.indexing_manager import IndexingTask
    from unified_kg_rag.ports import IndexingStats
    from unified_kg_rag.shared.write_fence import check_write_fence

    config = Config()
    manager = IndexingManager(
        config=config,
        vector_indexer=FakeVectorStore(opensearch_config=config.indexing.opensearch),
        graph_indexer=FakeGraphStore(),
    )
    written: list[float] = []

    def write(items: list[int]) -> IndexingStats:
        for _ in items:
            check_write_fence()  # each store request checks the fence
            time.sleep(0.2)
            written.append(time.time())
        return IndexingStats(total_items=len(items), successful_items=len(items))

    pipe = _stopping_pipeline()
    previous = signal.signal(
        signal.SIGALRM, lambda signum, _: pipe.request_stop(signal.SIGTERM)
    )
    token = install_write_fence(_StopFence(pipe))
    tasks = [IndexingTask(write, [list(range(20))], f"t{i}") for i in range(3)]
    started = time.time()
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        with pytest.raises(PipelineInterrupted):
            manager._run_indexing_phase(tasks)
        stopped = time.time()
        time.sleep(0.5)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        remove_write_fence(token)
    assert stopped - started < 1.5
    # Each task finished at most the request it was sending.
    assert len(written) <= 3 * 4
    assert not [at for at in written if at > stopped + 0.3]


def test_the_stop_fence_lets_the_main_thread_unwind() -> None:
    from unified_kg_rag.application.ingestion.pipeline import _StopFence

    pipe = _stopping_pipeline()
    pipe._stop_signal = signal.SIGTERM
    fence = _StopFence(pipe)
    fence.check()  # the main thread closes the attempts, uploads the checkpoint
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            fence.check()
        except BaseException as e:  # noqa: BLE001 - collected
            errors.append(e)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert [type(e) for e in errors] == [PipelineInterrupted]


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
    sent = mocker.patch("unified_kg_rag.adapters.storage.neptune_indexer.run_traversal")
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


@pytest.mark.parametrize("s3", [False, True])
def test_a_lost_lease_stops_the_run_at_the_next_stage(mocker, s3: bool) -> None:
    # Old behaviour: a run that lost its lease in a stage that does not
    # write (extraction) only noticed at the next store write: it started
    # the next stage and uploaded its checkpoint in between.
    pipe = _stopping_pipeline()
    pipe.stages = [SimpleNamespace(name="chunking"), SimpleNamespace(name="indexing")]
    pipe.s3_cache_manager = mocker.MagicMock() if s3 else None
    pipe.cache_manager = mocker.MagicMock()
    pipe.llm_response_cache = None
    executed: list[str] = []
    mocker.patch.object(
        pipe,
        "_execute_stage",
        side_effect=lambda stage, context: executed.append(stage.name)
        or SimpleNamespace(status=None),
    )
    mocker.patch.object(pipe, "_emit_stage_metrics")
    mocker.patch.object(pipe, "_should_stop_pipeline", return_value=False)
    mocker.patch.object(pipe, "_start_index", return_value=0)
    upload = mocker.patch.object(
        pipe, "_sync_cache_with_s3", wraps=pipe._sync_cache_with_s3
    )
    context = SimpleNamespace(stage_results=[], pipeline_id="pid")
    token = _fenced(_Fence(allowed=1))  # lost while the first stage runs
    try:
        with pytest.raises(NamespaceLeaseError):
            pipe._execute_pipeline_stages(context, None)  # type: ignore[arg-type]
    finally:
        remove_write_fence(token)
    assert executed == ["chunking"]
    if s3:
        pipe.s3_cache_manager.sync_pipeline_to_s3.assert_not_called()
    assert upload.call_count == int(s3)


def test_a_run_that_lost_its_lease_uploads_no_checkpoint_on_stop(mocker) -> None:
    pipe = _stopping_pipeline()
    pipe._checkpoint_id = "pid"
    upload = mocker.patch.object(pipe, "_upload_checkpoint")
    token = _fenced(_Fence(allowed=0))
    try:
        pipe._checkpoint_on_stop()
    finally:
        remove_write_fence(token)
    upload.assert_not_called()


def test_a_lease_under_three_times_the_longest_request_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipe = _stopping_pipeline()
    pipe.config = Config()
    pipe._doc_status = FakeDocStatusStore()
    with caplog.at_level("WARNING"):
        pipe._namespace_lease()
    assert "lease_seconds" not in caplog.text  # the default is long enough
    pipe.config.indexing.lease_seconds = 300
    with caplog.at_level("WARNING"):
        pipe._namespace_lease()
    assert "under three times the longest store write request" in caplog.text


def test_a_released_lease_fails_the_writes_of_orphaned_workers() -> None:
    # Old behaviour: the fence of the released lease kept passing in the
    # worker threads a failed block left running (their context still holds
    # it), for up to two thirds of a term after another run took the lease.
    from unified_kg_rag.shared.write_fence import check_write_fence

    store = FakeDocStatusStore()
    with NamespaceLease(store, "ns", 60):
        worker_context = contextvars.copy_context()
    assert FakeDocStatusStore.try_acquire_lease(store, "ns", "next-run", 60)
    with pytest.raises(NamespaceLeaseError, match="released"):
        worker_context.run(check_write_fence)


def _leased_pipeline(
    tmp_path: Path, mocker, store: FakeDocStatusStore
) -> DataIngestionPipeline:
    """A pipeline whose ``run`` takes ``store``'s lease and calls a mocked
    ``_execute_pipeline`` (patch it), with no S3 and no incremental delta."""
    pipe = object.__new__(DataIngestionPipeline)
    pipe.config = Config()
    pipe._doc_status = store
    pipe.source_directory = tmp_path
    pipe.pipeline_config = SimpleNamespace(
        s3_sync_enabled=False,
        pipeline_id=None,
        resume_from_stage=None,
        force_rebuild=False,
    )
    pipe.llm_response_cache = None
    pipe.state_manager = mocker.MagicMock()
    pipe.state_manager.pipeline_exists.return_value = False
    pipe._prepare_new_run = mocker.MagicMock(
        return_value=(
            SimpleNamespace(
                incremental_delta=None,
                status=PipelineStageStatus.COMPLETED,
                pipeline_id="pid",
                stage_results=[],
            ),
            None,
        )
    )
    for name in (
        "_set_corpus_fingerprint",
        "_validate_retired_scopes",
        "_use_source_directory",
    ):
        mocker.patch.object(pipe, name)
    return pipe


def test_ctrl_c_stops_the_workers_writes_before_the_lease_is_released(
    tmp_path: Path, mocker
) -> None:
    # Old behaviour: a KeyboardInterrupt did not set the stop fence, so the
    # worker a phase left running kept writing until (and, before the
    # lease's release marked it lost, after) the lease was released.
    from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor
    from unified_kg_rag.shared.write_fence import check_write_fence

    store = FakeDocStatusStore()
    pipe = _leased_pipeline(tmp_path, mocker, store)
    stopped = threading.Event()
    writes_after_stop: list[int] = []

    def worker() -> None:
        for _ in range(150):
            check_write_fence()
            if stopped.is_set():
                writes_after_stop.append(1)
            time.sleep(0.02)

    worker_future: list[Any] = []

    def execute(context, start):  # type: ignore[no-untyped-def]
        with ContextThreadPoolExecutor(max_workers=1) as pool:
            worker_future.append(pool.submit(worker))
            time.sleep(0.1)
            raise KeyboardInterrupt

    mocker.patch.object(pipe, "_execute_pipeline", side_effect=execute)
    # The run unwinds (closes its attempts, uploads its checkpoint) while
    # the worker still runs: the stop fence alone must end its writes.
    mocker.patch.object(
        pipe,
        "_checkpoint_on_stop",
        side_effect=lambda: concurrent.futures.wait(worker_future, timeout=2),
    )
    release = store.release_lease

    def released(namespace: str, owner: str) -> None:
        stopped.set()
        release(namespace, owner)

    mocker.patch.object(store, "release_lease", side_effect=released)
    with pytest.raises(KeyboardInterrupt):
        pipe.run(tmp_path, pipeline_id="pid")
    error = worker_future[0].exception(timeout=5)
    assert type(error) is PipelineInterrupted
    assert writes_after_stop == []


def test_the_stop_fence_exempts_the_thread_that_runs_the_pipeline() -> None:
    # Old behaviour: it exempted the main thread, so a pipeline run from
    # another thread failed its own unwinding writes (closing the attempts)
    # while the main thread's writes passed.
    from unified_kg_rag.application.ingestion.pipeline import _StopFence

    pipe = _stopping_pipeline()
    pipe._stop_signal = signal.SIGTERM
    outcome: list[Any] = []

    def run() -> None:  # a pipeline run in a worker thread
        fence = _StopFence(pipe)
        fence.check()  # its own unwinding writes pass
        outcome.append(fence)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join()
    assert len(outcome) == 1
    with pytest.raises(PipelineInterrupted):
        outcome[0].check()  # any other thread's, the main thread's included


def test_a_stop_does_not_wait_for_the_model_calls_in_flight() -> None:
    # Old behaviour: the batch's finally waited for the running calls, so a
    # SIGTERM waited for the slowest model call in flight (up to the call
    # timeout, past the task's stop timeout).
    from unified_kg_rag.shared.utils.langchain import BatchProcessor

    def call(_: dict[str, Any]) -> str:
        time.sleep(3)
        return "ok"

    def stop(signum: int, _: Any) -> None:
        raise PipelineInterrupted(signum)

    previous = signal.signal(signal.SIGALRM, stop)
    started = time.time()
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.3)
        with pytest.raises(PipelineInterrupted):
            BatchProcessor(max_concurrency=2).execute_with_fallback(
                list(range(4)),
                lambda xs: [{"x": x} for x in xs],
                call,
                "extraction",
                show_progress=False,
            )
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert time.time() - started < 1.5


def _slow_match(name: str) -> list[tuple[str, float]]:
    time.sleep(0.2)
    return []


def _slow_minhash(item: tuple[int, str], **_: Any) -> None:
    time.sleep(0.2)
    return None


class _RecordingProcessPool(ProcessPoolExecutor):
    """A real process pool that keeps the futures it hands out."""

    pools: list[_RecordingProcessPool] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.futures: list[concurrent.futures.Future[Any]] = []
        self.pools.append(self)

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        future = super().submit(fn, *args, **kwargs)
        self.futures.append(future)
        return future


def _stopped_after(seconds: float, block: Any) -> None:
    def stop(signum: int, _: Any) -> None:
        raise PipelineInterrupted(signum)

    previous = signal.signal(signal.SIGALRM, stop)
    try:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        with pytest.raises(PipelineInterrupted):
            block()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _assert_queued_tasks_cancelled(workers: int) -> None:
    (pool,) = _RecordingProcessPool.pools
    _RecordingProcessPool.pools = []
    try:
        ran = sum(f.done() and not f.cancelled() for f in pool.futures)
        # Only the running tasks and the few the pool already handed to its
        # workers' queue are not cancellable; the pool's manager thread
        # cancels the rest once it wakes (the old code ran them all).
        expected = len(pool.futures) - ran - 2 * workers - 1
        deadline = time.time() + 3
        while time.time() < deadline and (
            sum(f.cancelled() for f in pool.futures) < expected
        ):
            time.sleep(0.05)
        assert sum(f.cancelled() for f in pool.futures) >= max(expected, 1)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def test_a_stop_cancels_the_queued_tasks_of_the_resolver_process_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Old behaviour: the pool's own with-block exit (shutdown(wait=True))
    # reset the cancellation, so none of the queued names was cancelled and
    # they all ran before the interpreter could exit.
    from unified_kg_rag.domain.ingestion import graph_resolver

    monkeypatch.setattr(graph_resolver, "find_all_matches_for_entity_task", _slow_match)
    monkeypatch.setattr(graph_resolver, "ProcessPoolExecutor", _RecordingProcessPool)
    resolver = graph_resolver.EntityResolver(Config(), 2, True)

    def resolve() -> None:
        for _ in resolver._match_in_pool(MagicMock(), [f"n{i}" for i in range(40)]):
            pass

    _stopped_after(0.5, resolve)
    _assert_queued_tasks_cancelled(workers=2)


def test_a_stop_cancels_the_queued_tasks_of_the_dedup_process_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from unified_kg_rag.adapters.ingestion import loader as loader_mod
    from unified_kg_rag.domain.models import Document

    monkeypatch.setattr(loader_mod, "compute_minhash", _slow_minhash)
    monkeypatch.setattr(loader_mod, "ProcessPoolExecutor", _RecordingProcessPool)
    loader = loader_mod.DirectoryLoader(tmp_path)
    loader.max_workers = 2
    docs = [
        Document(
            page_content=f"text {i}",
            document_id=f"d{i}",
            file_name=f"d{i}.txt",
            file_path=str(tmp_path / f"d{i}.txt"),
        )
        for i in range(40)
    ]

    _stopped_after(0.5, lambda: loader._compute_minhashes(docs))
    _assert_queued_tasks_cancelled(workers=2)


def test_a_lost_lease_stops_delete_indices_at_once(mocker) -> None:
    # Old behaviour: each batch's NamespaceLeaseError was collected, the
    # later batches were still sent, and an AWSServiceError raised at the end.
    client = object.__new__(OpenSearchClient)
    batch = mocker.patch.object(
        client, "_delete_index_batch", side_effect=NamespaceLeaseError("lost")
    )
    with pytest.raises(NamespaceLeaseError):
        client.delete_indices([f"idx-{i}" for i in range(50)])
    assert batch.call_count == 1


@pytest.mark.parametrize("phase", ["retain", "collect"])
def test_a_lost_lease_is_not_swallowed_by_the_record_cleanup(
    mocker, phase: str
) -> None:
    # Old behaviour: the best-effort cleanup of extraction records logged the
    # lost lease as a warning and the run went on to its next write.
    config = Config()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(
        config=config, vector_indexer=vector, graph_indexer=FakeGraphStore()
    )
    lost = NamespaceLeaseError("lost")
    if phase == "retain":
        mocker.patch.object(vector, "retain_extraction_records", side_effect=lost)
        call = lambda: manager._retain_extraction_records([], ["default"])  # noqa: E731
    else:
        manager._removed_record_keys["default"] = {"k"}
        mocker.patch.object(vector, "find_documents", side_effect=lost)
        call = manager.collect_extraction_records
    with pytest.raises(NamespaceLeaseError):
        call()


def test_the_lease_warning_uses_the_stores_request_timeouts(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from unified_kg_rag.shared import write_fence

    monkeypatch.setattr(write_fence, "INGESTION_REQUEST_TIMEOUT_SECONDS", 300.0)
    pipe = _stopping_pipeline()
    pipe.config = Config()
    pipe._doc_status = FakeDocStatusStore()
    with caplog.at_level("WARNING"):
        pipe._namespace_lease()  # 600 s < 3 x 300 s
    assert "longest store write request (300s)" in caplog.text


def test_a_worker_checks_both_the_stop_and_the_lease_fence(
    tmp_path: Path, mocker
) -> None:
    # A run stacks its stop fence and its lease fence; a worker thread must
    # be stopped by either, the lease (innermost) first.
    from unified_kg_rag.shared import write_fence
    from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor
    from unified_kg_rag.shared.write_fence import check_write_fence

    store = FakeDocStatusStore()
    pipe = _leased_pipeline(tmp_path, mocker, store)
    seen: list[Any] = []

    def check() -> Any:
        try:
            check_write_fence()
        except (NamespaceLeaseError, PipelineInterrupted) as e:
            return type(e)
        return None

    def execute(context, start):  # type: ignore[no-untyped-def]
        lease = write_fence._FENCES.get()[-1]
        assert isinstance(lease, NamespaceLease)
        with ContextThreadPoolExecutor(max_workers=1) as pool:
            seen.append(pool.submit(check).result())
            pipe._stop_signal = signal.SIGTERM
            seen.append(pool.submit(check).result())
            pipe._stop_signal = None
            lease._lose("taken in the test")
            seen.append(pool.submit(check).result())
            pipe._stop_signal = signal.SIGTERM
            seen.append(pool.submit(check).result())
            pipe._stop_signal = None

    mocker.patch.object(pipe, "_execute_pipeline", side_effect=execute)
    pipe.run(tmp_path, pipeline_id="pid")
    assert seen == [
        None,
        PipelineInterrupted,
        NamespaceLeaseError,
        NamespaceLeaseError,
    ]


def test_a_raising_pool_block_does_not_wait_for_its_running_tasks() -> None:
    from unified_kg_rag.shared.utils.concurrency import ContextThreadPoolExecutor

    release = threading.Event()
    started = threading.Event()

    def task() -> None:
        started.set()
        release.wait(10)

    began = time.time()
    try:
        with pytest.raises(RuntimeError):
            with ContextThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(task)
                started.wait(5)
                raise RuntimeError("phase failed")
        assert time.time() - began < 2
        assert not future.done()
    finally:
        release.set()
