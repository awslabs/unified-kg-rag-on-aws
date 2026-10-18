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
from concurrent.futures import ThreadPoolExecutor
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
    # The batch's own check and its first attempt pass; the lease is lost
    # before the retry.
    token = _fenced(_Fence(allowed=2))
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
    time.sleep(0.05)
    return []


def test_a_stop_cancels_the_queued_tasks_of_the_resolver_process_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Old behaviour: the ProcessPoolExecutor block ran every queued name
    # before the stop got through (200 x 0.05 s on 2 workers = 5 s).
    from unified_kg_rag.domain.ingestion import graph_resolver

    monkeypatch.setattr(graph_resolver, "find_all_matches_for_entity_task", _slow_match)
    # A thread pool's block waits for the queued tasks the same way, without
    # forking the test process.
    monkeypatch.setattr(graph_resolver, "ProcessPoolExecutor", ThreadPoolExecutor)
    resolver = graph_resolver.EntityResolver(Config(), 2, True)

    def stop(signum: int, _: Any) -> None:
        raise PipelineInterrupted(signum)

    previous = signal.signal(signal.SIGALRM, stop)
    started = time.time()
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.5)
        with pytest.raises(PipelineInterrupted):
            for _ in resolver._match_in_pool(
                MagicMock(), [f"n{i}" for i in range(200)]
            ):
                pass
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert time.time() - started < 2.5


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
