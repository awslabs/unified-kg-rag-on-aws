# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A pipeline run holds its namespace's write lease (embedded stores).

While another run holds the lease, a run stops before reading or writing
anything; a finished run releases it.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_embedded_e2e import (
    ANSWER,
    DIM,
    FakeCrossEncoder,
    _config,
    _errors,
    _ingest,
    _providers,
)
from unified_kg_rag.domain.models import PipelineStageStatus
from unified_kg_rag.shared import NamespaceLeaseError

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

no_aws = e2e.no_aws


def test_a_leased_namespace_stops_a_run_before_it_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import unified_kg_rag.application.ingestion.namespace_lease as lease_mod

    # A takeover of an expired lease waits for the late writes of its holder
    # (up to its expiry plus the longest request): shorten that here.
    monkeypatch.setattr(lease_mod, "_TAKEOVER_MARGIN_SECONDS", 0.1)
    monkeypatch.setattr(lease_mod, "longest_write_request_seconds", lambda: 0.3)
    config = _config(tmp_path / "stores", tmp_path / "viz")
    providers = _providers(
        config, FakeOllama(e2e._respond, ANSWER, DIM), FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "a.txt").write_text("Vendor supplies Buyer.")
    registry = providers.build_doc_status()
    registry.try_acquire_lease("default", "other-host:1:run", 60)

    with pytest.raises(NamespaceLeaseError, match="other-host:1:run"):
        _ingest(config, providers, source, tmp_path / "run-1")
    assert registry.list_all() == []

    # The other run ends: this one indexes and leaves no lease behind.
    registry.release_lease("default", "other-host:1:run")
    context = _ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    assert len(registry.list_all()) == 1
    assert registry.lease_holder("default") is None
    # An expired lease does not block either; the run's first write waits
    # until the dead run's writes have landed.
    registry.try_acquire_lease("default", "dead-run", 0.1)
    time.sleep(0.2)
    started = time.time()
    context = _ingest(config, providers, source, tmp_path / "run-3")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    assert time.time() - started >= 0.2


def test_a_stopped_local_run_lets_the_next_one_write_at_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The embedded stores are written in the run's process: a stop leaves no
    # write in flight, so it releases the lease (after a Ctrl-C the next
    # local run stalled ~3 minutes in document_loading).
    import unified_kg_rag.application.ingestion.namespace_lease as lease_mod
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.shared import PipelineInterrupted

    monkeypatch.setattr(lease_mod, "longest_write_request_seconds", lambda: 180.0)
    config = _config(tmp_path / "stores", tmp_path / "viz")
    providers = _providers(
        config, FakeOllama(e2e._respond, ANSWER, DIM), FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "a.txt").write_text("Vendor supplies Buyer.")
    registry = providers.build_doc_status()

    def stopped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise PipelineInterrupted(15)

    with monkeypatch.context() as patch:
        patch.setattr(DataIngestionPipeline, "_execute_pipeline", stopped)
        with pytest.raises(PipelineInterrupted):
            _ingest(config, providers, source, tmp_path / "run-1")
    assert registry.lease_holder("default") is None
    started = time.time()
    context = _ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    assert time.time() - started < 60


class _RemoteStore:
    """A stand-in for a store written over the network (no ``in_process``)."""

    closed = False

    def close(self) -> None:
        self.closed = True


def _pipeline(tmp_path: Path, **injected: object):  # type: ignore[no-untyped-def]
    """A pipeline on the embedded config, with ``injected`` stores
    (``doc_status``, ``graph_indexer_factory``, ``vector_indexer_factory``)."""
    from unified_kg_rag.adapters.local.embeddings import OllamaEmbeddingFactory
    from unified_kg_rag.adapters.local.models import OllamaLanguageModelFactory
    from unified_kg_rag.adapters.local.rerank import CrossEncoderRerankFactory
    from unified_kg_rag.application.bootstrap import build_providers
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.domain.models import PipelineConfig

    config = _config(tmp_path / "stores", tmp_path / "viz")
    config.search.store_timeout_seconds = 7.0
    transport = httpx.MockTransport(FakeOllama(e2e._respond, ANSWER, DIM))
    providers = build_providers(
        config,
        llm_factory=OllamaLanguageModelFactory(config, transport=transport),
        embedding_factory=OllamaEmbeddingFactory(config, transport=transport),
        rerank_factory=CrossEncoderRerankFactory(
            config, loader=lambda *_, **__: FakeCrossEncoder("r")
        ),
        **injected,  # type: ignore[arg-type]
    )
    return DataIngestionPipeline(
        config,
        PipelineConfig(local_directory=tmp_path / "cache"),
        source_directory=tmp_path,
        providers=providers,
    )


def test_a_run_on_embedded_stores_holds_an_in_process_lease(tmp_path: Path) -> None:
    lease = _pipeline(tmp_path)._namespace_lease()
    assert lease.in_process_stores is True
    # A takeover waits one embedded write (the stores' busy timeout).
    assert lease.in_process_write_seconds == 7.0


@pytest.mark.parametrize("remote", ["registry", "graph", "vector"])
def test_one_injected_remote_store_makes_the_lease_remote(
    tmp_path: Path, remote: str
) -> None:
    # Old behaviour: the rule read the configured kinds (all sqlite here),
    # so an injected remote store was taken for an embedded one, and a stop
    # released the lease with that store's requests still in flight.
    from tests.fixtures.fakes.doc_status import FakeDocStatusStore

    store = _RemoteStore()
    injected: dict[str, object] = {
        "registry": {"doc_status": FakeDocStatusStore()},
        "graph": {"graph_indexer_factory": lambda: store},
        "vector": {"vector_indexer_factory": lambda: store},
    }[remote]
    lease = _pipeline(tmp_path, **injected)._namespace_lease()
    assert lease.in_process_stores is False
    assert store.closed is (remote != "registry")


def test_the_probe_leaves_a_store_injected_into_the_pipeline_open(
    tmp_path: Path,
) -> None:
    # The pipeline's vector_indexer= is the instance every stage writes
    # with: the in-process probe closed it before the run used it.
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.domain.models import PipelineConfig

    store = _RemoteStore()
    pipeline = DataIngestionPipeline(
        _config(tmp_path / "stores", tmp_path / "viz"),
        PipelineConfig(local_directory=tmp_path / "cache"),
        source_directory=tmp_path,
        vector_indexer=store,  # type: ignore[arg-type]
    )
    assert pipeline._namespace_lease().in_process_stores is False
    assert store.closed is False


def test_the_stages_leave_the_stores_injected_into_the_pipeline_open(
    tmp_path: Path,
) -> None:
    # Each stage closes the indexer it builds; one built from the pipeline's
    # vector_indexer= / graph_indexer= is the caller's instance, which the
    # community stage's whole-graph read closed before the indexing stage
    # wrote with it.
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.domain.models import PipelineConfig

    vector, graph = _RemoteStore(), _RemoteStore()
    pipeline = DataIngestionPipeline(
        _config(tmp_path / "stores", tmp_path / "viz"),
        PipelineConfig(local_directory=tmp_path / "cache"),
        source_directory=tmp_path,
        vector_indexer=vector,  # type: ignore[arg-type]
        graph_indexer=graph,  # type: ignore[arg-type]
    )
    built = pipeline.providers.build_graph_indexer()
    built.close()
    pipeline.providers.build_vector_indexer().close()
    pipeline.close()

    assert (vector.closed, graph.closed) == (False, False)
    # Everything but close is the injected instance's.
    graph.closed = True
    assert built.closed is True


def test_a_stopped_run_with_a_remote_store_leaves_its_lease_expired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Mixed stores: the remote one may still have a write in flight, so a
    # stop gives the lease up expired (owner kept) and the next run waits.
    import unified_kg_rag.application.ingestion.namespace_lease as lease_mod
    from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
    from unified_kg_rag.shared import PipelineInterrupted

    monkeypatch.setattr(lease_mod, "longest_write_request_seconds", lambda: 180.0)
    pipeline = _pipeline(tmp_path, graph_indexer_factory=_RemoteStore)
    (tmp_path / "a.txt").write_text("Vendor supplies Buyer.")

    def stopped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise PipelineInterrupted(15)

    monkeypatch.setattr(DataIngestionPipeline, "_execute_pipeline", stopped)
    with pytest.raises(PipelineInterrupted):
        pipeline.run(tmp_path, pipeline_id="run-1")
    assert pipeline._doc_status is not None
    holder = pipeline._doc_status.lease_holder("default")
    assert holder is not None and holder[1] <= time.time()
    assert pipeline.next_run_writes_from > time.time() + 60
