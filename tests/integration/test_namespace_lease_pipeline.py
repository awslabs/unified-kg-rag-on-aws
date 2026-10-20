# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A pipeline run holds its namespace's write lease (embedded stores).

While another run holds the lease, a run stops before reading or writing
anything; a finished run releases it.
"""

from __future__ import annotations

import time
from pathlib import Path

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
