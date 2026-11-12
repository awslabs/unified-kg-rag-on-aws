# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A killed ingestion process, and the run after it (embedded stores).

The crash tests elsewhere raise in-process; here an incremental run in a
child process is killed with SIGKILL halfway through the indexing stage
(nothing unwinds: no checkpoint upload, no lease release, no attempt
closed). Its lease, held as by another host's run (the SQLite registry
hands this host's killed run's on at once), blocks the next run until it
expires; then a rerun of the same pipeline id takes the lease over,
resumes, and leaves the stores
and the registry as a fresh build of the same corpus does.
"""

from __future__ import annotations

import os
import signal
import subprocess  # nosec B404 - runs the test's own driver module
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.rerank import FakeCrossEncoder
from tests.fixtures.killable_ingest import LEASE_SECONDS, crash_config
from tests.integration import test_embedded_e2e as e2e
from unified_kg_rag.domain.models import PipelineStageStatus
from unified_kg_rag.shared import NamespaceLeaseError

pytestmark = [pytest.mark.integration, pytest.mark.slow]

ROOT = Path(__file__).parents[2]
no_aws = e2e.no_aws


def _write(source: Path, files: dict[str, str | None]) -> None:
    source.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        if text is None:
            (source / name).unlink()
        else:
            (source / name).write_text(text, encoding="utf-8")


_FIRST = {
    "contract.txt": "Vendor supplies widgets to Buyer.",
    "logistics.txt": "Depot stores widgets for Vendor.",
}
_SECOND = {
    "freight.txt": "Carrier moves gadgets for Depot.",
    "logistics.txt": e2e.ANSWER,
    "contract.txt": None,
}


def _providers(config: Any) -> Any:
    return e2e._providers(
        config, FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM), FakeCrossEncoder("r")
    )


def _state(config: Any, providers: Any) -> tuple[Any, ...]:
    return (
        e2e._entity_names(providers),
        e2e._stored_texts(config, providers),
        e2e._registry(providers),
    )


@pytest.mark.timeout(240)
def test_a_killed_run_is_taken_over_once_its_lease_expires_and_converges(
    tmp_path: Path, no_aws  # noqa: F811 - the fixture
) -> None:
    stores, source = tmp_path / "stores", tmp_path / "corpus"
    config = crash_config(stores)
    providers = _providers(config)
    _write(source, _FIRST)
    context = e2e._ingest(config, providers, source, tmp_path / "run-1")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)

    # The incremental run, killed with the indexing half done.
    _write(source, _SECOND)
    cache, marker = tmp_path / "run-2", tmp_path / "indexing.marker"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "AWS_EC2_METADATA_DISABLED": "true"}
    child = subprocess.Popen(  # nosec B603 - the test's own module
        [
            sys.executable,
            "-m",
            "tests.fixtures.killable_ingest",
            str(stores),
            str(source),
            str(cache),
            str(marker),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        deadline = time.monotonic() + 120
        while not marker.exists():
            assert child.poll() is None, child.communicate()[0].decode()[-3000:]
            assert time.monotonic() < deadline, "the run never reached indexing"
            time.sleep(0.1)
    finally:
        child.send_signal(signal.SIGKILL)
        child.communicate()
    killed_at = time.time()
    assert child.returncode == -signal.SIGKILL

    # It died before it committed: the new and changed documents are not
    # recorded as processed.
    registry = providers.build_doc_status()
    assert e2e._registry(providers) != {
        "freight.txt": e2e.DocStatus.PROCESSED,
        "logistics.txt": e2e.DocStatus.PROCESSED,
    }
    holder = registry.lease_holder("default")
    assert holder is not None and holder[1] > time.time()
    # The SQLite registry hands a killed run's lease of this host on at once
    # (tests/unit/test_processes.py); a holder on another host it cannot
    # judge, whose lease runs to its expiry: as such a holder's.
    assert registry.try_acquire_lease(
        "default", "another-host:1:2:3:killed", holder[1] - time.time()
    )
    holder = registry.lease_holder("default")
    assert holder is not None and holder[1] > time.time()
    # Its lease stops the next run before it reads or writes anything.
    with pytest.raises(NamespaceLeaseError):
        e2e._ingest(config, _providers(config), source, cache)

    # Once it expires (no renewal since the kill), the rerun of the same
    # pipeline id takes the lease over and finishes the run.
    time.sleep(max(0.0, holder[1] - time.time()) + 0.5)
    assert time.time() - killed_at <= LEASE_SECONDS + 5
    context = e2e._ingest(config, _providers(config), source, cache)
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert registry.lease_holder("default") is None

    # As a fresh build of the same corpus.
    fresh = crash_config(tmp_path / "fresh")
    fresh_providers = _providers(fresh)
    context = e2e._ingest(fresh, fresh_providers, source, tmp_path / "fresh-run")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert _state(config, _providers(config)) == _state(fresh, fresh_providers)
    assert e2e._entity_names(fresh_providers) == {"Vendor", "Depot", "Carrier"}
