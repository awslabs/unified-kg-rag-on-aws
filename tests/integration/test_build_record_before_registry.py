# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""A run killed after its registry write leaves its build record (AWS-free).

The indexing stage wrote the suffix's build record after the registry
recorded the documents. A run killed in between (a first build, or a reset,
whose new indices hold no record) left the registry naming every document
indexed and no record: the next run found nothing to index, wrote no record
either, and every later query read the namespace as holding whatever its
config builds (the methodology checks fail open without a record). The
record is now written once the writes are judged recorded, before the
registry write.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as e2e
from tests.integration.test_embedded_e2e import no_aws  # noqa: F401 - a fixture
from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
from unified_kg_rag.domain.models import PipelineStageStatus

pytestmark = pytest.mark.integration


class _Killed(BaseException):
    """A process death: no ``except Exception`` handler absorbs it."""


def _record(config, providers) -> dict | None:  # noqa: ANN001
    vectors = providers.build_vector_indexer()
    try:
        return vectors.read_build_record(None)
    finally:
        vectors.close()


@pytest.mark.usefixtures("no_aws")
@pytest.mark.parametrize("reset", [False, True], ids=["first-build", "reset"])
def test_a_run_killed_after_its_registry_write_leaves_its_record(
    tmp_path: Path, mocker, reset: bool
) -> None:
    config = e2e._config(tmp_path / "stores", tmp_path / "viz")
    providers = e2e._providers(
        config, FakeOllama(e2e._respond, e2e.ANSWER, e2e.DIM), e2e.FakeCrossEncoder("r")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "contract.txt").write_text("Vendor supplies widgets to Buyer.")
    (source / "logistics.txt").write_text("Depot stores widgets for Vendor.")
    if reset:
        context = e2e._ingest(config, providers, source, tmp_path / "run-0")
        assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
        (source / "carrier.txt").write_text("Carrier ships widgets for Depot.")
        config.indexing.reset = True
    record_processed = IncrementalIndexer._record_processed

    def then_die(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        record_processed(self, *args, **kwargs)
        raise _Killed

    mocker.patch.object(IncrementalIndexer, "_record_processed", then_die)
    with pytest.raises(_Killed):
        e2e._ingest(config, providers, source, tmp_path / "run-1")
    mocker.stopall()

    killed = _record(config, providers)
    assert killed is not None
    assert killed["pipeline_id"] == "run-1"
    assert killed["mode"] == "full"
    config.indexing.reset = False
    context = e2e._ingest(config, providers, source, tmp_path / "run-2")
    assert context.status is PipelineStageStatus.COMPLETED, e2e._errors(context)
    assert context.incremental_delta is not None
    assert context.incremental_delta.to_process == []
    assert _record(config, providers) == killed
