# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental runs over indices with a small ``max_result_window`` (embedded).

The indexer read stored documents back a fixed 1,000 ids per search and
looked them up by field a configured window at a time, while the live index
returns at most the window it was created with: with
``index_settings.max_result_window`` below the corpus, every incremental run
failed in its indexing stage, and reverting the setting did not help (the
live index keeps its window until a reindex). The run's documents stayed
pending for good.
"""

from __future__ import annotations

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
from unified_kg_rag.domain.models import DocStatus, PipelineStageStatus

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

no_aws = e2e.no_aws


def _run(tmp_path: Path, source: Path, window: int | None, name: str):  # type: ignore[no-untyped-def]
    config = _config(tmp_path / "stores", tmp_path / "viz")
    if window is not None:
        config.indexing.opensearch.index_settings = {
            **config.indexing.opensearch.index_settings,
            "max_result_window": window,
        }
    providers = _providers(
        config, FakeOllama(e2e._respond, ANSWER, DIM), FakeCrossEncoder("r")
    )
    context = _ingest(config, providers, source, tmp_path / name)
    return context, providers.build_doc_status()


def test_incremental_runs_read_within_the_live_window(tmp_path: Path) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    for n in range(6):
        (source / f"d{n}.txt").write_text(f"Vendor ships widget lot {n} to Buyer.")

    context, _ = _run(tmp_path, source, 3, "full")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)

    # An added document: the run reads the stored units back (6 > 3).
    (source / "d6.txt").write_text("Depot stores gadgets for Vendor.")
    context, registry = _run(tmp_path, source, 3, "added")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)

    # The setting reverted (and raised past the live window) without a
    # reindex: the live indices still return 3 hits a search.
    (source / "d0.txt").write_text("Vendor ships gadget lot 0 to Carrier.")
    context, registry = _run(tmp_path, source, None, "reverted")
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    assert {r.status for r in registry.list_all()} == {DocStatus.PROCESSED}
