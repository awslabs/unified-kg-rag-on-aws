# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data-integrity regressions for incremental indexing (AWS-free).

Each test pins a failure mode where an incremental run could lose or orphan
indexed content. All data is synthetic.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    DocumentDelta,
    PipelineContext,
    PipelineStageStatus,
)
from unified_kg_rag.ports.indexer import IndexingStats
from unified_kg_rag.shared import PipelineStageError

pytestmark = pytest.mark.unit


def _context(**fields) -> PipelineContext:
    return PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory="/tmp/src",
        **fields,
    )


def _incremental_indexing_stage(mocker, store: FakeDocStatusStore, manager):
    from unified_kg_rag.application.ingestion import pipeline_stages as ps

    mocker.patch.object(ps, "IndexingManager", return_value=manager)
    cfg = Config()
    cfg.indexing.reset = False
    return ps.IndexingStage(
        config=cfg, boto_session=mocker.MagicMock(), doc_status=store
    )


# --- prune failure blocks commit --------------------------------------------


def test_prune_failure_blocks_commit_and_keeps_old_lineage(mocker) -> None:
    store = FakeDocStatusStore()
    old = DocStatusRecord(
        doc_id="doc-a",
        content_hash="old-hash",
        status=DocStatus.PROCESSED,
        entity_ids=["e-stale"],
    )
    store.put(old)

    manager = mocker.MagicMock()
    manager.initialize.return_value = True
    manager.config = Config()
    # The stale-artifact removal reports a failed delete.
    manager.delete_documents.return_value = {
        "neptune_delete_default": IndexingStats(total_items=1, failed_items=1)
    }
    stage = _incremental_indexing_stage(mocker, store, manager)
    context = _context(
        incremental_delta=DocumentDelta(changed=["doc-a"]),
        incremental_fingerprints={"doc-a": "new-hash"},
    )

    with pytest.raises(PipelineStageError, match="stale artifacts"):
        stage._execute_core(context)

    # Nothing was upserted and the registry still points at the old lineage,
    # so the next run re-detects the doc as changed and retries the prune.
    manager.index_delta.assert_not_called()
    assert store.get("doc-a") == old


def test_prune_changed_reports_success_and_failure() -> None:
    from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer

    class _Manager:
        def __init__(self, failed: int) -> None:
            self.failed = failed

        def delete_documents(self, ids_by_suffix):
            return {"x": IndexingStats(total_items=1, failed_items=self.failed)}

    store = FakeDocStatusStore()
    store.put(DocStatusRecord(doc_id="d", content_hash="h", entity_ids=["e1"]))
    delta = DocumentDelta(changed=["d"])

    assert IncrementalIndexer(store, _Manager(0)).prune_changed(delta) is True  # type: ignore[arg-type]
    assert IncrementalIndexer(store, _Manager(1)).prune_changed(delta) is False  # type: ignore[arg-type]
    # Nothing changed -> nothing to prune -> success.
    assert IncrementalIndexer(store, _Manager(1)).prune_changed(DocumentDelta()) is True  # type: ignore[arg-type]
