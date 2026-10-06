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


# --- shared entity keeps unchanged documents' lineage ------------------------


def test_delta_run_unions_an_entity_shared_with_unchanged_docs_by_default() -> None:
    from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
    from unified_kg_rag.application.ingestion.incremental import IncrementalIndexer
    from unified_kg_rag.application.storage.indexing_manager import IndexingManager
    from unified_kg_rag.domain.models import DocumentLineage, Entity

    config = Config()
    assert config.indexing.cross_run_merge is True
    graph = FakeGraphStore()
    vector = FakeVectorStore(opensearch_config=config.indexing.opensearch)
    manager = IndexingManager(config=config, vector_indexer=vector, graph_indexer=graph)
    store = FakeDocStatusStore()
    incremental = IncrementalIndexer(store, manager)

    # Run 1: "Vendor" appears in doc-a (chunk ta) and doc-b (chunk tb).
    incremental.commit(
        lineages=[
            DocumentLineage(doc_id="doc-a", entity_ids=["e-vendor"]),
            DocumentLineage(doc_id="doc-b", entity_ids=["e-vendor"]),
        ],
        fingerprints={"doc-a": "ha", "doc-b": "hb"},
        entities=[
            Entity(
                id="e-vendor",
                name="vendor",
                description="Vendor supplies parts.",
                text_unit_ids=["ta", "tb"],
            )
        ],
    )

    # Run 2: only doc-c is new; its extraction sees "Vendor" in chunk tc.
    incremental.commit(
        lineages=[DocumentLineage(doc_id="doc-c", entity_ids=["e-vendor"])],
        fingerprints={"doc-c": "hc"},
        entities=[
            Entity(
                id="e-vendor",
                name="vendor",
                description="Vendor invoices Buyer.",
                text_unit_ids=["tc"],
            )
        ],
    )

    stored = graph.data["entities"]["e-vendor"]
    # The unchanged docs' chunk lineage (followed by mix) and description survive.
    assert set(stored.text_unit_ids or []) == {"ta", "tb", "tc"}
    assert "Vendor supplies parts." in (stored.description or "")
    assert "Vendor invoices Buyer." in (stored.description or "")


# --- document ids: path + full content ---------------------------------------


_HEADER = "Master Services Agreement between Vendor and Buyer. " * 4


def _parse(path):
    from unified_kg_rag.adapters.ingestion.parser import ParserFactory

    return ParserFactory.create_parser(path, Config()).parse_file(path)


def test_same_filename_and_header_in_different_folders_get_distinct_ids(
    tmp_path,
) -> None:
    from unified_kg_rag.domain.ingestion.delta_detector import (
        assign_document_identity,
    )

    for folder, tail in (("region-a", "Fee: 100."), ("region-b", "Fee: 200.")):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "contract.txt").write_text(_HEADER + tail)
    # Identical bytes in a third folder must still be a different document.
    (tmp_path / "region-c").mkdir()
    (tmp_path / "region-c" / "contract.txt").write_text(_HEADER + "Fee: 100.")

    docs = [
        _parse(tmp_path / folder / "contract.txt")
        for folder in ("region-a", "region-b", "region-c")
    ]
    assert len({d.document_id for d in docs}) == 3

    for doc in docs:
        assign_document_identity(doc, tmp_path)
    assert [d.metadata["relative_path"] for d in docs] == [
        "region-a/contract.txt",
        "region-b/contract.txt",
        "region-c/contract.txt",
    ]
    assert len({d.document_id for d in docs}) == 3


def test_document_id_does_not_depend_on_where_the_corpus_lives(tmp_path) -> None:
    from unified_kg_rag.domain.ingestion.delta_detector import (
        assign_document_identity,
    )

    ids = []
    for root in (tmp_path / "checkout", tmp_path / "synced"):
        (root / "sub").mkdir(parents=True)
        (root / "sub" / "a.txt").write_text("Vendor ships goods to Buyer.")
        doc = _parse(root / "sub" / "a.txt")
        assign_document_identity(doc, root)
        ids.append(doc.document_id)
    assert ids[0] == ids[1]
