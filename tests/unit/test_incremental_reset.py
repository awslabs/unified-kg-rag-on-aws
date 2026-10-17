# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``indexing.reset`` with the doc-status registry enabled (AWS-free).

A reset clears the stores and the registry, so it must rebuild from the FULL
corpus and record every document again. Diffing against the registry before
the reset would index only the delta and lose the unchanged documents.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.domain.ingestion.delta_detector import (
    compute_content_hash,
    compute_doc_id,
    document_doc_id,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    Document,
    Entity,
    PipelineContext,
    PipelineStageStatus,
    TextUnit,
)
from unified_kg_rag.ports.indexer import IndexingStats

pytestmark = pytest.mark.unit


def _doc(path: str, text: str) -> Document:
    return Document(
        page_content=text,
        document_id=f"run-{path}",
        file_name=path.rsplit("/", 1)[-1],
        file_path=path,
    )


def _reset_config() -> Config:
    config = Config()
    config.aws.dynamodb.enabled = True
    config.indexing.reset = True
    return config


def _loading_stage(config: Config, store: FakeDocStatusStore):
    from unified_kg_rag.application.ingestion.pipeline_stages import (
        DocumentLoadingStage,
    )

    stage = DocumentLoadingStage.__new__(DocumentLoadingStage)
    stage.config = config  # type: ignore[attr-defined]
    stage._doc_status = store  # type: ignore[attr-defined]
    stage.loader = SimpleNamespace(source_directory=Path("/"))  # type: ignore[attr-defined]
    return stage


def _context(**fields) -> PipelineContext:
    return PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory="/",
        **fields,
    )


def test_reset_keeps_every_document_for_extraction() -> None:
    unchanged, edited = _doc("/a.txt", "Vendor ships."), _doc("/b.txt", "Buyer pays.")
    store = FakeDocStatusStore()
    store.put(
        DocStatusRecord(
            doc_id=document_doc_id(unchanged),
            content_hash=compute_content_hash(unchanged),
        )
    )
    stage = _loading_stage(_reset_config(), store)
    context = _context()

    kept, skipped = stage._apply_incremental_filter([unchanged, edited], context)

    assert kept == [unchanged, edited]
    assert skipped == 0
    delta = context.incremental_delta
    assert delta is not None
    assert sorted(delta.new) == sorted(
        [document_doc_id(unchanged), document_doc_id(edited)]
    )
    assert delta.changed == delta.unchanged == delta.deleted == []
    assert set(context.incremental_fingerprints) == set(delta.new)


def test_reset_records_the_full_corpus_after_rebuilding(mocker) -> None:
    from unified_kg_rag.application.ingestion import pipeline_stages as ps

    a, b = _doc("/a.txt", "Vendor ships."), _doc("/b.txt", "Buyer pays.")
    store = FakeDocStatusStore()
    store.put(
        DocStatusRecord(
            doc_id="stale-doc", content_hash="old", scope="default|/old-corpus"
        )
    )
    # Another namespace sharing the table: its stores were not cleared, so
    # its records stay. Every record of the cleared namespace goes, whatever
    # its scope.
    other = DocStatusRecord(
        doc_id=compute_doc_id("keep.txt", "default-y", "/corpus"),
        content_hash="h",
        scope="default-y|/corpus",
    )
    gone = DocStatusRecord(
        doc_id=compute_doc_id("gone.txt", "default", "/elsewhere"),
        content_hash="h",
        scope="default|/elsewhere",
    )
    for record in (other, gone):
        store.put(record)
    config = _reset_config()

    manager = mocker.MagicMock()
    manager.config = config
    manager.index_all_data.return_value = {
        "neptune_entities": IndexingStats(total_items=1, successful_items=1)
    }
    mocker.patch.object(ps, "IndexingManager", return_value=manager)
    stage = ps.IndexingStage(config=config, doc_status=store)

    loading = _loading_stage(config, store)
    context = _context()
    context.documents, _ = loading._apply_incremental_filter([a, b], context)
    context.text_units = [
        TextUnit(id="tu-a", text="Vendor ships.", document_ids=[a.document_id]),
        TextUnit(id="tu-b", text="Buyer pays.", document_ids=[b.document_id]),
    ]
    context.resolved_entities = [
        Entity(id="e-vendor", name="Vendor", text_unit_ids=["tu-a"]),
        Entity(id="e-buyer", name="Buyer", text_unit_ids=["tu-b"]),
    ]

    stage._execute_core(context)

    manager.index_all_data.assert_called_once()
    manager.index_delta.assert_not_called()
    records = {r.doc_id: r for r in store.list_all()}
    assert set(records) == {document_doc_id(a), document_doc_id(b), other.doc_id}
    assert records[other.doc_id] == other
    assert records[document_doc_id(a)].status == DocStatus.PROCESSED
    assert records[document_doc_id(a)].text_unit_ids == ["tu-a"]
    assert records[document_doc_id(b)].content_hash == compute_content_hash(b)


def test_failed_reset_rebuild_does_not_record_documents(mocker) -> None:
    from unified_kg_rag.application.ingestion import pipeline_stages as ps
    from unified_kg_rag.shared import PipelineStageError

    a = _doc("/a.txt", "Vendor ships.")
    store = FakeDocStatusStore()
    config = _reset_config()
    manager = mocker.MagicMock()
    manager.config = config
    manager.index_all_data.return_value = {
        "neptune_entities": IndexingStats(total_items=1, failed_items=1)
    }
    mocker.patch.object(ps, "IndexingManager", return_value=manager)
    stage = ps.IndexingStage(config=config, doc_status=store)
    context = _context()
    context.documents, _ = _loading_stage(config, store)._apply_incremental_filter(
        [a], context
    )
    context.text_units = [TextUnit(id="tu-a", text="x", document_ids=[a.document_id])]

    with pytest.raises(PipelineStageError):
        stage._execute_core(context)

    # Only the write-ahead record is left: PENDING, so the next run indexes
    # the document again (or removes what the rebuild wrote if it is gone).
    records = store.list_all()
    assert [r.doc_id for r in records] == [document_doc_id(a)]
    assert records[0].status is DocStatus.PENDING
    assert store.diff({document_doc_id(a): compute_content_hash(a)}).changed == [
        document_doc_id(a)
    ]


def _reset_stage(mocker, config: Config, store: FakeDocStatusStore):
    from unified_kg_rag.application.ingestion import pipeline_stages as ps

    manager = mocker.MagicMock()
    manager.config = config
    manager.index_all_data.return_value = {
        "neptune_entities": IndexingStats(total_items=1, successful_items=1)
    }
    mocker.patch.object(ps, "IndexingManager", return_value=manager)
    return ps.IndexingStage(config=config, doc_status=store)


def test_reset_keeps_the_records_of_another_namespace_of_the_suffix(
    mocker,
) -> None:
    a = _doc("/a.txt", "Vendor ships.")
    store = FakeDocStatusStore()
    # Another additional_suffix on the same item suffix: that namespace's
    # stores are intact, so its record stays.
    other_namespace = DocStatusRecord(
        doc_id=compute_doc_id("keep.txt", "default-y", "/corpus"),
        content_hash="h",
        suffix="default",
        file_path="keep.txt",
        text_unit_ids=["tu-kept"],
    )
    cleared = DocStatusRecord(
        doc_id=compute_doc_id("gone.txt", "default-x", "/corpus"),
        content_hash="h",
        suffix="default",
        file_path="gone.txt",
    )
    for record in (other_namespace, cleared):
        store.put(record)
    config = _reset_config()
    config.indexing.additional_suffix = "x"
    stage = _reset_stage(mocker, config, store)

    context = _context()
    context.documents, _ = _loading_stage(config, store)._apply_incremental_filter(
        [a], context
    )
    context.text_units = [
        TextUnit(id="tu-a", text="Vendor ships.", document_ids=[a.document_id])
    ]

    stage._execute_core(context)

    records = {r.doc_id: r for r in store.list_all()}
    assert records[other_namespace.doc_id] == other_namespace
    assert cleared.doc_id not in records
    assert document_doc_id(a) in records
    # Only the cleared namespace's partition was read.
    assert store.namespace_reads.count("default-x") >= 1
    assert "default-y" not in store.namespace_reads


def test_reset_fails_when_the_registry_cannot_be_cleared(mocker) -> None:
    # The stores are already wiped: a registry left with stale records would
    # make the next incremental run skip re-ingested documents as unchanged,
    # so the reset fails instead of logging and indexing on.
    from unified_kg_rag.shared.exceptions import DocStatusRegistryError

    class _UnreachableStore(FakeDocStatusStore):
        # The registry adapters raise their errors as DocStatusRegistryError.
        def list_namespace(self, namespace):
            raise DocStatusRegistryError("registry unreachable")

    store = _UnreachableStore()
    config = _reset_config()
    stage = _reset_stage(mocker, config, store)
    context = _context()
    context.text_units = [TextUnit(id="tu-a", text="x", document_ids=["doc-a"])]

    with pytest.raises(DocStatusRegistryError, match="registry unreachable"):
        stage._execute_core(context)
    stage.indexing_manager.index_all_data.assert_not_called()


def test_reset_surfaces_a_registry_configuration_error_as_itself(mocker) -> None:
    # A misconfigured registry is not a reachability problem: the reset must
    # not rewrap it as a DocStatusRegistryError telling the user to re-run
    # once the registry is reachable.
    from unified_kg_rag.shared.exceptions import (
        ConfigurationError,
        DocStatusRegistryError,
    )

    class _MisconfiguredStore(FakeDocStatusStore):
        def list_namespace(self, namespace):
            raise ConfigurationError("aws.dynamodb.table_name must be set")

    stage = _reset_stage(mocker, _reset_config(), _MisconfiguredStore())
    context = _context()
    context.text_units = [TextUnit(id="tu-a", text="x", document_ids=["doc-a"])]

    with pytest.raises(ConfigurationError, match="table_name must be set") as info:
        stage._execute_core(context)
    assert not isinstance(info.value, DocStatusRegistryError)
    stage.indexing_manager.index_all_data.assert_not_called()
