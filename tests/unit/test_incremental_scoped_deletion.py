# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Incremental deletion is scoped to the run's tenant and corpus (AWS-free).

``deleted`` used to be every registry document absent from the run's input,
across the whole table, with the doc id a hash of the local path only. A
per-file parse failure, a run over a subfolder, or another tenant's run then
removed indexed content. These tests drive the real parsing and loading stages
over synthetic files and the in-memory registry (plus the moto-backed DynamoDB
adapter for the diff contract).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import boto3
import pytest
from moto import mock_aws

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from unified_kg_rag.adapters.aws import DynamoDBDocStatusStore
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
)
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.ingestion.pipeline_stages import (
    DocumentLoadingStage,
    DocumentParsingStage,
)
from unified_kg_rag.domain.ingestion.delta_detector import (
    assign_document_identity,
    assign_registry_source,
    compute_content_hash,
    compute_doc_id,
    detect_delta,
    document_doc_id,
    legacy_doc_id,
    other_local_scopes,
    validate_retired_scopes,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    Document,
    DocumentDelta,
    PipelineContext,
    PipelineStageStatus,
)
from unified_kg_rag.shared import ConfigurationError
from unified_kg_rag.shared.utils.document_identity import is_local_path_scope

pytestmark = pytest.mark.unit


def _config(index_value: str | None = None) -> Config:
    config = Config()
    config.aws.dynamodb.enabled = True
    config.processing.document_parsing.index_value = index_value
    return config


def _write(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _run(
    root: Path, store: FakeDocStatusStore, config: Config, mocker
) -> PipelineContext:
    """Parse ``root`` and run the loading stage's incremental filter, then
    record every processed document the way a successful commit does."""
    mocker.patch("boto3.Session")
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=root,
    )
    DocumentParsingStage(config, source_directory=root)._execute_core(context)
    loading = DocumentLoadingStage(config, source_directory=root, doc_status=store)
    kept, _ = loading._apply_incremental_filter(list(context.documents), context)
    context.documents = kept

    lineages = build_document_lineage(
        documents=kept,
        text_units=[],
        entities=[],
        relationships=[],
        communities=[],
        claims=[],
    )
    indexer = IncrementalIndexer(
        store, mocker.MagicMock(), scope=context.incremental_scope
    )
    indexer._record_processed(lineages, context.incremental_fingerprints)
    return context


def _doc_id(context: PipelineContext, relative: str, namespace: str = "default"):
    """The registry key of ``relative`` in ``context``'s source scope."""
    source_scope = context.incremental_scope.split("|", 1)[1]
    return compute_doc_id(relative, namespace, source_scope)


def test_another_tenants_run_does_not_delete_this_tenants_documents(
    tmp_path, mocker
) -> None:
    # Both tenants' corpora are staged in the same local directory (as a
    # container does) and both contain contract.txt.
    staging = tmp_path / "staging"
    store = FakeDocStatusStore()

    _write(staging, {"contract.txt": "Vendor A supplies parts to Buyer."})
    first = _run(staging, store, _config("tenant-a"), mocker)
    assert len(first.incremental_delta.new) == 1

    for path in staging.iterdir():
        path.unlink()
    _write(staging, {"contract.txt": "Vendor B leases trucks to Buyer."})
    second = _run(staging, store, _config("tenant-b"), mocker)

    # Same relative path, different tenant -> a different document, and the
    # other tenant's record is not a deletion candidate.
    assert len(second.incremental_delta.new) == 1
    assert second.incremental_delta.deleted == []
    assert second.incremental_delta.changed == []
    assert len(store.list_all()) == 2


def test_a_file_that_fails_to_parse_is_not_deleted(tmp_path, mocker) -> None:
    root = _write(
        tmp_path / "corpus",
        {"a.txt": "Vendor ships goods.", "b.txt": "Buyer pays invoices."},
    )
    store = FakeDocStatusStore()
    _run(root, store, _config(), mocker)
    assert len(store.list_all()) == 2

    # b.txt now fails to parse (empty text is rejected by the parser).
    (root / "b.txt").write_text("", encoding="utf-8")
    context = _run(root, store, _config(), mocker)

    delta = context.incremental_delta
    assert delta.deleted == []
    assert delta.failed == [_doc_id(context, "b.txt")]
    assert delta.unchanged == [_doc_id(context, "a.txt")]
    assert store.get(_doc_id(context, "b.txt")) is not None


def test_a_subfolder_run_does_not_delete_the_parent_corpus(tmp_path, mocker) -> None:
    root = _write(
        tmp_path / "corpus",
        {"top.txt": "Vendor ships goods.", "sub/inner.txt": "Buyer pays."},
    )
    store = FakeDocStatusStore()
    _run(root, store, _config(), mocker)

    context = _run(root / "sub", store, _config(), mocker)
    assert context.incremental_delta.deleted == []


def test_a_removed_file_is_still_deleted_within_the_scope(tmp_path, mocker) -> None:
    root = _write(
        tmp_path / "corpus", {"a.txt": "Vendor ships goods.", "b.txt": "Buyer pays."}
    )
    store = FakeDocStatusStore()
    _run(root, store, _config(), mocker)

    (root / "b.txt").unlink()
    context = _run(root, store, _config(), mocker)
    assert context.incremental_delta.deleted == [_doc_id(context, "b.txt")]


def test_source_scope_setting_identifies_the_corpus_not_the_staging_dir(
    tmp_path, mocker
) -> None:
    # Two runs from different staging dirs of the same named source share a
    # scope; a different source does not.
    store = FakeDocStatusStore()
    config = _config()
    config.processing.document_parsing.source_scope = "s3://bucket/corpus-1/"
    first = _run(_write(tmp_path / "a", {"x.txt": "Vendor."}), store, config, mocker)
    second = _run(_write(tmp_path / "b", {"y.txt": "Buyer."}), store, config, mocker)
    assert first.incremental_scope == second.incremental_scope
    assert second.incremental_delta.deleted == [_doc_id(second, "x.txt")]

    other = _config()
    other.processing.document_parsing.source_scope = "s3://bucket/corpus-2/"
    third = _run(_write(tmp_path / "c", {"z.txt": "Carrier."}), store, other, mocker)
    assert third.incremental_delta.deleted == []


def test_committed_records_carry_scope_and_relative_path(tmp_path, mocker) -> None:
    root = _write(tmp_path / "corpus", {"docs/a.txt": "Vendor ships goods."})
    store = FakeDocStatusStore()
    context = _run(root, store, _config("tenant-a"), mocker)

    (record,) = store.list_all()
    assert record.scope == context.incremental_scope
    assert record.scope.startswith("tenant-a|")
    assert record.file_path == "docs/a.txt"
    assert record.doc_id == _doc_id(context, "docs/a.txt", "tenant-a")


# --- registry diff contract (fake and DynamoDB stay identical) ---------------


@pytest.fixture(params=["fake", "dynamodb"])
def registry(request):
    if request.param == "fake":
        yield FakeDocStatusStore()
        return
    with mock_aws():
        config = Config()
        config.aws.dynamodb.table_name = "test-doc-status"
        store = DynamoDBDocStatusStore(
            config, boto_session=boto3.Session(region_name="us-east-1")
        )
        _ = store.client
        yield store


def test_scoped_diff_only_deletes_records_of_the_same_scope(registry) -> None:
    for doc_id, scope in (("a", "s1"), ("b", "s1"), ("c", "s2"), ("legacy", None)):
        registry.put(
            DocStatusRecord(
                doc_id=doc_id,
                content_hash="h",
                status=DocStatus.PROCESSED,
                scope=scope,
            )
        )

    scoped = registry.diff({"a": "h"}, scope="s1")
    assert scoped.unchanged == ["a"]
    assert scoped.deleted == ["b"]

    # Without a scope the whole registry is the deletion candidate set.
    assert sorted(registry.diff({"a": "h"}).deleted) == ["b", "c", "legacy"]
    # The scope round-trips through the store.
    assert registry.get("c").scope == "s2"
    assert registry.get("legacy").scope is None


# --- adopting records keyed without the source scope ------------------------

_SCOPE = "default|s3://bucket/corpus-1/"
_SOURCE = "s3://bucket/corpus-1/"


def _scoped_document(relative: str, text: str) -> Document:
    document = Document(
        page_content=text,
        document_id=relative,
        file_name=relative,
        file_path=f"/staging/{relative}",
        file_type="txt",
        total_pages=1,
    )
    assign_document_identity(document, "/staging")
    assign_registry_source(document, _SOURCE)
    return document


def _legacy_record(document: Document, scope: str | None) -> DocStatusRecord:
    return DocStatusRecord(
        doc_id=legacy_doc_id(document),
        content_hash=compute_content_hash(document),
        status=DocStatus.PROCESSED,
        scope=scope,
        file_path="a.txt",
        entity_ids=["e1"],
        text_unit_ids=["t1"],
    )


def test_source_scope_is_part_of_the_doc_id() -> None:
    assert compute_doc_id("a.txt", "default", "s1") != compute_doc_id(
        "a.txt", "default", "s2"
    )
    assert compute_doc_id("a.txt", "default", "s1") != compute_doc_id("a.txt")
    document = _scoped_document("a.txt", "Vendor ships goods.")
    assert document_doc_id(document) == compute_doc_id("a.txt", "default", _SOURCE)
    assert legacy_doc_id(document) == compute_doc_id("a.txt", "default")


def test_a_legacy_record_of_the_scope_is_adopted(registry) -> None:
    document = _scoped_document("a.txt", "Vendor ships goods.")
    registry.put(_legacy_record(document, _SCOPE))
    doc_id = document_doc_id(document)

    delta, _ = detect_delta(
        [document],
        registry,
        scope=_SCOPE,
        legacy_doc_ids={doc_id: legacy_doc_id(document)},
    )

    assert delta.unchanged == [doc_id]
    assert delta.new == delta.deleted == []
    assert registry.get(legacy_doc_id(document)) is None
    adopted = registry.get(doc_id)
    assert adopted.entity_ids == ["e1"] and adopted.text_unit_ids == ["t1"]
    assert adopted.scope == _SCOPE


def test_a_failed_files_legacy_record_is_adopted_not_deleted(registry) -> None:
    document = _scoped_document("a.txt", "Vendor ships goods.")
    registry.put(_legacy_record(document, _SCOPE))
    doc_id = document_doc_id(document)

    # a.txt failed to parse this run, so no document reaches the diff.
    delta, _ = detect_delta(
        [],
        registry,
        scope=_SCOPE,
        failed_doc_ids=[doc_id],
        legacy_doc_ids={doc_id: legacy_doc_id(document)},
    )

    assert delta.deleted == []
    assert delta.failed == [doc_id]
    assert registry.get(doc_id).entity_ids == ["e1"]
    assert registry.get(legacy_doc_id(document)) is None


def test_an_interrupted_adoption_only_drops_the_stale_legacy_key(registry) -> None:
    document = _scoped_document("a.txt", "Vendor ships goods.")
    legacy = _legacy_record(document, _SCOPE)
    doc_id = document_doc_id(document)
    registry.put(legacy)
    registry.put(legacy.model_copy(update={"doc_id": doc_id}))

    delta, _ = detect_delta(
        [document],
        registry,
        scope=_SCOPE,
        legacy_doc_ids={doc_id: legacy_doc_id(document)},
    )

    assert delta.unchanged == [doc_id]
    assert delta.deleted == []
    assert [r.doc_id for r in registry.list_all()] == [doc_id]


@pytest.mark.parametrize("edited", [False, True])
def test_an_interrupted_adoption_of_a_scopeless_record_drops_it(
    registry, edited
) -> None:
    # Written before scopes existed (scope None), adopted under the current
    # key, then interrupted before the legacy key was deleted. A scope-None
    # record is never a deletion candidate, so only the adoption cleanup can
    # remove it; left behind, it keeps the old version's artifacts referenced.
    document = _scoped_document("a.txt", "Vendor ships goods.")
    legacy = _legacy_record(document, None)
    doc_id = document_doc_id(document)
    registry.put(legacy)
    registry.put(legacy.model_copy(update={"doc_id": doc_id, "scope": _SCOPE}))
    if edited:
        document = _scoped_document("a.txt", "Vendor ships other goods.")

    delta, _ = detect_delta(
        [document],
        registry,
        scope=_SCOPE,
        legacy_doc_ids={doc_id: legacy_doc_id(document)},
    )

    assert (delta.changed if edited else delta.unchanged) == [doc_id]
    assert delta.new == delta.deleted == []
    assert [r.doc_id for r in registry.list_all()] == [doc_id]
    assert registry.get(doc_id).entity_ids == ["e1"]


def test_an_interrupted_adoption_of_a_failed_file_drops_the_legacy_key(
    registry,
) -> None:
    document = _scoped_document("a.txt", "Vendor ships goods.")
    legacy = _legacy_record(document, None)
    doc_id = document_doc_id(document)
    registry.put(legacy)
    current = legacy.model_copy(
        update={"doc_id": doc_id, "scope": _SCOPE, "entity_ids": ["e2"]}
    )
    registry.put(current)

    delta, _ = detect_delta(
        [],
        registry,
        scope=_SCOPE,
        failed_doc_ids=[doc_id],
        legacy_doc_ids={doc_id: legacy_doc_id(document)},
    )

    # The current record is kept as is (not overwritten by the legacy copy).
    assert delta.failed == [doc_id] and delta.deleted == []
    assert registry.list_all() == [current]


def test_a_legacy_record_of_another_scope_is_not_adopted(registry) -> None:
    document = _scoped_document("a.txt", "Vendor ships goods.")
    other = _legacy_record(document, "default|s3://bucket/corpus-2/")
    registry.put(other)
    doc_id = document_doc_id(document)

    delta, _ = detect_delta(
        [document],
        registry,
        scope=_SCOPE,
        legacy_doc_ids={doc_id: legacy_doc_id(document)},
    )

    assert delta.new == [doc_id]
    assert delta.deleted == []
    assert registry.get(legacy_doc_id(document)) == other
    assert registry.get(doc_id) is None


def test_without_a_scope_nothing_is_adopted(registry) -> None:
    document = _scoped_document("a.txt", "Vendor ships goods.")
    registry.put(_legacy_record(document, None))
    doc_id = document_doc_id(document)

    delta, _ = detect_delta(
        [document], registry, legacy_doc_ids={doc_id: legacy_doc_id(document)}
    )

    assert delta.new == [doc_id]
    assert registry.get(legacy_doc_id(document)) is not None


# --- legacy adoption adjusts the delta instead of diffing again --------------


class _StaleDiffStore(FakeDocStatusStore):
    """Registry whose scan lags its writes, like an eventually consistent
    DynamoDB ``Scan``: every diff and listing answers from the records as
    they were at the first diff. Counts the diffs."""

    def __init__(self) -> None:
        super().__init__()
        self.diffs = 0
        self._snapshot: dict[str, DocStatusRecord] | None = None

    def diff(self, incoming: dict[str, str], scope: str | None = None):
        self.diffs += 1
        if self._snapshot is None:
            self._snapshot = dict(self._records)
        current, self._records = self._records, self._snapshot
        try:
            return super().diff(incoming, scope=scope)
        finally:
            self._records = current

    def list_all(self) -> list[DocStatusRecord]:
        return list((self._snapshot or self._records).values())


def _no_removal_manager() -> MagicMock:
    """An indexing manager that fails the test if any artifact is touched."""
    manager = MagicMock()
    manager.delete_documents.side_effect = AssertionError("nothing to delete")
    manager.remove_text_units_from_shared.side_effect = AssertionError(
        "nothing to strip"
    )
    return manager


@pytest.mark.parametrize(
    ("text", "status", "expected"),
    [
        ("Vendor ships goods.", DocStatus.PROCESSED, "unchanged"),
        ("Vendor ships other goods.", DocStatus.PROCESSED, "changed"),
        ("Vendor ships goods.", DocStatus.FAILED, "changed"),
    ],
)
def test_legacy_adoption_diffs_once_and_classifies_in_memory(
    text, status, expected
) -> None:
    store = _StaleDiffStore()
    stored = _scoped_document("a.txt", "Vendor ships goods.")
    legacy = _legacy_record(stored, _SCOPE).model_copy(update={"status": status})
    store.put(legacy)
    document = _scoped_document("a.txt", text)
    doc_id = document_doc_id(document)

    delta, _ = detect_delta(
        [document],
        store,
        scope=_SCOPE,
        legacy_doc_ids={doc_id: legacy_doc_id(document)},
    )

    assert store.diffs == 1
    assert getattr(delta, expected) == [doc_id]
    assert delta.new == delta.deleted == []
    assert store.get(legacy.doc_id) is None
    assert store.get(doc_id).text_unit_ids == ["t1"]


def test_a_stale_scan_after_adoption_deletes_nothing() -> None:
    # Diffing again after the re-key, a scan that still returned the legacy
    # key (scope = this run's) and not the new one read it as deleted and
    # pruned the artifacts the adopted document now owns.
    store = _StaleDiffStore()
    document = _scoped_document("a.txt", "Vendor ships goods.")
    store.put(_legacy_record(document, _SCOPE))
    doc_id = document_doc_id(document)

    delta, _ = detect_delta(
        [document],
        store,
        scope=_SCOPE,
        legacy_doc_ids={doc_id: legacy_doc_id(document)},
    )

    assert delta.deleted == [] and delta.unchanged == [doc_id]
    indexer = IncrementalIndexer(store, _no_removal_manager(), scope=_SCOPE)
    assert indexer.remove_changed_and_deleted(delta)
    assert store.get(doc_id).entity_ids == ["e1"]


def test_an_exhausted_adopted_document_is_not_read_back() -> None:
    # The retry limit uses the adopted record as written: a store that does
    # not return it yet would otherwise make it retry once more.
    store = _StaleDiffStore()
    document = _scoped_document("a.txt", "Vendor ships goods.")
    store.put(
        _legacy_record(document, _SCOPE).model_copy(
            update={"status": DocStatus.FAILED, "failure_count": 3}
        )
    )
    doc_id = document_doc_id(document)
    store.get = MagicMock(side_effect=AssertionError("read back"))  # type: ignore[method-assign]

    delta, _ = detect_delta(
        [document],
        store,
        scope=_SCOPE,
        max_failures=3,
        legacy_doc_ids={doc_id: legacy_doc_id(document)},
    )

    assert delta.unchanged == [doc_id] and delta.changed == []


# --- retiring source scopes ---------------------------------------------------

_OLD_SCOPE = "default|/corpora/old"


def _record(doc_id: str, scope: str | None) -> DocStatusRecord:
    return DocStatusRecord(
        doc_id=doc_id, content_hash="h", status=DocStatus.PROCESSED, scope=scope
    )


def test_diff_reports_the_stored_scopes(registry) -> None:
    for doc_id, scope in (("a", _SCOPE), ("b", _OLD_SCOPE), ("c", None)):
        registry.put(_record(doc_id, scope))

    delta = registry.diff({"a": "h"}, scope=_SCOPE)

    assert delta.stored_scopes == sorted([_SCOPE, _OLD_SCOPE])
    assert delta.deleted == []


def test_a_retired_scope_is_deleted_and_nothing_else(registry) -> None:
    document = _scoped_document("a.txt", "Vendor ships goods.")
    for doc_id, scope in (
        ("old-1", _OLD_SCOPE),
        ("old-2", _OLD_SCOPE),
        ("other", "default|/corpora/other"),
        ("tenant-b", "tenant-b|/corpora/old"),
        ("legacy", None),
    ):
        registry.put(_record(doc_id, scope))

    delta, _ = detect_delta(
        [document], registry, scope=_SCOPE, retired_scopes=[_OLD_SCOPE]
    )

    assert delta.new == [document_doc_id(document)]
    assert sorted(delta.deleted) == ["old-1", "old-2"]
    # Detection writes nothing: the records go when their removal succeeds.
    assert len(registry.list_all()) == 5


@pytest.mark.parametrize(
    ("scope", "retired"),
    [(_SCOPE, [_SCOPE]), (_SCOPE, [_OLD_SCOPE, _SCOPE]), (None, [_OLD_SCOPE])],
)
def test_retiring_the_runs_own_scope_fails_before_any_read(scope, retired) -> None:
    store = MagicMock()
    store.diff.side_effect = AssertionError("the registry must not be read")

    with pytest.raises(ConfigurationError, match="own registry scope"):
        detect_delta([], store, scope=scope, retired_scopes=retired)


@pytest.mark.parametrize(
    "own", ["/corpora/new", "/corpora/new/", "/corpora//new", " /corpora/new "]
)
def test_validate_retired_scopes_rejects_the_own_source_scope(own) -> None:
    with pytest.raises(ConfigurationError, match="own source scope"):
        validate_retired_scopes(["/corpora/old", own], "/corpora/new", "default")


def test_validate_retired_scopes_normalizes_paths_and_keeps_uris() -> None:
    assert validate_retired_scopes(
        ["/corpora/old/", "/corpora/old", "s3://bucket/old/", "corpus-a"],
        "s3://bucket/new/",
        "tenant-a",
    ) == [
        "tenant-a|/corpora/old",
        "tenant-a|s3://bucket/old/",
        "tenant-a|corpus-a",
    ]
    with pytest.raises(ConfigurationError, match="own source scope"):
        validate_retired_scopes(["s3://bucket/new/"], "s3://bucket/new/", "t")
    with pytest.raises(ConfigurationError, match="empty"):
        validate_retired_scopes(["  "], "/corpora/new", "default")


def test_other_local_scopes_names_only_local_directories_of_the_namespace() -> None:
    delta = DocumentDelta(
        stored_scopes=[
            "default|/corpora/new",
            "default|/corpora/old",
            "default|/corpora/retired",
            "default|s3://bucket/corpus/",
            "default|corpus-a",
            "tenant-b|/corpora/other",
        ]
    )

    assert other_local_scopes(
        delta, "default|/corpora/new", "default", ["default|/corpora/retired"]
    ) == ["/corpora/old"]


def test_is_local_path_scope_reads_only_the_string() -> None:
    assert is_local_path_scope("/corpora/gone-on-this-host")
    assert is_local_path_scope("C:/corpora/a")
    assert not is_local_path_scope("s3://bucket/prefix/")
    assert not is_local_path_scope("corpus-a")


def test_a_pipeline_retiring_without_the_registry_fails_fast(tmp_path) -> None:
    pipeline = object.__new__(DataIngestionPipeline)
    pipeline.config = _config()
    pipeline.config.aws.dynamodb.enabled = False
    pipeline.config.indexing.retire_source_scopes = ["/corpora/old"]
    pipeline._doc_status = None
    with pytest.raises(ConfigurationError, match="needs incremental indexing"):
        pipeline._validate_retired_scopes(tmp_path)

    pipeline._doc_status = FakeDocStatusStore()
    pipeline._validate_retired_scopes(tmp_path)
    pipeline.config.indexing.retire_source_scopes = [f"{tmp_path.as_posix()}/"]
    with pytest.raises(ConfigurationError, match="own source scope"):
        pipeline._validate_retired_scopes(tmp_path)
