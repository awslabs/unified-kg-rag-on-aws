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
from unified_kg_rag.adapters.doc_status.dynamodb import DynamoDBDocStatusStore
from unified_kg_rag.adapters.providers import Providers
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    build_document_lineage,
    retired_scope_readings,
)
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.ingestion.pipeline_stages import (
    DocumentLoadingStage,
    DocumentParsingStage,
)
from unified_kg_rag.domain.ingestion.delta_detector import (
    assign_document_identity,
    assign_registry_source,
    compute_doc_id,
    detect_delta,
    document_doc_id,
    other_local_scopes,
    retire_source_scopes,
    retired_scope_groups,
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
    config.providers.doc_status.enabled = True
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

    lineages = build_document_lineage(documents=kept, text_units=[])
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


@pytest.mark.parametrize("copy_name", ["a_copy.txt", "z_copy.txt"])
def test_dedup_keeps_the_indexed_original_over_a_new_copy(
    tmp_path, mocker, copy_name
) -> None:
    # The diff reads a document deduplication dropped as deleted: keeping a new
    # near-copy that sorts first would delete the indexed original.
    root = tmp_path / "corpus"
    store = FakeDocStatusStore()
    config = _config()
    config.processing.deduplicate = True
    text = "Vendor A supplies parts to Buyer under the yearly framework agreement."
    _write(root, {"original.txt": text})
    _run(root, store, config, mocker)

    _write(root, {copy_name: text})
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=root,
    )
    context.add_stage_result(
        DocumentParsingStage(config, source_directory=root).execute(context)
    )
    DocumentLoadingStage(config, source_directory=root, doc_status=store).execute(
        context
    )

    delta = context.incremental_delta
    assert (delta.new, delta.changed, delta.deleted) == ([], [], [])
    assert context.documents == []


@pytest.mark.parametrize("copy_name", ["a_copy.txt", "z_copy.txt"])
def test_dedup_drops_a_failed_original_for_its_new_copy(
    tmp_path, mocker, copy_name
) -> None:
    # The original used up its retries (FAILED, unchanged content), so it is
    # never re-indexed: keeping it over its near-copy would index neither.
    root = tmp_path / "corpus"
    store = FakeDocStatusStore()
    config = _config()
    config.processing.deduplicate = True
    text = "Vendor A supplies parts to Buyer under the yearly framework agreement."
    _write(root, {"original.txt": text})
    first = _run(root, store, config, mocker)
    original_id = _doc_id(first, "original.txt")
    record = store.get(original_id)
    assert record is not None
    store.put(
        record.model_copy(
            update={
                "status": DocStatus.FAILED,
                "failure_count": config.indexing.max_document_failures,
            }
        )
    )

    _write(root, {copy_name: text})
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=root,
    )
    context.add_stage_result(
        DocumentParsingStage(config, source_directory=root).execute(context)
    )
    DocumentLoadingStage(config, source_directory=root, doc_status=store).execute(
        context
    )

    delta = context.incremental_delta
    assert delta.new == [_doc_id(context, copy_name)]
    assert delta.deleted == [original_id]
    assert [d.file_name for d in context.documents] == [copy_name]


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


def _load_json_corpus(
    root: Path, store: FakeDocStatusStore, config: Config, mocker
) -> PipelineContext:
    """Run the loading stage over pre-parsed JSON (parsing disabled), then
    record every loaded document the way a successful commit does."""
    mocker.patch("boto3.Session")
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=root,
    )
    DocumentLoadingStage(config, source_directory=root, doc_status=store)._execute_core(
        context
    )
    lineages = build_document_lineage(documents=context.documents, text_units=[])
    IncrementalIndexer(
        store, mocker.MagicMock(), scope=context.incremental_scope
    )._record_processed(lineages, context.incremental_fingerprints)
    return context


def test_a_json_file_that_fails_to_load_is_not_deleted(tmp_path, mocker) -> None:
    # Same guarantee as for a parse failure, on the pre-parsed JSON path the
    # loading stage reads itself when parsing did not run.
    root = tmp_path / "corpus"
    root.mkdir()
    for name, text in {"a": "Vendor ships goods.", "b": "Buyer pays invoices."}.items():
        document = Document(
            page_content=text,
            document_id=name,
            file_name=f"{name}.txt",
            file_path=f"/source/{name}.txt",
        )
        (root / f"{name}.json").write_text(document.model_dump_json(), encoding="utf-8")
    store = FakeDocStatusStore()
    _load_json_corpus(root, store, _config(), mocker)
    assert len(store.list_all()) == 2

    (root / "b.json").write_text("{ not json", encoding="utf-8")
    context = _load_json_corpus(root, store, _config(), mocker)

    assert context.failed_source_files == [str((root / "b.json").resolve())]
    delta = context.incremental_delta
    assert delta.deleted == []
    assert delta.failed == [_doc_id(context, "b.json")]
    assert delta.unchanged == [_doc_id(context, "a.json")]
    assert store.get(_doc_id(context, "b.json")) is not None


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


# --- source scopes in the registry key ----------------------------------------

_SCOPE = "default|s3://bucket/corpus-1/"
_SOURCE = "s3://bucket/corpus-1/"


def _scoped_document(relative: str, text: str) -> Document:
    document = Document(
        page_content=text,
        document_id=relative,
        file_name=relative,
        file_path=f"/staging/{relative}",
    )
    assign_document_identity(document, "/staging")
    assign_registry_source(document, _SOURCE)
    return document


def test_source_scope_is_part_of_the_doc_id() -> None:
    assert compute_doc_id("a.txt", "default", "s1") != compute_doc_id(
        "a.txt", "default", "s2"
    )
    assert compute_doc_id("a.txt", "default", "s1") != compute_doc_id("a.txt")
    document = _scoped_document("a.txt", "Vendor ships goods.")
    assert document_doc_id(document) == compute_doc_id("a.txt", "default", _SOURCE)


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


def test_a_retired_scope_matches_every_stored_spelling_of_it(registry) -> None:
    # A configured source_scope is stored as given, so a local path can be
    # stored with a trailing or a repeated slash; the retire value is
    # normalized (validate_retired_scopes).
    document = _scoped_document("a.txt", "Vendor ships goods.")
    for doc_id, scope in (
        ("slash", f"{_OLD_SCOPE}/"),
        ("double", _OLD_SCOPE.replace("/old", "//old")),
        ("exact", _OLD_SCOPE),
        ("tenant-b", "tenant-b|/corpora/old/"),
        ("name", "default|corpora/old/"),
    ):
        registry.put(_record(doc_id, scope))

    delta, _ = detect_delta(
        [document], registry, scope=_SCOPE, retired_scopes=[_OLD_SCOPE]
    )

    assert sorted(delta.deleted) == ["double", "exact", "slash"]


def test_retiring_never_reaches_the_runs_own_stored_spelling() -> None:
    store = FakeDocStatusStore()
    store.put(_record("own", f"{_OLD_SCOPE}/"))
    store.put(_record("old", _OLD_SCOPE))
    delta = store.diff({}, scope=f"{_OLD_SCOPE}/")
    delta.deleted = []

    added = retire_source_scopes(store, delta, [_OLD_SCOPE], own_scope=f"{_OLD_SCOPE}/")

    assert (added, delta.deleted) == (1, ["old"])


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


def test_validate_retired_scopes_normalizes_paths_and_s3_uris() -> None:
    # An s3:// URI loses its trailing slash like a path (aws s3 sync reads
    # both spellings as one folder); other URIs and names are kept.
    assert validate_retired_scopes(
        [
            "/corpora/old/",
            "/corpora/old",
            "s3://bucket/old/",
            "s3://bucket/old",
            "https://host/old/",
            "corpus-a",
        ],
        "s3://bucket/new/",
        "tenant-a",
    ) == [
        "tenant-a|/corpora/old",
        "tenant-a|s3://bucket/old",
        "tenant-a|https://host/old/",
        "tenant-a|corpus-a",
    ]
    with pytest.raises(ConfigurationError, match="own source scope"):
        validate_retired_scopes(["s3://bucket/new/"], "s3://bucket/new/", "t")
    with pytest.raises(ConfigurationError, match="own source scope"):
        validate_retired_scopes(["s3://bucket/new"], "s3://bucket/new/", "t")
    with pytest.raises(ConfigurationError, match="empty"):
        validate_retired_scopes(["  "], "/corpora/new", "default")


def test_other_local_scopes_names_only_local_directories_of_the_namespace() -> None:
    delta = DocumentDelta(
        stored_scopes=[
            "default|/corpora/new",
            "default|/corpora/old",
            "default|/corpora/retired",
            "default|/corpora/retired/",
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
    pipeline.config.providers.doc_status.enabled = False
    pipeline.config.indexing.retire_source_scopes = ["/corpora/old"]
    pipeline.providers = Providers(pipeline.config)
    with pytest.raises(ConfigurationError, match="needs incremental indexing"):
        pipeline._validate_retired_scopes(tmp_path)

    pipeline.providers = Providers(pipeline.config, doc_status=FakeDocStatusStore())
    pipeline._validate_retired_scopes(tmp_path)
    pipeline.config.indexing.retire_source_scopes = [f"{tmp_path.as_posix()}/"]
    with pytest.raises(ConfigurationError, match="own source scope"):
        pipeline._validate_retired_scopes(tmp_path)


def test_retired_scope_readings(tmp_path: Path, monkeypatch) -> None:
    real = tmp_path / "real"
    (real / "old").mkdir(parents=True)
    (tmp_path / "link").symlink_to(real)
    monkeypatch.chdir(tmp_path)
    resolved = (real / "old").resolve().as_posix()
    assert retired_scope_readings("s3://bucket/old/") == (["s3://bucket/old/"], False)
    assert retired_scope_readings("./real/old/../old") == ([resolved], False)
    # One directory, two spellings: both retired.
    assert retired_scope_readings(str(tmp_path / "link" / "old")) == (
        [str(tmp_path / "link" / "old"), resolved],
        False,
    )
    # A name or a relative path: two readings, at most one may have records.
    assert retired_scope_readings("old") == (
        ["old", (tmp_path / "old").resolve().as_posix()],
        True,
    )


def _bare_value_groups(tmp_path: Path, monkeypatch) -> tuple[list, str, str]:
    monkeypatch.chdir(tmp_path)
    groups = retired_scope_groups(
        ["corpus-b"], "/corpora/new", "default", retired_scope_readings
    )
    name = "default|corpus-b"
    path = f"default|{(tmp_path / 'corpus-b').resolve().as_posix()}"
    return groups, name, path


def test_a_bare_value_retires_the_one_reading_with_records(
    tmp_path: Path, monkeypatch
) -> None:
    groups, name, _ = _bare_value_groups(tmp_path, monkeypatch)
    store = FakeDocStatusStore()
    store.put(_record("named", name))
    delta = store.diff({}, scope="default|/corpora/new")

    assert retire_source_scopes(store, delta, groups, "default|/corpora/new") == 1
    assert delta.deleted == ["named"]


def test_a_bare_value_with_records_under_both_readings_stops(
    tmp_path: Path, monkeypatch
) -> None:
    # Before, both readings were retired: a corpus configured with the
    # source_scope name "corpus-b" lost its documents to a retire of the
    # directory ./corpus-b (or the other way round).
    groups, name, path = _bare_value_groups(tmp_path, monkeypatch)
    store = FakeDocStatusStore()
    store.put(_record("named", name))
    store.put(_record("moved", path))
    delta = store.diff({}, scope="default|/corpora/new")

    with pytest.raises(ConfigurationError, match="ambiguous"):
        retire_source_scopes(store, delta, groups, "default|/corpora/new")
    assert delta.deleted == []


def test_a_bare_value_without_records_warns_once(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    groups, _, _ = _bare_value_groups(tmp_path, monkeypatch)
    store = FakeDocStatusStore()
    delta = store.diff({}, scope="default|/corpora/new")

    assert retire_source_scopes(store, delta, groups, "default|/corpora/new") == 0
    assert caplog.text.count("has no doc-status records") == 1
    assert "'corpus-b'" in caplog.text
