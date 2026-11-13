# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``indexing.max_deletion_share``: a run that would delete most of its
scope's documents asks first, and files it skipped are never deleted.

The share is of the records the diff read for the run's scope
(``DocumentDelta.scope_documents``); a retired scope's records were asked
for, and files the run could not read (failed or skipped) are kept.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.fixtures.doc_status_records import all_records
from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.registries import isolate_loaders
from unified_kg_rag.adapters.ingestion.file_loaders import TextFileLoader
from unified_kg_rag.adapters.ingestion.parser import ParserFactory
from unified_kg_rag.application.cli import run_ingestion_pipeline
from unified_kg_rag.application.ingestion.incremental import (
    IncrementalIndexer,
    _confirm_mass_deletion,
    build_document_lineage,
)
from unified_kg_rag.application.ingestion.pipeline_stages import (
    DocumentLoadingStage,
    DocumentParsingStage,
)
from unified_kg_rag.domain.ingestion.delta_detector import (
    assign_document_identity,
    assign_registry_source,
    detect_delta,
)
from unified_kg_rag.domain.ingestion.document_identity import RELATIVE_PATH_KEY
from unified_kg_rag.domain.ingestion.seed_graph import (
    SEED_SOURCE_KIND,
    seed_relative_path,
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
from unified_kg_rag.shared.exceptions import ConfirmationDeclinedError

pytestmark = pytest.mark.unit

_SCOPE = "default|s3://bucket/corpus-1/"
_OLD_SCOPE = "default|/corpora/old"


def _delta(deleted: int, kept: int, retired: int = 0) -> DocumentDelta:
    gone = [f"d{k}" for k in range(deleted)]
    old = [f"r{k}" for k in range(retired)]
    return DocumentDelta(
        unchanged=[f"u{k}" for k in range(kept)],
        deleted=gone + old,
        retired=old,
        scope_documents=deleted + kept,
    )


class _Asked:
    def __init__(self, answer: bool) -> None:
        self.answer = answer
        self.questions: list[str] = []

    def __call__(self, question: str) -> bool:
        self.questions.append(question)
        return self.answer


@pytest.mark.parametrize(
    ("deleted", "kept", "share", "asked"),
    [
        (6, 4, 0.5, True),  # 60% of 10
        (5, 5, 0.5, False),  # exactly the share
        (6, 3, 0.5, False),  # 9 documents: under the 10 a checked scope needs
        (1, 9, 0.0, True),  # 0.0: any deletion asks
        (10, 0, 1.0, False),  # 1.0: off
        (10, 0, None, False),  # null: off
    ],
)
def test_the_share_and_the_scope_size_decide_whether_the_run_asks(
    deleted: int, kept: int, share: float | None, asked: bool
) -> None:
    confirm = _Asked(True)
    _confirm_mass_deletion(_delta(deleted, kept), share, "s", confirm)
    assert bool(confirm.questions) is asked


def test_a_retired_scopes_records_do_not_count() -> None:
    # --retire-source-scope asked for those: 40 retired, 2 of 10 own deleted.
    confirm = _Asked(True)
    _confirm_mass_deletion(_delta(2, 8, retired=40), 0.5, "s", confirm)
    assert confirm.questions == []


def test_seed_shard_removals_do_not_count() -> None:
    # The configuration removed 30 shards; 2 of the 12 documents are gone.
    shards = [f"s{k}" for k in range(30)]
    delta = _delta(2, 10)
    delta.deleted += shards
    delta.seed_deleted = shards
    confirm = _Asked(True)
    _confirm_mass_deletion(delta, 0.5, "s", confirm)
    assert confirm.questions == []


def test_no_confirmation_or_a_no_stops_the_run() -> None:
    with pytest.raises(ConfirmationDeclinedError, match="--confirm-deletions"):
        _confirm_mass_deletion(_delta(8, 4), 0.5, "s3://bucket/corpus-1/", None)
    confirm = _Asked(False)
    with pytest.raises(ConfirmationDeclinedError, match="Deletion cancelled"):
        _confirm_mass_deletion(_delta(8, 4), 0.5, "s3://bucket/corpus-1/", confirm)
    assert confirm.questions == [
        "This run deletes 8 of the 12 documents the doc-status registry holds "
        "for source scope 's3://bucket/corpus-1/' (67%, over "
        "indexing.max_deletion_share 0.5): their files are not in the corpus, "
        "and what only they stated is removed from the index. A partial sync "
        "or a missing volume looks the same; continue?"
    ]


def _document(relative: str) -> Document:
    document = Document(
        page_content=f"{relative} text.",
        document_id=relative,
        file_name=relative,
        file_path=f"/staging/{relative}",
    )
    assign_document_identity(document, "/staging")
    assign_registry_source(document, "s3://bucket/corpus-1/")
    return document


def _record(doc_id: str, scope: str) -> DocStatusRecord:
    return DocStatusRecord(
        doc_id=doc_id, content_hash="h", status=DocStatus.PROCESSED, scope=scope
    )


def test_detect_delta_counts_the_scopes_records_and_marks_retired_ones() -> None:
    store = FakeDocStatusStore()
    kept = [_document(f"k{k}.txt") for k in range(3)]
    delta, fingerprints = detect_delta(kept, store, scope=_SCOPE)
    for doc_id in fingerprints:
        store.put(_record(doc_id, _SCOPE))
    for k in range(2):
        store.put(_record(f"gone-{k}", _SCOPE))
        store.put(_record(f"old-{k}", _OLD_SCOPE))
    store.put(_record("other", "default|/corpora/other"))

    delta, _ = detect_delta(
        [*kept, _document("new.txt")],
        store,
        scope=_SCOPE,
        failed_doc_ids=["gone-1"],
        retired_scopes=[_OLD_SCOPE],
    )

    # 3 still in the corpus + 2 gone (one of them unread, so kept); the new
    # document has no record and another scope's records are not this one's.
    assert delta.scope_documents == 5
    assert sorted(delta.deleted) == ["gone-0", "old-0", "old-1"]
    assert sorted(delta.retired) == ["old-0", "old-1"]


def test_detect_delta_leaves_seed_shards_out_of_the_share() -> None:
    store = FakeDocStatusStore()
    documents = [_document(f"k{k}.txt") for k in range(12)]
    shards = [_document(seed_relative_path("catalog", k, 4)) for k in range(4)]
    for shard in shards:
        shard.metadata["source_kind"] = SEED_SOURCE_KIND
    _, fingerprints = detect_delta([*documents, *shards], store, scope=_SCOPE)
    for doc_id, document in zip(fingerprints, [*documents, *shards], strict=True):
        store.put(
            _record(doc_id, _SCOPE).model_copy(
                update={"file_path": document.metadata[RELATIVE_PATH_KEY]}
            )
        )

    # 11 documents gone and the seed graph resharded (its 4 shards gone, 2 new).
    resharded = [_document(seed_relative_path("catalog", k, 2)) for k in range(2)]
    for shard in resharded:
        shard.metadata["source_kind"] = SEED_SOURCE_KIND
    delta, _ = detect_delta([documents[-1], *resharded], store, scope=_SCOPE)

    assert len(delta.deleted) == 15
    assert sorted(delta.seed_deleted) == sorted(list(fingerprints)[12:])
    assert delta.scope_documents == 12


# --- files no loader reads (O-H1) ----------------------------------------------


def _config() -> Config:
    config = Config()
    config.providers.doc_status.enabled = True
    return config


def _run(
    root: Path, store: FakeDocStatusStore, mocker: Any, confirm: Any = None
) -> PipelineContext:
    """Parse ``root``, run the loading stage's incremental filter, then
    record the processed documents as a successful commit does."""
    context = PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=root,
    )
    config = _config()
    DocumentParsingStage(config, source_directory=root)._execute_core(context)
    loading = DocumentLoadingStage(
        config, source_directory=root, doc_status=store, confirm_deletions=confirm
    )
    kept, _ = loading._apply_incremental_filter(list(context.documents), context)
    lineages = build_document_lineage(documents=kept, text_units=[])
    IncrementalIndexer(
        store, mocker.MagicMock(), scope=context.incremental_scope
    )._record_processed(lineages, context.incremental_fingerprints)
    return context


def test_a_file_skipped_for_its_type_is_kept_not_deleted(
    tmp_path: Path, mocker: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "a.txt").write_text("Vendor ships goods.")
    (root / "b.note").write_text("Buyer pays invoices.")
    store = FakeDocStatusStore()
    loaders = isolate_loaders(monkeypatch)
    ParserFactory.register_loader(".note", TextFileLoader, {}, "Note")
    _run(root, store, mocker)
    assert len(all_records(store)) == 2
    loaders.unregister(".note")

    context = _run(root, store, mocker)

    assert context.skipped_source_files == [str(root / "b.note")]
    assert context.failed_source_files == []
    delta = context.incremental_delta
    assert delta is not None
    assert delta.deleted == []
    assert len(delta.failed) == 1
    assert len(all_records(store)) == 2


def test_the_loading_stage_asks_before_its_first_write(
    tmp_path: Path, mocker: Any
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    for k in range(10):
        (root / f"d{k}.txt").write_text(f"Vendor {k} ships goods.")
    store = FakeDocStatusStore()
    _run(root, store, mocker)
    before = {r.doc_id: r for r in all_records(store)}
    for k in range(6):
        (root / f"d{k}.txt").unlink()
    (root / "new.txt").write_text("Carrier moves parts.")

    with pytest.raises(ConfirmationDeclinedError):
        _run(root, store, mocker)
    # No write-ahead record for new.txt, no attempt opened.
    assert {r.doc_id: r for r in all_records(store)} == before

    context = _run(root, store, mocker, confirm=lambda question: True)
    assert context.incremental_delta is not None
    assert len(context.incremental_delta.deleted) == 6


# --- run-ingestion ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "env", "expected"),
    [
        ([], None, False),
        (["--confirm-deletions"], None, True),
        ([], "1", True),
        ([], "no", False),
    ],
)
def test_the_deletion_confirmation_is_a_flag_or_an_env_var(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    env: str | None,
    expected: bool,
) -> None:
    monkeypatch.delenv("UKR_CONFIRM_RESET", raising=False)
    if env is None:
        monkeypatch.delenv("UKR_CONFIRM_DELETIONS", raising=False)
    else:
        monkeypatch.setenv("UKR_CONFIRM_DELETIONS", env)
    monkeypatch.setattr(sys, "argv", ["run-ingestion", *argv])
    args = run_ingestion_pipeline.CommandLineInterface().parse_args()
    assert args.confirm_deletions is expected
    assert args.confirm_reset is False


def test_an_unreadable_deletion_confirmation_env_var_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UKR_CONFIRM_DELETIONS", "maybe")
    monkeypatch.setattr(sys, "argv", ["run-ingestion"])
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.CommandLineInterface().parse_args()
    assert exc.value.code == 2


def test_each_confirmation_answers_its_own_question(mocker: Any) -> None:
    mocker.patch.object(run_ingestion_pipeline.sys.stdin, "isatty", return_value=False)
    runner: Any = run_ingestion_pipeline.IngestionPipelineRunner.__new__(
        run_ingestion_pipeline.IngestionPipelineRunner
    )
    runner.args = SimpleNamespace(
        yes=False, confirm_reset=True, confirm_deletions=False
    )
    with pytest.raises(
        ConfirmationDeclinedError,
        match=r"--confirm-deletions \(or UKR_CONFIRM_DELETIONS=1\)",
    ):
        runner._confirm_deletions("Delete them?")
    runner.args = SimpleNamespace(
        yes=False, confirm_reset=False, confirm_deletions=True
    )
    assert runner._confirm_deletions("Delete them?") is True
    with pytest.raises(ConfirmationDeclinedError):
        runner._confirm_reset("Drop it?")
    runner.args = SimpleNamespace(
        yes=True, confirm_reset=False, confirm_deletions=False
    )
    assert runner._confirm_deletions("Delete them?") is True
