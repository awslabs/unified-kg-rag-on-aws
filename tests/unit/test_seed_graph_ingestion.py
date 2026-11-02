# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed graphs in the ingestion stages: virtual documents no model reads.

Loading appends the shards after deduplication, chunking cuts them into
their units as written, translation, extraction, gleaning and claims leave
their units to the parse, resolution stores their records under the seed
fingerprint, and the text units name only that record. The registry keeps a
shard unchanged when only the extraction settings changed, and a reset asks
before it drops another scope's seed documents.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.seed_graphs import EDGES, NODES, PROCURE, lines
from unified_kg_rag.adapters.ingestion.claim_extractor import ClaimExtractionStats
from unified_kg_rag.adapters.ingestion.gleaner import GleaningStats
from unified_kg_rag.adapters.ingestion.graph_extractor import ExtractionStats
from unified_kg_rag.adapters.ingestion.translator import TranslationStats
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.cli import run_ingestion_pipeline
from unified_kg_rag.application.ingestion import incremental
from unified_kg_rag.application.ingestion import pipeline as pipeline_module
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.extraction_store import with_record_keys
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.ingestion.cache_keys import unit_record_fingerprint
from unified_kg_rag.domain.ingestion.contributions import record_key, record_payload
from unified_kg_rag.domain.ingestion.delta_detector import (
    assign_document_identity,
    assign_registry_source,
    detect_delta,
    document_doc_id,
    fingerprint_documents,
    keep_seed_settings_unchanged,
    registry_scope,
)
from unified_kg_rag.domain.ingestion.document_identity import (
    RELATIVE_PATH_KEY,
    compute_doc_id,
)
from unified_kg_rag.domain.ingestion.seed_graph import (
    SEED_RECORD_FINGERPRINT,
    SeedConversion,
    SeedConversionSettings,
    convert_seed,
    is_seed_path,
    parse_seed_unit,
    seed_text_units,
)
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    DocStatusRecord,
    Document,
    DocumentDelta,
    Entity,
    PipelineConfig,
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
    PipelineStageType,
    TextUnit,
    TextUnitExtraction,
)
from unified_kg_rag.domain.models.config import LanguageCode, SeedGraphConfig
from unified_kg_rag.shared.exceptions import (
    ConfirmationDeclinedError,
    PipelineExecutionError,
    PipelineStageError,
    SeedGraphError,
)

SEED_PATH = "seed_graphs/procure/0000-of-0001.jsonl"


def _conversion(shards: int = 1, items: int = 50) -> SeedConversion:
    return convert_seed(
        PROCURE,
        SeedConversionSettings(name="procure", shards=shards, max_items_per_unit=items),
        Config().processing.graph_extraction.extraction_schema(),
    )


def _context(**fields: Any) -> PipelineContext:
    return PipelineContext(
        pipeline_id="pid",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory="/corpus",
        **fields,
    )


def _document(name: str, text: str) -> Document:
    return Document(
        page_content=text,
        document_id=f"doc-{name}",
        file_name=name,
        file_path=f"/corpus/{name}",
        metadata={RELATIVE_PATH_KEY: name},
    )


def _seed_units(items: int = 1) -> list[TextUnit]:
    (document,) = _conversion(items=items).documents()
    return seed_text_units(document)


def _doc_unit(unit_id: str = "u1", text: str = "Acme Supplies sells.") -> TextUnit:
    return TextUnit(id=unit_id, text=text, document_ids=["doc-a"])


# -- Identity and paths ---------------------------------------------------------


def test_seed_paths_are_told_from_document_paths() -> None:
    assert is_seed_path(SEED_PATH)
    assert is_seed_path("seed_graphs/my-seed_2/0003-of-0016.jsonl")
    for path in (
        None,
        "",
        "seed_graphs/procure.jsonl",
        "docs/seed_graphs/procure/0000-of-0001.jsonl",
        "seed_graphs/Procure/0000-of-0001.jsonl",
        "seed_graphs/procure/0000-of-0001.json",
        "seed_graphs/procure/notes.jsonl",
    ):
        assert not is_seed_path(path), path


def test_a_shard_keeps_its_seed_path_whatever_the_corpus_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Resolved against the working directory, the relative seed path would
    # read as <cwd>/seed_graphs/... under a root above it.
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    ids = set()
    for root in (tmp_path, Path("."), tmp_path / "elsewhere"):
        (document,) = _conversion().documents()
        assign_document_identity(document, root)
        assert document.metadata[RELATIVE_PATH_KEY] == SEED_PATH
        ids.add(document.document_id)
    assert len(ids) == 1


# -- The settings-changed exemption ----------------------------------------------


def _record(path: str, scope: str, **fields: Any) -> DocStatusRecord:
    values: dict[str, Any] = {
        "doc_id": compute_doc_id(path, "default", scope.partition("|")[2]),
        "content_hash": "h",
        "scope": scope,
        "status": DocStatus.PROCESSED,
        "file_path": path,
        "extraction_fingerprint": "old",
    }
    values.update(fields)
    return DocStatusRecord(**values)


def test_only_settings_changed_seed_shards_move_back_to_unchanged() -> None:
    delta = DocumentDelta(
        changed=["seed", "doc", "edited-seed"],
        settings_changed=["seed", "doc"],
        unchanged=["same"],
    )
    moved = keep_seed_settings_unchanged(delta, {"seed", "edited-seed"})
    assert moved == ["seed"]
    assert delta.changed == ["doc", "edited-seed"]
    assert delta.settings_changed == ["doc"]
    assert delta.unchanged == ["same", "seed"]
    assert keep_seed_settings_unchanged(delta, set()) == []


def _registry_with(
    documents: list[Document], scope: str, **fields: Any
) -> FakeDocStatusStore:
    store = FakeDocStatusStore()
    for doc_id, content_hash in fingerprint_documents(documents).items():
        document = next(d for d in documents if document_doc_id(d) == doc_id)
        store.put(
            DocStatusRecord(
                doc_id=doc_id,
                content_hash=content_hash,
                scope=scope,
                status=DocStatus.PROCESSED,
                file_path=document.metadata[RELATIVE_PATH_KEY],
                extraction_fingerprint="old",
                **fields,
            )
        )
    return store


def _identified(documents: list[Document]) -> list[Document]:
    for document in documents:
        assign_document_identity(document, "/corpus")
        assign_registry_source(document, "/corpus")
    return documents


def test_detect_delta_keeps_seed_shards_on_a_settings_change() -> None:
    documents = _identified([_document("a.txt", "Acme."), *_conversion().documents()])
    doc_id, seed_id = (document_doc_id(d) for d in documents)
    scope = registry_scope("default", "/corpus")
    store = _registry_with(documents, scope)

    delta, _ = detect_delta(documents, store, scope=scope, extraction_fingerprint="new")

    assert (delta.changed, delta.settings_changed) == ([doc_id], [doc_id])
    assert delta.unchanged == [seed_id]
    # The same settings: nothing changes.
    delta, _ = detect_delta(documents, store, scope=scope, extraction_fingerprint="old")
    assert sorted(delta.unchanged) == sorted([doc_id, seed_id])


def test_an_exempt_quarantined_seed_shard_stays_releasable() -> None:
    documents = _identified(_conversion().documents())
    (seed_id,) = (document_doc_id(d) for d in documents)
    scope = registry_scope("default", "/corpus")
    store = _registry_with(documents, scope)
    store.put(
        store.get_many([seed_id])[seed_id].model_copy(
            update={"status": DocStatus.QUARANTINED}
        )
    )
    delta, _ = detect_delta(documents, store, scope=scope, extraction_fingerprint="new")
    assert delta.unchanged == [seed_id]
    assert delta.quarantined_unchanged == [seed_id]


def test_a_resumed_delta_still_applies_over_an_exempt_seed_shard() -> None:
    documents = _identified([_document("a.txt", "Acme."), *_conversion().documents()])
    scope = registry_scope("default", "/corpus")
    store = _registry_with(documents, scope, written_by="pid")
    delta, fingerprints = detect_delta(
        documents, store, scope=scope, extraction_fingerprint="new"
    )
    for record in store.get_many(delta.changed).values():
        store.put(record.model_copy(update={"attempted_hash": record.content_hash}))

    assert incremental.delta_still_applies(
        store,
        delta,
        fingerprints,
        max_attempts=3,
        extraction_fingerprint="new",
        scope=scope,
        pipeline_id="pid",
    )
    # A document shard the delta kept unchanged would be another run's work.
    (seed_id,) = delta.unchanged
    record = store.get_many([seed_id])[seed_id]
    store.put(record.model_copy(update={"file_path": "notes/0000-of-0001.jsonl"}))
    assert not incremental.delta_still_applies(
        store,
        delta,
        fingerprints,
        max_attempts=3,
        extraction_fingerprint="new",
        scope=scope,
        pipeline_id="pid",
    )


# -- The reset guard ---------------------------------------------------------------


def _guard(store: FakeDocStatusStore, confirm: Any) -> None:
    incremental._confirm_reset_of_seed_scopes(
        lambda: store, "default", "default|/corpus", confirm
    )


def test_a_reset_asks_only_for_another_scopes_seed_documents() -> None:
    store = FakeDocStatusStore()
    store.put(_record(SEED_PATH, "default|/corpus"))
    store.put(_record("a.txt", "default|other"))
    store.put(_record("seed_graphs/x/notes.txt", "default|other"))
    asked: list[str] = []
    # The run's own seed and another scope's documents: no question.
    _guard(store, lambda q: asked.append(q) or True)
    assert asked == []

    store.put(_record(SEED_PATH, "default|seeds-procure"))
    _guard(store, lambda q: asked.append(q) or True)
    assert asked == [
        "The reset of index namespace 'default' drops the seed graph documents "
        "of source scope(s) seeds-procure; continue?"
    ]
    with pytest.raises(PipelineExecutionError, match="Reset cancelled"):
        _guard(store, lambda q: False)
    with pytest.raises(ConfirmationDeclinedError, match="--confirm-reset"):
        _guard(store, None)


def test_a_reset_is_refused_before_any_registry_write(mocker) -> None:
    store = FakeDocStatusStore()
    store.put(_record(SEED_PATH, "default|seeds-procure"))
    config = Config()
    config.indexing.reset = True
    writes = mocker.spy(store, "put_many")
    with pytest.raises(PipelineExecutionError, match="seeds-procure"):
        incremental.filter_to_delta(
            [_document("a.txt", "Acme.")],
            _context(),
            config,
            lambda: store,
            Path("/corpus"),
        )
    writes.assert_not_called()
    documents, _ = incremental.filter_to_delta(
        [_document("a.txt", "Acme.")],
        _context(),
        config,
        lambda: store,
        Path("/corpus"),
        lambda q: True,
    )
    assert len(documents) == 1


def _seeds_alone(
    store: FakeDocStatusStore, reset: bool = False, confirm: Any = None
) -> list[Document]:
    """filter_to_delta over a corpus of the seed's shards alone."""
    config = Config()
    config.indexing.reset = reset
    documents, _ = incremental.filter_to_delta(
        list(_conversion().documents()),
        _context(),
        config,
        lambda: store,
        Path("/corpus"),
        confirm,
    )
    return documents


def test_seeds_alone_never_delete_the_scopes_documents(mocker) -> None:
    # An empty corpus (a volume not mounted yet) beside the scope's
    # documents: indexing the seeds alone would delete them.
    store = FakeDocStatusStore()
    store.put(_record("a.txt", "default|/corpus"))
    store.put(_record(SEED_PATH, "default|/corpus"))
    writes = mocker.spy(store, "put_many")
    with pytest.raises(PipelineStageError) as error:
        _seeds_alone(store)
    message = str(error.value)
    assert "holds 1 documents of source scope '/corpus' (a.txt)" in message
    assert "--retire-source-scope /corpus" in message
    assert "processing.document_parsing.source_scope" in message
    writes.assert_not_called()
    # A reset of the scope from seeds alone fails too, whatever the answer.
    with pytest.raises(PipelineStageError, match="a.txt"):
        _seeds_alone(store, reset=True, confirm=lambda q: True)
    writes.assert_not_called()


def test_seeds_alone_run_a_scope_without_documents() -> None:
    # A new scope, or a seed-only one; another scope's documents are not
    # this incremental run's.
    store = FakeDocStatusStore()
    assert len(_seeds_alone(store)) == 1
    store.put(_record("a.txt", "default|documents"))
    assert len(_seeds_alone(store)) == 1


def test_a_reset_from_seeds_alone_asks_before_it_drops_documents() -> None:
    store = FakeDocStatusStore()
    store.put(_record("a.txt", "default|documents"))
    asked: list[str] = []
    assert len(_seeds_alone(store, True, lambda q: asked.append(q) or True)) == 1
    assert asked == [
        "The reset of index namespace 'default' rebuilds it from this run's seed "
        "graphs alone (no source document reached the run), dropping the "
        "documents of source scope(s) documents; continue?"
    ]
    with pytest.raises(ConfirmationDeclinedError, match="Reset cancelled"):
        _seeds_alone(store, True, lambda q: False)
    with pytest.raises(ConfirmationDeclinedError, match="--confirm-reset"):
        _seeds_alone(store, True)
    # One question for both: another scope's seed and its documents.
    store.put(_record(SEED_PATH, "default|seeds-procure"))
    asked.clear()
    _seeds_alone(store, True, lambda q: asked.append(q) or True)
    assert asked == [
        "The reset of index namespace 'default' drops the seed graph documents "
        "of source scope(s) seeds-procure and rebuilds it from this run's seed "
        "graphs alone (no source document reached the run), dropping the "
        "documents of source scope(s) documents; continue?"
    ]


def test_a_new_scope_importing_a_seed_another_scope_holds_is_warned(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = FakeDocStatusStore()
    store.put(_record(SEED_PATH, "default|/old-corpus"))
    store.put(_record("seed_graphs/parts/0000-of-0001.jsonl", "default|other"))
    _seeds_alone(store)
    (warned,) = [
        r.getMessage() for r in caplog.records if "already imported" in r.getMessage()
    ]
    assert (
        "Seed graph 'procure' is already imported by source scope(s) /old-corpus"
        in warned
    )
    # Its second run: the scope is no longer new.
    for record in store.list_namespace("default"):
        if record.scope == "default|/corpus":
            store.put(record.model_copy(update={"status": DocStatus.PROCESSED}))
    caplog.clear()
    _seeds_alone(store)
    assert "already imported" not in caplog.text


def test_run_ingestion_answers_the_reset_with_its_confirm(mocker) -> None:
    built = mocker.patch.object(run_ingestion_pipeline, "DataIngestionPipeline")
    runner: Any = run_ingestion_pipeline.IngestionPipelineRunner.__new__(
        run_ingestion_pipeline.IngestionPipelineRunner
    )
    runner.args = SimpleNamespace(yes=True, verify_metadata=True)
    runner.config = Config()
    for name in (
        "_display_run_info",
        "_require_indexing_endpoints",
        "_handle_metadata_operations",
        "_build_metrics_sink",
    ):
        mocker.patch.object(runner, name, create=True)
    mocker.patch.object(runner, "_create_pipeline_config")
    mocker.patch.object(run_ingestion_pipeline, "display_ascii_art")
    mocker.patch.object(
        run_ingestion_pipeline, "missing_extras_error", return_value=None
    )
    runner.args.source_directory = runner.args.output_directory = None
    runner.run()
    confirm = built.call_args.kwargs["confirm"]
    assert confirm("Drop it?") is True


@pytest.mark.parametrize(
    ("argv", "env", "expected"),
    [
        ([], None, False),
        (["--confirm-reset"], None, True),
        ([], "1", True),
        ([], " TRUE ", True),
        ([], "yes", True),
        ([], "0", False),
        ([], "", False),
        (["--confirm-reset"], "0", True),
    ],
)
def test_the_reset_confirmation_is_a_flag_or_an_env_var(
    monkeypatch: pytest.MonkeyPatch,
    argv: list[str],
    env: str | None,
    expected: bool,
) -> None:
    if env is None:
        monkeypatch.delenv("UKR_CONFIRM_RESET", raising=False)
    else:
        monkeypatch.setenv("UKR_CONFIRM_RESET", env)
    monkeypatch.setattr(sys, "argv", ["run-ingestion", *argv])
    args = run_ingestion_pipeline.CommandLineInterface().parse_args()
    assert args.confirm_reset is expected
    assert args.yes is False


def test_an_unreadable_reset_confirmation_env_var_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UKR_CONFIRM_RESET", "maybe")
    monkeypatch.setattr(sys, "argv", ["run-ingestion"])
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.CommandLineInterface().parse_args()
    assert exc.value.code == 2


@pytest.mark.parametrize(("yes", "confirm_reset"), [(False, True), (True, False)])
def test_the_reset_confirmation_answers_resets_only(
    mocker, yes: bool, confirm_reset: bool
) -> None:
    mocker.patch.object(run_ingestion_pipeline.sys.stdin, "isatty", return_value=False)
    runner: Any = run_ingestion_pipeline.IngestionPipelineRunner.__new__(
        run_ingestion_pipeline.IngestionPipelineRunner
    )
    runner.args = SimpleNamespace(yes=yes, confirm_reset=confirm_reset)
    # --yes answers every question; --confirm-reset a reset's only, never an
    # integrity check's.
    assert runner._confirm_reset("Drop it?") is True
    if yes:
        assert runner._confirm("Continue despite integrity issues?") is True
    else:
        with pytest.raises(ConfirmationDeclinedError, match="pass --yes"):
            runner._confirm("Continue despite integrity issues?")
    runner.args = SimpleNamespace(yes=False, confirm_reset=False)
    with pytest.raises(
        ConfirmationDeclinedError, match=r"--confirm-reset \(or UKR_CONFIRM_RESET=1\)"
    ):
        runner._confirm_reset("Drop it?")


def test_a_reset_answered_no_stops_with_exit_1_and_no_traceback(
    mocker,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    question = "The reset of index namespace 'default' drops ...; continue?"
    mocker.patch.object(
        run_ingestion_pipeline, "load_cli_config", return_value=Config()
    )
    mocker.patch.object(run_ingestion_pipeline, "display_ascii_art")
    mocker.patch.object(
        run_ingestion_pipeline.IngestionPipelineRunner, "_require_indexing_endpoints"
    )
    built = mocker.patch.object(run_ingestion_pipeline, "DataIngestionPipeline")
    built.return_value.state_manager.pipeline_exists.return_value = False
    built.return_value.run.side_effect = ConfirmationDeclinedError(
        f"Reset cancelled: {question}"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-ingestion",
            "--source-directory",
            str(tmp_path),
            "--pipeline-id",
            "p1",
            "--output-format",
            "json",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.main()
    assert exc.value.code == 1
    out = capsys.readouterr()
    assert "Pipeline stopped before it wrote anything: Reset cancelled" in " ".join(
        out.err.split()
    )
    assert json.loads(out.out)["exit_code"] == 1
    # No recovery suggestions, no error logged.
    assert "Recovery suggestions" not in out.err
    assert not [r for r in caplog.records if r.levelname == "ERROR"]


# -- Pipeline start-up -------------------------------------------------------------


def test_the_pipeline_imports_its_seeds_before_it_builds_a_model(
    mocker, tmp_path: Path
) -> None:
    mocker.patch.object(
        pipeline_module, "build_providers", side_effect=AssertionError("built")
    )
    config = Config()
    config.processing.seed_graphs = [
        SeedGraphConfig(name="procure", kind="jsonl", path=str(tmp_path / "none"))
    ]
    with pytest.raises(SeedGraphError):
        DataIngestionPipeline(config, PipelineConfig(local_directory=tmp_path))


def test_the_pipeline_hands_its_seeds_to_the_loading_stage(
    mocker, tmp_path: Path
) -> None:
    seed = tmp_path / "procure.jsonl"
    seed.write_text(lines(*NODES, *EDGES))
    config = Config()
    config.processing.seed_graphs = [
        SeedGraphConfig(name="procure", kind="jsonl", path=str(seed))
    ]
    confirm = mocker.Mock()
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            local_directory=tmp_path / "cache",
            stages_enabled={
                stage: stage
                in {
                    PipelineStageType.DOCUMENT_PARSING,
                    PipelineStageType.DOCUMENT_LOADING,
                }
                for stage in PipelineStageType
            },
        ),
        source_directory=tmp_path,
        providers=mocker.MagicMock(),
        confirm=confirm,
    )
    (loading,) = [s for s in pipeline.stages if isinstance(s, ps.DocumentLoadingStage)]
    assert loading.seed_conversions == pipeline.seed_conversions
    assert [c.name for c in pipeline.seed_conversions] == ["procure"]
    assert loading.confirm is confirm
    # Rebuilt for another corpus root, it keeps them.
    other = tmp_path / "other"
    other.mkdir()
    pipeline._use_source_directory(other)
    (loading,) = [s for s in pipeline.stages if isinstance(s, ps.DocumentLoadingStage)]
    assert loading.seed_conversions == pipeline.seed_conversions
    assert loading.confirm is confirm


def test_a_bad_seed_exits_2_before_any_model_call(
    mocker, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = Config()
    config.processing.seed_graphs = [
        SeedGraphConfig(name="procure", kind="jsonl", path=str(tmp_path / "none"))
    ]
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    mocker.patch.object(
        run_ingestion_pipeline.IngestionPipelineRunner, "_require_indexing_endpoints"
    )
    mocker.patch.object(
        pipeline_module, "build_providers", side_effect=AssertionError("built")
    )
    source = tmp_path / "corpus"
    source.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-ingestion",
            "--source-directory",
            str(source),
            "--cache-directory",
            str(tmp_path / "cache"),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.main()
    assert exc.value.code == 2


# -- Parsing and loading ---------------------------------------------------------------


def _seeded_config() -> Config:
    config = Config()
    config.processing.seed_graphs = [
        SeedGraphConfig(name="procure", kind="jsonl", path="seed.jsonl")
    ]
    return config


@pytest.mark.parametrize("seeded", [True, False])
def test_an_empty_corpus_is_a_seed_only_run(tmp_path: Path, seeded: bool) -> None:
    config = _seeded_config() if seeded else Config()
    stage = ps.DocumentParsingStage(config, source_directory=tmp_path)
    context = _context(documents=[_document("stale.txt", "x")])
    result = stage.execute(context)
    expected = PipelineStageStatus.COMPLETED if seeded else PipelineStageStatus.FAILED
    assert result.status is expected
    if seeded:
        assert context.documents == []


def test_a_corpus_whose_files_all_failed_still_fails(tmp_path: Path) -> None:
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")
    stage = ps.DocumentParsingStage(_seeded_config(), source_directory=tmp_path)
    result = stage.execute(_context())
    assert result.status is PipelineStageStatus.FAILED


def _parsed() -> PipelineStageResult:
    now = datetime(2026, 1, 1)
    return PipelineStageResult(
        stage_name=PipelineStageType.DOCUMENT_PARSING.value,
        status=PipelineStageStatus.COMPLETED,
        start_time=now,
        end_time=now,
        duration_seconds=0,
    )


def test_loading_appends_the_seed_documents_after_deduplication(
    mocker, tmp_path: Path
) -> None:
    config = Config()
    config.processing.deduplicate = True
    config.processing.document_parsing.index_value = "tenant1"
    conversion = _conversion(shards=2)
    stage = ps.DocumentLoadingStage(
        config,
        source_directory=tmp_path,
        providers=mocker.MagicMock(),
        seed_conversions=[conversion],
    )
    dedupe = mocker.spy(stage.loader, "deduplicate_documents")
    documents = [_document("a.txt", "Acme."), _document("b.txt", "Acme.")]
    for document in documents:
        assign_document_identity(document, tmp_path)
    context = _context(documents=documents, stage_results=[_parsed()])

    inputs, outputs, metrics = stage._execute_core(context)

    # Seeds are neither deduplicated nor near-duplicate checked.
    assert all("seed_graph" not in d.metadata for d in dedupe.call_args.args[0])
    seeds = context.documents[1:]
    assert [d.metadata[RELATIVE_PATH_KEY] for d in seeds] == [
        shard.relative_path for shard in conversion.documents_content
    ]
    assert all(d.metadata["index"] == "tenant1" for d in seeds)
    assert (inputs, outputs) == (2 + len(seeds), 1 + len(seeds))
    assert (metrics["seed_documents"], metrics["seed_documents_skipped"]) == (
        len(seeds),
        0,
    )


def test_unchanged_seed_shards_are_skipped_like_documents(
    mocker, tmp_path: Path
) -> None:
    config = Config()
    store = FakeDocStatusStore()
    stage = ps.DocumentLoadingStage(
        config,
        source_directory=tmp_path,
        doc_status=store,
        providers=mocker.MagicMock(),
        seed_conversions=[_conversion(shards=2)],
    )
    context = _context(documents=[], stage_results=[_parsed()])
    stage._execute_core(context)
    shards = list(context.documents)
    assert context.incremental_fingerprints
    # Recorded as the run's scope: the next run diffs them with the corpus.
    scope = context.incremental_scope
    for document in shards:
        doc_id = document_doc_id(document)
        store.put(
            DocStatusRecord(
                doc_id=doc_id,
                content_hash=context.incremental_fingerprints[doc_id],
                scope=scope,
                status=DocStatus.PROCESSED,
                file_path=document.metadata[RELATIVE_PATH_KEY],
            )
        )
    context = _context(documents=[], stage_results=[_parsed()])
    _, outputs, metrics = stage._execute_core(context)
    assert (outputs, metrics["seed_documents_skipped"]) == (0, len(shards))


# -- Chunking, translation, extraction, gleaning, claims --------------------------


def test_chunking_cuts_seed_shards_without_the_chunker(mocker) -> None:
    chunker = mocker.MagicMock()
    chunker.chunk_documents.side_effect = lambda docs: [
        TextUnit(id=f"u-{d.file_name}", text=d.page_content) for d in docs
    ]
    mocker.patch.object(ps.ChunkerFactory, "create_chunker", return_value=chunker)
    stage = ps.TextChunkingStage(Config(), providers=mocker.MagicMock())
    (seed,) = _conversion(items=1).documents()
    context = _context(documents=[seed, _document("a.txt", "Acme.")])

    _, outputs, metrics = stage._execute_core(context)

    (documents,) = chunker.chunk_documents.call_args.args
    assert [d.file_name for d in documents] == ["a.txt"]
    expected = seed_text_units(seed, count_tokens=estimate_token_count)
    assert context.text_units[0].id == "u-a.txt"
    assert context.text_units[1:] == expected
    assert all(unit.n_tokens for unit in expected)
    assert (outputs, metrics["seed_units"]) == (1 + len(expected), len(expected))


def test_translation_leaves_seed_units_as_written(mocker) -> None:
    translator = mocker.MagicMock()

    def translate(units: list[TextUnit]) -> list[TextUnit]:
        for unit in units:
            unit.translated_texts = {"ko": "번역"}
        return units

    translator.translate_text_units.side_effect = translate
    translator.stats = TranslationStats()
    mocker.patch.object(ps, "TextUnitTranslator", return_value=translator)
    config = Config()
    config.processing.translation.enabled = True
    config.processing.translation.additional_target_languages = [LanguageCode.KO]
    stage = ps.TranslationStage(config=config)
    seed = _seed_units()[0]
    context = _context(text_units=[_doc_unit(), seed])

    stage._execute_core(context)

    (sent,) = translator.translate_text_units.call_args.args
    assert [u.id for u in sent] == ["u1"]
    assert [u.id for u in context.translated_units] == ["u1", seed.id]
    assert context.translated_units[1].translated_texts is None
    assert context.failed_text_unit_ids["translation"] == []


def _store(mocker, loaded: dict[str, dict[str, Any]] | None = None) -> Any:
    """The stages' extraction store: ``loaded`` payloads by fingerprint."""
    store = mocker.patch.object(ps, "ExtractionStore", autospec=True).return_value
    loaded = {} if loaded is None else loaded

    def load(units, fingerprint, parse):
        return {
            unit.id: parse(loaded[fingerprint], unit)
            for unit in units
            if fingerprint in loaded
        }

    store.load.side_effect = load
    store.save.return_value = set()
    return store


def test_extraction_parses_seed_units_and_never_sends_them(mocker) -> None:
    extractor = mocker.patch.object(ps, "GraphExtractor", autospec=True).return_value
    extractor.extract_units.side_effect = lambda units: (
        [TextUnitExtraction(text_unit_id=u.id) for u in units],
        ExtractionStats(),
    )
    seeds = _seed_units()
    stored = record_payload(parse_seed_unit(seeds[0]))
    loaded: dict[str, dict[str, Any]] = {}
    store = _store(mocker, loaded)
    stage = ps.GraphExtractionStage(Config(), providers=mocker.MagicMock())
    context = _context(text_units=[_doc_unit(), *seeds])

    _, _, metrics = stage._execute_core(context)

    (sent,) = extractor.extract_units.call_args.args
    assert [u.id for u in sent] == ["u1"]
    fingerprints = [call.args[1] for call in store.load.call_args_list]
    assert SEED_RECORD_FINGERPRINT in fingerprints
    records = {r.text_unit_id: r for r in context.unit_extractions}
    for unit in seeds:
        assert records[unit.id] == parse_seed_unit(unit)
    assert (metrics["seed_units"], metrics["text_units_reused"]) == (len(seeds), 0)
    assert metrics["seed_entities"] and metrics["seed_relationships"]

    # A stored seed record is reused (it is the parse of the same text).
    loaded[SEED_RECORD_FINGERPRINT] = stored
    stage = ps.GraphExtractionStage(Config(), providers=mocker.MagicMock())
    context = _context(text_units=[seeds[0]])
    stage._execute_core(context)
    (record,) = context.unit_extractions
    assert record.stored
    extractor.extract_units.assert_called_once()


def test_gleaning_refines_document_units_only(mocker) -> None:
    gleaner = mocker.patch.object(ps, "GraphGleaner", autospec=True).return_value
    gleaner.glean.side_effect = lambda units, records: (records, GleaningStats())
    seed = _seed_units()[0]
    records = [TextUnitExtraction(text_unit_id="u1"), parse_seed_unit(seed)]
    stage = ps.GleaningStage(Config())
    context = _context(text_units=[_doc_unit(), seed], unit_extractions=records)

    stage._execute_core(context)

    units, given = gleaner.glean.call_args.args
    assert [u.id for u in units] == ["u1"]
    assert given == records


def test_resolution_stores_seed_records_under_the_seed_fingerprint(mocker) -> None:
    store = _store(mocker)
    mocker.patch.object(ps, "DescriptionSummarizer", autospec=True)
    stage = ps.GraphResolutionStage(Config(), providers=mocker.MagicMock())
    seed = _seed_units()[0]
    context = _context(
        text_units=[_doc_unit(), seed],
        unit_extractions=[
            TextUnitExtraction(text_unit_id="u1"),
            parse_seed_unit(seed),
        ],
    )

    assert stage._store_records(context) == set()

    saved = {call.args[1]: call.args[0] for call in store.save.call_args_list}
    assert [u.id for u, _ in saved[SEED_RECORD_FINGERPRINT]] == [seed.id]
    document_fp = unit_record_fingerprint(Config(), gleaned=False)
    assert [u.id for u, _ in saved[document_fp]] == ["u1"]
    assert all(record.stored for record in context.unit_extractions)


def test_a_seed_record_not_stored_fails_its_unit(mocker) -> None:
    store = _store(mocker)
    seed = _seed_units()[0]
    store.save.side_effect = lambda items, fingerprint: (
        {seed.id} if fingerprint == SEED_RECORD_FINGERPRINT else set()
    )
    mocker.patch.object(ps, "DescriptionSummarizer", autospec=True)
    stage = ps.GraphResolutionStage(Config(), providers=mocker.MagicMock())
    context = _context(text_units=[seed], unit_extractions=[parse_seed_unit(seed)])
    assert stage._store_records(context) == {seed.id}


def test_claims_skip_seed_units(mocker) -> None:
    extractor = mocker.patch.object(ps, "ClaimExtractor", autospec=True).return_value
    extractor.extract_units.side_effect = lambda units, entities: (
        {},
        ClaimExtractionStats(failed_text_unit_ids=[u.id for u in units]),
    )
    store = _store(mocker)
    seed = _seed_units()[0]
    stage = ps.ClaimExtractionStage(Config(), providers=mocker.MagicMock())
    context = _context(
        text_units=[_doc_unit(), seed],
        unit_extractions=[
            TextUnitExtraction(text_unit_id="u1"),
            parse_seed_unit(seed),
        ],
    )

    stage._execute_core(context)

    units, _ = extractor.extract_units.call_args.args
    assert [u.id for u in units] == ["u1"]
    assert context.failed_text_unit_ids["claim_extraction"] == ["u1"]
    loaded = [u.id for call in store.load.call_args_list for u in call.args[0]]
    assert seed.id not in loaded
    # And no seed document is ever left without its claims.
    shard = _conversion().documents()[0]
    assign_document_identity(shard, "/corpus")
    context.documents = [shard]
    failures = incremental.run_failures(context, context.text_units)
    assert failures.claims_missing == set() and failures.failed == set()


def test_a_seed_unit_names_its_graph_record_only() -> None:
    config = Config()
    seed = _seed_units()[0]
    keyed = with_record_keys(
        config,
        [_doc_unit(), seed],
        [TextUnitExtraction(text_unit_id="u1"), parse_seed_unit(seed)],
        graph_failed=(),
        claims=True,
    )
    keys = {unit.id: unit.attributes["extraction_records"] for unit in keyed}
    assert keys[seed.id] == {"graph": record_key(SEED_RECORD_FINGERPRINT, seed.text)}
    assert set(keys["u1"]) == {"graph", "claims"}
    # A failed one names none.
    failed, _ = with_record_keys(
        config,
        [seed, _doc_unit()],
        [parse_seed_unit(seed)],
        graph_failed={seed.id},
        claims=True,
    )
    assert "extraction_records" not in (failed.attributes or {})


def test_seed_text_units_are_written_with_their_source_kind() -> None:
    indexer: Any = OpenSearchIndexer.__new__(OpenSearchIndexer)
    indexer.config = Config()
    seed = _seed_units()[0]
    doc = indexer._prepare_text_unit_doc(seed, ([0.0],))
    assert doc["source_kind"] == "seed_graph"
    assert doc["text"] == seed.text and doc["n_tokens"] == 0
    assert "source_kind" not in indexer._prepare_text_unit_doc(_doc_unit(), ([0.0],))


def test_the_seed_header_stays_out_of_every_unit() -> None:
    (document,) = _conversion(items=1).documents()
    header = document.page_content.split("\n", 1)[0]
    assert json.loads(header)["seed_graph"] == "procure"
    assert all(header not in unit.text for unit in seed_text_units(document))


# -- Community evidence on an incremental run -------------------------------------


@pytest.mark.parametrize(
    ("units", "kinds", "backed"),
    [
        (["sg-1", "u1"], ["document", "seed_graph"], True),
        # The document unit is one the run replaces (its document changed or
        # went): a full build no longer cites it.
        (["sg-1", "u-old"], ["document", "seed_graph"], False),
        (["sg-1"], ["seed_graph"], False),
        # Stored without lineage: the source kinds decide (none: written
        # before seed graphs, or document-only).
        ([], None, True),
        ([], ["document", "seed_graph"], True),
        ([], ["seed_graph"], False),
    ],
)
def test_a_stored_member_evidences_a_community_through_a_kept_document(
    units: list[str], kinds: list[str] | None, backed: bool
) -> None:
    entity = Entity(id="e", name="Vendor", text_unit_ids=units, source_kinds=kinds)
    assert ps._document_backed(entity, removed={"u-old"}) is backed
