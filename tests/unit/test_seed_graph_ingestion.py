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
import logging
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.fixtures.fakes.doc_status import FakeDocStatusStore
from tests.fixtures.fakes.stores import FakeGraphStore, FakeVectorStore
from tests.fixtures.seed_graphs import EDGES, NODES, PROCURE, lines, seed_settings
from unified_kg_rag.adapters.ingestion.claim_extractor import ClaimExtractionStats
from unified_kg_rag.adapters.ingestion.description_summarizer import (
    SUMMARIZED_FROM_ATTRIBUTE,
)
from unified_kg_rag.adapters.ingestion.gleaner import GleaningStats
from unified_kg_rag.adapters.ingestion.graph_extractor import ExtractionStats
from unified_kg_rag.adapters.ingestion.translator import TranslationStats
from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.adapters.storage.opensearch_indexer import OpenSearchIndexer
from unified_kg_rag.application.cli import preflight, run_ingestion_pipeline
from unified_kg_rag.application.ingestion import incremental
from unified_kg_rag.application.ingestion import pipeline as pipeline_module
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.extraction_store import with_record_keys
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.storage import stored_items
from unified_kg_rag.application.storage.indexing_manager import (
    stripped_descriptions,
    unit_records,
)
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
from unified_kg_rag.domain.ingestion.derivation import Derived
from unified_kg_rag.domain.ingestion.document_identity import (
    RELATIVE_PATH_KEY,
    compute_doc_id,
)
from unified_kg_rag.domain.ingestion.seed_graph import (
    SEED_RECORD_FINGERPRINT,
    SeedConversion,
    convert_seed,
    is_seed_path,
    parse_seed_unit,
    seed_text_units,
)
from unified_kg_rag.domain.models import (
    Community,
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
    DocStatusRegistryError,
    PipelineExecutionError,
    PipelineStageError,
    SeedGraphError,
)
from unified_kg_rag.shared.utils import compute_hash

SEED_PATH = "seed_graphs/procure/0000-of-0001.jsonl"


def _conversion(shards: int = 1, items: int = 50) -> SeedConversion:
    return convert_seed(
        PROCURE,
        seed_settings(shards=shards, max_items_per_unit=items),
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
        lambda: store, "default", "/corpus", confirm
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
    # A seed-only scope (its seed records) beside another scope's documents.
    store = FakeDocStatusStore()
    store.put(_record(SEED_PATH, "default|/corpus"))
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


def _respelled(config: Config) -> Config:
    config.processing.document_parsing.source_scope = "s3://bkt/corpus"
    return config


def _seeds_alone_in(
    store: FakeDocStatusStore, source_scope: str, reset: bool, confirm: Any = None
) -> list[Document]:
    """``_seeds_alone`` of a run whose source scope is ``source_scope``."""
    config = Config()
    config.indexing.reset = reset
    config.processing.document_parsing.source_scope = source_scope
    documents, _ = incremental.filter_to_delta(
        list(_conversion().documents()),
        _context(),
        config,
        lambda: store,
        Path("/corpus"),
        confirm,
    )
    return documents


@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("confirm", [None, lambda q: True])
def test_seeds_alone_never_delete_a_respelled_scopes_documents(
    mocker, reset: bool, confirm: Any
) -> None:
    # The scope was indexed as s3://bkt/corpus/; this run spells it without
    # the slash (the same S3 folder): its documents are the run's own.
    store = FakeDocStatusStore()
    store.put(_record("a.txt", "default|s3://bkt/corpus/"))
    store.put(_record(SEED_PATH, "default|s3://bkt/corpus/"))
    writes = mocker.spy(store, "put_many")
    with pytest.raises(PipelineStageError) as error:
        _seeds_alone_in(store, "s3://bkt/corpus", reset, confirm)
    assert "holds 1 documents of source scope" in str(error.value)
    assert "(a.txt)" in str(error.value)
    writes.assert_not_called()


@pytest.mark.parametrize("confirm", [None, lambda q: True])
def test_a_reset_from_seeds_alone_of_a_scope_without_records_is_refused(
    mocker, confirm: Any
) -> None:
    # A new or mistyped scope: no answer makes the namespace's documents
    # go for the seeds alone.
    store = FakeDocStatusStore()
    store.put(_record("a.txt", "default|s3://bkt/corpus/"))
    store.put(_record("b.txt", "default|documents"))
    writes = mocker.spy(store, "put_many")
    asked: list[str] = []
    with pytest.raises(PipelineStageError) as error:
        _seeds_alone_in(
            store,
            "s3://bkt/corpus2",
            True,
            confirm and (lambda q: asked.append(q) or True),
        )
    message = str(error.value)
    assert "holds no record of source scope 's3://bkt/corpus2'" in message
    assert "documents of source scope(s) documents, s3://bkt/corpus/" in message
    assert "without indexing.reset first" in message
    assert asked == []
    writes.assert_not_called()
    # Its incremental run is a new seed-only scope beside them.
    assert len(_seeds_alone_in(store, "s3://bkt/corpus2", False)) == 1


def test_a_respelled_seed_only_scope_reset_asks_as_its_own() -> None:
    # Its seed records under another spelling are its own: asked about the
    # documents only, not about "another scope's" seed.
    store = FakeDocStatusStore()
    store.put(_record(SEED_PATH, "default|s3://bkt/seeds/"))
    store.put(_record("a.txt", "default|documents"))
    asked: list[str] = []
    _seeds_alone_in(store, "s3://bkt/seeds", True, lambda q: asked.append(q) or True)
    assert asked == [
        "The reset of index namespace 'default' rebuilds it from this run's seed "
        "graphs alone (no source document reached the run), dropping the "
        "documents of source scope(s) documents; continue?"
    ]


def test_a_reset_of_a_respelled_scope_does_not_ask_about_its_own_seed() -> None:
    store = FakeDocStatusStore()
    store.put(_record(SEED_PATH, "default|s3://bkt/corpus/"))
    store.put(_record("a.txt", "default|s3://bkt/corpus/"))
    config = _respelled(Config())
    config.indexing.reset = True
    asked: list[str] = []
    documents, _ = incremental.filter_to_delta(
        [_document("a.txt", "Acme."), *_conversion().documents()],
        _context(),
        config,
        lambda: store,
        Path("/corpus"),
        lambda q: asked.append(q) or True,
    )
    assert len(documents) == 2
    assert asked == []


def test_a_seed_only_scope_run_reads_its_own_records_not_the_namespace(
    mocker,
) -> None:
    store = FakeDocStatusStore()
    for record in (
        _record(SEED_PATH, "default|/corpus"),
        _record("a.txt", "default|documents"),
    ):
        store.put(record)
    listed = mocker.spy(store, "list_namespace")
    _seeds_alone(store)
    _seeds_alone(store)
    listed.assert_not_called()
    # A document of the scope is still found, from the delta's ids.
    store.put(_record("b.txt", "default|/corpus"))
    with pytest.raises(PipelineStageError, match=r"\(b\.txt\)"):
        _seeds_alone(store)
    listed.assert_not_called()


def test_an_unreadable_registry_fails_the_seed_guard() -> None:
    # A listing that fails is no empty namespace: the guard stops the run
    # rather than read it as holding nothing.
    class Unreadable:
        def list_namespace(self, namespace: str) -> list[DocStatusRecord]:
            raise RuntimeError("throttled")

    with pytest.raises(DocStatusRegistryError, match="throttled") as error:
        incremental._namespace_records(
            lambda: Unreadable(), "default", "the seed check"  # type: ignore[arg-type,return-value]
        )
    assert "could not read namespace 'default' for the seed check" in str(error.value)


def test_a_seed_only_run_counts_a_file_that_failed_to_load() -> None:
    # Parsing off: the loader failed to read the scope's only document; it
    # is not deleted, and it is no seed-only scope's either.
    store = FakeDocStatusStore()
    store.put(_record("a.txt", "default|/corpus"))
    context = _context(failed_source_files=["/corpus/a.txt"])
    with pytest.raises(PipelineStageError, match=r"\(a\.txt\)"):
        incremental.filter_to_delta(
            list(_conversion().documents()),
            context,
            Config(),
            lambda: store,
            Path("/corpus"),
        )


def test_a_reset_reads_its_quarantined_documents_from_its_listing(mocker) -> None:
    store = FakeDocStatusStore()
    (document,) = _identified([_document("a.txt", "Acme.")])
    (doc_id, content_hash), *_ = fingerprint_documents([document]).items()
    store.put(
        _record(
            "a.txt",
            "default|/corpus",
            doc_id=doc_id,
            content_hash=content_hash,
            status=DocStatus.QUARANTINED,
            attempt_count=3,
        )
    )
    config = Config()
    config.indexing.reset = True
    read = mocker.spy(store, "get_many")
    documents, _ = incremental.filter_to_delta(
        [document], _context(), config, lambda: store, Path("/corpus")
    )
    assert documents == []
    read.assert_not_called()


# -- Seeds alone with the registry off ---------------------------------------------


def _registry_off_stage(
    mocker,
    graph: Any = False,
    vector: Any = False,
    confirm: Any = None,
    units: Any = False,
) -> tuple[Any, Any]:
    """A loading stage with the registry off whose stores answer
    ``holds_any_item`` with ``graph`` and ``vector``, and the vector store
    ``holds_any_text_unit`` with ``units`` (an exception: raised)."""
    stage: Any = ps.DocumentLoadingStage.__new__(ps.DocumentLoadingStage)
    stage.config = Config()
    stage.confirm = confirm
    providers = mocker.MagicMock()
    for probe, held in (
        (providers.build_graph_indexer.return_value.holds_any_item, graph),
        (providers.build_vector_indexer.return_value.holds_any_item, vector),
        (providers.build_vector_indexer.return_value.holds_any_text_unit, units),
    ):
        if isinstance(held, Exception):
            probe.side_effect = held
        else:
            probe.return_value = held
    stage.providers = providers
    return stage, providers


def test_registry_off_seeds_alone_build_an_empty_index_unasked(mocker) -> None:
    # A first seed-only build: the index holds nothing to lose.
    stage, providers = _registry_off_stage(mocker)
    stage._confirm_seeds_alone()
    for build in (providers.build_graph_indexer, providers.build_vector_indexer):
        build.return_value.holds_any_item.assert_called_once_with("default")
        build.return_value.close.assert_called_once()
    vector = providers.build_vector_indexer.return_value
    vector.holds_any_text_unit.assert_called_once_with("default")


@pytest.mark.parametrize(
    ("graph", "vector", "units"),
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (None, False, False),
        (False, False, None),
        (False, RuntimeError("down"), False),
    ],
    ids=[
        "graph-holds",
        "vector-holds",
        "text-units-only",
        "cannot-tell",
        "units-cannot-tell",
        "read-fails",
    ],
)
def test_registry_off_seeds_alone_over_an_index_ask_first(
    mocker, graph: Any, vector: Any, units: Any
) -> None:
    question = "rebuild the index 'default' from the seed graphs alone"
    stage, providers = _registry_off_stage(mocker, graph, vector, units=units)
    with pytest.raises(ConfirmationDeclinedError, match="--confirm-reset") as error:
        stage._confirm_seeds_alone()
    assert question in str(error.value)
    assert "incremental indexing is off" in str(error.value)
    stage.confirm = lambda q: False
    with pytest.raises(ConfirmationDeclinedError, match="Build cancelled"):
        stage._confirm_seeds_alone()
    asked: list[str] = []
    stage.confirm = lambda q: asked.append(q) or True
    stage._confirm_seeds_alone()
    assert len(asked) == 1 and question in asked[0]
    for build in (providers.build_graph_indexer, providers.build_vector_indexer):
        assert build.return_value.close.called or build.call_count == 0


def test_registry_off_confirmation_names_the_namespace(mocker) -> None:
    stage, _ = _registry_off_stage(mocker, graph=True)
    stage.config.processing.document_parsing.index_value = "tenant"
    stage.config.indexing.additional_suffix = "v2"
    asked: list[str] = []
    stage.confirm = lambda q: asked.append(q) or True
    stage._confirm_seeds_alone()
    stage.providers.build_graph_indexer.return_value.holds_any_item.assert_called_with(
        "tenant"
    )
    assert "rebuild the index 'tenant-v2' from the seed graphs alone" in asked[0]


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


def _runner(mocker, config: Config, **args: Any) -> Any:
    """An ingestion runner with ``args``, its pipeline mocked."""
    runner: Any = run_ingestion_pipeline.IngestionPipelineRunner.__new__(
        run_ingestion_pipeline.IngestionPipelineRunner
    )
    runner.args = SimpleNamespace(
        verify_metadata=True, source_directory=None, output_directory=None, **args
    )
    runner.config = config
    for name in (
        "_display_run_info",
        "_require_indexing_endpoints",
        "_handle_metadata_operations",
        "_build_metrics_sink",
    ):
        mocker.patch.object(runner, name, create=True)
    mocker.patch.object(runner, "_create_pipeline_config")
    mocker.patch.object(run_ingestion_pipeline, "display_ascii_art")
    return runner


@pytest.mark.parametrize(("yes", "confirm_reset"), [(True, False), (False, True)])
def test_run_ingestion_answers_the_reset_with_its_confirm(
    mocker, yes: bool, confirm_reset: bool
) -> None:
    built = mocker.patch.object(run_ingestion_pipeline, "DataIngestionPipeline")
    mocker.patch.object(run_ingestion_pipeline.sys.stdin, "isatty", return_value=False)
    runner = _runner(mocker, Config(), yes=yes, confirm_reset=confirm_reset)
    mocker.patch.object(
        run_ingestion_pipeline, "missing_extras_error", return_value=None
    )
    runner.run()
    confirm = built.call_args.kwargs["confirm"]
    assert confirm("Drop it?") is True


def test_run_ingestion_preflights_the_seed_sources_extras(
    mocker, capsys: pytest.CaptureFixture[str]
) -> None:
    # Only run-ingestion loads the seeds: an rdf seed without the ontology
    # extra fails before the pipeline is built.
    built = mocker.patch.object(run_ingestion_pipeline, "DataIngestionPipeline")
    mocker.patch.object(preflight, "_installed", side_effect=lambda m: m != "rdflib")
    config = Config()
    config.processing.seed_graphs = [
        SeedGraphConfig(name="procure", kind="rdf", path="procure.ttl")
    ]
    runner = _runner(mocker, config, yes=False, confirm_reset=False)
    with pytest.raises(SystemExit) as exc:
        runner.run()
    assert exc.value.code == 2
    assert "processing.seed_graphs 'procure' kind 'rdf'" in capsys.readouterr().err
    built.assert_not_called()


def test_an_unanswerable_reset_question_names_its_flag(
    mocker, capsys: pytest.CaptureFixture[str]
) -> None:
    mocker.patch.object(run_ingestion_pipeline.sys.stdin, "isatty", return_value=False)
    runner = _runner(mocker, Config(), yes=False, confirm_reset=False)
    with pytest.raises(ConfirmationDeclinedError):
        runner._confirm_reset("Drop it?")
    printed = " ".join(capsys.readouterr().out.split())
    assert "Re-run with --confirm-reset (or UKR_CONFIRM_RESET=1) to proceed" in printed


@pytest.mark.parametrize(
    ("argv", "env", "expected"),
    [
        ([], None, False),
        (["--confirm-reset"], None, True),
        ([], "1", True),
        ([], " TRUE ", True),
        ([], "yes", True),
        ([], "0", False),
        ([], "false", False),
        ([], " No ", False),
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

    def load(units, fingerprint, parse, legacy_fingerprints=()):
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
            # u1 states an entity: a unit that states none is not asked.
            TextUnitExtraction(
                text_unit_id="u1",
                entities=[Entity(id="e1", name="Vendor", text_unit_ids=["u1"])],
            ),
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


SEED_UNIT = "sg-00000000-0000-0000-0000-000000000001"


def _evidence(
    mocker, graph: Any, *, reset: bool = False
) -> tuple[Any, PipelineContext, Any]:
    """A community stage of an incremental run (``reset``: a reset's) whose
    graph store is ``graph``, and the run's context: Vendor (default suffix)
    and Harbor (suffix ``other``), each stated by the seed alone."""
    config = Config()
    config.indexing.reset = reset
    # No detector (it would build the report model): only the stored reads.
    mocker.patch.object(ps, "CommunityDetector")
    providers = mocker.MagicMock()
    providers.build_graph_indexer.return_value = graph
    stage = ps.CommunityDetectionStage(
        config=config, providers=providers, doc_status=FakeDocStatusStore()
    )
    context = _context(incremental_delta=DocumentDelta(new=["seed-doc"]))
    context.resolved_entities = [
        Entity(id="vendor", name="Vendor", text_unit_ids=[SEED_UNIT]),
        Entity(
            id="harbor",
            name="Harbor",
            text_unit_ids=[SEED_UNIT],
            attributes={"index": "other"},
        ),
    ]
    return stage, context, providers


def _communities() -> list[Community]:
    return [
        Community(
            id=f"c-{member}",
            name=member,
            level="0",
            parent="",
            children=[],
            entity_ids=[member],
            text_unit_ids=[SEED_UNIT],
        )
        for member in ("vendor", "harbor")
    ]


def _stored_graph() -> FakeGraphStore:
    graph = FakeGraphStore()
    graph.upsert_entities(
        [
            Entity(
                id="vendor",
                name="Vendor",
                description="Vendor sells.",
                text_unit_ids=["u1"],
            ),
            Entity(
                id="harbor",
                name="Harbor",
                description="Harbor docks.",
                text_unit_ids=["u2"],
                attributes={"index": "other"},
            ),
        ]
    )
    return graph


def _read(stage: Any, context: PipelineContext) -> tuple[Any, dict[str, Any]]:
    """The evidenced communities and the backed members' descriptions."""
    stored = stage._evidenced_by_stored_members(context, _communities())
    return stored.evidenced, {
        entity_id: entity.description for entity_id, entity in stored.backed.items()
    }


def test_stored_members_are_read_in_their_own_suffix(mocker) -> None:
    stage, context, _ = _evidence(mocker, _stored_graph())
    assert _read(stage, context) == (
        {"c-vendor", "c-harbor"},
        {"vendor": "Vendor sells.", "harbor": "Harbor docks."},
    )
    stored = stage._evidenced_by_stored_members(context, _communities())
    # Tagged with the suffix they were read from, as the indexing groups
    # the summaries made of them.
    assert ps.BaseIndexer.get_suffix(stored.backed["harbor"]) == "other"


def test_only_the_backed_members_community_is_evidenced(mocker) -> None:
    # Harbor is stored from the seed alone: its community stays seed-only.
    graph = FakeGraphStore()
    graph.upsert_entities(
        [
            Entity(id="vendor", name="Vendor", text_unit_ids=["u1"]),
            Entity(
                id="harbor",
                name="Harbor",
                text_unit_ids=[SEED_UNIT],
                attributes={"index": "other"},
            ),
        ]
    )
    stage, context, _ = _evidence(mocker, graph)
    evidenced, _ = _read(stage, context)
    assert evidenced == {"c-vendor"}


def test_a_deleted_documents_units_evidence_nothing(mocker) -> None:
    # u1 is the unit of a document the run deletes: a full build no longer
    # holds it, so Vendor's stored copy evidences nothing.
    stage, context, _ = _evidence(mocker, _stored_graph())
    stage._doc_status.put(
        DocStatusRecord(
            doc_id="gone",
            content_hash="h",
            status=DocStatus.PROCESSED,
            text_unit_ids=["u1"],
        )
    )
    context.incremental_delta = DocumentDelta(new=["seed-doc"], deleted=["gone"])
    assert _read(stage, context) == ({"c-harbor"}, {"harbor": "Harbor docks."})


def test_the_stored_members_graph_store_is_closed(mocker) -> None:
    graph = _stored_graph()
    close = mocker.spy(graph, "close")
    stage, context, providers = _evidence(mocker, graph)
    stage._evidenced_by_stored_members(context, _communities())
    assert close.call_count == 1
    # No stored member cites a unit the run replaces: no record is read.
    providers.build_vector_indexer.assert_not_called()


@pytest.mark.parametrize("recorded", [True, False])
def test_a_replaced_units_lines_leave_the_stored_description(
    mocker, recorded: bool
) -> None:
    # Vendor's stored copy cites the document's u1 and the seed unit this
    # run replaces (an edited seed): its description is derived again from
    # u1's record, as the indexing's strip does. Without u1's record it is
    # kept as stored, as the strip keeps it.
    graph = _stored_graph()
    graph.upsert_entities(
        [
            Entity(
                id="vendor",
                name="Vendor",
                description="Vendor sells.\nSeed v1 says vendor.",
                text_unit_ids=["u1", SEED_UNIT],
            )
        ]
    )
    vector = FakeVectorStore()
    close = mocker.spy(vector, "close")
    vector.upsert_text_units(
        [TextUnit(id="u1", text="", attributes={"extraction_records": {"graph": "k1"}})]
    )
    if recorded:
        record = TextUnitExtraction(
            text_unit_id="u1",
            entities=[Entity(id="vendor", name="Vendor", description="Vendor sells.")],
        )
        vector.write_extraction_records({"k1": record_payload(record)})
    stage, context, providers = _evidence(mocker, graph)
    providers.build_vector_indexer.return_value = vector
    context.text_units = [TextUnit(id=SEED_UNIT, text="seed v2")]
    evidenced, descriptions = _read(stage, context)
    assert evidenced == {"c-vendor", "c-harbor"}
    assert descriptions == {
        "vendor": (
            "Vendor sells." if recorded else "Vendor sells.\nSeed v1 says vendor."
        ),
        "harbor": "Harbor docks.",
    }
    assert close.call_count == 1


def _stripped_vendor(mocker, records: dict[str, Any]) -> tuple[Any, PipelineContext]:
    """The community stage of a seed edit: Vendor's stored copy cites the
    document units of ``records`` (unit -> record or raw payload) and the
    seed unit the run replaces; the run states Vendor with the seed's new
    line."""
    graph = _stored_graph()
    graph.upsert_entities(
        [
            Entity(
                id="vendor-old",
                name="Vendor",
                description="Vendor sells.\nVendor buys.\nSeed v1 says vendor.",
                text_unit_ids=[*records, SEED_UNIT],
            )
        ]
    )
    vector = FakeVectorStore()
    vector.upsert_text_units(
        [
            TextUnit(id=u, text="", attributes={"extraction_records": {"graph": u}})
            for u in records
        ]
    )
    vector.write_extraction_records(
        {
            u: record_payload(r) if isinstance(r, TextUnitExtraction) else r
            for u, r in records.items()
        }
    )
    stage, context, providers = _evidence(mocker, graph)
    providers.build_vector_indexer.return_value = vector
    context.text_units = [TextUnit(id=SEED_UNIT, text="seed v2")]
    context.resolved_entities[0] = Entity(
        id="vendor-old",
        name="Vendor",
        type="ORGANIZATION",
        description="Seed v2 says vendor.",
        text_unit_ids=[SEED_UNIT],
    )
    return stage, context


def _vendor_record(unit: str, entity_id: str, description: str) -> TextUnitExtraction:
    return TextUnitExtraction(
        text_unit_id=unit,
        entities=[Entity(id=entity_id, name="Vendor", description=description)],
    )


def _communities_of(*members: str) -> list[Community]:
    return [
        Community(
            id=f"c-{member}",
            name=member,
            level="0",
            parent="",
            children=[],
            entity_ids=[member],
            text_unit_ids=[SEED_UNIT],
        )
        for member in members
    ]


def test_a_stripped_member_follows_its_rekeyed_id(mocker) -> None:
    # The records name Vendor under another id (ids follow the current
    # name): the derivation re-keys the stored copy, and its description is
    # the re-keyed entity's.
    stage, context = _stripped_vendor(
        mocker,
        {
            "u1": _vendor_record("u1", "vendor", "Vendor sells."),
            "u3": _vendor_record("u3", "vendor", "Vendor buys."),
        },
    )
    stored = stage._evidenced_by_stored_members(context, _communities_of("vendor-old"))
    assert stored.backed["vendor-old"].description == "Vendor sells.\nVendor buys."


def test_a_stripped_member_no_record_states_keeps_its_description(mocker) -> None:
    # No record of its other units states Vendor: the derivation keeps it
    # as it was, and so does its description.
    other = TextUnitExtraction(
        text_unit_id="u1", entities=[Entity(id="buyer", name="Buyer")]
    )
    stage, context = _stripped_vendor(mocker, {"u1": other})
    stored = stage._evidenced_by_stored_members(context, _communities_of("vendor-old"))
    assert stored.backed["vendor-old"].description == (
        "Vendor sells.\nVendor buys.\nSeed v1 says vendor."
    )


def test_a_stripped_member_the_derivation_lacks_keeps_its_description(
    mocker,
) -> None:
    # Defensive: a derivation without an entity for the stripped one (no
    # derivation drops an entity today) leaves its stored description, not
    # an empty one.
    vector = FakeVectorStore()
    vector.upsert_text_units(
        [TextUnit(id="u1", text="", attributes={"extraction_records": {"graph": "u1"}})]
    )
    mocker.patch.object(stored_items, "derive", return_value=Derived())
    vendor = Entity(
        id="vendor",
        name="Vendor",
        description="Vendor sells.\nSeed v1 says vendor.",
        text_unit_ids=["u1", SEED_UNIT],
    )
    assert stripped_descriptions(
        Config(), vector, "default", [vendor], {SEED_UNIT}
    ) == {"vendor": "Vendor sells.\nSeed v1 says vendor."}


@pytest.mark.parametrize(
    "payload",
    [
        {"entities": [{"name": "Vendor"}]},  # no id: a ValueError
        {"entities": [], "rejected": 5},  # not a mapping: a TypeError
    ],
)
def test_an_unreadable_record_is_ignored_by_unit_records(
    payload: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    vector = FakeVectorStore()
    vector.upsert_text_units(
        [
            TextUnit(id=u, text="", attributes={"extraction_records": {"graph": u}})
            for u in ("u1", "u2")
        ]
    )
    good = _vendor_record("u2", "vendor", "Vendor sells.")
    vector.write_extraction_records({"u1": payload, "u2": record_payload(good)})
    with caplog.at_level(logging.WARNING):
        records, claims = unit_records(vector, "default", ["u1", "u2"])
    assert set(records) == {"u2"} and claims == {}
    assert "Ignoring an unreadable extraction record of text unit" in caplog.text


class _FakeSummarizer:
    """Summarizes each entity it is given to ``Summary of <name>.``, marked
    as ``DescriptionSummarizer`` marks it."""

    def __init__(self) -> None:
        self.asked: list[list[str]] = []

    def summarize_entities(self, entities: list[Entity]) -> list[Entity]:
        self.asked.append([e.description or "" for e in entities])
        for entity in entities:
            entity.attributes = {
                **(entity.attributes or {}),
                SUMMARIZED_FROM_ATTRIBUTE: compute_hash(entity.description or ""),
            }
            entity.description = f"Summary of {entity.name}."
        return entities


def test_a_derived_member_description_is_summarized_as_the_indexing_does(
    mocker,
) -> None:
    # Over the threshold (4 tokens), Vendor's description derived without
    # the replaced unit is summarized, and so is that summary followed by
    # the seed's line, as the indexing summarizes its strip and then its
    # merge. Both summaries are kept for the indexing to reuse, marked with
    # what they were made from; Harbor's fits and is not summarized.
    stage, context = _stripped_vendor(
        mocker,
        {
            "u1": _vendor_record("u1", "vendor-old", "Vendor sells."),
            "u3": _vendor_record("u3", "vendor-old", "Vendor buys."),
        },
    )
    summarization = stage.config.processing.graph_extraction.description_summarization
    summarization.force_summary_threshold_tokens = 4
    summarizer = _FakeSummarizer()
    stage._description_summarizer = summarizer
    stored = stage._evidenced_by_stored_members(
        context, _communities_of("vendor-old", "harbor")
    )
    assert stage._report_descriptions(context, stored) == {
        "vendor-old": "Summary of Vendor.",
        "harbor": "Harbor docks.",
    }
    derived = "Vendor sells.\nVendor buys."
    merged = "Summary of Vendor.\nSeed v2 says vendor."
    assert summarizer.asked == [[derived], [merged]]
    assert [
        (e.id, e.description, (e.attributes or {})[SUMMARIZED_FROM_ATTRIBUTE])
        for e in context.member_summaries
    ] == [
        ("vendor-old", "Summary of Vendor.", compute_hash(derived)),
        ("vendor-old", "Summary of Vendor.", compute_hash(merged)),
    ]


def test_only_a_stripped_member_description_is_summarized_before_its_merge(
    mocker,
) -> None:
    # Every description is over the threshold (1 token). Harbor cites no
    # unit the run replaces: its stored description is what the indexing
    # merges this run's lines into, so it is not summarized on its own
    # first; Vendor's, derived again without the replaced unit, is.
    stage, context = _stripped_vendor(
        mocker,
        {
            "u1": _vendor_record("u1", "vendor-old", "Vendor sells."),
            "u3": _vendor_record("u3", "vendor-old", "Vendor buys."),
        },
    )
    summarization = stage.config.processing.graph_extraction.description_summarization
    summarization.force_summary_threshold_tokens = 1
    summarizer = _FakeSummarizer()
    stage._description_summarizer = summarizer
    stored = stage._evidenced_by_stored_members(
        context, _communities_of("vendor-old", "harbor")
    )
    assert summarizer.asked == [["Vendor sells.\nVendor buys."]]
    assert stored.backed["harbor"].description == "Harbor docks."


def test_a_failed_member_summary_is_not_kept_for_the_indexing(mocker) -> None:
    # The model gave no summary: the description stays as derived, and
    # nothing is offered (a stored copy's mark would name another text).
    stage, context = _stripped_vendor(
        mocker,
        {
            "u1": _vendor_record("u1", "vendor-old", "Vendor sells."),
            "u3": _vendor_record("u3", "vendor-old", "Vendor buys."),
        },
    )
    summarization = stage.config.processing.graph_extraction.description_summarization
    summarization.force_summary_threshold_tokens = 4
    summarizer = mocker.MagicMock()
    summarizer.summarize_entities.side_effect = lambda entities: entities
    stage._description_summarizer = summarizer
    stored = stage._evidenced_by_stored_members(context, _communities_of("vendor-old"))
    assert stage._report_descriptions(context, stored) == {
        "vendor-old": "Vendor sells.\nVendor buys.\nSeed v2 says vendor."
    }
    assert summarizer.summarize_entities.call_count == 2
    assert context.member_summaries == []


def test_without_summarization_a_derived_description_is_read_whole(mocker) -> None:
    # As a full build without summarization reads it: no summarizer is
    # built, nothing is kept for the indexing.
    stage, context = _stripped_vendor(
        mocker,
        {
            "u1": _vendor_record("u1", "vendor-old", "Vendor sells."),
            "u3": _vendor_record("u3", "vendor-old", "Vendor buys."),
        },
    )
    summarization = stage.config.processing.graph_extraction.description_summarization
    summarization.force_summary_threshold_tokens = 4
    summarization.enabled = False
    stored = stage._evidenced_by_stored_members(context, _communities_of("vendor-old"))
    assert stage._report_descriptions(context, stored) == {
        "vendor-old": "Vendor sells.\nVendor buys.\nSeed v2 says vendor."
    }
    assert stage._description_summarizer is None
    assert context.member_summaries == []


def test_an_unattributed_member_reads_its_stored_type_and_description(
    mocker,
) -> None:
    # Harbor is stated by the run with neither a type nor a description
    # (the endpoint of a seed edge): its stored copy fills both, though it
    # evidences nothing (seed-only). Vendor, stated with a description, is
    # not filled.
    graph = FakeGraphStore()
    graph.upsert_entities(
        [
            Entity(id="vendor", name="Vendor", type="ORG", text_unit_ids=["u1"]),
            Entity(
                id="harbor",
                name="Harbor",
                type="LOCATION",
                description="Harbor docks.",
                text_unit_ids=[SEED_UNIT],
                attributes={"index": "other"},
            ),
        ]
    )
    stage, context, _ = _evidence(mocker, graph)
    context.resolved_entities[0].description = "Seed says vendor."
    stored = stage._evidenced_by_stored_members(context, _communities())
    assert stored.evidenced == {"c-vendor"}
    assert set(stored.unattributed) == {"harbor"}
    assert stored.unattributed["harbor"].type == "LOCATION"
    assert stage._report_descriptions(context, stored) == {
        "vendor": "Seed says vendor.",
        "harbor": "Harbor docks.",
    }


def test_a_reset_reads_no_stored_member(mocker) -> None:
    # A reset rebuilds from this corpus alone: what the stores held before
    # evidences nothing.
    stage, context, providers = _evidence(mocker, _stored_graph(), reset=True)
    assert _read(stage, context) == (set(), {})
    providers.build_graph_indexer.assert_not_called()


def test_an_unreadable_member_leaves_its_community_without_a_report(
    mocker, caplog: pytest.LogCaptureFixture
) -> None:
    graph = _stored_graph()
    mocker.patch.object(graph, "read_entities", side_effect=RuntimeError("down"))
    stage, context, _ = _evidence(mocker, graph)
    with caplog.at_level(logging.WARNING):
        # None: unread, not "evidenced by none" (the communities count as
        # failed, see test_unread_members_count_their_communities_failed).
        assert _read(stage, context) == (None, {})
    assert "Could not read the stored members of 2 communities" in caplog.text
