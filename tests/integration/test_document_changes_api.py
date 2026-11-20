# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``DataIngestionPipeline.insert_documents`` / ``delete_documents`` on the
embedded stores (no container, no AWS): documents given from memory go
through the same incremental path as a corpus run, and the scope's other
documents are kept instead of deleted."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.doc_status_records import all_records
from tests.fixtures.fakes.model_servers import FakeOllama
from tests.integration import test_embedded_e2e as embedded
from tests.integration.test_embedded_e2e import (
    ANSWER,
    DIM,
    FakeCrossEncoder,
    _config,
    _entity_names,
    _errors,
    _providers,
    _respond,
    _stored,
    _stored_texts,
)
from unified_kg_rag import DocumentNotFoundError, InputDocument
from unified_kg_rag.application.ingestion import pipeline_stages as ps
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import (
    Config,
    DocStatus,
    Document,
    Methodology,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)
from unified_kg_rag.shared.exceptions import ConfigurationError

pytestmark = [pytest.mark.integration, pytest.mark.usefixtures("no_aws")]

# The fixture that patches every AWS client to fail.
no_aws = embedded.no_aws


@pytest.fixture
def setup(tmp_path: Path) -> Iterator[tuple[Config, Any, DataIngestionPipeline]]:
    config = _config(tmp_path / "stores", tmp_path / "viz")
    providers = _providers(
        config, FakeOllama(_respond, ANSWER, DIM), FakeCrossEncoder("rerank")
    )
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(
            stages_enabled=dict.fromkeys(PipelineStageType, True),
            local_directory=tmp_path / "cache",
        ),
        providers=providers,
    )
    for stage in pipeline.stages:
        if isinstance(stage, ps.GraphResolutionStage):
            stage.resolver.entity_resolver.use_process_pool = False
            stage.resolver.relationship_resolver.use_process_pool = False
    yield config, providers, pipeline
    pipeline.close()


def _records(providers: Any) -> dict[str, DocStatus]:
    return {r.file_path: r.status for r in all_records(providers.build_doc_status())}


def _ok(context: PipelineContext) -> PipelineContext:
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    return context


def _unit_filters(config: Config, providers: Any) -> dict[str, Any]:
    prefix = config.indexing.opensearch.text_units_index_prefix
    return {
        hit["_source"]["text"]
        .strip(): hit["_source"]
        .get("attributes", {})
        .get("filters")
        for hit in _stored(config, providers, prefix)
    }


def test_documents_from_memory_are_added_updated_and_deleted(
    setup, caplog: pytest.LogCaptureFixture
) -> None:
    config, providers, pipeline = setup

    # Added: Korean text, filters, a nested id.
    first = _ok(
        pipeline.insert_documents(
            [
                InputDocument(
                    id="contracts/supply.txt",
                    text="Vendor supplies widgets to Buyer. 공급자는 납품한다.",
                    filters={"category": "contract", "year": 2024},
                ),
                {"id": "ticket-1", "text": "Depot stores widgets for Vendor."},
            ]
        )
    )
    assert _records(providers) == {
        "contracts/supply.txt": DocStatus.PROCESSED,
        "ticket-1": DocStatus.PROCESSED,
    }
    assert _entity_names(providers) == {"Vendor", "Buyer", "Depot"}
    filters = _unit_filters(config, providers)
    # The given filters and the id's path filters (file, folder).
    assert filters["Vendor supplies widgets to Buyer. 공급자는 납품한다."] == {
        "category": "contract",
        "year": 2024,
        "file": "contracts/supply.txt",
        "folder": ["contracts"],
    }
    assert filters["Depot stores widgets for Vendor."] == {"file": "ticket-1"}
    assert set(first.incremental_delta.new) and not first.incremental_delta.kept

    # Another insert keeps what it does not name (a corpus run deletes it).
    second = _ok(
        pipeline.insert_documents(
            [InputDocument(id="ticket-2", text="Carrier ships widgets for Buyer.")]
        )
    )
    assert len(second.incremental_delta.kept) == 2
    assert second.incremental_delta.deleted == []
    assert set(_records(providers)) == {"contracts/supply.txt", "ticket-1", "ticket-2"}
    assert _entity_names(providers) == {"Vendor", "Buyer", "Depot", "Carrier"}

    # Updated: the same id with other text is changed, re-extracted.
    third = _ok(
        pipeline.insert_documents(
            [InputDocument(id="ticket-1", text="Depot stores gadgets for Buyer.")]
        )
    )
    assert len(third.incremental_delta.changed) == 1
    assert "Depot stores gadgets for Buyer." in _stored_texts(config, providers)
    assert "Depot stores widgets for Vendor." not in _stored_texts(config, providers)

    # Deleted by id: only what it alone stated goes. A deletion stages no
    # document, which is no empty corpus to warn of.
    with caplog.at_level("WARNING"):
        fourth = _ok(pipeline.delete_documents(["ticket-2"]))
    assert "No supported files found" not in caplog.text
    assert len(fourth.incremental_delta.deleted) == 1
    assert set(_records(providers)) == {"contracts/supply.txt", "ticket-1"}
    assert _entity_names(providers) == {"Vendor", "Buyer", "Depot"}
    assert "Carrier ships widgets for Buyer." not in _stored_texts(config, providers)

    # Unknown ids are refused before anything is written.
    with pytest.raises(DocumentNotFoundError, match="'ticket-9'"):
        pipeline.delete_documents(["ticket-9", "ticket-1"])
    assert set(_records(providers)) == {"contracts/supply.txt", "ticket-1"}
    # The pipeline's own settings are left as they were.
    assert config.processing.document_parsing.source_scope is None
    assert any(isinstance(s, ps.DocumentParsingStage) for s in pipeline.stages)


def test_scopes_keep_their_documents_apart(setup, tmp_path: Path) -> None:
    config, providers, pipeline = setup
    _ok(pipeline.insert_documents([{"id": "a", "text": "Vendor supplies Buyer."}]))
    _ok(
        pipeline.insert_documents(
            [{"id": "a", "text": "Depot stores widgets for Carrier."}],
            source_scope="crm",
        )
    )
    scopes = sorted(r.scope for r in all_records(providers.build_doc_status()))
    assert (
        len(scopes) == 2 and scopes[0].endswith("|api") and scopes[1].endswith("|crm")
    )
    # Deleting a from crm leaves the api scope's a.
    _ok(pipeline.delete_documents(["a"], source_scope="crm"))
    assert _stored_texts(config, providers) == {"Vendor supplies Buyer."}
    # A corpus directory still runs as before, in its own scope.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "notes.txt").write_text("Carrier works with Depot.")
    _ok(pipeline.run(corpus))
    assert sorted(_records(providers)) == ["a", "notes.txt"]


def test_the_registry_is_required(tmp_path: Path) -> None:
    config = _config(tmp_path / "stores", tmp_path / "viz")
    config.providers.doc_status.enabled = False
    providers = _providers(
        config, FakeOllama(_respond, ANSWER, DIM), FakeCrossEncoder("rerank")
    )
    pipeline = DataIngestionPipeline(
        config, PipelineConfig(local_directory=tmp_path / "cache"), providers=providers
    )
    try:
        for call in (
            lambda: pipeline.insert_documents([{"id": "a", "text": "Vendor."}]),
            lambda: pipeline.delete_documents(["a"]),
        ):
            with pytest.raises(ConfigurationError, match="doc-status registry"):
                call()
    finally:
        pipeline.close()


@pytest.mark.parametrize(
    ("given", "match"),
    [
        ([], "at least one document"),
        ([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}], "more than once"),
        ([{"id": "../escape", "text": "x"}], "relative path"),
        ([{"id": "/abs", "text": "x"}], "relative path"),
        ([{"id": "a", "text": "   "}], "text is empty"),
        ([{"id": "a", "text": "x", "filters": {"a.b": 1}}], "filter key"),
        ([{"id": "a", "text": "x", "filters": {"seed_graph": 1}}], "filter key"),
        ([{"id": "a", "text": "x", "metadata": {"filters": {}}}], "set by ingestion"),
    ],
)
def test_invalid_documents_are_refused_before_a_run(
    setup, given: list[dict[str, Any]], match: str
) -> None:
    _, providers, pipeline = setup
    with pytest.raises(ValueError, match=match):
        pipeline.insert_documents(given)
    assert _records(providers) == {}


def test_a_reset_cannot_run_document_changes(setup) -> None:
    config, _, pipeline = setup
    config.indexing.reset = True
    with pytest.raises(ConfigurationError, match="cannot reset"):
        pipeline.insert_documents([{"id": "a", "text": "Vendor."}])


def test_an_added_methodology_is_refused_with_advice_that_works(
    tmp_path: Path,
) -> None:
    # Regression: a methodology added to an API-fed namespace failed the
    # loading stage, after the run took the lease, advising a run of "the
    # whole corpus" the api scope does not have.
    asked: list[str] = []

    def respond(system: str, human: str) -> str:
        asked.append(system)
        return _respond(system, human)

    config = _config(tmp_path / "stores", tmp_path / "viz")
    config.indexing.methodologies = [Methodology.LIGHTRAG]
    providers = _providers(
        config, FakeOllama(respond, ANSWER, DIM), FakeCrossEncoder("rerank")
    )

    def pipeline_of() -> DataIngestionPipeline:
        pipeline = DataIngestionPipeline(
            config,
            PipelineConfig(
                stages_enabled=dict.fromkeys(PipelineStageType, True),
                local_directory=tmp_path / "cache",
            ),
            providers=providers,
        )
        for stage in pipeline.stages:
            if isinstance(stage, ps.GraphResolutionStage):
                stage.resolver.entity_resolver.use_process_pool = False
                stage.resolver.relationship_resolver.use_process_pool = False
        return pipeline

    def methodologies() -> list[str]:
        indexer = providers.build_vector_indexer()
        try:
            return list((indexer.read_build_record(None) or {})["methodologies"])
        finally:
            indexer.close()

    documents = {
        "contracts/supply.txt": "Vendor supplies widgets to Buyer.",
        "ticket-1.txt": "Depot stores widgets for Vendor.",
    }
    pipeline = pipeline_of()
    try:
        _ok(
            pipeline.insert_documents(
                [{"id": key, "text": text} for key, text in documents.items()]
            )
        )
    finally:
        pipeline.close()

    config.indexing.methodologies = [Methodology.LIGHTRAG, Methodology.GRAPHRAG]
    pipeline = pipeline_of()
    try:
        asked.clear()
        for call in (
            lambda: pipeline.insert_documents([{"id": "b", "text": "Carrier."}]),
            lambda: pipeline.delete_documents(["ticket-1.txt"]),
        ):
            with pytest.raises(ConfigurationError, match="does not hold") as refused:
                call()
            assert "indexing.methodologies: lightrag" in str(refused.value)
            assert "source_scope: 'api'" in str(refused.value)
        assert asked == []
        assert set(_records(providers)) == set(documents)
        assert not config.indexing.reset

        # The advice: a run over a directory of the namespace's documents,
        # each a file named by its id (ids the parser reads), in the api
        # scope, adds it from the stored extractions.
        corpus = tmp_path / "corpus"
        for key, text in documents.items():
            (corpus / key).parent.mkdir(parents=True, exist_ok=True)
            (corpus / key).write_text(text)
        config.processing.document_parsing.source_scope = "api"
        _ok(pipeline.run(corpus))
        assert not any("knowledge graph extraction expert" in s for s in asked)
        assert "graphrag" in methodologies()
        assert set(_records(providers)) == set(documents)
        # Inserts run again once the namespace holds what they build.
        config.processing.document_parsing.source_scope = None
        _ok(pipeline.insert_documents([{"id": "b", "text": "Carrier ships."}]))
        assert set(_records(providers)) == {*documents, "b"}
    finally:
        pipeline.close()


def test_fewer_methodologies_are_refused_beside_another_scope(
    tmp_path: Path,
) -> None:
    # Document changes with fewer methodologies than the namespace holds
    # would stop maintaining the dropped one for every scope's documents:
    # refused before any model call while another scope is held, allowed
    # while the api scope is the namespace's only one.
    asked: list[str] = []

    def respond(system: str, human: str) -> str:
        asked.append(system)
        return _respond(system, human)

    def pipeline_in(root: Path) -> tuple[Config, Any, DataIngestionPipeline]:
        config = _config(root / "stores", root / "viz")
        providers = _providers(
            config, FakeOllama(respond, ANSWER, DIM), FakeCrossEncoder("rerank")
        )
        pipeline = DataIngestionPipeline(
            config,
            PipelineConfig(
                stages_enabled=dict.fromkeys(PipelineStageType, True),
                local_directory=root / "cache",
            ),
            providers=providers,
        )
        for stage in pipeline.stages:
            if isinstance(stage, ps.GraphResolutionStage):
                stage.resolver.entity_resolver.use_process_pool = False
                stage.resolver.relationship_resolver.use_process_pool = False
        return config, providers, pipeline

    def record(providers: Any) -> dict[str, Any]:
        indexer = providers.build_vector_indexer()
        try:
            return dict(indexer.read_build_record(None) or {})
        finally:
            indexer.close()

    # Beside a directory corpus: refused.
    config, providers, pipeline = pipeline_in(tmp_path / "shared")
    try:
        _ok(pipeline.insert_documents([{"id": "a", "text": "Vendor supplies Buyer."}]))
        assert record(providers)["source_scopes"] == ["api"]
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "notes.txt").write_text("Bank lends money to Vendor.")
        _ok(pipeline.run(corpus))
        before = record(providers)
        assert before["source_scopes"] == sorted([str(corpus.resolve()), "api"])

        config.indexing.methodologies = [Methodology.LIGHTRAG]
        asked.clear()
        for call in (
            lambda: pipeline.insert_documents([{"id": "b", "text": "Carrier."}]),
            lambda: pipeline.delete_documents(["a"]),
        ):
            with pytest.raises(ConfigurationError, match="leaves out graphrag") as no:
                call()
            assert str(corpus.resolve()) in str(no.value)
        assert asked == []
        assert set(_records(providers)) == {"a", "notes.txt"}
        assert record(providers) == before
    finally:
        pipeline.close()

    # The api scope alone: narrowing is allowed.
    config, providers, pipeline = pipeline_in(tmp_path / "alone")
    try:
        _ok(pipeline.insert_documents([{"id": "a", "text": "Vendor supplies Buyer."}]))
        config.indexing.methodologies = [Methodology.LIGHTRAG]
        _ok(pipeline.insert_documents([{"id": "b", "text": "Carrier ships."}]))
        narrowed = record(providers)
        assert narrowed["methodologies"] == ["lightrag", "hipporag"]
        assert narrowed["source_scopes"] == ["api"]
    finally:
        pipeline.close()


def test_delete_documents_closes_a_registry_it_builds(
    setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A pipeline built without its registry builds one to check the ids:
    # it was left open.
    _, providers, pipeline = setup
    built = providers.build_doc_status()
    closed: list[bool] = []
    monkeypatch.setattr(built, "close", lambda: closed.append(True))
    monkeypatch.setattr(pipeline, "_doc_status", None)
    monkeypatch.setattr(providers, "build_doc_status", lambda: built)
    with pytest.raises(DocumentNotFoundError):
        pipeline.delete_documents(["never-inserted"])
    assert closed == [True]


class _Killed(BaseException):
    """A process death: no ``except Exception`` handler absorbs it."""


@pytest.mark.parametrize("after", [False, True], ids=["first-only", "both"])
def test_a_deletion_killed_after_its_first_record_delete_is_retried(
    setup, mocker, after: bool
) -> None:
    # The registry deletes the documents' records one by one, last. A run
    # killed after the first left the others: retried under its pipeline
    # id, the deletion refused the id already gone ("nothing was deleted"),
    # so it could not be finished.
    from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore

    config, providers, pipeline = setup
    _ok(
        pipeline.insert_documents(
            [
                {"id": "a", "text": "Vendor supplies widgets to Buyer."},
                {"id": "b", "text": "Depot stores widgets for Vendor."},
                {"id": "c", "text": "Carrier ships widgets for Buyer."},
            ]
        )
    )
    delete = SQLiteDocStatusStore.delete
    calls = {"n": 0}

    def dies_second(self, doc_id):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if calls["n"] == 2:
            if after:
                delete(self, doc_id)
            raise _Killed
        return delete(self, doc_id)

    mocker.patch.object(SQLiteDocStatusStore, "delete", dies_second)
    with pytest.raises(_Killed):
        pipeline.delete_documents(["a", "b"], pipeline_id="del-1")
    mocker.stopall()
    assert len(_records(providers)) == (1 if after else 2)

    _ok(pipeline.delete_documents(["a", "b"], pipeline_id="del-1"))

    assert set(_records(providers)) == {"c"}
    assert _entity_names(providers) == {"Carrier", "Buyer"}
    assert _stored_texts(config, providers) == {"Carrier ships widgets for Buyer."}
    # Another id is still refused: only those the interrupted run deleted
    # are taken as done.
    with pytest.raises(DocumentNotFoundError, match="'a'"):
        pipeline.delete_documents(["a", "c"], pipeline_id="del-2")


def test_a_deletion_names_the_documents_it_removes(
    setup, caplog: pytest.LogCaptureFixture
) -> None:
    # Regression (rv24): only counts were logged, then the registry record
    # that named the document was deleted: no audit of what was removed.
    import logging

    config, providers, pipeline = setup
    _ok(
        pipeline.insert_documents(
            [
                {"id": "ticket-1", "text": "Vendor supplies Buyer."},
                {"id": "ticket-2", "text": "Depot stores widgets for Carrier."},
            ]
        )
    )
    with caplog.at_level(logging.INFO):
        _ok(pipeline.delete_documents(["ticket-2"]))
    (audit,) = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.INFO and "changed documents" in r.getMessage()
    ]
    assert "of 1 deleted or changed documents from 'default': ticket-2" in audit
    assert "ticket-1" not in audit


_SUPPLY = "Vendor supplies widgets to Buyer under the yearly framework agreement."
_STORAGE = "Depot stores widgets for Carrier in the harbor warehouse every week."


def _texts(config: Config, providers: Any) -> str:
    """The stored text units' texts, in one string (a unit is a chunk)."""
    return " ".join(sorted(_stored_texts(config, providers)))


def _loading_metrics(context: PipelineContext) -> dict[str, Any]:
    (result,) = [
        r
        for r in context.stage_results
        if r.stage_name == PipelineStageType.DOCUMENT_LOADING.value
    ]
    return result.metrics


def test_an_update_to_a_copy_deletes_the_replaced_text(setup) -> None:
    # Regression (rv42): with processing.deduplicate, a batch that updated b
    # to a copy of a dropped b's new text and kept its old one (records,
    # units, graph) as a document the run does not name; a corpus run
    # deletes it.
    config, providers, pipeline = setup
    config.processing.deduplicate = True
    _ok(pipeline.insert_documents([{"id": "a", "text": _SUPPLY}]))
    _ok(pipeline.insert_documents([{"id": "b", "text": _STORAGE}]))
    assert _entity_names(providers) == {"Vendor", "Buyer", "Depot", "Carrier"}

    context = _ok(
        pipeline.insert_documents(
            [{"id": "a", "text": _SUPPLY}, {"id": "b", "text": _SUPPLY}]
        )
    )

    assert set(_records(providers)) == {"a"}
    assert "harbor" not in _texts(config, providers)
    assert _entity_names(providers) == {"Vendor", "Buyer"}
    assert len(context.incremental_delta.deleted) == 1
    metrics = _loading_metrics(context)
    assert (metrics["deduplicated"], metrics["documents_deduplicated"]) == (["b"], 1)


def test_new_copies_are_named_and_never_recorded(setup) -> None:
    # A batch of an update to a copy (b), a new copy (c) and a new document
    # (d): b's earlier text is deleted, c is named and gets no record (there
    # is nothing of it to delete), d is added.
    config, providers, pipeline = setup
    config.processing.deduplicate = True
    _ok(
        pipeline.insert_documents(
            [{"id": "a", "text": _SUPPLY}, {"id": "b", "text": _STORAGE}]
        )
    )

    context = _ok(
        pipeline.insert_documents(
            [
                {"id": "a", "text": _SUPPLY},
                {"id": "b", "text": _SUPPLY},
                {"id": "c", "text": _SUPPLY},
                {"id": "d", "text": "Carrier ships widgets for Buyer."},
            ]
        )
    )

    assert set(_records(providers)) == {"a", "d"}
    texts = _texts(config, providers)
    assert "harbor" not in texts and "Carrier ships widgets for Buyer." in texts
    assert _loading_metrics(context)["deduplicated"] == ["b", "c"]
    with pytest.raises(DocumentNotFoundError, match="'c'"):
        pipeline.delete_documents(["c"])


def test_copies_to_delete_are_asked_about_as_in_a_corpus_run(setup) -> None:
    # Over indexing.max_deletion_share of the scope, the copies a run of
    # changes would delete are asked about, as a corpus run asks.
    from unified_kg_rag.shared.exceptions import ConfirmationDeclinedError

    config, providers, pipeline = setup
    config.processing.deduplicate = True
    config.indexing.max_deletion_share = 0.1
    stored = [
        {"id": f"t{n}", "text": f"Depot stores crate {n} for Carrier."}
        for n in range(9)
    ]
    _ok(pipeline.insert_documents([{"id": "a", "text": _SUPPLY}, *stored]))

    with pytest.raises(ConfirmationDeclinedError, match="near-copies") as asked:
        pipeline.insert_documents(
            [
                {"id": "a", "text": _SUPPLY},
                {"id": "t1", "text": _SUPPLY},
                {"id": "t2", "text": _SUPPLY},
            ]
        )
    assert len(_records(providers)) == 10
    # The batch, not a corpus, holds the copies; Python's remedy is named.
    message = str(asked.value)
    assert "sent in this batch" in message and "in the corpus" not in message
    assert "DataIngestionPipeline(confirm_deletions=...)" in message


def test_delete_documents_of_most_of_a_scope_is_not_asked(setup) -> None:
    # The documents delete_documents names are the caller's request, left
    # out of the indexing.max_deletion_share question (no one can answer it
    # here, so asking would refuse the deletion).
    config, providers, pipeline = setup
    config.indexing.max_deletion_share = 0.1
    stored = [
        {"id": f"t{n}", "text": f"Depot stores crate {n} for Carrier."}
        for n in range(11)
    ]
    _ok(pipeline.insert_documents([{"id": "a", "text": _SUPPLY}, *stored]))

    _ok(pipeline.delete_documents([d["id"] for d in stored]))

    assert set(_records(providers)) == {"a"}


def test_an_update_to_a_copy_killed_mid_deletion_is_finished(setup, mocker) -> None:
    # Killed while the commit deletes b's record, the run retried under its
    # pipeline id finishes the deletion.
    from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore

    config, providers, pipeline = setup
    config.processing.deduplicate = True
    _ok(
        pipeline.insert_documents(
            [{"id": "a", "text": _SUPPLY}, {"id": "b", "text": _STORAGE}]
        )
    )
    mocker.patch.object(SQLiteDocStatusStore, "delete", side_effect=_Killed)
    batch = [{"id": "a", "text": _SUPPLY}, {"id": "b", "text": _SUPPLY}]
    with pytest.raises(_Killed):
        pipeline.insert_documents(batch, pipeline_id="copy-1")
    mocker.stopall()

    _ok(pipeline.insert_documents(batch, pipeline_id="copy-1"))

    assert set(_records(providers)) == {"a"}
    assert "harbor" not in _texts(config, providers)
    assert _entity_names(providers) == {"Vendor", "Buyer"}


def test_a_budget_stop_names_the_id_an_insert_resumes_with(setup) -> None:
    # insert_documents without a pipeline_id runs under a new one, which the
    # stop's advice ("re-run with the same pipeline id") did not name.
    from unified_kg_rag.shared.exceptions import RunBudgetExhausted

    config, providers, pipeline = setup
    config.processing.max_run_model_calls = 1
    batch = [{"id": "a", "text": _SUPPLY}, {"id": "b", "text": _STORAGE}]
    with pytest.raises(RunBudgetExhausted) as stopped:
        pipeline.insert_documents(batch)
    pipeline_id = pipeline.pipeline_id
    assert pipeline_id and f"same pipeline id ({pipeline_id!r})" in str(stopped.value)

    config.processing.max_run_model_calls = None
    _ok(pipeline.insert_documents(batch, pipeline_id=pipeline_id))
    assert set(_records(providers)) == {"a", "b"}


def test_a_document_a_failed_write_records_failed_is_named_to_re_insert(
    setup, mocker, caplog: pytest.LogCaptureFixture
) -> None:
    # A batch merges into an entity a stored document it does not send also
    # states, and that entity's write fails: the stored document is recorded
    # FAILED so a later run writes it again. A corpus run re-reads its file;
    # a run of document changes keeps it until it is inserted again, so the
    # warning names it and says so.
    import logging

    from unified_kg_rag.adapters.storage.sqlite_graph_indexer import (
        SQLiteGraphIndexer,
    )

    config, providers, pipeline = setup
    config.retry.ingestion_item.max_attempts = 1
    _ok(pipeline.insert_documents([{"id": "a", "text": _SUPPLY}]))
    mocker.patch.object(
        SQLiteGraphIndexer, "upsert_entities", side_effect=ConnectionError("down")
    )
    with caplog.at_level(logging.WARNING):
        pipeline.insert_documents([{"id": "n", "text": "Buyer orders widgets."}])
    mocker.stopall()
    assert _records(providers)["a"] is DocStatus.FAILED
    (warning,) = [
        r.getMessage() for r in caplog.records if "outside this run" in r.getMessage()
    ]
    assert warning.endswith(": a") and "insert_documents" in warning

    _ok(pipeline.insert_documents([{"id": "a", "text": _SUPPLY}]))
    assert _records(providers)["a"] is DocStatus.PROCESSED


def test_a_directory_scope_given_through_a_symlink_is_the_corpus_run_s(
    setup, tmp_path: Path
) -> None:
    # A corpus run keys its records by its resolved directory; a run of
    # changes given that directory through a symlink used the path as
    # given, opened a second scope and indexed the corpus's documents again.
    config, providers, pipeline = setup
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "notes.txt").write_text(_SUPPLY)
    link = tmp_path / "link"
    link.symlink_to(corpus, target_is_directory=True)
    _ok(pipeline.run(link))

    _ok(
        pipeline.insert_documents(
            [
                {"id": "notes.txt", "text": _SUPPLY},
                {"id": "more.txt", "text": _STORAGE},
            ],
            source_scope=f"{link}/",
        )
    )
    scopes = {r.scope for r in all_records(providers.build_doc_status())}
    assert sorted(_records(providers)) == ["more.txt", "notes.txt"]
    assert len(scopes) == 1 and scopes.pop().endswith(f"|{corpus.resolve()}")

    _ok(pipeline.delete_documents(["more.txt"], source_scope=str(link)))
    assert sorted(_records(providers)) == ["notes.txt"]


def _scopes(providers: Any) -> set[str]:
    return {
        r.scope.partition("|")[2] for r in all_records(providers.build_doc_status())
    }


@pytest.mark.parametrize(
    "spelling", ["trailing", "doubled", "dotdot", "symlink", "other_host_under_link"]
)
def test_an_explicit_corpus_scope_is_taken_as_the_corpus_run_spells_it(
    setup, tmp_path: Path, spelling: str
) -> None:
    # A corpus run keys its records by processing.document_parsing.
    # source_scope as spelled. The same string given to insert_documents and
    # delete_documents names the same records: no second scope, no
    # DocumentNotFoundError, whatever the directory resolves to here.
    config, providers, pipeline = setup
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "notes.txt").write_text(_SUPPLY)
    link = tmp_path / "link"
    link.symlink_to(corpus, target_is_directory=True)
    scope = {
        "trailing": f"{corpus}/",
        "doubled": str(corpus).replace("/corpus", "//corpus"),
        "dotdot": f"{corpus}/../corpus",
        "symlink": str(link),
        "other_host_under_link": f"{link}/elsewhere",
    }[spelling]
    config.processing.document_parsing.source_scope = scope
    _ok(pipeline.run(corpus))

    _ok(
        pipeline.insert_documents(
            [{"id": "more.txt", "text": _STORAGE}], source_scope=scope
        )
    )
    assert _scopes(providers) == {scope}
    assert sorted(_records(providers)) == ["more.txt", "notes.txt"]

    _ok(pipeline.delete_documents(["notes.txt"], source_scope=scope))
    assert sorted(_records(providers)) == ["more.txt"]
    assert _scopes(providers) == {scope}


def test_another_spelling_of_a_path_of_another_host_is_compared_as_a_string(
    setup, tmp_path: Path
) -> None:
    # A configured scope naming no directory here (the corpus another host
    # syncs) is matched by its spelling alone: trailing and doubled slashes
    # name the same records, with no filesystem to resolve it against.
    config, providers, pipeline = setup
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "notes.txt").write_text(_SUPPLY)
    scope = f"{tmp_path}/elsewhere"
    config.processing.document_parsing.source_scope = scope
    _ok(pipeline.run(corpus))

    _ok(
        pipeline.insert_documents(
            [{"id": "more.txt", "text": _STORAGE}], source_scope=f"{scope}/"
        )
    )
    assert _scopes(providers) == {scope}
    _ok(
        pipeline.delete_documents(
            ["notes.txt"], source_scope=scope.replace("/elsewhere", "//elsewhere/")
        )
    )
    assert sorted(_records(providers)) == ["more.txt"]


@pytest.mark.parametrize("spelling", ["link", "link/", "resolved"])
def test_a_record_inserted_through_a_symlink_by_0_3_0_stays_reachable(
    setup, tmp_path: Path, spelling: str
) -> None:
    # 0.3.0's insert_documents keyed an absolute path as given: a record
    # under a symlinked spelling is deleted by that spelling, another
    # spelling of it, or the directory it resolves to.
    config, providers, pipeline = setup
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    link = tmp_path / "link"
    link.symlink_to(corpus, target_is_directory=True)
    _ok(
        pipeline._run_changes(
            [InputDocument(id="old.txt", text=_STORAGE)], (), str(link), None
        )
    )
    assert _scopes(providers) == {str(link)}
    scope = {"link": str(link), "link/": f"{link}/", "resolved": str(corpus)}[spelling]

    _ok(
        pipeline.insert_documents(
            [{"id": "new.txt", "text": _SUPPLY}], source_scope=scope
        )
    )
    assert _scopes(providers) == {str(link)}
    _ok(pipeline.delete_documents(["old.txt"], source_scope=scope))
    assert sorted(_records(providers)) == ["new.txt"]


def _insert_as_0_3_0(
    pipeline: DataIngestionPipeline,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    name: str,
    text: str,
) -> None:
    # 0.3.0's insert_documents keyed its records by the scope as given,
    # never by another stored spelling of it.
    from unified_kg_rag.application.ingestion import incremental

    with monkeypatch.context() as patched:
        patched.setattr(
            incremental, "stored_scope_spelling", lambda given, *a, **k: given
        )
        _ok(pipeline._run_changes([InputDocument(id=name, text=text)], (), scope, None))


def _by_scope(providers: Any) -> dict[str, set[str]]:
    held: dict[str, set[str]] = {}
    for r in all_records(providers.build_doc_status()):
        held.setdefault(r.scope.partition("|")[2], set()).add(r.file_path)
    return held


def test_two_alike_spellings_held_are_each_taken_as_given(
    setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rv45 ingestion low: a registry holding /h/c and /h/c/ (0.3.0 keyed
    # each as given). A run of changes takes the scope as given when the
    # registry holds it, before any look-alike spelling: taking a
    # look-alike first sent each run on to the other spelling, forever.
    config, providers, pipeline = setup
    plain, slashed = f"{tmp_path}/h/c", f"{tmp_path}/h/c/"
    _insert_as_0_3_0(pipeline, monkeypatch, plain, "a.txt", _SUPPLY)
    _insert_as_0_3_0(pipeline, monkeypatch, slashed, "b.txt", _STORAGE)

    for number, given in enumerate((plain, slashed, f"{tmp_path}/h/c//")):
        _ok(
            pipeline.insert_documents(
                [{"id": f"n{number}.txt", "text": f"{_SUPPLY} Lot {number}."}],
                source_scope=given,
            )
        )
    # A spelling it does not hold takes the first held one, sorted.
    assert _by_scope(providers) == {
        plain: {"a.txt", "n0.txt", "n2.txt"},
        slashed: {"b.txt", "n1.txt"},
    }
    _ok(pipeline.delete_documents(["n1.txt"], source_scope=slashed))
    _ok(pipeline.delete_documents(["n0.txt"], source_scope=plain))
    _ok(pipeline.delete_documents(["n2.txt"], source_scope=f"{tmp_path}/h/c//"))
    assert _by_scope(providers) == {plain: {"a.txt"}, slashed: {"b.txt"}}

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text(_SUPPLY)
    config.processing.document_parsing.source_scope = slashed
    _ok(pipeline.run(corpus))
    assert _by_scope(providers) == {plain: {"a.txt"}, slashed: {"a.txt"}}


def test_a_deletion_split_across_spellings_names_where_each_id_is(
    setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # rv45 ingestion low: the ids are under two spellings of the scope. The
    # call deletes nothing (one call deletes under one spelling) and names
    # the spelling each missing id is under; on a tie the scope as given is
    # the one looked in.
    config, providers, pipeline = setup
    plain, slashed = f"{tmp_path}/h/c", f"{tmp_path}/h/c/"
    _insert_as_0_3_0(pipeline, monkeypatch, plain, "a.txt", _SUPPLY)
    _insert_as_0_3_0(pipeline, monkeypatch, slashed, "b.txt", _STORAGE)
    with pytest.raises(DocumentNotFoundError) as raised:
        pipeline.delete_documents(["a.txt", "b.txt", "z.txt"], source_scope=plain)
    message = str(raised.value)
    assert "No document 'b.txt', 'z.txt' in the doc-status" in message
    assert f"source scope {plain!r}: nothing was deleted" in message
    assert f"; 'b.txt' is under {slashed!r}, another spelling of it" in message
    assert "'z.txt' is" not in message
    assert _by_scope(providers) == {plain: {"a.txt"}, slashed: {"b.txt"}}


def test_a_changes_run_warns_of_copies_under_another_spelling_of_its_directory(
    setup,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    # rv45 ingestion low: a corpus run keyed the resolved directory and
    # 0.3.0's insert_documents the path through a symlink. A run of changes
    # through the link is keyed by the link (the registry holds it), so an
    # update of a corpus document there is a second copy: it is named.
    config, providers, pipeline = setup
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "notes.txt").write_text(_SUPPLY)
    link = tmp_path / "link"
    link.symlink_to(corpus, target_is_directory=True)
    _ok(pipeline.run(corpus))
    with caplog.at_level("WARNING"):
        # One spelling held: nothing to warn of.
        _ok(
            pipeline.insert_documents(
                [{"id": "other.txt", "text": _STORAGE}], source_scope=str(corpus)
            )
        )
    assert "stored twice" not in caplog.text
    _insert_as_0_3_0(pipeline, monkeypatch, str(link), "more.txt", _STORAGE)

    with caplog.at_level("WARNING"):
        _ok(
            pipeline.insert_documents(
                [{"id": "notes.txt", "text": _STORAGE}], source_scope=str(link)
            )
        )
    warned = [r.getMessage() for r in caplog.records if "stored twice" in r.message]
    assert len(warned) == 1
    assert repr(str(corpus.resolve())) in warned[0]
    assert f"same directory as the source scope {str(link)!r}" in warned[0]

    # rv46 ingestion low: a corpus run over the same mixed registry keeps
    # both copies too, and is warned the same way (one shared check).
    caplog.clear()
    with caplog.at_level("WARNING"):
        _ok(pipeline.run(corpus))
    warned = [r.getMessage() for r in caplog.records if "stored twice" in r.message]
    assert len(warned) == 1
    assert repr(str(link)) in warned[0]
    assert f"same directory as the source scope {str(corpus.resolve())!r}" in (
        warned[0]
    )


def _dedup_corpus(tmp_path: Path) -> tuple[Path, Path]:
    """A corpus of z.txt and other.txt, and a symlink to it."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "z.txt").write_text(_SUPPLY)
    (corpus / "other.txt").write_text(_STORAGE)
    link = tmp_path / "link"
    link.symlink_to(corpus, target_is_directory=True)
    return corpus, link


@pytest.mark.parametrize("spelling", ["resolved", "trailing", "link"])
def test_deduplication_keeps_the_indexed_copy_under_any_spelling_of_the_scope(
    setup, tmp_path: Path, spelling: str
) -> None:
    # rv48 ingestion medium: deduplication looked the copies up under the
    # scope as given, the diff under the spelling the registry holds. Under
    # another spelling no copy read as indexed, so input order kept the new
    # copy a and deleted the indexed z it duplicates.
    config, providers, pipeline = setup
    config.processing.deduplicate = True
    corpus, link = _dedup_corpus(tmp_path)
    _ok(pipeline.run(corpus))
    resolved = corpus.resolve().as_posix()
    scope = {"resolved": resolved, "trailing": f"{resolved}/", "link": str(link)}[
        spelling
    ]

    context = _ok(
        pipeline.insert_documents(
            [{"id": "a.txt", "text": _SUPPLY}, {"id": "z.txt", "text": _SUPPLY}],
            source_scope=scope,
        )
    )

    assert sorted(_records(providers)) == ["other.txt", "z.txt"]
    assert _loading_metrics(context)["deduplicated"] == ["a.txt"]
    assert context.incremental_delta.deleted == []
    assert _scopes(providers) == {resolved}


def test_a_corpus_run_keeps_the_indexed_copy_under_a_respelled_explicit_scope(
    setup, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # The same for a corpus run whose configured source scope gained a
    # trailing slash since the run that indexed z. The spelling is resolved
    # once: deduplication and the diff key by it, and it is named once.
    config, providers, pipeline = setup
    config.processing.deduplicate = True
    corpus, _ = _dedup_corpus(tmp_path)
    resolved = corpus.resolve().as_posix()
    config.processing.document_parsing.source_scope = resolved
    _ok(pipeline.run(corpus))
    (corpus / "a.txt").write_text(_SUPPLY)

    config.processing.document_parsing.source_scope = f"{resolved}/"
    with caplog.at_level("WARNING"):
        context = _ok(pipeline.run(corpus))

    assert sorted(_records(providers)) == ["other.txt", "z.txt"]
    assert _loading_metrics(context)["deduplicated"] == ["a.txt"]
    assert _scopes(providers) == {resolved}
    respelled = [r for r in caplog.records if "using the stored spelling" in r.message]
    assert len(respelled) == 1


def test_deduplication_gives_a_failed_copy_way_under_another_spelling(
    setup, tmp_path: Path
) -> None:
    # A FAILED copy gives way to any other (it may have used up its
    # retries), under any spelling of the scope: z, first in the batch and
    # FAILED, is dropped (and its record deleted), its copy a indexed.
    config, providers, pipeline = setup
    config.processing.deduplicate = True
    corpus, link = _dedup_corpus(tmp_path)
    _ok(pipeline.run(corpus))
    store = providers.build_doc_status()
    (record,) = [r for r in all_records(store) if r.file_path == "z.txt"]
    store.put(
        record.model_copy(update={"status": DocStatus.FAILED, "error_info": "boom"})
    )

    context = _ok(
        pipeline.insert_documents(
            [{"id": "z.txt", "text": _SUPPLY}, {"id": "a.txt", "text": _SUPPLY}],
            source_scope=str(link),
        )
    )

    assert _records(providers) == {
        "a.txt": DocStatus.PROCESSED,
        "other.txt": DocStatus.PROCESSED,
    }
    assert _loading_metrics(context)["deduplicated"] == ["z.txt"]


def _parsing_metrics(context: PipelineContext) -> dict[str, Any]:
    (result,) = [
        r
        for r in context.stage_results
        if r.stage_name == PipelineStageType.DOCUMENT_PARSING.value
    ]
    return result.metrics


def test_an_unchanged_file_is_not_parsed_again_under_a_respelled_scope(
    setup, tmp_path: Path
) -> None:
    # The parse of an unchanged file is skipped by its record, looked up
    # under the spelling the registry holds: a configured scope that gained
    # a trailing slash parsed every file again.
    config, providers, pipeline = setup
    corpus, _ = _dedup_corpus(tmp_path)
    resolved = corpus.resolve().as_posix()
    config.processing.document_parsing.source_scope = resolved
    _ok(pipeline.run(corpus))

    config.processing.document_parsing.source_scope = f"{resolved}/"
    context = _ok(pipeline.run(corpus))

    metrics = _parsing_metrics(context)
    assert (metrics["files_unparsed"], metrics["parsed_files"]) == (2, [])
    assert len(context.incremental_delta.unchanged) == 2
    assert _scopes(providers) == {resolved}


def _scoped_records(providers: Any) -> set[tuple[str, str]]:
    return {
        (r.scope.partition("|")[2], r.file_path)
        for r in all_records(providers.build_doc_status())
    }


@pytest.mark.parametrize("deduplicate", [False, True])
def test_a_corpus_run_follows_no_symlink_to_a_stored_scope(
    setup, tmp_path: Path, caplog: pytest.LogCaptureFixture, deduplicate: bool
) -> None:
    # rv49 ingestion medium: with deduplication on, every corpus run read as
    # a run of document changes (the loading stage tested that the pipeline
    # had installed its deletions callable, which it always does), so a
    # corpus run of the resolved directory followed the symlink to the scope
    # the registry held, took it over and deleted mem.txt, which only a run
    # of changes had stated there.
    config, providers, pipeline = setup
    config.processing.deduplicate = deduplicate
    corpus, link = _dedup_corpus(tmp_path)
    config.processing.document_parsing.source_scope = str(link)
    _ok(pipeline.run(corpus))
    _ok(
        pipeline.insert_documents(
            [{"id": "mem.txt", "text": "Carrier ships parts to Depot every Monday."}],
            source_scope=str(link),
        )
    )
    config.processing.document_parsing.source_scope = None

    with caplog.at_level("WARNING"):
        _ok(pipeline.run(corpus))

    resolved = corpus.resolve().as_posix()
    assert _scoped_records(providers) == {
        (resolved, "other.txt"),
        (resolved, "z.txt"),
        (str(link), "mem.txt"),
        (str(link), "other.txt"),
        (str(link), "z.txt"),
    }
    assert not [r for r in caplog.records if "using the stored spelling" in r.message]


@pytest.mark.parametrize("deduplicate", [False, True])
def test_a_corpus_run_keeps_its_symlinked_source_scope(
    setup, tmp_path: Path, deduplicate: bool
) -> None:
    # The other way round: an explicit symlinked source scope is the corpus
    # run's own, whatever processing.deduplicate is; it is not replaced by
    # the resolved directory the registry holds.
    config, providers, pipeline = setup
    config.processing.deduplicate = deduplicate
    corpus, link = _dedup_corpus(tmp_path)
    _ok(pipeline.run(corpus))
    config.processing.document_parsing.source_scope = str(link)

    _ok(pipeline.run(corpus))

    assert _scopes(providers) == {corpus.resolve().as_posix(), str(link)}


def test_a_json_corpus_run_keeps_each_documents_file_path(tmp_path: Path) -> None:
    # A corpus of pre-parsed Document JSON (no parsing stage) keeps the
    # original path each document records; only documents given from memory
    # take their id as their path.
    config = _config(tmp_path / "stores", tmp_path / "viz")
    providers = _providers(
        config, FakeOllama(_respond, ANSWER, DIM), FakeCrossEncoder("rerank")
    )
    stages = dict.fromkeys(PipelineStageType, True)
    stages[PipelineStageType.DOCUMENT_PARSING] = False
    pipeline = DataIngestionPipeline(
        config,
        PipelineConfig(stages_enabled=stages, local_directory=tmp_path / "cache"),
        providers=providers,
    )
    corpus = tmp_path / "parsed"
    corpus.mkdir()
    original = "/archive/contracts/supply.pdf"
    (corpus / "supply.json").write_text(
        Document(
            page_content=_SUPPLY,
            document_id="supply",
            file_name="supply.pdf",
            file_path=original,
        ).model_dump_json()
    )
    try:
        context = _ok(pipeline.run(corpus))
    finally:
        pipeline.close()

    assert [d.file_path for d in context.documents] == [original]


@pytest.mark.parametrize("deduplicate", [False, True])
def test_a_respelled_scope_reads_the_namespace_scopes_once(
    setup, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, deduplicate: bool
) -> None:
    # The first lookup that finds nothing under the scope as given (the
    # parsing stage's, or deduplication's) reads the namespace's stored
    # scopes; the run keeps them, so the diff is keyed by the stored
    # spelling at once instead of diffing under the scope as given first.
    config, providers, pipeline = setup
    config.processing.deduplicate = deduplicate
    corpus, _ = _dedup_corpus(tmp_path)
    resolved = corpus.resolve().as_posix()
    config.processing.document_parsing.source_scope = resolved
    _ok(pipeline.run(corpus))
    config.processing.document_parsing.source_scope = f"{resolved}/"
    store_type = type(providers.build_doc_status())
    diff = store_type.diff
    incoming: list[dict[str, str]] = []

    def counted(self: Any, fingerprints: dict[str, str], *args: Any, **kwargs: Any):
        incoming.append(fingerprints)
        return diff(self, fingerprints, *args, **kwargs)

    monkeypatch.setattr(store_type, "diff", counted)
    context = _ok(pipeline.run(corpus))

    assert len(context.incremental_delta.unchanged) == 2
    assert [bool(fingerprints) for fingerprints in incoming] == [False, True]
