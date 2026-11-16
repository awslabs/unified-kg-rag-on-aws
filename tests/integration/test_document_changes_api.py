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
    assert filters["Vendor supplies widgets to Buyer. 공급자는 납품한다."] == {
        "category": "contract",
        "year": 2024,
    }
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

    with pytest.raises(ConfirmationDeclinedError, match="near-copies"):
        pipeline.insert_documents(
            [
                {"id": "a", "text": _SUPPLY},
                {"id": "t1", "text": _SUPPLY},
                {"id": "t2", "text": _SUPPLY},
            ]
        )
    assert len(_records(providers)) == 10


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
