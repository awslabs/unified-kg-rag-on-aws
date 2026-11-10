# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""An incremental run parses only the source files that changed.

Each record holds its file's source digest (the file's bytes and the parser
that read it); a file whose digest is its PROCESSED record's is not parsed
again. Runs on the embedded stores with the fake model server
(``test_embedded_e2e``): no AWS, no container. Synthetic data only.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.fixtures.fakes.model_servers import FakeOllama
from tests.fixtures.fakes.rerank import FakeCrossEncoder
from tests.integration.test_embedded_e2e import (
    ANSWER,
    DIM,
    _config,
    _errors,
    _ingest,
    _providers,
    _respond,
    _stored_texts,
    no_aws,
)
from unified_kg_rag.adapters.doc_status.sqlite import SQLiteDocStatusStore
from unified_kg_rag.adapters.ingestion import parser as parser_module
from unified_kg_rag.adapters.ingestion.file_loaders import TextFileLoader
from unified_kg_rag.application.ingestion import source_parsing
from unified_kg_rag.domain.models import Config, DocStatus, PipelineStageStatus

pytestmark = pytest.mark.integration

__all__ = ["no_aws"]  # the fixture, imported for the tests here

TEXTS = {
    "contract.txt": "Vendor supplies widgets to Buyer.",
    "logistics.txt": "Depot stores widgets for Vendor.",
    "freight.txt": "Carrier moves gadgets for Depot.",
}


@pytest.fixture
def parsed(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The names of the files parsed, in order."""
    names: list[str] = []
    parse_file = parser_module.FileParser.parse_file

    def counted(self, file_path, index_value=None):  # noqa: ANN001, ANN202
        names.append(Path(file_path).name)
        return parse_file(self, file_path, index_value)

    monkeypatch.setattr(parser_module.FileParser, "parse_file", counted)
    return names


def _corpus(tmp_path: Path) -> Path:
    source = tmp_path / "corpus"
    source.mkdir()
    for name, text in TEXTS.items():
        (source / name).write_text(text, encoding="utf-8")
    return source


def _records(config: Config) -> dict[str, tuple[DocStatus, str | None]]:
    store = SQLiteDocStatusStore(config)
    try:
        return {
            str(record.file_path): (record.status, record.source_digest)
            for namespace in ("default",)
            for record in store.list_namespace(namespace)
        }
    finally:
        store.close()


def _setup(tmp_path: Path):  # noqa: ANN202
    config = _config(tmp_path / "stores", tmp_path / "viz")
    server = FakeOllama(_respond, ANSWER, DIM)
    providers = _providers(config, server, FakeCrossEncoder("synthetic-reranker"))
    return config, server, providers


def _run(config, providers, source: Path, cache: Path):  # noqa: ANN001, ANN202
    context = _ingest(config, providers, source, cache)
    assert context.status is PipelineStageStatus.COMPLETED, _errors(context)
    return context


def test_a_run_without_changes_parses_no_file(
    tmp_path: Path, no_aws, parsed: list[str]
) -> None:
    config, _, providers = _setup(tmp_path)
    source = _corpus(tmp_path)
    _run(config, providers, source, tmp_path / "run-1")
    assert sorted(parsed) == sorted(TEXTS)
    records = _records(config)
    assert {name: status for name, (status, _) in records.items()} == dict.fromkeys(
        TEXTS, DocStatus.PROCESSED
    )
    assert all(digest for _, digest in records.values())

    parsed.clear()
    context = _run(config, providers, source, tmp_path / "run-2")
    assert parsed == []
    delta = context.incremental_delta
    assert delta is not None and delta.is_empty
    assert len(delta.unchanged) == len(TEXTS)
    assert _stored_texts(config, providers) == set(TEXTS.values())
    parsing = next(
        r for r in context.stage_results if r.stage_name == "document_parsing"
    )
    assert parsing.metrics["files_unparsed"] == len(TEXTS)

    # One changed file is the only one parsed, and indexed with its new text.
    (source / "freight.txt").write_text(ANSWER, encoding="utf-8")
    parsed.clear()
    context = _run(config, providers, source, tmp_path / "run-3")
    assert parsed == ["freight.txt"]
    assert context.incremental_delta is not None
    assert len(context.incremental_delta.changed) == 1
    texts = {**TEXTS, "freight.txt": ANSWER}
    assert _stored_texts(config, providers) == set(texts.values())


def test_a_corpus_read_through_a_symlink_is_not_parsed_again(
    tmp_path: Path, no_aws, parsed: list[str]
) -> None:
    # The registry keys a corpus by its resolved directory: the parsing stage
    # must look its records up under the same source scope.
    config, _, providers = _setup(tmp_path)
    linked = tmp_path / "linked"
    linked.symlink_to(_corpus(tmp_path), target_is_directory=True)
    _run(config, providers, linked, tmp_path / "run-1")
    parsed.clear()
    _run(config, providers, linked, tmp_path / "run-2")
    assert parsed == []


def test_records_of_an_earlier_release_get_their_digest_once(
    tmp_path: Path, no_aws, parsed: list[str]
) -> None:
    # A record written before source digests existed: its file is parsed
    # (the content hash finds it unchanged) and the record gets the digest,
    # so the next run parses nothing. Nothing is re-extracted.
    config, server, providers = _setup(tmp_path)
    source = _corpus(tmp_path)
    _run(config, providers, source, tmp_path / "run-1")
    store = SQLiteDocStatusStore(config)
    store.put_many(
        [
            record.model_copy(update={"source_digest": None})
            for record in store.list_namespace("default")
        ]
    )
    store.close()

    parsed.clear()
    tokens = server.input_tokens
    context = _run(config, providers, source, tmp_path / "run-2")
    assert sorted(parsed) == sorted(TEXTS)
    assert context.incremental_delta is not None
    assert context.incremental_delta.is_empty
    assert all(digest for _, digest in _records(config).values())
    assert server.input_tokens == tokens  # no model call

    parsed.clear()
    _run(config, providers, source, tmp_path / "run-3")
    assert parsed == []


def test_unparsed_files_the_run_processes_again_are_read_first(
    tmp_path: Path, no_aws, parsed: list[str]
) -> None:
    # New extraction settings re-extract every document: the files skipped
    # by their digest are parsed before any stage reads their text.
    config, _, providers = _setup(tmp_path)
    source = _corpus(tmp_path)
    _run(config, providers, source, tmp_path / "run-1")
    config.processing.graph_extraction.max_entities_per_chunk += 1

    parsed.clear()
    context = _run(config, providers, source, tmp_path / "run-2")
    assert sorted(parsed) == sorted(TEXTS)
    delta = context.incremental_delta
    assert delta is not None and len(delta.changed) == len(TEXTS)
    assert all(document.page_content for document in context.documents)
    assert _stored_texts(config, providers) == set(TEXTS.values())


def test_another_parser_parses_every_file_again(
    tmp_path: Path, no_aws, parsed: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _, providers = _setup(tmp_path)
    source = _corpus(tmp_path)
    _run(config, providers, source, tmp_path / "run-1")

    class OtherTextLoader(TextFileLoader):
        """Another loader for .txt: the same text, another parser."""

    configs = dict(parser_module.ParserFactory._loader_configs)
    monkeypatch.setattr(parser_module.ParserFactory, "_loader_configs", configs)
    parser_module.ParserFactory.register_loader(".txt", OtherTextLoader, {}, "Text")
    parsed.clear()
    context = _run(config, providers, source, tmp_path / "run-2")
    assert sorted(parsed) == sorted(TEXTS)
    # The text is the same: nothing to re-extract.
    assert context.incremental_delta is not None
    assert context.incremental_delta.is_empty


def test_a_registry_off_run_parses_every_file(
    tmp_path: Path, no_aws, parsed: list[str]
) -> None:
    config, _, providers = _setup(tmp_path)
    source = _corpus(tmp_path)
    _run(config, providers, source, tmp_path / "run-1")
    off = config.model_copy(deep=True)
    off.providers.doc_status.enabled = False
    off_providers = _providers(
        off, FakeOllama(_respond, ANSWER, DIM), FakeCrossEncoder("synthetic-reranker")
    )
    parsed.clear()
    _run(off, off_providers, source, tmp_path / "run-2")
    assert sorted(parsed) == sorted(TEXTS)


def test_files_parse_in_a_pool_when_they_would_take_longer_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _corpus(tmp_path)
    paths = sorted(source.iterdir())
    config = Config()
    here, _ = source_parsing.SourceFileParser(config, source).parse(paths)
    # Any parse time beats a pool that starts at once.
    monkeypatch.setattr(source_parsing, "_POOL_START_SECONDS", -1.0)
    monkeypatch.setattr(source_parsing, "default_max_workers", lambda: 2)
    pooled: list[int] = []
    pool = source_parsing.SourceFileParser._parse_in_pool

    def counted(self, paths, results, start, workers):  # noqa: ANN001, ANN202
        pooled.append(len(paths) - start)
        return pool(self, paths, results, start, workers)

    monkeypatch.setattr(source_parsing.SourceFileParser, "_parse_in_pool", counted)
    there, failed = source_parsing.SourceFileParser(config, source).parse(paths)
    assert pooled == [len(paths)]
    assert failed == []
    assert {p: d.model_dump() for p, d in there.items()} == {
        p: d.model_dump() for p, d in here.items()
    }


def test_a_slow_first_parse_of_a_file_type_starts_no_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The first file of a type also pays its loader's imports: timing it
    # started a pool (seconds) for a corpus of text files that parse in
    # microseconds.
    source = tmp_path / "corpus"
    source.mkdir()
    for i in range(20):
        (source / f"doc_{i:02d}.txt").write_text(f"Vendor {i} ships.", encoding="utf-8")
    parse_here = source_parsing.SourceFileParser._parse_here
    # The parser's clock, advanced only by the parses below: each costs what
    # it is given here, whatever the host's load (a stall on a busy CI runner
    # must not read as a slow file type).
    now = [0.0]
    monkeypatch.setattr(
        source_parsing, "time", SimpleNamespace(perf_counter=lambda: now[0])
    )
    calls: list[int] = []

    def slow_first(self, path):  # noqa: ANN001, ANN202
        calls.append(1)
        # The first file alone would outlast a pool's start; the other 19
        # together take about a tenth of that start.
        cost = 2 * source_parsing._POOL_START_SECONDS
        now[0] += cost if len(calls) == 1 else cost / 400
        return parse_here(self, path)

    monkeypatch.setattr(source_parsing.SourceFileParser, "_parse_here", slow_first)
    monkeypatch.setattr(source_parsing, "default_max_workers", lambda: 2)
    pooled: list[int] = []
    monkeypatch.setattr(
        source_parsing.SourceFileParser,
        "_parse_in_pool",
        lambda *args: pooled.append(1),
    )
    documents, failed = source_parsing.SourceFileParser(Config(), source).parse(
        sorted(source.iterdir())
    )
    assert pooled == [] and failed == [] and len(documents) == 20


def test_a_loader_that_does_not_pickle_parses_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class LocalLoader(TextFileLoader):
        """Defined in a function: no worker can import it."""

    configs = dict(parser_module.ParserFactory._loader_configs)
    monkeypatch.setattr(parser_module.ParserFactory, "_loader_configs", configs)
    parser_module.ParserFactory.register_loader(".txt", LocalLoader)
    monkeypatch.setattr(source_parsing, "_POOL_START_SECONDS", -1.0)
    monkeypatch.setattr(source_parsing, "default_max_workers", lambda: 2)
    source = _corpus(tmp_path)
    documents, failed = source_parsing.SourceFileParser(Config(), source).parse(
        sorted(source.iterdir())
    )
    assert failed == []
    assert {d.page_content for d in documents.values()} == set(TEXTS.values())
