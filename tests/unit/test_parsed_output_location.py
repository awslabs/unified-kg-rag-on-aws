# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parsed-document output must never land in (or be re-read from) the corpus.

Regression for the default that wrote ``<stem>.json`` into the source directory:
the loading stage then tried to read every raw file as Document JSON ("Failed to
load document ... Expecting value"), and because ``.json`` is itself a parseable
source format, a re-run ingested the previous run's output as new documents.
Runs only the parsing + loading stages; AWS-free.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pytest

from unified_kg_rag.adapters.ingestion.parser import ParserFactory
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.application.ingestion.pipeline_stages import (
    DocumentLoadingStage,
    DocumentParsingStage,
)
from unified_kg_rag.domain.models import (
    Config,
    Document,
    PipelineConfig,
    PipelineContext,
    PipelineStageStatus,
    PipelineStageType,
)

pytestmark = pytest.mark.unit

_DOCS = {
    "alpha.txt": "Vendor A supplies widgets to Buyer B under a framework agreement.",
    "beta.txt": "Buyer B pays Vendor A a fixed monthly fee of 1,000 credits.",
    "nested/gamma.txt": "Vendor C audits the widget deliveries every quarter.",
}
_PREP_STAGES = {PipelineStageType.DOCUMENT_PARSING, PipelineStageType.DOCUMENT_LOADING}


def _write_corpus(root: Path) -> None:
    for name, text in _DOCS.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _pipeline(
    source: Path,
    cache: Path,
    target: Path | None = None,
    force_rebuild: bool = False,
) -> DataIngestionPipeline:
    config = Config()
    pipeline_config = PipelineConfig(
        stages_enabled={st: st in _PREP_STAGES for st in PipelineStageType},
        local_directory=cache,
        force_rebuild=force_rebuild,
    )
    return DataIngestionPipeline(
        config=config,
        pipeline_config=pipeline_config,
        source_directory=source,
        target_directory=target,
    )


def _completed(context) -> bool:
    return all(r.status == PipelineStageStatus.COMPLETED for r in context.stage_results)


def _stage_result(context, stage_type: PipelineStageType):
    return next(r for r in context.stage_results if r.stage_name == stage_type.value)


def _assert_only_source_documents(context) -> None:
    # Exactly the raw corpus files were parsed (no earlier .json output or cache
    # file was picked up as a source) and each document holds its source text.
    parsing = _stage_result(context, PipelineStageType.DOCUMENT_PARSING)
    assert parsing.input_count == len(_DOCS)
    texts = sorted(d.content.text.strip() for d in context.documents)
    assert texts == sorted(_DOCS.values())


def test_default_run_leaves_source_untouched(tmp_path: Path, caplog) -> None:
    source, cache = tmp_path / "corpus", tmp_path / "cache"
    _write_corpus(source)
    before = _snapshot(source)

    with caplog.at_level(logging.WARNING):
        context = _pipeline(source, cache).run(source, pipeline_id="run-1")

    assert _completed(context)
    assert _snapshot(source) == before
    assert len(context.documents) == len(_DOCS)
    assert "Failed to load" not in caplog.text
    # Parsed output goes to the pipeline-owned dir under the cache dir.
    out = cache / DocumentParsingStage.DEFAULT_OUTPUT_SUBDIR / "run-1"
    assert sorted(p.name for p in out.glob("*.json")) == [
        "alpha.json",
        "beta.json",
        "gamma.json",
    ]


def test_loading_reuses_parsed_documents_without_failures(tmp_path: Path) -> None:
    source, cache = tmp_path / "corpus", tmp_path / "cache"
    _write_corpus(source)
    context = _pipeline(source, cache).run(source, pipeline_id="run-1")

    loading = _stage_result(context, PipelineStageType.DOCUMENT_LOADING)
    assert loading.input_count == len(_DOCS)
    assert loading.output_count == len(_DOCS)
    assert loading.metrics["failed_files"] == []


@pytest.mark.parametrize("same_pipeline_id", [True, False])
def test_rerun_on_same_source_yields_same_document_count(
    tmp_path: Path, same_pipeline_id: bool
) -> None:
    source, cache = tmp_path / "corpus", tmp_path / "cache"
    _write_corpus(source)
    first = _pipeline(source, cache).run(source, pipeline_id="run-1")
    second = _pipeline(source, cache, force_rebuild=True).run(
        source, pipeline_id="run-1" if same_pipeline_id else "run-2"
    )

    assert len(first.documents) == len(second.documents) == len(_DOCS)
    assert sorted(d.file_name for d in second.documents) == sorted(
        Path(n).name for n in _DOCS
    )
    _assert_only_source_documents(second)


def test_cache_dir_inside_source_is_not_parsed(tmp_path: Path) -> None:
    # e.g. `run-ingestion --source-directory .` with the default ./cache
    source = tmp_path / "corpus"
    cache = source / "cache"
    _write_corpus(source)
    _pipeline(source, cache).run(source, pipeline_id="run-1")
    context = _pipeline(source, cache, force_rebuild=True).run(
        source, pipeline_id="run-2"
    )
    assert len(context.documents) == len(_DOCS)
    _assert_only_source_documents(context)


def test_explicit_target_directory_still_written(tmp_path: Path) -> None:
    source, cache, target = tmp_path / "corpus", tmp_path / "cache", tmp_path / "out"
    _write_corpus(source)
    before = _snapshot(source)
    context = _pipeline(source, cache, target=target).run(source, pipeline_id="r")

    assert _snapshot(source) == before
    assert len(context.documents) == len(_DOCS)
    assert sorted(p.name for p in target.glob("*.json")) == [
        "alpha.json",
        "beta.json",
        "gamma.json",
    ]
    assert not (cache / DocumentParsingStage.DEFAULT_OUTPUT_SUBDIR).exists()


def test_explicit_target_inside_source_is_not_reparsed(tmp_path: Path) -> None:
    source, cache = tmp_path / "corpus", tmp_path / "cache"
    target = source / "parsed"
    _write_corpus(source)
    _pipeline(source, cache, target=target).run(source, pipeline_id="run-1")
    context = _pipeline(source, cache, target=target, force_rebuild=True).run(
        source, pipeline_id="run-2"
    )
    assert len(context.documents) == len(_DOCS)
    _assert_only_source_documents(context)


def test_target_equal_to_source_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "corpus"
    source.mkdir()
    with pytest.raises(ValueError, match="must not be the source directory"):
        DocumentParsingStage(
            config=Config(), source_directory=source, target_directory=source
        )


def test_loading_without_parsing_reads_only_json(tmp_path: Path, caplog) -> None:
    # Pre-parsed corpus mode (parsing stage disabled): stray raw files beside
    # the Document JSON are ignored instead of failing as invalid JSON.
    config = Config()
    source = tmp_path / "preparsed"
    source.mkdir()
    _write_corpus(tmp_path / "raw")
    for raw in sorted((tmp_path / "raw").rglob("*.txt")):
        parsed = ParserFactory.create_parser(raw, config).parse_file(raw, None)
        parsed.to_json_file(source / f"{raw.stem}.json")
    (source / "notes.txt").write_text("not a document json", encoding="utf-8")

    stage = DocumentLoadingStage(config=config, source_directory=source)
    context = PipelineContext(
        pipeline_id="p",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=source,
    )
    with caplog.at_level(logging.WARNING):
        result = stage.execute(context)

    assert result.status == PipelineStageStatus.COMPLETED
    assert result.input_count == len(_DOCS)
    assert len(context.documents) == len(_DOCS)
    assert all(isinstance(d, Document) for d in context.documents)
    assert "Failed to load" not in caplog.text
