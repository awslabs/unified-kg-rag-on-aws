# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source files the parser cannot read are reported, not silently skipped."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pytest

from unified_kg_rag.application.ingestion.pipeline_stages import DocumentParsingStage
from unified_kg_rag.domain.ingestion.cache_keys import corpus_manifest_fingerprint
from unified_kg_rag.domain.models import (
    Config,
    PipelineContext,
    PipelineStageStatus,
)
from unified_kg_rag.shared import PipelineStageError

pytestmark = pytest.mark.unit

_OPTIONAL = {".md", ".markdown", ".htm", ".html"}


def _stage(source: Path) -> DocumentParsingStage:
    stage = DocumentParsingStage(Config(), source_directory=source)
    # Behave as if the optional 'unstructured' extra is not installed.
    stage.supported_extensions = [
        e for e in stage.supported_extensions if e not in _OPTIONAL
    ]
    return stage


def _skip_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "Skipping" in r.getMessage()]


def test_skipped_files_are_warned_once_with_counts_and_remedies(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    for name in ("a.md", "b.md", "page.html", "notes.docx", "keep.txt", ".hidden"):
        (tmp_path / name).write_text("Vendor ships goods to Buyer.")

    with caplog.at_level(logging.WARNING):
        files, skipped = _stage(tmp_path)._discover_source_files()

    assert [f.name for f in files] == ["keep.txt"]
    # Hidden files are neither read nor reported.
    assert sorted(Path(p).name for p in skipped) == [
        "a.md",
        "b.md",
        "notes.docx",
        "page.html",
    ]
    (warning,) = _skip_warnings(caplog)
    assert f"Skipping 4 file(s) in '{tmp_path}'" in warning
    assert "2 '.md'" in warning and "1 '.html'" in warning and "1 '.docx'" in warning
    assert "for .html, .md, install the optional 'unstructured' extra" in warning
    assert "uv sync --extra unstructured" in warning
    assert "for .docx, convert them" in warning
    assert "keeps what they indexed before" in warning


def test_the_stage_reports_skipped_files_apart_from_failed_ones(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.md").write_text("Vendor ships goods to Buyer.")
    (tmp_path / "keep.txt").write_text("Vendor ships goods to Buyer.")
    context = PipelineContext(
        pipeline_id="p",
        config={},
        status=PipelineStageStatus.RUNNING,
        start_time=datetime(2026, 1, 1),
        source_directory=tmp_path,
    )

    _, _, metrics = _stage(tmp_path)._execute_core(context)

    assert context.skipped_source_files == [str(tmp_path / "a.md")]
    assert context.failed_source_files == []
    assert metrics is not None
    assert (metrics["files_skipped"], metrics["files_failed"]) == (1, 0)


def test_files_under_hidden_directories_are_neither_parsed_nor_fingerprinted(
    tmp_path: Path,
) -> None:
    # One rule for discovery and the corpus fingerprint: a file under a hidden
    # directory used to be parsed while its edits never changed the cache key.
    (tmp_path / "keep.txt").write_text("Vendor ships goods to Buyer.")
    (tmp_path / ".drafts").mkdir()
    hidden = tmp_path / ".drafts" / "note.txt"
    hidden.write_text("Draft text.")

    assert [f.name for f in _stage(tmp_path)._discover_files()] == ["keep.txt"]
    before = corpus_manifest_fingerprint(tmp_path, [".txt"])
    hidden.write_text("Edited draft text.")
    assert corpus_manifest_fingerprint(tmp_path, [".txt"]) == before


def test_hidden_corpus_root_is_not_excluded(tmp_path: Path) -> None:
    root = tmp_path / ".cache" / "corpus"
    root.mkdir(parents=True)
    (root / "keep.txt").write_text("Vendor ships goods to Buyer.")
    assert [f.name for f in _stage(root)._discover_files()] == ["keep.txt"]


def test_first_stage_with_no_parseable_files_names_the_source_directory(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.md").write_text("Vendor ships goods to Buyer.")
    stage = _stage(tmp_path)

    result = stage.execute(
        PipelineContext(
            pipeline_id="p",
            config={},
            status=PipelineStageStatus.RUNNING,
            start_time=datetime(2026, 1, 1),
            source_directory=tmp_path,
        )
    )

    assert result.error_message is not None
    assert str(tmp_path) in result.error_message
    assert "previous stage" not in result.error_message


def test_first_stage_empty_input_raises_without_previous_stage_hint(
    tmp_path: Path,
) -> None:
    stage = _stage(tmp_path)
    with pytest.raises(PipelineStageError) as exc:
        stage._validate_critical_stage_output(input_count=0, output_count=0)
    assert "previous stage" not in str(exc.value)
    assert str(tmp_path) in str(exc.value)
