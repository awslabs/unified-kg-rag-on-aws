# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source files the parser cannot read are reported, not silently skipped."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pytest

from unified_kg_rag.application.ingestion.pipeline_stages import DocumentParsingStage
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


def test_skipped_extensions_are_warned_once_each(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    for name in ("a.md", "b.md", "page.html", "notes.docx", "keep.txt", ".hidden"):
        (tmp_path / name).write_text("Vendor ships goods to Buyer.")

    with caplog.at_level(logging.WARNING):
        files = _stage(tmp_path)._discover_files()

    assert [f.name for f in files] == ["keep.txt"]
    warnings = sorted(_skip_warnings(caplog))
    assert len(warnings) == 3  # .docx, .html, .md; hidden files are not reported
    md = next(w for w in warnings if "'.md'" in w)
    assert "2 " in md and "uv sync --extra unstructured" in md
    docx = next(w for w in warnings if "'.docx'" in w)
    assert "unstructured" not in docx


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
