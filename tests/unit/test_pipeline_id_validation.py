# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""pipeline_id is validated where it enters the system (AWS-free).

Regression: the id became a local cache path (``<cache>/<pipeline_id>``) and
an S3 prefix unchecked, so ``../x`` or ``a/b`` escaped the cache directory and
the S3 sync uploaded whatever JSON sat under the resolved path. The id now
must be one safe path segment (letters, digits, ``.``, ``_``, ``-``; no
separator, no ``..``) at the CLI (``--pipeline-id`` and the
``GRAPHRAG_PIPELINE_ID`` variable the Step Functions task sets), in
``PipelineConfig`` and in ``DataIngestionPipeline.run``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from unified_kg_rag.application.cli import run_ingestion_pipeline
from unified_kg_rag.application.ingestion.pipeline import DataIngestionPipeline
from unified_kg_rag.domain.models import PipelineConfig

pytestmark = pytest.mark.unit

_BAD_IDS = ["../escape", "a/b", "..", ".hidden", "a..b", "run 1", "a\\b", "*"]
_GOOD_IDS = ["run-001", "pipeline-1a2b3c4d-20260101_120000", "Run-1", "run.1"]


def _parser():
    return run_ingestion_pipeline.CommandLineInterface._setup_arguments()


@pytest.mark.parametrize("bad", _BAD_IDS)
def test_cli_rejects_unsafe_pipeline_id(bad: str) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args(["--pipeline-id", bad])


@pytest.mark.parametrize("bad", ["../escape", "a/b"])
def test_cli_rejects_unsafe_pipeline_id_from_env(
    bad: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GRAPHRAG_PIPELINE_ID", bad)
    with pytest.raises(SystemExit):
        _parser().parse_args([])


def test_cli_treats_empty_env_pipeline_id_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GRAPHRAG_PIPELINE_ID", "")
    assert _parser().parse_args([]).pipeline_id is None


@pytest.mark.parametrize("good", _GOOD_IDS)
def test_cli_accepts_safe_pipeline_id(good: str) -> None:
    assert _parser().parse_args(["--pipeline-id", good]).pipeline_id == good


@pytest.mark.parametrize("bad", ["../escape", "a/b", ".."])
def test_pipeline_config_rejects_unsafe_pipeline_id(bad: str) -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(pipeline_id=bad)


def test_pipeline_config_accepts_safe_pipeline_id() -> None:
    assert PipelineConfig(pipeline_id="run-001").pipeline_id == "run-001"


@pytest.mark.parametrize("bad", ["../escape", "a/b"])
def test_pipeline_run_rejects_unsafe_pipeline_id(bad: str) -> None:
    pipe = object.__new__(DataIngestionPipeline)
    pipe.pipeline_config = SimpleNamespace(pipeline_id=None)
    with pytest.raises(ValueError, match="pipeline_id"):
        DataIngestionPipeline._resolve_pipeline_id(pipe, Path("/tmp/src"), bad)
