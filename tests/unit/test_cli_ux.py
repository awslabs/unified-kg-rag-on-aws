# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the CLIs print and accept for a person at the terminal (AWS-free):
the namespace flags of run-ingestion, run-rag's search hits and sources,
the interactive commands and the failure messages."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from unified_kg_rag.application.cli import run_ingestion_pipeline
from unified_kg_rag.domain.models import Config, Methodology

pytestmark = pytest.mark.unit


def _ing_parser() -> argparse.ArgumentParser:
    return run_ingestion_pipeline.CommandLineInterface._setup_arguments()


def _ingestion_runner(
    mocker, tmp_path: Path, config: Config, *flags: str
) -> run_ingestion_pipeline.IngestionPipelineRunner:
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
    source = tmp_path / "src"
    source.mkdir(exist_ok=True)
    args = _ing_parser().parse_args(
        [
            "--source-directory",
            str(source),
            "--cache-directory",
            str(tmp_path / "cache"),
            *flags,
        ]
    )
    return run_ingestion_pipeline.IngestionPipelineRunner(args)


# --- run-ingestion: namespace flags -------------------------------------


def test_ingestion_namespace_flags_override_the_config_for_the_run(
    mocker, tmp_path
) -> None:
    config = Config(processing={"document_parsing": {"index_value": "from-file"}})
    runner = _ingestion_runner(
        mocker,
        tmp_path,
        config,
        "--suffix",
        "tenant-a",
        "--methodologies",
        "lightrag, raptor",
        "--reset",
    )
    assert runner.config.processing.document_parsing.index_value == "tenant-a"
    assert runner.config.indexing.methodologies == [
        Methodology.LIGHTRAG,
        Methodology.RAPTOR,
    ]
    assert runner.config.indexing.reset is True
    # The loaded config is not changed: a later run of the same file does
    # not reset again.
    assert config.indexing.reset is False


def test_ingestion_without_namespace_flags_keeps_the_config(mocker, tmp_path) -> None:
    config = Config(processing={"document_parsing": {"index_value": "from-file"}})
    runner = _ingestion_runner(mocker, tmp_path, config)
    assert runner.config.processing.document_parsing.index_value == "from-file"
    assert runner.config.indexing.reset is False


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--methodologies", "lightrg"], "unknown methodology 'lightrg'"),
        (["--methodologies", "lightrag,,raptor"], "empty methodology name"),
        (["--suffix", "Bad Name"], "--suffix"),
    ],
    ids=["unknown-methodology", "empty-name", "bad-suffix"],
)
def test_ingestion_namespace_flag_errors_are_usage_errors(
    mocker, tmp_path, capsys, flags: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as exited:
        _ingestion_runner(mocker, tmp_path, Config(), *flags)
    assert exited.value.code == 2
    assert message in capsys.readouterr().err


def test_ingestion_banner_names_the_namespace_as_run_rag_does(mocker, tmp_path) -> None:
    config = Config(indexing={"additional_suffix": "v2"})
    runner = _ingestion_runner(mocker, tmp_path, config, "--suffix", "tenant-a")
    printed = mocker.patch.object(run_ingestion_pipeline.console, "print")
    runner._display_run_info()
    (panel,), _ = printed.call_args
    text = str(panel.renderable)
    assert "Index Namespace:[/dim] 'tenant-a-v2'" in text
    assert "Methodologies:[/dim] graphrag, lightrag" in text
