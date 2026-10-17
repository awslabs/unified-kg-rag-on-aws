# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every CLI follows one convention for file flags, output format and exit
codes (``application/cli/conventions.py``)."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest

from unified_kg_rag.application.cli import (
    prepare_benchmark,
    run_evaluation,
    run_ingestion_pipeline,
    run_prompt_tuning,
    run_rag_chain,
    run_visualization,
)
from unified_kg_rag.application.cli.conventions import exits_on_sigterm
from unified_kg_rag.domain.models import (
    Config,
    PipelineConfig,
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
)
from unified_kg_rag.shared.utils.display import console

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _restore_console():
    # --output-format json moves the shared console to stderr.
    yield
    console.stderr = False


PARSERS: dict[str, Callable[[], argparse.ArgumentParser]] = {
    "run-ingestion": run_ingestion_pipeline.CommandLineInterface._setup_arguments,
    "run-rag": run_rag_chain.CommandLineInterface._setup_arguments,
    "run-eval": run_evaluation.CommandLineInterface._setup_arguments,
    "run-visualization": run_visualization._build_parser,
    "run-prompt-tuning": run_prompt_tuning._build_parser,
    "prepare-benchmark": prepare_benchmark._parser,
}

# The file and directory flags any CLI may have: an input corpus directory,
# an input data file, where the command writes, and the config file.
FILE_FLAGS = {
    "--source-directory",
    "--data-path",
    "--output-directory",
    "--output-path",
    "--cache-directory",
    "--config-path",
}


def _options(parser: argparse.ArgumentParser) -> list[str]:
    return [o for action in parser._actions for o in action.option_strings]


@pytest.mark.parametrize("cli", PARSERS)
def test_file_flags_follow_one_naming_scheme(cli: str) -> None:
    options = _options(PARSERS[cli]())
    file_like = [o for o in options if re.search(r"-(dir|directory|path)$", o)]
    assert set(file_like) <= FILE_FLAGS, f"{cli}: {sorted(set(file_like) - FILE_FLAGS)}"
    for legacy in ("--output", "--source", "--output-dir", "--target-directory"):
        assert legacy not in options, f"{cli} still has {legacy}"


@pytest.mark.parametrize("cli", PARSERS)
def test_every_cli_offers_json_output(cli: str) -> None:
    parser = PARSERS[cli]()
    action = next(a for a in parser._actions if "--output-format" in a.option_strings)
    assert (action.default, list(action.choices or [])) == ("text", ["text", "json"])


@pytest.mark.parametrize("cli", PARSERS)
def test_flag_abbreviations_are_rejected(cli: str) -> None:
    # A prefix such as --output-dir must not silently match a longer flag.
    assert PARSERS[cli]().allow_abbrev is False


def test_sigterm_exits_143_through_finally_blocks() -> None:
    previous = signal.getsignal(signal.SIGTERM)
    cleaned = []

    @exits_on_sigterm
    def main() -> int:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            pytest.fail("the SIGTERM handler did not run")
        finally:
            cleaned.append(True)

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 143
    assert cleaned == [True]
    # The handler is the CLI's only while it runs.
    assert signal.getsignal(signal.SIGTERM) is previous


@pytest.mark.parametrize(
    "main",
    [
        run_ingestion_pipeline.main,
        run_rag_chain.main,
        run_evaluation.main,
        run_visualization.main,
        run_prompt_tuning.main,
        prepare_benchmark.main,
    ],
)
def test_every_cli_main_handles_sigterm(main: Callable[..., object]) -> None:
    assert getattr(main, "__wrapped__", None) is not None


def test_ingestion_rejects_json_with_the_cloudwatch_metrics_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CloudWatch EMF records go to stdout, where the JSON result goes.
    monkeypatch.setattr(
        sys,
        "argv",
        ["run-ingestion", "--output-format", "json", "--metrics-sink", "cloudwatch"],
    )
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.CommandLineInterface().parse_args()
    assert exc.value.code == 2


def test_ingestion_json_result_lists_each_stage(
    config: Config, mocker, tmp_path: Path
) -> None:
    mocker.patch.object(run_ingestion_pipeline, "get_config", return_value=config)
    src = tmp_path / "src"
    src.mkdir()
    args = run_ingestion_pipeline.CommandLineInterface._setup_arguments().parse_args(
        ["--source-directory", str(src), "--output-format", "json"]
    )
    runner = run_ingestion_pipeline.IngestionPipelineRunner(args)
    runner.pipeline = mocker.MagicMock(executed_stages=frozenset({"text_chunking"}))
    context = PipelineContext(
        pipeline_id="run-1",
        config=PipelineConfig(),
        status=PipelineStageStatus.FAILED,
        start_time=datetime.now(),
        source_directory=src,
        stage_results=[
            PipelineStageResult(
                stage_name="text_chunking",
                status=PipelineStageStatus.FAILED,
                start_time=datetime.now(),
                error_message="boom",
            )
        ],
    )
    result = json.loads(json.dumps(runner._json_result(context)))
    assert result["pipeline_id"] == "run-1"
    assert result["status"] == "failed"
    assert result["stages"] == [
        {
            "stage_name": "text_chunking",
            "status": "failed",
            "duration_seconds": None,
            "input_count": 0,
            "output_count": 0,
            "error_message": "boom",
            "executed": True,
        }
    ]


async def test_eval_json_output_is_the_only_stdout(
    mocker, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config = Config()
    config.evaluation.enabled_evaluators = []
    mocker.patch.object(run_evaluation, "display_ascii_art")
    data = tmp_path / "eval.json"
    data.write_text(json.dumps([{"question": "Who ships?", "answer": "Vendor"}]))

    class _Chain:
        async def ainvoke(self, inputs, config=None):
            meta = {"search_strategy": "local", "processing_time": 0.1}
            return {"answer": "Vendor", "sources": [], "metadata": meta}

    args = run_evaluation.CommandLineInterface._setup_arguments().parse_args(
        [
            "--data-path",
            str(data),
            "--output-directory",
            str(tmp_path / "out"),
            "--output-format",
            "json",
        ]
    )
    runner = run_evaluation.EvaluationRunner(args, _Chain(), config)  # type: ignore[arg-type]
    assert await runner.run() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["output_directory"] == str(tmp_path / "out")
    assert result["failure_budget_breaches"] == []
    assert [
        (r["strategy"], r["repeat"], r["total_queries"]) for r in result["runs"]
    ] == [("auto", 1, 1)]


def test_visualization_json_output_names_the_written_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = tmp_path / "empty.json"
    data.write_text(json.dumps({"nodes": [], "edges": []}), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-visualization",
            "--data-path",
            str(data),
            "--output-directory",
            str(tmp_path / "out"),
            "--renderers",
            "static",
            "--output-format",
            "json",
        ],
    )
    monkeypatch.setattr(run_visualization, "get_config", lambda _path=None: Config())
    assert run_visualization.main() == 1  # an empty graph renders nothing
    assert json.loads(capsys.readouterr().out) == {
        "output_directory": str(tmp_path / "out"),
        "files": [],
    }


def test_a_missing_input_file_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["run-visualization", "--data-path", str(tmp_path / "nope.json")]
    )
    monkeypatch.setattr(run_visualization, "get_config", lambda _path=None: Config())
    with pytest.raises(SystemExit) as exc:
        run_visualization.main()
    assert exc.value.code == 2
