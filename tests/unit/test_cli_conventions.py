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
from unified_kg_rag.application.cli.conventions import (
    exits_on_sigterm,
    json_output,
    print_json,
)
from unified_kg_rag.domain.models import (
    Config,
    PipelineConfig,
    PipelineContext,
    PipelineStageResult,
    PipelineStageStatus,
)
from unified_kg_rag.shared import ConfigurationError
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
    ("stop", "code"),
    [(lambda: os.kill(os.getpid(), signal.SIGTERM), 143), (KeyboardInterrupt, 130)],
)
def test_json_output_prints_a_status_when_stopped(
    capsys, monkeypatch, stop, code
) -> None:
    # Before, a stopped JSON run printed nothing a consumer could parse.
    monkeypatch.setattr(console, "stderr", console.stderr)

    @exits_on_sigterm
    def main() -> None:
        json_output(argparse.Namespace(output_format="json"))
        if isinstance(stop, type):
            raise stop
        stop()

    with pytest.raises((SystemExit, KeyboardInterrupt)):
        main()
    assert json.loads(capsys.readouterr().out) == {
        "status": "interrupted",
        "exit_code": code,
    }


def test_a_printed_json_result_is_not_followed_by_a_status(capsys, monkeypatch) -> None:
    monkeypatch.setattr(console, "stderr", console.stderr)

    @exits_on_sigterm
    def main() -> None:
        json_output(argparse.Namespace(output_format="json"))
        print_json({"done": True})
        raise SystemExit(143)

    with pytest.raises(SystemExit):
        main()
    assert json.loads(capsys.readouterr().out) == {"done": True}


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
    mocker.patch.object(run_ingestion_pipeline, "load_cli_config", return_value=config)
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
    # Each run reports its guardrail blocks (none here).
    assert result["runs"][0]["guardrail_blocked_counts"] == {}


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
    monkeypatch.setattr(run_visualization, "load_cli_config", lambda _args: Config())
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
    monkeypatch.setattr(run_visualization, "load_cli_config", lambda _args: Config())
    with pytest.raises(SystemExit) as exc:
        run_visualization.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("content", ["{bad", "[]", '{"nodes": [{}]}'])
def test_an_invalid_visualization_data_file_is_a_usage_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, content: str
) -> None:
    # Before, it exited 1, while run-eval exits 2 on an invalid dataset.
    data = tmp_path / "bad.json"
    data.write_text(content)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-visualization",
            "--data-path",
            str(data),
            "--output-directory",
            str(tmp_path / "out"),
        ],
    )
    monkeypatch.setattr(run_visualization, "load_cli_config", lambda _args: Config())
    assert run_visualization.main() == 2


def _raise_config_error(*_args: object, **_kwargs: object) -> None:
    raise ConfigurationError("index built with the registry off")


def test_a_configuration_error_while_running_exits_2_in_every_cli(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # run-eval exited 2 on a ConfigurationError met while running, the other
    # CLIs 1: one exit code for one kind of error.
    monkeypatch.setattr(run_visualization, "load_cli_config", lambda _args: Config())
    monkeypatch.setattr(run_prompt_tuning, "load_cli_config", lambda _args: Config())
    data = tmp_path / "graph.json"
    data.write_text("{}")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Vendor ships parts to Buyer.")
    monkeypatch.setattr(run_visualization, "run_visualization", _raise_config_error)
    monkeypatch.setattr(run_prompt_tuning, "PromptTuner", _raise_config_error)
    monkeypatch.setattr(run_rag_chain.RAGChainRunner, "__init__", _raise_config_error)
    monkeypatch.setattr(
        run_ingestion_pipeline.IngestionPipelineRunner, "__init__", _raise_config_error
    )
    cases = [
        (run_visualization.main, ["run-visualization", "--data-path", str(data)]),
        (
            run_prompt_tuning.main,
            [
                "run-prompt-tuning",
                "--source-directory",
                str(corpus),
                "--output-path",
                str(tmp_path / "tuned.yaml"),
            ],
        ),
        (run_rag_chain.main, ["run-rag", "--query", "Who ships parts?"]),
        (run_ingestion_pipeline.main, ["run-ingestion"]),
    ]
    for main, argv in cases:
        monkeypatch.setattr(sys, "argv", argv)
        try:
            code = main()  # run-visualization and run-prompt-tuning return it
        except SystemExit as exc:
            code = exc.code
        assert code == 2, argv[0]


def _raise_interrupt(*_args: object, **_kwargs: object) -> None:
    raise KeyboardInterrupt


def test_ctrl_c_exits_130_with_a_json_status_in_every_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Before, run-prompt-tuning and run-visualization returned 130 without
    # the JSON status, and run-rag died with a KeyboardInterrupt traceback.
    monkeypatch.setattr(console, "stderr", console.stderr)
    monkeypatch.setattr(run_visualization, "load_cli_config", lambda _args: Config())
    monkeypatch.setattr(run_prompt_tuning, "load_cli_config", lambda _args: Config())
    data = tmp_path / "graph.json"
    data.write_text("{}")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Vendor ships parts to Buyer.")
    monkeypatch.setattr(run_visualization, "run_visualization", _raise_interrupt)
    monkeypatch.setattr(run_prompt_tuning, "PromptTuner", _raise_interrupt)

    def _interrupted_run(coro: object) -> None:
        # The runner has started (and chosen JSON) when the loop, waiting on
        # I/O, gets the Ctrl-C.
        coro.close()  # type: ignore[attr-defined]
        json_output(argparse.Namespace(output_format="json"))
        raise KeyboardInterrupt

    monkeypatch.setattr(run_rag_chain.asyncio, "run", _interrupted_run)
    json_flags = ["--output-format", "json"]
    cases = [
        (
            run_visualization.main,
            ["run-visualization", "--data-path", str(data), *json_flags],
        ),
        (
            run_prompt_tuning.main,
            ["run-prompt-tuning", "--source-directory", str(corpus), *json_flags],
        ),
        (run_rag_chain.main, ["run-rag", "--query", "Who ships?", *json_flags]),
    ]
    for main, argv in cases:
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 130, argv[0]
        assert json.loads(capsys.readouterr().out) == {
            "status": "interrupted",
            "exit_code": 130,
        }, argv[0]


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_interrupted_ingestion_json_names_the_pipeline_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    signum: int,
) -> None:
    # Before, the status gave no id to rerun with --pipeline-id.
    from types import SimpleNamespace

    from unified_kg_rag.shared import PipelineInterrupted

    monkeypatch.setattr(console, "stderr", console.stderr)
    monkeypatch.setattr(
        run_ingestion_pipeline, "load_cli_config", lambda _args: Config()
    )

    def _run(self: run_ingestion_pipeline.IngestionPipelineRunner) -> None:
        self.pipeline = SimpleNamespace(pipeline_id="run-7")  # type: ignore[assignment]
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise PipelineInterrupted(signum)

    monkeypatch.setattr(run_ingestion_pipeline.IngestionPipelineRunner, "run", _run)
    src = tmp_path / "src"
    src.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run-ingestion",
            "--source-directory",
            str(src),
            "--cache-directory",
            str(tmp_path / "cache"),
            "--output-format",
            "json",
        ],
    )
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.main()
    assert exc.value.code == 128 + signum
    assert json.loads(capsys.readouterr().out) == {
        "status": "interrupted",
        "exit_code": 128 + signum,
        "pipeline_id": "run-7",
    }


@pytest.mark.parametrize("value", ["0", "-3"])
def test_prompt_tuning_rejects_a_non_positive_max_docs(value: str) -> None:
    # Before, --max-docs 0 was accepted and then sampled nothing.
    with pytest.raises(SystemExit) as exc:
        run_prompt_tuning._build_parser().parse_args(
            ["--source-directory", ".", "--max-docs", value]
        )
    assert exc.value.code == 2


def _raise_runtime_error(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("renderer crashed")


def test_a_failed_json_run_prints_a_failed_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Before, run-visualization and run-prompt-tuning exited 1 with nothing
    # on stdout in JSON mode.
    monkeypatch.setattr(console, "stderr", console.stderr)
    monkeypatch.setattr(run_visualization, "load_cli_config", lambda _args: Config())
    monkeypatch.setattr(run_prompt_tuning, "load_cli_config", lambda _args: Config())
    monkeypatch.setattr(run_visualization, "run_visualization", _raise_runtime_error)
    data = tmp_path / "graph.json"
    data.write_text("{}")
    empty = tmp_path / "empty"
    empty.mkdir()  # no documents: prompt tuning fails
    json_flags = ["--output-format", "json"]
    cases = [
        (
            run_visualization.main,
            ["run-visualization", "--data-path", str(data), *json_flags],
        ),
        (
            run_prompt_tuning.main,
            ["run-prompt-tuning", "--source-directory", str(empty), *json_flags],
        ),
    ]
    for main, argv in cases:
        monkeypatch.setattr(sys, "argv", argv)
        assert main() == 1, argv[0]
        assert json.loads(capsys.readouterr().out) == {
            "status": "failed",
            "exit_code": 1,
        }, argv[0]


def test_prepare_benchmark_ctrl_c_exits_130_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Before, a Ctrl-C escaped as a KeyboardInterrupt traceback.
    monkeypatch.setattr(console, "stderr", console.stderr)
    source = tmp_path / "dev.jsonl"
    source.write_text("")
    monkeypatch.setattr(prepare_benchmark.musique, "prepare", _raise_interrupt)
    with pytest.raises(SystemExit) as exc:
        prepare_benchmark.main(
            [
                "musique",
                "--output-directory",
                str(tmp_path / "out"),
                "--data-path",
                str(source),
                "--output-format",
                "json",
            ]
        )
    assert exc.value.code == 130
    assert json.loads(capsys.readouterr().out)["status"] == "interrupted"


@pytest.mark.parametrize(
    "line",
    [
        "[1, 2]",
        '{"id": "x"}',
        '{"id": "x", "question": "q", "answer": "a"}',
        # Wrong types (before, they reached the dataset as is).
        '{"id": "x", "question": 7, "answer": "a", "paragraphs": []}',
        '{"id": "x", "question": "q", "answer": "a", "paragraphs": [],'
        ' "answer_aliases": "a"}',
        '{"id": "x", "question": "q", "answer": "a", "paragraphs": [],'
        ' "answer_aliases": [1]}',
        # Before, these reached eval_data.json or were silently accepted.
        '{"id": "x", "question": "  ", "answer": "a", "paragraphs": []}',
        '{"id": "x", "question": "q", "answer": "a", "paragraphs": [],'
        ' "answerable": "false"}',
        '{"id": "x", "question": "q", "answer": "a",'
        ' "paragraphs": [{"title": "T", "paragraph_text": 3}]}',
        '{"id": "x", "question": "q", "answer": "a", "paragraphs": []}\n'
        '{"id": "x", "question": "q2", "answer": "b", "paragraphs": []}',
    ],
)
def test_a_malformed_musique_item_is_a_usage_error(tmp_path: Path, line: str) -> None:
    # Before, it crashed with a KeyError or AttributeError (exit 1).
    source = tmp_path / "dev.jsonl"
    source.write_text(line + "\n")
    with pytest.raises(SystemExit) as exc:
        prepare_benchmark.main(
            [
                "musique",
                "--output-directory",
                str(tmp_path / "out"),
                "--data-path",
                str(source),
            ]
        )
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        # An argparse error.
        ["run-rag", "--output-format", "json", "--top-k", "0", "--query", "q"],
        # A config file that fails to load, before json_output ran.
        [
            "run-rag",
            "--output-format=json",
            "--query",
            "q",
            "--config-path",
            "nope.yaml",
        ],
        ["run-eval", "--output-format", "json", "--config-path", "nope.yaml"],
        ["run-ingestion", "--output-format", "json", "--config-path", "nope.yaml"],
    ],
)
def test_an_early_usage_error_still_prints_a_json_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    # Before, stdout was empty: the error came before json_output(args).
    monkeypatch.setattr(console, "stderr", console.stderr)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    main = {
        "run-rag": run_rag_chain.main,
        "run-eval": run_evaluation.main,
        "run-ingestion": run_ingestion_pipeline.main,
    }[argv[0]]
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out) == {"status": "failed", "exit_code": 2}


def test_a_stopped_cli_exits_without_joining_the_threads_it_abandoned() -> None:
    # Old behaviour: the SystemExit(143) left the interpreter joining the
    # threads of the work the stop abandoned (a model call in flight), so a
    # stopped task lived on for as long as that call took.
    import subprocess
    import textwrap
    import time

    script = textwrap.dedent("""
        import json, os, signal, threading, time
        from unified_kg_rag.application.cli.conventions import (
            exits_on_sigterm, json_output, print_json,
        )
        import argparse

        @exits_on_sigterm
        def main():
            json_output(argparse.Namespace(output_format="json"))
            threading.Thread(target=time.sleep, args=(30,)).start()
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(5)

        main()
    """)
    started = time.time()
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 143
    assert time.time() - started < 20
    assert json.loads(result.stdout) == {"status": "interrupted", "exit_code": 143}


@pytest.mark.parametrize("code", [1, 2])
def test_a_json_run_that_exits_with_an_error_prints_a_failed_status(
    capsys, monkeypatch, code: int
) -> None:
    monkeypatch.setattr(console, "stderr", console.stderr)

    @exits_on_sigterm
    def main() -> None:
        json_output(argparse.Namespace(output_format="json"))
        raise SystemExit(code)

    with pytest.raises(SystemExit):
        main()
    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "exit_code": code,
    }


def test_a_stopped_cli_leaves_no_pool_worker_running() -> None:
    # Old behaviour: os._exit left the abandoned process pool's workers
    # running on (a local kill; a container stops them with the task).
    import subprocess
    import textwrap
    import time

    script = textwrap.dedent("""
        import os, signal, threading, time
        from concurrent.futures import ProcessPoolExecutor
        from unified_kg_rag.application.cli.conventions import exits_on_sigterm
        from unified_kg_rag.shared.utils.concurrency import shut_down_after

        @exits_on_sigterm
        def main():
            with shut_down_after(ProcessPoolExecutor(max_workers=2)) as ex:
                futures = [ex.submit(time.sleep, 30) for _ in range(4)]
                time.sleep(1)
                print(" ".join(str(p) for p in ex._processes), flush=True)
                os.kill(os.getpid(), signal.SIGTERM)
                futures[0].result()

        main()
    """)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 143
    workers = [int(pid) for pid in result.stdout.split()]
    assert len(workers) == 2
    time.sleep(0.5)
    for pid in workers:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
