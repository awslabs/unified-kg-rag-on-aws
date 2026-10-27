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
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

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
def _ctrl_c_raises():
    """Python's own SIGINT handler, whatever the suite inherited: under
    ``nohup`` (or ``cmd &`` in a script) SIGINT starts ignored, and the CLIs
    keep an inherited ignore (tested on its own)."""
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    yield
    signal.signal(signal.SIGINT, previous)


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


def test_a_sigterm_cancels_the_running_main_task_instead_of_raising() -> None:
    # Raised into the event loop, the SystemExit could land in the loop's
    # own code and leave nest_asyncio's run() waiting forever (2 of 150
    # stops mid-run). The stop now cancels the task at its next await.
    import asyncio

    from unified_kg_rag.application.cli.conventions import stoppable

    events: list[str] = []

    async def run() -> int:
        try:
            os.kill(os.getpid(), signal.SIGTERM)
            events.append("went on to the next await")
            await asyncio.sleep(5)
            events.append("not cancelled")
        except asyncio.CancelledError:
            events.append("cancelled")
            raise
        finally:
            await asyncio.sleep(0)  # an async aclose() still runs
            events.append("closed")
        return 0

    @exits_on_sigterm
    def main() -> int:
        return asyncio.run(stoppable(run()))

    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 143
    assert events == ["went on to the next await", "cancelled", "closed"]
    assert signal.getsignal(signal.SIGTERM) is previous


def test_a_ctrl_c_cancels_the_running_main_task_and_exits_130() -> None:
    # Python's own SIGINT handler raises KeyboardInterrupt into the loop,
    # as SIGTERM's SystemExit did: after the JSON 8 of 80 run-rag and 12 of
    # 120 run-eval Ctrl-Cs hung. The CLI's handler cancels the task too.
    import asyncio

    from unified_kg_rag.application.cli.conventions import stoppable

    events: list[str] = []

    async def run() -> int:
        try:
            os.kill(os.getpid(), signal.SIGINT)
            events.append("went on to the next await")
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            events.append("cancelled")
            raise
        finally:
            events.append("closed")
        return 0

    @exits_on_sigterm
    def main() -> int:
        return asyncio.run(stoppable(run()))

    previous = signal.getsignal(signal.SIGINT)
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 130
    assert events == ["went on to the next await", "cancelled", "closed"]
    assert signal.getsignal(signal.SIGINT) is previous


def test_a_cli_keeps_an_inherited_ignored_ctrl_c() -> None:
    # Started under nohup (or `cmd &` in a script), SIGINT is ignored: main
    # runs with it ignored, and it is ignored again after main.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    seen: list[object] = []

    @exits_on_sigterm
    def main() -> int:
        seen.append(signal.getsignal(signal.SIGINT))
        os.kill(os.getpid(), signal.SIGINT)  # ignored
        return 0

    assert main() == 0
    assert seen == [signal.SIG_IGN]
    assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN


def test_a_ctrl_c_outside_the_loop_is_the_clis_keyboardinterrupt() -> None:
    import asyncio

    from unified_kg_rag.application.cli.conventions import (
        interruptible,
        stoppable,
    )

    async def run() -> str:
        try:
            with interruptible():  # an interactive prompt
                os.kill(os.getpid(), signal.SIGINT)
        except KeyboardInterrupt:
            return "conversation ended"
        return "not raised"

    @exits_on_sigterm
    def main() -> str:
        return asyncio.run(stoppable(run()))

    assert main() == "conversation ended"


def test_a_finished_cli_ends_its_process_with_its_exit_code_at_once() -> None:
    # The interpreter's shutdown resets the signal handlers, so a SIGTERM
    # in it ended a finished run with -15 (85 of 160 run-rag stops after
    # the JSON). The process now ends once main and the exit handlers ran,
    # without waiting through the shutdown (a thread left running here).
    import time

    started = time.time()
    result = _run_cli_script("""
        atexit.register(lambda: print("exit handlers ran", flush=True))

        @exits_on_sigterm
        def main():
            threading.Thread(target=time.sleep, args=(30,)).start()
            return 0

        sys.exit(main())
        """)
    assert result.returncode == 0, result.stderr
    assert time.time() - started < 20
    assert "exit handlers ran" in result.stdout


def test_a_sigterm_at_an_interruptible_prompt_raises_at_once() -> None:
    # An interactive session waits in a blocking prompt on the loop's
    # thread, so a cancel would wait for the next line (until the watchdog
    # ended it, 10 s later).
    import asyncio

    from unified_kg_rag.application.cli.conventions import (
        interruptible,
        stoppable,
    )

    events: list[str] = []

    async def run() -> int:
        try:
            with interruptible():
                os.kill(os.getpid(), signal.SIGTERM)
                events.append("not raised")
        finally:
            events.append("closed")
        return 0

    @exits_on_sigterm
    def main() -> int:
        return asyncio.run(stoppable(run()))

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 143
    assert events == ["closed"]


def test_a_stopped_cli_that_does_not_unwind_is_ended_after_the_grace_period() -> None:
    # A run that swallowed the stop ran on until it was killed.
    import time

    started = time.time()
    result = _run_cli_script("""
        import argparse, asyncio
        from unified_kg_rag.application.cli import conventions
        from unified_kg_rag.application.cli.conventions import json_output, stoppable

        conventions._STOP_GRACE_SECONDS = 0.5

        async def run():
            os.kill(os.getpid(), signal.SIGTERM)
            while True:
                try:
                    await asyncio.sleep(0.01)
                except BaseException:
                    pass

        @exits_on_sigterm
        def main():
            json_output(argparse.Namespace(output_format="json"))
            asyncio.run(stoppable(run()))

        main()
        """)
    assert result.returncode == 143, result.stderr
    assert time.time() - started < 20
    assert json.loads(result.stdout) == {"status": "interrupted", "exit_code": 143}


def test_the_stop_grace_leaves_cleanup_time_inside_the_stop_timeout() -> None:
    # 10 s cut run-eval's upload of the finished runs, which the 120 s
    # StopTimeout of the app container (iac compute stack) leaves time for.
    from unified_kg_rag.application.cli import conventions

    assert 60 <= conventions._STOP_GRACE_SECONDS < 120


def test_a_second_signal_ends_a_stopping_cli_at_once() -> None:
    # Ignored, it left the process to the full grace period.
    import time

    started = time.time()
    result = _run_cli_script("""
        import argparse, asyncio
        from unified_kg_rag.application.cli import conventions
        from unified_kg_rag.application.cli.conventions import json_output, stoppable

        conventions._STOP_GRACE_SECONDS = 40.0

        async def run():
            os.kill(os.getpid(), signal.SIGTERM)
            try:
                await asyncio.sleep(30)
            finally:
                threading.Timer(0.2, os.kill, (os.getpid(), signal.SIGINT)).start()
                time.sleep(30)  # a cleanup that hangs

        @exits_on_sigterm
        def main():
            json_output(argparse.Namespace(output_format="json"))
            asyncio.run(stoppable(run()))

        main()
        """)
    assert result.returncode == 143, result.stderr
    assert time.time() - started < 20
    assert json.loads(result.stdout) == {"status": "interrupted", "exit_code": 143}


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_a_text_mode_stop_in_the_loop_says_so_once(signum: int) -> None:
    # The cancelled task exited 130 or 143 without a word.
    result = _run_cli_script(f"""
        import asyncio
        from unified_kg_rag.application.cli.conventions import stoppable

        signal.signal(signal.SIGINT, signal.default_int_handler)

        async def run():
            os.kill(os.getpid(), {int(signum)})
            await asyncio.sleep(30)

        @exits_on_sigterm
        def main():
            try:
                asyncio.run(stoppable(run()))
            except KeyboardInterrupt:
                print("interrupted by user")

        main()
        """)
    name = signal.Signals(signum).name
    assert result.returncode == 128 + signum, result.stderr
    assert result.stdout.count("Stopped by") == 1, result.stdout
    assert f"Stopped by {name}." in result.stdout
    assert "interrupted by user" not in result.stdout


def test_a_sigterm_after_the_result_is_printed_ends_the_process_at_once(
    capsys, monkeypatch
) -> None:
    # Raised into run-rag's event loop as its coroutine returned, the
    # SystemExit could leave nest_asyncio's run() waiting forever (5 of 23
    # stops right after the JSON hung until SIGKILL).
    from unified_kg_rag.application.cli import conventions

    monkeypatch.setattr(console, "stderr", console.stderr)
    monkeypatch.setitem(conventions._exit_at_once, "enabled", True)
    monkeypatch.setattr(conventions, "_exit_finished", lambda _code: None)
    events: list[object] = []
    monkeypatch.setattr(conventions, "_exit_stopped", events.append)

    @exits_on_sigterm
    def main() -> int:
        json_output(argparse.Namespace(output_format="json"))
        print_json({"ok": True})
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            events.append("unwound")
        return 0

    previous = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(SystemExit):
            main()
    finally:
        signal.signal(signal.SIGTERM, previous)
    # Ended from the handler, not after unwinding through the loop.
    assert events[:2] == [143, "unwound"]
    assert json.loads(capsys.readouterr().out) == {"ok": True}


def test_a_sigterm_after_main_returned_ends_the_process_with_143(
    capsys, monkeypatch
) -> None:
    # The default handler was restored as main returned, so a SIGTERM
    # while the process tore down (thread joins, exit handlers) ended it
    # with -15 after the success document.
    from unified_kg_rag.application.cli import conventions

    monkeypatch.setattr(console, "stderr", console.stderr)
    monkeypatch.setitem(conventions._exit_at_once, "enabled", True)
    monkeypatch.setattr(conventions, "_exit_finished", lambda _code: None)
    ended: list[int] = []
    monkeypatch.setattr(conventions, "_exit_stopped", ended.append)
    previous = signal.getsignal(signal.SIGTERM)

    @exits_on_sigterm
    def main() -> int:
        json_output(argparse.Namespace(output_format="json"))
        print_json({"ok": True})
        return 0

    try:
        assert main() == 0
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler) and handler is not previous
        handler(signal.SIGTERM, None)
    finally:
        signal.signal(signal.SIGTERM, previous)
    assert ended == [143]
    # One document: the result, not a status after it.
    assert json.loads(capsys.readouterr().out) == {"ok": True}


@pytest.mark.parametrize("output_format", ["json", "text"])
def test_a_sigterm_while_a_cli_loads_exits_143(
    tmp_path: Path, output_format: str
) -> None:
    # A SIGTERM during a CLI's imports (seconds) ended it with -15 and no
    # JSON: the console-script entry point installs a handler before them
    # (the real one, its import of the CLI stood in for).
    import subprocess

    script = (
        "import os, signal, sys\n"
        "from unified_kg_rag.application.cli import launch\n"
        "assert 'langchain_core' not in sys.modules and 'rich' not in sys.modules\n"
        "def loading(name):\n"
        "    os.kill(os.getpid(), signal.SIGTERM)\n"
        "    signal.pause()\n"
        "launch.importlib.import_module = loading\n"
        "launch.run_rag()\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, "--output-format", output_format],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=tmp_path,
    )
    assert done.returncode == 143, done.stderr
    if output_format == "json":
        assert json.loads(done.stdout) == {"status": "interrupted", "exit_code": 143}
    else:
        assert done.stdout == ""


def test_a_ctrl_c_while_a_cli_loads_exits_130() -> None:
    # Only SIGTERM was handled: a Ctrl-C during the imports printed a
    # traceback and, with --output-format json, no JSON.
    import subprocess

    script = (
        "import os, signal\n"
        "from unified_kg_rag.application.cli import launch\n"
        "def loading(name):\n"
        "    os.kill(os.getpid(), signal.SIGINT)\n"
        "    signal.pause()\n"
        "launch.importlib.import_module = loading\n"
        "launch.run_rag()\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, "--output-format", "json"],
        capture_output=True,
        text=True,
        timeout=60,
        # SIGINT handled, whatever the suite inherited (nohup ignores it).
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_DFL),
    )
    assert done.returncode == 130, done.stderr
    assert json.loads(done.stdout) == {"status": "interrupted", "exit_code": 130}


def test_a_ctrl_c_ignored_by_the_parent_stays_ignored_while_a_cli_loads() -> None:
    # Started with SIGINT ignored (`cmd &` in a non-interactive shell,
    # nohup), a CLI exited 130 on a SIGINT during its imports.
    import subprocess

    script = (
        "import os, signal, types\n"
        "from unified_kg_rag.application.cli import launch\n"
        "def loading(name):\n"
        "    os.kill(os.getpid(), signal.SIGINT)\n"
        "    return types.SimpleNamespace(main=lambda: print("
        "signal.getsignal(signal.SIGINT) is signal.SIG_IGN))\n"
        "launch.importlib.import_module = loading\n"
        "launch.run_rag()\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script, "--output-format", "json"],
        capture_output=True,
        text=True,
        timeout=60,
        preexec_fn=lambda: signal.signal(signal.SIGINT, signal.SIG_IGN),
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == "True\n"


def test_a_cli_main_gets_the_ctrl_c_handler_back(monkeypatch) -> None:
    # Once loaded, Ctrl-C raises KeyboardInterrupt in main as before.
    from types import SimpleNamespace

    from unified_kg_rag.application.cli import launch

    cli = SimpleNamespace(main=lambda: signal.getsignal(signal.SIGINT))
    monkeypatch.setattr(launch.importlib, "import_module", lambda _name: cli)
    previous = signal.getsignal(signal.SIGTERM)
    interrupt = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        # The handler main starts with is the one the process had, not
        # merely another than the launcher's.
        assert launch.run_rag() is signal.default_int_handler
    finally:
        signal.signal(signal.SIGTERM, previous)
        signal.signal(signal.SIGINT, interrupt)


@pytest.mark.parametrize(
    ("entry", "module"),
    [
        ("run_ingestion", "run_ingestion_pipeline"),
        ("run_rag", "run_rag_chain"),
        ("run_eval", "run_evaluation"),
        ("run_visualization", "run_visualization"),
        ("run_prompt_tuning", "run_prompt_tuning"),
        ("prepare_benchmark", "prepare_benchmark"),
    ],
)
def test_each_launcher_entry_runs_its_own_cli(
    monkeypatch: pytest.MonkeyPatch, entry: str, module: str
) -> None:
    from types import SimpleNamespace

    from unified_kg_rag.application.cli import launch

    loaded: list[str] = []

    def load(name: str) -> SimpleNamespace:
        loaded.append(name)
        return SimpleNamespace(main=lambda: "ran")

    monkeypatch.setattr(launch.importlib, "import_module", load)
    previous = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
    try:
        assert getattr(launch, entry)() == "ran"
    finally:
        signal.signal(signal.SIGTERM, previous[0])
        signal.signal(signal.SIGINT, previous[1])
    assert loaded == [f"unified_kg_rag.application.cli.{module}"]


def test_every_console_script_starts_through_the_launcher() -> None:
    import tomllib

    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    scripts = tomllib.loads(pyproject.read_text())["project"]["scripts"]
    from unified_kg_rag.application.cli import launch

    for name, target in scripts.items():
        module, _, function = target.partition(":")
        assert module == "unified_kg_rag.application.cli.launch", name
        assert callable(getattr(launch, function)), name


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
    # Named like the status of a run that fails before it prints.
    assert result["error"] == "Pipeline failed at stage(s): text_chunking"
    assert result["exit_code"] == 1
    assert result["failed_documents"] == 0
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


async def test_ingestion_json_result_counts_the_failed_documents() -> None:
    # The result named no failed document: a run that recorded some FAILED
    # read as a clean success. It counts the records written FAILED, not the
    # documents a stage failed on (a model outage records them PENDING).
    from unified_kg_rag.domain.models import Document, TextUnit

    documents = [
        Document(page_content=t, document_id=t, file_name=t, file_path=t)
        for t in ("a", "b", "c")
    ]
    units = [
        TextUnit(id=f"u{i}", text="x", document_ids=[doc])
        for i, doc in enumerate(["a", "a", "b", "c"])
    ]
    context = PipelineContext(
        pipeline_id="run-1",
        config=PipelineConfig(),
        status=PipelineStageStatus.COMPLETED,
        start_time=datetime.now(),
        documents=documents,
        text_units=units,
        source_directory=Path("src"),
        failed_text_unit_ids={"graph_extraction": ["u0", "u1"], "gleaning": ["u3"]},
    )
    context.record_failed_documents(["doc-a"])
    runner = run_ingestion_pipeline.IngestionPipelineRunner.__new__(
        run_ingestion_pipeline.IngestionPipelineRunner
    )
    runner.pipeline = MagicMock(executed_stages=frozenset())
    assert runner._json_result(context)["failed_documents"] == 1
    # Each recording adds to the run's; a document recorded twice counts once.
    context.record_failed_documents(["doc-c"])
    assert runner._json_result(context)["failed_documents"] == 2
    context.record_failed_documents(["doc-a"])
    assert context.recorded_failed_doc_ids == ["doc-a", "doc-c"]


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


def test_an_unreachable_store_exits_1_in_run_ingestion(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The pipeline raises it unwrapped now (the facade exports it): still a
    # failed run, exit 1, not an unexpected crash.
    from unified_kg_rag.shared import StoreConnectionError

    def unreachable(*_args: object, **_kwargs: object) -> None:
        raise StoreConnectionError("Failed to connect to OpenSearch.")

    monkeypatch.setattr(
        run_ingestion_pipeline.IngestionPipelineRunner, "__init__", unreachable
    )
    monkeypatch.setattr(sys, "argv", ["run-ingestion"])
    with pytest.raises(SystemExit) as exited:
        run_ingestion_pipeline.main()
    assert exited.value.code == 1
    out = capsys.readouterr().out
    assert "Failed to connect to OpenSearch" in out
    assert "unexpected" not in out


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


@pytest.mark.parametrize("waits", [False, True], ids=["no_wait", "next_run_waits"])
@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_interrupted_ingestion_json_names_the_pipeline_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    signum: int,
    waits: bool,
) -> None:
    # Before, the status gave no id to rerun with --pipeline-id, nor (only
    # the console message did) when the next run may write.
    import time
    from datetime import datetime
    from types import SimpleNamespace

    from unified_kg_rag.shared import PipelineInterrupted

    monkeypatch.setattr(console, "stderr", console.stderr)
    monkeypatch.setattr(
        run_ingestion_pipeline, "load_cli_config", lambda _args: Config()
    )

    until = float(int(time.time()) + 185) if waits else 0.0

    def _run(self: run_ingestion_pipeline.IngestionPipelineRunner) -> None:
        self.pipeline = SimpleNamespace(  # type: ignore[assignment]
            pipeline_id="run-7", next_run_writes_from=until
        )
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
    expected = {
        "status": "interrupted",
        "exit_code": 128 + signum,
        "pipeline_id": "run-7",
    }
    if waits:
        expected["next_run_writes_from"] = datetime.fromtimestamp(
            until, tz=UTC
        ).isoformat()
    assert json.loads(capsys.readouterr().out) == expected


def test_ingestion_stopped_after_its_result_prints_one_document(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A SIGTERM between the result and the end of pipeline.close() printed
    # an interrupted status after it, and told the user to resume a run
    # that had finished. The runner's own run() prints the result and
    # notes it (reported); only the pipeline is stood in for, its close()
    # stopped by the signal.
    from unified_kg_rag.shared import PipelineInterrupted

    monkeypatch.setattr(console, "stderr", console.stderr)
    monkeypatch.setattr(
        run_ingestion_pipeline, "load_cli_config", lambda _args: Config()
    )
    src = tmp_path / "src"
    src.mkdir()

    class _Pipeline:
        pipeline_id = "run-7"
        next_run_writes_from = 0.0
        executed_stages: frozenset[str] = frozenset()

        def __init__(self, **_kwargs: Any) -> None:
            pass

        def request_stop(self, signum: int) -> None:
            pass

        def close(self) -> None:
            raise PipelineInterrupted(signal.SIGTERM)

    runner_class = run_ingestion_pipeline.IngestionPipelineRunner
    monkeypatch.setattr(run_ingestion_pipeline, "DataIngestionPipeline", _Pipeline)
    monkeypatch.setattr(runner_class, "_require_indexing_endpoints", lambda *_: None)
    monkeypatch.setattr(runner_class, "_handle_metadata_operations", lambda _s: False)
    monkeypatch.setattr(runner_class, "_validate_pipeline_integrity", lambda _s: True)
    monkeypatch.setattr(
        runner_class,
        "_execute_pipeline",
        lambda _s: PipelineContext(
            pipeline_id="run-7",
            config=PipelineConfig(),
            status=PipelineStageStatus.COMPLETED,
            start_time=datetime.now(),
            source_directory=src,
        ),
    )
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
    assert exc.value.code == 143
    out, err = capsys.readouterr()
    result = json.loads(out)
    assert (result["pipeline_id"], result["status"]) == ("run-7", "completed")
    assert "--pipeline-id" not in err
    assert "after it finished" in err


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
    # on stdout in JSON mode, then without the error.
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
            "renderer crashed",
        ),
        (
            run_prompt_tuning.main,
            ["run-prompt-tuning", "--source-directory", str(empty), *json_flags],
            "No text documents found",
        ),
    ]
    for main, argv, error in cases:
        monkeypatch.setattr(sys, "argv", argv)
        assert main() == 1, argv[0]
        status = json.loads(capsys.readouterr().out)
        assert error in status.pop("error"), argv[0]
        assert status == {"status": "failed", "exit_code": 1}, argv[0]


def _json_status(main: Callable[[], object], argv: list[str], capsys: Any) -> Any:
    """The exit code of ``main`` run with ``argv`` and its JSON status."""
    sys.argv = argv
    try:
        code = main()  # run-visualization and run-prompt-tuning return it
    except SystemExit as exc:
        code = exc.code
    return code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(
    ("cli", "error", "code"),
    [
        ("run-rag", ConfigurationError("index built with the registry off"), 2),
        ("run-rag", RuntimeError("store unreachable"), 1),
        ("run-visualization", ConfigurationError("unknown renderer option"), 2),
        ("run-prompt-tuning", ConfigurationError("no tuning model set"), 2),
        ("run-prompt-tuning", RuntimeError("model unreachable"), 1),
    ],
)
def test_a_json_run_failing_while_running_names_the_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    cli: str,
    error: Exception,
    code: int,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(console, "stderr", console.stderr)
    monkeypatch.setattr(sys, "argv", sys.argv)
    monkeypatch.setattr(run_visualization, "load_cli_config", lambda _args: Config())
    monkeypatch.setattr(run_prompt_tuning, "load_cli_config", lambda _args: Config())
    monkeypatch.setattr(run_rag_chain.RAGChainRunner, "__init__", fail)
    monkeypatch.setattr(run_visualization, "run_visualization", fail)
    monkeypatch.setattr(run_prompt_tuning, "PromptTuner", fail)
    data = tmp_path / "graph.json"
    data.write_text("{}")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text("Vendor ships parts to Buyer.")
    main, argv = {
        "run-rag": (run_rag_chain.main, ["--query", "Who ships?"]),
        "run-visualization": (run_visualization.main, ["--data-path", str(data)]),
        "run-prompt-tuning": (
            run_prompt_tuning.main,
            ["--source-directory", str(corpus)],
        ),
    }[cli]
    got = _json_status(main, [cli, *argv, "--output-format", "json"], capsys)
    assert got == (code, {"status": "failed", "exit_code": code, "error": str(error)})


@pytest.mark.parametrize("cli", ["run-rag", "run-visualization", "run-prompt-tuning"])
def test_a_json_run_with_an_unknown_flag_names_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], cli: str
) -> None:
    # The CLIs' ArgumentParser exits through usage_error: a plain argparse
    # error printed a status without the error.
    monkeypatch.setattr(console, "stderr", console.stderr)
    monkeypatch.setattr(sys, "argv", sys.argv)
    main, argv = {
        "run-rag": (run_rag_chain.main, ["--query", "Who ships?"]),
        "run-visualization": (run_visualization.main, ["--data-path", "g.json"]),
        "run-prompt-tuning": (run_prompt_tuning.main, ["--source-directory", "c"]),
    }[cli]
    code, status = _json_status(
        main, [cli, *argv, "--bogus", "--output-format", "json"], capsys
    )
    assert code == 2
    assert status.pop("error") == "unrecognized arguments: --bogus"
    assert status == {"status": "failed", "exit_code": 2}


def test_an_invalid_visualization_data_file_names_the_error_in_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(console, "stderr", console.stderr)
    monkeypatch.setattr(sys, "argv", sys.argv)
    monkeypatch.setattr(run_visualization, "load_cli_config", lambda _args: Config())
    data = tmp_path / "bad.json"
    data.write_text("[]")
    argv = ["--data-path", str(data), "--output-directory", str(tmp_path / "out")]
    code, status = _json_status(
        run_visualization.main,
        ["run-visualization", *argv, "--output-format", "json"],
        capsys,
    )
    assert code == 2
    assert status.pop("error")
    assert status == {"status": "failed", "exit_code": 2}


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
    ("argv", "error"),
    [
        # An argparse error.
        (
            ["run-rag", "--output-format", "json", "--top-k", "0", "--query", "q"],
            "--top-k",
        ),
        # An unknown flag, and a parser.error call: no exception to name (the
        # status had no error).
        (
            ["run-ingestion", "--output-format", "json", "--no-such-flag"],
            "--no-such-flag",
        ),
        (["run-rag", "--output-format", "json"], "--query"),
        # A config file that fails to load, before json_output ran.
        (
            [
                "run-rag",
                "--output-format=json",
                "--query",
                "q",
                "--config-path",
                "nope.yaml",
            ],
            "nope.yaml",
        ),
        (
            ["run-eval", "--output-format", "json", "--config-path", "nope.yaml"],
            "nope.yaml",
        ),
        (
            ["run-ingestion", "--output-format", "json", "--config-path", "nope.yaml"],
            "nope.yaml",
        ),
    ],
)
def test_an_early_usage_error_still_prints_a_json_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
    error: str,
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
    status = json.loads(capsys.readouterr().out)
    # The error it exited for is named (it was not).
    assert error in status.pop("error")
    assert status == {"status": "failed", "exit_code": 2}


@pytest.mark.parametrize(
    ("error", "code"),
    [
        ("refused", 2),
        ("stage", 1),
        ("unexpected", 1),
    ],
)
def test_a_failed_ingestion_status_names_the_error_and_pipeline_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    error: str,
    code: int,
) -> None:
    # A run the store refused before any model call printed only
    # {"status": "failed", "exit_code": 2}: no reason and no run to resume.
    # A failed stage and an unexpected error name them too.
    from unified_kg_rag.shared.exceptions import (
        PipelineExecutionError,
        StoreRequestRejectedError,
    )

    raised = {
        "refused": StoreRequestRejectedError("unknown setting [index.refresh_intervl]"),
        "stage": PipelineExecutionError("stage 'indexing' failed"),
        "unexpected": RuntimeError("boom"),
    }[error]
    monkeypatch.setattr(console, "stderr", console.stderr)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "src").mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["run-ingestion", "--output-format", "json", "--source-directory", "src"],
    )

    def failed(self) -> None:
        self.pipeline = MagicMock(pipeline_id="run-7")
        raise raised

    monkeypatch.setattr(run_ingestion_pipeline.IngestionPipelineRunner, "run", failed)
    with pytest.raises(SystemExit) as exc:
        run_ingestion_pipeline.main()
    assert exc.value.code == code
    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "exit_code": code,
        "error": str(raised),
        "pipeline_id": "run-7",
    }


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


def test_a_stopped_cli_leaves_no_pool_worker_running(tmp_path: Path) -> None:
    # Old behaviour: os._exit left the abandoned process pool's workers
    # running on (a local kill; a container stops them with the task).
    import subprocess
    import textwrap
    import time

    pids = tmp_path / "workers"
    script = textwrap.dedent(f"""
        import os, signal, threading, time
        from concurrent.futures import ProcessPoolExecutor
        from unified_kg_rag.application.cli.conventions import exits_on_sigterm
        from unified_kg_rag.shared.utils.concurrency import shut_down_after

        @exits_on_sigterm
        def main():
            with shut_down_after(ProcessPoolExecutor(max_workers=2)) as ex:
                futures = [ex.submit(time.sleep, 30) for _ in range(4)]
                time.sleep(1)
                with open({str(pids)!r}, "w") as f:
                    f.write(" ".join(str(p) for p in ex._processes))
                os.kill(os.getpid(), signal.SIGTERM)
                futures[0].result()

        main()
    """)
    workers: list[int] = []
    try:
        # No pipes: a worker left running would hold them open.
        result = subprocess.run(
            [sys.executable, "-c", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
        workers = [int(pid) for pid in pids.read_text().split()]
        assert result.returncode == 143
        assert len(workers) == 2
        time.sleep(0.5)
        for pid in workers:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        for pid in workers:  # never leave one running, whatever failed
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_an_explicit_argv_decides_json_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # main(argv) is read for --output-format json, not sys.argv, and the
    # console moves to stderr before main runs, so an early error is JSON.
    monkeypatch.setattr(console, "stderr", False)
    monkeypatch.setattr(sys, "argv", ["run-rag"])
    seen: list[bool] = []

    @exits_on_sigterm
    def main(argv: list[str]) -> int:
        seen.append(console.stderr)
        raise SystemExit(2)

    with pytest.raises(SystemExit):
        main(["--output-format", "json"])
    assert seen == [True]
    assert json.loads(capsys.readouterr().out) == {"status": "failed", "exit_code": 2}


def _run_cli_script(body: str) -> Any:
    import subprocess
    import textwrap

    script = textwrap.dedent("""
        import atexit, logging, logging.handlers, os, signal, sys, threading, time
        from unified_kg_rag.application.cli.conventions import exits_on_sigterm
        from unified_kg_rag.shared import observability
        """) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=60
    )


def test_ctrl_c_ends_the_process_at_once_after_flushing() -> None:
    # The KeyboardInterrupt path exits at once too (130), and the tracing
    # backend and buffered log handlers are flushed before it does.
    import time

    started = time.time()
    result = _run_cli_script("""
        class Backend:
            def shutdown(self):
                print("tracing flushed", flush=True)

        observability.install(Backend())
        target = logging.StreamHandler(sys.stdout)
        logging.getLogger("t").addHandler(
            logging.handlers.MemoryHandler(100, flushLevel=100, target=target)
        )

        @exits_on_sigterm
        def main():
            logging.getLogger("t").warning("buffered log line")
            threading.Thread(target=time.sleep, args=(30,)).start()
            raise KeyboardInterrupt

        main()
        """)
    assert result.returncode == 130
    assert time.time() - started < 20
    assert "tracing flushed" in result.stdout
    assert "buffered log line" in result.stdout


def test_a_failed_cli_exits_normally() -> None:
    # Only a signal's exit code (above 128) ends the process at once: an
    # ordinary failure (exit 1) still runs the interpreter's exit handlers.
    result = _run_cli_script("""
        atexit.register(lambda: print("exit handlers ran", flush=True))

        @exits_on_sigterm
        def main():
            raise SystemExit(1)

        main()
        """)
    assert result.returncode == 1
    assert "exit handlers ran" in result.stdout


def test_a_stopped_ingestion_says_when_the_next_run_may_write() -> None:
    import time
    from types import SimpleNamespace

    from unified_kg_rag.application.cli.run_ingestion_pipeline import _next_run_wait

    waiting = SimpleNamespace(
        pipeline=SimpleNamespace(next_run_writes_from=time.time() + 185)
    )
    assert "waits until" in _next_run_wait(waiting)  # type: ignore[arg-type]
    done = SimpleNamespace(pipeline=SimpleNamespace(next_run_writes_from=0.0))
    assert _next_run_wait(done) == ""  # type: ignore[arg-type]
    assert _next_run_wait(None) == ""
