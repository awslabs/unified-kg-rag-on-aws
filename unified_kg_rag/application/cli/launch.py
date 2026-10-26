# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console-script entry points (``[project.scripts]``).

Each installs a SIGTERM and a SIGINT handler, then imports its CLI. The
imports take seconds (LangChain, the model and store SDKs), and a SIGTERM in
that window ended the process with -15 (a Ctrl-C with a traceback) and, with
``--output-format json``, no JSON on stdout. Here it exits ``128 + signal``
(143, or 130 for SIGINT) and prints the status document the CLI would have
(:mod:`.conventions`). Standard library only, so the handlers are in place
before anything heavy loads; the SIGINT handler is restored before ``main``
runs, and the CLI's own handlers (``exits_on_sigterm``) then replace both.
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import sys
from typing import Any


def wants_json(argv: list[str]) -> bool:
    """Whether ``argv`` asks for ``--output-format json`` (read before the
    command line is parsed; ``conventions.json_output`` then decides)."""
    return any(
        arg == "--output-format=json"
        or (arg == "--output-format" and argv[i + 1 : i + 2] == ["json"])
        for i, arg in enumerate(argv)
    )


def _stopped_while_loading(signum: int, _frame: object) -> None:
    """End a CLI stopped before its ``main`` ran: nothing ran, so there is
    nothing to unwind or flush but the status document."""
    exit_code = 128 + signum
    if wants_json(sys.argv[1:]):
        status = {"status": "interrupted", "exit_code": exit_code}
        sys.stdout.write(json.dumps(status, indent=2) + "\n")
        sys.stdout.flush()
    os._exit(exit_code)


def _run(module: str) -> Any:
    signal.signal(signal.SIGTERM, _stopped_while_loading)
    interrupt = signal.getsignal(signal.SIGINT)
    # A SIGINT the parent ignores (`cmd &` in a script, nohup) stays ignored.
    if interrupt is not signal.SIG_IGN:
        signal.signal(signal.SIGINT, _stopped_while_loading)
    cli = importlib.import_module(f"unified_kg_rag.application.cli.{module}")
    # Ctrl-C raises KeyboardInterrupt again once main runs.
    signal.signal(signal.SIGINT, interrupt)
    return cli.main()


def run_ingestion() -> Any:
    return _run("run_ingestion_pipeline")


def run_rag() -> Any:
    return _run("run_rag_chain")


def run_eval() -> Any:
    return _run("run_evaluation")


def run_visualization() -> Any:
    return _run("run_visualization")


def run_prompt_tuning() -> Any:
    return _run("run_prompt_tuning")


def prepare_benchmark() -> Any:
    return _run("prepare_benchmark")
