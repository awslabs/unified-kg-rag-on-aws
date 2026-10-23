# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conventions every CLI follows: exit codes and machine-readable output.

Exit codes:

- ``0``: success.
- ``1`` (:data:`EXIT_FAILURE`): the run failed (a failed stage, query or
  upload, a breached evaluation failure budget, nothing rendered).
- ``2`` (:data:`EXIT_USAGE`): the command line or the configuration is
  wrong (argparse errors, an invalid flag combination, a missing input, an
  invalid configuration file, an unset store endpoint: nothing was run), or
  a ``ConfigurationError`` met while running.
- ``128 + signal``: stopped by a signal, ``130`` for Ctrl-C (SIGINT) and
  ``143`` for SIGTERM (ECS ``StopTask``, a Step Functions timeout).

``--output-format json`` prints one JSON document with the run's result to
stdout and sends everything else (banner, progress, logs) to stderr, so the
output pipes into ``jq``. A run stopped by a signal before it printed one
prints ``{"status": "interrupted", "exit_code": 143}`` (or 130) instead, and
one that fails before it printed one ``{"status": "failed", "exit_code": 1}``
(or 2) with the ``error`` it failed with, when it knows one (and
``run-ingestion`` the ``pipeline_id``, once resolved).
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, ParamSpec, TypeVar

from unified_kg_rag.shared import observability
from unified_kg_rag.shared.utils.display import console

_P = ParamSpec("_P")
_R = TypeVar("_R")

EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 128 + signal.SIGINT  # 130, Ctrl-C
EXIT_TERMINATED = 128 + signal.SIGTERM  # 143

# Whether this process prints JSON, and whether it printed its document.
_json = {"enabled": False, "printed": False}
# Whether a stopped CLI ends its process at once (see _exit_stopped); the
# test suite turns it off to see the SystemExit.
_exit_at_once = {"enabled": True}
# How long a stopped CLI waits for its terminated worker processes to end.
_CHILD_EXIT_SECONDS = 1.0


def usage_error(message: str) -> NoReturn:
    """Report a usage or configuration error as argparse does and exit 2."""
    print(f"{Path(sys.argv[0]).name}: error: {message}", file=sys.stderr)
    print_status(EXIT_USAGE, message)
    raise SystemExit(EXIT_USAGE)


def exits_on_sigterm(main: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorate a CLI ``main``: SIGTERM exits it with ``128 + SIGTERM`` (143),
    running ``finally`` blocks on the way out.

    In the container the CLI is PID 1 (the entrypoint execs it), and the
    kernel does not deliver a signal to PID 1 that it has no handler for, so
    without one a stopped task ran on until it was killed. The previous
    handler is restored when ``main`` returns; off the main thread, where
    Python cannot install a handler, ``main`` runs unchanged. With JSON
    output, a run stopped, or failed, before it printed its document prints a
    status document, so a consumer parsing stdout gets JSON either way. A
    run stopped by a signal (Ctrl-C included) then ends its process at once
    (:func:`_exit_stopped`), without waiting for the threads it abandoned.
    """

    def _exit(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    @functools.wraps(main)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if threading.current_thread() is not threading.main_thread():
            return main(*args, **kwargs)
        argv = next((a for a in args if isinstance(a, list)), sys.argv[1:])
        # Set from the command line before it is parsed, so an argparse
        # error or a config that fails to load still prints a JSON status.
        _json.update(enabled=_wants_json(argv), printed=False)
        if _json["enabled"]:
            console.stderr = True
        previous = signal.signal(signal.SIGTERM, _exit)
        try:
            result = main(*args, **kwargs)
            if isinstance(result, int) and result:  # mains that return the code
                print_status(result)
            return result
        except KeyboardInterrupt:
            print_status(EXIT_INTERRUPTED)
            _exit_stopped(EXIT_INTERRUPTED)
            raise
        except SystemExit as e:
            if isinstance(e.code, int) and e.code:
                # The error the CLI exited for, if it exited handling one.
                print_status(e.code, e.__context__)
                if e.code > 128:
                    _exit_stopped(e.code)
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)

    return wrapper


def _wants_json(argv: list[str]) -> bool:
    """Whether ``argv`` asks for ``--output-format json`` (read before the
    command line is parsed; :func:`json_output` then decides)."""
    return any(
        arg == "--output-format=json"
        or (arg == "--output-format" and argv[i + 1 : i + 2] == ["json"])
        for i, arg in enumerate(argv)
    )


def add_output_format_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-format",
        default="text",
        choices=["text", "json"],
        help="'text' (default) for people; 'json' prints one JSON result to "
        "stdout (for a run that failed or was stopped before it, a status with "
        "the exit code and the error) and everything else to stderr",
    )


def json_output(args: argparse.Namespace) -> bool:
    """Whether the run prints JSON; the console writes to stderr if so."""
    as_json = getattr(args, "output_format", "text") == "json"
    console.stderr = as_json
    _json["enabled"] = as_json
    return as_json


def print_json(result: Any) -> None:
    """Print ``result`` as the run's one JSON document on stdout."""
    _json["printed"] = True
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def print_status(
    exit_code: int, error: BaseException | str | None = None, **fields: Any
) -> None:
    """Print the status document of a run that exits ``exit_code`` without
    having printed its result (once, and only with JSON output): a failed
    one names its ``error``; ``fields`` (``pipeline_id``) are added."""
    if _json["enabled"] and not _json["printed"]:
        stopped = exit_code in (EXIT_INTERRUPTED, EXIT_TERMINATED)
        status: dict[str, Any] = {
            "status": "interrupted" if stopped else "failed",
            "exit_code": exit_code,
        }
        if not stopped and isinstance(error, str | Exception) and str(error):
            status["error"] = str(error)
        print_json({**status, **fields})


def _exit_stopped(exit_code: int) -> None:
    """End the process of a run stopped by a signal now, with ``exit_code``.

    A stop does not wait for the work in flight: the model calls and pool
    tasks it abandoned run on in threads the interpreter would join at exit,
    keeping the stopped process alive (past ECS's stop timeout, so it was
    killed) for as long as they take. The ``finally`` blocks of the run have
    run by now; the logs, the tracing backend and stdout are flushed, then
    the process exits without joining those threads. The worker processes of
    an abandoned process pool are terminated first: a container stops them
    with the task, but a local ``kill`` would leave them running on.
    """
    if not _exit_at_once["enabled"]:
        return
    # A pool worker inherits the CLI's SIGTERM handler, and the pool's task
    # wrapper catches the SystemExit it raises: kill what SIGTERM left.
    children = multiprocessing.active_children()
    for child in children:
        child.terminate()
    deadline = time.monotonic() + _CHILD_EXIT_SECONDS
    for child in children:
        child.join(max(0.0, deadline - time.monotonic()))
        if child.is_alive():
            child.kill()
    observability.uninstall()
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
