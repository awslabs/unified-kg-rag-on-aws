# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conventions every CLI follows: exit codes and machine-readable output.

Exit codes:

- ``0``: success.
- ``1`` (:data:`EXIT_FAILURE`): the run failed (a failed stage, query or
  upload, a breached evaluation failure budget, nothing rendered).
- ``2`` (:data:`EXIT_USAGE`): the command line or the configuration is
  wrong (argparse errors, an invalid flag combination, a missing input, an
  invalid configuration file, an unset store endpoint). Nothing was run.
- ``128 + signal``: stopped by a signal, ``130`` for Ctrl-C (SIGINT) and
  ``143`` for SIGTERM (ECS ``StopTask``, a Step Functions timeout).

``--output-format json`` prints one JSON document with the run's result to
stdout and sends everything else (banner, progress, logs) to stderr, so the
output pipes into ``jq``. A run stopped by a signal before it printed one
prints ``{"status": "interrupted", "exit_code": 143}`` (or 130) instead.
"""

from __future__ import annotations

import argparse
import functools
import json
import signal
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn, ParamSpec, TypeVar

from unified_kg_rag.shared.utils.display import console

_P = ParamSpec("_P")
_R = TypeVar("_R")

EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 128 + signal.SIGINT  # 130, Ctrl-C
EXIT_TERMINATED = 128 + signal.SIGTERM  # 143

# Whether this process prints JSON, and whether it printed its document.
_json = {"enabled": False, "printed": False}


def usage_error(message: str) -> NoReturn:
    """Report a usage or configuration error as argparse does and exit 2."""
    print(f"{Path(sys.argv[0]).name}: error: {message}", file=sys.stderr)
    raise SystemExit(EXIT_USAGE)


def exits_on_sigterm(main: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorate a CLI ``main``: SIGTERM exits it with ``128 + SIGTERM`` (143),
    running ``finally`` blocks on the way out.

    In the container the CLI is PID 1 (the entrypoint execs it), and the
    kernel does not deliver a signal to PID 1 that it has no handler for, so
    without one a stopped task ran on until it was killed. The previous
    handler is restored when ``main`` returns; off the main thread, where
    Python cannot install a handler, ``main`` runs unchanged. With JSON
    output, a run stopped before it printed its document prints a status
    document, so a consumer parsing stdout gets JSON either way.
    """

    def _exit(signum: int, _frame: object) -> None:
        raise SystemExit(128 + signum)

    @functools.wraps(main)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if threading.current_thread() is not threading.main_thread():
            return main(*args, **kwargs)
        _json.update(enabled=False, printed=False)
        previous = signal.signal(signal.SIGTERM, _exit)
        try:
            return main(*args, **kwargs)
        except KeyboardInterrupt:
            _print_interrupted(EXIT_INTERRUPTED)
            raise
        except SystemExit as e:
            if e.code in (EXIT_INTERRUPTED, EXIT_TERMINATED):
                _print_interrupted(e.code)
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)

    return wrapper


def add_output_format_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-format",
        default="text",
        choices=["text", "json"],
        help="'text' (default) for people; 'json' prints one JSON result to "
        "stdout and everything else to stderr",
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


def _print_interrupted(exit_code: int) -> None:
    if _json["enabled"] and not _json["printed"]:
        print_json({"status": "interrupted", "exit_code": exit_code})
