# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conventions every CLI follows: exit codes and machine-readable output.

Exit codes:

- ``0``: success.
- ``1`` (:data:`EXIT_FAILURE`): the run failed (a failed stage, also one
  that met a configuration error, a failed query or upload, a breached
  evaluation failure budget, nothing rendered).
- ``2`` (:data:`EXIT_USAGE`): the command line or the configuration is
  wrong (argparse errors, an invalid flag combination, a missing input, an
  invalid configuration file, an unset store endpoint: nothing was run), or
  the run refused it before its first model call (a ``ConfigurationError``
  outside a stage: the store refused to create an index with the run's
  settings, or an incremental run's live indices were built for others).
- ``128 + signal``: stopped by a signal, ``130`` for Ctrl-C (SIGINT) and
  ``143`` for SIGTERM (ECS ``StopTask``, a Step Functions timeout).

``--output-format json`` prints one JSON document with the run's result to
stdout and sends everything else (banner, progress, logs) to stderr, so the
output pipes into ``jq``. A run stopped by a signal before it printed one
prints ``{"status": "interrupted", "exit_code": 143}`` (or 130) instead, and
one that fails before it printed one ``{"status": "failed", "exit_code": 1}``
(or 2) with the ``error`` it failed with (an argparse error included, see
:class:`ArgumentParser`), and ``run-ingestion`` the ``pipeline_id`` once
resolved.
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import contextlib
import functools
import json
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any, NoReturn, ParamSpec, TypeVar

from unified_kg_rag.application.cli.launch import wants_json
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
# How long a CLI stopped by a signal in its event loop may take to unwind
# (its finally blocks, run-eval's upload of the finished runs) before it is
# ended anyway: a cap on any hang on the way out, inside the 120 s StopTimeout
# the stack gives the app container (ECS's default is 30 s). A second signal
# ends it at once.
_STOP_GRACE_SECONDS = 100.0
# The task a stop cancels (see stoppable), the signal that stopped the
# run once one did, and whether the loop's thread waits in a blocking call
# (see interruptible).
_stop: dict[str, Any] = {"task": None, "signum": None, "interruptible": False}
# Set when a stop starts the grace period of the watchdog thread, and when
# a second signal ends it.
_stop_started = threading.Event()
_stop_now = threading.Event()
# Held while the one JSON document is printed: the watchdog may print the
# status from its own thread. Reentrant, as a signal handler may print the
# status on the main thread in the middle of print_json.
_print_lock = threading.RLock()


def usage_error(message: str) -> NoReturn:
    """Report a usage or configuration error as argparse does and exit 2."""
    print(f"{Path(sys.argv[0]).name}: error: {message}", file=sys.stderr)
    print_status(EXIT_USAGE, message)
    raise SystemExit(EXIT_USAGE)


def failed(exit_code: int, error: BaseException | str) -> int:
    """``exit_code``, for a ``main`` that returns it after logging ``error``:
    the JSON status names the error (:func:`print_status`)."""
    print_status(exit_code, error)
    return exit_code


class ArgumentParser(argparse.ArgumentParser):
    """``argparse.ArgumentParser`` whose errors (an unknown flag, a missing
    or invalid value) exit through :func:`usage_error`, so the JSON status
    names them."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        usage_error(message)


def exits_on_sigterm(main: Callable[_P, _R]) -> Callable[_P, _R]:
    """Decorate a CLI ``main``: SIGTERM exits it with ``128 + SIGTERM`` (143),
    running ``finally`` blocks on the way out, and Ctrl-C (SIGINT) with 130.

    In a container without an init process the CLI is PID 1 (the entrypoint
    execs it), and the kernel does not deliver a signal to PID 1 that it has
    no handler for, so without one a stopped task ran on until it was
    killed. Off the main thread, where Python cannot install a handler,
    ``main`` runs unchanged. With JSON output, a run stopped, or failed,
    before it printed its document prints a status document, so a consumer
    parsing stdout gets JSON either way. A run stopped by a signal then ends
    its process at once (:func:`_exit_stopped`), without waiting for the
    threads it abandoned.

    While an event loop runs the coroutine a CLI wrapped in :func:`stoppable`
    (outside an :func:`interruptible` call), either signal cancels that task
    instead of raising into the loop, where the ``SystemExit`` or
    ``KeyboardInterrupt`` could land in the loop's own code and leave
    nest_asyncio's ``run()`` waiting forever; its ``finally`` blocks run and
    ``main`` then exits ``128 + signal``. Elsewhere SIGTERM raises
    ``SystemExit`` and SIGINT ``KeyboardInterrupt``, as Python's own handler
    does (a SIGINT the parent ignores stays ignored). A watchdog thread ends
    a process stopped so (a SIGTERM, or a Ctrl-C in the loop) still running
    :data:`_STOP_GRACE_SECONDS` later, or at a second signal (with the status
    document, if none was printed), so no such stop can hang. A handler a
    CLI installs over these (``run-ingestion``'s checkpointed SIGTERM) and a
    Ctrl-C the CLI handles outside the loop are not watched.

    When ``main`` ends, the process ends with its exit code at once
    (:func:`_exit_finished`): the interpreter's shutdown resets the signal
    handlers, so a stop during it ended the process with -15 or -2 after the
    result. Until then a stop during the teardown ends it with
    ``128 + signal``. With the stop-at-once switch off (the test suite),
    ``main`` returns or raises and the previous handlers are restored.
    """

    def _stop_signal(signum: int, frame: object) -> None:
        if _json["printed"]:
            # The result is out: end now. Raised into an event loop as its
            # coroutine returns, the SystemExit could leave nest_asyncio's
            # run() waiting forever on a task it cancelled.
            _stopped_in_teardown(signum, frame)
            raise SystemExit(128 + signum)
        if _stop["signum"] is not None:
            # Already stopping: a second signal ends the process now.
            _stop_now.set()
            return
        task = _stop["task"]
        in_loop = (
            task is not None
            and task.get_loop().is_running()
            and not _stop["interruptible"]
        )
        if not in_loop and signum == signal.SIGINT:
            # Ctrl-C outside the loop (an interactive prompt, the CLI's
            # synchronous code): what the CLI handles.
            raise KeyboardInterrupt
        _stop["signum"] = signum
        if _exit_at_once["enabled"]:
            _stop_started.set()
        if not in_loop:
            raise SystemExit(128 + signum)
        # Cancel the main task on the loop; the wrapper exits 128 + signum
        # once main returns or raises.
        assert task is not None
        if not task.done():
            task.get_loop().call_soon_threadsafe(task.cancel)

    @functools.wraps(main)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if threading.current_thread() is not threading.main_thread():
            return main(*args, **kwargs)
        argv = next((a for a in args if isinstance(a, list)), sys.argv[1:])
        # Set from the command line before it is parsed, so an argparse
        # error or a config that fails to load still prints a JSON status.
        _json.update(enabled=wants_json(argv), printed=False)
        _stop.update(task=None, signum=None, interruptible=False)
        _start_stop_watchdog()
        if _json["enabled"]:
            console.stderr = True
        previous = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGTERM, signal.SIGINT)
            if signum == signal.SIGTERM
            or signal.getsignal(signum) is not signal.SIG_IGN
        }
        for signum in previous:
            signal.signal(signum, _stop_signal)
        code: int | str | None = 0
        exited: SystemExit | None = None
        try:
            try:
                result = main(*args, **kwargs)
            finally:
                if _stop["signum"] is not None:
                    # A stop the event loop took as a cancellation (main
                    # raised CancelledError) or one that came as main's
                    # coroutine ended. The JSON status says so; in text mode,
                    # this line (the CLI's own KeyboardInterrupt messages
                    # are for a Ctrl-C outside the loop, which sets no stop).
                    if not _json["enabled"]:
                        console.print(
                            "\n[yellow]Stopped by "
                            f"{signal.Signals(_stop['signum']).name}.[/yellow]"
                        )
                    raise SystemExit(128 + _stop["signum"])
            if isinstance(result, int) and result:  # mains that return the code
                print_status(result)
            code = result if isinstance(result, int) else 0
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
            if not _exit_at_once["enabled"]:
                raise
            code, exited = e.code, e  # a stop's code ended the process above
        finally:
            for signum, handler in previous.items():
                signal.signal(
                    signum,
                    _stopped_in_teardown if _exit_at_once["enabled"] else handler,
                )
        _exit_finished(code)
        if exited is not None:
            raise exited
        return result

    return wrapper


async def stoppable(main: Awaitable[_R]) -> _R:
    """Await ``main`` as the task a SIGTERM or Ctrl-C cancels: a CLI wraps
    the coroutine it runs (``asyncio.run(stoppable(...))``) under
    :func:`exits_on_sigterm`."""
    _stop["task"] = asyncio.current_task()
    return await main


@contextlib.contextmanager
def interruptible() -> Iterator[None]:
    """Mark a blocking call a coroutine makes on the loop's thread (an
    interactive prompt): a SIGTERM during it raises ``SystemExit`` there and
    a Ctrl-C ``KeyboardInterrupt``, as outside a loop, since it cannot land
    in the loop's code and a cancel would wait for the call to return."""
    _stop["interruptible"] = True
    try:
        yield
    finally:
        _stop["interruptible"] = False


def _stop_watchdog() -> None:
    """End the process a grace period after a stop, if it is still running,
    or at once at a second signal.

    Started before any stop (starting a thread in a signal handler could
    wait forever on a lock the interrupted code holds) and waits for it.
    """
    _stop_started.wait()
    _stop_now.wait(_STOP_GRACE_SECONDS)
    if not _exit_at_once["enabled"]:  # the test suite
        return
    exit_code = 128 + (_stop["signum"] or signal.SIGTERM)
    # Not waiting long on a main thread stuck writing to stdout.
    if _print_lock.acquire(timeout=1.0):
        try:
            print_status(exit_code)
        except Exception:  # noqa: BLE001, S110 - ending anyway
            pass
        finally:
            _print_lock.release()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001, S110 - ending anyway
            pass
    os._exit(exit_code)


def _start_stop_watchdog() -> None:
    """Start the watchdog thread (once per process) and disarm it."""
    _stop_started.clear()
    _stop_now.clear()
    if not _exit_at_once["enabled"] or any(
        t.name == "stop-watchdog" for t in threading.enumerate()
    ):
        return
    threading.Thread(target=_stop_watchdog, name="stop-watchdog", daemon=True).start()


def _exit_finished(code: int | str | None) -> None:
    """End the process of a CLI whose ``main`` finished, with ``code``.

    What the interpreter's exit would do first is done here: the tracing
    backend is shut down (its spans flushed) and the exit handlers run
    (``logging.shutdown`` among them); the threads left are not joined, as
    the work is done. The interpreter's shutdown then resets the signal
    handlers, so a SIGTERM during it ended the process with -15 after the
    result (most stops in the half second after it): ending now keeps the
    exit code. A no-op with the stop-at-once switch off.
    """
    if not _exit_at_once["enabled"]:
        return
    if isinstance(code, str):  # sys.exit("message"), as the interpreter does
        print(code, file=sys.stderr)
        code = EXIT_FAILURE
    observability.uninstall()
    atexit._run_exitfuncs()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code or 0)


def _stopped_in_teardown(signum: int, _frame: object) -> None:
    """End a CLI whose ``main`` has returned, stopped while it tears down."""
    print_status(128 + signum)
    _exit_stopped(128 + signum)


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
    with _print_lock:
        _json["printed"] = True
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def print_status(
    exit_code: int, error: BaseException | str | None = None, **fields: Any
) -> None:
    """Print the status document of a run that exits ``exit_code`` without
    having printed its result (once, and only with JSON output): a failed
    one names its ``error``; ``fields`` (``pipeline_id``) are added."""
    with _print_lock:
        if not _json["enabled"] or _json["printed"]:
            return
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
