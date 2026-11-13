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
prints ``{"status": "interrupted", "exit_code": 143, "error_type":
"Interrupted"}`` (or 130) instead, and one that fails before it printed one
``{"status": "failed", "exit_code": 1, "error_type": ...}`` (or 2) with the
``error`` it failed with (an argparse error included, see
:class:`ArgumentParser`), and ``run-ingestion`` the ``pipeline_id`` once
resolved. A result document that shows a failure (a failed ``run-rag``
query, a ``run-ingestion`` run a stage failed) carries the same keys
(:func:`failure_fields`).
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import contextlib
import functools
import ipaddress
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

from unified_kg_rag import __version__
from unified_kg_rag.application.cli.display import console
from unified_kg_rag.application.cli.launch import wants_json
from unified_kg_rag.shared import observability, setup_logging
from unified_kg_rag.shared.exceptions import (
    ErrorClass,
    classify,
    error_text,
    query_failure_ids,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")

EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 128 + signal.SIGINT  # 130, Ctrl-C
EXIT_TERMINATED = 128 + signal.SIGTERM  # 143

# The exit code of a query failure of each ErrorClass (run-rag, run-eval;
# docs/reference/exit-codes-and-json.md): a request or a setting to change
# (a filter key, an argument, a methodology or a suffix the namespace does
# not hold, a setting) exits 2, the rest 1.
EXIT_CODES: dict[ErrorClass, int] = {
    ErrorClass.INVALID_FILTER: EXIT_USAGE,
    ErrorClass.INVALID_ARGUMENT: EXIT_USAGE,
    ErrorClass.NOT_INDEXED: EXIT_USAGE,
    ErrorClass.INDEX_NOT_FOUND: EXIT_USAGE,
    ErrorClass.CONFIGURATION: EXIT_USAGE,
    ErrorClass.TIMEOUT: EXIT_FAILURE,
    ErrorClass.STORE_UNREACHABLE: EXIT_FAILURE,
    ErrorClass.MODEL_UNREACHABLE: EXIT_FAILURE,
    ErrorClass.FAILURE: EXIT_FAILURE,
}

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
_stop: dict[str, Any] = {
    "task": None,
    "signum": None,
    "interruptible": False,
    "line_printed": False,
}
# Set when a stop starts the grace period of the watchdog thread, and when
# a second signal ends it.
_stop_started = threading.Event()
_stop_now = threading.Event()
# Held while the one JSON document is printed: the watchdog may print the
# status from its own thread. Reentrant, as a signal handler may print the
# status on the main thread in the middle of print_json.
_print_lock = threading.RLock()


# The error_type of a failure without an exception: a usage error (an
# argparse error, a refused flag) or a run that failed.
USAGE_ERROR = "UsageError"
RUN_FAILED = "RunFailed"
INTERRUPTED = "Interrupted"


def usage_error(
    message: str | BaseException,
    error_type: str | None = None,
    *,
    error_class: ErrorClass | None = None,
) -> NoReturn:
    """Report a usage or configuration error as argparse does and exit 2.

    ``message`` is the text, or the exception the run refused, which then
    names the ``error_type`` (its class) and the ``error_class``
    (:func:`error_class_for`) of the JSON status unless they are given; a
    text is a :data:`USAGE_ERROR` of class ``invalid_argument`` by
    default."""
    text = error_text(message) if isinstance(message, BaseException) else message
    print(f"{Path(sys.argv[0]).name}: error: {text}", file=sys.stderr)
    print_status(EXIT_USAGE, message, error_type=error_type, error_class=error_class)
    raise SystemExit(EXIT_USAGE)


def error_type_of(exit_code: int, error: BaseException | str | None) -> str:
    """The ``error_type`` of a failure: the exception's class name, else
    :data:`USAGE_ERROR` for exit 2 and :data:`RUN_FAILED` otherwise."""
    if isinstance(error, BaseException) and not isinstance(error, SystemExit):
        return type(error).__name__
    return USAGE_ERROR if exit_code == EXIT_USAGE else RUN_FAILED


def error_class_for(exit_code: int, error: BaseException | str | None) -> ErrorClass:
    """The ``error_class`` of a failure: the exception's (:func:`classify`),
    else ``invalid_argument`` for exit 2 (a usage error) and ``failure``. An
    exception of no other class that a CLI refused as a usage error (an
    invalid input file) is ``invalid_argument`` too: the class agrees with
    the exit code (:data:`EXIT_CODES`)."""
    error_class = ErrorClass.FAILURE
    if isinstance(error, BaseException) and not isinstance(error, SystemExit):
        error_class = classify(error)
    if error_class is ErrorClass.FAILURE and exit_code == EXIT_USAGE:
        return ErrorClass.INVALID_ARGUMENT
    return error_class


def failure_fields(
    exit_code: int,
    error: BaseException | str | None,
    error_type: str | None = None,
    *,
    error_class: ErrorClass | None = None,
) -> dict[str, Any]:
    """The keys every JSON failure document has: ``status`` ``failed``, the
    ``exit_code``, the ``error`` message (when there is one), its
    ``error_class`` (:class:`ErrorClass`, the stable key a caller branches
    on: :func:`error_class_for` unless given) and ``error_type`` (the
    exception's class name, informational: :func:`error_type_of` unless
    given), and the ``query_id`` and ``trace_id`` of the query it failed,
    when it carries them (``query_failure_ids``). A CLI whose result
    document shows a failure merges them into it."""
    fields: dict[str, Any] = {"status": "failed", "exit_code": exit_code}
    if isinstance(error, str | Exception) and str(error):
        fields["error"] = error if isinstance(error, str) else error_text(error)
    fields["error_class"] = (error_class or error_class_for(exit_code, error)).value
    fields["error_type"] = error_type or error_type_of(exit_code, error)
    fields.update(query_failure_ids(error))
    return fields


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


def failure_rate(value: str) -> float:
    """A ``--max-failure-rate`` value: a fraction from 0.0 to 1.0."""
    rate = float(value)
    if not 0.0 <= rate <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0.0 and 1.0")
    return rate


def _same_file(a: Path, b: Path) -> bool:
    if a.exists() and b.exists():
        return a.samefile(b)
    return a.resolve() == b.resolve()


def _loaded_schema_file(config: Any) -> Path | None:
    """The local file the config's extraction schema source names (a plain
    path or a ``file://`` URL), or None for none, S3 or another scheme."""
    from unified_kg_rag.adapters.source_files.locations import resolve_source_path
    from unified_kg_rag.shared.exceptions import ConfigurationError

    source = config.processing.graph_extraction.schema_source.path
    if not source:
        return None
    try:
        resolved = resolve_source_path(source, refusal="")
    except ConfigurationError:
        return None
    return resolved if isinstance(resolved, Path) else None


def refuse_overwrites(
    outputs: list[tuple[str, Path | None]],
    *,
    force: bool,
    config_path: str | None,
    config: Any,
    remedy: str = "write the output to another file",
) -> None:
    """Exit 2 (:func:`usage_error`) before a CLI writes anything when one of
    its ``outputs`` (``(flag, path)``, a None path is not written) would
    replace an input of the run, the ``--config-path`` file or the schema
    file the config loads (always), names an existing file without
    ``--force``, or when two outputs name one file (``remedy`` ends the
    message of a refused input)."""
    written = [(flag, path) for flag, path in outputs if path is not None]
    inputs = [
        (Path(config_path) if config_path else None, "the --config-path file"),
        (
            _loaded_schema_file(config),
            "the schema file the config loads "
            "(processing.graph_extraction.schema_source)",
        ),
    ]
    for flag, path in written:
        for source, name in inputs:
            if source is not None and _same_file(path, source):
                usage_error(f"{flag} '{path}' is {name}; {remedy}")
    for i, (flag, path) in enumerate(written):
        for other, other_path in written[i + 1 :]:
            if _same_file(path, other_path):
                usage_error(f"{flag} and {other} name one file: '{path}'")
    for flag, path in written:
        if path.exists() and not force:
            usage_error(
                f"{flag} '{path}' exists; pass --force to replace it, or name "
                "another file"
            )


def is_loopback(host: str) -> bool:
    """Whether ``host`` names a loopback address (``127.0.0.0/8``, ``::1``,
    ``localhost``): reachable from this machine only."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def host_without_port(header: str) -> str:
    """The host of an HTTP ``Host`` header, without its port: ``[::1]:8501``
    and ``[::1]`` give ``[::1]``, ``127.0.0.2:8501`` gives ``127.0.0.2``."""
    header = header.strip().lower()
    if header.startswith("["):
        end = header.find("]")
        return header[: end + 1] if end != -1 else header
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


def add_removed_flag(parser: argparse.ArgumentParser, flag: str, reason: str) -> None:
    """Make ``flag``, removed from this release, a usage error (exit 2) that
    says why and what replaces it, instead of argparse's "unrecognized
    arguments"; it is left out of ``--help``. It takes no value: a value
    after it is then the unknown argument argparse names."""

    class _Removed(argparse.Action):
        def __call__(self, parser: Any, *_: Any) -> None:
            parser.error(f"{flag} was removed: {reason}")

    parser.add_argument(
        flag,
        nargs=0,
        action=_Removed,
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )


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
    ``KeyboardInterrupt`` could land in the loop's own code instead of the
    task (which, under the nest_asyncio patch the CLIs once applied, left
    ``run()`` waiting forever); its ``finally`` blocks run and
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
            # coroutine returns, the SystemExit could leave the loop's run()
            # waiting on a task it cancelled.
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
        _stop.update(task=None, signum=None, interruptible=False, line_printed=False)
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
                    _print_stop_line(_stop["signum"])
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


def _print_stop_line(signum: int) -> None:
    """In text mode, print the line saying which signal stopped the run,
    once: main's exit and the watchdog may both get to it."""
    with _print_lock:
        if _json["enabled"] or _stop["line_printed"]:
            return
        _stop["line_printed"] = True
        console.print(f"\n[yellow]Stopped by {signal.Signals(signum).name}.[/yellow]")


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
    signum = _stop["signum"] or signal.SIGTERM
    exit_code = 128 + signum
    # Not waiting long on a main thread stuck writing to stdout.
    if _print_lock.acquire(timeout=1.0):
        try:
            print_status(exit_code)
            _print_stop_line(signum)  # unless main's own exit printed it
        except Exception:  # noqa: BLE001  # nosec B110 - ending anyway
            pass
        finally:
            _print_lock.release()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001  # nosec B110 - ending anyway
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


LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def add_output_format_argument(parser: argparse.ArgumentParser) -> None:
    """Add the flags every CLI has: ``--output-format``, ``--log-level``
    and ``--version``."""
    parser.add_argument(
        "--output-format",
        default="text",
        choices=["text", "json"],
        help="'text' (default) for people; 'json' prints one JSON result to "
        "stdout (for a run that failed or was stopped before it, a status with "
        "the exit code and the error) and everything else to stderr",
    )
    add_log_level_and_version_arguments(parser)


def add_log_level_and_version_arguments(parser: argparse.ArgumentParser) -> None:
    """Add ``--log-level`` and ``--version`` (a CLI whose stdout is a
    protocol, run-mcp-server, takes these without ``--output-format``)."""
    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS,
        help="Log level of this run (default: logging.level, INFO; DEBUG logs "
        "hold user and corpus text). With --output-format text the logs are "
        "plain lines unless logging.log_format is set",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Print the version and exit",
    )


def setup_cli_logging(config: Any, args: argparse.Namespace) -> None:
    """``setup_logging`` for a CLI run: ``--log-level`` over
    ``logging.level``; with ``--output-format json`` the logs go to stderr,
    and in text mode, unless ``logging.log_format`` is set (a deployment
    sets ``structured``), they are plain lines, not JSON records between
    the CLI's own output."""
    overrides: dict[str, Any] = {}
    if level := getattr(args, "log_level", None):
        overrides["level"] = level
    as_json = getattr(args, "output_format", "text") == "json"
    if not as_json and "log_format" not in config.logging.model_fields_set:
        overrides["log_format"] = "plain"
    setup_logging(config, overrides or None, stream=sys.stderr if as_json else None)


def json_output(args: argparse.Namespace) -> bool:
    """Whether the run prints JSON; the console writes to stderr if so."""
    as_json = getattr(args, "output_format", "text") == "json"
    console.stderr = as_json
    _json["enabled"] = as_json
    return as_json


def print_json(result: Any) -> None:
    """Print ``result`` as the run's one JSON document on stdout, with the
    package ``version`` that wrote it (a key every document has)."""
    if isinstance(result, dict):
        result = {"version": __version__, **result}
    with _print_lock:
        _json["printed"] = True
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


def print_status(
    exit_code: int,
    error: BaseException | str | None = None,
    *,
    error_type: str | None = None,
    error_class: ErrorClass | None = None,
    **fields: Any,
) -> None:
    """Print the status document of a run that exits ``exit_code`` without
    having printed its result (once, and only with JSON output): a failed
    one names its ``error`` and ``error_type`` (:func:`failure_fields`), a
    stopped one has ``error_type`` ``Interrupted``; ``fields``
    (``pipeline_id``) are added."""
    with _print_lock:
        if not _json["enabled"] or _json["printed"]:
            return
        if exit_code in (EXIT_INTERRUPTED, EXIT_TERMINATED):
            status: dict[str, Any] = interrupted_fields(exit_code)
        else:
            status = failure_fields(
                exit_code, error, error_type, error_class=error_class
            )
        print_json({**status, **fields})


def interrupted_fields(exit_code: int) -> dict[str, Any]:
    """The keys of a stopped run's JSON status."""
    return {"status": "interrupted", "exit_code": exit_code, "error_type": INTERRUPTED}


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
