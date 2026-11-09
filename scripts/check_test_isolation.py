# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run each test file in a process of its own: none may need another first.

The suite runs every file in a few shared processes (``pytest -n``), so a
file that only passes after another one bound a lazy import, registered a
plugin or filled a cache passes there and fails when run alone, as a
developer runs the file they are changing. The ``test-isolation`` job of
``.github/workflows/quality.yml`` runs this over ``tests/unit``,
``tests/integration`` and ``tests/property``.

Usage::

    python scripts/check_test_isolation.py [DIRECTORY ...] [--jobs N] [--timeout S]

Exit status 0 when every file passes on its own (a file whose tests all
skip, pytest's exit 5, passes), 1 when one fails (each is named with the
end of its output). A file still running after ``--timeout`` seconds is
stopped and fails as timed out, named like any other, rather than hanging
the job until its own timeout names nothing.
"""

from __future__ import annotations

import argparse
import os
import subprocess  # nosec B404 - fixed argv, no shell
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# pytest's exit codes that pass: all tests passed, or none was collected (a
# module-level skip of an optional dependency).
_PASSING = (0, 5)
# Lines of a failing file's output shown.
_TAIL = 30
# The exit status reported for a file stopped at the timeout (timeout(1)'s).
TIMED_OUT = 124
# Seconds one file may run: well past the slowest file alone (about four
# minutes on a loaded host), well under the CI job's 25 minutes.
_DEFAULT_TIMEOUT = 600


def run_alone(path: Path, timeout: float = _DEFAULT_TIMEOUT) -> tuple[Path, int, str]:
    """``path``'s pytest exit code and output, in a fresh interpreter;
    :data:`TIMED_OUT` and what it printed when it ran past ``timeout``
    seconds (the process is killed)."""
    argv = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(path)]
    try:
        done = subprocess.run(  # nosec B603 - fixed argv, no shell
            argv, capture_output=True, text=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        printed = "".join(
            out.decode(errors="replace") if isinstance(out, bytes) else out
            for out in (e.stdout, e.stderr)
            if out
        )
        return path, TIMED_OUT, f"{printed}\ntimed out after {timeout:g} s"
    return path, done.returncode, done.stdout + done.stderr


def failures(
    directories: list[Path], jobs: int, timeout: float = _DEFAULT_TIMEOUT
) -> list[tuple[Path, int, str]]:
    """The files of ``directories`` that fail when run on their own."""
    files = sorted(p for d in directories for p in d.rglob("test_*.py"))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda path: run_alone(path, timeout), files))
    return [result for result in results if result[1] not in _PASSING]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "directories", nargs="*", type=Path, default=[Path("tests/unit")]
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    parser.add_argument(
        "--timeout",
        type=float,
        default=_DEFAULT_TIMEOUT,
        help="seconds one file may run before it fails as timed out",
    )
    args = parser.parse_args(argv)
    failed = failures(args.directories, args.jobs, args.timeout)
    for path, code, output in failed:
        tail = "\n".join(output.splitlines()[-_TAIL:])
        print(f"::error file={path}::fails when run alone (exit {code})\n{tail}\n")
    print(f"{len(failed)} test file(s) fail when run alone")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
