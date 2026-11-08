# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run each test file in a process of its own: none may need another first.

The suite runs every file in a few shared processes (``pytest -n``), so a
file that only passes after another one bound a lazy import, registered a
plugin or filled a cache passes there and fails when run alone, as a
developer runs the file they are changing. The ``test-isolation`` job of
``.github/workflows/quality.yml`` runs this over ``tests/unit``.

Usage::

    python scripts/check_test_isolation.py [DIRECTORY ...] [--jobs N]

Exit status 0 when every file passes on its own (a file whose tests all
skip, pytest's exit 5, passes), 1 when one fails (each is named with the
end of its output).
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


def run_alone(path: Path) -> tuple[Path, int, str]:
    """``path``'s pytest exit code and output, in a fresh interpreter."""
    done = subprocess.run(  # nosec B603 - fixed argv, no shell
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return path, done.returncode, done.stdout + done.stderr


def failures(directories: list[Path], jobs: int) -> list[tuple[Path, int, str]]:
    """The files of ``directories`` that fail when run on their own."""
    files = sorted(p for d in directories for p in d.rglob("test_*.py"))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(run_alone, files))
    return [result for result in results if result[1] not in _PASSING]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "directories", nargs="*", type=Path, default=[Path("tests/unit")]
    )
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)
    failed = failures(args.directories, args.jobs)
    for path, code, output in failed:
        tail = "\n".join(output.splitlines()[-_TAIL:])
        print(f"::error file={path}::fails when run alone (exit {code})\n{tail}\n")
    print(f"{len(failed)} test file(s) fail when run alone")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
