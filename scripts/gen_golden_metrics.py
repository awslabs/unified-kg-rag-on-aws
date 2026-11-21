# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate tests/fixtures/golden_metrics.json, the golden-metrics baseline.

The golden-metrics gate (``tests/integration/test_golden_metrics.py``, the
``golden-metrics`` CI job) ingests the synthetic structural and bridge
corpora with deterministic fake models and fails when any score or answer
fingerprint of the held-out ``test`` half (or of an unsplit dataset)
differs from this baseline. Run this only to accept a change you meant: the diff it prints,
and the baseline's diff in the pull request, show every score that moved
(docs/guides/evaluate.md "Golden metrics").

Usage:
    uv run python scripts/gen_golden_metrics.py
"""

from __future__ import annotations

import os

# Nothing in the run calls AWS: an ambient instance role, profile or key
# must not be found if something tried.
for _name in (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
):
    os.environ.pop(_name, None)
os.environ.update(
    {
        "AWS_EC2_METADATA_DISABLED": "true",
        "AWS_CONFIG_FILE": os.devnull,
        "AWS_SHARED_CREDENTIALS_FILE": os.devnull,
    }
)

import argparse  # noqa: E402
import contextlib  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures import golden_metrics  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="golden-metrics-") as work:
        log = Path(work) / "run.log"
        try:
            # The runs' progress and tables go to a log, shown on failure.
            with log.open("w", encoding="utf-8") as out:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                    measured = golden_metrics.measure(Path(work))
        except Exception:
            print(log.read_text(encoding="utf-8")[-20_000:], file=sys.stderr)
            raise
    path = golden_metrics.BASELINE
    if path.exists():
        gated, reported = golden_metrics.compare(
            golden_metrics.read_baseline(path), measured
        )
        for line in gated:
            print(f"gated: {line}")
        for line in reported:
            print(f"tune: {line}")
        if not gated and not reported:
            print("No score changed.")
    golden_metrics.write_baseline(measured, path)
    print(f"Wrote {path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
