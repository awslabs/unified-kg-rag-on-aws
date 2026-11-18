# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail on an npm advisory at moderate or above that is not accepted.

Reads ``npm audit --json --audit-level=none`` output on stdin and exits
non-zero when it reports an advisory at moderate severity or above whose id
``--accepted`` does not list with a date not yet passed. ``npm audit`` has no
per-advisory ignore, and with ``--audit-level=none`` it exits 0 even when the
registry cannot be reached, so output without an audit report fails too.

The accepted file holds one advisory per line, as
``.github/pip-audit-accepted.txt`` does: the id, the date it is accepted until
(YYYY-MM-DD; past it the advisory gates again), the package and its version,
and why it does not apply.

Usage::

    npm audit --json --audit-level=none \\
        | python check_npm_audit.py --accepted .github/npm-audit-accepted.txt
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

GATING = {"moderate", "high", "critical"}
_ENTRY = re.compile(r"^(GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}) (\S+) ")


def accepted_ids(text: str, today: datetime.date) -> set[str]:
    """The ids of the entries whose date has not passed.

    Raises:
        ValueError: An entry has no YYYY-MM-DD date after its id.
    """
    accepted = set()
    for line in text.splitlines():
        match = _ENTRY.match(line)
        if match is None:
            continue
        advisory, until = match.groups()
        try:
            expiry = datetime.date.fromisoformat(until)
        except ValueError as exc:
            raise ValueError(f"{advisory}: no YYYY-MM-DD date after the id") from exc
        if expiry < today:
            print(f"::warning::{advisory} was accepted until {until}: it gates again")
        else:
            accepted.add(advisory)
    return accepted


def advisories(report: dict[str, object]) -> dict[str, tuple[str, str]]:
    """Advisory id -> (package, severity), at moderate or above.

    Raises:
        ValueError: The output is not an audit report (e.g. a registry error).
    """
    vulnerabilities = report.get("vulnerabilities")
    if "error" in report or not isinstance(vulnerabilities, dict):
        raise ValueError(f"not an npm audit report: {report.get('message', report)}")
    found = {}
    for name, entry in vulnerabilities.items():
        # A dict in `via` is an advisory against this package; a string names
        # the vulnerable dependency it reaches it through.
        for via in entry.get("via", []):
            if isinstance(via, dict) and via.get("severity") in GATING:
                advisory = str(via.get("url", "")).rsplit("/", 1)[-1] or str(via)
                found[advisory] = (name, str(via["severity"]))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--accepted", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        accepted = accepted_ids(
            args.accepted.read_text(encoding="utf-8"), datetime.date.today()
        )
        found = advisories(json.load(sys.stdin))
    except ValueError as exc:
        print(f"::error::{exc}")
        return 1
    for advisory in sorted(accepted - found.keys()):
        print(f"{advisory} is accepted but no longer reported: remove its line")
    failing = {k: v for k, v in found.items() if k not in accepted}
    for advisory, (name, severity) in sorted(found.items()):
        status = "FAIL" if advisory in failing else "accepted"
        print(f"{status} {advisory} {name} ({severity})")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
