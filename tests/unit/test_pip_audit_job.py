# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The security workflow audits the locked Python dependencies.

No CI job scanned them for known vulnerabilities (ASH's grype and syft
cannot run there). Each advisory the job accepts is listed with its
package and the reason it does not apply.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
_ACCEPTED = REPO_ROOT / ".github" / "pip-audit-accepted.txt"


def _entries() -> list[str]:
    return [
        line
        for line in _ACCEPTED.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def test_the_job_audits_the_lock_with_the_accepted_advisories() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/security.yml").read_text(encoding="utf-8")
    )
    (run,) = [
        step["run"]
        for step in workflow["jobs"]["pip-audit"]["steps"]
        if "pip-audit==" in step.get("run", "")
    ]
    # Every extra, without the dev group, from the lock, hashed.
    assert "uv export --frozen --all-extras --no-dev" in run
    assert "--disable-pip" in run and "--no-deps" not in run
    assert ".github/pip-audit-accepted.txt" in run
    # An entry past its date is not ignored: the advisory gates again.
    assert '"$until" < "$today"' in run
    assert re.search(r"pip-audit==\d+\.\d+\.\d+", run)
    assert "--exclude-newer" in run


def test_each_accepted_advisory_names_its_package_and_reason() -> None:
    entries = _entries()
    assert entries
    ids = []
    for line in entries:
        advisory, until, package, version, reason = line.split(" ", 4)
        assert re.fullmatch(r"(PYSEC|GHSA|CVE)-[0-9A-Za-z-]+", advisory), line
        # Accepted until a date (past it the advisory gates again), at most
        # a year out, so the reason is re-read.
        expiry = datetime.date.fromisoformat(until)
        assert expiry <= datetime.date.today() + datetime.timedelta(days=366), line
        assert re.fullmatch(r"\d+(\.\d+)*", version), line
        assert package and len(reason) > 40, line
        ids.append(advisory)
    assert len(ids) == len(set(ids))
