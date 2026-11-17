# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console-web job audits the npm lock with the accepted advisories.

``npm audit`` has no per-advisory ignore, so an advisory without a fixed
release in a devDependency the built client never bundles failed the job with
nothing to bump. .github/scripts/check_npm_audit.py gates the JSON report
against .github/npm-audit-accepted.txt, as the pip-audit job does.
"""

from __future__ import annotations

import datetime
import importlib.util
import io
import json
import re
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "check_npm_audit.py"
ACCEPTED = REPO_ROOT / ".github" / "npm-audit-accepted.txt"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_npm_audit", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_npm_audit = _load()


def _report(**advisories: str) -> dict[str, object]:
    """An npm audit report: package -> severity of an advisory against it."""
    vulnerabilities: dict[str, object] = {}
    for n, (name, severity) in enumerate(advisories.items()):
        vulnerabilities[name] = {
            "via": [
                {
                    "url": f"https://github.com/advisories/GHSA-aaaa-bbbb-000{n}",
                    "severity": severity,
                }
            ]
        }
        # A dependent reaches it through the package: not an advisory itself.
        vulnerabilities[f"{name}-dependent"] = {"via": [name]}
    return {"vulnerabilities": vulnerabilities, "metadata": {}}


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    report: object,
    accepted: str,
) -> int:
    path = tmp_path / "accepted.txt"
    path.write_text(accepted, encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(report)))
    code: int = check_npm_audit.main(["--accepted", str(path)])
    return code


def test_the_job_gates_the_report_on_the_accepted_list() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    )
    (step,) = [
        step
        for step in workflow["jobs"]["console-web"]["steps"]
        if "npm audit" in step.get("run", "")
    ]
    assert step["shell"] == "bash"  # pipefail: a failing npm audit fails the step
    assert "npm audit --json --audit-level=none" in step["run"]
    assert "check_npm_audit.py" in step["run"]
    assert ".github/npm-audit-accepted.txt" in step["run"]


def test_an_unaccepted_advisory_at_moderate_or_above_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = _report(acme="low", globex="moderate")
    assert _run(monkeypatch, tmp_path, report, "") == 1
    assert _run(monkeypatch, tmp_path, _report(acme="low"), "") == 0


def test_an_accepted_advisory_passes_until_its_date(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report = _report(acme="high")
    future = (datetime.date.today() + datetime.timedelta(days=30)).isoformat()
    past = (datetime.date.today() - datetime.timedelta(days=1)).isoformat()
    line = "GHSA-aaaa-bbbb-0000 {} acme 1.0.0 (a reason)\n"
    assert _run(monkeypatch, tmp_path, report, line.format(future)) == 0
    assert _run(monkeypatch, tmp_path, report, line.format(past)) == 1
    assert _run(monkeypatch, tmp_path, report, line.format("soon")) == 1


def test_output_without_an_audit_report_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # --audit-level=none exits 0 when the registry cannot be reached.
    error = {"message": "request failed", "error": {"summary": "", "detail": ""}}
    assert _run(monkeypatch, tmp_path, error, "") == 1


def test_each_accepted_advisory_names_its_package_and_reason() -> None:
    entries = [
        line
        for line in ACCEPTED.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    ids = []
    for line in entries:
        advisory, until, package, version, reason = line.split(" ", 4)
        assert re.fullmatch(r"GHSA-[0-9a-z]{4}-[0-9a-z]{4}-[0-9a-z]{4}", advisory)
        # At most a year out, so the reason is re-read.
        expiry = datetime.date.fromisoformat(until)
        assert expiry <= datetime.date.today() + datetime.timedelta(days=366), line
        assert re.fullmatch(r"\d+(\.\d+)*", version), line
        assert package and len(reason) > 40, line
        ids.append(advisory)
    assert len(ids) == len(set(ids))
    # The script reads every entry the test does.
    text = ACCEPTED.read_text(encoding="utf-8")
    assert check_npm_audit.accepted_ids(text, datetime.date.min) == set(ids)
