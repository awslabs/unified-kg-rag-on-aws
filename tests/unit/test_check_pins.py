# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/check_pins.py, the weekly check of the pins Dependabot does not move.

Its patterns must keep matching the files that hold the pins (a pin moved or
reformatted would otherwise be checked no more), and each kind of drift
must be reported. The upstream lookups are replaced: no network.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_pins.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_pins", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_pins = _load()


def test_the_pins_are_read_from_the_files_that_hold_them() -> None:
    pins = check_pins.read_pins()
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    compute = (REPO_ROOT / "iac" / "stacks" / "compute_stack.py").read_text(
        encoding="utf-8"
    )
    assert pins.base_ref == "public.ecr.aws/docker/library/python:3.12-slim"
    assert f"{pins.base_ref}@{pins.base_digest}" in dockerfile
    assert f"{pins.uv_repo}:{pins.uv_version}@{pins.uv_digest}" in dockerfile
    assert f'COLLECTOR_VERSION = "{pins.collector_version}"' in compute
    assert pins.collector_digest in compute
    assert re.fullmatch(r"\d+\.\d+\.\d+", pins.cdk_cli_version)
    # The CDK CLI CI installs is the one the IaC README names.
    readme = (REPO_ROOT / "iac" / "README.md").read_text(encoding="utf-8")
    assert f"aws-cdk@{pins.cdk_cli_version}" in readme


def _lookups(pins: Any, **changes: str) -> Any:
    """Lookups answering what the repository pins, except ``changes``."""
    collector = f"{check_pins._COLLECTOR_REPOSITORY}:{pins.collector_version}"
    digests = {
        pins.base_ref: changes.get("base", pins.base_digest),
        f"{pins.uv_repo}:{pins.uv_version}": changes.get("uv_digest", pins.uv_digest),
        collector: changes.get("collector_digest", pins.collector_digest),
    }
    releases = {
        check_pins.UV_RELEASES: changes.get("uv", pins.uv_version),
        check_pins.ADOT_RELEASES: changes.get("adot", pins.collector_version),
    }
    npm = {check_pins.CDK_CLI_LATEST: changes.get("cdk", pins.cdk_cli_version)}
    return check_pins.Lookups(
        digest=digests.__getitem__,
        latest_release=releases.__getitem__,
        latest_npm=npm.__getitem__,
    )


def test_current_pins_report_nothing() -> None:
    pins = check_pins.read_pins()
    assert check_pins.check(pins, _lookups(pins)) == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"base": "sha256:" + "1" * 64}, "Base image"),
        ({"uv": "99.0.0"}, "uv 99.0.0 is released"),
        ({"uv_digest": "sha256:" + "2" * 64}, "the tag was re-pushed"),
        ({"adot": "v9.0.0"}, "ADOT collector v9.0.0 is released"),
        ({"collector_digest": "sha256:" + "3" * 64}, "the tag was re-pushed"),
        ({"cdk": "9.0.0"}, "AWS CDK CLI 9.0.0 is released"),
    ],
)
def test_each_drift_is_reported(change: dict[str, str], expected: str) -> None:
    pins = check_pins.read_pins()
    (line,) = check_pins.check(pins, _lookups(pins, **change))
    assert expected in line


def test_an_older_upstream_release_is_not_drift() -> None:
    # A "latest" release behind the pin (a yanked one) reports nothing.
    pins = check_pins.read_pins()
    assert check_pins.check(pins, _lookups(pins, uv="0.0.1", cdk="1.0.0")) == []


def test_the_workflow_runs_the_script_weekly_and_files_one_issue() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/pin-freshness.yml").read_text(encoding="utf-8")
    )
    # PyYAML reads the `on:` key as True.
    assert workflow[True]["schedule"]
    (job,) = workflow["jobs"].values()
    assert job["permissions"] == {"contents": "read", "issues": "write"}
    runs = " ".join(step.get("run", "") for step in job["steps"])
    assert "scripts/check_pins.py --report" in runs
    assert "gh issue edit" in runs and "gh issue create" in runs
