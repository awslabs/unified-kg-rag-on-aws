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
    # Every image scan runs the one pinned Trivy image.
    quality = (REPO_ROOT / ".github" / "workflows" / "quality.yml").read_text(
        encoding="utf-8"
    )
    trivy = f"{pins.trivy_repo}:{pins.trivy_version}@{pins.trivy_digest}"
    assert quality.count(trivy) == quality.count("aquasec/trivy:") >= 2
    # Each apt package the runtime stage installs, with its pin.
    install = dockerfile.split("apt-get install", 1)[1].split("rm -rf", 1)[0]
    assert [f"{name}={version}" for name, version in pins.apt_pins] == re.findall(
        r"[a-z0-9][a-z0-9.+-]*=\S+", install
    )
    assert {name for name, _ in pins.apt_pins} >= {"libgomp1", "zlib1g"}


def test_differing_trivy_pins_are_refused(tmp_path: Path) -> None:
    for relative in ("docker/Dockerfile", "iac/stacks/compute_stack.py"):
        (tmp_path / relative).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / relative).write_text(
            (REPO_ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8"
        )
    quality = (REPO_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    pinned = check_pins._TRIVY_IMAGE.search(quality).group(0)
    other = pinned.replace(pinned.split("@")[1], "sha256:" + "4" * 64)
    workflow = tmp_path / ".github/workflows/quality.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text(quality.replace(pinned, other, 1), encoding="utf-8")
    with pytest.raises(ValueError, match="different Trivy images"):
        check_pins.read_pins(tmp_path)


# The version the base image ships of an apt package the image upgrades.
_OLD = "0.9-1"


def _lookups(pins: Any, **changes: str) -> Any:
    """Lookups answering what the repository pins, except ``changes``: the
    base image's repositories offer each pinned apt version, which the base
    image has not installed (or ships an older version of)."""
    collector = f"{check_pins._COLLECTOR_REPOSITORY}:{pins.collector_version}"
    digests = {
        pins.base_ref: changes.get("base", pins.base_digest),
        f"{pins.uv_repo}:{pins.uv_version}": changes.get("uv_digest", pins.uv_digest),
        collector: changes.get("collector_digest", pins.collector_digest),
        f"{pins.trivy_repo}:{pins.trivy_version}": changes.get(
            "trivy_digest", pins.trivy_digest
        ),
    }
    releases = {
        check_pins.UV_RELEASES: changes.get("uv", pins.uv_version),
        check_pins.ADOT_RELEASES: changes.get("adot", pins.collector_version),
        check_pins.TRIVY_RELEASES: changes.get("trivy", f"v{pins.trivy_version}"),
    }
    npm = {check_pins.CDK_CLI_LATEST: changes.get("cdk", pins.cdk_cli_version)}

    def apt_policy(image: str, packages: tuple[str, ...]) -> dict[str, tuple[str, str]]:
        assert image == f"{pins.base_ref}@{pins.base_digest}"
        pinned = dict(pins.apt_pins)
        assert packages == tuple(pinned)
        return {
            name: (
                changes.get(f"{name}_installed", _OLD),
                changes.get(f"{name}_candidate", pinned[name]),
            )
            for name in packages
        }

    return check_pins.Lookups(
        digest=digests.__getitem__,
        latest_release=releases.__getitem__,
        latest_npm=npm.__getitem__,
        apt_policy=apt_policy,
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
        ({"trivy": "v9.0.0"}, "Trivy 9.0.0 is released"),
        ({"trivy_digest": "sha256:" + "5" * 64}, "the tag was re-pushed"),
        (
            {"zlib1g_candidate": "9:9.9-9"},
            "apt `zlib1g`: the base image's repositories now offer `9:9.9-9`",
        ),
        (
            {"libgomp1_candidate": "99-1"},
            "apt `libgomp1`: the base image's repositories now offer `99-1`",
        ),
    ],
)
def test_each_drift_is_reported(change: dict[str, str], expected: str) -> None:
    pins = check_pins.read_pins()
    (line,) = check_pins.check(pins, _lookups(pins, **change))
    assert expected in line


def test_a_base_image_that_ships_the_pinned_version_drops_the_pin() -> None:
    pins = check_pins.read_pins()
    zlib = dict(pins.apt_pins)["zlib1g"]
    (line,) = check_pins.check(pins, _lookups(pins, zlib1g_installed=zlib))
    assert "base image already ships the pinned" in line and "zlib1g" in line


def test_apt_cache_policy_output_is_read() -> None:
    text = (
        "libgomp1:\n  Installed: (none)\n  Candidate: 14.2.0-19\n"
        "  Version table:\n     14.2.0-19 500\n"
        "zlib1g:\n  Installed: 1:1.3-1+b1\n  Candidate: 1:1.3-1+deb13u1\n"
    )
    assert check_pins.parse_apt_policy(text, ("libgomp1", "zlib1g")) == {
        "libgomp1": ("(none)", "14.2.0-19"),
        "zlib1g": ("1:1.3-1+b1", "1:1.3-1+deb13u1"),
    }
    with pytest.raises(ValueError, match="no candidate"):
        check_pins.parse_apt_policy(text, ("libgomp1", "openssl"))


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
    # The weekly image scan runs the quality workflow's one Trivy pin.
    assert "docker build -f docker/Dockerfile" in runs
    assert "grep -oE 'aquasec/trivy:" in runs and "quality.yml" in runs
    assert "aquasec/trivy:0" not in runs


def test_only_the_pinned_sources_are_fetched() -> None:
    # The network lookups read four fixed URLs; anything else is refused
    # before a request is built, so the fetch cannot be pointed elsewhere.
    with pytest.raises(ValueError, match="not one of the pinned sources"):
        check_pins._get_json("https://example.com/latest")
    assert check_pins._ALLOWED_URLS == {
        check_pins.UV_RELEASES,
        check_pins.ADOT_RELEASES,
        check_pins.CDK_CLI_LATEST,
        check_pins.TRIVY_RELEASES,
    }
