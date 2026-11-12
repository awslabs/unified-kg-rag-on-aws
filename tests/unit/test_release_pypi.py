# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The release workflow's PyPI job: prepared, off by default.

It uploads only on a tag push with the repository variable PYPI_PUBLISH set
to 'true', through trusted publishing (no stored token), in the pypi
environment, and only the files the build job hashed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def _jobs() -> dict:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    )
    assert workflow["permissions"] == {"contents": "read"}
    return workflow["jobs"]


def test_pypi_publishing_runs_only_on_a_tag_with_the_variable_set() -> None:
    job = _jobs()["publish-pypi"]
    condition = " ".join(job["if"].split())
    assert "github.event_name == 'push'" in condition
    assert "startsWith(github.ref, 'refs/tags/v')" in condition
    assert "vars.PYPI_PUBLISH == 'true'" in condition
    # After the GitHub release: PyPI gets what the release published.
    assert set(job["needs"]) == {"build", "github-release"}


def test_pypi_publishing_is_trusted_publishing_in_the_pypi_environment() -> None:
    job = _jobs()["publish-pypi"]
    assert job["permissions"] == {"id-token": "write"}
    assert job["environment"]["name"] == "pypi"
    steps = job["steps"]
    (publish,) = [s for s in steps if "gh-action-pypi-publish" in s.get("uses", "")]
    assert re.fullmatch(
        r"pypa/gh-action-pypi-publish@[0-9a-f]{40}", publish["uses"]
    ), "pin a commit"
    assert "password" not in publish.get("with", {})
    # The built files, checksum-verified before the upload.
    verify = steps.index(
        next(s for s in steps if "sha256sum --check" in s.get("run", ""))
    )
    assert verify < steps.index(publish)
    (download,) = [s for s in steps if "download-artifact" in s.get("uses", "")]
    assert download["with"]["name"] == "release-artifacts"
    assert publish["with"]["packages-dir"] == "dist/"
