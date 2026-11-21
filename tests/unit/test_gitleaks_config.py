# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The gitleaks allowlist and the image that applies it, in CI and locally.

The allowlist in ``.gitleaks.toml`` matches a line's text under one rule:
a line-numbered fingerprint (``.gitleaksignore``) went stale as soon as a
line was added above it, and a global allowlist would hide a real key of
any rule. The ``gitleaks`` job of ``security.yml`` and the command in
CONTRIBUTING.md run the same pinned image, so a local clean scan means a
clean job.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
_IMAGE = re.compile(r"zricethezav/gitleaks@sha256:[0-9a-f]{64}")


def test_the_allowlist_extends_the_default_rules_one_rule_at_a_time() -> None:
    config = tomllib.loads((REPO_ROOT / ".gitleaks.toml").read_text(encoding="utf-8"))
    assert config["extend"] == {"useDefault": True}
    assert set(config) == {"extend", "allowlists"}, "no rule of our own"
    assert config["allowlists"]
    for allowlist in config["allowlists"]:
        assert allowlist["targetRules"], "scope each allowlist to its rules"
        assert allowlist["regexTarget"] == "line"
        assert allowlist["regexes"]
        assert set(allowlist) <= {
            "description",
            "targetRules",
            "regexTarget",
            "regexes",
        }, "paths, commits or stop words would allow more than a line"
    assert not (REPO_ROOT / ".gitleaksignore").exists()


def test_ci_and_contributing_run_the_same_pinned_image() -> None:
    workflow_text = (REPO_ROOT / ".github/workflows/security.yml").read_text(
        encoding="utf-8"
    )
    job = yaml.safe_load(workflow_text)["jobs"]["gitleaks"]
    (step,) = [s for s in job["steps"] if "run" in s]
    assert "run dir ." in step["run"]
    assert "run git . --log-opts=" in step["run"]
    pinned = set(_IMAGE.findall(workflow_text))
    documented = set(
        _IMAGE.findall((REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8"))
    )
    assert len(pinned) == 1
    assert pinned == documented
