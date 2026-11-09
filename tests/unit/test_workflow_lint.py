# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The workflows are linted (actionlint) locally and in CI, pinned.

A broken expression, an unknown action input or a shell error in a
``run:`` script is found only when the workflow runs, on GitHub, often on
a schedule nobody watches (pin-freshness, fidelity). The pre-commit hook
lints them; the quality job runs every pre-commit hook it does not run
itself, so CI lints them with the same pin.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_actionlint_is_a_pinned_pre_commit_hook_ci_runs() -> None:
    config = yaml.safe_load(
        (REPO_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    )
    (repo,) = [r for r in config["repos"] if r["repo"].endswith("/actionlint")]
    assert re.fullmatch(r"[0-9a-f]{40}", repo["rev"]), "pin a commit, not a tag"
    assert [hook["id"] for hook in repo["hooks"]] == ["actionlint"]
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    )
    (step,) = [
        s
        for s in workflow["jobs"]["quality"]["steps"]
        if "pre-commit run --all-files" in s.get("run", "")
    ]
    skipped = step.get("env", {}).get("SKIP", "").split(",")
    assert "actionlint" not in skipped
