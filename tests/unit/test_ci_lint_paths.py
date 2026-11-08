# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The quality job lints every tracked Python file.

ruff, black and isort ran on unified_kg_rag, tests, iac and scripts only,
so the console demo, the image's S3 sync script and the workflow scripts
were never checked.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess  # nosec B404 - runs git on a fixed argv
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "quality.yml"


def _lint_paths(tool: str) -> list[str]:
    steps = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))["jobs"]["quality"][
        "steps"
    ]
    (run,) = [s["run"] for s in steps if s.get("run", "").startswith(f"uv run {tool} ")]
    return [arg for arg in shlex.split(run)[3:] if not arg.startswith("-")]


@pytest.mark.skipif(
    shutil.which("git") is None or not (REPO_ROOT / ".git").exists(),
    reason="needs a git checkout",
)
def test_every_tracked_python_file_is_linted() -> None:
    tracked = subprocess.run(  # nosec B603 B607 - fixed argv
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert tracked
    for tool in ("ruff", "black", "isort"):
        paths = _lint_paths(tool)
        unlinted = [
            f
            for f in tracked
            if not any(f == p or f.startswith(f"{p}/") for p in paths)
        ]
        assert unlinted == [], f"{tool} skips {unlinted}"
