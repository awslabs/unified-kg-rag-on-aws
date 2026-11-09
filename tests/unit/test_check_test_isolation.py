# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/check_test_isolation.py fails a test file that passes only after
another file ran in its process, and CI runs it over tests/unit."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "check_test_isolation.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_test_isolation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.timeout(60)
def test_a_file_that_needs_another_first_fails(tmp_path: Path) -> None:
    script = _script()
    (tmp_path / "test_a_sets.py").write_text(
        "import builtins\n\n\ndef test_sets():\n    builtins.SHARED = 1\n"
    )
    # Passes after test_a_sets.py in one process, fails alone.
    (tmp_path / "test_b_needs.py").write_text(
        "import builtins\n\n\ndef test_needs():\n    assert builtins.SHARED == 1\n"
    )
    (tmp_path / "test_c_skips.py").write_text(
        "import pytest\n\npytest.importorskip('no_such_module_anywhere')\n"
    )
    failed = script.failures([tmp_path], jobs=3)
    assert [(path.name, code) for path, code, _ in failed] == [("test_b_needs.py", 1)]
    assert script.main([str(tmp_path), "--jobs", "2"]) == 1


def test_ci_runs_it_over_the_unit_tests() -> None:
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    )
    runs = [step.get("run", "") for step in workflow["jobs"]["test-isolation"]["steps"]]
    assert any("scripts/check_test_isolation.py tests/unit" in run for run in runs)
