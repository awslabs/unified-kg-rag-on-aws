# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
""".github/scripts/check_licenses.py over the console's npm lockfile.

The web client bundles its runtime npm dependencies; their licenses went
unchecked while the Python dependencies' were. The lockfile's own license
fields are read (no network), devDependencies left out.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "check_licenses.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_licenses", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_licenses = _load()


def _lock(tmp_path: Path, **packages: dict[str, object]) -> Path:
    lock = {
        "lockfileVersion": 3,
        "packages": {
            "": {"name": "console-web", "license": "Apache-2.0"},
            **{f"node_modules/{name}": entry for name, entry in packages.items()},
        },
    }
    path = tmp_path / "package-lock.json"
    path.write_text(json.dumps(lock), encoding="utf-8")
    return path


def test_runtime_packages_are_read_without_dev_or_links(tmp_path: Path) -> None:
    lock = json.loads(
        _lock(
            tmp_path,
            react={"version": "19.0.0", "license": "MIT"},
            vite={"version": "7.0.0", "license": "MIT", "dev": True},
            local={"link": True},
        ).read_text(encoding="utf-8")
    )
    lock["packages"]["node_modules/a/node_modules/b"] = {
        "version": "1.0.0",
        "license": "ISC",
    }
    assert check_licenses.npm_packages(lock) == [
        {"Name": "react", "Version": "19.0.0", "License": "MIT"},
        {"Name": "b", "Version": "1.0.0", "License": "ISC"},
    ]


@pytest.mark.parametrize(
    ("entry", "status"),
    [
        ({"version": "1.0.0", "license": "MIT"}, 0),
        ({"version": "1.0.0", "license": "GPL-3.0-only"}, 1),
        ({"version": "1.0.0"}, 1),  # unlabeled
        ({"version": "1.0.0", "license": "GPL-3.0-only", "dev": True}, 0),
    ],
    ids=["permissive", "copyleft", "unlabeled", "copyleft-dev-only"],
)
def test_a_bundled_copyleft_or_unlabeled_license_fails(
    tmp_path: Path, entry: dict[str, object], status: int
) -> None:
    lock = _lock(tmp_path, react={"version": "19.0.0", "license": "MIT"}, pkg=entry)
    assert check_licenses.main(["--npm-lock", str(lock)]) == status


def test_the_console_lockfile_passes_and_ci_checks_it() -> None:
    lock = REPO_ROOT / "console" / "web" / "package-lock.json"
    assert check_licenses.main(["--npm-lock", str(lock)]) == 0
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github/workflows/quality.yml").read_text(encoding="utf-8")
    )
    runs = [step.get("run", "") for step in workflow["jobs"]["console-web"]["steps"]]
    assert (
        "python3 ../../.github/scripts/check_licenses.py --npm-lock package-lock.json"
        in runs
    )
