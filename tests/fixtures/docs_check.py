# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load ``scripts/check_docs.py`` (not a package module) and build doc trees.

The docs tests and the ``--help`` epilog test share its slug function, so a
link the checker accepts is the link the epilog test accepts.
"""

from __future__ import annotations

import importlib.util
import subprocess
from functools import cache
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


@cache
def check_docs() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "check_docs", REPO_ROOT / "scripts" / "check_docs.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git_tree(root: Path, files: dict[str, str]) -> Path:
    """Write ``files`` under ``root`` and track them in a fresh Git repo."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root
