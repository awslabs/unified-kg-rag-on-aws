# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Importing a CLI module leaves ``sys.path`` alone.

Regression: run_rag_chain put ``unified_kg_rag/application`` at the front of
``sys.path`` on import, a leftover from running the file as a script, so its
subpackages (``cli``, ``ingestion``, ``retrieval``, ``storage``) shadowed any
top-level module of the same name in the importing process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import unified_kg_rag.application.cli as cli_package

pytestmark = pytest.mark.unit

_PROBE = """
import importlib, json, sys
before = list(sys.path)
changed = {}
for name in sys.argv[1:]:
    importlib.import_module(name)
    if sys.path != before:
        changed[name] = [p for p in sys.path if p not in before]
        sys.path[:] = before
print(json.dumps(changed))
"""


def test_importing_cli_modules_does_not_change_sys_path() -> None:
    modules = sorted(
        f"{cli_package.__name__}.{path.stem}"
        for path in Path(cli_package.__file__).parent.glob("*.py")
        if path.stem != "__init__"
    )
    assert any(m.endswith(".run_rag_chain") for m in modules)
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, *modules],
        capture_output=True,
        text=True,
        check=True,
        timeout=110,
    )
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {}
