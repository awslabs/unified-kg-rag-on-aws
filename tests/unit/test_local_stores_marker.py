# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every test against the local containers carries the ``local_stores``
marker, the one way such a test is skipped, so the CI local-stores job
(``-m local_stores``) runs all of them.

It listed its files, and the container cases of two files that skipped on
``LOCAL_STORES`` themselves ran nowhere (R2E-1).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from tests.fixtures.store_backends import BACKENDS, GRAPH_BACKENDS

pytestmark = pytest.mark.unit

_TESTS = Path(__file__).parents[1]
_ROOT = _TESTS.parent
# A read of the variable: getenv / environ with the literal or the
# store_backends constant.
_READS = re.compile(
    r"""(?:getenv|environ(?:\.get)?)\s*[\(\[]\s*(?:["']LOCAL_STORES["']|LOCAL_STORES_ENV)"""
)
# The skip (tests/conftest.py) and the harness's failure for an unmarked
# container test (tests/fixtures/store_backends.py).
_ALLOWED = {_TESTS / "conftest.py", _TESTS / "fixtures" / "store_backends.py"}


def test_only_the_marker_skips_on_local_stores() -> None:
    readers = sorted(
        str(path.relative_to(_ROOT))
        for path in _TESTS.rglob("*.py")
        if path not in _ALLOWED and _READS.search(path.read_text(encoding="utf-8"))
    )
    assert (
        readers == []
    ), "mark the test pytest.mark.local_stores instead of reading LOCAL_STORES"


def test_every_container_backend_carries_the_marker() -> None:
    for param in (*BACKENDS, *GRAPH_BACKENDS):
        kind = str(param.values[0]).partition("+")[0]
        names = {mark.name for mark in param.marks}
        assert ("local_stores" in names) == (kind == "container"), param


def test_the_ci_job_selects_by_the_marker() -> None:
    workflow = yaml.safe_load(
        (_ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    )
    runs = [
        step["run"]
        for step in workflow["jobs"]["local-stores"]["steps"]
        if (step.get("env") or {}).get("LOCAL_STORES") == "1"
    ]
    assert runs
    for run in runs:
        assert '-m "local_stores' in run, run
        # No file list of its own to fall behind.
        assert "tests/" not in run, run
