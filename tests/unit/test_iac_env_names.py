# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every environment variable the CDK stacks inject names something the app
reads.

The IaC is a separate project (the CDK is not installed here), so its stack
sources are read as text: a ``UKR__`` variable must resolve to a config key,
or the app fails on it as an unknown key; a single-underscore ``UKR_``
variable must be read somewhere in the package, or it is silently ignored.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from unified_kg_rag.application.config import ENV_PREFIX, _env_path

_ROOT = Path(__file__).resolve().parents[2]
_STACKS = sorted((_ROOT / "iac" / "stacks").glob("*.py"))
_NAMES = sorted(
    {
        name
        for path in _STACKS
        for name in re.findall(r"[\"'](UKR_[A-Z0-9_]+)[\"']", path.read_text())
    }
)


def test_the_stacks_inject_variables() -> None:
    # The extraction itself: an empty list would pass every check below.
    assert _STACKS and "UKR__AWS__S3__BUCKET_NAME" in _NAMES
    assert "UKR_PIPELINE_ID" in _NAMES


@pytest.mark.parametrize(
    "name", [n for n in _NAMES if n.startswith(ENV_PREFIX)] or ["<none>"]
)
def test_a_config_variable_names_a_config_key(name: str) -> None:
    assert _env_path(name) is not None, f"{name} names no config key"


@pytest.mark.parametrize(
    "name", [n for n in _NAMES if not n.startswith(ENV_PREFIX)] or ["<none>"]
)
def test_another_variable_is_read_by_the_app(name: str) -> None:
    package = _ROOT / "unified_kg_rag"
    readers = [
        path
        for path in package.rglob("*.py")
        if re.search(rf"getenv\(\s*\"{name}\"|environ\[\"{name}\"\]", path.read_text())
    ]
    assert readers, f"{name} is injected but nothing in the package reads it"
