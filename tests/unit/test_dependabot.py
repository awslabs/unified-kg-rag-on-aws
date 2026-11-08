# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The console web client's Dependabot entry holds back the majors that
cannot land on their own (two red PRs were closed by hand)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_CONFIG = Path(__file__).resolve().parents[2] / ".github" / "dependabot.yml"


def test_typescript_and_node_types_majors_are_held_for_the_console() -> None:
    updates = yaml.safe_load(_CONFIG.read_text(encoding="utf-8"))["updates"]
    (console,) = [u for u in updates if u["directory"] == "/console/web"]
    ignored = {i["dependency-name"]: i for i in console["ignore"]}
    for name in ("typescript", "@types/node"):
        # Majors only: minors and patches still come in the weekly group.
        assert ignored[name] == {
            "dependency-name": name,
            "update-types": ["version-update:semver-major"],
        }
