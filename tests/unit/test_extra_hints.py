# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One remedy for a missing extra (``shared.extras.extra_hint``), named the
same way by every message that names one."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

from unified_kg_rag.application.cli import run_console
from unified_kg_rag.shared.extras import extra_hint

pytestmark = pytest.mark.unit

PACKAGE = Path(__file__).parents[2] / "unified_kg_rag"


def test_the_hint_fits_a_checkout_and_a_wheel() -> None:
    hint = extra_hint("viz")
    # Added to the sync that was run, or inexact: `uv sync --extra viz`
    # alone uninstalled every other extra.
    assert "add `--extra viz` to the `uv sync` you ran" in hint
    assert "uv sync --inexact --extra viz" in hint
    assert not re.search(r"uv sync --extra", hint)
    # The package is not on PyPI: a wheel install names the wheel file.
    assert "pip install '<wheel file>[viz]'" in hint
    assert "unified-kg-rag-on-aws[" not in hint


def test_no_message_writes_its_own_install_command() -> None:
    # Five forms at nine sites: `uv sync --extra X` (exact: it removed the
    # other extras), `pip install -e '.[X]'` (a checkout only) and the
    # unpublished `unified-kg-rag-on-aws[X]`.
    own = re.compile(r"uv sync --extra|pip install -e|unified-kg-rag-on-aws\[")
    offenders = [
        f"{path.relative_to(PACKAGE)}:{n}"
        for path in sorted(PACKAGE.rglob("*.py"))
        if path.name != "extras.py"
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if own.search(line)
    ]
    assert offenders == []


def test_the_console_names_the_package_not_the_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real = importlib.util.find_spec

    def without_multipart(name: str, *args: object) -> object:
        return None if name == "multipart" else real(name, *args)  # type: ignore[arg-type]

    monkeypatch.setattr(importlib.util, "find_spec", without_multipart)
    assert run_console.missing_extra() == ["python-multipart"]
