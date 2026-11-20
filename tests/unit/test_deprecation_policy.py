# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The deprecated names are removed in the release CONTRIBUTING.md's
deprecation policy names (``REMOVED_IN``), and every place a user reads of
them says which release that is."""

from __future__ import annotations

from pathlib import Path

import pytest
from packaging.version import Version

from unified_kg_rag import __version__
from unified_kg_rag.shared.utils.deprecation import REMOVED_IN

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_package_is_older_than_the_release_that_removes_them() -> None:
    # 0.5.0.dev0 is before 0.5.0: the release branch's version reaches it.
    assert Version(__version__) < Version(REMOVED_IN), (
        f"{__version__} is the release that removes the deprecated names: "
        "remove them (CHANGELOG *Deprecated* lists them) and move REMOVED_IN "
        "to the release that removes the next ones"
    )


def _section(text: str, heading: str, level: str) -> str:
    """``heading``'s section of ``text``: up to the next heading of its
    ``level`` (``"## "``) or a higher one."""
    start = text.index(heading)
    ends = [
        found
        for marker in {"\n## ", f"\n{level}"}
        if (found := text.find(marker, start + len(heading))) != -1
    ]
    return text[start : min(ends, default=len(text))]


def test_the_docs_name_the_release() -> None:
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased = _section(changelog, "## [Unreleased]", "## ")
    if "### Deprecated" in unreleased:
        deprecated = _section(unreleased, "### Deprecated", "### ")
        assert f"removed in {REMOVED_IN}" in deprecated
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    policy = _section(contributing, "### Deprecating a public name", "### ")
    assert f"removed in {REMOVED_IN}" in policy
    operations = (REPO_ROOT / "docs" / "operations.md").read_text(encoding="utf-8")
    upgrading = _section(operations, "## Upgrading the library", "## ")
    assert f"removed in {REMOVED_IN}" in upgrading
