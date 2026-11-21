# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The deprecated names are removed in the release CONTRIBUTING.md's
deprecation policy names (``REMOVED_IN``), and every place a user reads of
them says which release that is."""

from __future__ import annotations

import re
import tokenize
from collections.abc import Iterator
from pathlib import Path

import pytest
from packaging.version import Version

import unified_kg_rag
from unified_kg_rag import __version__
from unified_kg_rag.application.retrieval.rag_chain import RAGInput
from unified_kg_rag.domain.models.config._migration import (
    DEPRECATED_KEYS,
    deprecated_key_warning,
)
from unified_kg_rag.domain.models.namespace import (
    DEPRECATED_NAMESPACE_NAMES,
    deprecated_namespace_warning,
)
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


# The root's renamed names (TransientRetryConfig -> RetryPolicy).
_RENAMED = unified_kg_rag._RENAMED


def test_the_deprecated_section_lists_every_deprecated_key_and_name() -> None:
    # The release that removes the names works from this list.
    changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    deprecated = _section(
        _section(changelog, "## [Unreleased]", "## "), "### Deprecated", "### "
    )
    # The names in its code spans; ``RAGInput.suffix`` lists ``suffix``.
    named = {
        token
        for span in re.findall(r"`([^`]*)`", deprecated)
        for token in re.split(r"[^\w.\-]+", span)
    }
    names = [*DEPRECATED_KEYS, *DEPRECATED_NAMESPACE_NAMES, *_RENAMED]
    missing = [
        name
        for name in names
        if not any(t == name or t.endswith(f".{name}") for t in named)
    ]
    assert not missing, f"CHANGELOG *Deprecated* does not list {missing}"


def test_every_deprecation_text_names_the_release() -> None:
    removed = f"(removed in {REMOVED_IN})"
    for key in DEPRECATED_KEYS:
        assert removed in deprecated_key_warning(key), key
    for name in DEPRECATED_NAMESPACE_NAMES:
        assert removed in deprecated_namespace_warning(name), name
    for name in _RENAMED:
        with pytest.warns(DeprecationWarning, match=re.escape(removed)):
            getattr(unified_kg_rag, name)
    with pytest.warns(DeprecationWarning, match=re.escape(removed)):
        RAGInput(query="q", enable_thinking=True)


def _strings(path: Path) -> Iterator[tuple[int, str]]:
    """The string literals of ``path`` (an f-string's literal parts too)."""
    kinds = {tokenize.STRING, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)}
    with path.open("rb") as source:
        for token in tokenize.tokenize(source.readline):
            if token.type in kinds:
                yield token.start[0], token.string


def test_no_deprecation_text_is_written_by_hand() -> None:
    # Each goes through deprecation_message, which names REMOVED_IN: a
    # literal "is deprecated" elsewhere would not.
    package = REPO_ROOT / "unified_kg_rag"
    helper = package / "shared" / "utils" / "deprecation.py"
    by_hand = [
        f"{path.relative_to(REPO_ROOT)}:{line}"
        for path in sorted(package.rglob("*.py"))
        if path != helper
        for line, text in _strings(path)
        if "is deprecated" in text
    ]
    assert not by_hand, by_hand
