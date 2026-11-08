# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The design reference's index and its parts agree.

``docs/design.md`` lists every section heading of the design reference with
a summary and a link; each section's text lives in one part file
(``docs/design/*.md``, ``docs/guides/extend.md`` for §15). Code and docs cite
the section numbers and the heading ids, so a heading must keep its text in
both places, and the split itself moved text without editing it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.fixtures.design_docs import INDEX, as_one_file, blocks, heading_title, join
from tests.fixtures.docs_check import REPO_ROOT, check_docs

pytestmark = pytest.mark.unit

# The sections the index keeps in full rather than as a summary.
INDEX_SECTIONS = {
    "## Table of Contents",
    "## 1. Overview and Design Philosophy",
    "## 16. Further Reading",
}


def _parts(root: Path = REPO_ROOT) -> dict[str, str]:
    files = check_docs().tracked_files(root)
    return {
        rel: (root / rel).read_text(encoding="utf-8")
        for rel in (p.relative_to(root).as_posix() for p in files)
        if rel.endswith(".md") and rel.startswith(check_docs().DESIGN_PARTS)
    }


def _headings(text: str) -> list[str]:
    return [b.heading for b in blocks(text)[1:]]  # type: ignore[misc]


def test_every_part_heading_is_in_the_index_and_the_reverse() -> None:
    index = _headings((REPO_ROOT / INDEX).read_text(encoding="utf-8"))
    parts = _parts()
    assert len(parts) > 5, sorted(parts)
    in_parts = [h for text in parts.values() for h in _headings(text)]
    assert len(in_parts) == len(set(in_parts)), "a heading in two parts"
    missing = [h for h in in_parts if h not in index]
    assert not missing, f"part headings the index lacks: {missing}"
    extra = [h for h in index if h not in in_parts and h not in INDEX_SECTIONS]
    assert not extra, f"index headings no part holds: {extra}"
    # Each part holds its sections in the index's order.
    for path, text in parts.items():
        own = _headings(text)
        assert own == [h for h in index if h in own], path


def test_each_index_entry_links_its_section() -> None:
    docs = check_docs()
    held = {h: path for path, text in _parts().items() for h in _headings(text)}
    for block in blocks((REPO_ROOT / INDEX).read_text(encoding="utf-8"))[1:]:
        if block.heading in INDEX_SECTIONS:
            continue
        path = held[block.heading]  # type: ignore[index]
        slug = docs.slugify(heading_title(block.heading))  # type: ignore[arg-type]
        targets = {t for _, t, _ in docs.links(block.text)}
        rel = Path(path).relative_to("docs").as_posix()
        assert f"./{rel}#{slug}" in targets, (block.heading, targets)


def test_section_numbers_map_to_the_file_holding_them() -> None:
    sections = check_docs().design_sections()
    assert sections["1"] == INDEX
    assert sections["16"] == INDEX
    assert sections["4.6"] == "docs/design/ingestion.md"
    assert sections["6.1.1"] == "docs/design/retrieval-graphrag-lightrag.md"
    assert sections["6.6"] == "docs/design/structured-queries.md"
    assert sections["8.4"] == "docs/design/surfaces.md"
    assert sections["15"] == "docs/guides/extend.md"
    index = (REPO_ROOT / INDEX).read_text(encoding="utf-8")
    numbers = check_docs()._section_numbers(index)
    assert set(numbers) == set(sections)


def test_join_restores_the_single_file_and_sees_an_edit() -> None:
    files = {INDEX, "docs/design/a.md"}
    original = (
        "# Title\n\nIntro, [b](#2-b).\n\n## 1. A\n\nKept.\n\n---\n\n"
        '<a id="old-b"></a>\n\n## 2. B\n\nSee [a](#1-a) and '
        "[g](./guides/g.md).\n\n### 2.1 C\n\nText `[x](y)`.\n"
    )
    index = (
        "# Title\n\nIntro, [b](#2-b).\n\n## 1. A\n\nKept.\n\n---\n\n"
        '<a id="old-b"></a>\n\n## 2. B\n\nSummary.\n\n### 2.1 C\n\nSummary.\n'
    )
    part = (
        "# Part\n\nPart of the [index](../design.md).\n\n## 2. B\n\n"
        "See [a](../design.md#1-a) and [g](../guides/g.md).\n\n"
        "### 2.1 C\n\nText `[x](y)`.\n"
    )
    whole = as_one_file(original, files)
    assert join(index, {"docs/design/a.md": part}, files) == whole
    edited = part.replace("See", "Read")
    assert join(index, {"docs/design/a.md": edited}, files) != whole


def _git(*args: str) -> str | None:
    try:
        done = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
        )
    except OSError:
        return None
    return done.stdout if done.returncode == 0 else None


def test_the_split_moved_text_without_editing_it() -> None:
    # The commit that split the single file, and the file it split.
    added = _git(
        "log", "--diff-filter=A", "--format=%H", "--", "docs/design/architecture.md"
    )
    if not added:
        pytest.skip("no Git history of the split")
    split = added.split()[-1]
    original = _git("show", f"{split}^:{INDEX}")
    if original is None:
        pytest.skip("the commit before the split is not fetched (a shallow clone)")
    listed = _git("ls-tree", "-r", "--name-only", split, "docs/design", "docs/guides")
    assert listed
    names = [
        n
        for n in listed.split()
        if n.endswith(".md") and n.startswith(check_docs().DESIGN_PARTS)
    ]
    parts = {name: _git("show", f"{split}:{name}") or "" for name in names}
    index = _git("show", f"{split}:{INDEX}")
    assert index is not None
    files = set(parts) | {INDEX}
    assert join(index, parts, files) == as_one_file(original, files)
