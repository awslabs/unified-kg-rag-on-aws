# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every relative link, fragment and image in the docs resolves.

``scripts/check_docs.py`` does the checking; run it directly for the full list
of failures. The synthetic trees below pin the GitHub slug rules it relies on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.docs_check import check_docs, git_tree

pytestmark = pytest.mark.unit

REPO = "https://github.com/awslabs/unified-kg-rag-on-aws/blob/main/"


def test_every_doc_link_resolves() -> None:
    report = check_docs().check_links()
    assert not report.failures, "broken doc links (run `uv run python "
    "scripts/check_docs.py`):\n" + "\n".join(report.failures)
    # The checker found the links: an empty scan would pass silently.
    assert report.counts["links"] > 100
    assert report.counts["images"] > 0
    assert report.counts["code citations"] > 0


@pytest.mark.parametrize(
    ("title", "slug"),
    [
        ("2. Configuration", "2-configuration"),
        ("`run-rag` flags (CLI)", "run-rag-flags-cli"),
        ("A — B & C", "a--b--c"),
        ("[Linked](./x.md) heading", "linked-heading"),
        ("snake_case <sup>new</sup>", "snake_case-new"),
        ("2. 설정", "2-설정"),
    ],
)
def test_slugify_follows_github(title: str, slug: str) -> None:
    assert check_docs().slugify(title) == slug


def test_duplicate_headings_and_html_ids_are_fragments(tmp_path: Path) -> None:
    page = tmp_path / "a.md"
    page.write_text(
        "# Setup\n## Setup\n## Setup-1\n### Setup\n"
        '<a id="custom"></a>\n```\n# not a heading\n```\n',
        encoding="utf-8",
    )
    assert check_docs().fragments(page) == {
        "setup",
        "setup-1",
        "setup-1-1",
        "setup-2",
        "custom",
    }


def _failures(root: Path, files: dict[str, str]) -> list[str]:
    return check_docs().check_links(git_tree(root, files)).failures


def test_valid_links_pass(tmp_path: Path) -> None:
    assert not _failures(
        tmp_path,
        {
            "README.md": "[g](docs/g.md#2-설정) [h](docs/g.md#2-%EC%84%A4%EC%A0%95)"
            " [d](docs/g.md#dup-1) [top](#readme) [badge ![b](img.png)](LICENSE)"
            ' <img src="img.png"> [ext](https://example.org/missing.md)'
            f" [abs]({REPO}docs/g.md#dup)\n# Readme\n",
            "LICENSE": "",
            "img.png": "",
            "docs/g.md": "## 2. 설정\n## Dup\n## Dup\n",
            "unified_kg_rag/m.py": f'URL = "{REPO}docs/g.md#dup-1"  # docs/g.md\n'
            "# https://example.org/blob/main/docs/missing.md\n",
        },
    )


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        ({"a.md": "[x](missing.md)"}, "a.md:1: missing target: missing.md"),
        ({"a.md": "\n![x](gone.png)"}, "a.md:2: missing target: gone.png"),
        ({"a.md": '<img src="gone.png">'}, "a.md:1: missing target: gone.png"),
        ({"a.md": "[![b](gone.png)](a.md)"}, "a.md:1: missing target: gone.png"),
        ({"a.md": "x\n[a wrapped\nlink](gone.md)"}, "a.md:2: missing target: gone.md"),
        ({"a.md": "[x](#nope)\n# Yes"}, "a.md:1: missing fragment #nope in a.md"),
        (
            {"a.md": "[x](b.md#dup-2)", "b.md": "# Dup\n# Dup"},
            "a.md:1: missing fragment #dup-2 in b.md",
        ),
        (
            {"iac/s.py": "# see docs/gone.md\n"},
            "iac/s.py:1: missing target: docs/gone.md",
        ),
        (
            {"scripts/s.py": f"U = '{REPO}docs/b.md#nope'", "docs/b.md": "# B"},
            "scripts/s.py:1: missing fragment #nope in docs/b.md",
        ),
    ],
)
def test_broken_links_are_reported(
    tmp_path: Path, files: dict[str, str], expected: str
) -> None:
    assert _failures(tmp_path, files) == [expected]


def test_code_spans_and_fences_are_not_links(tmp_path: Path) -> None:
    assert not _failures(
        tmp_path,
        {
            "a.md": "`[x](missing.md)` ``[y](gone.md)``\n"
            "- item\n\n  ```md\n  [z](gone.md)\n  ```\n"
            "~~~\n[w](gone.md)\n~~~\n",
        },
    )


def test_untracked_files_are_not_checked(tmp_path: Path) -> None:
    git_tree(tmp_path, {"a.md": "# A"})
    (tmp_path / "local.md").write_text("[x](missing.md)", encoding="utf-8")
    assert not check_docs().check_links(tmp_path).failures


def test_the_script_exits_non_zero_on_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    module = check_docs()
    git_tree(tmp_path, {"README.md": "[x](missing.md)"})
    monkeypatch.setattr(module, "TRANSLATED_PAGES", [])
    assert module.main(tmp_path) == 1
    assert "FAIL README.md:1: missing target: missing.md" in capsys.readouterr().out


def test_a_link_to_a_file_git_does_not_track_fails(tmp_path: Path) -> None:
    """A target that exists only on disk (untracked or ignored) is broken in CI."""
    root = git_tree(tmp_path, {"README.md": "[t](docs/t.md)\n"})
    (root / "docs").mkdir()
    (root / "docs" / "t.md").write_text("# T\n", encoding="utf-8")
    report = check_docs().check_links(root)
    assert report.failures == ["README.md:1: target not tracked by Git: docs/t.md"]
