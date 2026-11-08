# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every relative link, fragment and image in the docs resolves.

``scripts/check_docs.py`` does the checking; run it directly for the full list
of failures. The synthetic trees below pin the GitHub slug rules it relies on.
"""

from __future__ import annotations

import subprocess
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
        ({"a.md": '<a href="gone.md">x</a>'}, "a.md:1: missing target: gone.md"),
        ({"a.md": "[x][r]\n\n[r]: gone.md"}, "a.md:3: missing target: gone.md"),
        (
            {"a.md": "See `unified_kg_rag/gone.py`."},
            "a.md:1: missing code path: unified_kg_rag/gone.py",
        ),
        (
            {"a.md": "x\n`domain/gone.py`"},
            "a.md:2: missing code path: domain/gone.py",
        ),
        (
            {"tests/t.py": "# docs/gone.md\n"},
            "tests/t.py:1: missing target: docs/gone.md",
        ),
        (
            {"pyproject.toml": "# docs/gone.md\n"},
            "pyproject.toml:1: missing target: docs/gone.md",
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


def test_a_link_inside_an_html_comment_is_checked(tmp_path: Path) -> None:
    # A commented-out placeholder (a recording to add later) must resolve
    # once uncommented; a code span inside the comment is still no link.
    files = {
        "a.md": "# A\n\n<!-- ![demo](./assets/demo.gif)\n`[x](gone.md)` -->\n",
    }
    assert _failures(tmp_path, files) == ["a.md:3: missing target: ./assets/demo.gif"]


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


def test_a_code_citation_of_a_file_git_does_not_track_fails(tmp_path: Path) -> None:
    root = git_tree(
        tmp_path,
        {"unified_kg_rag/m.py": '"""See docs/t.md#t for it."""\n', "a.md": "# A"},
    )
    (root / "docs").mkdir()
    (root / "docs" / "t.md").write_text("# T\n", encoding="utf-8")
    report = check_docs().check_links(root)
    assert report.counts["code citations"] == 1
    assert report.failures == [
        "unified_kg_rag/m.py:1: target not tracked by Git: docs/t.md#t"
    ]


def test_a_tracked_symlink_to_an_untracked_file_fails(tmp_path: Path) -> None:
    # The checkout holds the link, not what it points to.
    root = git_tree(tmp_path, {"README.md": "[o](alias.md)\n"})
    (root / "outputs").mkdir()
    (root / "outputs" / "o.md").write_text("# O\n", encoding="utf-8")
    (root / "alias.md").symlink_to("outputs/o.md")
    subprocess.run(["git", "add", "alias.md"], cwd=root, check=True)
    report = check_docs().check_links(root)
    assert report.failures == ["README.md:1: target not tracked by Git: alias.md"]


def test_a_tracked_symlink_to_a_tracked_file_passes(tmp_path: Path) -> None:
    root = git_tree(tmp_path, {"README.md": "[o](alias.md)\n", "docs/o.md": "# O"})
    (root / "alias.md").symlink_to("docs/o.md")
    subprocess.run(["git", "add", "alias.md"], cwd=root, check=True)
    assert check_docs().check_links(root).failures == []


def test_a_link_to_a_directory_git_tracks_nothing_in_fails(tmp_path: Path) -> None:
    root = git_tree(
        tmp_path, {"README.md": "[o](outputs/) [d](docs/)\n", "docs/a.md": "# A"}
    )
    (root / "outputs").mkdir()
    (root / "outputs" / "x.txt").write_text("x", encoding="utf-8")
    assert check_docs().check_links(root).failures == [
        "README.md:1: target not tracked by Git: outputs/"
    ]


def test_inline_code_paths_that_exist_pass(tmp_path: Path) -> None:
    report = check_docs().check_links(
        git_tree(
            tmp_path,
            {
                "a.md": "`unified_kg_rag/m.py`, `domain/x.py`, `ports/__init__`,"
                " `domain/x.attr`, `docs/` and `docs/b.md#b`; `docs/<page>.md`,"
                " `a/b.py` and `unified_kg_rag/query` are not checked.",
                "unified_kg_rag/m.py": "",
                "unified_kg_rag/domain/x.py": "",
                "unified_kg_rag/ports/__init__.py": "",
                "docs/b.md": "# B",
            },
        )
    )
    assert report.failures == []
    assert report.counts["code paths"] == 6


def test_a_code_citation_ends_at_a_full_stop(tmp_path: Path) -> None:
    files = {"unified_kg_rag/m.py": '"""See docs/b.md#b."""\n', "docs/b.md": "# B"}
    assert _failures(tmp_path, files) == []


_DESIGN = {
    "docs/design.md": "# D\n## 1. One\n## 2. Two\n### 2.1 Sub\n## 3. Three\n",
    "docs/design/two.md": "# Two\n## 2. Two\n### 2.1 Sub\n",
    "docs/design/three.md": "# Three\n## 3. Three\n",
}


def test_section_refs_name_design_sections(tmp_path: Path) -> None:
    root = git_tree(
        tmp_path,
        {
            **_DESIGN,
            "README.md": "See docs/design.md §2.1, §3 and [§2](docs/design/two.md#2-two)"
            " and [§1](docs/design.md#1-one).\n",
            "unified_kg_rag/m.py": '"""Cites docs/design.md §2.1."""\n',
        },
    )
    report = check_docs().check_links(root)
    assert not report.failures
    assert report.counts["section refs"] == 5
    assert check_docs().design_sections(root) == {
        "1": "docs/design.md",
        "2": "docs/design/two.md",
        "2.1": "docs/design/two.md",
        "3": "docs/design/three.md",
    }


def test_an_unknown_section_or_a_link_to_the_wrong_part_fails(tmp_path: Path) -> None:
    failures = _failures(
        tmp_path,
        {
            **_DESIGN,
            "README.md": "See [§3](docs/design/two.md#2-two).\n",
            "unified_kg_rag/m.py": '"""Cites docs/design.md §4.2."""\n',
        },
    )
    assert any("README.md:1: links §3 to docs/design/two.md" in f for f in failures)
    assert any("m.py:1: no design section §4.2" in f for f in failures)


def test_docs_cite_no_pull_request_numbers_and_diagrams_are_whole() -> None:
    report = check_docs().check_hygiene()
    assert not report.failures, "\n".join(report.failures)
    assert report.counts["diagrams"] >= 6


@pytest.mark.parametrize(
    ("text", "flagged"),
    [
        ("Fixed in #627 last week.", ["#627"]),
        ("(#618), and", ["#618"]),
        ("See [§6.1.1](#611-global-search) and `#999` in code.", []),
        ("&#1234; and issue#123 and docs/a.md#100", []),
        ("Version 2.19 and #12 stay.", []),
    ],
)
def test_pull_request_numbers_outside_the_changelog_fail(
    tmp_path: Path, text: str, flagged: list[str]
) -> None:
    files = {
        "docs/a.md": text,
        "CHANGELOG.md": "- A fix (#627).",
        ".github/t.md": "Closes #123",
    }
    failures = check_docs().check_hygiene(git_tree(tmp_path, files)).failures
    assert [f.split("number ")[1].split(" ")[0] for f in failures] == flagged
    assert all(f.startswith("docs/a.md:1:") for f in failures)


def _png(*chunks: tuple[bytes, bytes], crc_ok: bool = True) -> bytes:
    import struct
    import zlib

    out = b"\x89PNG\r\n\x1a\n"
    for kind, body in chunks:
        crc = zlib.crc32(kind + body) ^ (0 if crc_ok else 1)
        out += struct.pack(">I", len(body)) + kind + body + struct.pack(">I", crc)
    return out


@pytest.mark.parametrize(
    ("data", "problems"),
    [
        (_png((b"zTXt", b"mxGraphModel\0\0x"), (b"IEND", b"")), []),
        (_png((b"tEXt", b"mxfile\0x"), (b"IEND", b"")), []),
        (
            _png((b"zTXt", b"mxGraphModel\0\0x"), (b"IEND", b""), crc_ok=False),
            ["zTXt chunk CRC mismatch", "IEND chunk CRC mismatch"],
        ),
        (_png((b"zTXt", b"mxGraphModel\0\0x")), ["no IEND chunk"]),
        (
            _png((b"zTXt", b"mxGraphModel\0\0x"))[:-6],
            ["truncated zTXt chunk", "no IEND chunk", "no embedded draw.io source"],
        ),
        (_png((b"IEND", b"")), ["no embedded draw.io source"]),
        (b"GIF89a", ["not a PNG"]),
    ],
    ids=["ztxt", "text", "crc", "no-iend", "truncated", "no-source", "not-png"],
)
def test_a_diagram_must_be_a_whole_png_with_its_source(
    tmp_path: Path, data: bytes, problems: list[str]
) -> None:
    assert check_docs().png_problems(data) == problems
    root = git_tree(tmp_path, {"README.md": "# R"})
    (root / "assets").mkdir()
    (root / "assets" / "x.drawio.png").write_bytes(data)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    failures = check_docs().check_hygiene(root).failures
    assert failures == [f"assets/x.drawio.png: {p}" for p in problems]
