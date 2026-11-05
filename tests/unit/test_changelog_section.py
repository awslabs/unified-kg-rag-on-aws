# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/changelog_section.py: the release notes the release workflow publishes."""

from __future__ import annotations

import importlib.util
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

import unified_kg_rag

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "changelog_section.py"

CHANGELOG = """\
# Changelog

## [Unreleased]

### Added
- Something new.

## [0.3.0] - 2026-12-01

### Fixed
- A bug (#1).

## [0.3.0rc1] - 2026-11-20

### Fixed
- An earlier bug.

[Unreleased]: https://example.invalid/compare/v0.3.0...HEAD
[0.3.0]: https://example.invalid/compare/v0.3.0rc1...v0.3.0
"""


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("changelog_section", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extracts_one_section_up_to_the_next_heading() -> None:
    notes = _script().extract_section(CHANGELOG, "0.3.0")
    assert notes == "### Fixed\n- A bug (#1).\n"


def test_last_section_stops_at_the_link_references() -> None:
    notes = _script().extract_section(CHANGELOG, "0.3.0rc1")
    assert notes == "### Fixed\n- An earlier bug.\n"


def test_version_is_matched_exactly() -> None:
    # 0.3.0 must not match the 0.3.0rc1 heading, nor a regex wildcard.
    module = _script()
    with pytest.raises(module.ChangelogError, match=r"no `## \[0\.3\]`"):
        module.extract_section(CHANGELOG, "0.3")
    with pytest.raises(module.ChangelogError):
        module.extract_section(CHANGELOG.replace("0.3.0]", "0x3x0]"), "0.3.0")


def test_empty_section_fails() -> None:
    module = _script()
    with pytest.raises(module.ChangelogError, match="empty"):
        module.extract_section("## [1.0.0]\n\n## [0.9.0]\n- x\n", "1.0.0")


@pytest.mark.parametrize(
    ("given", "expected"),
    [("v0.3.0", "0.3.0"), ("0.3.0", "0.3.0"), ("refs/tags/v1.2.3", "1.2.3")],
)
def test_normalize_version(given: str, expected: str) -> None:
    assert _script().normalize_version(given) == expected


def test_tag_must_match_the_package_version() -> None:
    module = _script()
    init = '"""Doc."""\n__version__ = "0.3.0"\n'
    assert module.release_notes(CHANGELOG, "v0.3.0", init_text=init).startswith(
        "### Fixed"
    )
    with pytest.raises(module.ChangelogError, match="does not match"):
        module.release_notes(CHANGELOG, "v0.3.1", init_text=init)


def test_package_version_needs_exactly_one_assignment() -> None:
    module = _script()
    with pytest.raises(module.ChangelogError):
        module.package_version("x = 1\n")
    with pytest.raises(module.ChangelogError):
        module.package_version('__version__ = "1"\n__version__ = "2"\n')


def test_fallback_to_unreleased_only_when_asked() -> None:
    module = _script()
    with pytest.raises(module.ChangelogError):
        module.release_notes(CHANGELOG, "0.4.0.dev0")
    notes = module.release_notes(CHANGELOG, "0.4.0.dev0", fallback_unreleased=True)
    assert notes == "### Added\n- Something new.\n"


def test_main_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _script()
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    assert module.main(["v0.3.0", "--changelog", str(changelog)]) == 0
    assert capsys.readouterr().out == "### Fixed\n- A bug (#1).\n"
    assert module.main(["v9.9.9", "--changelog", str(changelog)]) == 1
    assert "no `## [9.9.9]` section" in capsys.readouterr().err


def test_script_reads_the_real_package_version() -> None:
    module = _script()
    init_text = module.PACKAGE_INIT.read_text(encoding="utf-8")
    assert module.package_version(init_text) == unified_kg_rag.__version__


def test_version_has_one_source() -> None:
    # pyproject derives the version from __init__.py rather than repeating it,
    # so a bump edits one line and the tag check reads the same value.
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    assert "version" not in pyproject["project"]
    assert "version" in pyproject["project"]["dynamic"]
    assert pyproject["tool"]["hatch"]["version"]["path"] == "unified_kg_rag/__init__.py"


def test_changelog_has_an_unreleased_section() -> None:
    # The dry run publishes nothing but still renders the notes it would use.
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert _script().extract_section(text, "Unreleased")


def test_short_notes_are_unchanged() -> None:
    assert (
        _script().fit_release_body("- a\n", "https://x.invalid", limit=100) == "- a\n"
    )


def test_long_notes_are_cut_at_a_line_break_with_a_link() -> None:
    module = _script()
    notes = "".join(f"- entry {i}\n" for i in range(100))
    body = module.fit_release_body(notes, "https://x.invalid/CHANGELOG.md", limit=200)
    assert len(body) <= 200
    assert body.endswith("The full notes: https://x.invalid/CHANGELOG.md\n")
    kept = body.split("\n\n... ")[0].splitlines()
    assert kept and all(line.startswith("- entry ") for line in kept)
    assert notes.startswith("\n".join(kept) + "\n")


def test_real_unreleased_notes_fit_a_release_body() -> None:
    # Whatever the [Unreleased] section grows to, the published body fits.
    module = _script()
    text = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    body = module.fit_release_body(
        module.extract_section(text, "Unreleased"), "https://x.invalid"
    )
    assert len(body) <= module.GITHUB_RELEASE_BODY_LIMIT


def test_main_prints_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert _script().main(["--print-package-version"]) == 0
    assert capsys.readouterr().out == f"{unified_kg_rag.__version__}\n"
