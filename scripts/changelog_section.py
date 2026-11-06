# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Print one version's CHANGELOG.md section, for the GitHub release notes.

The section is the body under ``## [<version>]`` (optionally followed by
`` - <date>``) up to the next ``## `` heading or the link references at the
bottom of the file. ``--check-package`` also fails unless the version (a tag
such as ``v0.3.0`` or a bare ``0.3.0``) equals ``unified_kg_rag.__version__``,
read from the source file without importing the package.
``--fallback-unreleased`` prints the ``[Unreleased]`` section when the
version has none (the release workflow's dry run on a dev version); an empty
``[Unreleased]`` (right after a release) then prints a one-line placeholder.
``--link-base`` resolves repo-relative link targets against a URL such as
``https://github.com/<owner>/<repo>/blob/<tag>/``, the way GitHub resolves them
when it renders CHANGELOG.md (a release body resolves none of them). Notes
longer than a GitHub release body allows (125,000 characters, less a margin)
are cut at a line break and end with a link to the full section
(``--full-notes-url``). ``.github/workflows/release.yml`` runs it;
``tests/unit/test_changelog_section.py`` covers it.

Usage:
    uv run python scripts/changelog_section.py --print-package-version
    uv run python scripts/changelog_section.py v0.3.0 --check-package \
        --link-base https://github.com/awslabs/unified-kg-rag-on-aws/blob/v0.3.0/
    uv run python scripts/changelog_section.py 0.3.0.dev0 --fallback-unreleased
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
PACKAGE_INIT = REPO_ROOT / "unified_kg_rag" / "__init__.py"

UNRELEASED = "Unreleased"
# GitHub rejects a release body over 125,000 characters; keep a margin.
GITHUB_RELEASE_BODY_LIMIT = 120_000

_VERSION_LINE = re.compile(r'^__version__ = "([^"]+)"$', re.MULTILINE)
# A link reference definition at the bottom: ``[0.3.0]: https://...``.
_LINK_REFERENCE = re.compile(r"^\[[^\]]+\]:\s")
# An inline link or image target, ``](target`` / ``![alt](target``, and a link
# reference definition's target, ``[label]: target``.
_INLINE_TARGET = re.compile(
    r"(?P<head>(?P<bang>!?)\[[^\[\]]*\]\()(?P<target><[^>]*>|[^)\s]+)"
)
_REFERENCE_TARGET = re.compile(
    r"^(?P<head> {0,3}\[[^\]]+\]:[ \t]*)(?P<target>\S+)", re.MULTILINE
)
# A scheme (``https:``, ``mailto:``), a protocol-relative ``//host`` or a
# fragment-only ``#x`` target is left as it is.
_NOT_REPO_RELATIVE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//|#)")
NO_UNRELEASED_CHANGES = "No unreleased changes.\n"


class ChangelogError(Exception):
    """The changelog or the package version does not support the release."""


def normalize_version(tag_or_version: str) -> str:
    """``v0.3.0`` -> ``0.3.0``; a bare version is returned unchanged."""
    value = tag_or_version.strip()
    if value.startswith("refs/tags/"):
        value = value.removeprefix("refs/tags/")
    return value[1:] if value.startswith("v") else value


def package_version(init_text: str) -> str:
    """The ``__version__`` string assigned in the package's ``__init__``."""
    matches = _VERSION_LINE.findall(init_text)
    if len(matches) != 1:
        raise ChangelogError(
            f'expected one `__version__ = "..."` line, found {len(matches)}'
        )
    return str(matches[0])


def extract_section(
    changelog_text: str, version: str, *, allow_empty: bool = False
) -> str:
    """The body of the ``## [version]`` section, stripped.

    An empty section fails unless ``allow_empty`` (then it is ``""``).
    """
    heading = re.compile(rf"^## \[{re.escape(version)}\](?:\s+-\s+\S.*)?\s*$")
    lines = changelog_text.splitlines()
    start = next((i for i, line in enumerate(lines) if heading.match(line)), None)
    if start is None:
        raise ChangelogError(f"CHANGELOG.md has no `## [{version}]` section")
    body: list[str] = []
    for line in lines[start + 1 :]:
        if line.startswith("## ") or _LINK_REFERENCE.match(line):
            break
        body.append(line)
    text = "\n".join(body).strip()
    if not text:
        if allow_empty:
            return ""
        raise ChangelogError(f"the `## [{version}]` section of CHANGELOG.md is empty")
    return text + "\n"


def absolutize_links(notes: str, base: str) -> str:
    """Resolve repo-relative link and image targets in ``notes`` against ``base``.

    ``base`` is the URL of the directory CHANGELOG.md sits in, such as
    ``https://github.com/<owner>/<repo>/blob/<tag>/``; images resolve against
    the matching ``/raw/`` URL so they load as files. Absolute URLs, ``mailto:``
    and fragment-only targets are unchanged.
    """
    base = base if base.endswith("/") else base + "/"
    image_base = base.replace("/blob/", "/raw/", 1)

    def resolve(target: str, root: str) -> str:
        bracketed = target.startswith("<") and target.endswith(">")
        inner = target[1:-1] if bracketed else target
        if not inner or _NOT_REPO_RELATIVE.match(inner):
            return target
        resolved = urljoin(root, inner)
        return f"<{resolved}>" if bracketed else resolved

    def inline(match: re.Match[str]) -> str:
        root = image_base if match["bang"] else base
        return match["head"] + resolve(match["target"], root)

    def reference(match: re.Match[str]) -> str:
        return match["head"] + resolve(match["target"], base)

    return _REFERENCE_TARGET.sub(reference, _INLINE_TARGET.sub(inline, notes))


def fit_release_body(
    notes: str, full_notes_url: str, limit: int = GITHUB_RELEASE_BODY_LIMIT
) -> str:
    """``notes`` unchanged if short enough, else cut at a line break plus a link."""
    if len(notes) <= limit:
        return notes
    footer = f"\n\n... The full notes: {full_notes_url}\n"
    room = limit - len(footer)
    if room <= 0:
        raise ChangelogError(f"a {limit}-character limit leaves no room for notes")
    cut = notes.rfind("\n", 0, room + 1)
    return notes[: cut if cut > 0 else room].rstrip() + footer


def release_notes(
    changelog_text: str,
    tag_or_version: str,
    *,
    init_text: str | None = None,
    fallback_unreleased: bool = False,
) -> str:
    """The notes for a release; checks the package version when given its source."""
    version = normalize_version(tag_or_version)
    if init_text is not None:
        current = package_version(init_text)
        if current != version:
            raise ChangelogError(
                f"tag/version {tag_or_version!r} does not match the package "
                f"version {current!r} (unified_kg_rag/__init__.py)"
            )
    try:
        return extract_section(changelog_text, version)
    except ChangelogError:
        if not fallback_unreleased:
            raise
        # Right after a release [Unreleased] is empty; the dry run still builds.
        notes = extract_section(changelog_text, UNRELEASED, allow_empty=True)
        return notes or NO_UNRELEASED_CHANGES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "version", nargs="?", help="a tag such as v0.3.0, or a bare version"
    )
    parser.add_argument(
        "--print-package-version",
        action="store_true",
        help="print unified_kg_rag.__version__ and exit",
    )
    parser.add_argument(
        "--check-package",
        action="store_true",
        help="fail unless the version equals unified_kg_rag.__version__",
    )
    parser.add_argument(
        "--fallback-unreleased",
        action="store_true",
        help="print the [Unreleased] section when the version has none",
    )
    parser.add_argument("--changelog", type=Path, default=CHANGELOG)
    parser.add_argument(
        "--link-base",
        help=(
            "resolve repo-relative link targets against this URL, e.g. "
            "https://github.com/<owner>/<repo>/blob/<tag>/"
        ),
    )
    parser.add_argument(
        "--full-notes-url",
        default="CHANGELOG.md",
        help="where truncated notes point to for the full section",
    )
    args = parser.parse_args(argv)
    try:
        init_text = PACKAGE_INIT.read_text(encoding="utf-8")
        if args.print_package_version:
            print(package_version(init_text))
            return 0
        if args.version is None:
            parser.error("a version is required")
        notes = release_notes(
            args.changelog.read_text(encoding="utf-8"),
            args.version,
            init_text=init_text if args.check_package else None,
            fallback_unreleased=args.fallback_unreleased,
        )
        if args.link_base:
            notes = absolutize_links(notes, args.link_base)
        notes = fit_release_body(notes, args.full_notes_url)
    except ChangelogError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
