# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the documentation's links, fragments and EN/KO structure.

Every tracked ``*.md`` file is scanned for relative links and images: the
target must exist and a ``#fragment`` into a Markdown file must match one of its
headings (GitHub slug rules, duplicate headings get ``-1``, ``-2``...) or an
``<a id>``/``<a name>``. Code and configuration under ``unified_kg_rag/``,
``scripts/`` and ``iac/`` that cite ``docs/*.md`` paths or this repository's
``blob/main/docs/...#fragment`` URLs are checked the same way. Each EN/KO pair in
:data:`TRANSLATED_PAGES` must keep the same structure, unless the Korean page
carries :data:`LAG_NOTE`. ``tests/unit/test_docs_links.py`` and
``tests/unit/test_docs_translations.py`` run these checks in CI.

Usage:
    uv run python scripts/check_docs.py   # exit 1 and list every failure
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter
from functools import cache
from pathlib import Path
from urllib.parse import unquote

REPO_ROOT = Path(__file__).resolve().parent.parent

# The pages with a Korean twin (``<page>.ko.md`` next to ``<page>.md``), as
# repo-relative paths without the suffix. Adding a ``.ko.md`` file that is not
# listed here fails the check.
TRANSLATED_PAGES = ["README", "docs/index", "docs/concepts/overview", "docs/user-guide"]

# The note a Korean page carries while its translation lags the English page
# (CONTRIBUTING.md, Documentation); such a page is reported, not failed.
LAG_NOTE = "이 번역은 영문판보다 늦을 수 있습니다"

REPO_URL = "https://github.com/awslabs/unified-kg-rag-on-aws/blob/main/"
CODE_DIRS = ("unified_kg_rag", "scripts", "iac")
SKIP_PARTS = {".git", ".venv", "node_modules", "references", "cdk.out"}

_FENCE = re.compile(r"(`{3,}|~{3,})")
_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*$")
_HTML_ID = re.compile(r"<a\s[^>]*?\b(?:id|name)\s*=\s*[\"']([^\"']+)[\"']", re.I)
_CODE_SPAN = re.compile(r"(`+)(?!`)[\s\S]*?(?<!`)\1(?!`)")
_COMMENT = re.compile(r"<!--[\s\S]*?-->")
# (target "title"); <target> allows spaces, a bare one balanced parentheses.
_TARGET = r"\(\s*(<[^>\n]*>|[^\s()]*(?:\([^\s()]*\)[^\s()]*)*)(?:\s+[\"'(][^)]*)?\)"
# ![alt](target), and [text](target) whose text may hold an image or brackets.
_IMAGE = re.compile(r"!\[[^\[\]]*\]" + _TARGET)
_LINK = re.compile(r"(?<!!)\[(?:[^\[\]]|\[[^\[\]]*\])*\]" + _TARGET)
_IMG_SRC = re.compile(r"<img\s[^>]*?\bsrc\s*=\s*[\"']([^\"']+)[\"']", re.I)
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
# docs/<path>.md[#fragment] inside code, docstrings, comments and YAML.
_CODE_CITE = re.compile(r"(?<![\w-])docs/[\w./-]+?\.md(?:#[^\s\"'`)\]>,;]+)?")
_TOKEN_START = re.compile(r"[\s\"'`(\[<=]")


class Report:
    """The outcome of a check: failures, lagging pages and what was counted."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.lagging: list[str] = []
        self.counts: Counter[str] = Counter()


def slugify(text: str) -> str:
    """GitHub's heading slug: lowercase, punctuation dropped, spaces to ``-``."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text).strip().lower()
    return "".join(c for c in text if c.isalnum() or c in "-_ ").replace(" ", "-")


def _split(text: str) -> tuple[list[tuple[int, str]], int]:
    """The ``(line number, line)`` pairs outside fenced code blocks, and the
    number of fenced blocks (indented fences inside list items included)."""
    prose: list[tuple[int, str]] = []
    fence, blocks = "", 0
    for number, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        match = _FENCE.match(stripped)
        if fence:
            if match and stripped == match.group(1) and stripped[0] == fence[0]:
                fence = "" if len(stripped) >= len(fence) else fence
            continue
        if match:
            fence, blocks = match.group(1), blocks + 1
            continue
        prose.append((number, line))
    return prose, blocks


def prose_lines(text: str) -> list[tuple[int, str]]:
    """The ``(line number, line)`` pairs outside fenced code blocks."""
    return _split(text)[0]


def headings(text: str) -> list[tuple[int, str]]:
    """The ``(level, title)`` of every ATX heading outside code blocks."""
    out = []
    for _, line in prose_lines(text):
        match = _HEADING.match(line)
        if match:
            out.append((len(match.group(1)), match.group(2) or ""))
    return out


@cache
def fragments(path: Path) -> frozenset[str]:
    """Every fragment (heading id) GitHub renders for a Markdown file."""
    text = path.read_text(encoding="utf-8")
    seen: dict[str, int] = {}
    out: set[str] = set()
    for _, title in headings(text):
        base = slug = slugify(title)
        while slug in out:
            seen[base] = seen.get(base, 0) + 1
            slug = f"{base}-{seen[base]}"
        out.add(slug)
    for _, line in prose_lines(text):
        out.update(_HTML_ID.findall(line))
    return frozenset(out)


def links(text: str) -> list[tuple[int, str, bool]]:
    """The ``(line, target, is_image)`` of every link outside code.

    Paragraphs are scanned whole, so a link whose text wraps is found.
    """
    paragraphs: list[list[tuple[int, str]]] = []
    for number, line in prose_lines(text):
        if not line.strip():
            continue
        if paragraphs and paragraphs[-1][-1][0] == number - 1:
            paragraphs[-1].append((number, line))
        else:
            paragraphs.append([(number, line)])
    out = []
    for paragraph in paragraphs:
        first = paragraph[0][0]
        body = "\n".join(line for _, line in paragraph)
        # Blank code and comments out, keeping offsets and newlines.
        for pattern in (_CODE_SPAN, _COMMENT):
            body = pattern.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), body)
        found = [(m.start(), m.group(1), True) for m in _IMAGE.finditer(body)]
        found += [(m.start(), m.group(1), False) for m in _LINK.finditer(body)]
        found += [(m.start(), m.group(1), True) for m in _IMG_SRC.finditer(body)]
        for start, target, image in sorted(found):
            out.append((first + body.count("\n", 0, start), target.strip("<>"), image))
    return out


def tracked_files(root: Path) -> list[Path]:
    """The files Git tracks under ``root``, or every file when Git is absent."""
    try:
        listed = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.split("\0")
        files = [root / name for name in listed if name]
    except (OSError, subprocess.CalledProcessError):
        files = [p for p in root.rglob("*") if p.is_file()]
    return sorted(
        p
        for p in files
        if p.is_file() and not SKIP_PARTS & set(p.relative_to(root).parts)
    )


def _check_target(
    root: Path, source: Path, where: str, target: str, base: Path, report: Report
) -> None:
    path, _, fragment = target.partition("#")
    dest = (base / unquote(path)).resolve() if path else source
    if not dest.exists():
        report.failures.append(f"{where}: missing target: {target}")
    elif fragment and dest.suffix == ".md":
        known = {a.lower() for a in fragments(dest)}
        if unquote(fragment).lower() not in known:
            try:
                shown = dest.relative_to(root.resolve()).as_posix()
            except ValueError:
                shown = str(dest)
            report.failures.append(f"{where}: missing fragment #{fragment} in {shown}")


def check_links(root: Path = REPO_ROOT) -> Report:
    """Check Markdown links and images, then code-cited doc paths."""
    report = Report()
    files = tracked_files(root)
    for md in (p for p in files if p.suffix == ".md"):
        rel = md.relative_to(root).as_posix()
        for number, target, image in links(md.read_text(encoding="utf-8")):
            base = md.parent
            if target.startswith(REPO_URL):
                target, base = target[len(REPO_URL) :], root
            elif _SCHEME.match(target) or target.startswith("//"):
                continue
            report.counts["images" if image else "links"] += 1
            _check_target(root, md, f"{rel}:{number}", target, base, report)
    code = [
        p
        for p in files
        if p.suffix != ".md" and p.relative_to(root).parts[0] in CODE_DIRS
    ]
    for path in code:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(root).as_posix()
        for number, line in enumerate(text.splitlines(), 1):
            for match in _CODE_CITE.finditer(line):
                start = max(
                    (m.end() for m in _TOKEN_START.finditer(line, 0, match.start())),
                    default=0,
                )
                token = line[start : match.end()]
                # Another host's URL (a placeholder or a fork) is not ours.
                if "://" in token and not token.startswith(REPO_URL):
                    continue
                report.counts["code citations"] += 1
                _check_target(
                    root, path, f"{rel}:{number}", match.group(), root, report
                )
    return report


def structure(text: str, page: Path, root: Path) -> dict[str, object]:
    """What an EN/KO pair must share: heading levels, code blocks, table rows
    and repository link targets (fragments dropped, a translated ``x.md`` read
    as ``x.ko.md``, links between the pair itself ignored)."""
    prose, blocks = _split(text)
    rows = sum(1 for _, line in prose if line.lstrip().startswith("|"))
    translated = {f"{p}.md" for p in TRANSLATED_PAGES}
    stem = page.relative_to(root).as_posix().removesuffix(".md").removesuffix(".ko")
    targets = set()
    for _, target, _ in links(text):
        path = target.partition("#")[0]
        # Same-page fragments and external URLs are translated (Korean slugs,
        # localized sites); only the files the pair links to must agree.
        if not path or _SCHEME.match(path) or path.startswith("//"):
            continue
        resolved = (page.parent / unquote(path)).resolve()
        try:
            rel = resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            rel = path
        if rel in (f"{stem}.md", f"{stem}.ko.md"):
            continue
        if rel in translated:
            rel = rel.removesuffix(".md") + ".ko.md"
        targets.add(rel)
    return {
        "headings per level": dict(Counter(level for level, _ in headings(text))),
        "fenced code blocks": blocks,
        "table rows": rows,
        "link targets": targets,
    }


def check_translation(page: str, root: Path = REPO_ROOT) -> Report:
    """Compare one :data:`TRANSLATED_PAGES` pair."""
    report = Report()
    en, ko = root / f"{page}.md", root / f"{page}.ko.md"
    for path in (en, ko):
        if not path.exists():
            report.failures.append(f"{page}: missing {path.relative_to(root)}")
    if report.failures:
        return report
    ko_text = ko.read_text(encoding="utf-8")
    a = structure(en.read_text(encoding="utf-8"), en, root)
    b = structure(ko_text, ko, root)
    problems = []
    for key in a:
        if a[key] != b[key]:
            if isinstance(a[key], set):
                en_only = sorted(a[key] - b[key])  # type: ignore[operator]
                ko_only = sorted(b[key] - a[key])  # type: ignore[operator]
                detail = f"EN only {en_only}, KO only {ko_only}"
            else:
                detail = f"EN {a[key]}, KO {b[key]}"
            problems.append(f"{page}: {key} differ: {detail}")
    if problems and LAG_NOTE in ko_text:
        report.lagging.append(f"{page}.ko.md lags its English page")
    else:
        report.failures.extend(problems)
    return report


def check_translations(root: Path = REPO_ROOT) -> Report:
    """Check every translated pair and that no other ``.ko.md`` exists."""
    report = Report()
    listed = {f"{p}.ko.md" for p in TRANSLATED_PAGES}
    for path in tracked_files(root):
        rel = path.relative_to(root).as_posix()
        if rel.endswith(".ko.md") and rel not in listed:
            report.failures.append(
                f"{rel}: not in TRANSLATED_PAGES (scripts/check_docs.py)"
            )
    for page in TRANSLATED_PAGES:
        result = check_translation(page, root)
        report.failures += result.failures
        report.lagging += result.lagging
    return report


def main(root: Path = REPO_ROOT) -> int:
    links_report = check_links(root)
    translations = check_translations(root)
    for line in links_report.failures + translations.failures:
        print(f"FAIL {line}")
    for line in translations.lagging:
        print(f"LAG  {line}")
    counts = ", ".join(f"{n} {kind}" for kind, n in sorted(links_report.counts.items()))
    broken = len(links_report.failures)
    print(f"checked {counts}: {broken} broken")
    print(
        f"checked {len(TRANSLATED_PAGES)} translated pages: "
        f"{len(translations.failures)} failures, {len(translations.lagging)} lagging"
    )
    return 1 if broken or translations.failures else 0


if __name__ == "__main__":
    sys.exit(main())
