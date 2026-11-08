# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the documentation's links, fragments and EN/KO structure.

Every tracked ``*.md`` file is scanned for relative links, images,
``<a href>`` and reference-style link definitions: the target must exist and
be tracked by Git (a file, or a directory holding one; not through a tracked
symlink to an untracked file), and a ``#fragment`` into a Markdown file must
match one of its headings (GitHub slug rules, duplicate headings get ``-1``,
``-2``...) or an ``<a id>``/``<a name>``. An inline code span that names a
repository path (``unified_kg_rag/...``, ``docs/...``, a package layer such as
``domain/models/config/_aws.py``, a module such as ``ports/__init__``) must name a
tracked file or directory. Code and configuration under :data:`CODE_DIRS` and
:data:`CODE_FILES` that cite ``docs/*.md`` paths or this repository's
``blob/main/docs/...#fragment`` URLs are checked the same way. Each EN/KO pair
in :data:`TRANSLATED_PAGES` must keep the same structure (headings, code
blocks, table rows, link targets, inline code spans; every number of the
English page in the Korean one), its Korean page must link the Korean twin of
a translated page and mark a link to an English-only page "(영문)", unless the
Korean page carries :data:`LAG_NOTE`. ``tests/unit/test_docs_links.py`` and
``tests/unit/test_docs_translations.py`` run these checks in CI, and
``tests/unit/test_docs_facts.py`` checks the counts and lists the docs state
against the code.

The design reference is an index (:data:`DESIGN_INDEX`, every section heading)
over part files (:data:`DESIGN_PARTS`) that hold the sections' text.
:func:`design_sections` maps each section number to the file holding it, so
every ``§N`` reference in the docs and the code must name a section, and a
link whose text names ``§N`` must point at the index or at the part holding
it.

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
# repo-relative paths without the suffix: the Tier-1 pages. Adding a
# ``.ko.md`` file that is not listed here fails the check.
TRANSLATED_PAGES = [
    "README",
    "docs/index",
    "docs/getting-started/local",
    "docs/getting-started/aws",
    "docs/concepts/overview",
    "docs/concepts/retrieval",
    "docs/concepts/scopes-and-namespaces",
    "docs/guides/configure",
    "docs/guides/ingest",
    "docs/guides/multilingual",
    "docs/guides/query",
    "docs/guides/structured-queries",
    "docs/guides/incremental",
    "docs/guides/domain-adaptation",
    "docs/guides/evaluate",
    "docs/guides/mcp",
    "docs/guides/console",
]

# The note a Korean page carries while its translation lags the English page
# (CONTRIBUTING.md, Documentation); such a page is reported, not failed.
LAG_NOTE = "이 번역은 영문판보다 늦을 수 있습니다"

REPO_URL = "https://github.com/awslabs/unified-kg-rag-on-aws/blob/main/"
CODE_DIRS = ("unified_kg_rag", "scripts", "iac", "tests", ".github", "docker")
# Root files outside CODE_DIRS whose docs/*.md citations are checked too.
CODE_FILES = ("config-template.yaml", "pyproject.toml")
# Files whose doc paths are the checker's own test inputs, not citations.
CITATION_FIXTURES = (
    "tests/unit/test_changelog_section.py",
    "tests/unit/test_design_split.py",
    "tests/unit/test_docs_links.py",
    "tests/unit/test_docs_translations.py",
)
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
_A_HREF = re.compile(r"<a\s[^>]*?\bhref\s*=\s*[\"']([^\"']+)[\"']", re.I)
# A reference-style link definition: [label]: target "title".
_REF_DEF = re.compile(r"^ {0,3}\[[^\]]+\]:[ \t]*(<[^>\n]*>|\S+)", re.M)
# An inline code span's text.
_CODE_TEXT = re.compile(r"(`+)(?!`)([\s\S]*?)(?<!`)\1(?!`)")
# A code span that names a repository path: it starts at a top-level
# directory (or a package layer, read under unified_kg_rag/) and holds no
# placeholder, glob or space.
_PATH_SPAN = re.compile(r"[\w.-]+/(?:[\w.-]+/)*[\w.-]*(?:#[\w-]+)?")
_PATH_ROOTS = (
    "unified_kg_rag/",
    "scripts/",
    "iac/",
    "docs/",
    "tests/",
    "docker/",
    ".github/",
    "assets/",
)
_PACKAGE_LAYERS = (
    "domain/",
    "adapters/",
    "application/",
    "ports/",
    "shared/",
    "evaluation/",
)
# Code spans under those roots that are not paths of this repository: an
# example mount, a metric namespace, an example file name and two files of
# the upstream libraries the overview names relative to their own roots.
EXTERNAL_PATHS = frozenset(
    {
        "docker/ontology/procure.ttl",
        "unified_kg_rag/query",
        "docs/Terms.pdf",
        # Split so the citation check of this file does not read it as ours.
        "docs/index/" "byog.md",
        "evaluation/eval_rag_quality.py",
    }
)
_SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
# The design reference: the index lists every section heading, and each
# section's text lives in one part (§15, the extension guide, in the guides).
DESIGN_INDEX = "docs/design.md"
DESIGN_PARTS = ("docs/design/", "docs/guides/extend.md")
# A numbered heading ("6.1.1 Global search") and a reference to one ("§6.1.1").
_SECTION_NUMBER = re.compile(r"(\d+(?:\.\d+)*)\.?(?:[ \t]|$)")
_SECTION_REF = re.compile(r"§[ \t]?(\d+(?:\.\d+)*)")
# docs/<path>.md[#fragment] inside code, docstrings, comments and YAML. A
# heading id holds no "." (slugs drop it), so a sentence's full stop ends it.
_CODE_CITE = re.compile(r"(?<![\w-])docs/[\w./-]+?\.md(?:#[^\s\"'`)\]>,;.]+)?")
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


def _paragraphs(text: str, *, comments: bool = False) -> list[tuple[int, str]]:
    """The ``(first line, text)`` of each paragraph outside fenced code, with
    code spans blanked, and HTML comments unless ``comments`` (offsets and
    newlines kept)."""
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
        body = "\n".join(line for _, line in paragraph)
        for pattern in (_CODE_SPAN,) if comments else (_CODE_SPAN, _COMMENT):
            body = pattern.sub(lambda m: re.sub(r"[^\n]", " ", m.group()), body)
        out.append((paragraph[0][0], body))
    return out


def links(text: str, *, comments: bool = False) -> list[tuple[int, str, bool]]:
    """The ``(line, target, is_image)`` of every link outside code, and with
    ``comments`` also inside HTML comments (a commented-out image or link,
    such as a placeholder for a recording, must resolve once uncommented).

    Paragraphs are scanned whole, so a link whose text wraps is found.
    """
    out = []
    for first, body in _paragraphs(text, comments=comments):
        found = [(m.start(), m.group(1), True) for m in _IMAGE.finditer(body)]
        found += [(m.start(), m.group(1), False) for m in _LINK.finditer(body)]
        found += [(m.start(), m.group(1), True) for m in _IMG_SRC.finditer(body)]
        found += [(m.start(), m.group(1), False) for m in _A_HREF.finditer(body)]
        found += [(m.start(), m.group(1), False) for m in _REF_DEF.finditer(body)]
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


def _is_tracked(dest: Path, tracked: set[Path]) -> bool:
    """Whether ``dest`` (resolved) is a tracked file, or a directory holding
    one."""
    if dest.is_dir():
        return any(path.is_relative_to(dest) for path in tracked)
    return dest in tracked


def _check_target(
    root: Path,
    source: Path,
    where: str,
    target: str,
    base: Path,
    report: Report,
    tracked: set[Path] | None = None,
) -> None:
    path, _, fragment = target.partition("#")
    dest = (base / unquote(path)).resolve() if path else source
    if not dest.exists():
        report.failures.append(f"{where}: missing target: {target}")
    elif tracked is not None and not _is_tracked(dest, tracked):
        # On disk but not committed (untracked or ignored): CI would miss it.
        report.failures.append(f"{where}: target not tracked by Git: {target}")
    elif fragment and dest.suffix == ".md":
        known = {a.lower() for a in fragments(dest)}
        if unquote(fragment).lower() not in known:
            try:
                shown = dest.relative_to(root.resolve()).as_posix()
            except ValueError:
                shown = str(dest)
            report.failures.append(f"{where}: missing fragment #{fragment} in {shown}")


def code_paths(text: str) -> list[tuple[int, str]]:
    """The ``(line, path)`` of every inline code span outside fences that
    names a repository path (:data:`_PATH_ROOTS`, :data:`_PACKAGE_LAYERS`)."""
    out = []
    for number, line in prose_lines(text):
        for match in _CODE_TEXT.finditer(_COMMENT.sub("", line)):
            span = match.group(2).strip()
            if not _PATH_SPAN.fullmatch(span):
                continue
            path = span.partition("#")[0]
            if path.startswith(_PATH_ROOTS + _PACKAGE_LAYERS):
                out.append((number, path))
    return out


def _path_exists(root: Path, path: str, tracked: set[Path]) -> bool:
    """Whether a code span's path is tracked: as written, under the package
    for a layer path, or as a module (``x/y`` for ``x/y.py``, ``x/y.attr``)."""
    bases = [root / "unified_kg_rag"] if path.startswith(_PACKAGE_LAYERS) else []
    bases.append(root)
    stem = path.rstrip("/")
    candidates = [stem, f"{stem}.py"]
    if "." in stem.rsplit("/", 1)[-1]:
        candidates.append(stem.rsplit(".", 1)[0] + ".py")
    return any(
        (base / c).exists() and _is_tracked((base / c).resolve(), tracked)
        for base in bases
        for c in candidates
    )


def check_links(root: Path = REPO_ROOT) -> Report:
    """Check Markdown links and images, then code-cited doc paths."""
    report = Report()
    files = tracked_files(root)
    # A tracked symlink is not its target: one to an untracked file is
    # broken in CI (its target, when tracked, is listed itself).
    tracked = {p.resolve() for p in files if not p.is_symlink()}
    for md in (p for p in files if p.suffix == ".md"):
        rel = md.relative_to(root).as_posix()
        text = md.read_text(encoding="utf-8")
        for number, target, image in links(text, comments=True):
            base = md.parent
            if target.startswith(REPO_URL):
                target, base = target[len(REPO_URL) :], root
            elif _SCHEME.match(target) or target.startswith("//"):
                continue
            report.counts["images" if image else "links"] += 1
            _check_target(root, md, f"{rel}:{number}", target, base, report, tracked)
        for number, code_path in code_paths(text):
            if code_path in EXTERNAL_PATHS:
                continue
            report.counts["code paths"] += 1
            if not _path_exists(root, code_path, tracked):
                report.failures.append(
                    f"{rel}:{number}: missing code path: {code_path}"
                )
    code = [
        p
        for p in files
        if p.suffix != ".md"
        and (
            p.relative_to(root).parts[0] in CODE_DIRS
            or p.relative_to(root).as_posix() in CODE_FILES
        )
        and p.relative_to(root).as_posix() not in CITATION_FIXTURES
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
                    root, path, f"{rel}:{number}", match.group(), root, report, tracked
                )
    _check_section_refs(root, files, code, report)
    return report


def _section_numbers(text: str) -> list[str]:
    """The numbers of the numbered ##-and-deeper headings of a page."""
    out = []
    for level, title in headings(text):
        match = _SECTION_NUMBER.match(title)
        if level > 1 and match:
            out.append(match.group(1))
    return out


def design_sections(root: Path = REPO_ROOT) -> dict[str, str]:
    """Each section number of the design reference, and the repo-relative
    file holding its text: a part, or the index for a section it keeps."""
    sections: dict[str, str] = {}
    paths = [p.relative_to(root).as_posix() for p in tracked_files(root)]
    for rel in paths:
        if rel.endswith(".md") and rel.startswith(DESIGN_PARTS):
            text = (root / rel).read_text(encoding="utf-8")
            sections.update(dict.fromkeys(_section_numbers(text), rel))
    if DESIGN_INDEX in paths:
        text = (root / DESIGN_INDEX).read_text(encoding="utf-8")
        for number in _section_numbers(text):
            sections.setdefault(number, DESIGN_INDEX)
    return sections


def _check_section_refs(
    root: Path, files: list[Path], code: list[Path], report: Report
) -> None:
    """Every ``§N`` names a design section, and a link whose text names one
    points at the index or at the part holding it."""
    sections = design_sections(root)
    if not sections:
        return
    design = set(sections.values()) | {DESIGN_INDEX}
    for path in [p for p in files if p.suffix == ".md"] + code:
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for match in _SECTION_REF.finditer(line):
                report.counts["section refs"] += 1
                if match.group(1) not in sections:
                    report.failures.append(
                        f"{rel}:{number}: no design section §{match.group(1)}"
                    )
        if path.suffix != ".md":
            continue
        for first, paragraph in _paragraphs(text):
            for m in _LINK_TEXT.finditer(paragraph):
                link = m.group(2).strip("<>").partition("#")[0]
                if not link or _SCHEME.match(link):
                    continue
                try:
                    dest = (path.parent / unquote(link)).resolve()
                    target = dest.relative_to(root.resolve()).as_posix()
                except ValueError:
                    continue
                if target not in design or target == DESIGN_INDEX:
                    continue
                for ref in _SECTION_REF.findall(m.group(1)):
                    if sections.get(ref) != target:
                        line = first + paragraph.count("\n", 0, m.start())
                        report.failures.append(
                            f"{rel}:{line}: links §{ref} to {target}, but its "
                            f"text is in {sections.get(ref)}"
                        )


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
        "inline code": inline_code(text),
    }


def _prose(text: str) -> str:
    """The text outside fenced code blocks and HTML comments."""
    return _COMMENT.sub(" ", "\n".join(line for _, line in prose_lines(text)))


def inline_code(text: str) -> set[str]:
    """The distinct inline code spans, whitespace collapsed and ``<...>``
    placeholders (translated) blanked."""
    return {
        re.sub(r"<[^<>]*>", "<>", re.sub(r"\s+", " ", m.group(2).strip()))
        for m in _CODE_TEXT.finditer(_prose(text))
    }


_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
# Korean writes 100,000 as 10만.
_MAN = re.compile(r"(\d+(?:\.\d+)?)만")


def numbers(text: str, korean: bool = False) -> set[str]:
    """The distinct numbers of the prose outside code and link targets,
    thousands separators dropped (and ``N만`` read as ``N * 10,000``)."""
    prose = _CODE_TEXT.sub(" ", _prose(text))
    prose = re.sub(r"\]\([^)]*\)", "]", re.sub(r"<[^>]+>", " ", prose))
    if korean:
        prose = _MAN.sub(lambda m: f"{round(float(m.group(1)) * 10000)}", prose)
    return {n.replace(",", "") for n in _NUMBER.findall(prose)}


_LINK_TEXT = re.compile(r"(?<!!)\[((?:[^\[\]]|\[[^\[\]]*\])*)\]" + _TARGET)
# Where a Korean clause ends: a table cell, or a sentence's full stop.
_CLAUSE_END = re.compile(r"\||(?<=[다요])\.(?=\s|$)")
ENGLISH_MARK = "영문"


def korean_link_problems(text: str, page: Path, root: Path) -> list[str]:
    """Links of a Korean page that skip a Korean twin, or that reach an
    English-only page without "(영문)" in the same clause."""
    translated = {f"{p}.md" for p in TRANSLATED_PAGES}
    stem = page.relative_to(root).as_posix().removesuffix(".ko.md")
    rel_page = page.relative_to(root).as_posix()
    problems = []
    for first, paragraph in _paragraphs(text):
        for m in _LINK_TEXT.finditer(paragraph):
            path = m.group(2).strip("<>").partition("#")[0]
            if not path.endswith(".md") or _SCHEME.match(path):
                continue
            try:
                rel = (page.parent / unquote(path)).resolve()
                rel_s = rel.relative_to(root.resolve()).as_posix()
            except ValueError:
                continue
            if rel_s.endswith(".ko.md") or rel_s == f"{stem}.md":
                continue
            where = f"{rel_page}:{first + paragraph.count(chr(10), 0, m.start())}"
            if rel_s in translated:
                if m.group(1).strip() != "English":
                    problems.append(
                        f"{where}: links {rel_s}, which has a Korean twin: "
                        f"link {rel_s.removesuffix('.md')}.ko.md"
                    )
                continue
            starts = [0] + [e.end() for e in _CLAUSE_END.finditer(paragraph)]
            start = max(x for x in starts if x <= m.start())
            end = next(
                (e.start() for e in _CLAUSE_END.finditer(paragraph, m.end())),
                len(paragraph),
            )
            if ENGLISH_MARK not in paragraph[start:end]:
                problems.append(
                    f"{where}: link to the English-only {rel_s} needs "
                    f'"({ENGLISH_MARK})" in its clause'
                )
    return problems


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
    missing = numbers(en.read_text(encoding="utf-8")) - numbers(ko_text, korean=True)
    if missing:
        problems.append(f"{page}: numbers differ: EN only {sorted(missing)}")
    problems += korean_link_problems(ko_text, ko, root)
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
