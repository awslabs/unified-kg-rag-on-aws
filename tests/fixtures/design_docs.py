# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read the design reference as heading blocks, and join its index and parts.

``docs/design.md`` is an index: every section heading of the reference with
a summary and a link. The text of each section lives in one part file
(``docs/design/*.md``, and ``docs/guides/extend.md`` for §15). Joining the
index and the parts (each moved section's text in place of its summary)
gives the reference as one document again, so a test can compare it with
the single file the parts were split from.
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass
from itertools import pairwise

from tests.fixtures.docs_check import check_docs

INDEX = "docs/design.md"
# A heading level the index lists: ## sections, ### and #### subsections.
_LEVELS = (2, 3, 4)
_ID_TAG = re.compile(r"^<a (?:id|name)=\"[^\"]+\"></a>$")
# An inline link's or image's target, up to its first space or ")".
_TARGET = re.compile(r"\]\(([^)\s]+)")


@dataclass(frozen=True)
class Block:
    """One heading and the text up to the next heading.

    ``prefix`` holds the ``<a id>`` lines (and the blank lines after them)
    that stand right before the heading: a legacy heading id that belongs
    to the heading, not to the section before it. The first block of a file
    has no heading: the title and the text before the first section.
    """

    heading: str | None
    prefix: str
    text: str


def heading_title(line: str) -> str:
    """The text of a heading line, without its ``#`` marks."""
    match = check_docs()._HEADING.match(line)
    assert match, line
    return match.group(2) or ""


def blocks(text: str) -> list[Block]:
    """``text`` cut at every ##, ### and #### heading outside code."""
    docs = check_docs()
    lines = text.splitlines(keepends=True)
    starts = []
    for number, line in docs.prose_lines(text):
        match = docs._HEADING.match(line)
        if match and len(match.group(1)) in _LEVELS:
            starts.append(number - 1)
    bounds = [0, *starts, len(lines)]
    chunks = [lines[a:b] for a, b in pairwise(bounds)]
    prefixes: list[list[str]] = [[] for _ in chunks]
    for i in range(1, len(chunks)):
        before = chunks[i - 1]
        cut = len(before)
        while cut and not before[cut - 1].strip():
            cut -= 1
        first = cut
        while first and _ID_TAG.match(before[first - 1].strip()):
            first -= 1
        if first < cut:
            prefixes[i], chunks[i - 1] = before[first:], before[:first]
    out = []
    for i, chunk in enumerate(chunks):
        heading = chunk[0].rstrip("\n") if i else None
        out.append(Block(heading, "".join(prefixes[i]), "".join(chunk)))
    return out


def canonical_links(text: str, path: str, design_files: set[str]) -> str:
    """``text`` (the file ``path``) with each inline link target written from
    the repository root, and a link into any design file read as a link
    into the index: where a link points, not how its file reaches it."""
    docs = check_docs()
    prose = {number for number, _ in docs.prose_lines(text)}
    base = posixpath.dirname(path)

    def canonical(target: str) -> str:
        if docs._SCHEME.match(target) or target.startswith("//"):
            return target
        link, _, fragment = target.partition("#")
        dest = posixpath.normpath(posixpath.join(base, link)) if link else path
        if dest in design_files and fragment:
            dest = INDEX
        return f"@{dest}" + (f"#{fragment}" if fragment else "")

    out = []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        if number in prose:
            blanked = docs._CODE_SPAN.sub(lambda m: " " * len(m.group()), line)
            pieces, last = [], 0
            for match in _TARGET.finditer(blanked):
                pieces += [line[last : match.start(1)], canonical(match.group(1))]
                last = match.end(1)
            line = "".join(pieces) + line[last:]
        out.append(line)
    return "".join(out)


def _normalized(block: Block, text: str) -> tuple[str | None, str, str]:
    # A part file ends in one newline where the section, inside the single
    # file, ended in the blank line before the next heading.
    return block.heading, block.prefix, text.rstrip("\n")


def as_one_file(text: str, design_files: set[str]) -> list[tuple[str | None, str, str]]:
    """The single-file reference as normalized blocks."""
    return [
        _normalized(b, canonical_links(b.text, INDEX, design_files))
        for b in blocks(text)
    ]


def join(
    index: str, parts: dict[str, str], design_files: set[str]
) -> list[tuple[str | None, str, str]]:
    """The index with each section's summary replaced by its text from the
    part that holds it, as normalized blocks (the parts' own titles and
    introductions dropped)."""
    held: dict[str, tuple[str, Block]] = {}
    for path, text in parts.items():
        for block in blocks(text)[1:]:
            assert block.heading not in held, f"{block.heading} in two parts"
            held[block.heading] = (path, block)  # type: ignore[index]
    out = []
    for block in blocks(index):
        if block.heading in held:
            path, part = held.pop(block.heading)
            text = canonical_links(part.text, path, design_files)
        else:
            text = canonical_links(block.text, INDEX, design_files)
        out.append(_normalized(block, text))
    assert not held, f"sections of no index heading: {sorted(held)}"
    return out
