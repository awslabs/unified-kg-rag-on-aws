# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CHANGELOG.md's ``[Unreleased]`` section stays one coherent document.

Branches that each add entries and then rebase leave the section damaged in
ways a reader trusts and a reviewer misses: a ``(#PR)`` placeholder the
landing script never filled, two *Upgrading* steps with one number (an old
and a new version of the same instruction), a "step N" reference past the
last step (left by a step removed or merged), the same passage kept twice by
a conflict resolution (often on adjacent lines, both versions of one
wrapped line), a line a conflict resolution joined unwrapped, and an extra
the install instructions do not name.
"""

from __future__ import annotations

import re
import tomllib

import pytest

from tests.fixtures.docs_check import REPO_ROOT

pytestmark = pytest.mark.unit

CHANGELOG = REPO_ROOT / "CHANGELOG.md"
# A passage this many words long repeated further than this many words away
# is a duplicate (a phrase repeated within one sentence is not). Words, not
# lines: both versions of one conflicting line sit on adjacent lines.
SHINGLE_WORDS = 14
SHINGLE_MIN_DISTANCE = 40
# Outside tables, no line is longer (the section wraps at 118).
MAX_LINE = 120
_PLACEHOLDER = re.compile(r"\(#\(?PR\)?\)|#\(PR\)|\(#PR\b", re.I)


def _unreleased(text: str) -> str:
    start = text.index("## [Unreleased]")
    end = re.search(r"^(?:## \[|\[Unreleased\]:)", text[start + 1 :], re.M)
    return text[start : start + 1 + end.start()] if end else text[start:]


def _subsection(section: str, title: str) -> str:
    body = section.split(f"\n### {title}\n", 1)[1]
    return body.split("\n### ", 1)[0]


def _placeholders(section: str) -> list[str]:
    return [line for line in section.splitlines() if _PLACEHOLDER.search(line)]


def _step_numbers(upgrading: str) -> list[int]:
    return [int(n) for n in re.findall(r"^(\d+)\. ", upgrading, re.M)]


def _step_references(section: str) -> list[int]:
    """Every step number a "step N" or "steps N-M" names, across line
    breaks."""
    text = " ".join(section.split())
    return [
        int(n)
        for match in re.finditer(r"\bsteps?\s+(\d+)(?:\s*-\s*(\d+))?", text, re.I)
        for n in match.groups()
        if n
    ]


def _repeated_passages(section: str) -> list[tuple[int, int, str]]:
    """``(first line, repeat line, passage)`` of every run of
    ``SHINGLE_WORDS`` words that appears again more than
    ``SHINGLE_MIN_DISTANCE`` words after its first occurrence."""
    words = [
        (word.lower().strip(".,;:()"), number)
        for number, line in enumerate(section.splitlines(), 1)
        for word in line.split()
        if re.search(r"\w", word)  # not a bullet or a dash
    ]
    first: dict[tuple[str, ...], int] = {}
    repeats = []
    for i in range(len(words) - SHINGLE_WORDS + 1):
        key = tuple(word for word, _ in words[i : i + SHINGLE_WORDS])
        if key in first and i - first[key] > SHINGLE_MIN_DISTANCE:
            repeats.append((words[first[key]][1], words[i][1], " ".join(key)))
        first.setdefault(key, i)
    return repeats


def _long_lines(section: str) -> list[tuple[int, int]]:
    """``(line, length)`` of every line over ``MAX_LINE`` outside a table."""
    return [
        (number, len(line))
        for number, line in enumerate(section.splitlines(), 1)
        if len(line) > MAX_LINE and not line.lstrip().startswith("|")
    ]


def _extras() -> list[str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as file:
        return list(tomllib.load(file)["project"]["optional-dependencies"])


SECTION = _unreleased(CHANGELOG.read_text(encoding="utf-8"))


def test_the_unreleased_section_has_no_pr_placeholder() -> None:
    assert _placeholders(SECTION) == []


def test_the_upgrading_steps_are_numbered_once_each_in_order() -> None:
    steps = _step_numbers(_subsection(SECTION, "Upgrading"))
    assert steps, "no numbered Upgrading step found"
    assert steps == list(range(1, len(steps) + 1)), steps


def test_every_step_reference_names_a_step() -> None:
    last = max(_step_numbers(_subsection(SECTION, "Upgrading")))
    references = _step_references(SECTION)
    assert references, "no step reference found"
    assert [n for n in references if not 1 <= n <= last] == []


def test_no_passage_is_repeated() -> None:
    repeats = _repeated_passages(SECTION)
    assert not repeats, repeats[:3]


def test_no_line_is_over_long() -> None:
    assert _long_lines(SECTION) == []


def test_every_extra_is_named_where_the_section_lists_them() -> None:
    breaking = _subsection(SECTION, "Breaking changes")
    (extras_entry,) = [
        entry for entry in breaking.split("\n- ") if "Extras: `" in entry
    ]
    step_one = _subsection(SECTION, "Upgrading").split("\n1. ", 1)[1]
    step_one = step_one.split("\n2. ", 1)[0]
    for extra in _extras():
        assert f"`{extra}`" in extras_entry, f"Breaking extras omit `{extra}`"
        assert f"`{extra}`" in step_one, f"Upgrading step 1 omits `{extra}`"


@pytest.mark.parametrize(
    ("line", "flagged"),
    [
        ("  extracted again (#(PR)).", True),
        ("  extracted again (#PR).", True),
        ("- (#PR) citations: answers", True),
        ("  extracted again (#628).", False),
        ("  the PR body names it", False),
    ],
)
def test_the_placeholder_scan(line: str, flagged: bool) -> None:
    assert bool(_placeholders(line)) is flagged


def test_the_step_scan_sees_a_repeated_number() -> None:
    text = "1. one\n   more\n2. two\n2. two again\n3. three\n"
    assert _step_numbers(text) == [1, 2, 2, 3]


def test_the_step_reference_scan() -> None:
    text = "delete it in step\n   13); follows steps 6-9 above; Step 2 first"
    assert _step_references(text) == [13, 6, 9, 2]
    assert _step_references("a two-step upgrade, 3 steps") == []


def test_the_repeat_scan() -> None:
    passage = " ".join(f"w{i}" for i in range(SHINGLE_WORDS))
    filler = " ".join(f"f{i}" for i in range(SHINGLE_MIN_DISTANCE))
    # Within SHINGLE_MIN_DISTANCE words: one sentence repeating itself.
    near = f"- {passage} {' '.join(filler.split()[: -SHINGLE_WORDS])}\n  {passage}\n"
    # Both versions of one line, on adjacent lines, past it.
    adjacent = f"- {passage} {filler}\n  {passage}\n"
    lines = "\n".join(filler.split())
    far = f"- {passage}\n{lines}\n- {passage}\n"
    short = " ".join(passage.split()[:-1])
    assert _repeated_passages(near) == []
    assert _repeated_passages(f"- {short}\n{lines}\n- {short}\n") == []
    assert [r[:2] for r in _repeated_passages(adjacent)] == [(1, 2)]
    assert [r[:2] for r in _repeated_passages(far)] == [(1, SHINGLE_MIN_DISTANCE + 2)]


def test_the_line_length_scan() -> None:
    long = "x" * (MAX_LINE + 1)
    text = f"- {'y' * (MAX_LINE - 2)}\n  {long}\n| {long} |\n"
    assert _long_lines(text) == [(2, MAX_LINE + 3)]
