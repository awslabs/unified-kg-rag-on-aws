# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CHANGELOG.md's ``[Unreleased]`` section stays one coherent document.

Branches that each add entries and then rebase leave the section damaged in
ways a reader trusts and a reviewer misses: a ``(#PR)`` placeholder the
landing script never filled, two *Upgrading* steps with one number (an old
and a new version of the same instruction), a "step N" reference past the
last step (left by a step removed or merged), the same passage kept twice by
a conflict resolution (often on adjacent lines, both versions of one
wrapped line), a shorter fragment of an entry kept twice in it (the old and
the new tail of one sentence, the second starting a line), two entries of a
subsection that contradict each other ("no longer exports X" beside
"exports X", sharing a quoted message), a line a conflict resolution
joined unwrapped, an extra the install instructions do not name, and a
*Fixed*, *Changed*, *Deprecated* or *Removed* entry (a change a 0.2.3 user
compares with what they run) that names a config key 0.2.3 did not have: a
feature no release had belongs in its *Added* entry.
"""

from __future__ import annotations

import importlib
import re
import subprocess  # nosec B404 - fixed argv, no shell
import sys
import tomllib
import typing
from pathlib import Path

import pytest
from pydantic import BaseModel

from tests.fixtures.docs_check import REPO_ROOT

pytestmark = pytest.mark.unit

CHANGELOG = REPO_ROOT / "CHANGELOG.md"
# A passage this many words long repeated further than this many words away
# is a duplicate (a phrase repeated within one sentence is not). Words, not
# lines: both versions of one conflicting line sit on adjacent lines. A
# repeat that starts a line is one at any distance: a conflict resolution
# that kept both versions of a wrapped line starts the second on its own
# line, often fewer than SHINGLE_MIN_DISTANCE words after the first.
SHINGLE_WORDS = 14
SHINGLE_MIN_DISTANCE = 40
# Within one entry (a bullet or an Upgrading step), a run of this many words
# that comes again starting a line is the old and the new version of one
# wrapped sentence a merge kept both of; a run of ENTRY_REPEAT_WORDS words
# that comes again anywhere in it is one too.
FRAGMENT_WORDS = 6
ENTRY_REPEAT_WORDS = 10
# A quoted span this many words long that two contradicting entries share
# says they describe the same behaviour.
SHARED_SPAN_WORDS = 3
# Outside tables, no line is longer (the section wraps at 118).
MAX_LINE = 120
_PLACEHOLDER = re.compile(r"\(#\(?PR\)?\)|#\(PR\)|\(#PR\b", re.I)


def _unreleased(text: str) -> str:
    start = text.index("## [Unreleased]")
    end = re.search(r"^(?:## \[|\[Unreleased\]:)", text[start + 1 :], re.M)
    return text[start : start + 1 + end.start()] if end else text[start:]


def _subsection(section: str, title: str) -> str:
    """The body of ``section``'s ``### title``, empty when it has none (a
    section just after a release cut holds few headings)."""
    parts = section.split(f"\n### {title}\n", 1)
    return parts[1].split("\n### ", 1)[0] if len(parts) > 1 else ""


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
    ``SHINGLE_MIN_DISTANCE`` words after its first occurrence, or again at
    the start of a line."""
    words = [
        (word.lower().strip(".,;:()"), number)
        for number, line in enumerate(section.splitlines(), 1)
        for word in line.split()
        if re.search(r"\w", word)  # not a bullet or a dash
    ]
    line_starts = {
        i for i, (_, number) in enumerate(words) if i == 0 or words[i - 1][1] != number
    }
    first: dict[tuple[str, ...], int] = {}
    repeats = []
    for i in range(len(words) - SHINGLE_WORDS + 1):
        key = tuple(word for word, _ in words[i : i + SHINGLE_WORDS])
        if key in first and (i - first[key] > SHINGLE_MIN_DISTANCE or i in line_starts):
            repeats.append((words[first[key]][1], words[i][1], " ".join(key)))
        first.setdefault(key, i)
    return repeats


def _entries(section: str) -> list[list[tuple[int, str]]]:
    """The section's entries: each bullet or numbered step at the margin
    with its continuation lines, as ``(line number, line)``."""
    entries: list[list[tuple[int, str]]] = []
    for number, line in enumerate(section.splitlines(), 1):
        if re.match(r"(- |\d+\. )", line):
            entries.append([(number, line)])
        elif entries and line.startswith(" ") and line.strip():
            entries[-1].append((number, line))
        elif entries and entries[-1]:
            entries.append([])  # a heading, a blank or a paragraph ends it
    return [entry for entry in entries if entry]


def _words(entry: list[tuple[int, str]]) -> list[tuple[str, int, bool]]:
    """``(word, line number, starts its line)`` of an entry's words."""
    words = []
    for number, line in entry:
        first = True
        for word in line.split():
            if re.search(r"\w", word):  # not a bullet or a dash
                words.append((word.lower().strip(".,;:()"), number, first))
                first = False
    return words


def _repeated_fragments(section: str) -> list[tuple[int, int, str]]:
    """``(first line, repeat line, fragment)`` of every run of
    ``FRAGMENT_WORDS`` words an entry holds again starting a line, and of
    ``ENTRY_REPEAT_WORDS`` words it holds again anywhere (not overlapping
    the first)."""
    repeats = []
    for entry in _entries(section):
        words = _words(entry)
        for size, at_line_start in (
            (FRAGMENT_WORDS, True),
            (ENTRY_REPEAT_WORDS, False),
        ):
            first: dict[tuple[str, ...], int] = {}
            for i in range(len(words) - size + 1):
                key = tuple(word for word, _, _ in words[i : i + size])
                seen = first.setdefault(key, i)
                if i - seen >= size and (words[i][2] or not at_line_start):
                    repeats.append((words[seen][1], words[i][1], " ".join(key)))
    return sorted(set(repeats))


def _stem(word: str) -> str:
    for suffix in ("es", "s", "ed", "ing"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _contradictions(section: str) -> list[tuple[int, int, str]]:
    """``(line, line, verb)`` of two entries of one subsection where one
    says it "no longer <verb>s" what the other says it does, both quoting
    the same span of at least ``SHARED_SPAN_WORDS`` words."""
    lines = section.splitlines()
    # Each line's subsection: the count of `### ` headings up to it.
    subsection_of = [0]
    for line in lines:
        subsection_of.append(subsection_of[-1] + line.startswith("### "))
    found = []
    for subsection in sorted(set(subsection_of)):
        entries = [
            (entry[0][0], " ".join(line.strip() for _, line in entry))
            for entry in _entries(section)
            if subsection_of[entry[0][0]] == subsection
        ]
        spans = [
            {
                span
                for span in re.findall(r"`([^`]+)`", text)
                if len(span.split()) >= SHARED_SPAN_WORDS
            }
            for _, text in entries
        ]
        for a, (line_a, text_a) in enumerate(entries):
            denied = {
                _stem(m.group(1).lower())
                for m in re.finditer(r"\bno longer (\w+)", text_a)
            }
            for b, (line_b, text_b) in enumerate(entries):
                if a == b or not denied or not spans[a] & spans[b]:
                    continue
                # What the entry says, not what it quotes.
                prose = re.sub(r"`[^`]*`", "``", text_b)
                for m in re.finditer(r"\b(\w+)\b", prose):
                    negated = re.search(
                        r"(?:no longer|not|never|n't)\s+$", prose[: m.start()]
                    )
                    if _stem(m.group(1).lower()) in denied and not negated:
                        found.append((line_a, line_b, _stem(m.group(1).lower())))
                        break
    return found


def _long_lines(section: str) -> list[tuple[int, int]]:
    """``(line, length)`` of every line over ``MAX_LINE`` outside a table."""
    return [
        (number, len(line))
        for number, line in enumerate(section.splitlines(), 1)
        if len(line) > MAX_LINE and not line.lstrip().startswith("|")
    ]


# The last release, its commit and its config keys (dotted, every nested
# field), read from that commit's config models by
# ``config_keys_at`` (the CI checkout is shallow, so the list is kept).
RELEASE = "0.2.3"
RELEASE_COMMIT = "81ea4987"
RELEASE_KEYS = REPO_ROOT / "tests/fixtures/config_keys_0_2_3.txt"
# Keys a Fixed or Changed entry may name although 0.2.3 had none: the knob
# the change itself added to tune or undo it, or the new name of a 0.2.3 key
# (*Breaking changes* lists the renames). Each must still be named.
NEW_KEYS_IN_CHANGES = {
    "evaluation.answer_span_output_ceiling": "the short-answer ceiling it added",
    "evaluation.repeat_extractions": "how repeats extract, which the change added",
    "evaluation.run_order": "keeps the earlier, sequential order",
    "evaluation.run_subdirectories": "keeps the flat layout",
    "custom_prompts.text_chunking_": "the chunker prompt it made overridable",
    "custom_prompts.text_translation_": "the translation prompt it made overridable",
    "graph.community_detection.max_cluster_size": "the leaf bound the change added",
    "indexing.neptune.deterministic_order": "keeps the store's edge order",
    "models.embedding.dimension": "indexing.opensearch.embedding_dimension, renamed",
    "models.roles.fixing": "fixing.model_id, renamed",
    "processing.embedding_batch_size": "the bound the change added",
    "processing.embedding_concurrency": "the bound the change added",
    "search.auto_routable_strategies": "how auto opts out of a strategy",
    "search.local_search.seed_score_ratio": "the seed threshold the fix added",
    "search.output_ceilings": "the short-answer ceilings it added",
    "search.query_extraction_cache_size": "the cache the change added",
    "search.store_timeout_seconds": "the store timeout the fix added",
}
# A backticked config key, bare, set (``search.top_k=20``, ``search.top_k:
# 20``) or a family (``custom_prompts.text_chunking_*``); a call
# (``export_data()``) or a file is none.
_CONFIG_KEY = re.compile(r"([a-z_]+(?:\.[a-z0-9_]+)+)\*?(?:\s*[=:][\s\S]*)?")
_FILE_SUFFIXES = frozenset({"py", "md", "json", "jsonl", "yaml", "yml", "toml", "txt"})
PACKAGE = REPO_ROOT / "unified_kg_rag"


def config_keys_at(commit: str, root: Path) -> set[str]:
    """Every dotted config key of ``commit``'s ``Config``: its config models
    (``domain/models/config.py`` with the two modules it imports, read with
    ``git show``) imported in a package of their own."""
    package = root / "release_models"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in ("config", "evaluation", "retrieval"):
        source = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["git", "show", f"{commit}:unified_kg_rag/domain/models/{name}.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        (package / f"{name}.py").write_text(source)
    sys.path.insert(0, str(root))
    try:
        config = importlib.import_module("release_models.config").Config
    finally:
        sys.path.remove(str(root))
        for module in [m for m in sys.modules if m.startswith("release_models")]:
            del sys.modules[module]

    def models(annotation: object) -> list[type[BaseModel]]:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return [annotation]
        return [m for arg in typing.get_args(annotation) for m in models(arg)]

    keys: set[str] = set()

    def walk(model: type[BaseModel], prefix: str) -> None:
        for name, field in model.model_fields.items():
            keys.add(f"{prefix}{name}")
            for nested in models(field.annotation):
                walk(nested, f"{prefix}{name}.")

    walk(config, "")
    return keys


def _a_module(name: str) -> bool:
    """Whether the dotted ``name`` is a module or package of the package
    (``evaluation.gates``) or a file (``search.py``), not a key."""
    parts = name.split(".")
    path = PACKAGE.joinpath(*parts)
    return (
        parts[-1] in _FILE_SUFFIXES
        or path.is_dir()
        or path.with_suffix(".py").is_file()
    )


def _new_config_keys(section: str, known: set[str], top: set[str]) -> list[str]:
    """``<subsection>: <key>`` of each config key a Changed, Fixed,
    Deprecated or Removed entry names that is not in ``known``: a
    backticked dotted name, bare or set to a value, under a top-level
    section of ``top`` or of ``known`` (a section the release had and the
    code since removed), that names no module or file."""
    sections = top | {key for key in known if "." not in key}
    found = []
    for title in ("Changed", "Fixed", "Deprecated", "Removed"):
        for span in re.findall(r"`([^`]+)`", _subsection(section, title)):
            match = _CONFIG_KEY.fullmatch(span)
            if not match:
                continue
            key = match.group(1)
            if key.split(".")[0] in sections and key not in known:
                if not _a_module(key):
                    found.append(f"{title}: {key}")
    return found


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


def test_no_fragment_is_repeated_within_an_entry() -> None:
    repeats = _repeated_fragments(SECTION)
    assert not repeats, repeats[:3]


def test_no_two_entries_contradict() -> None:
    assert _contradictions(SECTION) == []


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


def test_fixed_and_changed_entries_name_only_keys_the_release_had() -> None:
    from unified_kg_rag.domain.models import Config

    known = set(RELEASE_KEYS.read_text(encoding="utf-8").split())
    new = _new_config_keys(SECTION, known, set(Config.model_fields))
    unexplained = sorted(
        {n for n in new if n.split(": ", 1)[1] not in NEW_KEYS_IN_CHANGES}
    )
    assert unexplained == [], f"a key {RELEASE} had not: move it to Added"
    named = {n.split(": ", 1)[1] for n in new}
    assert sorted(set(NEW_KEYS_IN_CHANGES) - named) == [], "drop these entries"


def test_the_release_keys_are_its_commit_s(tmp_path: Path) -> None:
    has = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
        ["git", "cat-file", "-e", f"{RELEASE_COMMIT}^{{commit}}"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if has.returncode:
        pytest.skip(f"a shallow checkout without {RELEASE_COMMIT}")
    keys = config_keys_at(RELEASE_COMMIT, tmp_path)
    assert {"search.local_search", "processing.chunking.chunk_overlap"} <= keys
    assert keys == set(RELEASE_KEYS.read_text(encoding="utf-8").split())


def test_the_new_key_scan() -> None:
    section = (
        "\n### Changed\n- `search.top_k` is 10; `search.new_knob: 3` sets it, "
        "`run-eval` and `metadata.degraded` are no keys.\n"
        "### Fixed\n- `indexing.old.thing` reads; `search.top_k=20` and "
        "`search.other=3` are set.\n"
        # A module, a file and a call are no keys.
        "- `evaluation.gates`, `search.py` and "
        "`graph.visualization.export_data()` changed.\n"
        # A section the release had and the code removed is still one.
        "### Removed\n- `fixing.model_new` is gone.\n"
        "### Security\n- `search.x`\n"
    )
    known = {"search", "search.top_k", "indexing", "indexing.old", "fixing"}
    top = {"search", "indexing", "evaluation", "graph"}
    assert _new_config_keys(section, known, top) == [
        "Changed: search.new_knob",
        "Fixed: indexing.old.thing",
        "Fixed: search.other",
        "Removed: fixing.model_new",
    ]
    # Just after a release cut the section may hold no Changed heading.
    assert _new_config_keys("\n### Added\n- `search.n`\n", known, top) == []


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
    # Within SHINGLE_MIN_DISTANCE words, mid-line: one sentence repeating
    # itself.
    near = f"- {passage} {' '.join(filler.split()[: -SHINGLE_WORDS])} {passage}\n"
    # Within it too, but starting a line: both versions of a wrapped line.
    near_line = f"- x {passage}\n  {passage}\n"
    # Both versions of one line, on adjacent lines, past it.
    adjacent = f"- {passage} {filler}\n  {passage}\n"
    lines = "\n".join(filler.split())
    far = f"- {passage}\n{lines}\n- {passage}\n"
    short = " ".join(passage.split()[:-1])
    assert _repeated_passages(near) == []
    assert [r[:2] for r in _repeated_passages(near_line)] == [(1, 2)]
    assert _repeated_passages(f"- {short}\n{lines}\n- {short}\n") == []
    assert [r[:2] for r in _repeated_passages(adjacent)] == [(1, 2)]
    assert [r[:2] for r in _repeated_passages(far)] == [(1, SHINGLE_MIN_DISTANCE + 2)]


def test_the_fragment_scan() -> None:
    # The old and the new tail of one wrapped sentence, both kept: the
    # second starts a line, fewer words than SHINGLE_WORDS long.
    kept_both = (
        "- Local mode runs with reranking off, logged once; `local` declares it.\n"
        "  with reranking off, logged once; `local` declares httpx.\n"
    )
    assert [r[:2] for r in _repeated_fragments(kept_both)] == [(1, 2)]
    # The same fragment in two entries, or mid-line in one, is wording.
    apart = (
        "- one stops before any model call (exit 2), naming it.\n"
        "- another stops before any model call (exit 2), naming it.\n"
    )
    mid_line = (
        "- it stops before any model call here, and then before any model call there.\n"
    )
    assert _repeated_fragments(apart) == []
    assert _repeated_fragments(mid_line) == []
    # A long run repeated anywhere in one entry is a duplicate too.
    run = " ".join(f"w{i}" for i in range(ENTRY_REPEAT_WORDS))
    assert [r[:2] for r in _repeated_fragments(f"- {run} x {run}\n")] == [(1, 1)]
    short = " ".join(run.split()[:-1])
    assert _repeated_fragments(f"- {short} x {short}\n") == []


def test_the_contradiction_scan() -> None:
    message = "`Communities must be detected before exporting`"
    no_longer = f"- A run no longer exports the file (or fails to, {message}).\n"
    exports = f"- A run exports the whole graph (or none, or {message}).\n"
    assert [r[2] for r in _contradictions(no_longer + exports)] == ["export"]
    # Without a shared quoted span the two describe different things.
    assert _contradictions(no_longer + "- A run exports the whole graph.\n") == []
    # A negated mention agrees.
    agrees = f"- It does not export it ({message}).\n"
    assert _contradictions(no_longer + agrees) == []
    # Other subsections are other documents.
    assert _contradictions(no_longer + "### Fixed\n" + exports) == []


def test_the_line_length_scan() -> None:
    long = "x" * (MAX_LINE + 1)
    text = f"- {'y' * (MAX_LINE - 2)}\n  {long}\n| {long} |\n"
    assert _long_lines(text) == [(2, MAX_LINE + 3)]
