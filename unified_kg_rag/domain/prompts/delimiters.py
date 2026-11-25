# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep untrusted prompt inputs inside the tags that delimit them.

Prompts wrap corpus text, retrieved context and reports in a tag pair
(``<input_text>{input_text}</input_text>``) and tell the model that everything
inside is data. A value containing ``</input_text>`` would end that block
early, and whatever followed it would read as instructions. The chain builder
neutralises the delimiter tags of a prompt in every input value before the
template is formatted. Pure functions, no backend imports.
"""

import re
import unicodedata
from collections.abc import Iterable
from functools import lru_cache
from typing import NamedTuple

# A tag pair whose content holds a template placeholder and no other tag:
# ``<context>\n{context}\n</context>``.
_DELIMITED_PLACEHOLDER = re.compile(
    r"<([A-Za-z_][\w\-]*)>[^<]*?(?<!\{)\{[A-Za-z_]\w*\}(?!\})[^<]*?</\1>"
)


def delimiter_tags(*templates: str) -> frozenset[str]:
    """Names of the tags that delimit a placeholder in ``templates``."""
    return frozenset(
        match.group(1)
        for template in templates
        for match in _DELIMITED_PLACEHOLDER.finditer(template)
    )


def neutralise_delimiters(value: str, tags: Iterable[str]) -> str:
    """Escape the opening and closing forms of ``tags`` inside ``value``.

    ``</context>``, ``< / Context >`` and ``<context>`` become ``&lt;/context>``
    and so on (case-insensitive, whitespace-tolerant), so the value can no
    longer end its block or open a new one; other text, including other tags,
    is kept as is.

    A form that only looks like one is escaped too: characters shown as
    nothing or on the one before them (:data:`_HIDDEN_CATEGORIES`: a
    zero-width space, a tag character, a variation selector, an accent)
    inside it, a ``<`` or ``/`` written as a lookalike (fullwidth ``＜``,
    small ``﹤``, ``‹``, fullwidth ``／``), and a name that differs from a
    tag's only in non-ASCII letters or marks (``ｃontext`` with a fullwidth
    letter, ``cоntext`` with a Cyrillic ``о``, ``İnput_text`` with a dotted
    capital I): the tag names are ASCII, so such a word can only imitate
    one. Its ``<`` (or lookalike) becomes ``&lt;``.
    """
    names = frozenset(name.casefold() for name in tags)
    if not names:
        return value
    marks = _marks()
    pieces: list[str] = []
    kept = 0  # value[:kept] is in pieces
    resume = 0  # where the next form may start (after the last one's word)
    for opener in marks.opener.finditer(value):
        if opener.start() < resume:
            continue
        # The gap, an optional slash and its gap, then the word: as a
        # possessive "<gap>(?:/<gap>)?<word>+".
        start = _run_end(marks.gap, value, opener.end())
        if start < len(value) and value[start] in marks.slashes:
            start = _run_end(marks.gap, value, start + 1)
        end = _run_end(marks.word, value, start)
        if end == start:
            continue
        resume = end
        if _imitates(value[start:end], names):
            pieces += (value[kept : opener.start()], "&lt;")
            kept = opener.end()
    if not pieces:
        return value
    pieces.append(value[kept:])
    return "".join(pieces)


class _Marks(NamedTuple):
    opener: re.Pattern[str]  # a "<" or a lookalike
    slashes: frozenset[str]  # a "/" and its lookalikes
    # A run of the gap's characters (white space, hidden ones) and of a
    # tag-like word's (word characters, "-", hidden ones), the hidden ones
    # of the Basic Multilingual Plane alone (see _run_end).
    gap: re.Pattern[str]
    word: re.Pattern[str]


# The confusables of "<" and "/" in Unicode's UTS #39 data that NFKC leaves
# alone (MODIFIER LETTER LEFT ARROWHEAD, CANADIAN SYLLABICS PA, RUNIC LETTER
# KAUNA, the single and heavy left-pointing angle quotation marks; FRACTION
# SLASH, DIVISION SLASH, BIG SOLIDUS). Those NFKC folds to "<" or "/" (the
# fullwidth and small forms) are found in the character data.
_LESS_THAN_CONFUSABLES = "\u02c2\u1438\u16b2\u2039\u276e"
_SLASH_CONFUSABLES = "\u2044\u2215\u29f8"

# The categories of the characters a form may hide in, in any plane (the tag
# characters U+E0020-E007F and some format controls lie beyond the Basic
# Multilingual Plane): invisible format characters (Cf: zero-width space and
# joiners, bidi marks, the byte-order mark, the soft hyphen) and combining
# marks (Mn, Me: accents, variation selectors), which show on the character
# before them, a "<" too.
_HIDDEN_CATEGORIES = frozenset({"Cf", "Mn", "Me"})


def _hidden(char: str) -> bool:
    return unicodedata.category(char) in _HIDDEN_CATEGORIES


def _run_end(run: re.Pattern[str], value: str, at: int) -> int:
    """Where the run of ``run``'s characters at ``at`` ends, a hidden
    character beyond the Basic Multilingual Plane (a tag character) taken
    in too: ``run`` names the plane's own, read once, and those beyond it
    are read here as they come (rare, and reading every plane's character
    data would take a second)."""
    while True:
        at = run.match(value, at).end()  # type: ignore[union-attr]  # matches ""
        if at < len(value) and ord(value[at]) > 0xFFFF and _hidden(value[at]):
            at += 1
            continue
        return at


@lru_cache(maxsize=1)
def _marks() -> _Marks:
    """The characters shown as a "<" or a "/", and the runs of a gap and a
    word, read once from the Basic Multilingual Plane's character data (no
    compatibility mapping beyond it folds to "<" or "/")."""
    less_thans, slashes, hidden = ["<"], ["/"], []
    for code in range(0x80, 0x10000):
        char = chr(code)
        if _hidden(char):
            hidden.append(char)
        # A compatibility mapping to the character alone ("<wide> 003C"):
        # what NFKC folds to it.
        elif (mapping := unicodedata.decomposition(char)).endswith(" 003C"):
            less_thans.append(char)
        elif mapping.endswith(" 002F"):
            slashes.append(char)
    less_than = f"[{re.escape(''.join(less_thans) + _LESS_THAN_CONFUSABLES)}]"
    shown_as_nothing = re.escape("".join(hidden))
    return _Marks(
        re.compile(less_than),
        frozenset([*slashes, *_SLASH_CONFUSABLES]),
        re.compile(f"[\\s{shown_as_nothing}]*+"),
        re.compile(f"[\\w\\-{shown_as_nothing}]*+"),
    )


def _imitates(word: str, names: frozenset[str]) -> bool:
    """Whether ``word``, its invisible format characters dropped, NFKC-folded
    and casefolded, and its combining marks then dropped (those it has, and
    those casefolding adds: ``İ`` folds to ``i`` and a combining dot), is
    one of the tag ``names`` (casefolded ASCII), or differs from one only
    in non-ASCII letters where it keeps at least one of its ASCII
    characters (a word of another script alone imitates no tag)."""
    if word.isascii():  # nothing to fold or drop: most words
        return word.casefold() in names
    visible = "".join(c for c in word if unicodedata.category(c) != "Cf")
    folded = "".join(
        c for c in unicodedata.normalize("NFKC", visible).casefold() if not _hidden(c)
    )
    if folded in names:
        return True
    return any(
        len(folded) == len(name)
        and any(a == b for a, b in zip(folded, name, strict=True))
        and all(
            a == b or (not a.isascii() and a.isalpha())
            for a, b in zip(folded, name, strict=True)
        )
        for name in names
    )
