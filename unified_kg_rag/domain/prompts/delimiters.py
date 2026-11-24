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

    A form that only looks like one is escaped too: invisible format
    characters (a zero-width space) inside it, a ``<`` or ``/`` written as a
    lookalike (fullwidth ``＜``, small ``﹤``, ``‹``, fullwidth ``／``), and a
    name that differs from a tag's only in non-ASCII letters (``ｃontext``
    with a fullwidth letter, ``cоntext`` with a Cyrillic ``о``): the tag
    names are ASCII, so such a word can only imitate one. Its ``<`` (or
    lookalike) becomes ``&lt;``.
    """
    names = frozenset(name.casefold() for name in tags)
    if not names:
        return value
    marks = _marks()
    if marks.opener.search(value) is None:
        return value

    def escaped(match: re.Match[str]) -> str:
        if not _imitates(match.group(3), names, marks.invisible):
            return match.group(0)
        return "&lt;" + match.group(2) + match.group(3)

    return marks.tag.sub(escaped, value)


class _Marks(NamedTuple):
    invisible: frozenset[str]
    opener: re.Pattern[str]  # a "<" or a lookalike
    tag: re.Pattern[str]  # an opener, a gap, an optional slash, a word


# The confusables of "<" and "/" in Unicode's UTS #39 data that NFKC leaves
# alone (MODIFIER LETTER LEFT ARROWHEAD, CANADIAN SYLLABICS PA, RUNIC LETTER
# KAUNA, the single and heavy left-pointing angle quotation marks; FRACTION
# SLASH, DIVISION SLASH, BIG SOLIDUS). Those NFKC folds to "<" or "/" (the
# fullwidth and small forms) are found in the character data.
_LESS_THAN_CONFUSABLES = "\u02c2\u1438\u16b2\u2039\u276e"
_SLASH_CONFUSABLES = "\u2044\u2215\u29f8"


@lru_cache(maxsize=1)
def _marks() -> _Marks:
    """The characters shown as a "<" or a "/", and the invisible format
    characters (category Cf: zero-width space and joiners, bidi marks, the
    byte-order mark, the soft hyphen), read once from the Basic Multilingual
    Plane's character data."""
    less_thans, slashes, invisible = ["<"], ["/"], []
    for code in range(0x80, 0x10000):
        char = chr(code)
        if unicodedata.category(char) == "Cf":
            invisible.append(char)
        # A compatibility mapping to the character alone ("<wide> 003C"):
        # what NFKC folds to it.
        elif (mapping := unicodedata.decomposition(char)).endswith(" 003C"):
            less_thans.append(char)
        elif mapping.endswith(" 002F"):
            slashes.append(char)
    less_than = f"[{re.escape(''.join(less_thans) + _LESS_THAN_CONFUSABLES)}]"
    slash = f"[{re.escape(''.join(slashes) + _SLASH_CONFUSABLES)}]"
    hidden = re.escape("".join(invisible))
    gap = f"[\\s{hidden}]*+"
    return _Marks(
        frozenset(invisible),
        re.compile(less_than),
        re.compile(f"({less_than})({gap}(?:{slash}{gap})?)([\\w\\-{hidden}]++)"),
    )


def _imitates(word: str, names: frozenset[str], invisible: frozenset[str]) -> bool:
    """Whether ``word``, its invisible characters dropped and NFKC-folded,
    is one of the tag ``names`` (casefolded ASCII), or differs from one
    only in non-ASCII letters where it keeps at least one of its ASCII
    characters (a word of another script alone imitates no tag)."""
    folded = unicodedata.normalize(
        "NFKC", "".join(c for c in word if c not in invisible)
    ).casefold()
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
