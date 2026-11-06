# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where sentences end, for cutting a text short without breaking one.

Script-aware, not language-specific: a full-width or ideographic stop
(``。！？．｡``) ends a sentence wherever it stands, since Chinese and Japanese
write no space after it; a Latin stop (``.!?…``, used by the Latin and
Cyrillic languages and by Korean) ends one only before whitespace or the end
of the text, and a ``.`` not before a lowercase letter, so neither ``3.5``
nor ``e.g. the`` is a boundary. Closing quotes and brackets after a stop
stay with its sentence, and a line break ends one too (a list item or a
heading has no stop).
"""

from __future__ import annotations

import re

__all__ = ["sentence_ends"]

_CLOSERS = "\"'’”)\\]}»」』〉》】）｣"
_DENSE_STOP = re.compile(f"[。！？．｡‼⁇-⁉][{_CLOSERS}]*")
_LATIN_STOP = re.compile(f"[.!?…]+[{_CLOSERS}]*(?=\\s|$)")
_LINE_BREAK = re.compile(r"[^\S\n]*\n")


def sentence_ends(text: str) -> list[int]:
    """The offsets in ``text`` just past the end of each sentence, ascending
    and distinct; ``text[:offset]`` is then whole sentences. The end of the
    text is not one unless a sentence ends there."""
    ends = {match.end() for match in _DENSE_STOP.finditer(text)}
    for match in _LATIN_STOP.finditer(text):
        rest = text[match.end() :].lstrip()
        # "e.g. the", "approx. three": a period before a lowercase word.
        if not (match.group().startswith(".") and rest[:1].islower()):
            ends.add(match.end())
    ends.update(match.start() for match in _LINE_BREAK.finditer(text))
    return sorted(end for end in ends if text[:end].strip())
