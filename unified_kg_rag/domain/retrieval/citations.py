# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Citation markers in an answer (``search.citations``, docs/design.md §7.1).

The answer context numbers its evidence sections ``[1]``, ``[2]``, ... and
the answer prompt asks the model to cite them with the same markers in every
language. A marker is read the same in every answer language: after NFKC
normalization (full-width ``［１］`` becomes ``[1]``), square or lenticular
brackets (``【1】``) around one or more ASCII numbers separated by commas,
semicolons or ``、`` (``[1, 2]``, ``[1、2]``), or a short range (``[1-3]``,
up to :data:`MAX_RANGE` numbers). Adjacent markers (``[1][3]``) are two.

:class:`CitationMarkerScanner` reads the same markers from an answer that
arrives in chunks: a marker split across chunks is held back until it
closes, so the chunks yield exactly the markers of the whole text.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection

__all__ = [
    "MAX_MARKER_CHARS",
    "MAX_RANGE",
    "CitationMarkerScanner",
    "cited_markers",
    "strip_markers",
]

# The longest range a marker may span ([1-20]); a longer one ([1-500]) is
# not read as citing every number in it.
MAX_RANGE = 20
# A pending marker held back by the scanner longer than this is not one.
MAX_MARKER_CHARS = 48

_OPEN = "[【"
_CLOSE = "]】"
_NUMBER = r"[0-9]{1,4}"
_SEPARATOR = r"\s*[,;、]\s*"
_RANGE = r"\s*[-–~]\s*"
_ITEM = rf"{_NUMBER}(?:{_RANGE}{_NUMBER})?"
_MARKER = re.compile(
    rf"[{re.escape(_OPEN)}]\s*({_ITEM}(?:{_SEPARATOR}{_ITEM})*)\s*[{re.escape(_CLOSE)}]"
)
_ITEMS = re.compile(rf"({_NUMBER})(?:{_RANGE}({_NUMBER}))?")
# The start of a marker that has not closed yet: an opening bracket and only
# what a marker may hold after it.
_PENDING = re.compile(rf"[{re.escape(_OPEN)}][0-9\s,;、\-–~]*$")


# A marker as an answer writes it, before normalization (full-width
# brackets, digits and separators included), for :func:`strip_markers`. Up
# to three digits: a bracketed year ("[2020]") is text, not a marker.
_RAW_NUMBER = r"[0-9０-９]{1,3}"
_RAW_ITEM = rf"{_RAW_NUMBER}(?:\s*[-–~～－]\s*{_RAW_NUMBER})?"
_RAW_MARKER = re.compile(
    rf"\s*[\[［【]\s*{_RAW_ITEM}(?:\s*[,，;；、]\s*{_RAW_ITEM})*\s*[\]］】]"
)


def strip_markers(text: str) -> str:
    """``text`` without its citation markers (and the space before each), for
    scoring an answer's words against a reference (``answer_match``)."""
    return _RAW_MARKER.sub("", text or "")


def _numbers(group: str) -> list[int]:
    numbers: list[int] = []
    for match in _ITEMS.finditer(group):
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) else first
        if last < first or last - first >= MAX_RANGE:
            # A reversed or long range is two numbers, not every one between.
            numbers.extend(dict.fromkeys((first, last)))
            continue
        numbers.extend(range(first, last + 1))
    return numbers


def _scan(text: str, valid: Collection[int] | None) -> list[int]:
    found: list[int] = []
    for match in _MARKER.finditer(text):
        found.extend(n for n in _numbers(match.group(1)) if valid is None or n in valid)
    return found


def cited_markers(text: str, valid: Collection[int] | None = None) -> list[int]:
    """The marker numbers ``text`` cites, each once, in the order they first
    appear; only those in ``valid`` when it is given."""
    normalized = unicodedata.normalize("NFKC", text or "")
    return list(dict.fromkeys(_scan(normalized, valid)))


class CitationMarkerScanner:
    """The citation markers of an answer read chunk by chunk.

    ``feed`` returns the markers each chunk completes (in order, repeats
    included) and ``finish`` those of the held-back tail; together they are
    the markers of the joined text (``cited_markers`` without the
    deduplication). ``valid`` keeps only those numbers (the context's
    markers), as in :func:`cited_markers`.
    """

    def __init__(self, valid: Collection[int] | None = None) -> None:
        self._valid = None if valid is None else frozenset(valid)
        self._pending = ""
        self._seen: dict[int, None] = {}

    @property
    def markers(self) -> list[int]:
        """Every marker read so far, each once, in first-seen order."""
        return list(self._seen)

    def feed(self, chunk: str) -> list[int]:
        text = self._pending + unicodedata.normalize("NFKC", chunk or "")
        opened = max(text.rfind(c) for c in _OPEN)
        self._pending = ""
        if opened >= 0 and len(text) - opened <= MAX_MARKER_CHARS:
            tail = text[opened:]
            if _PENDING.match(tail):
                # Not closed yet: it may still become a marker.
                self._pending, text = tail, text[:opened]
        return self._read(text)

    def finish(self) -> list[int]:
        """The markers of the held-back tail (none if it never closed)."""
        text, self._pending = self._pending, ""
        return self._read(text)

    def _read(self, text: str) -> list[int]:
        found = _scan(text, self._valid)
        self._seen.update(dict.fromkeys(found))
        return found
