# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The explicit decline marker of an answer (``search.abstention``).

Under ``search.abstention.mode: mark`` the answer prompt asks the model to
start a declined answer with :data:`ABSTENTION_MARKER`. Detection reads only
the first characters after NFKC normalization, so it is the same in every
answer language: the token itself stays ASCII, full-width brackets
(``［NO_ANSWER］``, common in CJK output) normalize to ASCII, lenticular
brackets (``【NO_ANSWER】``) are accepted as written, and leading Markdown
emphasis, quotes or a heading mark are skipped.

A strategy that judges its own context sets
``SearchResult.metadata[CONTEXT_SUFFICIENT_KEY] = False`` (and optionally
``INSUFFICIENCY_REASON_KEY``); the answer step then applies
``search.abstention.on_insufficient``.
"""

from __future__ import annotations

import re
import unicodedata

from unified_kg_rag.domain.prompts.retrieval import ABSTENTION_MARKER

__all__ = [
    "ABSTENTION_MARKER",
    "ABSTENTION_REASON_KEY",
    "CONTEXT_SUFFICIENT_KEY",
    "INSUFFICIENCY_REASON_KEY",
    "MARKER_LOOKAHEAD",
    "REASON_INSUFFICIENT_CONTEXT",
    "REASON_MARKER",
    "REASON_NO_CONTEXT",
    "MarkerStreamFilter",
    "split_marker",
]

# SearchResult.metadata keys a strategy sets to flag its context.
CONTEXT_SUFFICIENT_KEY = "context_sufficient"
INSUFFICIENCY_REASON_KEY = "insufficiency_reason"
# RAGOutput.metadata key naming why an answer is abstained, and its values.
ABSTENTION_REASON_KEY = "abstention_reason"
REASON_NO_CONTEXT = "no_context"
REASON_MARKER = "marker"
REASON_INSUFFICIENT_CONTEXT = "insufficient_context"

# Characters a streamed answer holds back, at most, before the marker is
# ruled in or out (leading decoration plus the token), and the characters
# detection reads: the stream and the whole answer decide alike.
MARKER_LOOKAHEAD = 64

# Decoration a model puts before the token: whitespace, Markdown emphasis,
# heading and quote marks, code ticks, quotes.
_LEAD = r"[\s*_#>`\"'“”‘’「」『』]*"
# The token, tolerant of case, a space or hyphen for the underscore and
# lenticular brackets; NFKC already turned full-width forms into ASCII.
_TOKEN = r"[\[【]\s*NO[\s_-]?ANSWER\s*[\]】]"
# What follows it: closing emphasis or quotes, then an optional separator.
_TAIL = r"[*_`\"'“”‘’」』]*\s*(?:[:\-–—|]\s*)?"
_MARKER = re.compile(_LEAD + _TOKEN + _TAIL, re.IGNORECASE)


def split_marker(answer: str) -> tuple[bool, str]:
    """``(True, the answer after the marker)`` when ``answer`` starts with
    the decline marker, else ``(False, answer)``.

    The remainder is the model's own explanation, stripped of surrounding
    whitespace; it may be empty.
    """
    end = _marker_end(answer)
    if end is None:
        return False, answer
    return True, answer[end:].strip()


def _marker_end(answer: str) -> int | None:
    """Where the decline marker that starts ``answer`` ends (``None``: it
    does not start with one). Only the first :data:`MARKER_LOOKAHEAD`
    characters are read, so a stream decides on what it holds back."""
    head = answer[:MARKER_LOOKAHEAD]
    normalized = unicodedata.normalize("NFKC", head)
    match = _MARKER.match(normalized)
    if match is None:
        return None
    # NFKC can change the length of what precedes the end of the match
    # (one full-width character stays one character, but a compatibility
    # ligature can expand), so the cut is found again on the original.
    return _original_length(head, match.end())


def _original_length(head: str, normalized_end: int) -> int:
    """How many characters of ``head`` normalize to its first
    ``normalized_end`` normalized characters."""
    for index in range(len(head) + 1):
        if len(unicodedata.normalize("NFKC", head[:index])) >= normalized_end:
            return index
    return len(head)


def _could_become_marker(text: str) -> bool:
    """Whether more characters could still make ``text`` start with the
    marker (it is a prefix of one)."""
    normalized = unicodedata.normalize("NFKC", text)
    lead = re.match(_LEAD, normalized)
    rest = normalized[lead.end() if lead else 0 :]
    if not rest:
        return True
    probe = re.sub(r"[\s_-]", "", rest).upper()
    return any(
        candidate.startswith(probe) for candidate in ("[NOANSWER]", "【NOANSWER】")
    )


class MarkerStreamFilter:
    """Strips the decline marker from a streamed answer.

    ``feed`` returns the text that may be shown now: nothing while the start
    of the answer could still be the marker (at most
    :data:`MARKER_LOOKAHEAD` characters; after the marker, until the first
    character of the explanation), then everything. ``marked`` tells, once
    decided, whether the answer started with the marker. With
    ``strip=False`` the text passes through unchanged (still detected).

    What it shows is what :func:`split_marker` leaves of the whole answer,
    up to the whitespace that ends it (a stream cannot take back what it
    showed): the explanation is cut at the end of the marker and only its
    leading whitespace is dropped, so a chunk boundary never joins two
    words.
    """

    def __init__(self, *, strip: bool = True) -> None:
        self._strip = strip
        self._buffer = ""
        self._decided = False
        self.marked = False

    def feed(self, chunk: str) -> str:
        if self._decided:
            return chunk
        self._buffer += chunk
        end = _marker_end(self._buffer)
        if end is not None:
            # Marked; the explanation starts at its first visible character
            # (more decoration or a separator may still follow the token).
            rest = self._buffer[end:].lstrip()
            return self._decide(True, rest) if rest else ""
        if len(self._buffer) < MARKER_LOOKAHEAD and _could_become_marker(self._buffer):
            return ""
        return self._decide(False, self._buffer)

    def finish(self) -> str:
        """The text still held back when the stream ends."""
        if self._decided:
            return ""
        marked, rest = split_marker(self._buffer)
        return self._decide(marked, rest if marked else self._buffer)

    def _decide(self, marked: bool, rest: str) -> str:
        self._decided = True
        self.marked = marked
        text = rest if marked and self._strip else self._buffer
        self._buffer = ""
        return text
