"""Splitting the query entity extraction's answer into entity names.

The entity prompt (``EntityExtractionPrompt``) asks for a comma-separated
list in the query language. A model writing Chinese or Japanese separates
items with the ideographic comma ``、``, the fullwidth comma ``，`` or a
semicolon (``；``/``;``), and some answer one item per line: a split on the
ASCII comma alone returns the whole list as one "entity", which then matches
no graph node. :func:`split_entity_list` splits on all of them.
"""

from __future__ import annotations

import re

from unified_kg_rag.domain.languages import profiles

# Item separators: ASCII, fullwidth, ideographic and Arabic commas, ASCII,
# fullwidth and Arabic semicolons, and line breaks.
_SEPARATORS = frozenset(",，、،;；؛\n\r")
# Quote pairs a model wraps an item (or the whole list) in, every
# language's (``LanguageProfile.quote_pairs``).
_QUOTES = {
    opening: closing
    for language in profiles()
    for opening, closing in language.quote_pairs
}
_OPENING_QUOTES = frozenset(_QUOTES)
_CLOSING_QUOTES = frozenset(_QUOTES.values())
# A list marker opening a line ("- ", "* ", "• ", "1. ", "2) ").
_LIST_MARKER = re.compile(r"^(?:[-*•・]|\d{1,3}[.)])\s+")
# What an item loses at its ends: whitespace, quotes and an ideographic
# full stop.
_ITEM_EDGES = re.compile(r"^[\s\"'“”‘’「」『』。]+|[\s\"'“”‘’「」『』。]+$")


def _unwrapped(text: str) -> str:
    """``text`` without one pair of quotes around all of it (the prompt's
    example shows the list quoted), when no other quote is inside."""
    if len(text) >= 2 and _QUOTES.get(text[0]) == text[-1]:
        inner = text[1:-1]
        if not any(c in _OPENING_QUOTES or c in _CLOSING_QUOTES for c in inner):
            return inner
    return text


def _items(text: str) -> list[str]:
    """The raw items of ``text``: split at every separator outside quotes,
    except an ASCII comma between two digits (``1,000``)."""
    items: list[str] = []
    current: list[str] = []
    closing: str | None = None
    for index, char in enumerate(text):
        if closing is not None:
            if char == closing:
                closing = None
            current.append(char)
            continue
        if char in _QUOTES and not "".join(current).strip():
            closing = _QUOTES[char]
            current.append(char)
            continue
        if char in _SEPARATORS:
            between_digits = (
                char == ","
                and 0 < index < len(text) - 1
                and text[index - 1].isdigit()
                and text[index + 1].isdigit()
            )
            if not between_digits:
                items.append("".join(current))
                current = []
                continue
        current.append(char)
    items.append("".join(current))
    return items


def split_entity_list(text: str) -> list[str]:
    """The entity names of the entity prompt's answer ``text``, in order,
    without repeats.

    Splits at commas (``,``, ``，``, ``、``), semicolons (``;``, ``；``) and
    line breaks, outside a quoted item (``"Vendor, Inc."`` stays one) and
    not inside a number (``1,000``). Each item loses its surrounding
    whitespace and quotes, a list marker and a trailing full stop; empty
    items are dropped.
    """
    names: list[str] = []
    for raw in _items(_unwrapped(text.strip())):
        item = _ITEM_EDGES.sub("", _LIST_MARKER.sub("", raw.strip()))
        if item:
            names.append(item)
    return list(dict.fromkeys(names))
