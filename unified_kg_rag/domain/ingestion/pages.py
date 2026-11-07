# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Page numbers of the text units of a paged document (docs/design.md §7.1).

A loader that emits one document per page (PDF) gives each page a 0-based
``page`` number. The document converter joins the pages into one text and
records where each starts (:data:`PAGE_OFFSETS_KEY`: ``[offset, page]``
pairs, 1-based pages, ascending offsets). The chunker then locates each
chunk in that text and stores the first and last page it covers in the text
unit's attributes (:data:`PAGE_START_KEY`, :data:`PAGE_END_KEY`); chunk
boundaries, texts and ids are unchanged.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

__all__ = [
    "PAGE_END_KEY",
    "PAGE_OFFSETS_KEY",
    "PAGE_START_KEY",
    "chunk_page_ranges",
    "page_of",
    "valid_offsets",
]

# Document metadata: where each page starts in ``page_content``.
PAGE_OFFSETS_KEY = "page_offsets"
# Text unit attributes: the first and last page (1-based) the unit covers.
PAGE_START_KEY = "page_start"
PAGE_END_KEY = "page_end"


def page_of(offsets: Sequence[Sequence[int]], position: int) -> int | None:
    """The page the character at ``position`` is on (``None`` before the
    first page starts)."""
    starts = [int(o[0]) for o in offsets]
    index = bisect.bisect_right(starts, position) - 1
    return int(offsets[index][1]) if index >= 0 else None


def valid_offsets(value: object) -> list[tuple[int, int]] | None:
    """``value`` as ``[(offset, page), ...]`` when it is a well-formed
    :data:`PAGE_OFFSETS_KEY` record (ascending offsets, positive pages)."""
    if not isinstance(value, list | tuple) or not value:
        return None
    pairs: list[tuple[int, int]] = []
    for item in value:
        if (
            not isinstance(item, list | tuple)
            or len(item) != 2
            or not all(isinstance(v, int) and not isinstance(v, bool) for v in item)
        ):
            return None
        offset, page = item
        if offset < 0 or page < 1 or (pairs and offset < pairs[-1][0]):
            return None
        pairs.append((offset, page))
    return pairs


def chunk_page_ranges(
    content: str, offsets: Sequence[Sequence[int]], chunks: Sequence[str]
) -> list[tuple[int, int] | None]:
    """The (first, last) page of each of ``chunks`` (in document order) in
    ``content``, ``None`` for a chunk not found in it.

    A chunk is searched from just after the previous found chunk's start (a
    chunk may overlap the one before it), so repeated text earlier in the
    document is never matched; a chunk not found leaves the search where it
    was, and no page is guessed for it.
    """
    ranges: list[tuple[int, int] | None] = []
    cursor = 0
    for chunk in chunks:
        text = chunk.strip()
        start = content.find(text, cursor) if text else -1
        if start < 0:
            ranges.append(None)
            continue
        cursor = start + 1
        first = page_of(offsets, start)
        last = page_of(offsets, start + len(text) - 1)
        if first is None or last is None:
            ranges.append(None)
            continue
        ranges.append((first, last))
    return ranges
