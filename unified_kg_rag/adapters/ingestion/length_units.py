# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the chunk size settings count (``processing.chunking.length_unit``).

``chars`` measures with ``len``. ``tokens`` measures with the local
script-aware estimate (:func:`estimate_token_count`), never a model's
tokenizer, so chunk boundaries stay deterministic and need no network.

The estimate is not additive: ``max(words, dense + other // 4, 1)`` of a
joined text can exceed the sum of its pieces' estimates (the ``// 4`` floors
per piece). LangChain's splitter budgets a chunk by that sum, so with the
estimate as its length function a chunk could exceed its size.
:class:`TokenBudgetTextSplitter` measures the joined text instead, from
per-piece counts kept in a sliding window, so every chunk it emits is within
``chunk_size`` estimated tokens.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
from typing import Any, NamedTuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from unified_kg_rag.adapters.llm.token_estimate import (
    dense_char_count,
    estimate_token_count,
)
from unified_kg_rag.domain.ingestion.chunk_length import LATIN_CHARS_PER_TOKEN
from unified_kg_rag.domain.models import ChunkLengthUnit


def chunk_length_function(unit: ChunkLengthUnit) -> Callable[[str], int]:
    """The measure of ``unit``."""
    return estimate_token_count if unit == ChunkLengthUnit.TOKENS else len


class _Piece(NamedTuple):
    """The counts the token estimate of a text is computed from."""

    dense: int
    other: int
    words: int
    # Whether the text starts / ends inside a word (non-whitespace), so a
    # neighbour's word continues it.
    opens_word: bool
    closes_word: bool

    @classmethod
    def of(cls, text: str) -> _Piece:
        dense = dense_char_count(text)
        return cls(
            dense=dense,
            other=len(text) - dense,
            words=len(text.split()),
            opens_word=bool(text) and not text[0].isspace(),
            closes_word=bool(text) and not text[-1].isspace(),
        )


class _Window:
    """The token estimate of a growing and shrinking concatenation.

    Of non-empty pieces: an empty one would hide the seam between its
    neighbours (the splitter drops empty splits and separators).
    """

    def __init__(self) -> None:
        self._pieces: deque[_Piece] = deque()
        self._dense = 0
        self._other = 0
        # Words of the pieces minus the words that span a seam.
        self._words = 0

    def _seam(self, left: _Piece, right: _Piece) -> int:
        return 1 if left.closes_word and right.opens_word else 0

    def push(self, piece: _Piece) -> None:
        if self._pieces:
            self._words -= self._seam(self._pieces[-1], piece)
        self._pieces.append(piece)
        self._dense += piece.dense
        self._other += piece.other
        self._words += piece.words

    def pop_front(self) -> None:
        piece = self._pieces.popleft()
        if self._pieces:
            self._words += self._seam(piece, self._pieces[0])
        self._dense -= piece.dense
        self._other -= piece.other
        self._words -= piece.words

    def tokens(self, *appended: _Piece) -> int:
        """The estimate of the window's text followed by ``appended``."""
        dense, other, words = self._dense, self._other, self._words
        last = self._pieces[-1] if self._pieces else None
        for piece in appended:
            if last is not None:
                words -= self._seam(last, piece)
            dense += piece.dense
            other += piece.other
            words += piece.words
            last = piece
        if dense + other == 0:
            return 0
        return max(words, dense + other // LATIN_CHARS_PER_TOKEN, 1)


class GrowingText:
    """A text measured by ``length_function`` as it grows at its end.

    In tokens the measure comes from running counts (:class:`_Window`), so
    each extension costs its new characters, not the whole text again. The
    text is measured only when asked.
    """

    def __init__(self, text: str, length_function: Callable[[str], int]) -> None:
        self.text = text
        self._length_function = length_function
        self._size: int | None = None
        self._window: _Window | None = None

    @property
    def size(self) -> int:
        if self._size is None:
            self._size = self._length_function(self.text)
        return self._size

    def _counts(self) -> _Window | None:
        """The running counts of the text, in tokens only."""
        if self._window is None and self._length_function is estimate_token_count:
            self._window = _Window()
            if self.text:
                self._window.push(_Piece.of(self.text))
        return self._window

    def size_of(self, longer: str) -> int:
        """The measure of ``longer``, a text that starts with this one."""
        window = self._counts()
        if window is None or not longer.startswith(self.text):
            return self._length_function(longer)
        tail = longer[len(self.text) :]
        return window.tokens(_Piece.of(tail)) if tail else self.size

    def extend(self, longer: str, size: int) -> None:
        """Become ``longer``, whose measure :meth:`size_of` gave as ``size``."""
        window = self._window
        if window is not None and longer.startswith(self.text):
            tail = longer[len(self.text) :]
            if tail:
                window.push(_Piece.of(tail))
        else:
            self._window = None
        self.text, self._size = longer, size


class TokenBudgetTextSplitter(RecursiveCharacterTextSplitter):
    """A recursive splitter whose chunks fit ``chunk_size`` estimated tokens.

    Splits like :class:`RecursiveCharacterTextSplitter` with
    ``length_function=estimate_token_count``; only the merge of the pieces
    into chunks differs, measuring the joined text rather than summing the
    pieces (see the module docstring). Overlap is bounded the same way.
    """

    def __init__(self, **kwargs: Any) -> None:
        # With keep_separator the recursive split leaves each separator in
        # a split and merges with separator "", which _merge_splits relies on.
        if not kwargs.get("keep_separator", True):
            raise ValueError("TokenBudgetTextSplitter needs keep_separator")
        super().__init__(length_function=estimate_token_count, **kwargs)

    def _merge_splits(self, splits: Iterable[str], separator: str) -> list[str]:
        docs: list[str] = []
        current: list[str] = []
        window = _Window()
        for split in splits:
            piece = _Piece.of(split)
            if current and window.tokens(piece) > self._chunk_size:
                doc = self._join_docs(current, separator)
                if doc is not None:
                    docs.append(doc)
                while current and (
                    window.tokens() > self._chunk_overlap
                    or window.tokens(piece) > self._chunk_size
                ):
                    current.pop(0)
                    window.pop_front()
            window.push(piece)
            current.append(split)
        doc = self._join_docs(current, separator)
        if doc is not None:
            docs.append(doc)
        return docs
