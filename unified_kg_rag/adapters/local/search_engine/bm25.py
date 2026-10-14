# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""BM25 scoring as OpenSearch computes it (Lucene's ``LegacyBM25Similarity``).

Per query term, with ``N`` the documents that have the field, ``n`` those of
them that contain the term, ``tf`` its frequency in the document, ``dl`` the
document's field length in tokens and ``avgdl`` the mean length over the
``N`` documents::

    idf   = ln(1 + (N - n + 0.5) / (n + 0.5))
    score = idf * (k1 + 1) * tf / (tf + k1 * (1 - b + b * dl / avgdl))

with ``k1 = 1.2`` and ``b = 0.75``. Lucene 8 dropped the ``(k1 + 1)`` factor
from ``BM25Similarity``, but OpenSearch's default similarity keeps it (its
explain output shows it as a boost of 2.2); it scales every score alike, so
it changes the scores, not the ranking. Lucene stores ``dl`` as a one-byte
norm (``SmallFloat.intToByte4``), so a long field's length is rounded down to
one of 256 values; ``quantized_length`` does the same so scores match.
Statistics are per field, per index. A query term repeated in the query counts
once per repetition, as the clauses of Lucene's boolean query do.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable

K1 = 1.2
B = 0.75


def _long_to_int4(value: int) -> int:
    bits = value.bit_length()
    if bits < 4:
        return value
    shift = bits - 4
    encoded = (value >> shift) & 0x07
    return encoded | ((shift + 1) << 3)


def _int4_to_long(value: int) -> int:
    bits = value & 0x07
    shift = (value >> 3) - 1
    return bits if shift == -1 else (bits | 0x08) << shift


_MAX_INT4 = _long_to_int4(2**31 - 1)
_NUM_FREE_VALUES = 255 - _MAX_INT4


def quantized_length(length: int) -> int:
    """``length`` after Lucene's norm round trip (``intToByte4``/``byte4ToInt``)."""
    if length < _NUM_FREE_VALUES:
        return length
    encoded = _NUM_FREE_VALUES + _long_to_int4(length - _NUM_FREE_VALUES)
    return _NUM_FREE_VALUES + _int4_to_long(encoded - _NUM_FREE_VALUES)


def idf(doc_freq: int, doc_count: int) -> float:
    return math.log(1 + (doc_count - doc_freq + 0.5) / (doc_freq + 0.5))


def term_score(
    tf: int,
    length: int,
    avg_length: float,
    term_idf: float,
    k1: float = K1,
    b: float = B,
) -> float:
    """One query term's score in one document (``length`` already quantized)."""
    return term_idf * (k1 + 1) * tf / (tf + k1 * (1 - b + b * length / avg_length))


class FieldIndex:
    """The inverted index and statistics of one text field of one index."""

    def __init__(self) -> None:
        self.postings: dict[str, dict[str, int]] = {}
        self.lengths: dict[str, int] = {}
        self._total_terms = 0

    def add(self, doc_id: str, tokens: list[str]) -> None:
        """Index ``tokens`` as the field of ``doc_id`` (a document added once)."""
        if not tokens:
            return  # Lucene counts only documents with a term in the field.
        for term, count in Counter(tokens).items():
            self.postings.setdefault(term, {})[doc_id] = count
        self.lengths[doc_id] = quantized_length(len(tokens))
        self._total_terms += len(tokens)

    @property
    def doc_count(self) -> int:
        return len(self.lengths)

    @property
    def avg_length(self) -> float:
        return self._total_terms / self.doc_count if self.doc_count else 0.0

    def docs_with(self, term: str) -> Iterable[str]:
        return self.postings.get(term, {}).keys()

    def score(self, query_tokens: list[str]) -> dict[str, float]:
        """Summed term scores of every document matching any query token."""
        scores: dict[str, float] = {}
        avg_length = self.avg_length
        for term in query_tokens:
            postings = self.postings.get(term)
            if not postings:
                continue
            term_idf = idf(len(postings), self.doc_count)
            for doc_id, tf in postings.items():
                scores[doc_id] = scores.get(doc_id, 0.0) + term_score(
                    tf, self.lengths[doc_id], avg_length, term_idf
                )
        return scores
