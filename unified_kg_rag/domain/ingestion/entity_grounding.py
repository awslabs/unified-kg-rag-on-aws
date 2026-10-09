# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provenance grounding for extracted entities.

The extraction LLM is asked to emit, for every entity, a verbatim ``source_text``
span copied from the chunk it read. A *grounded* entity is one whose evidence
span actually occurs in that chunk; an *ungrounded* one is a hallucination — the
model invented an entity from its own domain priors rather than the document
(for example, a contract term with a duration that no clause in the corpus
states).

This module holds the pure, technology-agnostic grounding check (no boto3 /
LangChain). It is deliberately conservative: it only *rejects* an entity when we
are confident the evidence is absent, and it degrades to "grounded" whenever the
signal is too weak to judge (no span supplied, very short span) so that turning
the gate on never silently deletes legitimate entities.
"""

from __future__ import annotations

import unicodedata

from unified_kg_rag.shared.utils.scripts import (
    drop_dense_script_spaces,
    has_dense_script,
    is_dense_script_char,
)

__all__ = [
    "char_bigram_overlap_ratio",
    "is_grounded",
    "normalize_for_grounding",
    "span_length",
    "token_overlap_ratio",
]


def normalize_for_grounding(text: str | None) -> str:
    """Casefold + NFKC normalize, drop punctuation, and collapse whitespace.

    Punctuation (Unicode category ``P*``) becomes a space, so "2년이다." and
    "2년이다" compare equal; symbols (category ``S*``, e.g. "$", "+") are
    content and are kept. Word boundaries are kept: grounding compares running prose, where
    collapsing every symbol would make unrelated spans look equal.
    Unicode-aware so CJK / accented text matches.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = "".join(
        " " if unicodedata.category(ch).startswith("P") else ch for ch in normalized
    )
    # Collapse all Unicode whitespace runs to a single space so spans that
    # differ only in wrapping/indentation still match.
    return " ".join(normalized.split())


def _tokens(text: str) -> list[str]:
    return normalize_for_grounding(text).split()


def span_length(span: str) -> int:
    """Length of a span in approximate tokens.

    A Han, Hangul or Kana character counts as one (one character is a
    syllable or morpheme, and ~1 model token), any other whitespace token as
    one. A Japanese or Chinese sentence has no spaces, so a whitespace count
    would call every such span one token long.
    """
    length = 0
    for token in _tokens(span):
        dense = sum(1 for ch in token if is_dense_script_char(ch))
        length += dense if dense else 1
    return length


def token_overlap_ratio(span: str, source: str) -> float:
    """Fraction of the span's tokens that also appear in the source text.

    Used as a fuzzy fallback when the span is not a verbatim substring (the LLM
    lightly paraphrased or fixed whitespace). Returns 0.0 for an empty span so
    an empty/whitespace span never counts as grounded via this path.
    """
    span_tokens = _tokens(span)
    if not span_tokens:
        return 0.0
    source_tokens = set(_tokens(source))
    hits = sum(1 for t in span_tokens if t in source_tokens)
    return hits / len(span_tokens)


def _bigrams(text: str) -> set[str]:
    return {text[i : i + 2] for i in range(len(text) - 1)} or {text}


def char_bigram_overlap_ratio(span: str, source: str) -> float:
    """Fraction of the span's character bigrams that also occur in the source.

    The fuzzy fallback for dense-script spans, compared with the spaces next
    to Han/Hangul/Kana letters removed. Whitespace tokens fail there: a
    Korean paraphrase changes the particles attached to every word
    ("보증 기간이" -> "보증기간은"), and Japanese/Chinese have no spaces.
    """
    span_text = drop_dense_script_spaces(normalize_for_grounding(span))
    if not span_text:
        return 0.0
    span_bigrams = _bigrams(span_text)
    source_bigrams = _bigrams(drop_dense_script_spaces(normalize_for_grounding(source)))
    return len(span_bigrams & source_bigrams) / len(span_bigrams)


def is_grounded(
    source_text: str | None,
    chunk_text: str,
    *,
    min_span_tokens: int = 4,
    min_overlap_ratio: float = 0.6,
) -> bool:
    """Decide whether an evidence ``source_text`` span is grounded in ``chunk_text``.

    Decision order (conservative — bias toward keeping the entity):
    1. No span supplied, or the chunk is empty → ``True`` (cannot judge; don't
       penalize models/configs that don't emit spans).
    2. Span shorter than ``min_span_tokens`` after normalization → ``True``
       (too short to distinguish a real short name from a coincidence; the
       confidence/threshold path guards these instead). Length is
       :func:`span_length`: a Han/Hangul/Kana character counts as a token.
    3. Verbatim (normalized) substring match → ``True``. Punctuation is
       ignored, and so is spacing next to Han/Hangul/Kana letters.
    4. Overlap fallback ≥ ``min_overlap_ratio`` → ``True`` (handles light
       paraphrase / whitespace edits): token overlap, or character-bigram
       overlap (:func:`char_bigram_overlap_ratio`) for dense-script spans.
    5. Otherwise → ``False`` (ungrounded; likely hallucinated).
    """
    if not source_text or not chunk_text:
        return True

    norm_span = normalize_for_grounding(source_text)
    norm_chunk = normalize_for_grounding(chunk_text)
    if not norm_span or not norm_chunk:
        return True

    if span_length(norm_span) < min_span_tokens:
        return True

    if has_dense_script(norm_span):
        if drop_dense_script_spaces(norm_span) in drop_dense_script_spaces(norm_chunk):
            return True
        return char_bigram_overlap_ratio(norm_span, norm_chunk) >= min_overlap_ratio

    if norm_span in norm_chunk:
        return True

    return token_overlap_ratio(source_text, chunk_text) >= min_overlap_ratio
