# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provenance grounding for extracted entities.

The extraction LLM is asked to emit, for every entity, a verbatim ``source_text``
span copied from the chunk it read. A *grounded* entity is one whose evidence
span actually occurs in that chunk; an *ungrounded* one is a hallucination — the
model invented an entity from its own domain priors rather than the document
(observed in real E2E: a "Warranty Period / 24 months / Provisional Acceptance
Date" entity materialized from a corpus that contained no such clause).

This module holds the pure, technology-agnostic grounding check (no boto3 /
LangChain). It is deliberately conservative: it only *rejects* an entity when we
are confident the evidence is absent, and it degrades to "grounded" whenever the
signal is too weak to judge (no span supplied, very short span) so that turning
the gate on never silently deletes legitimate entities.
"""

from __future__ import annotations

import unicodedata

__all__ = ["normalize_for_grounding", "token_overlap_ratio", "is_grounded"]


def normalize_for_grounding(text: str | None) -> str:
    """Casefold + NFKC normalize and collapse whitespace for substring matching.

    Mirrors the spirit of :func:`normalize_name` but keeps word boundaries and
    does NOT strip punctuation-run-to-space aggressively beyond whitespace —
    grounding compares running prose, where collapsing every symbol would make
    unrelated spans look equal. Unicode-aware so CJK / accented text matches.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    # Collapse all Unicode whitespace runs to a single space so spans that
    # differ only in wrapping/indentation still match.
    return " ".join(normalized.split())


def _tokens(text: str) -> list[str]:
    return normalize_for_grounding(text).split()


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
       confidence/threshold path guards these instead).
    3. Verbatim (normalized) substring match → ``True``.
    4. Token-overlap fallback ≥ ``min_overlap_ratio`` → ``True`` (handles light
       paraphrase / whitespace edits).
    5. Otherwise → ``False`` (ungrounded; likely hallucinated).
    """
    if not source_text or not chunk_text:
        return True

    norm_span = normalize_for_grounding(source_text)
    norm_chunk = normalize_for_grounding(chunk_text)
    if not norm_span or not norm_chunk:
        return True

    if len(norm_span.split()) < min_span_tokens:
        return True

    if norm_span in norm_chunk:
        return True

    return token_overlap_ratio(source_text, chunk_text) >= min_overlap_ratio
