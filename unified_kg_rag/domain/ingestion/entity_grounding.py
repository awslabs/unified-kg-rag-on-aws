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

import re
import unicodedata
from functools import lru_cache

from unified_kg_rag.shared.utils.scripts import (
    drop_dense_script_spaces,
    has_dense_script,
    is_dense_script_char,
)

__all__ = [
    "char_bigram_overlap_ratio",
    "is_grounded",
    "is_name_grounded",
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


# Name matching (is_name_grounded) folds both sides further than
# normalize_for_grounding: accents, dots, hyphens and apostrophes vary between
# an LLM's spelling of a name and the document's ("Nestlé"/"Nestle",
# "U.S."/"US", "E-Mail"/"email").

# Characters that join a word rather than separate words: dots, apostrophes
# and dashes (category Pd, checked separately). One view of the text removes
# them ("u.s." -> "us"), the other turns them into spaces ("vendor-buyer" ->
# "vendor buyer"), so either spelling of a name matches either of the text.
_NAME_JOINERS = frozenset(".'\u2019\u02bc")
_POSSESSIVE = re.compile(r"['\u2019\u02bc]s\b")
_PARENTHETICAL = re.compile(r"\(([^()]*)\)")
# A last-token spelling and its canonical short form.
_CORPORATE_SUFFIXES = {
    "corporation": "corp",
    "incorporated": "inc",
    "company": "co",
    "limited": "ltd",
}
# Shortest stem a trailing "s"/"es" may be added to: "us" is not "u" plural.
_MIN_PLURAL_STEM = 3


def _strip_marks(text: str) -> str:
    """NFKD, drop combining marks on non-dense letters, recompose (NFC).

    Marks on Kana (dakuten) are kept: "ガ" and "カ" are different letters.
    """
    kept: list[str] = []
    base_dense = False
    for ch in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(ch):
            if base_dense:
                kept.append(ch)
            continue
        base_dense = is_dense_script_char(ch)
        kept.append(ch)
    return unicodedata.normalize("NFC", "".join(kept))


def _split_scripts(token: str) -> list[str]:
    """Split a token where it changes between dense and other script."""
    parts: list[str] = []
    start = 0
    for i in range(1, len(token)):
        if is_dense_script_char(token[i]) != is_dense_script_char(token[i - 1]):
            parts.append(token[start:i])
            start = i
    parts.append(token[start:])
    return parts


@lru_cache(maxsize=256)
def _name_tokens(text: str, join: bool) -> tuple[str, ...]:
    """Fold ``text`` for name matching and split it into tokens.

    Accents stripped and casefolded; a possessive "'s" dropped; joiners
    removed (``join``) or turned into spaces; other punctuation separates
    tokens, symbols are content. A token never mixes dense and other script,
    so "vendor는" yields "vendor" and "는".
    """
    folded = _POSSESSIVE.sub("", _strip_marks(text).casefold())
    chars: list[str] = []
    for ch in folded:
        category = unicodedata.category(ch)
        if ch in _NAME_JOINERS or category == "Pd":
            chars.append("" if join else " ")
        elif category.startswith("P"):
            chars.append(" ")
        else:
            chars.append(ch)
    return tuple(
        part for token in "".join(chars).split() for part in _split_scripts(token)
    )


def _is_dense(token: str) -> bool:
    return is_dense_script_char(token[0])


def _same_last_token(name_token: str, text_token: str, *, suffix: bool) -> bool:
    """Equal up to a trailing "s"/"es", or (``suffix``) a corporate suffix.

    The suffix table applies only after a main name: a bare "Co" is not
    "company".
    """
    if name_token == text_token:
        return True
    if suffix and _CORPORATE_SUFFIXES.get(
        name_token, name_token
    ) == _CORPORATE_SUFFIXES.get(text_token, text_token):
        return True
    short, long = sorted((name_token, text_token), key=len)
    return len(short) >= _MIN_PLURAL_STEM and long in (f"{short}s", f"{short}es")


def _latin_segment_found(segment: list[str], tokens: tuple[str, ...]) -> bool:
    """Whether ``segment`` occurs as consecutive whole tokens of ``tokens``."""
    *head, last = segment
    width = len(segment)
    for start in range(len(tokens) - width + 1):
        if list(tokens[start : start + width - 1]) == head and _same_last_token(
            last, tokens[start + width - 1], suffix=width > 1
        ):
            return True
    return False


def _segments(tokens: tuple[str, ...]) -> list[tuple[bool, list[str]]]:
    """Group consecutive tokens of one script: (is_dense, tokens)."""
    groups: list[tuple[bool, list[str]]] = []
    for token in tokens:
        dense = _is_dense(token)
        if groups and groups[-1][0] == dense:
            groups[-1][1].append(token)
        else:
            groups.append((dense, [token]))
    return groups


def _dense_segment_found(word: str, tokens: tuple[str, ...], dense_text: str) -> bool:
    """Substring of the space-dropped text; one character only as a token."""
    if len(word) == 1:
        return word in tokens
    return word in dense_text


def _single_name_grounded(name: str, chunk_text: str) -> bool:
    for join in (True, False):
        name_tokens = _name_tokens(name, join)
        if not name_tokens:
            continue
        text_tokens = _name_tokens(chunk_text, join)
        dense_text = drop_dense_script_spaces(" ".join(text_tokens))
        if all(
            (
                _dense_segment_found("".join(segment), text_tokens, dense_text)
                if dense
                else _latin_segment_found(segment, text_tokens)
            )
            for dense, segment in _segments(name_tokens)
        ):
            return True
    return False


def is_name_grounded(name: str | None, chunk_text: str) -> bool:
    """Decide whether an entity ``name`` itself occurs in ``chunk_text``.

    :func:`is_grounded` checks an evidence span and calls a short span
    grounded because it is too short to judge, so it cannot vouch for a name
    (most names are one to three tokens). Both sides are folded: NFKD with
    combining marks stripped and casefolded ("Nestlé" = "Nestle"), a
    possessive "'s" dropped, and dots, hyphens and apostrophes either
    removed or read as spaces ("U.S." = "US", "Co-Op" = "CoOp", "Vendor" in
    "Vendor-Buyer").

    The name is matched per script segment:

    - other scripts (Latin, digits): whole tokens, consecutive in the text,
      so "Ven" does not match "Vendor" nor "AI" "aim". The last token may
      differ by a trailing "s"/"es" ("Vendors" = "Vendor") or, after a main
      name, a corporate suffix (Corporation/Corp, Incorporated/Inc,
      Company/Co, Limited/Ltd).
    - Han/Hangul/Kana: a substring of the text with the spaces next to those
      letters removed, since particles attach to the name ("벤더는" contains
      "벤더"). A one-character segment matches only as a standalone token: as
      a substring it would match almost any text.

    A name with a parenthetical alias ("한빛전자(Hanbit Electronics)") is
    grounded when the main name or an alias is. An empty chunk cannot be
    judged and counts as grounded, as in :func:`is_grounded`; an empty name
    does not.
    """
    if not name or not _name_tokens(name, False):
        return False
    if not _name_tokens(chunk_text or "", False):
        return True
    candidates = [_PARENTHETICAL.sub(" ", name), *_PARENTHETICAL.findall(name)]
    return any(
        _single_name_grounded(candidate, chunk_text)
        for candidate in candidates
        if candidate.strip()
    )
