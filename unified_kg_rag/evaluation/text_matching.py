# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic text normalization shared by the LLM-free evaluators."""

from __future__ import annotations

import re
import string
import unicodedata

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_ASCII_PUNCTUATION = frozenset(string.punctuation)


def _is_punctuation(ch: str) -> bool:
    # SQuAD removes ASCII string.punctuation (which includes symbols such as
    # "$" and "%"); Unicode punctuation (category P*) covers CJK and fullwidth
    # marks such as 「」 and 。.
    return ch in _ASCII_PUNCTUATION or unicodedata.category(ch).startswith("P")


def normalize_answer(text: str) -> str:
    """SQuAD v1.1 answer normalization, preceded by Unicode NFKC.

    NFKC folds compatibility forms (fullwidth digits/letters, ligatures, the
    "…" ellipsis) to their canonical characters; then, as in the official
    SQuAD script: lowercase, delete punctuation, drop the English articles
    ``a``/``an``/``the``, and collapse whitespace. Punctuation is deleted, not
    replaced by a space, so "1,000" equals "1000" (and "3.5" becomes "35").
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(ch for ch in text if not _is_punctuation(ch))
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def tokenize(text: str) -> list[str]:
    """Whitespace tokens of the normalized text (``normalize_answer``)."""
    return normalize_answer(text).split()


def _is_hangul(ch: str) -> bool:
    return "가" <= ch <= "힣"


def is_spaceless_script(text: str) -> bool:
    """True if the text has CJK characters and no internal whitespace.

    Such a phrase is not whitespace-tokenizable (Chinese/Japanese), or is a
    single Korean word that takes attached particles, so contiguous word-token
    matching would miss it; it is matched as a substring instead.
    """
    if any(ch.isspace() for ch in text.strip()):
        return False
    return any(
        "぀" <= ch <= "鿿" or _is_hangul(ch)  # Hiragana, Katakana, CJK ideographs
        for ch in text
    )


# Korean particles/copula endings attached to a word ("서울은", "서울에서",
# "서울입니다") are at most this many syllables in the common cases.
_MAX_KOREAN_SUFFIX = 3


def _token_matches(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    suffix = actual[len(expected) :]
    return (
        actual.startswith(expected)
        and _is_hangul(expected[-1])
        and 0 < len(suffix) <= _MAX_KOREAN_SUFFIX
        and all(_is_hangul(ch) for ch in suffix)
    )


def phrase_in_text(phrase: str, text: str) -> bool:
    """True if ``phrase`` appears in ``text`` after normalization.

    Space-delimited phrases match a contiguous run of whole word tokens, so
    "AI" does not match inside "airport". A Hangul token also matches a token
    that extends it by up to three Hangul syllables, so "서울 특별시" matches
    "서울 특별시는" (attached particles). A phrase that is a single CJK run
    (``is_spaceless_script``) is matched as a substring of the normalized text.

    LIMITATION (CJK): without a morphological segmenter there is no morpheme
    boundary, so a short expected word can match inside a longer one ("가나"
    within "가나상사", "서울" within "서울대학교"). This over-counts rather
    than under-counts; both uses (coverage, containment) are recall-style.
    """
    if is_spaceless_script(phrase):
        cleaned = normalize_answer(phrase)
        return bool(cleaned) and cleaned in normalize_answer(text)

    expected = tokenize(phrase)
    if not expected:
        return False
    actual = tokenize(text)
    window = len(expected)
    return any(
        all(
            _token_matches(e, a)
            for e, a in zip(expected, actual[start : start + window], strict=True)
        )
        for start in range(len(actual) - window + 1)
    )
