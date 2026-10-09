# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic text normalization shared by the LLM-free evaluators."""

from __future__ import annotations

import re
import string
import unicodedata

from unified_kg_rag.shared.utils.scripts import (
    drop_dense_script_spaces,
    has_dense_script,
    is_dense_script_char,
    is_hangul_syllable,
)

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


def is_spaceless_script(text: str) -> bool:
    """True if the text has CJK characters and no internal whitespace.

    Such a phrase is not whitespace-tokenizable (Chinese/Japanese), or is a
    single Korean word that takes attached particles, so contiguous word-token
    matching would miss it; it is matched as a substring instead.
    """
    if any(ch.isspace() for ch in text.strip()):
        return False
    return has_dense_script(text)


def _continues_word(a: str, b: str) -> bool:
    """True if adjacent characters ``a`` and ``b`` belong to one word or number.

    Digits (of any script, including Han numerals) and non-CJK letters run
    together: "2년" inside "12년" or "二年" inside "十二年" is another number.
    """

    def wordlike(ch: str) -> bool:
        return ch.isnumeric() or (ch.isalpha() and not is_dense_script_char(ch))

    return wordlike(a) and wordlike(b)


def _substring_at_boundary(needle: str, haystack: str) -> bool:
    start = haystack.find(needle)
    while start != -1:
        end = start + len(needle)
        if not (
            (start and _continues_word(haystack[start - 1], needle[0]))
            or (end < len(haystack) and _continues_word(needle[-1], haystack[end]))
        ):
            return True
        start = haystack.find(needle, start + 1)
    return False


# Korean particles/copula endings attached to a word ("서울은", "서울에서",
# "서울입니다") are at most this many syllables in the common cases.
_MAX_KOREAN_SUFFIX = 3


def _token_matches(expected: str, actual: str) -> bool:
    # Any token may carry an attached particle: "서울은", "2024년에", "AWS는".
    if expected == actual:
        return True
    suffix = actual[len(expected) :]
    return (
        actual.startswith(expected)
        and 0 < len(suffix) <= _MAX_KOREAN_SUFFIX
        and all(is_hangul_syllable(ch) for ch in suffix)
    )


def _compact_window_matches(expected: str, actual: list[str]) -> bool:
    """True if a run of whole tokens, spacing ignored, equals ``expected``.

    The run may end in an attached particle (``_token_matches``), so the
    gold "가나다 상사" (compacted "가나다상사") matches "가나다상사는" and
    "가나다 상사는", but not "가나다상사관리본부".
    """
    for start in range(len(actual)):
        joined = ""
        for token in actual[start:]:
            joined = drop_dense_script_spaces(f"{joined} {token}" if joined else token)
            if _token_matches(expected, joined):
                return True
            if len(joined) > len(expected) + _MAX_KOREAN_SUFFIX:
                break
    return False


def phrase_in_text(phrase: str, text: str) -> bool:
    """True if ``phrase`` appears in ``text`` after normalization.

    Space-delimited phrases match a contiguous run of whole word tokens, so
    "AI" does not match inside "airport". Any token also matches a token that
    extends it by up to three Hangul syllables, so "서울 특별시" matches
    "서울 특별시는" and "AWS" matches "AWS는" (attached particles). A phrase
    containing CJK is also compared with the spaces next to CJK letters
    removed, so "가나다 상사" matches "가나다상사이다" and "3억 원" matches
    "3억원". A phrase that is a single CJK run (``is_spaceless_script``) is
    matched as a substring of the normalized text (spacing ignored), but not
    where a digit or Latin letter continues it on either side: "2년" does
    not match "12년", nor "二年" "十二年".

    LIMITATION (CJK): without a morphological segmenter there is no morpheme
    boundary, so a short expected word can match inside a longer one ("가나"
    within "가나상사", "서울" within "서울대학교"). This over-counts rather
    than under-counts; both uses (coverage, containment) are recall-style.
    """
    if is_spaceless_script(phrase):
        cleaned = normalize_answer(phrase)
        return bool(cleaned) and _substring_at_boundary(
            cleaned, drop_dense_script_spaces(normalize_answer(text))
        )

    expected = tokenize(phrase)
    if not expected:
        return False
    actual = tokenize(text)
    window = len(expected)
    if any(
        all(
            _token_matches(e, a)
            for e, a in zip(expected, actual[start : start + window], strict=True)
        )
        for start in range(len(actual) - window + 1)
    ):
        return True
    return has_dense_script(phrase) and _compact_window_matches(
        drop_dense_script_spaces(" ".join(expected)), actual
    )
