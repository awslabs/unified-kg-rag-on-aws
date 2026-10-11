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
# A comma followed by exactly three digits inside a number ("1,000,000").
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_DASHES = "\u002d\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
# A dash between two word characters separates them: a range ("1990-1995"
# is two numbers) or a compound ("5-year" equals "5 year").
_JOINING_DASH = re.compile(rf"(?<=\w)[{_DASHES}](?=\w)")
# A minus sign starting a number ("-5", "(−5)"); folded to ASCII "-".
_LEADING_MINUS = re.compile(rf"(?<!\w)[{_DASHES}](?=\d)")
# Punctuation that carries meaning and is kept: a leading minus ("-5" is not
# "5"), a decimal point or comma between digits ("1.5" is not "15"), a percent
# sign after a number ("3.5%" is not "3.5") and "+"/"#" ending a word ("C++",
# "C#").
_KEPT_PUNCTUATION = re.compile(
    r"(?<!\w)-(?=\d)|(?<=\d)[.,](?=\d)|(?<=\d)%|(?<=\w)[+#]+"
)
# "3.5 %" is written "3.5%".
_SPACED_PERCENT = re.compile(r"(?<=\d)\s+%")
_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)*%?")


def _is_punctuation(ch: str) -> bool:
    # ASCII string.punctuation (which includes symbols such as "$" and "%")
    # plus Unicode punctuation (category P*), which covers CJK and fullwidth
    # marks such as 「」 and 。.
    return ch in _ASCII_PUNCTUATION or unicodedata.category(ch).startswith("P")


_WORD_CHAR = re.compile(r"\w")
# A space between an ASCII word and a lone capital letter ending a word: the
# "A" of "Type A" or "Plan B", folded like "Type-A" into one token. A capital
# starting a symbol ("C++", "C#") or an abbreviation ("U.S.") is not one.
_SPACED_LONE_CAPITAL = re.compile(r"(?<=[A-Za-z0-9]) (?=[A-Z](?![\w+#]|\.\w))")


def _is_lone_letter(text: str, index: int, step: int) -> bool:
    """Whether ``text[index]`` is a letter forming a one-character word.

    ``step`` is the direction away from the dash (-1 left, +1 right); the
    word is one character long when the next character in that direction is
    not a word character (or the text ends).
    """
    if not 0 <= index < len(text) or not text[index].isalpha():
        return False
    beyond = index + step
    return not (0 <= beyond < len(text) and _WORD_CHAR.match(text[beyond]))


def _split_or_join_dash(match: re.Match[str]) -> str:
    """A dash between words is a space, except next to a lone letter.

    "1990-1995" and "5-year" split into separate tokens, but "Type-A" and
    "A-1" stay one token: split, the lone "a" would be dropped as an article
    and "Type-A" would match "Type-B". Only the characters next to the dash
    are inspected, so the pass stays linear in the text length.
    """
    text = match.string
    if _is_lone_letter(text, match.start() - 1, -1) or _is_lone_letter(
        text, match.end(), 1
    ):
        return ""
    return " "


def normalize_answer(text: str) -> str:
    """SQuAD-style answer normalization, preceded by Unicode NFKC.

    NFKC folds compatibility forms (fullwidth digits/letters, ligatures, the
    "…" ellipsis) to their canonical characters; then: lowercase, delete
    punctuation, drop the English articles ``a``/``an``/``the``, and collapse
    whitespace. Unlike the official SQuAD v1.1 script, Unicode punctuation is
    deleted too, and punctuation that changes meaning is kept: a leading
    minus ("-5" stays distinct from "5"), a decimal point or comma between
    digits ("1.5" from "15"), a percent sign after a number ("3.5%" from
    "3.5") and a trailing "+"/"#" ("C++", "C#"). A dash between two words or
    numbers becomes a space, so "1990-1995" is two numbers and "5-year"
    equals "5 year"; next to a lone letter it is deleted instead, and so is
    the space before a lone capital ending a word, so "Type-A", "Type A" and
    "TypeA" are one token that does not match "Type B". A lowercase lone
    letter after a space ("a vendor") is left alone, as an article or a
    word. Thousands separators are deleted, so "1,000" equals
    "1000". Scores are therefore not comparable with leaderboard numbers
    computed by the official script.

    A text consisting only of articles ("A", "The") keeps them rather than
    normalizing to an empty string.
    """
    text = _SPACED_LONE_CAPITAL.sub("", unicodedata.normalize("NFKC", text))
    text = text.lower()
    text = _THOUSANDS_SEPARATOR.sub("", text)
    text = _JOINING_DASH.sub(_split_or_join_dash, text)
    text = _LEADING_MINUS.sub("-", text)
    text = _SPACED_PERCENT.sub("%", text)
    kept = {i for m in _KEPT_PUNCTUATION.finditer(text) for i in range(*m.span())}
    text = "".join(
        ch for i, ch in enumerate(text) if i in kept or not _is_punctuation(ch)
    )
    without_articles = " ".join(_ARTICLES.sub(" ", text).split())
    return without_articles or " ".join(text.split())


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
        # A "." or "," survives normalization only between digits ("1.5"),
        # a "-" only as the sign of a number ("-2년" is not "2년").
        return (
            ch.isnumeric()
            or ch in ".,-"
            or (ch.isalpha() and not is_dense_script_char(ch))
        )

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
# Korean magnitude numerals: after a number they scale its value ("3억" is
# 300 million), so a bare gold number never matches a token they extend.
_KOREAN_MAGNITUDES = frozenset("십백천만억조경")
# Korean counters and units that keep a number's value ("2024년", "3명",
# "5,000원"), longest first; a particle may follow them ("2024년에").
_KOREAN_UNITS = (
    "개월",
    "년",
    "월",
    "일",
    "주",
    "시",
    "분",
    "초",
    "명",
    "개",
    "회",
    "건",
    "번",
    "배",
    "원",
)


def _is_particle(suffix: str) -> bool:
    return 0 < len(suffix) <= _MAX_KOREAN_SUFFIX and all(
        is_hangul_syllable(ch) for ch in suffix
    )


def _number_suffix_keeps_value(suffix: str) -> bool:
    """True if Hangul ``suffix`` after a number is a unit and/or a particle."""
    if not suffix or suffix[0] in _KOREAN_MAGNITUDES:
        return False
    for unit in _KOREAN_UNITS:
        if suffix.startswith(unit):
            rest = suffix[len(unit) :]
            return not rest or _is_particle(rest)
    return _is_particle(suffix)


def _token_matches(expected: str, actual: str) -> bool:
    # A word may carry an attached particle: "서울은", "AWS는". A bare number
    # may carry a unit and a particle ("2024년에", "3명이"), but not a
    # magnitude numeral ("3억", "5천"), which changes the value.
    if expected == actual:
        return True
    if not actual.startswith(expected):
        return False
    suffix = actual[len(expected) :]
    if _NUMBER.fullmatch(expected):
        return _number_suffix_keeps_value(suffix)
    return _is_particle(suffix)


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
    "AI" does not match inside "airport". Any token except a bare number also
    matches a token that extends it by up to three Hangul syllables, so
    "서울 특별시" matches "서울 특별시는" and "AWS" matches "AWS는" (attached
    particles). A bare number also matches it followed by a Korean unit and
    a particle ("2024" in "2024년에"), but not by a magnitude numeral ("3" is
    not "3억", which changes the value). A phrase
    containing CJK is also compared with the spaces next to CJK letters
    removed, so "가나다 상사" matches "가나다상사이다" and "3억 원" matches
    "3억원". A phrase that is a single CJK run (``is_spaceless_script``) is
    matched as a substring of the normalized text (spacing ignored), but not
    where a digit, decimal point or Latin letter continues it on either side:
    "2년" does not match "12년" or "1.2년", nor "二年" "十二年".

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
