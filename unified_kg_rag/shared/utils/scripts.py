# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unicode script tests for dense (space-optional) writing systems.

Han, Hangul and Kana pack a morpheme into one or two characters and do not
need spaces between words: one character carries what a whole Latin word
carries. Several text heuristics (edit-distance fuzziness, character
shingles, word-token counts) are tuned for Latin words and need to know when
they face such text. The tests below go by Unicode block, not by language.
"""

from __future__ import annotations

import re

__all__ = [
    "drop_dense_script_spaces",
    "has_dense_script",
    "is_dense_script_char",
    "is_hangul_syllable",
]

# Letter blocks only: CJK punctuation and full-width forms are excluded (NFKC
# folds the latter to ASCII).
_DENSE_SCRIPT_LETTER_RANGES: tuple[tuple[int, int], ...] = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3130, 0x318F),  # Hangul Compatibility Jamo
    (0x31F0, 0x31FF),  # Katakana Phonetic Extensions
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0xD7B0, 0xD7FF),  # Hangul Jamo Extended-B
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x323AF),  # CJK Unified Ideographs Extensions B-H
)


def is_dense_script_char(ch: str) -> bool:
    """True if ``ch`` is a Han, Hangul or Kana letter."""
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _DENSE_SCRIPT_LETTER_RANGES)


def has_dense_script(text: str) -> bool:
    """True if ``text`` contains at least one Han, Hangul or Kana letter."""
    return any(is_dense_script_char(ch) for ch in text)


def is_hangul_syllable(ch: str) -> bool:
    """True if ``ch`` is a precomposed Hangul syllable (가-힣)."""
    return "가" <= ch <= "힣"


_WHITESPACE = re.compile(r"\s+")


def drop_dense_script_spaces(text: str) -> str:
    """Remove whitespace next to a Han, Hangul or Kana letter.

    Word spacing is optional in these scripts ("가나다 상사" / "가나다상사",
    "3억 원" / "3억원"), so two spellings that differ only there compare
    equal afterwards. Other whitespace runs collapse to one space.
    """

    def replace(match: re.Match[str]) -> str:
        before = text[match.start() - 1] if match.start() else ""
        after = text[match.end()] if match.end() < len(text) else ""
        dense = (bool(before) and is_dense_script_char(before)) or (
            bool(after) and is_dense_script_char(after)
        )
        return "" if dense else " "

    return _WHITESPACE.sub(replace, text)
