# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Script-aware token-count estimate, for models without a counting API."""

import re

# Codepoint ranges that tokenize at roughly ONE token per character (often more)
# under BPE tokenizers: CJK ideographs + Japanese kana + Hangul + CJK
# punctuation/full-width forms. A plain chars/4 estimate under-counts these ~4x,
# which is exactly where embedding truncation silently failed on CJK corpora.
_DENSE_SCRIPT_RANGES: tuple[tuple[int, int], ...] = (
    (0x1100, 0x11FF),  # Hangul Jamo
    (0x2E80, 0x2FDF),  # CJK radicals / Kangxi
    (0x3000, 0x303F),  # CJK symbols and punctuation
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xA960, 0xA97F),  # Hangul Jamo Extended-A
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xFF00, 0xFFEF),  # Half/full-width forms
)


# One compiled character class over the ranges above. The estimate runs over
# every context section of every query, and a per-character Python loop over
# the ranges cost ~1.5 ms per 1.5 KB section; the regex scan runs in C.
_DENSE_SCRIPT_RE = re.compile(
    "[" + "".join(f"\\u{lo:04X}-\\u{hi:04X}" for lo, hi in _DENSE_SCRIPT_RANGES) + "]"
)


# Characters per token of the text outside the dense scripts.
LATIN_CHARS_PER_TOKEN = 4


def dense_char_count(text: str) -> int:
    """How many characters of ``text`` are in a dense script (~1 token each)."""
    return len(_DENSE_SCRIPT_RE.findall(text))


def estimate_token_count(text: str) -> int:
    """Script-aware token-count estimate for the API-unavailable fallback.

    Bedrock's CountTokens API does not support embedding models, so embedding
    truncation always lands on this estimate. Nor does it support every language
    model: Claude 5 (and Opus 4.8) reject it on ``bedrock-runtime``, so context
    budgeting for those models lands here too. A flat ~4-chars-per-token estimate
    under-counts space-less dense scripts (CJK, kana, Hangul) ~4x — a whole
    Korean/Japanese sentence is few "words" and few chars-over-4 but many tokens
    — which let over-limit chunks slip past truncation and fail the embedding
    call. We count dense-script characters at ~1 token each and the remaining
    (largely Latin) text at ~4 chars/token, then floor with the whitespace word
    count so space-delimited text is never under-counted.
    """
    if not text:
        return 0
    dense_chars = dense_char_count(text)
    other_chars = len(text) - dense_chars
    # ~1 token per dense-script char + ~4 chars per token for the rest.
    char_estimate = dense_chars + (other_chars // LATIN_CHARS_PER_TOKEN)
    word_count = len(text.split())
    return max(word_count, char_estimate, 1)
