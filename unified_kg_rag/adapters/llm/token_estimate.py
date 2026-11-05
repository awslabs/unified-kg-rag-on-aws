# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Script-aware token-count estimate, for models without a counting API."""

import re

from unified_kg_rag.domain.ingestion.chunk_length import LATIN_CHARS_PER_TOKEN
from unified_kg_rag.shared.utils.scripts import DENSE_SCRIPT_TOKEN_RANGES

# One compiled character class over the dense-script ranges, which
# tokenize at roughly ONE token per character (often more) under BPE
# tokenizers: a plain chars/4 estimate under-counts them ~4x, which is
# exactly where embedding truncation silently failed on CJK corpora. The
# estimate runs over every context section of every query, and a
# per-character Python loop over the ranges cost ~1.5 ms per 1.5 KB section;
# the regex scan runs in C.
_DENSE_SCRIPT_RE = re.compile(
    "["
    + "".join(f"\\u{lo:04X}-\\u{hi:04X}" for lo, hi in DENSE_SCRIPT_TOKEN_RANGES)
    + "]"
)


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
    return estimate_from_counts(dense_chars, len(text) - dense_chars, len(text.split()))


def estimate_from_counts(dense: int, other: int, words: int) -> int:
    """The estimate of a non-empty text of ``dense`` dense-script
    characters, ``other`` other characters and ``words`` whitespace-split
    words: ~1 token per dense-script character plus ~4 characters per token
    for the rest, floored by the word count and by 1. The one formula of
    :func:`estimate_token_count` and of the chunker's running counts."""
    return max(words, dense + other // LATIN_CHARS_PER_TOKEN, 1)


def truncate_to_estimate(text: str, max_tokens: int) -> str:
    """``text`` cut so its estimated token count is at most ``max_tokens``."""
    if estimate_token_count(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    # Largest prefix that fits; the estimate is monotone in the prefix length.
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_token_count(text[:middle]) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low]
