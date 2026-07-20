# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from functools import lru_cache
from typing import Any

from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

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


def _is_dense_script_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _DENSE_SCRIPT_RANGES)


def estimate_token_count(text: str) -> int:
    """Script-aware token-count estimate for the API-unavailable fallback.

    Bedrock's CountTokens API does not support embedding models, so embedding
    truncation always lands on this estimate. A flat ~4-chars-per-token estimate
    under-counts space-less dense scripts (CJK, kana, Hangul) ~4x — a whole
    Korean/Japanese sentence is few "words" and few chars-over-4 but many tokens
    — which let over-limit chunks slip past truncation and fail the embedding
    call. We count dense-script characters at ~1 token each and the remaining
    (largely Latin) text at ~4 chars/token, then floor with the whitespace word
    count so space-delimited text is never under-counted.
    """
    if not text:
        return 0
    dense_chars = sum(1 for ch in text if _is_dense_script_char(ch))
    other_chars = len(text) - dense_chars
    # ~1 token per dense-script char + ~4 chars per token for the rest.
    char_estimate = dense_chars + (other_chars // 4)
    word_count = len(text.split())
    return max(word_count, char_estimate, 1)


class BedrockTokenCounter:
    """Token counter using the Bedrock count_tokens API for accurate token measurement.

    The Bedrock ``count_tokens`` API is the single source of truth. If a call
    fails (e.g. transient error or a model that does not support the API), it
    degrades to a script-aware estimate purely to keep the pipeline running — no
    third-party tokenizer is used, so counting stays consistent with the model.
    """

    MAX_TRUNCATION_ITERATIONS: int = 8

    def __init__(
        self,
        model_id: str,
        client: Any,
        cache_maxsize: int = 1024,
    ) -> None:
        self.model_id = model_id
        self._client = client

        @lru_cache(maxsize=cache_maxsize)
        def _cached_count(text: str) -> int:
            return self._call_bedrock_count_tokens(text)

        self._cached_count = _cached_count

    def count_tokens(self, text: str) -> int:
        """Count tokens via the Bedrock count_tokens API (LRU-cached).

        Degrades to a whitespace word count only if the API call fails, so the
        pipeline never crashes on an unsupported model or transient error.
        """
        if not text:
            return 0
        try:
            return self._cached_count(text)
        except Exception as e:
            logger.debug(
                "Bedrock count_tokens failed for model '%s': %s. Degrading to "
                "script-aware estimate.",
                self.model_id,
                e,
            )
            return estimate_token_count(text)

    def truncate_to_token_limit(self, text: str, max_tokens: int) -> tuple[str, int]:
        """Truncate text to fit within max_tokens using ratio-based estimation and verification.

        Returns:
            A tuple of (truncated_text, final_token_count).
        """
        if not text:
            return text, 0

        token_count = self.count_tokens(text)
        if token_count <= max_tokens:
            return text, token_count

        ratio = max_tokens / token_count
        char_limit = int(len(text) * ratio * 0.95)
        truncated = text[:char_limit]

        for _ in range(self.MAX_TRUNCATION_ITERATIONS):
            current_count = self.count_tokens(truncated)
            if current_count <= max_tokens:
                slack = max_tokens - current_count
                if slack > max_tokens * 0.05 and len(truncated) < len(text):
                    chars_per_token = len(truncated) / max(current_count, 1)
                    extra_chars = int(slack * chars_per_token * 0.8)
                    candidate = text[: len(truncated) + extra_chars]
                    candidate_count = self.count_tokens(candidate)
                    if candidate_count <= max_tokens:
                        truncated = candidate
                        current_count = candidate_count
                        continue
                return truncated, current_count

            overshoot = current_count - max_tokens
            chars_per_token = len(truncated) / max(current_count, 1)
            reduce_chars = max(int(overshoot * chars_per_token * 1.1), 1)
            truncated = truncated[: len(truncated) - reduce_chars]

        final_count = self.count_tokens(truncated)
        return truncated, final_count

    def _call_bedrock_count_tokens(self, text: str) -> int:
        response = self._client.count_tokens(
            modelId=self.model_id,
            converse={
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": text}],
                    }
                ]
            },
        )
        return int(response["totalTokens"])
