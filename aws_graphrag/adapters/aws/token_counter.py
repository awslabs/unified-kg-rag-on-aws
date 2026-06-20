# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
from __future__ import annotations

from functools import lru_cache
from typing import Any

from aws_graphrag.shared import get_logger

logger = get_logger(__name__)


class BedrockTokenCounter:
    """Token counter using the Bedrock count_tokens API for accurate token measurement.

    The Bedrock ``count_tokens`` API is the single source of truth. If a call
    fails (e.g. transient error or a model that does not support the API), it
    degrades to a whitespace word count purely to keep the pipeline running — no
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
                "Bedrock count_tokens failed for model '%s': %s. Degrading to whitespace word count.",
                self.model_id,
                e,
            )
            return len(text.split())

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
