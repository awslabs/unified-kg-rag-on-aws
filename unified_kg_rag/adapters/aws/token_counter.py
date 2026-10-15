# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import threading
from functools import lru_cache
from typing import Any

from unified_kg_rag.adapters.llm.token_estimate import estimate_token_count
from unified_kg_rag.shared import get_logger

logger = get_logger(__name__)

# Bedrock error codes meaning "this model/operation will never accept
# CountTokens" (no IAM permission, unknown operation on an old endpoint, unknown
# model). Throttling, 5xx, model-not-ready and timeouts are transient and
# deliberately absent: they must not permanently disable the API for a model.
_UNSUPPORTED_ERROR_CODES: frozenset[str] = frozenset(
    {
        "AccessDeniedException",
        "UnknownOperationException",
        "ResourceNotFoundException",
    }
)

# ``ValidationException`` is ambiguous: Bedrock returns it both for a model that
# does not support CountTokens and for a bad *input* (e.g. a blank text block).
# Only the former is permanent, so it is recognised by its message; any other
# ValidationException fails just the current call.
_VALIDATION_UNSUPPORTED_MARKERS: tuple[str, ...] = (
    "doesn't support",
    "does not support",
    "not supported",
    "unsupported",
)

# Process-wide negative cache of model ids whose CountTokens call failed with a
# non-transient error. ``lru_cache`` does not memoize exceptions, so without
# this every count for such a model paid a failing network round trip.
_unsupported_model_ids: set[str] = set()
_unsupported_lock = threading.Lock()


def is_count_tokens_unsupported_error(error: BaseException) -> bool:
    """Whether a CountTokens failure means the model will never support it.

    Client-side request-shape errors and a client lacking the operation
    (botocore too old) are deterministic too, so they count as unsupported. A
    ``ValidationException`` counts only when its message says the model or
    operation is unsupported; otherwise it is an input problem for that call.
    """
    # Only a Bedrock call raises these, so botocore is already loaded here.
    from botocore.exceptions import ClientError, ParamValidationError

    if isinstance(error, ClientError):
        err = error.response.get("Error", {})
        code = str(err.get("Code", ""))
        if code == "ValidationException":
            message = str(err.get("Message", "")).lower()
            return any(marker in message for marker in _VALIDATION_UNSUPPORTED_MARKERS)
        return code in _UNSUPPORTED_ERROR_CODES
    return isinstance(error, (ParamValidationError, AttributeError))


def is_count_tokens_known_unsupported(model_id: str) -> bool:
    with _unsupported_lock:
        return model_id in _unsupported_model_ids


def mark_count_tokens_unsupported(model_id: str, reason: object) -> None:
    """Record that ``model_id`` rejects CountTokens; logs only the first time."""
    with _unsupported_lock:
        if model_id in _unsupported_model_ids:
            return
        _unsupported_model_ids.add(model_id)
    logger.info(
        "Bedrock count_tokens is unsupported for model '%s' (%s); using the "
        "script-aware estimate for the rest of this process.",
        model_id,
        reason,
    )


class BedrockTokenCounter:
    """Token counter using the Bedrock count_tokens API for accurate token measurement.

    The Bedrock ``count_tokens`` API is the single source of truth. If a call
    fails (e.g. transient error or a model that does not support the API), it
    degrades to a script-aware estimate purely to keep the pipeline running — no
    third-party tokenizer is used, so counting stays consistent with the model.

    A non-transient failure (see ``is_count_tokens_unsupported_error``) marks the
    model unsupported process-wide, so later counts skip the API entirely.
    Callers that already know the model cannot be counted (embedding and rerank
    models do not accept the Converse input this counter sends) pass
    ``api_supported=False`` and may pass ``client=None``.

    ``estimate_multiplier`` scales the local estimate for a model whose
    tokenizer counts more tokens than the one the estimate is calibrated to
    (``LanguageModelInfo.token_estimate_multiplier``); API counts are exact and
    never scaled.
    """

    MAX_TRUNCATION_ITERATIONS: int = 8

    def __init__(
        self,
        model_id: str,
        client: Any,
        cache_maxsize: int = 1024,
        *,
        api_supported: bool = True,
        estimate_multiplier: float = 1.0,
    ) -> None:
        self.model_id = model_id
        self._client = client
        self._api_supported = api_supported and client is not None
        self._estimate_multiplier = estimate_multiplier

        @lru_cache(maxsize=cache_maxsize)
        def _cached_count(text: str) -> int:
            return self._call_bedrock_count_tokens(text)

        self._cached_count = _cached_count

    @property
    def counts_via_api(self) -> bool:
        """Whether a count is a CountTokens round trip (not a local estimate).

        False from the start for a counter built without API support, and once
        the model is known to reject CountTokens.
        """
        return self._api_supported and not is_count_tokens_known_unsupported(
            self.model_id
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens via the Bedrock count_tokens API (LRU-cached).

        Degrades to the script-aware estimate if the API call fails, so the
        pipeline never crashes on an unsupported model or transient error. Once
        the model is known to be unsupported, the API is no longer called.
        """
        if not text:
            return 0
        # Converse rejects blank text blocks with a ValidationException; never
        # send one (it would also be indistinguishable from a model rejection).
        if not text.strip():
            return self._estimate(text)
        if not self.counts_via_api:
            return self._estimate(text)
        try:
            return self._cached_count(text)
        except Exception as e:
            if is_count_tokens_unsupported_error(e):
                mark_count_tokens_unsupported(self.model_id, e)
                return self._estimate(text)
            logger.debug(
                "Bedrock count_tokens failed for model '%s': %s. Degrading to "
                "script-aware estimate.",
                self.model_id,
                e,
            )
            return self._estimate(text)

    def _estimate(self, text: str) -> int:
        estimate = estimate_token_count(text)
        if self._estimate_multiplier == 1.0:
            return estimate
        return math.ceil(estimate * self._estimate_multiplier)

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
        # The request nests the payload under `input` and the response returns
        # `inputTokens`. A top-level `converse=` raises ParamValidationError and
        # `totalTokens` raises KeyError — both were silently swallowed by the
        # estimate fallback, so every count degraded rather than failing loudly.
        response = self._client.count_tokens(
            modelId=self.model_id,
            input={
                "converse": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": text}],
                        }
                    ]
                }
            },
        )
        return int(response["inputTokens"])
